"""Runtime shims for ``nexus.extensions.tools.search_stateful``.

The upstream ``search_stateful`` backend authenticates both the serper search API
and the jina_ai page-fetch API with a single ``(user, apikey)`` pair. When the
two providers' budgets live on separate accounts, set:

    SEARCH_STATEFUL_JINA_USER
    SEARCH_STATEFUL_JINA_API_KEY

and call :func:`install`. The shim wraps
``GoogleApiBackend.fetch`` so jina_ai calls run under the secondary account while
serper keeps using the primary account. The ``search_stateful`` package itself is
not modified - this lives in the task layer because only tasks / generators know
about account-budget policies.

Installation is idempotent; :func:`install` is a no-op when the env vars are
unset, so it is always safe to call.
"""

from __future__ import annotations

import functools
import logging
import os

from typing import Any


logger = logging.getLogger(__name__)

_SHIM_FLAG = '_search_stateful_dual_account_installed'
_SHIM_CREDENTIALS = '_search_stateful_dual_account_credentials'


def _get_provider_credentials(provider_config: dict | None) -> tuple[str | None, str | None]:
    """Extract `(user, apikey)` from a nested provider config."""
    if not isinstance(provider_config, dict):
        return None, None
    return (
        provider_config.get('serper_app_id') or provider_config.get('jina_app_id') or provider_config.get('user'),
        provider_config.get('serper_app_key') or provider_config.get('jina_app_key') or provider_config.get('apikey'),
    )


def normalize_tool_config(raw_config: dict | None) -> tuple[dict, str | None, str | None]:
    """Normalize search_stateful config and extract optional dedicated Jina creds."""
    config = dict(raw_config or {})
    serper_cfg = config.pop('serper', None)
    jina_cfg = config.pop('jina', None)

    serper_user, serper_apikey = _get_provider_credentials(serper_cfg)
    if serper_user:
        config.setdefault('user', serper_user)
    if serper_apikey:
        config.setdefault('apikey', serper_apikey)

    jina_user, jina_apikey = _get_provider_credentials(jina_cfg)
    if isinstance(jina_cfg, dict) and (jina_provider := jina_cfg.get('jina_provider') or jina_cfg.get('provider')):
        config.setdefault('jina_provider', jina_provider)

    return config, jina_user, jina_apikey


def prepare_tool_configs(tool_configs: dict[str, Any]) -> None:
    """Normalize search_stateful config in-place and install the Jina credential shim."""
    raw_search_stateful_config = tool_configs.get('search_stateful')
    normalized_search_stateful_config, jina_user, jina_apikey = normalize_tool_config(raw_search_stateful_config)
    if raw_search_stateful_config is not None:
        tool_configs['search_stateful'] = normalized_search_stateful_config
    install(jina_user=jina_user, jina_apikey=jina_apikey)


def install(*, jina_user: str | None = None, jina_apikey: str | None = None) -> None:
    """Install the dual-account shim. Idempotent; no-op when credentials are unset."""
    jina_user = jina_user or os.getenv('SEARCH_STATEFUL_JINA_USER')
    jina_apikey = jina_apikey or os.getenv('SEARCH_STATEFUL_JINA_API_KEY')
    if not (jina_user and jina_apikey):
        return

    from nexus.extensions.tools.search_stateful import backend as _backend

    cls = _backend.GoogleApiBackend
    setattr(cls, _SHIM_CREDENTIALS, (jina_user, jina_apikey))
    if getattr(cls, _SHIM_FLAG, False):
        return

    original_fetch = cls.fetch

    @functools.wraps(original_fetch)
    async def fetch_with_jina_account(self, url: str):
        # Swap credentials only for the duration of this fetch. Each search_stateful
        # tool call creates its own backend instance, so this swap stays request-local.
        current_jina_user, current_jina_apikey = getattr(cls, _SHIM_CREDENTIALS, (None, None))
        if not (current_jina_user and current_jina_apikey):
            return await original_fetch(self, url)
        saved_user, saved_apikey = self.user, self.apikey
        self.user, self.apikey = current_jina_user, current_jina_apikey
        try:
            return await original_fetch(self, url)
        finally:
            self.user, self.apikey = saved_user, saved_apikey

    cls.fetch = fetch_with_jina_account
    setattr(cls, _SHIM_FLAG, True)
    logger.debug(
        'search_stateful dual-account shim installed: jina_ai routed through %s',
        jina_user,
    )
