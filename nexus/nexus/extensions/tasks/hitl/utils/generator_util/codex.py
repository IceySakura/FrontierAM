"""The ``worker_config`` a Codex backend accepts.

Every key here is one the Codex manager reads: the npm package to install, what goes
into its ``config.toml``, and its turn timeout. What only Claude Code accepts lives in
:mod:`claude_code`, and what holds for any harness in :mod:`worker`.
"""

import json
import os

from pathlib import Path
from typing import Any

from nexus.extensions.tasks.hitl.utils.generator_util.env import _env_bool, _env_list
from nexus.extensions.tasks.hitl.utils.generator_util.worker import (
    _read_user_memory,
    _worker_context_window,
    _worker_reasoning_effort,
    _worker_turn_timeout,
    _worker_version,
    worker_mcp_servers,
)


DEFAULT_TURN_TIMEOUT_SECONDS = 7200

DEFAULT_CODEX_VERSION = '0.154.0'

CATALOG_DIR = Path(__file__).resolve().parents[2] / 'assets' / 'codex'
"""Where a bare ``HITL_CODEX_MODEL_CATALOG`` name is looked up."""

MULTI_AGENT_VERSIONS = frozenset({'disabled', 'v1', 'v2'})
"""What a catalog entry's ``multi_agent_version`` takes. Codex drops any other value."""

DEFAULT_MULTI_AGENT_VERSION = 'disabled'


def _read_model_catalog() -> dict[str, Any] | None:
    """The catalog this run gives Codex, or None to leave the bundled one in effect.

    ``HITL_CODEX_MODEL_CATALOG`` names a file in :data:`CATALOG_DIR`, with or without
    the ``.json`` suffix, or is a path of its own. Unset means no catalog.

    A catalog entry decides the request shape Codex builds for its slug, which the
    bundled one gets wrong for some models against this proxy. Naming a file rather
    than patching the bundled entry keeps that decision in data: a slug the file does
    not carry makes Codex fail, where a patch that missed would leave the worker
    running with no tools and say nothing.
    """
    raw = os.environ.get('HITL_CODEX_MODEL_CATALOG', '').strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    if not path.is_absolute() and not path.exists():
        path = CATALOG_DIR / (raw if raw.endswith('.json') else f'{raw}.json')
    if not path.is_file():
        raise ValueError(
            f'HITL_CODEX_MODEL_CATALOG names no file: {path}. '
            f'Available in {CATALOG_DIR}: {sorted(p.stem for p in CATALOG_DIR.glob("*.json"))}'
        )
    catalog = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(catalog, dict) or not catalog.get('models'):
        raise ValueError(f'{path} carries no "models" array')
    return _without_async_user_input(_with_multi_agent_version(_with_responses_lite(catalog, path=path)))


ASYNC_USER_INPUT_TOOL = 'send_user_message_async'
"""The ``experimental_supported_tools`` entry that puts ``request_user_input_async`` on the wire.

Carrying it is what offers the tool: ``features.send_message_to_user_async`` does not gate it
either way, and that flag's own name belongs to a different tool.
"""


def _without_async_user_input(catalog: dict[str, Any]) -> dict[str, Any]:
    """Drop :data:`ASYNC_USER_INPUT_TOOL` from every entry unless the run asks for it.

    The worker runs under ``--dangerously-bypass-approvals-and-sandbox``, where that tool
    answers ``{"accepted":true}`` and emits the question as an ``agent_message``: it reaches
    the user simulator as ordinary speech with no ``<worker_question>`` tag, so the worker
    waits for an answer this harness never sends. ``HITL_CODEX_USER_INPUT_TOOLS=1`` keeps the
    entry as the catalog file has it. Other entries in the list, ``clock`` among them, are
    offered independently and stay.

    The synchronous ``request_user_input`` is not a catalog field; it leaves the wire through
    ``user_input_tools`` in the worker config, which the backend writes into ``config.toml``.
    """
    if _env_bool('HITL_CODEX_USER_INPUT_TOOLS', default=False):
        return catalog
    models = [
        {
            **entry,
            'experimental_supported_tools': [
                tool for tool in entry.get('experimental_supported_tools') or [] if tool != ASYNC_USER_INPUT_TOOL
            ],
        }
        for entry in catalog['models']
    ]
    return {**catalog, 'models': models}


def _with_multi_agent_version(catalog: dict[str, Any]) -> dict[str, Any]:
    """Apply ``HITL_CODEX_MULTI_AGENT`` to every entry.

    ``disabled`` leaves Codex with no agent tools and no multi-agent developer messages,
    which is 3387 fewer prompt bytes on the gpt-5.6-sol entry. ``v2`` is what the
    catalogs in :data:`CATALOG_DIR` carry and what the runs so far sent.
    """
    version = os.environ.get('HITL_CODEX_MULTI_AGENT', '').strip() or DEFAULT_MULTI_AGENT_VERSION
    if version not in MULTI_AGENT_VERSIONS:
        raise ValueError(f'HITL_CODEX_MULTI_AGENT must be one of {sorted(MULTI_AGENT_VERSIONS)}, got {version!r}')
    models = [{**entry, 'multi_agent_version': version} for entry in catalog['models']]
    return {**catalog, 'models': models}


def _with_responses_lite(catalog: dict[str, Any], *, path: Path) -> dict[str, Any]:
    """Apply ``HITL_CODEX_RESPONSES_LITE`` to every entry, when it is set.

    Unset leaves the file as it is. Lite requests carry a different shape on the wire,
    and which one serves this proxy better is open, so the run can flip it without
    keeping a second catalog around.
    """
    raw = os.environ.get('HITL_CODEX_RESPONSES_LITE', '').strip()
    if not raw:
        return catalog
    lite = _env_bool('HITL_CODEX_RESPONSES_LITE')
    models = [{**entry, 'use_responses_lite': lite} for entry in catalog['models']]
    return {**catalog, 'models': models}


def _build_codex_worker_config(profile: str, *, session_id: str = '') -> dict[str, Any]:
    """Build the Codex worker config dict for the selected profile.

    ``profile`` picks the host client's protocol, which the worker reaches through
    this session's Responses proxy; the proxy substitutes the real model on every
    call, so ``codex_model`` only names what Codex writes in its own ``config.toml``.
    """
    del profile, session_id

    config: dict[str, Any] = {
        'version': _worker_version(DEFAULT_CODEX_VERSION),
        'timeout': _worker_turn_timeout(DEFAULT_TURN_TIMEOUT_SECONDS),
    }

    npm_registry = os.environ.get('HITL_WORKER_NPM_REGISTRY', '').strip()
    if npm_registry:
        config['npm_registry'] = npm_registry

    codex_model = os.environ.get('HITL_CODEX_MODEL', '').strip()
    if codex_model:
        config['codex_model'] = codex_model

    catalog = _read_model_catalog()
    if catalog is not None:
        config['model_catalog_json'] = catalog

    # True names the provider "azure", which keeps store=true, replays the reasoning
    # item ids and posts compaction to /responses/compact.
    config['use_azure_provider'] = _env_bool('HITL_CODEX_AZURE_PROVIDER', default=False)

    # False stops the converter minting an ``rs_*`` for a plaintext
    # ``reasoning_content``. Codex replays that id on the next turn and a stateless
    # upstream answers ``rs_ not found``. A run serving HY3/GLM through this backend
    # sets it, which is what keeps their plaintext CoT replayable.
    config['wrap_plaintext_reasoning_content'] = _env_bool('HITL_CODEX_WRAP_PLAINTEXT_REASONING', default=False)

    # Codex compacts at min(auto_compact_token_limit, model_context_window * 90%).
    window = _worker_context_window()
    if window:
        config['model_context_window'] = window
        config['auto_compact_token_limit'] = window

    # Codex builds its own requests and the proxy forwards the effort it sends; the
    # host client's request args do not override it. Unset leaves the catalog entry's
    # ``default_reasoning_level``.
    effort = _worker_reasoning_effort()
    if effort:
        config['reasoning_effort'] = effort

    # ``-c key.path=value``, the form Codex takes on the command line. A key names its
    # own place in the config, so this reaches a top-level key and a table's field
    # alike.
    overrides = _env_list('HITL_CODEX_CONFIG_OVERRIDES')
    if overrides:
        config['extra_cli_args'] = [arg for override in overrides for arg in ('-c', override)]

    # False keeps ``request_user_input`` off the wire, which Codex would otherwise offer and
    # then answer itself with ``unavailable in Default mode``. The async half of that pair is
    # a catalog field instead, applied by :func:`_without_async_user_input`.
    config['user_input_tools'] = _env_bool('HITL_CODEX_USER_INPUT_TOOLS', default=False)

    user_memory = _read_user_memory()
    if user_memory:
        config['user_memory'] = user_memory

    # Registered per invocation as ``-c mcp_servers.*`` by the backend, which passes
    # them again on a resumed turn.
    mcp_servers = worker_mcp_servers(config)
    if mcp_servers:
        config['mcp_servers'] = mcp_servers

    return config


__all__ = [
    'CATALOG_DIR',
    'DEFAULT_CODEX_VERSION',
    'DEFAULT_MULTI_AGENT_VERSION',
    'DEFAULT_TURN_TIMEOUT_SECONDS',
    'MULTI_AGENT_VERSIONS',
    '_build_codex_worker_config',
]
