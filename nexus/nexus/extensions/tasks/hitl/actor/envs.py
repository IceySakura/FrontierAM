"""Environment handed to a worker process in the sandbox.

``session_env`` builds what does not depend on the harness: the host credentials
forwarded into the sandbox. What the harness itself needs comes from
``SessionManager.harness_env``, which each backend declares.
"""

import os

from collections.abc import Mapping

from nexus.extensions.tasks.hitl.actor.base import Session
from nexus.extensions.tasks.hitl.utils import coerce


# Host environment forwarded into the sandbox when present: the credentials and
# limits a worker needs.
_PASSTHROUGH = (
    'GH_TOKEN',
    'GITHUB_TOKEN',
    'HUGGINGFACE_API_TOKEN',
    'MODAL_TOKEN_ID',
    'MODAL_TOKEN_SECRET',
    'OPENAI_WHISPER_BASE_URL',
    'OPENAI_WHISPER_API_KEY',
)


def session_env(
    config: Mapping[str, object],
    *,
    session: Session,
    harness_env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """The environment for one session's worker process.

    ``harness_env`` is what the backend's own ``harness_env(config)`` returned.
    ``extra_env`` overrides it.
    """
    if session is None:
        raise ValueError('session must be a Session')

    env = {key: value for key in _PASSTHROUGH if (value := os.environ.get(key))}
    env.update(harness_env or {})

    env.update(coerce.string_map(config.get('extra_env')))

    return env


__all__ = ['session_env']
