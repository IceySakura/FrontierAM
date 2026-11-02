"""Which module builds the ``worker_config`` for a given backend.

The config keys are each harness's own, so the builder has to match the backend a
run names. Sits above :mod:`claude_code` and :mod:`codex`, which it selects
between.
"""

from typing import Any

from nexus.extensions.tasks.hitl.utils.generator_util.claude_code import _build_worker_config
from nexus.extensions.tasks.hitl.utils.generator_util.codex import _build_codex_worker_config


def worker_config_for(backend: str, profile: str, *, session_id: str = '') -> dict[str, Any]:
    """The ``worker_config`` for this backend, from the module that owns its keys.

    Matched on the backend's suffix, so a new backend of an existing harness needs
    no entry here. Raises for a harness with no builder rather than handing over
    another one's keys, which the manager reads as settings it never declares.
    """
    if backend.endswith('_codex'):
        return _build_codex_worker_config(profile, session_id=session_id)
    if backend.endswith('_claude_code'):
        return _build_worker_config(profile, session_id=session_id)
    raise ValueError(
        f'no worker_config builder for backend {backend!r}. Add one in utils/generator_util/ and name it here.'
    )


__all__ = ['worker_config_for']
