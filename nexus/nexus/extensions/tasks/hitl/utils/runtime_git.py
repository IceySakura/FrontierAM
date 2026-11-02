"""Host-side callers for the sandbox git helpers.

:mod:`nexus.extensions.tasks.hitl.utils.git` imports nothing but the standard
library because its functions are shipped into the sandbox by ``runtime.invoke``.
The wrappers here run on the host: they hold the execution context, decide whether
the work is still needed, and swallow failures that must not lose a run.
"""

import logging

from nexus.core import require_context
from nexus.extensions.tasks.hitl.utils.git import cleanup_test_file_conflicts


logger = logging.getLogger(__name__)

_CLEANED_FLAG = 'hitl_test_file_conflicts_cleaned'


async def cleanup_test_conflicts(
    runtime: object,
    *,
    working_dir: str,
    test_patch: str,
    base_commit: str,
) -> None:
    """Revert worker edits to files the dataset's test patch controls.

    Runs at most once per task: the flag lives in ``task_config`` so a second call
    from a later step is a no-op. A failure is logged and swallowed, because the
    worker's patch is still worth extracting when the cleanup could not run.
    """
    if not test_patch:
        return
    ctx = require_context()
    if ctx.task_config.get(_CLEANED_FLAG):
        logger.info('[hitl/git] test file conflicts already cleaned; skipping')
        return
    try:
        cleaned = await runtime.invoke(  # type: ignore[attr-defined]
            cleanup_test_file_conflicts,
            working_dir=working_dir,
            test_patch=test_patch,
            base_commit=base_commit,
        )
        ctx.task_config[_CLEANED_FLAG] = True
        if cleaned:
            logger.info('[hitl/git] cleaned test file conflicts before patch extraction: %s', cleaned)
    except Exception as exc:
        logger.warning('[hitl/git] test file cleanup failed: %s', exc, exc_info=True)


__all__ = ['cleanup_test_conflicts']
