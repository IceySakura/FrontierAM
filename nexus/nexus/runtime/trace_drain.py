from __future__ import annotations

import logging

from typing import TYPE_CHECKING, Literal


if TYPE_CHECKING:
    from nexus.runtime.client.base import Runtime


logger = logging.getLogger(__name__)
TraceDrainErrorPolicy = Literal['raise', 'warn']


async def drain_runtime_trace(
    runtime: Runtime,
    *,
    error_policy: TraceDrainErrorPolicy,
    run_id: str,
    runtime_name: str,
) -> None:
    """Apply the run's error policy only to runtime trace draining."""
    try:
        await runtime.drain_trace()
    except Exception as exc:
        if error_policy != 'warn':
            raise
        logger.warning(
            'Runtime trace drain failed; continuing with potentially incomplete trace: '
            'run_id=%s runtime=%s error_type=%s error=%s',
            run_id,
            runtime_name,
            type(exc).__name__,
            exc,
            exc_info=True,
        )
