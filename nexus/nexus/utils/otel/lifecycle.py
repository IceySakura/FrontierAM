from __future__ import annotations

import asyncio
import logging

from nexus.utils.otel.auto import is_enabled


logger = logging.getLogger(__name__)


def _get_tracer_provider():
    try:
        from opentelemetry import trace
    except ModuleNotFoundError as exc:
        if not exc.name or not exc.name.startswith('opentelemetry'):
            raise
        return None
    return trace.get_tracer_provider()


def force_flush(timeout_millis: int = 30_000) -> bool:
    if not is_enabled():
        return True

    provider = _get_tracer_provider()
    flush = getattr(provider, 'force_flush', None)
    return True if flush is None else bool(flush(timeout_millis=timeout_millis))


def force_flush_best_effort(timeout_millis: int = 5_000) -> bool:
    try:
        flushed = force_flush(timeout_millis=timeout_millis)
    except Exception:
        logger.warning('OpenTelemetry force flush failed', exc_info=True)
        return False
    if not flushed:
        logger.warning('OpenTelemetry force flush timed out')
    return flushed


async def async_force_flush_best_effort(timeout_millis: int = 5_000) -> bool:
    """Run the potentially blocking provider flush outside the event loop."""
    return await asyncio.to_thread(force_flush_best_effort, timeout_millis=timeout_millis)


def shutdown() -> None:
    if not is_enabled():
        return

    close = getattr(_get_tracer_provider(), 'shutdown', None)
    if close is not None:
        close()
