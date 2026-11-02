"""Shared context variables for the trace module."""

from __future__ import annotations

from contextvars import ContextVar

from nexus.trace.processors import SpanProcessor


# Context variable to store context-local processors
# This ensures processors registered in one thread/coroutine don't affect others
context_processors: ContextVar[list[SpanProcessor] | None] = ContextVar('context_processors', default=None)


def get_processor(processor_type: type[SpanProcessor]) -> SpanProcessor | None:
    """Get a span processor by type from the current context.

    Reads from the context-local processor list (context_processors ContextVar).
    Returns the first instance matching processor_type via isinstance.

    Args:
        processor_type: A SpanProcessor subclass type (e.g., HytunerProcessor).

    Returns:
        The SpanProcessor instance, or None if no match is found.
    """
    processors = context_processors.get()
    if not processors:
        return None
    for p in processors:
        if isinstance(p, processor_type):
            return p
    return None
