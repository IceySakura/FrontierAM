"""Task phase span management: flat, context-manager, and close-only usage.

See docs/explanation/tracing-convention.md for the full specification.

Usage::

    from nexus.trace.phases import phase

    # Flat — switch phase, auto-close previous
    phase('task.init')
    ...
    phase('task.agent')          # closes task.init, opens task.agent

    # Context manager — closes on __exit__
    with phase('task.init.get_runtime'):
        ...

    # Close only — no new span
    phase()
"""

from __future__ import annotations

from logging import getLogger
from typing import Any

from nexus.trace.controller import SpanController
from nexus.trace.tracer import (
    _active_span_stack,
    end_span,
    get_current_span_id,
    start_span,
    update_span,
)
from nexus.trace.types import Status


logger = getLogger(__name__)


def _close_to_ancestor(new_name: str | None) -> None:
    """Close spans from the stack top until we reach the right nesting level for *new_name*.

    Rules:
    - If *new_name* starts with ``<top_span_name>.`` the top span is a valid
      parent → stop (don't close it).
    - Otherwise close the top span and repeat.
    - Stop if the stack is empty or if the top span was not opened by ``phase()``
      (i.e. it has no ``phase`` attribute — this avoids accidentally closing
      the outer ``@observe`` span).
    """
    controller = SpanController.get_instance_sync()
    stack = _active_span_stack.get() or []

    while stack:
        top_id = stack[-1]
        try:
            top_span = controller.get_span(top_id)
        except KeyError:
            break

        # Don't close spans that were not created by phase()
        if top_span.attributes.get('type') != 'task_phase':
            break

        # If new_name is a child of the current top → keep it
        if new_name is not None and new_name.startswith(top_span.name + '.'):
            break

        # If new_name equals top span name exactly → close it (replacing with same phase)
        # Otherwise it's a sibling or unrelated → also close
        update_span(top_id, status=Status(code='OK'))
        end_span(top_id)

        # Re-read stack (end_span pops it)
        stack = _active_span_stack.get() or []


def _infer_phase(name: str) -> str:
    """Extract the phase component from a span name like ``task.init.get_runtime`` → ``init``."""
    parts = name.split('.')
    if len(parts) >= 2:
        return parts[1]
    return name


class _PhaseContext:
    """Returned by :func:`phase`.  Works as a context manager or can be ignored (flat call)."""

    def __init__(self, span_id: str | None):
        self._span_id = span_id

    def __enter__(self) -> _PhaseContext:
        return self

    def set_attributes(self, attributes: dict[str, Any]) -> None:
        """Add attributes while this phase span is active."""
        if self._span_id is not None:
            update_span(self._span_id, attributes=attributes)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        if self._span_id is None:
            return
        # Only close if this span is still on the stack top
        current = get_current_span_id()
        if current == self._span_id:
            if exc_type is not None:
                update_span(self._span_id, status=Status(code='ERROR', message=str(exc_val)))
            else:
                update_span(self._span_id, status=Status(code='OK'))
            end_span(self._span_id)
            self._span_id = None


def phase(
    name: str | None = None,
    *,
    attributes: dict[str, Any] | None = None,
    close_previous: bool = True,
) -> _PhaseContext:
    """Open a task-phase span, or close the current one.

    Args:
        name: The span name (e.g. ``'task.init'``, ``'task.init.get_runtime'``).
            When *None*, the current phase span is closed without opening a new one.
        attributes: Additional span attributes. The ``type`` and ``phase``
            compatibility attributes are always set by this function.
        close_previous: Whether to apply the flat phase replacement rules
            before opening the span. Structured executors can disable this
            when phase spans need to nest.

    Returns:
        A :class:`_PhaseContext` that can be used as a context manager
        (``with phase(...):``).  If used as a bare call the span stays open
        until the next ``phase()`` call closes it.
    """
    # Close-only mode
    if name is None:
        _close_to_ancestor(None)
        return _PhaseContext(None)

    # Close siblings/ancestors that need to be closed before opening new span
    if close_previous:
        _close_to_ancestor(name)

    # Open new phase span
    phase_value = _infer_phase(name)
    span_attributes = dict(attributes or {})
    span_attributes.update({
        'type': 'task_phase',
        'phase': phase_value,
    })
    stub = start_span(
        name,
        attributes=span_attributes,
    )
    return _PhaseContext(stub.span_id)


__all__ = ['phase']
