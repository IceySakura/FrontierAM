from __future__ import annotations

from typing import Any, Literal

from nexus.core.registry import Registry
from nexus.trace.events import SpanEndEvent, SpanStartEvent, SpanUpdateEvent
from nexus.trace.scopes import DEFAULT_SCOPE
from nexus.trace.types import Span


class SpanProcessor:
    """Abstract base class for span processors.

    Hook invocation contract:

    - Processors with ``dispatch_mode = 'inline'`` have hooks invoked
      immediately on the caller's thread. This is the default for processors
      that do not declare a mode.
    - Processors with ``dispatch_mode = 'background'`` have ``on_start`` /
      ``on_update`` / ``on_end`` invoked on a dedicated background dispatcher
      thread, NOT on the caller's thread or the event loop. Use this for hooks
      that may block, perform IO, or do substantial aggregation work.
    - The ``span`` passed to ``on_start`` and ``on_end`` is a deep-copied,
      read-only snapshot of the span as it was when the event occurred. The
      ``span`` passed to ``on_update`` is a lightweight snapshot with metadata,
      current attributes/status, and only the events from the current
      ``SpanUpdateEvent``. Mutating it has no effect and the snapshot will not
      reflect later updates to the same span.
    - Do NOT call ``start_span`` / ``get_current_span`` / ``add_event`` or other
      tracer APIs from inside a hook. The dispatcher thread does not carry the
      caller's context, and enqueuing a new event from the (sole) worker thread
      while the bounded queue is full would deadlock the dispatcher. Use the
      ``span`` snapshot argument instead.
    - Background hooks are invoked strictly FIFO in background queue order.
      Inline hooks run immediately on the caller's thread. An exception raised
      by one processor is logged and isolated; it does not skip sibling
      processors using the same dispatcher.
    """

    dispatch_mode: Literal['background', 'inline'] = 'inline'
    processor_scopes: frozenset[str] = frozenset({DEFAULT_SCOPE})

    def on_start(self, span: Span, event: SpanStartEvent) -> None:
        """Called when a span is started.

        Args:
            span: Read-only deep-copied snapshot of the newly created span
            event: The SpanStartEvent that triggered the creation
        """
        pass

    def on_update(self, span: Span, event: SpanUpdateEvent) -> None:
        """Called when a span is updated.

        Args:
            span: Read-only lightweight snapshot of the span with updates applied;
                ``span.events`` contains only events from this update.
            event: The SpanUpdateEvent that triggered the update
        """
        pass

    def on_end(self, span: Span, event: SpanEndEvent) -> None:
        """Called when a span is ended.

        Args:
            span: Read-only deep-copied snapshot of the completed span (with end_time set)
            event: The SpanEndEvent that triggered the end
        """
        pass

    def finalize(self) -> None:
        """Called when the processor context exits.

        Use this for cleanup operations like flushing buffers, saving files,
        or releasing resources. This is called automatically by processor_context,
        AFTER the background dispatcher queue has been drained. Inline hooks run
        immediately, so all on_start/on_update/on_end hooks for this context are
        guaranteed to have run before finalize().

        Subclasses should override this method if they need cleanup logic.
        """
        pass

    def get_result(self) -> Any:
        """Get the result of this processor after finalization.

        This method is called after finalize() to retrieve any results
        the processor has collected. Override this in subclasses to
        return processor-specific results.

        Returns:
            Any result the processor wants to expose, or None by default.
        """
        return None

    def snapshot_result(self) -> Any:
        """Get a read-only snapshot of the processor's current state.

        Unlike get_result(), this method:
        - Can be called at any time during execution (before finalize())
        - Does NOT finalize or mutate processor state
        - Is purely read-only

        Returns None by default. Subclasses override this to expose
        their current collected data.
        """
        return None


def validate_span_processor(obj: Any, _name: str, _meta: dict) -> None:
    """Validate that an object is a valid SpanProcessor class."""
    if not isinstance(obj, type):
        raise TypeError(f'SpanProcessor must be a class, got {type(obj)}')
    if not issubclass(obj, SpanProcessor):
        raise TypeError(f'SpanProcessor must be a subclass of SpanProcessor, got {obj}')


span_processor_registry = Registry[type[SpanProcessor]]('span_processor', validate_span_processor)
