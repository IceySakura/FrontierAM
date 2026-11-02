from __future__ import annotations

from logging import getLogger
from typing import Any, Literal, cast

from nexus.trace.context import context_processors as _context_processors
from nexus.trace.dispatcher import BackgroundSpanDispatcher, HookItem, InlineSpanDispatcher
from nexus.trace.events import SpanEndEvent, SpanEvent, SpanStartEvent, SpanUpdateEvent
from nexus.trace.processors import SpanProcessor
from nexus.trace.scopes import DEFAULT_SCOPE, RUN_SCOPE, validate_processor_scope, validate_processor_scopes
from nexus.trace.types import Span
from nexus.utils.otel.nexus_bridge import NexusOTelBridge, get_nexus_bridge
from nexus.utils.singleton import Singleton


logger = getLogger(__name__)


class SpanController(Singleton):
    def __init__(self, *, otel_bridge: NexusOTelBridge | Any | None = None):
        self.active_span_dict: dict[str, Span] = {}
        self.active_span_scopes: dict[str, str] = {}
        self._local_span_ids: set[str] = set()
        self.processors: list[SpanProcessor] = []
        self.inline_dispatcher = InlineSpanDispatcher()
        self.background_dispatcher = BackgroundSpanDispatcher()
        self.otel_bridge = get_nexus_bridge() if otel_bridge is None else otel_bridge

    def add_processor(self, processor: SpanProcessor) -> None:
        """Add a span processor to receive callbacks."""
        validate_processor_scopes(getattr(processor, 'processor_scopes', frozenset({DEFAULT_SCOPE})))
        self.processors.append(processor)

    def remove_processor(self, processor: SpanProcessor) -> None:
        """Remove a span processor."""
        if processor in self.processors:
            self.processors.remove(processor)

    def get_all_processors(self, *, processor_scope: str = DEFAULT_SCOPE) -> list[SpanProcessor]:
        """Get all processors including global and context-local ones."""
        processor_scope = validate_processor_scope(processor_scope)
        global_processors = self.processors[:]
        context_processors = _context_processors.get() or []
        processors = []
        for processor in global_processors + context_processors:
            processor_scopes = validate_processor_scopes(
                getattr(processor, 'processor_scopes', frozenset({DEFAULT_SCOPE}))
            )
            if processor_scope in processor_scopes:
                processors.append(processor)
        return processors

    def _dispatch(
        self,
        hook: Literal['on_start', 'on_update', 'on_end'],
        span: Span,
        event: SpanEvent,
        *,
        processor_scope: str,
    ) -> None:
        """Snapshot the span and dispatch processor hooks.

        The processor list (including context-local processors) is resolved here,
        on the caller's thread, so the background dispatcher does not depend on the
        caller's contextvar context. Start/end hooks receive a full deep-copied
        span snapshot. Update hooks receive a lightweight snapshot whose
        ``events`` only contain the current update event batch; the complete
        accumulated event history is available on the end hook.
        """
        processors = self.get_all_processors(processor_scope=processor_scope)
        if not processors:
            return
        inline_processors = [
            processor for processor in processors if getattr(processor, 'dispatch_mode', 'inline') == 'inline'
        ]
        background_processors = [
            processor for processor in processors if getattr(processor, 'dispatch_mode', 'inline') != 'inline'
        ]
        if inline_processors:
            snapshot = self._snapshot_for_dispatch(hook, span, event)
            self.inline_dispatcher.submit(HookItem(hook=hook, span=snapshot, event=event, processors=inline_processors))
        if background_processors:
            snapshot = self._snapshot_for_dispatch(hook, span, event)
            self.background_dispatcher.submit(
                HookItem(hook=hook, span=snapshot, event=event, processors=background_processors)
            )

    def _snapshot_for_dispatch(
        self,
        hook: Literal['on_start', 'on_update', 'on_end'],
        span: Span,
        event: SpanEvent,
    ) -> Span:
        if hook != 'on_update':
            return span.model_copy(deep=True)

        return self._snapshot_for_update_hook(span, cast(SpanUpdateEvent, event))

    def _snapshot_for_update_hook(self, span: Span, event: SpanUpdateEvent) -> Span:
        """Build a lightweight update snapshot without historical span events."""
        return Span(
            trace_id=span.trace_id,
            span_id=span.span_id,
            parent_span_id=span.parent_span_id,
            name=span.name,
            start_time_unix_nano=span.start_time_unix_nano,
            end_time_unix_nano=span.end_time_unix_nano,
            attributes=dict(span.attributes),
            events=list(event.events or []),
            status=span.status,
        )

    def _mirror_safely(self, operation: str, *args: Any) -> None:
        try:
            getattr(self.otel_bridge, operation)(*args)
        except Exception:
            logger.warning('OTel bridge %s failed; Nexus trace lifecycle is unchanged', operation, exc_info=True)

    def add_event(
        self,
        event: SpanEvent,
        *,
        processor_scope: str | None = None,
        origin: Literal['local', 'replay'] = 'local',
    ) -> None:
        """Apply one Nexus event; replay events dispatch normally but are not mirrored to OTel."""
        if origin not in {'local', 'replay'}:
            raise ValueError(f'unknown span event origin: {origin!r}')
        if event.type == 'START':
            event = cast(SpanStartEvent, event)
            resolved_scope = validate_processor_scope(processor_scope or DEFAULT_SCOPE)
            span = Span(
                trace_id=event.trace_id,
                span_id=event.span_id,
                parent_span_id=event.parent_span_id,
                name=event.name,
                attributes=event.attributes,
                events=event.events,
                status=event.status,
                start_time_unix_nano=event.time_unix_nano,
            )
            self.active_span_dict[event.span_id] = span
            self.active_span_scopes[event.span_id] = resolved_scope
            if origin == 'local':
                self._local_span_ids.add(event.span_id)
            else:
                self._local_span_ids.discard(event.span_id)
            try:
                self._dispatch('on_start', span, event, processor_scope=resolved_scope)
            except BaseException:
                self.active_span_dict.pop(event.span_id, None)
                self.active_span_scopes.pop(event.span_id, None)
                self._local_span_ids.discard(event.span_id)
                raise
            if origin == 'local':
                self._mirror_safely('start_mirror', span)

        elif event.type == 'UPDATE':
            event = cast(SpanUpdateEvent, event)
            if event.span_id not in self.active_span_dict:
                logger.warning(f'span {event.span_id} not found, skip update')
                return
            resolved_scope = self._resolve_active_scope(event.span_id, processor_scope)
            span = self.active_span_dict[event.span_id]
            if event.attributes:
                span.attributes.update(event.attributes)
            if event.events and resolved_scope != RUN_SCOPE:
                span.events.extend(event.events)
            if event.status:
                span.status = event.status
            self._dispatch('on_update', span, event, processor_scope=resolved_scope)
            if origin == 'local' and event.span_id in self._local_span_ids:
                self._mirror_safely('update_mirror', span, event)

        elif event.type == 'END':
            event = cast(SpanEndEvent, event)
            if event.span_id not in self.active_span_dict:
                logger.warning(f'span {event.span_id} not found, skip end')
                return
            resolved_scope = self._resolve_active_scope(event.span_id, processor_scope)
            span = self.active_span_dict[event.span_id]
            span.end_time_unix_nano = event.time_unix_nano
            # Snapshot+enqueue BEFORE removing from the active dict.
            self._dispatch('on_end', span, event, processor_scope=resolved_scope)
            if origin == 'local' and event.span_id in self._local_span_ids:
                self._mirror_safely('end_mirror', span, event)
            self.active_span_dict.pop(event.span_id)
            self.active_span_scopes.pop(event.span_id, None)
            self._local_span_ids.discard(event.span_id)

    def get_span(self, span_id: str) -> Span:
        return self.active_span_dict[span_id]

    def _resolve_active_scope(self, span_id: str, processor_scope: str | None) -> str:
        active_scope = self.active_span_scopes.get(span_id, DEFAULT_SCOPE)
        if processor_scope is None:
            return validate_processor_scope(active_scope)
        requested_scope = validate_processor_scope(processor_scope)
        if requested_scope != active_scope:
            raise ValueError(
                f'cannot change processor_scope for span {span_id!r}: '
                f'started with {active_scope!r}, got {requested_scope!r}'
            )
        return requested_scope

    def shutdown(self, timeout: float = 5.0) -> None:
        """Drain pending hooks and finalize global processors.

        Intended to be called on process/runner exit. Context-local processors
        registered via ``processor_context`` are finalized by that context's exit,
        not here.
        """
        if not self.background_dispatcher.barrier(timeout=timeout):
            logger.warning('SpanController.shutdown: background dispatcher barrier timed out after %ss', timeout)
        for processor in self.processors:
            try:
                processor.finalize()
            except Exception as e:
                logger.error(f'Error finalizing processor {processor}: {e}')
