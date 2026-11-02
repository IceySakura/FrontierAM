"""Live API projection built from the unified raw run trace stream."""

from __future__ import annotations

import asyncio
import os

from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import replace
from typing import Any

from nexus.runner.trace import (
    RunTraceContext,
    RunTraceCursorExpired,
    RunTraceProjection,
    RunTraceProjectionStream,
    RunTraceProjector,
    RunTraceSink,
    RunTraceStreamEvent,
)
from nexus.trace.events import SpanEndEvent, SpanEvent, SpanStartEvent, SpanUpdateEvent
from nexus.trace.records import SpanEventRecord, span_event_from_record
from nexus.trace.types import Span


# Per-string byte/char budget applied to projected live ``trace.event`` payloads.
# Long agent tasks can emit ``openai_completion`` spans whose attributes carry
# whole-file tool results and multi-thousand-character reasoning traces. Streamed
# verbatim they inflate a single SSE connection to tens of MB and stall the shared
# executor loop while deep-copying and JSON-serialising each batch. The raw records
# remain fully persisted in the trace journal (tdsql); only the reader-facing live
# projection is capped. ``0`` (or a negative override) disables truncation.
DEFAULT_LIVE_TRACE_MAX_STRING_CHARS = 16_384
MAX_LIVE_TRACE_MAX_STRING_CHARS = 4_194_304
_LIVE_TRACE_MAX_STRING_CHARS_ENV = 'NEXUS_LIVE_TRACE_MAX_STRING_CHARS'


def _resolve_default_max_string_chars() -> int:
    """Read the runtime override for the per-string live projection budget.

    Returns ``0`` to disable truncation. Falls back to
    :data:`DEFAULT_LIVE_TRACE_MAX_STRING_CHARS` when the env variable is unset,
    empty, or non-integer; negative values (and ``0``) disable truncation.
    """
    raw = os.environ.get(_LIVE_TRACE_MAX_STRING_CHARS_ENV)
    if raw is None or raw.strip() == '':
        return DEFAULT_LIVE_TRACE_MAX_STRING_CHARS
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_LIVE_TRACE_MAX_STRING_CHARS
    if value <= 0:
        return 0
    return min(value, MAX_LIVE_TRACE_MAX_STRING_CHARS)


def _truncate_live_value(value: Any, *, max_chars: int) -> Any:
    """Return a copy of ``value`` with overlong strings replaced by a marker.

    Recurses through dicts and lists so nested ``openai_completion`` attributes
    (message content, ``reasoning_content``, tool results) are bounded. A new
    structure is always returned so the projection engine's retained span state
    is never mutated in place.
    """
    if isinstance(value, str):
        if len(value) > max_chars:
            omitted = len(value) - max_chars
            return f'{value[:max_chars]}…[truncated {omitted} chars]'
        return value
    if isinstance(value, dict):
        return {key: _truncate_live_value(item, max_chars=max_chars) for key, item in value.items()}
    if isinstance(value, list):
        return [_truncate_live_value(item, max_chars=max_chars) for item in value]
    return value


class LiveRunTraceProjector(RunTraceProjector):
    """Project raw run and ordinary span callbacks into the live API view."""

    def __init__(self, *, max_string_chars: int | None = None) -> None:
        self._max_string_chars = (
            _resolve_default_max_string_chars() if max_string_chars is None else max(0, max_string_chars)
        )

    @property
    def max_string_chars(self) -> int:
        return self._max_string_chars

    def project(
        self,
        context: RunTraceContext,
        span: Span,
        event: SpanEvent,
        record: SpanEventRecord,
    ) -> Sequence[RunTraceProjection]:
        if context.run_kind != 'live':
            return ()
        if span.name == 'run.live':
            projections = self._project_run_update(event)
        else:
            projections = self._project_trace_event(context, span, event, record)
        return self._cap_projection_payloads(projections)

    def _cap_projection_payloads(self, projections: list[RunTraceProjection]) -> list[RunTraceProjection]:
        if self._max_string_chars <= 0 or not projections:
            return projections
        return [
            replace(projection, data=_truncate_live_value(projection.data, max_chars=self._max_string_chars))
            for projection in projections
        ]

    @staticmethod
    def _project_run_update(event: SpanEvent) -> list[RunTraceProjection]:
        if not isinstance(event, SpanUpdateEvent):
            return []
        result: list[RunTraceProjection] = []
        for item in event.events or []:
            if item.name != 'run.event':
                continue
            event_type = item.attributes.get('event_type')
            if not isinstance(event_type, str):
                continue
            event_data = item.attributes.get('event_data') or {}
            if not isinstance(event_data, dict):
                event_data = {'value': event_data}
            result.append(
                RunTraceProjection(
                    type=event_type,
                    data=dict(event_data),
                    time_unix_nano=item.time_unix_nano,
                )
            )
        return result

    def _project_trace_event(
        self,
        context: RunTraceContext,
        span: Span,
        event: SpanEvent,
        record: SpanEventRecord,
    ) -> list[RunTraceProjection]:
        if isinstance(event, SpanStartEvent):
            if span.name != 'openai_completion':
                return []
            payload = event.model_dump(mode='json')
        elif isinstance(event, SpanUpdateEvent):
            return [
                RunTraceProjection(
                    type='trace.event',
                    data=self._trace_event_data(
                        context,
                        span,
                        {**item.model_dump(mode='json'), 'type': item.name},
                    ),
                    time_unix_nano=item.time_unix_nano,
                )
                for item in event.events or []
            ]
        elif isinstance(event, SpanEndEvent):
            if span.name != 'openai_completion':
                return []
            payload = event.model_dump(mode='json')
            payload['attributes'] = span.attributes
            payload['status'] = span.status.model_dump(mode='json')
        else:
            return []
        return [
            RunTraceProjection(
                type='trace.event',
                data=self._trace_event_data(context, span, payload),
                time_unix_nano=record.time_unix_nano,
            )
        ]

    @staticmethod
    def _trace_event_data(context: RunTraceContext, span: Span, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            'run_id': context.run_id,
            'event': {
                'span_id': span.span_id,
                'span_name': span.name,
                'trace_id': span.trace_id,
                'event': payload,
            },
        }


class RunTraceProjectionEngine:
    """Reconstruct span state and assign projection cursors without controller side effects."""

    def __init__(self, context: RunTraceContext, projector: RunTraceProjector | None = None) -> None:
        self.context = context
        self.projector = projector or LiveRunTraceProjector()
        self._spans: dict[str, Span] = {}
        self._next_id = 1

    def feed(self, record: SpanEventRecord) -> list[RunTraceStreamEvent]:
        span, event = self._apply_record(record)
        if span is None or event is None:
            return []
        result: list[RunTraceStreamEvent] = []
        for projection in self.projector.project(self.context, span, event, record):
            result.append(
                RunTraceStreamEvent(
                    type=projection.type,
                    data=projection.data,
                    time_unix_nano=projection.time_unix_nano,
                    id=self._next_id,
                )
            )
            self._next_id += 1
        if record.event_type == 'end':
            self._spans.pop(record.span_id, None)
        return result

    def _apply_record(self, record: SpanEventRecord) -> tuple[Span | None, SpanEvent | None]:
        event = span_event_from_record(record)
        if isinstance(event, SpanStartEvent):
            span = Span(
                trace_id=record.trace_id,
                span_id=record.span_id,
                parent_span_id=record.parent_span_id or '',
                name=event.name,
                start_time_unix_nano=record.time_unix_nano,
                attributes=dict(event.attributes),
                events=list(event.events),
                status=event.status,
            )
            self._spans[record.span_id] = span
            return span.model_copy(deep=True), event

        span = self._spans.get(record.span_id)
        if span is None:
            return None, None
        if isinstance(event, SpanUpdateEvent):
            if event.attributes:
                span.attributes.update(event.attributes)
            if event.status:
                span.status = event.status
            # Update projections only read span identity (span_id / name / trace_id)
            # and the update's own ``events`` — never ``attributes``. Deep-copying the
            # retained span here would clone the *entire* attribute payload on every
            # streaming delta. For an ``openai_completion`` span that payload holds the
            # full request ``inputs`` (an inlined image or long chat history can be many
            # MB), so a single long streaming run performs tens of thousands of multi-MB
            # deep copies on the shared executor loop, starving live SSE ``collect``.
            # A shallow copy is sufficient and keeps the retained span isolated from the
            # projection (attributes are shared by reference but never mutated in place).
            snapshot = span.model_copy(update={'events': list(event.events or [])})
            return snapshot, event
        if isinstance(event, SpanEndEvent):
            span.end_time_unix_nano = record.time_unix_nano
            return span.model_copy(deep=True), event
        return None, None


class LiveRunTraceView(RunTraceSink, RunTraceProjectionStream):
    """Bounded active live projection fed with raw records as an internal sink."""

    def __init__(self, *, max_events: int = 10_000) -> None:
        if max_events < 1:
            raise ValueError('max_events must be at least 1')
        self._events: deque[RunTraceStreamEvent] = deque(maxlen=max_events)
        self._condition = asyncio.Condition()
        self._engine: RunTraceProjectionEngine | None = None
        self._started = False
        self._closed = False

    @property
    def latest_id(self) -> int:
        return self._events[-1].id if self._events else 0

    async def start(self, context: RunTraceContext) -> None:
        if self._started:
            raise RuntimeError('live run trace view is already started')
        self._engine = RunTraceProjectionEngine(context)
        self._started = True

    async def on_event(self, context: RunTraceContext, event: SpanEventRecord) -> None:  # noqa: ARG002
        if not self._started or self._closed or self._engine is None:
            raise RuntimeError('live run trace view is not active')
        projected = self._engine.feed(event)
        if not projected:
            return
        async with self._condition:
            self._events.extend(projected)
            self._condition.notify_all()

    async def close(self, context: RunTraceContext) -> None:  # noqa: ARG002
        async with self._condition:
            self._closed = True
            self._condition.notify_all()

    def snapshot_after(self, cursor: int, *, limit: int | None = None) -> list[RunTraceStreamEvent]:
        if cursor < 0:
            raise ValueError('cursor must not be negative')
        if limit is not None and limit < 1:
            raise ValueError('limit must be at least 1')
        if self._events and cursor < self._events[0].id - 1:
            raise RunTraceCursorExpired(f'live projection cursor {cursor} is older than {self._events[0].id - 1}')
        events = [event for event in self._events if event.id > cursor]
        return events[:limit] if limit is not None else events

    async def wait_after(
        self,
        cursor: int,
        *,
        timeout: float | None = None,
        limit: int | None = None,
    ) -> list[RunTraceStreamEvent]:
        available = self.snapshot_after(cursor, limit=limit)
        if available or self._closed:
            return available

        async def wait() -> None:
            async with self._condition:
                await self._condition.wait_for(lambda: self.latest_id > cursor or self._closed)

        try:
            if timeout is None:
                await wait()
            else:
                await asyncio.wait_for(wait(), timeout=timeout)
        except asyncio.TimeoutError:
            return []
        return self.snapshot_after(cursor, limit=limit)


def project_live_trace_records(
    context: RunTraceContext,
    records: Iterable[SpanEventRecord],
    *,
    cursor: int = 0,
    limit: int | None = None,
) -> list[RunTraceStreamEvent]:
    """Project a complete persisted raw journal and then apply the live cursor."""
    if cursor < 0:
        raise ValueError('cursor must not be negative')
    if limit is not None and limit < 1:
        raise ValueError('limit must be at least 1')
    engine = RunTraceProjectionEngine(context)
    result: list[RunTraceStreamEvent] = []
    for record in records:
        for event in engine.feed(record):
            if event.id <= cursor:
                continue
            result.append(event)
            if limit is not None and len(result) >= limit:
                return result
    return result
