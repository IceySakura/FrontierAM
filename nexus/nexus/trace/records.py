from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from nexus.trace.events import SpanEndEvent, SpanEvent, SpanStartEvent, SpanUpdateEvent
from nexus.trace.types import Event, Span, Status


class SpanEventRecord(BaseModel):
    """Durable trace event journal record."""

    model_config = ConfigDict(frozen=True)

    record_id: int
    event_type: Literal['start', 'update', 'end']
    time_unix_nano: int
    trace_id: str
    span_id: str
    parent_span_id: str | None = None
    event_data: dict[str, Any]


def span_event_record_from_span(
    *,
    record_id: int,
    span: Span,
    event: SpanEvent,
) -> SpanEventRecord:
    """Build a durable record from a span processor callback."""
    event_data = event.model_dump(mode='json')
    event_data['trace_id'] = span.trace_id
    event_data['parent_span_id'] = span.parent_span_id
    return SpanEventRecord(
        record_id=record_id,
        event_type=_event_type(event),
        time_unix_nano=event.time_unix_nano,
        trace_id=span.trace_id,
        span_id=span.span_id,
        parent_span_id=span.parent_span_id,
        event_data=event_data,
    )


def span_event_from_record(record: SpanEventRecord) -> SpanEvent:
    """Reconstruct the span event stored in a durable journal record."""
    data = dict(record.event_data)
    if record.event_type == 'start':
        return SpanStartEvent(
            span_id=record.span_id,
            time_unix_nano=record.time_unix_nano,
            parent_span_id=record.parent_span_id or data.get('parent_span_id') or '',
            trace_id=record.trace_id or data.get('trace_id') or '',
            name=str(data.get('name') or 'unknown'),
            attributes=dict(data.get('attributes') or {}),
            events=[Event.model_validate(item) for item in data.get('events') or []],
            status=Status.model_validate(data.get('status') or {}),
        )
    if record.event_type == 'update':
        return SpanUpdateEvent(
            span_id=record.span_id,
            time_unix_nano=record.time_unix_nano,
            attributes=data.get('attributes'),
            events=[Event.model_validate(item) for item in data.get('events') or []] or None,
            status=Status.model_validate(data['status']) if data.get('status') else None,
        )
    if record.event_type == 'end':
        return SpanEndEvent(span_id=record.span_id, time_unix_nano=record.time_unix_nano)
    raise ValueError(f'unknown span event record type: {record.event_type}')


def _event_type(event: SpanEvent) -> Literal['start', 'update', 'end']:
    if isinstance(event, SpanStartEvent) or event.type == 'START':
        return 'start'
    if isinstance(event, SpanUpdateEvent) or event.type == 'UPDATE':
        return 'update'
    if isinstance(event, SpanEndEvent) or event.type == 'END':
        return 'end'
    raise ValueError(f'unknown span event type: {event.type}')
