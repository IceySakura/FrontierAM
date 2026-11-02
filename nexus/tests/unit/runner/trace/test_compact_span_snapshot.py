"""Regression coverage for :func:`_compact_span_snapshot`.

An ``openai_completion`` span carries the full request ``inputs`` (an inlined image or
long chat history is regularly multi-MB) on ``attributes``. The runner-level snapshot
that :class:`RunTraceProcessor` places on every span callback used to deep-copy those
attributes on every streaming delta, even though the sink write path never reads them.
For a long streaming run this saturated the inline dispatcher and starved live SSE
``collect``. These tests lock in that streaming updates take an identity-only snapshot,
while start/end still capture the full state (so sinks can persist it).
"""

from __future__ import annotations

from nexus.runner.trace.trace import _compact_span_snapshot
from nexus.trace.events import SpanEndEvent, SpanStartEvent, SpanUpdateEvent
from nexus.trace.records import span_event_record_from_span
from nexus.trace.types import Event, Span, Status


class _CountingStr(str):
    """A ``str`` subclass whose ``__deepcopy__`` increments a class-level counter."""

    deep_copies: int = 0

    def __deepcopy__(self, memo: dict) -> _CountingStr:  # noqa: D401 - protocol
        type(self).deep_copies += 1
        return self


def _reset_counter() -> None:
    _CountingStr.deep_copies = 0


def _span(*, attributes: dict) -> Span:
    return Span(
        trace_id='trace-1',
        span_id='span-oc-1',
        parent_span_id='run-span-1',
        name='openai_completion',
        start_time_unix_nano=1_000_000_000,
        attributes=attributes,
    )


def test_update_snapshot_drops_retained_span_attributes_and_events() -> None:
    """SpanUpdate snapshots must carry only span identity, never the retained attributes."""
    _reset_counter()
    huge_input = _CountingStr('IMG_PAYLOAD')
    span = _span(attributes={'inputs': {'image': huge_input}})
    # Also seed a large retained-events list; the snapshot must not clone it either.
    span.events = [Event(name=f'evt-{i}', time_unix_nano=i, attributes={}) for i in range(5)]

    update = SpanUpdateEvent(
        span_id=span.span_id,
        time_unix_nano=2_000_000_000,
        events=[Event(name='chat.completion.delta', time_unix_nano=2_000_000_000, attributes={'delta': 'x'})],
    )

    snapshot = _compact_span_snapshot(span, update)

    # Identity must still round-trip so the sink record can be built.
    assert snapshot.trace_id == span.trace_id
    assert snapshot.span_id == span.span_id
    assert snapshot.parent_span_id == span.parent_span_id
    assert snapshot.name == span.name
    # Retained attributes / events must NOT be cloned onto the snapshot at all.
    assert snapshot.attributes == {}
    assert snapshot.events == []
    # And, crucially, the retained multi-MB attribute value was never deep-copied.
    assert _CountingStr.deep_copies == 0


def test_update_snapshot_produces_a_valid_sink_record() -> None:
    """Downstream sink code reads span identity from the snapshot; verify it still works."""
    span = _span(attributes={'inputs': {'image': 'IMG'}})
    update = SpanUpdateEvent(
        span_id=span.span_id,
        time_unix_nano=2_000_000_000,
        events=[Event(name='chat.completion.delta', time_unix_nano=2_000_000_000, attributes={'delta': 'y'})],
    )

    snapshot = _compact_span_snapshot(span, update)
    record = span_event_record_from_span(record_id=42, span=snapshot, event=update)

    assert record.record_id == 42
    assert record.event_type == 'update'
    assert record.trace_id == span.trace_id
    assert record.span_id == span.span_id
    assert record.parent_span_id == span.parent_span_id
    # The update's own delta events (from ``event.model_dump()``) must be preserved.
    assert record.event_data['events'][0]['name'] == 'chat.completion.delta'
    assert record.event_data['events'][0]['attributes'] == {'delta': 'y'}


def test_start_snapshot_still_deep_copies_retained_span() -> None:
    """SpanStart is O(spans), not O(deltas), so the deep copy is still safe and required."""
    _reset_counter()
    huge_input = _CountingStr('IMG_PAYLOAD')
    span = _span(attributes={'inputs': {'image': huge_input}})

    start = SpanStartEvent(
        span_id=span.span_id,
        time_unix_nano=1_000_000_000,
        parent_span_id=span.parent_span_id,
        trace_id=span.trace_id,
        name=span.name,
        attributes=dict(span.attributes),
        events=[],
        status=Status(),
    )

    snapshot = _compact_span_snapshot(span, start)

    assert snapshot.attributes == {'inputs': {'image': huge_input}}
    # The retained attribute value was deep-copied exactly once — start still captures it.
    assert _CountingStr.deep_copies == 1


def test_end_snapshot_still_deep_copies_retained_span() -> None:
    """SpanEnd is O(spans), so keep the deep copy — sinks persist the final attributes/events."""
    _reset_counter()
    huge_input = _CountingStr('IMG_PAYLOAD')
    span = _span(attributes={'inputs': {'image': huge_input}})
    span.end_time_unix_nano = 3_000_000_000

    end = SpanEndEvent(span_id=span.span_id, time_unix_nano=3_000_000_000)

    snapshot = _compact_span_snapshot(span, end)

    assert snapshot.attributes == {'inputs': {'image': huge_input}}
    assert _CountingStr.deep_copies == 1


def test_streaming_deltas_never_deep_copy_the_retained_input() -> None:
    """50 streaming deltas must add zero deep copies of the retained multi-MB payload."""
    _reset_counter()
    huge_input = _CountingStr('IMG_PAYLOAD')
    span = _span(attributes={'inputs': {'image': huge_input}})

    for record_id in range(50):
        update = SpanUpdateEvent(
            span_id=span.span_id,
            time_unix_nano=(record_id + 2) * 1_000_000_000,
            events=[
                Event(
                    name='chat.completion.delta',
                    time_unix_nano=(record_id + 2) * 1_000_000_000,
                    attributes={'delta': 'x'},
                )
            ],
        )
        _compact_span_snapshot(span, update)

    assert _CountingStr.deep_copies == 0
