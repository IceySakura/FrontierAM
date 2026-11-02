from __future__ import annotations

import pytest

from nexus.trace.controller import SpanController
from nexus.trace.events import SpanEndEvent, SpanStartEvent, SpanUpdateEvent
from nexus.trace.processors.base import SpanProcessor
from nexus.trace.types import Event, Span


class NonDeepcopyable:
    def __deepcopy__(self, memo: dict[object, object]) -> object:  # noqa: ARG002
        raise TypeError('cannot deepcopy')


def test_update_hook_span_does_not_include_historical_events() -> None:
    controller = SpanController()
    update_event_counts: list[tuple[int, int]] = []
    end_event_counts: list[int] = []

    class Rec(SpanProcessor):
        def on_update(self, span: Span, event: SpanUpdateEvent) -> None:
            update_event_counts.append((len(span.events), len(event.events or [])))

        def on_end(self, span: Span, event: SpanEndEvent) -> None:  # noqa: ARG002
            end_event_counts.append(len(span.events))

    controller.add_processor(Rec())

    ev = SpanStartEvent(parent_span_id='', name='op')
    controller.add_event(ev)
    controller.add_event(
        SpanUpdateEvent(
            span_id=ev.span_id,
            events=[Event(time_unix_nano=1, name='response.output_text.delta', attributes={'delta': 'a'})],
        )
    )
    controller.add_event(
        SpanUpdateEvent(
            span_id=ev.span_id,
            events=[Event(time_unix_nano=2, name='response.output_text.delta', attributes={'delta': 'b'})],
        )
    )
    controller.add_event(SpanEndEvent(span_id=ev.span_id))

    assert update_event_counts == [(1, 1), (1, 1)]
    assert end_event_counts == [2]


def test_run_scope_span_does_not_accumulate_lifecycle_events() -> None:
    controller = SpanController()
    event = SpanStartEvent(parent_span_id='', name='run.live')
    controller.add_event(event, processor_scope='run')

    for index in range(3):
        controller.add_event(
            SpanUpdateEvent(
                span_id=event.span_id,
                events=[Event(time_unix_nano=index, name='run.event', attributes={'index': index})],
            ),
            processor_scope='run',
        )

    assert controller.get_span(event.span_id).events == []


def test_update_hook_span_uses_shallow_attribute_snapshot() -> None:
    controller = SpanController()
    value = NonDeepcopyable()
    seen: list[object] = []

    class Rec(SpanProcessor):
        def on_update(self, span: Span, event: SpanUpdateEvent) -> None:  # noqa: ARG002
            seen.append(span.attributes['conn'])

    controller.add_processor(Rec())

    ev = SpanStartEvent(parent_span_id='', name='op')
    controller.add_event(ev)
    controller.add_event(
        SpanUpdateEvent(
            span_id=ev.span_id,
            attributes={'conn': value},
            events=[Event(time_unix_nano=1, name='response.output_text.delta', attributes={'delta': 'a'})],
        )
    )

    assert seen == [value]


def test_end_hook_still_requires_deepcopyable_attributes() -> None:
    controller = SpanController()
    value = NonDeepcopyable()

    class Rec(SpanProcessor):
        pass

    controller.add_processor(Rec())

    ev = SpanStartEvent(parent_span_id='', name='op')
    controller.add_event(ev)
    controller.add_event(
        SpanUpdateEvent(
            span_id=ev.span_id,
            attributes={'conn': value},
        )
    )

    with pytest.raises(TypeError, match='cannot deepcopy'):
        controller.add_event(SpanEndEvent(span_id=ev.span_id))
