from __future__ import annotations

import pytest

from nexus.trace.controller import SpanController
from nexus.trace.events import SpanEndEvent, SpanStartEvent, SpanUpdateEvent
from nexus.trace.processors.base import SpanProcessor
from nexus.trace.types import Span


def test_processor_scope_routes_events_to_matching_processors() -> None:
    controller = SpanController()
    seen: list[tuple[str, str, str]] = []

    class DefaultOnly(SpanProcessor):
        processor_scopes = frozenset({'default'})

        def on_start(self, span: Span, event: SpanStartEvent) -> None:  # noqa: ARG002
            seen.append(('default', 'start', span.name))

        def on_update(self, span: Span, event: SpanUpdateEvent) -> None:  # noqa: ARG002
            seen.append(('default', 'update', span.name))

    class RunOnly(SpanProcessor):
        processor_scopes = frozenset({'run'})

        def on_start(self, span: Span, event: SpanStartEvent) -> None:  # noqa: ARG002
            seen.append(('run', 'start', span.name))

        def on_update(self, span: Span, event: SpanUpdateEvent) -> None:  # noqa: ARG002
            seen.append(('run', 'update', span.name))

    controller.add_processor(DefaultOnly())
    controller.add_processor(RunOnly())

    default_start = SpanStartEvent(parent_span_id='', name='default.op')
    controller.add_event(default_start)
    controller.add_event(SpanUpdateEvent(span_id=default_start.span_id, attributes={'phase': 'default'}))

    run_start = SpanStartEvent(parent_span_id='', name='run.live')
    controller.add_event(run_start, processor_scope='run')
    controller.add_event(SpanUpdateEvent(span_id=run_start.span_id, attributes={'phase': 'run'}))
    controller.add_event(SpanEndEvent(span_id=default_start.span_id))
    controller.add_event(SpanEndEvent(span_id=run_start.span_id))

    assert seen == [
        ('default', 'start', 'default.op'),
        ('default', 'update', 'default.op'),
        ('run', 'start', 'run.live'),
        ('run', 'update', 'run.live'),
    ]


def test_default_and_run_are_the_only_processor_scopes() -> None:
    from nexus.runner.trace import RunTraceProcessor

    assert SpanProcessor.processor_scopes == frozenset({'default'})
    assert RunTraceProcessor.processor_scopes == frozenset({'default', 'run'})


def test_unknown_processor_scope_is_rejected() -> None:
    controller = SpanController()
    start = SpanStartEvent(parent_span_id='', name='legacy.trace')

    with pytest.raises(ValueError, match='unknown processor_scope'):
        controller.add_event(start, processor_scope='trace')

    with pytest.raises(ValueError, match='unknown processor_scope'):
        controller.get_all_processors(processor_scope='trace')


def test_unknown_span_event_origin_is_rejected_before_state_changes() -> None:
    controller = SpanController()
    start = SpanStartEvent(parent_span_id='', name='legacy.trace')

    with pytest.raises(ValueError, match='unknown span event origin'):
        controller.add_event(start, origin='forwarded')  # type: ignore[arg-type]

    assert start.span_id not in controller.active_span_dict


def test_context_processor_scope_is_validated_before_registration() -> None:
    from nexus.trace.tracer import processor_context

    class InvalidScope(SpanProcessor):
        processor_scopes = frozenset({'legacy'})

    with pytest.raises(ValueError, match='unknown processor_scope values'):
        with processor_context([InvalidScope()]):
            raise AssertionError('context must not be entered')


def test_start_dispatch_failure_does_not_leave_active_span() -> None:
    controller = SpanController()
    start = SpanStartEvent(parent_span_id='', name='default.op')

    def fail_dispatch(*_args, **_kwargs):
        raise RuntimeError('dispatch failed')

    controller._dispatch = fail_dispatch
    with pytest.raises(RuntimeError, match='dispatch failed'):
        controller.add_event(start)

    assert start.span_id not in controller.active_span_dict
    assert start.span_id not in controller.active_span_scopes


def test_span_scope_cannot_change_after_start() -> None:
    controller = SpanController()
    start = SpanStartEvent(parent_span_id='', name='default.op')
    controller.add_event(start)

    with pytest.raises(ValueError, match='cannot change processor_scope'):
        controller.add_event(SpanUpdateEvent(span_id=start.span_id), processor_scope='run')

    with pytest.raises(ValueError, match='cannot change processor_scope'):
        controller.add_event(SpanEndEvent(span_id=start.span_id), processor_scope='run')


@pytest.mark.parametrize('failure_stage', ['start', 'update', 'end'])
def test_bridge_failure_never_changes_controller_semantics(failure_stage: str) -> None:
    seen: list[tuple[str, str]] = []

    class BrokenBridge:
        def start_mirror(self, span: Span) -> None:
            if failure_stage == 'start':
                raise RuntimeError('bridge start failed')

        def update_mirror(self, span: Span, event: SpanUpdateEvent) -> None:
            if failure_stage == 'update':
                raise RuntimeError('bridge update failed')

        def end_mirror(self, span: Span, event: SpanEndEvent) -> None:
            if failure_stage == 'end':
                raise RuntimeError('bridge end failed')

    class Recorder(SpanProcessor):
        def on_start(self, span: Span, event: SpanStartEvent) -> None:  # noqa: ARG002
            seen.append(('start', span.name))

        def on_update(self, span: Span, event: SpanUpdateEvent) -> None:  # noqa: ARG002
            seen.append(('update', str(span.attributes.get('phase'))))

        def on_end(self, span: Span, event: SpanEndEvent) -> None:  # noqa: ARG002
            seen.append(('end', span.name))

    controller = SpanController(otel_bridge=BrokenBridge())
    controller.add_processor(Recorder())
    start = SpanStartEvent(parent_span_id='', name='operation')

    controller.add_event(start)
    controller.add_event(SpanUpdateEvent(span_id=start.span_id, attributes={'phase': 'updated'}))
    controller.add_event(SpanEndEvent(span_id=start.span_id))

    assert seen == [('start', 'operation'), ('update', 'updated'), ('end', 'operation')]
    assert start.span_id not in controller.active_span_dict
    assert start.span_id not in controller.active_span_scopes


def test_end_dispatch_failure_keeps_active_span_and_does_not_call_bridge() -> None:
    bridge_calls: list[str] = []

    class Bridge:
        def start_mirror(self, span: Span) -> None:
            bridge_calls.append('start')

        def update_mirror(self, span: Span, event: SpanUpdateEvent) -> None:
            bridge_calls.append('update')

        def end_mirror(self, span: Span, event: SpanEndEvent) -> None:
            bridge_calls.append('end')

    controller = SpanController(otel_bridge=Bridge())
    start = SpanStartEvent(parent_span_id='', name='operation')
    controller.add_event(start)

    def fail_dispatch(*_args, **_kwargs):
        raise RuntimeError('dispatch failed')

    controller._dispatch = fail_dispatch
    with pytest.raises(RuntimeError, match='dispatch failed'):
        controller.add_event(SpanEndEvent(span_id=start.span_id))

    assert bridge_calls == ['start']
    assert start.span_id in controller.active_span_dict
    assert start.span_id in controller.active_span_scopes
