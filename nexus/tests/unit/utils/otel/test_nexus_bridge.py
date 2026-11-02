from __future__ import annotations

import asyncio

from contextvars import copy_context
from dataclasses import dataclass
from typing import Any

import pytest

from opentelemetry import context, trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from nexus.trace.controller import SpanController
from nexus.trace.events import SpanEndEvent, SpanStartEvent, SpanUpdateEvent
from nexus.trace.processors import SpanProcessor
from nexus.trace.types import Event, Span, Status


@dataclass
class BridgeHarness:
    bridge: Any
    controller: SpanController
    exporter: InMemorySpanExporter
    tracer: Any


@pytest.fixture
def bridge_harness(monkeypatch: pytest.MonkeyPatch) -> BridgeHarness:
    from nexus.utils.otel.nexus_bridge import NexusOTelBridge

    monkeypatch.setenv('NEXUS_OTEL_ENABLED', 'true')
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    bridge = NexusOTelBridge(tracer_provider=provider)
    controller = SpanController(otel_bridge=bridge)
    return BridgeHarness(
        bridge=bridge,
        controller=controller,
        exporter=exporter,
        tracer=provider.get_tracer('tests.trace.otel.bridge'),
    )


def _span(name: str, *, span_id: str, parent_span_id: str = '', **attributes: Any) -> Span:
    return Span(
        trace_id='nexus-trace',
        span_id=span_id,
        parent_span_id=parent_span_id,
        name=name,
        start_time_unix_nano=1,
        attributes=attributes,
    )


def test_local_start_update_end_exports_one_mirror(bridge_harness: BridgeHarness) -> None:
    start = SpanStartEvent(
        parent_span_id='',
        name='agent.inference',
        attributes={'type': 'agent_inference'},
        events=[Event(time_unix_nano=2, name='safe', attributes={'rpc.method': 'start'})],
    )
    bridge_harness.controller.add_event(start)
    bridge_harness.controller.add_event(
        SpanUpdateEvent(
            span_id=start.span_id,
            attributes={'gen_ai.request.model': 'gpt-5.5'},
            events=[Event(time_unix_nano=3, name='updated', attributes={'rpc.method': 'update'})],
            status=Status(code='OK'),
        )
    )
    bridge_harness.controller.add_event(SpanEndEvent(span_id=start.span_id, time_unix_nano=4))

    [span] = bridge_harness.exporter.get_finished_spans()
    assert span.name == 'agent.inference'
    assert span.instrumentation_scope.name == 'nexus.utils.otel.nexus_bridge'
    assert span.attributes['nexus.span_id'] == start.span_id
    assert span.attributes['gen_ai.request.model'] == 'gpt-5.5'
    assert [(event.name, event.attributes['rpc.method']) for event in span.events] == [
        ('safe', 'start'),
        ('updated', 'update'),
    ]
    assert span.status.status_code.name == 'OK'
    assert span.start_time == start.time_unix_nano
    assert span.end_time == 4


@pytest.mark.parametrize(
    ('updates', 'expected_status'),
    [
        ([Status(code='OK'), Status(code='ERROR', message='late failure')], 'ERROR'),
        ([Status(code='ERROR', message='temporary'), Status(code='UNSET')], 'UNSET'),
    ],
)
def test_mirror_applies_only_the_final_nexus_status_at_end(
    bridge_harness: BridgeHarness,
    updates: list[Status],
    expected_status: str,
) -> None:
    start = SpanStartEvent(parent_span_id='', name='status.transition')
    bridge_harness.controller.add_event(start)

    for status in updates:
        bridge_harness.controller.add_event(SpanUpdateEvent(span_id=start.span_id, status=status))
        assert bridge_harness.bridge._mirrors[start.span_id].status.status_code.name == 'UNSET'

    bridge_harness.controller.add_event(SpanEndEvent(span_id=start.span_id))
    [finished] = bridge_harness.exporter.get_finished_spans()
    assert finished.status.status_code.name == expected_status


def test_initial_ok_is_not_terminal_when_span_context_body_fails(bridge_harness: BridgeHarness) -> None:
    from nexus.trace import span_context

    previous_controller = SpanController._instance
    SpanController._instance = bridge_harness.controller

    def fail_after_starting_ok() -> None:
        with span_context('status.failure', status=Status(code='OK')) as stub:
            assert bridge_harness.bridge._mirrors[stub.span_id].status.status_code.name == 'UNSET'
            raise RuntimeError('late failure')

    try:
        with pytest.raises(RuntimeError, match='late failure'):
            fail_after_starting_ok()
    finally:
        SpanController._instance = previous_controller

    [finished] = bridge_harness.exporter.get_finished_spans()
    assert finished.status.status_code.name == 'ERROR'


def test_end_refreshes_final_safe_attributes_from_nexus_state(bridge_harness: BridgeHarness) -> None:
    start = SpanStartEvent(parent_span_id='', name='final.attributes')
    bridge_harness.controller.add_event(start)
    bridge_harness.controller.get_span(start.span_id).attributes['gen_ai.response.model'] = 'final-model'

    bridge_harness.controller.add_event(SpanEndEvent(span_id=start.span_id))

    [finished] = bridge_harness.exporter.get_finished_spans()
    assert finished.attributes['gen_ai.response.model'] == 'final-model'


def test_run_scope_update_event_is_mirrored_without_accumulating_on_nexus_span(
    bridge_harness: BridgeHarness,
) -> None:
    start = SpanStartEvent(parent_span_id='', name='run.live')
    bridge_harness.controller.add_event(start, processor_scope='run')
    update = SpanUpdateEvent(
        span_id=start.span_id,
        events=[Event(time_unix_nano=7, name='run.event', attributes={'rpc.method': 'run.update'})],
    )

    bridge_harness.controller.add_event(update, processor_scope='run')

    assert bridge_harness.controller.get_span(start.span_id).events == []
    bridge_harness.controller.add_event(SpanEndEvent(span_id=start.span_id), processor_scope='run')
    [span] = bridge_harness.exporter.get_finished_spans()
    assert [(event.name, event.attributes['rpc.method']) for event in span.events] == [('run.event', 'run.update')]


def test_replay_dispatches_but_does_not_mirror(bridge_harness: BridgeHarness) -> None:
    starts: list[str] = []

    class Recorder(SpanProcessor):
        def on_start(self, span: Span, event: SpanStartEvent) -> None:  # noqa: ARG002
            starts.append(span.name)

    bridge_harness.controller.add_processor(Recorder())
    start = SpanStartEvent(parent_span_id='', name='runtime.tool')

    bridge_harness.controller.add_event(start, origin='replay')
    bridge_harness.controller.add_event(SpanEndEvent(span_id=start.span_id), origin='replay')

    assert starts == ['runtime.tool']
    assert bridge_harness.exporter.get_finished_spans() == ()


def test_local_mirror_parent_beats_current_remote_context(bridge_harness: BridgeHarness) -> None:
    with bridge_harness.tracer.start_as_current_span('remote-parent') as remote_parent:
        parent = _span('nexus-parent', span_id='nexus-parent')
        bridge_harness.bridge.start_mirror(parent)
        child = _span('nexus-child', span_id='nexus-child', parent_span_id=parent.span_id)
        bridge_harness.bridge.start_mirror(child)
        bridge_harness.bridge.end_mirror(child, SpanEndEvent(span_id=child.span_id))
        bridge_harness.bridge.end_mirror(parent, SpanEndEvent(span_id=parent.span_id))

    spans = {span.name: span for span in bridge_harness.exporter.get_finished_spans()}
    assert spans['nexus-child'].parent.span_id == spans['nexus-parent'].context.span_id
    assert spans['nexus-parent'].parent.span_id == remote_parent.get_span_context().span_id


def test_current_remote_context_is_parent_without_local_mirror(bridge_harness: BridgeHarness) -> None:
    with bridge_harness.tracer.start_as_current_span('remote-parent') as remote_parent:
        child = _span('nexus-child', span_id='nexus-child', parent_span_id='remote-nexus-id')
        bridge_harness.bridge.start_mirror(child)
        bridge_harness.bridge.end_mirror(child, SpanEndEvent(span_id=child.span_id))

    spans = [span for span in bridge_harness.exporter.get_finished_spans() if span.name == 'nexus-child']
    assert spans[0].parent.span_id == remote_parent.get_span_context().span_id


def test_duplicate_start_does_not_replace_or_export_a_second_mirror(
    bridge_harness: BridgeHarness,
    caplog: pytest.LogCaptureFixture,
) -> None:
    first = _span('first', span_id='same')
    duplicate = _span('duplicate', span_id='same')

    first_mirror = bridge_harness.bridge.start_mirror(first)
    duplicate_mirror = bridge_harness.bridge.start_mirror(duplicate)
    bridge_harness.bridge.end_mirror(first, SpanEndEvent(span_id=first.span_id))

    assert duplicate_mirror is first_mirror
    assert [span.name for span in bridge_harness.exporter.get_finished_spans()] == ['first']
    assert 'duplicate OTel mirror START' in caplog.text


def test_failed_otel_end_removes_mirror_state(monkeypatch: pytest.MonkeyPatch) -> None:
    from nexus.utils.otel.nexus_bridge import NexusOTelBridge

    monkeypatch.setenv('NEXUS_OTEL_ENABLED', 'true')

    class FailingSpan:
        def set_attributes(self, attributes: dict[str, Any]) -> None:  # noqa: ARG002
            pass

        def add_event(self, *args: Any, **kwargs: Any) -> None:
            pass

        def set_status(self, status: Any) -> None:
            pass

        def end(self, *, end_time: int) -> None:
            raise RuntimeError(f'end failed at {end_time}')

    failing_span = FailingSpan()

    class Tracer:
        def start_span(self, *args: Any, **kwargs: Any) -> FailingSpan:
            return failing_span

    class Provider:
        def get_tracer(self, scope: str) -> Tracer:
            assert scope == 'nexus.utils.otel.nexus_bridge'
            return Tracer()

    bridge = NexusOTelBridge(tracer_provider=Provider())
    span = _span('failing', span_id='failing')
    bridge.start_mirror(span)

    with pytest.raises(RuntimeError, match='end failed'):
        bridge.end_mirror(span, SpanEndEvent(span_id=span.span_id, time_unix_nano=9))

    assert span.span_id not in bridge._mirrors


@pytest.mark.asyncio
async def test_concurrent_task_activation_is_isolated(bridge_harness: BridgeHarness) -> None:
    parent = _span('nexus-parent', span_id='parent')
    child = _span('nexus-child', span_id='child', parent_span_id='parent')
    bridge_harness.bridge.start_mirror(parent)
    bridge_harness.bridge.start_mirror(child)
    bridge_harness.bridge.activate_mirror(parent.span_id)
    parent_current = trace.get_current_span().get_span_context().span_id

    child_started = asyncio.Event()
    release_child = asyncio.Event()

    async def child_task() -> None:
        assert trace.get_current_span().get_span_context().span_id == parent_current
        assert bridge_harness.bridge.deactivate_mirror(parent.span_id) is False
        assert trace.get_current_span().get_span_context().span_id == parent_current
        assert bridge_harness.bridge.activate_mirror(child.span_id) is True
        child_current = trace.get_current_span().get_span_context().span_id
        assert child_current != parent_current
        child_started.set()
        await release_child.wait()
        assert bridge_harness.bridge.deactivate_mirror(child.span_id) is True
        assert trace.get_current_span().get_span_context().span_id == parent_current

    task = asyncio.create_task(child_task())
    await child_started.wait()
    assert trace.get_current_span().get_span_context().span_id == parent_current
    release_child.set()
    await task
    assert bridge_harness.bridge.deactivate_mirror(parent.span_id) is True
    assert trace.get_current_span().get_span_context().is_valid is False
    bridge_harness.bridge.end_mirror(child, SpanEndEvent(span_id=child.span_id))
    bridge_harness.bridge.end_mirror(parent, SpanEndEvent(span_id=parent.span_id))


@pytest.mark.asyncio
async def test_sibling_tasks_restore_their_inherited_parent_independently(
    bridge_harness: BridgeHarness,
) -> None:
    parent = _span('parent', span_id='parent')
    children = [
        _span('first', span_id='first', parent_span_id=parent.span_id),
        _span('second', span_id='second', parent_span_id=parent.span_id),
    ]
    bridge_harness.bridge.start_mirror(parent)
    for child in children:
        bridge_harness.bridge.start_mirror(child)
    assert bridge_harness.bridge.activate_mirror(parent.span_id) is True
    parent_context = trace.get_current_span().get_span_context()

    async def activate_child(child: Span) -> None:
        assert trace.get_current_span().get_span_context() == parent_context
        assert bridge_harness.bridge.activate_mirror(child.span_id) is True
        await asyncio.sleep(0)
        assert trace.get_current_span().get_span_context() != parent_context
        assert bridge_harness.bridge.deactivate_mirror(child.span_id) is True
        assert trace.get_current_span().get_span_context() == parent_context

    await asyncio.gather(*(activate_child(child) for child in children))
    assert trace.get_current_span().get_span_context() == parent_context
    assert bridge_harness.bridge.deactivate_mirror(parent.span_id) is True
    for child in children:
        bridge_harness.bridge.end_mirror(child, SpanEndEvent(span_id=child.span_id))
    bridge_harness.bridge.end_mirror(parent, SpanEndEvent(span_id=parent.span_id))


def test_duplicate_end_is_idempotent(bridge_harness: BridgeHarness) -> None:
    span = _span('single', span_id='single')
    bridge_harness.bridge.start_mirror(span)

    bridge_harness.bridge.end_mirror(span, SpanEndEvent(span_id=span.span_id))
    bridge_harness.bridge.end_mirror(span, SpanEndEvent(span_id=span.span_id))

    assert [finished.name for finished in bridge_harness.exporter.get_finished_spans()] == ['single']


def test_activation_record_can_restore_context_after_mirror_is_removed(bridge_harness: BridgeHarness) -> None:
    parent = _span('nexus-parent', span_id='parent')
    bridge_harness.bridge.start_mirror(parent)
    assert bridge_harness.bridge.activate_mirror(parent.span_id) is True

    bridge_harness.bridge.end_mirror(parent, SpanEndEvent(span_id=parent.span_id))

    assert bridge_harness.bridge.deactivate_mirror(parent.span_id) is True
    assert trace.get_current_span().get_span_context().is_valid is False


def test_copied_python_context_never_consumes_the_original_activation(
    bridge_harness: BridgeHarness,
) -> None:
    parent = _span('nexus-parent', span_id='parent')
    bridge_harness.bridge.start_mirror(parent)
    assert bridge_harness.bridge.activate_mirror(parent.span_id) is True
    parent_current = trace.get_current_span().get_span_context().span_id

    copied = copy_context()
    assert copied.run(bridge_harness.bridge.deactivate_mirror, parent.span_id) is True
    assert copied.run(trace.get_current_span).get_span_context().is_valid is False
    assert trace.get_current_span().get_span_context().span_id == parent_current

    assert bridge_harness.bridge.deactivate_mirror(parent.span_id) is True
    assert trace.get_current_span().get_span_context().is_valid is False
    bridge_harness.bridge.end_mirror(parent, SpanEndEvent(span_id=parent.span_id))


def test_disabled_bridge_does_not_import_or_touch_otel(monkeypatch: pytest.MonkeyPatch) -> None:
    from nexus.utils.otel.nexus_bridge import NexusOTelBridge

    monkeypatch.setenv('NEXUS_OTEL_ENABLED', 'false')
    bridge = NexusOTelBridge(tracer_provider=object())
    span = _span('disabled', span_id='disabled')

    assert bridge.start_mirror(span) is None
    assert bridge.activate_mirror(span.span_id) is False
    assert bridge.deactivate_mirror(span.span_id) is False
    bridge.update_mirror(span, SpanUpdateEvent(span_id=span.span_id))
    bridge.end_mirror(span, SpanEndEvent(span_id=span.span_id))


def test_best_effort_activation_api_contains_bridge_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    from nexus.utils.otel.nexus_bridge import NexusOTelBridge

    bridge = NexusOTelBridge()

    def fail(_span_id: str) -> bool:
        raise RuntimeError('bridge failure')

    monkeypatch.setattr(bridge, 'activate_mirror', fail)
    monkeypatch.setattr(bridge, 'deactivate_mirror', fail)

    assert bridge.try_activate_mirror('span-1') is False
    assert bridge.try_deactivate_mirror('span-1') is False


def test_nexus_span_is_current_otel_parent(bridge_harness: BridgeHarness) -> None:
    from nexus.trace import span_context

    previous_controller = SpanController._instance
    SpanController._instance = bridge_harness.controller
    try:
        with span_context('nexus.parent'):
            with bridge_harness.tracer.start_as_current_span('native.child'):
                pass
    finally:
        SpanController._instance = previous_controller

    spans = {span.name: span for span in bridge_harness.exporter.get_finished_spans()}
    assert spans['native.child'].parent.span_id == spans['nexus.parent'].context.span_id


def test_end_dispatch_failure_restores_current_mirror_for_retry(bridge_harness: BridgeHarness) -> None:
    from nexus.trace import end_span, start_span

    previous_controller = SpanController._instance
    SpanController._instance = bridge_harness.controller
    original_dispatch = bridge_harness.controller._dispatch
    current_during_failed_end: list[bool] = []

    stub = start_span('nexus.parent')
    mirror_context = trace.get_current_span().get_span_context()
    assert mirror_context.is_valid is True

    def fail_end(hook: str, *args: Any, **kwargs: Any) -> None:
        if hook == 'on_end':
            current_during_failed_end.append(trace.get_current_span().get_span_context().is_valid)
            raise RuntimeError('dispatch failed')
        original_dispatch(hook, *args, **kwargs)

    bridge_harness.controller._dispatch = fail_end
    try:
        with pytest.raises(RuntimeError, match='dispatch failed'):
            end_span(stub.span_id)

        assert current_during_failed_end == [False]
        assert trace.get_current_span().get_span_context() == mirror_context
        assert stub.span_id in bridge_harness.controller.active_span_dict
        assert stub.span_id in bridge_harness.bridge._mirrors

        bridge_harness.controller._dispatch = original_dispatch
        end_span(stub.span_id)
        assert trace.get_current_span().get_span_context().is_valid is False
        assert stub.span_id not in bridge_harness.controller.active_span_dict
        assert stub.span_id not in bridge_harness.bridge._mirrors
    finally:
        bridge_harness.controller._dispatch = original_dispatch
        SpanController._instance = previous_controller


def test_ending_non_current_parent_does_not_detach_child(bridge_harness: BridgeHarness) -> None:
    from nexus.trace import end_span, start_span

    previous_controller = SpanController._instance
    SpanController._instance = bridge_harness.controller
    try:
        parent = start_span('parent')
        parent_otel_context = trace.get_current_span().get_span_context()
        assert parent_otel_context.is_valid is True
        child = start_span('child')
        child_otel_context = trace.get_current_span().get_span_context()
        assert child_otel_context.is_valid is True
        assert child_otel_context != parent_otel_context

        end_span(parent.span_id)
        assert trace.get_current_span().get_span_context() == child_otel_context

        end_span(child.span_id)
        assert trace.get_current_span().get_span_context() == parent_otel_context

        end_span(parent.span_id)
        assert trace.get_current_span().get_span_context().is_valid is False
    finally:
        SpanController._instance = previous_controller

    assert sorted(span.name for span in bridge_harness.exporter.get_finished_spans()) == ['child', 'parent']


def test_nested_same_mirror_activation_is_strict_lifo(bridge_harness: BridgeHarness) -> None:
    parent = _span('parent', span_id='parent')
    bridge_harness.bridge.start_mirror(parent)

    assert bridge_harness.bridge.activate_mirror(parent.span_id) is True
    assert bridge_harness.bridge.activate_mirror(parent.span_id) is True
    assert bridge_harness.bridge.deactivate_mirror(parent.span_id) is True
    assert trace.get_current_span().get_span_context().is_valid is True
    assert bridge_harness.bridge.deactivate_mirror(parent.span_id) is True
    assert trace.get_current_span().get_span_context().is_valid is False
    assert bridge_harness.bridge.deactivate_mirror(parent.span_id) is False

    bridge_harness.bridge.end_mirror(parent, SpanEndEvent(span_id=parent.span_id))


@pytest.fixture(autouse=True)
def _restore_otel_context() -> Any:
    before = context.get_current()
    yield
    assert context.get_current() == before
