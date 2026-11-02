from __future__ import annotations

import logging

import pytest

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from nexus.trace import (
    Event,
    RemoteTraceContext,
    SpanUpdateEvent,
    Status,
    run_trace_context,
    span_context,
    update_span,
    with_remote_trace_context,
)
from nexus.trace.controller import SpanController
from nexus.trace.processors import SpanProcessor
from nexus.trace.types import Span
from nexus.utils.otel.nexus_bridge import NexusOTelBridge
from nexus.utils.otel.propagation import use_remote_context


@pytest.fixture
def otel_trace(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv('NEXUS_OTEL_ENABLED', 'true')
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    controller = SpanController(otel_bridge=NexusOTelBridge(tracer_provider=provider))
    previous_controller = SpanController._instance
    SpanController._instance = controller
    try:
        yield provider.get_tracer('tests.remote-context'), exporter
    finally:
        SpanController._instance = previous_controller


def test_remote_trace_context_carries_run_attributes_from_current_span() -> None:
    with run_trace_context(run_id='run-1', run_span_id='run-span-1', run_kind='single'):
        with span_context('client.op'):
            context = RemoteTraceContext.from_current_span()

    assert context is not None
    assert context.run_id == 'run-1'
    assert context.run_span_id == 'run-span-1'
    assert context.run_kind == 'single'


def test_with_remote_trace_context_injects_run_attributes_into_child_spans() -> None:
    seen = {}

    with with_remote_trace_context(
        'trace-1',
        'parent-span',
        run_id='run-1',
        run_span_id='run-span-1',
        run_kind='single',
    ):
        with span_context('server.op') as span:
            assert span.span is not None
            seen.update(span.span.attributes)

    assert seen['_run_id'] == 'run-1'
    assert seen['_run_span_id'] == 'run-span-1'
    assert seen['_run_kind'] == 'single'


def test_task_span_starts_a_new_root_outside_run_context() -> None:
    with span_context('outer') as outer_stub:
        with span_context('task.independent', attributes={'type': 'task'}) as task_stub:
            outer = outer_stub.span
            task = task_stub.span

    assert outer is not None
    assert task is not None
    assert task.parent_span_id == ''
    assert task.trace_id != outer.trace_id


def test_remote_run_context_keeps_task_in_the_remote_trace() -> None:
    with with_remote_trace_context(
        'trace-1',
        'parent-span',
        run_id='run-1',
        run_span_id='run-span-1',
        run_kind='single',
    ):
        with span_context('task.remote', attributes={'type': 'task'}) as task_stub:
            task = task_stub.span

    assert task is not None
    assert task.parent_span_id == 'parent-span'
    assert task.trace_id == 'trace-1'


def test_span_context_preserves_an_explicit_error_status() -> None:
    with span_context('operation') as span_stub:
        span = span_stub.span
        span_stub.update(status=Status(code='ERROR', message='operation failed'))

    assert span is not None
    assert span.status == Status(code='ERROR', message='operation failed')


def test_remote_trace_context_round_trips_otel_carrier(otel_trace) -> None:
    otel_tracer, exporter = otel_trace

    with span_context('nexus-client'):
        with otel_tracer.start_as_current_span('native-parent') as parent:
            remote = RemoteTraceContext.from_current_span()

    assert remote is not None
    assert remote.otel_carrier is not None
    round_tripped = RemoteTraceContext.model_validate(remote.model_dump())
    assert round_tripped.otel_carrier == remote.otel_carrier
    with use_remote_context(remote.otel_carrier):
        with otel_tracer.start_as_current_span('native-child'):
            pass

    spans = {span.name: span for span in exporter.get_finished_spans()}
    assert spans['native-child'].parent.span_id == parent.get_span_context().span_id


def test_remote_carrier_is_fallback_parent_for_server_nexus_span(otel_trace) -> None:
    otel_tracer, exporter = otel_trace
    with otel_tracer.start_as_current_span('remote-parent'):
        with span_context('nexus-client'):
            remote = RemoteTraceContext.from_current_span()

    assert remote is not None
    assert remote.otel_carrier is not None
    with with_remote_trace_context(
        remote.trace_id,
        remote.parent_span_id,
        otel_carrier=remote.otel_carrier,
    ):
        with span_context('nexus-server'):
            pass

    spans = {span.name: span for span in exporter.get_finished_spans()}
    assert spans['nexus-server'].parent.span_id == spans['nexus-client'].context.span_id


def test_remote_carrier_does_not_override_an_existing_otel_parent(otel_trace) -> None:
    otel_tracer, exporter = otel_trace
    with otel_tracer.start_as_current_span('payload-parent'):
        with span_context('payload-nexus'):
            remote = RemoteTraceContext.from_current_span()

    assert remote is not None
    with otel_tracer.start_as_current_span('http-server-parent'):
        with with_remote_trace_context(
            remote.trace_id,
            remote.parent_span_id,
            otel_carrier=remote.otel_carrier,
        ):
            with span_context('nexus-server'):
                pass

    spans = {span.name: span for span in exporter.get_finished_spans()}
    assert spans['nexus-server'].parent.span_id == spans['http-server-parent'].context.span_id


def test_remote_parent_updates_dispatch_without_missing_otel_mirror_warning(
    otel_trace,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _, exporter = otel_trace
    controller = SpanController.get_instance_sync()
    updates: list[str] = []

    class Recorder(SpanProcessor):
        def on_update(self, span: Span, event: SpanUpdateEvent) -> None:  # noqa: ARG002
            updates.extend(item.name for item in event.events or ())

    controller.add_processor(Recorder())
    caplog.set_level(logging.WARNING)

    with with_remote_trace_context('trace-1', 'remote-parent'):
        update_span(
            'remote-parent',
            events=[Event(time_unix_nano=1, name='agent.stream', attributes={})],
        )

    assert updates == ['agent.stream']
    assert exporter.get_finished_spans() == ()
    assert 'OTel mirror for Nexus span remote-parent is missing' not in caplog.text


def test_offline_replay_marks_every_controller_event_as_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    from nexus.trace.tracer import replay_span_events

    controller = SpanController()
    origins: list[str] = []
    original_add_event = controller.add_event

    def record_origin(event, **kwargs):  # noqa: ANN001, ANN003
        origins.append(kwargs.get('origin'))
        original_add_event(event, **kwargs)

    monkeypatch.setattr(controller, 'add_event', record_origin)
    monkeypatch.setattr(SpanController, 'get_instance_sync', lambda: controller)
    replay_span_events(
        [
            {
                'event_type': 'start',
                'span_id': 'span-1',
                'trace_id': 'trace-1',
                'parent_span_id': '',
                'time_unix_nano': 1,
                'event_data': {'name': 'replayed'},
            },
            {'event_type': 'end', 'span_id': 'span-1', 'time_unix_nano': 2},
        ]
    )

    assert origins == ['replay', 'replay']
