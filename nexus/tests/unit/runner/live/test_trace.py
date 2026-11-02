from __future__ import annotations

import asyncio

import pytest

from nexus.runner.trace import RunTraceSink
from nexus.trace.controller import SpanController
from nexus.trace.events import SpanStartEvent, SpanUpdateEvent
from nexus.trace.records import SpanEventRecord
from nexus.trace.types import Event


class RecordingSink(RunTraceSink):
    def __init__(self) -> None:
        self.started = False
        self.events: list[SpanEventRecord] = []
        self.closed = False

    async def start(self, context) -> None:  # noqa: ANN001
        self.started = True

    async def on_event(self, context, event: SpanEventRecord) -> None:  # noqa: ANN001
        self.events.append(event)

    async def close(self, context) -> None:  # noqa: ANN001
        self.closed = True


class FailingStartSink(RecordingSink):
    async def start(self, context) -> None:  # noqa: ANN001
        raise OSError('cannot start')


class FailingMetadataSink(RecordingSink):
    def metadata(self):  # noqa: ANN201
        raise OSError('metadata failed')


class MutatingSink(RecordingSink):
    async def on_event(self, context, event: SpanEventRecord) -> None:  # noqa: ANN001
        event.event_data['name'] = 'mutated'
        await super().on_event(context, event)


class BlockingCloseSink(RecordingSink):
    def __init__(self) -> None:
        super().__init__()
        self.close_started = asyncio.Event()
        self.allow_close = asyncio.Event()

    async def close(self, context) -> None:  # noqa: ANN001
        self.close_started.set()
        await self.allow_close.wait()
        await super().close(context)


class BlockingEventSink(RecordingSink):
    def __init__(self) -> None:
        super().__init__()
        self.event_started = asyncio.Event()
        self.allow_event = asyncio.Event()

    async def on_event(self, context, event: SpanEventRecord) -> None:  # noqa: ANN001
        self.event_started.set()
        await self.allow_event.wait()
        await super().on_event(context, event)


def _controller_with_processor():
    from nexus.runner.trace import RunTraceProcessor

    SpanController._instance = None
    controller = SpanController.get_instance_sync()
    processor = RunTraceProcessor()
    controller.add_processor(processor)
    return controller, processor


@pytest.mark.asyncio
async def test_run_trace_session_activate_makes_task_a_direct_child() -> None:
    from nexus.runner.trace import RunTraceSession
    from nexus.trace import span_context

    controller, processor = _controller_with_processor()
    sink = RecordingSink()
    session = RunTraceSession(
        run_id='run-parent',
        run_kind='single',
        processor=processor,
        span_controller=controller,
        backend_sinks={'test': sink},
    )

    await session.start()
    run_span = controller.get_span(session.run_span_id)
    with session.activate():
        with span_context('task.child', attributes={'type': 'task'}) as task_stub:
            task_span = task_stub.span
    await session.close()

    assert task_span is not None
    assert task_span.parent_span_id == session.run_span_id
    assert task_span.trace_id == run_span.trace_id


@pytest.mark.asyncio
async def test_run_session_activation_restores_outer_otel_context_after_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from nexus.runner.trace import RunTraceProcessor, RunTraceSession
    from nexus.utils.otel.nexus_bridge import NexusOTelBridge

    monkeypatch.setenv('NEXUS_OTEL_ENABLED', 'true')
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    controller = SpanController(otel_bridge=NexusOTelBridge(tracer_provider=provider))
    previous_controller = SpanController._instance
    SpanController._instance = controller
    processor = RunTraceProcessor()
    controller.add_processor(processor)
    session = RunTraceSession(
        run_id='run-otel',
        run_kind='live',
        processor=processor,
        span_controller=controller,
        backend_sinks={'test': RecordingSink()},
    )
    otel_tracer = provider.get_tracer('tests.run-session')

    try:
        with otel_tracer.start_as_current_span('outer') as outer:
            await session.start()
            nexus_run_span = controller.get_span(session.run_span_id)
            assert nexus_run_span.attributes['run_id'] == 'run-otel'
            assert nexus_run_span.attributes['run_kind'] == 'live'
            assert nexus_run_span.attributes['_run_id'] == 'run-otel'
            assert nexus_run_span.attributes['_run_span_id'] == session.run_span_id
            assert nexus_run_span.attributes['_run_kind'] == 'live'
            run_mirror = controller.otel_bridge._mirrors[session.run_span_id]
            with session.activate():
                assert trace.get_current_span().get_span_context() == run_mirror.get_span_context()
                await session.close()
                assert session.run_span_id not in controller.otel_bridge._mirrors
                assert trace.get_current_span().get_span_context() == run_mirror.get_span_context()
            assert trace.get_current_span().get_span_context() == outer.get_span_context()
    finally:
        SpanController._instance = previous_controller

    spans = {span.name: span for span in exporter.get_finished_spans()}
    assert spans['run.live'].parent.span_id == spans['outer'].context.span_id
    assert spans['run.live'].attributes['nexus.run.id'] == 'run-otel'
    assert spans['run.live'].attributes['nexus.run.span_id'] == session.run_span_id
    assert spans['run.live'].attributes['nexus.run.kind'] == 'live'


@pytest.mark.asyncio
async def test_session_fans_raw_records_to_backend_and_live_view() -> None:
    from nexus.runner.live.trace import LiveRunTraceView
    from nexus.runner.trace import RunTraceSession

    controller, processor = _controller_with_processor()
    sink = RecordingSink()
    view = LiveRunTraceView()
    session = RunTraceSession(
        run_id='run-1',
        run_kind='live',
        processor=processor,
        span_controller=controller,
        backend_sinks={'test': sink},
        internal_sinks=[view],
    )

    await session.start()
    await session.emit('run.status', status='running')
    await session.close()

    assert [record.record_id for record in sink.events] == list(range(1, len(sink.events) + 1))
    assert [event.type for event in view.snapshot_after(0)] == ['run.status']
    assert sink.closed is True


@pytest.mark.asyncio
async def test_processor_requires_explicit_run_attributes_for_normal_spans() -> None:
    from nexus.runner.trace import RunTraceSession

    controller, processor = _controller_with_processor()
    sink = RecordingSink()
    session = RunTraceSession(
        run_id='run-1',
        run_kind='single',
        processor=processor,
        span_controller=controller,
        backend_sinks={'test': sink},
    )
    await session.start()
    run_trace_id = controller.get_span(session.run_span_id).trace_id

    controller.add_event(SpanStartEvent(parent_span_id='', trace_id=run_trace_id, name='unattributed', attributes={}))
    attributed = SpanStartEvent(
        parent_span_id=session.run_span_id,
        trace_id=run_trace_id,
        name='attributed',
        attributes={'_run_id': 'run-1', '_run_span_id': session.run_span_id, '_run_kind': 'single'},
    )
    controller.add_event(attributed)
    await processor.flush_run(session.run_span_id)
    await session.close()

    names = [record.event_data.get('name') for record in sink.events if record.event_type == 'start']
    assert 'unattributed' not in names
    assert 'attributed' in names


@pytest.mark.asyncio
async def test_same_run_id_sessions_are_isolated_by_run_span_id() -> None:
    from nexus.runner.trace import RunTraceSession

    controller, processor = _controller_with_processor()
    first_sink = RecordingSink()
    second_sink = RecordingSink()
    first = RunTraceSession(
        run_id='shared',
        run_kind='single',
        run_span_id='run-span-1',
        processor=processor,
        span_controller=controller,
        backend_sinks={'first': first_sink},
    )
    second = RunTraceSession(
        run_id='shared',
        run_kind='single',
        run_span_id='run-span-2',
        processor=processor,
        span_controller=controller,
        backend_sinks={'second': second_sink},
    )

    await first.start()
    await second.start()
    await first.emit('run.status', status='first')
    await second.emit('run.status', status='second')
    await first.close()
    await second.close()

    assert all(record.span_id != 'run-span-2' for record in first_sink.events)
    assert all(record.span_id != 'run-span-1' for record in second_sink.events)


@pytest.mark.asyncio
async def test_processor_serializes_cross_thread_callbacks() -> None:
    from nexus.runner.trace import RunTraceSession

    controller, processor = _controller_with_processor()
    sink = RecordingSink()
    session = RunTraceSession(
        run_id='run-1',
        run_kind='single',
        run_span_id='run-span-1',
        processor=processor,
        span_controller=controller,
        backend_sinks={'test': sink},
    )
    await session.start()

    def emit_updates() -> None:
        for index in range(20):
            controller.add_event(
                SpanUpdateEvent(
                    span_id=session.run_span_id,
                    events=[Event(time_unix_nano=index, name='run.event', attributes={'index': index})],
                ),
                processor_scope='run',
            )

    await asyncio.to_thread(emit_updates)
    await session.close()

    assert [record.record_id for record in sink.events] == list(range(1, len(sink.events) + 1))


@pytest.mark.asyncio
async def test_processor_fails_run_when_bounded_queue_overflows() -> None:
    from nexus.runner.trace import RunTraceContext, RunTraceProcessingError, RunTraceProcessor

    controller = SpanController()
    processor = RunTraceProcessor(max_queued_events=1)
    controller.add_processor(processor)
    sink = BlockingEventSink()
    context = RunTraceContext(run_id='run-1', run_span_id='run-span-1', run_kind='single')
    await processor.register_run(context=context, sinks=[sink])
    controller.add_event(
        SpanStartEvent(
            span_id='run-span-1',
            parent_span_id='',
            name='run.single',
            attributes={'run_id': 'run-1', 'run_kind': 'single'},
        ),
        processor_scope='run',
    )
    await sink.event_started.wait()
    for index in range(2):
        controller.add_event(
            SpanUpdateEvent(
                span_id='run-span-1',
                events=[Event(time_unix_nano=index, name='run.event', attributes={'index': index})],
            ),
            processor_scope='run',
        )
    sink.allow_event.set()

    with pytest.raises(RunTraceProcessingError):
        await processor.close_run('run-span-1')
    assert processor._states_by_run_span_id == {}


@pytest.mark.asyncio
async def test_sink_start_failure_rolls_back_started_sinks_and_state() -> None:
    from nexus.runner.trace import RunTraceSession, RunTraceSinkError

    controller, processor = _controller_with_processor()
    started = RecordingSink()
    failed = FailingStartSink()
    session = RunTraceSession(
        run_id='run-1',
        run_kind='single',
        processor=processor,
        span_controller=controller,
        backend_sinks={'started': started, 'failed': failed},
    )

    with pytest.raises(RunTraceSinkError, match='cannot start'):
        await session.start()

    assert started.closed is True
    assert failed.closed is True
    assert processor._states_by_run_span_id == {}


@pytest.mark.asyncio
async def test_sink_metadata_failure_fails_close_before_envelope_is_available() -> None:
    from nexus.runner.trace import RunTraceSession, RunTraceSinkError

    controller, processor = _controller_with_processor()
    session = RunTraceSession(
        run_id='run-1',
        run_kind='single',
        processor=processor,
        span_controller=controller,
        backend_sinks={'failed': FailingMetadataSink()},
    )
    await session.start()

    with pytest.raises(RunTraceSinkError, match='metadata failed'):
        await session.close()
    with pytest.raises(RuntimeError, match='unavailable'):
        session.artifact_metadata()


@pytest.mark.asyncio
async def test_sink_record_mutation_does_not_affect_other_sinks() -> None:
    from nexus.runner.trace import RunTraceSession

    controller, processor = _controller_with_processor()
    mutating = MutatingSink()
    recording = RecordingSink()
    session = RunTraceSession(
        run_id='run-1',
        run_kind='single',
        processor=processor,
        span_controller=controller,
        backend_sinks={'mutating': mutating, 'recording': recording},
    )

    await session.start()
    await session.close()

    assert mutating.events[0].event_data['name'] == 'mutated'
    assert recording.events[0].event_data['name'] == 'run.single'


@pytest.mark.asyncio
async def test_cancelled_close_caller_does_not_interrupt_sink_cleanup() -> None:
    from nexus.runner.trace import RunTraceSession

    controller, processor = _controller_with_processor()
    sink = BlockingCloseSink()
    session = RunTraceSession(
        run_id='run-1',
        run_kind='single',
        processor=processor,
        span_controller=controller,
        backend_sinks={'test': sink},
    )
    await session.start()
    close_caller = asyncio.create_task(session.close())
    await sink.close_started.wait()
    close_caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await close_caller
    sink.allow_close.set()
    await session.close()

    assert sink.closed is True
    assert processor._states_by_run_span_id == {}
