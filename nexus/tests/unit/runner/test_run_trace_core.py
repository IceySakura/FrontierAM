from __future__ import annotations

import asyncio
import time

from typing import Any

import pytest

from pydantic import ValidationError

from nexus.trace.controller import SpanController
from nexus.trace.records import SpanEventRecord


class RecordingSink:
    def __init__(self, *, metadata: dict[str, Any] | None = None, fail_write: bool = False) -> None:
        self.events: list[SpanEventRecord] = []
        self.started = False
        self.closed = False
        self._metadata = metadata or {}
        self._fail_write = fail_write

    async def start(self, context) -> None:  # noqa: ANN001
        self.started = True

    async def on_event(self, context, event: SpanEventRecord) -> None:  # noqa: ANN001
        if self._fail_write:
            raise OSError('write failed')
        self.events.append(event)

    async def close(self, context) -> None:  # noqa: ANN001
        self.closed = True

    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata)


def _record(record_id: int, *, event_type: str = 'run.status', data: dict[str, Any] | None = None) -> SpanEventRecord:
    return SpanEventRecord(
        record_id=record_id,
        event_type='update',
        time_unix_nano=int(time.time() * 1e9),
        trace_id='trace-1',
        span_id='run-span-1',
        parent_span_id='',
        event_data={
            'type': 'UPDATE',
            'span_id': 'run-span-1',
            'events': [
                {
                    'time_unix_nano': int(time.time() * 1e9),
                    'name': 'run.event',
                    'attributes': {'event_type': event_type, 'event_data': data or {}},
                }
            ],
        },
    )


@pytest.mark.asyncio
@pytest.mark.parametrize('fail_write', [False, True])
async def test_local_trace_metrics_measure_actual_batch_io(tmp_path, monkeypatch, fail_write):
    from unittest.mock import MagicMock

    from nexus.runner.trace import RunTraceContext
    from nexus.runner.trace import local
    from nexus.utils.platform_metrics import metric_scope

    recorder = MagicMock(spec=['record'])
    context = RunTraceContext(run_id='run-1', run_span_id='run-span-1', run_kind='single')
    sink = local.RunTraceLocalSink(tmp_path, 'execution', 'assignment', 'attempt', batch_size=2)
    if fail_write:

        def fail(*args):
            raise OSError('disk write failed')

        monkeypatch.setattr(local, '_append_records', fail)
    with metric_scope(recorder, environment='test'):
        await sink.start(context)
        await sink.on_event(context, _record(1))
        await sink.on_event(context, _record(2))
        if fail_write:
            with pytest.raises(OSError, match='disk write failed'):
                await sink.close(context)
        else:
            await sink.close(context)
            assert len(sink.trace_events_path.read_text().splitlines()) == 2
    calls = {call.args[0]: call.args[1:] for call in recorder.record.call_args_list}
    assert calls['persistence_operations'][0] == 1
    assert calls['persistence_failures'][0] == int(fail_write)
    assert calls['persistence_operations'][1]['source'] == 'local_trace'
    assert calls['persistence_operations'][1]['stage'] == 'trace_write'
    assert calls['persistence_operations'][1]['environment'] == 'test'
    assert calls['persistence_duration_distribution'][0] >= 0


def test_run_trace_session_requires_a_backend_sink() -> None:
    from nexus.runner.trace import RunTraceProcessor, RunTraceSession

    with pytest.raises(ValueError, match='at least one backend sink'):
        RunTraceSession(
            run_id='run-1',
            run_kind='single',
            processor=RunTraceProcessor(),
            backend_sinks={},
        )


@pytest.mark.asyncio
async def test_session_metadata_contains_only_closed_backend_sinks() -> None:
    from nexus.runner.trace import RunTraceProcessor, RunTraceSession

    SpanController._instance = None
    controller = SpanController.get_instance_sync()
    processor = RunTraceProcessor()
    controller.add_processor(processor)
    backend = RecordingSink(metadata={'path': '/trace.jsonl'})
    internal = RecordingSink(metadata={'internal': True})
    session = RunTraceSession(
        run_id='run-1',
        run_kind='single',
        processor=processor,
        span_controller=controller,
        backend_sinks={'local': backend},
        internal_sinks=[internal],
    )

    await session.start()
    with pytest.raises(RuntimeError, match='before close'):
        session.artifact_metadata()
    await session.close()

    assert session.artifact_metadata() == {
        'version': 4,
        'run_id': 'run-1',
        'run_span_id': session.run_span_id,
        'run_kind': 'single',
        'sinks': [{'name': 'local', 'metadata': {'path': '/trace.jsonl'}}],
    }
    assert backend.closed is True
    assert internal.closed is True


@pytest.mark.asyncio
async def test_any_sink_write_failure_fails_the_session() -> None:
    from nexus.runner.trace import RunTraceProcessor, RunTraceSession, RunTraceSinkError

    SpanController._instance = None
    controller = SpanController.get_instance_sync()
    processor = RunTraceProcessor()
    controller.add_processor(processor)
    sink = RecordingSink()
    session = RunTraceSession(
        run_id='run-fail',
        run_kind='single',
        processor=processor,
        span_controller=controller,
        backend_sinks={'broken': sink},
    )

    await session.start()
    sink._fail_write = True
    with pytest.raises(RunTraceSinkError, match='write failed'):
        await session.close()


@pytest.mark.asyncio
async def test_sink_cancelled_error_during_write_fails_without_stopping_cleanup() -> None:
    from nexus.runner.trace import RunTraceProcessor, RunTraceSession, RunTraceSinkError

    class CancelledWriteSink(RecordingSink):
        cancel_write = False

        async def on_event(self, context, event: SpanEventRecord) -> None:  # noqa: ANN001
            if self.cancel_write:
                raise asyncio.CancelledError
            await super().on_event(context, event)

    SpanController._instance = None
    controller = SpanController.get_instance_sync()
    processor = RunTraceProcessor()
    controller.add_processor(processor)
    broken = CancelledWriteSink()
    healthy = RecordingSink()
    session = RunTraceSession(
        run_id='run-cancelled-write',
        run_kind='single',
        processor=processor,
        span_controller=controller,
        backend_sinks={'broken': broken, 'healthy': healthy},
    )

    await session.start()
    broken.cancel_write = True

    with pytest.raises(RunTraceSinkError, match='CancelledError'):
        await asyncio.wait_for(session.emit('run.status', status='running'), timeout=1)
    with pytest.raises(RunTraceSinkError, match='CancelledError'):
        await asyncio.wait_for(session.close(), timeout=1)

    assert broken.closed is True
    assert healthy.closed is True


@pytest.mark.asyncio
async def test_sink_cancelled_error_during_start_is_a_sink_failure() -> None:
    from nexus.runner.trace import RunTraceProcessor, RunTraceSession, RunTraceSinkError

    class CancelledStartSink(RecordingSink):
        async def start(self, context) -> None:  # noqa: ANN001
            raise asyncio.CancelledError

    SpanController._instance = None
    controller = SpanController.get_instance_sync()
    processor = RunTraceProcessor()
    controller.add_processor(processor)
    sink = CancelledStartSink()
    session = RunTraceSession(
        run_id='run-cancelled-start',
        run_kind='single',
        processor=processor,
        span_controller=controller,
        backend_sinks={'broken': sink},
    )

    with pytest.raises(RunTraceSinkError, match='start: RuntimeError: .*CancelledError'):
        await asyncio.wait_for(session.start(), timeout=1)

    assert sink.closed is True


@pytest.mark.asyncio
async def test_cancelling_session_start_cancels_blocked_sink_start() -> None:
    from nexus.runner.trace import RunTraceProcessor, RunTraceSession

    class BlockingStartSink(RecordingSink):
        def __init__(self) -> None:
            super().__init__()
            self.entered = asyncio.Event()
            self.cancelled = False

        async def start(self, context) -> None:  # noqa: ANN001
            self.entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled = True
                raise

    SpanController._instance = None
    controller = SpanController.get_instance_sync()
    processor = RunTraceProcessor()
    controller.add_processor(processor)
    sink = BlockingStartSink()
    session = RunTraceSession(
        run_id='run-cancel-start-caller',
        run_kind='single',
        processor=processor,
        span_controller=controller,
        backend_sinks={'local': sink},
    )

    start_task = asyncio.create_task(session.start())
    await sink.entered.wait()
    start_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await start_task

    assert sink.cancelled is True
    assert sink.closed is True


@pytest.mark.asyncio
async def test_sink_cancelled_error_during_close_does_not_skip_remaining_sinks() -> None:
    from nexus.runner.trace import RunTraceProcessor, RunTraceSession, RunTraceSinkError

    class CancelledCloseSink(RecordingSink):
        async def close(self, context) -> None:  # noqa: ANN001
            self.closed = True
            raise asyncio.CancelledError

    SpanController._instance = None
    controller = SpanController.get_instance_sync()
    processor = RunTraceProcessor()
    controller.add_processor(processor)
    broken = CancelledCloseSink()
    healthy = RecordingSink()
    session = RunTraceSession(
        run_id='run-cancelled-close',
        run_kind='single',
        processor=processor,
        span_controller=controller,
        backend_sinks={'broken': broken, 'healthy': healthy},
    )

    await session.start()
    with pytest.raises(RunTraceSinkError, match='CancelledError'):
        await asyncio.wait_for(session.close(), timeout=1)

    assert broken.closed is True
    assert healthy.closed is True


@pytest.mark.asyncio
async def test_sink_cancelled_error_during_metadata_is_reported_as_sink_failure() -> None:
    from nexus.runner.trace import RunTraceProcessor, RunTraceSession, RunTraceSinkError

    class CancelledMetadataSink(RecordingSink):
        def metadata(self) -> dict[str, Any]:
            raise asyncio.CancelledError

    SpanController._instance = None
    controller = SpanController.get_instance_sync()
    processor = RunTraceProcessor()
    controller.add_processor(processor)
    sink = CancelledMetadataSink()
    session = RunTraceSession(
        run_id='run-cancelled-metadata',
        run_kind='single',
        processor=processor,
        span_controller=controller,
        backend_sinks={'broken': sink},
    )

    await session.start()
    with pytest.raises(RunTraceSinkError, match='metadata: CancelledError'):
        await asyncio.wait_for(session.close(), timeout=1)


@pytest.mark.asyncio
async def test_in_memory_journal_exposes_raw_cursor_and_reports_expiry() -> None:
    from nexus.runner.trace import RunTraceContext, RunTraceCursorExpired, RunTraceInMemoryJournal

    context = RunTraceContext(run_id='run-1', run_span_id='run-span-1', run_kind='live')
    journal = RunTraceInMemoryJournal(max_records=2)
    await journal.start(context)
    await journal.on_event(context, _record(1))
    await journal.on_event(context, _record(2))
    await journal.on_event(context, _record(3))

    assert journal.latest_record_id == 3
    assert [item.record_id for item in journal.snapshot_after(1)] == [2, 3]
    with pytest.raises(RunTraceCursorExpired):
        journal.snapshot_after(0)


@pytest.mark.asyncio
async def test_in_memory_journal_timeout_returns_no_records() -> None:
    from nexus.runner.trace import RunTraceContext, RunTraceInMemoryJournal

    context = RunTraceContext(run_id='run-1', run_span_id='run-span-1', run_kind='live')
    journal = RunTraceInMemoryJournal()
    await journal.start(context)

    assert await journal.wait_after(0, timeout=0.01) == []

    await journal.close(context)


def test_iter_run_events_decodes_raw_lifecycle_events() -> None:
    from nexus.runner.trace import iter_run_events

    events = list(iter_run_events(_record(7, event_type='run.output', data={'success': True}), 'run-span-1'))

    assert len(events) == 1
    assert events[0].type == 'run.output'
    assert events[0].data == {'success': True}
    assert events[0].raw_record_id == 7


def test_span_event_record_is_immutable() -> None:
    record = _record(1)

    with pytest.raises(ValidationError):
        record.record_id = 2
