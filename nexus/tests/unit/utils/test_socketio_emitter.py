from __future__ import annotations

import asyncio
import logging

import pytest

from nexus.utils.socketio import AsyncSocketIOEmitter, SocketIOEventLoggerAdapter


class RecordingSio:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.batches: list[list[dict]] = []
        self.payloads: list[dict] = []

    async def emit(self, event_name, payload):  # noqa: ANN001, ARG002
        if self.error is not None:
            raise self.error
        self.payloads.append(payload)
        self.batches.append(payload['items'])


def test_socketio_event_logger_downgrades_routine_event_traffic(caplog) -> None:
    logger = logging.getLogger('test.socketio.event.logger')
    adapter = SocketIOEventLoggerAdapter(logger, {})

    with caplog.at_level(logging.DEBUG, logger=logger.name):
        adapter.info('received event "%s" from %s [%s]', 'worker_log', 'sid', '/worker')
        adapter.info('emitting event "%s" to %s [%s]', 'worker_log', 'all', '/')
        adapter.info('received ack from %s [%s]', 'sid', '/worker')
        adapter.info('other socketio lifecycle event')

    levels = {record.getMessage(): record.levelno for record in caplog.records}
    assert levels['received event "worker_log" from sid [/worker]'] == logging.DEBUG
    assert levels['emitting event "worker_log" to all [/]'] == logging.DEBUG
    assert levels['received ack from sid [/worker]'] == logging.DEBUG
    assert levels['other socketio lifecycle event'] == logging.INFO


@pytest.mark.asyncio
async def test_flush_waits_until_all_queued_items_are_emitted() -> None:
    sio = RecordingSio()
    emitter = AsyncSocketIOEmitter(sio, 'span', batch_size=2, flush_interval=0.01)
    for index in range(3):
        assert emitter.enqueue({'index': index}) is True

    await emitter.flush()
    await emitter.close()

    assert [item for batch in sio.batches for item in batch] == [
        {'index': 0},
        {'index': 1},
        {'index': 2},
    ]


@pytest.mark.asyncio
async def test_started_emitter_accepts_items_from_worker_threads() -> None:
    sio = RecordingSio()
    emitter = AsyncSocketIOEmitter(sio, 'span', flush_interval=0.01)
    await emitter.start()

    assert await asyncio.to_thread(emitter.enqueue, {'span': 1}) is True

    await emitter.flush()
    await emitter.close()
    assert sio.batches == [[{'span': 1}]]


@pytest.mark.asyncio
async def test_barrier_emits_a_sequenced_marker_after_queued_items() -> None:
    sio = RecordingSio()
    emitter = AsyncSocketIOEmitter(sio, 'span', flush_interval=0.01)
    await emitter.start()
    assert emitter.enqueue({'span': 1}) is True

    batch_id = await emitter.barrier()
    await emitter.close()

    assert batch_id == 2
    assert sio.payloads == [
        {'batch_id': 1, 'items': [{'span': 1}]},
        {'batch_id': 2, 'barrier': True, 'items': []},
    ]


@pytest.mark.asyncio
async def test_flush_surfaces_socketio_emit_failure() -> None:
    emitter = AsyncSocketIOEmitter(RecordingSio(error=OSError('socket failed')), 'span', flush_interval=0.01)
    assert emitter.enqueue({'span': 1}) is True

    with pytest.raises(OSError, match='socket failed'):
        await emitter.flush()

    with pytest.raises(OSError, match='socket failed'):
        await emitter.close()


@pytest.mark.asyncio
async def test_unexpected_transport_cancel_is_reported_as_emitter_failure() -> None:
    emitter = AsyncSocketIOEmitter(RecordingSio(error=asyncio.CancelledError()), 'span', flush_interval=0.01)
    assert emitter.enqueue({'span': 1}) is True

    with pytest.raises(RuntimeError, match='transport was unexpectedly cancelled'):
        await emitter.flush()

    with pytest.raises(RuntimeError, match='transport was unexpectedly cancelled'):
        await emitter.close()


@pytest.mark.asyncio
async def test_cancelling_emit_batch_cancels_blocked_transport() -> None:
    class BlockingSio:
        def __init__(self) -> None:
            self.entered = asyncio.Event()
            self.cancelled = False

        async def emit(self, event_name, payload) -> None:  # noqa: ANN001, ARG002
            self.entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled = True
                raise

    sio = BlockingSio()
    emitter = AsyncSocketIOEmitter(sio, 'span')
    emit_task = asyncio.create_task(emitter._emit_batch([{'span': 1}]))
    await sio.entered.wait()
    emit_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await emit_task

    assert sio.cancelled is True


@pytest.mark.asyncio
async def test_queue_overflow_is_reported_by_flush(monkeypatch) -> None:
    emitter = AsyncSocketIOEmitter(RecordingSio(), 'span', max_queue_size=1)
    monkeypatch.setattr(emitter, '_ensure_worker_started', lambda loop: None)
    assert emitter.enqueue({'span': 1}) is True
    assert emitter.enqueue({'span': 2}) is False

    with pytest.raises(RuntimeError, match='queue is full'):
        await asyncio.wait_for(emitter.flush(), timeout=1)


def test_enqueue_without_an_event_loop_is_not_silently_dropped() -> None:
    emitter = AsyncSocketIOEmitter(RecordingSio(), 'span')

    assert emitter.enqueue({'span': 1}) is False

    with pytest.raises(RuntimeError, match='running event loop'):
        asyncio.run(emitter.flush())


def test_best_effort_emitter_recovers_after_prestart_item_is_dropped() -> None:
    sio = RecordingSio()
    emitter = AsyncSocketIOEmitter(sio, 'log', drop_unbound_items=True)

    assert emitter.enqueue({'before': 'start'}) is False

    async def emit_after_start() -> None:
        await emitter.start()
        assert emitter.enqueue({'after': 'start'}) is True
        await emitter.flush()
        await emitter.close()

    asyncio.run(emit_after_start())

    assert sio.batches == [[{'after': 'start'}]]
