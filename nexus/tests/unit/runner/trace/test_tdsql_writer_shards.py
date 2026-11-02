from __future__ import annotations

import asyncio

import pytest

from nexus.runner.trace import RunTraceContext
from nexus.runner.trace.tdsql import config as config_module
from nexus.runner.trace.tdsql import writer as writer_module
from nexus.runner.trace.tdsql.config import resolve_writer_shards
from nexus.runner.trace.tdsql.sink_source import TDSQLBoundlessTraceSink, TDSQLTraceSink
from nexus.runner.trace.tdsql.writer import TDSQLTraceWriter
from nexus.trace import SpanEventRecord


SHARDS_ENV = 'NEXUS_TRAJ_TDSQL_WRITER_SHARDS'
DB_CONFIG = {'host': 'mysql.example', 'user': 'runner', 'db': 'traces'}


@pytest.fixture(autouse=True)
def clean_writer_environment(monkeypatch):
    monkeypatch.setattr(config_module, '_manager_memory_bytes', lambda: 0)
    for name in [
        SHARDS_ENV,
        'NEXUS_TRAJ_TDSQL_MAX_QUEUE_ITEMS',
        'NEXUS_TRAJ_TDSQL_MAX_QUEUE_BYTES',
    ]:
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize(
    ('memory_bytes', 'expected'),
    [
        (None, 1),
        (0, 1),
        (2 * 1024**3, 1),
        (4 * 1024**3, 1),
        (8 * 1024**3 - 1, 1),
        (8 * 1024**3, 2),
        (12 * 1024**3 - 1, 2),
        (12 * 1024**3, 3),
        (16 * 1024**3 - 1, 3),
        (16 * 1024**3, 4),
        (64 * 1024**3, 4),
    ],
)
def test_automatic_shards_follow_manager_memory_with_four_shard_cap(monkeypatch, memory_bytes, expected):
    monkeypatch.setattr(config_module, '_manager_memory_bytes', lambda: memory_bytes or 0)
    assert resolve_writer_shards() == expected
    for sink_class in [TDSQLTraceSink, TDSQLBoundlessTraceSink]:
        assert sink_class(**DB_CONFIG).writer_shards == expected


def test_explicit_shards_override_auto_and_can_exceed_cap(monkeypatch):
    monkeypatch.setattr(config_module, '_manager_memory_bytes', lambda: 8 * 1024**3)
    monkeypatch.setenv(SHARDS_ENV, '8')
    assert resolve_writer_shards() == 8
    assert TDSQLTraceSink(**DB_CONFIG, writer_shards=1).writer_shards == 1
    assert TDSQLTraceSink(**DB_CONFIG, writer_shards=6).writer_shards == 6
    monkeypatch.setenv(SHARDS_ENV, 'invalid')
    assert TDSQLTraceSink(**DB_CONFIG, writer_shards=6).writer_shards == 6


@pytest.mark.parametrize('value', [0, -1, True, 1.5, '2'])
def test_invalid_explicit_shards_are_rejected(value):
    with pytest.raises(ValueError, match='writer_shards must be a positive integer'):
        resolve_writer_shards(value)


@pytest.mark.parametrize('value', ['', 'invalid', '0', '-1', '1.5'])
def test_invalid_shard_environment_is_rejected(monkeypatch, value):
    monkeypatch.setenv(SHARDS_ENV, value)
    with pytest.raises(ValueError, match='must be a positive integer'):
        resolve_writer_shards()


@pytest.fixture
def ready_writer_registry(monkeypatch):
    async def ready(self, _db_config):
        self._started = True
        self._conn = object()
        return True

    monkeypatch.setattr(TDSQLTraceWriter, 'ensure_started', ready)
    monkeypatch.setattr(writer_module, '_writers', {})
    monkeypatch.setattr(writer_module, '_disabled_counts', {})


async def _two_writers(*, db_config=None, **options):
    config = DB_CONFIG if db_config is None else db_config
    pairs = [
        (f'span-{i}', await writer_module.get_writer(config, run_span_id=f'span-{i}', writer_shards=2, **options))
        for i in range(10)
    ]
    first = pairs[0]
    other = next(pair for pair in pairs if pair[1] is not first[1])
    return first, other


@pytest.mark.asyncio
async def test_writer_routing_is_stable_lazy_and_keeps_full_per_shard_limits(ready_writer_registry):
    (span, first), (_, second) = await _two_writers(max_queue_items=1, max_queue_bytes=1024)
    assert len(writer_module._writers) == 2
    again = await writer_module.get_writer(
        dict(reversed(DB_CONFIG.items())),
        run_span_id=span,
        writer_shards=2,
        max_queue_items=1,
        max_queue_bytes=1024,
    )
    assert again is first
    for writer in [first, second]:
        assert writer.max_queue_items == 1
        assert writer.max_queue_bytes == 1024
        assert writer.enqueue_span_event(('x' * 1024,)) == 1
        assert writer.enqueue_span_event(('x',)) == 0
    single = await writer_module.get_writer(DB_CONFIG, writer_shards=1, run_span_id='a')
    same_single = await writer_module.get_writer(DB_CONFIG, writer_shards=1, run_span_id='b')
    assert single is same_single
    assert single is not first


@pytest.mark.asyncio
async def test_blocked_writer_does_not_block_other_shard(ready_writer_registry, monkeypatch):
    (_, first), (_, second) = await _two_writers()
    entered, release, other_done = asyncio.Event(), asyncio.Event(), asyncio.Event()
    writes = {first: [], second: []}

    async def flush(self, batch):
        if self is first:
            entered.set()
            await release.wait()
        writes[self].extend(write.seq for write in batch)
        self._advance_settled(max(write.seq for write in batch))
        if self is second:
            other_done.set()

    monkeypatch.setattr(TDSQLTraceWriter, '_flush_batch', flush)
    monkeypatch.setattr(writer_module, '_BATCH_WINDOW_SECONDS', 0)
    for writer in [first, second]:
        writer._event = asyncio.Event()
        writer._io_loop = asyncio.get_running_loop()
    assert first.enqueue_span_event(('first',)) == 1
    loops = [asyncio.create_task(writer._writer_loop()) for writer in [first, second]]
    try:
        await asyncio.wait_for(entered.wait(), timeout=1)
        assert first.enqueue_span_event(('second',)) == 2
        assert second.enqueue_span_event(('independent',)) == 1
        await asyncio.wait_for(other_done.wait(), timeout=1)
        assert first.settled_through == 0
        assert second.settled_through == 1
        release.set()
        await first.wait_until_settled(2, timeout=1)
        assert writes[first] == [1, 2]
    finally:
        release.set()
        for writer in [first, second]:
            writer._closing = True
            writer._event.set()
        await asyncio.wait_for(asyncio.gather(*loops), timeout=1)


@pytest.mark.parametrize('sink_class', [TDSQLTraceSink, TDSQLBoundlessTraceSink])
@pytest.mark.asyncio
async def test_sink_keeps_its_shard_and_waits_for_its_own_sequence(
    ready_writer_registry,
    monkeypatch,
    sink_class,
):
    monkeypatch.setattr(config_module, '_manager_memory_bytes', lambda: 8 * 1024**3)
    sink = sink_class(**DB_CONFIG, close_flush_timeout_seconds=1)
    context = RunTraceContext(
        run_id='run-1',
        run_span_id='span-0',
        run_kind='single',
        metadata={
            'storage_scope': 'principal_v1',
            'principal_scope_id': 'a' * 32,
            'parent_created_at': '2026-09-25 00:00:00',
        },
    )
    await sink.start(context)

    async def emit(record_id):
        await sink.on_event(
            context,
            SpanEventRecord(
                record_id=record_id,
                event_type='update',
                time_unix_nano=record_id,
                trace_id='trace-1',
                span_id='span-0',
                parent_span_id='',
                event_data={'events': []},
            ),
        )

    await emit(1)
    own_writer = sink._writer
    monkeypatch.setenv(SHARDS_ENV, '8')
    await emit(2)
    assert sink.writer_shards == 2
    assert sink._writer is own_writer
    (_, first), (_, second) = await _two_writers(db_config=sink._db_config(), dialect=sink.dialect)
    assert own_writer in (first, second)
    other = second if first is own_writer else first
    other._advance_settled(100)
    closing = asyncio.create_task(sink.close())
    try:
        await asyncio.sleep(0)
        assert not closing.done()
        own_writer._advance_settled(2)
        await asyncio.wait_for(closing, timeout=1)
        assert sink.metadata()['status'] == 'succeeded'
    finally:
        if not closing.done():
            closing.cancel()
            await asyncio.gather(closing, return_exceptions=True)
