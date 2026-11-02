from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest

from nexus.runner.trace import RunTraceContext
from nexus.runner.trace.tdsql import config as config_module
from nexus.runner.trace.tdsql import writer as writer_module
from nexus.runner.trace.tdsql.sink_source import TDSQLBoundlessTraceSink, TDSQLTraceSink
from nexus.runner.trace.tdsql.writer import TDSQLTraceWriter
from nexus.trace import SpanEventRecord


ITEMS_ENV = 'NEXUS_TRAJ_TDSQL_MAX_QUEUE_ITEMS'
BYTES_ENV = 'NEXUS_TRAJ_TDSQL_MAX_QUEUE_BYTES'


@pytest.fixture(autouse=True)
def clear_queue_environment(monkeypatch):
    monkeypatch.delenv(ITEMS_ENV, raising=False)
    monkeypatch.delenv(BYTES_ENV, raising=False)
    monkeypatch.delenv('NEXUS_TRAJ_TDSQL_WRITER_SHARDS', raising=False)
    monkeypatch.setattr(config_module, '_manager_memory_bytes', lambda: 0)


@pytest.fixture(params=[TDSQLTraceSink, TDSQLBoundlessTraceSink, TDSQLTraceWriter])
def make_backend(request):
    def make(**kwargs):
        if request.param is TDSQLTraceWriter:
            return request.param(**kwargs)
        return request.param(host='mysql.example', user='runner', db='traces', **kwargs)

    return make


def test_queue_defaults_and_environment_are_resolved_at_creation(make_backend, monkeypatch):
    default = make_backend()
    assert (default.max_queue_items, default.max_queue_bytes) == (10000, 1536 * 1024**2)
    monkeypatch.setenv(ITEMS_ENV, '25')
    monkeypatch.setenv(BYTES_ENV, '4096')
    configured = make_backend()
    assert (configured.max_queue_items, configured.max_queue_bytes) == (25, 4096)
    assert (default.max_queue_items, default.max_queue_bytes) == (10000, 1536 * 1024**2)


@pytest.mark.parametrize(('field', 'env'), [('max_queue_items', ITEMS_ENV), ('max_queue_bytes', BYTES_ENV)])
def test_explicit_queue_limit_overrides_environment_independently(make_backend, monkeypatch, field, env):
    monkeypatch.setenv(env, 'invalid')
    configured = make_backend(**{field: 42})
    assert getattr(configured, field) == 42


@pytest.mark.parametrize('field', ['max_queue_items', 'max_queue_bytes'])
@pytest.mark.parametrize('value', [0, -1, 1.5, float('inf'), float('nan'), True, '123'])
def test_invalid_queue_limit_is_rejected(make_backend, field, value):
    with pytest.raises(ValueError, match=f'{field} must be a positive integer'):
        make_backend(**{field: value})


@pytest.mark.parametrize('env', [ITEMS_ENV, BYTES_ENV])
@pytest.mark.parametrize('value', ['', 'invalid', '0', '-1', '1.5'])
def test_invalid_queue_environment_is_rejected(make_backend, monkeypatch, env, value):
    monkeypatch.setenv(env, value)
    with pytest.raises(ValueError, match='must be a positive integer'):
        make_backend()


def test_item_limit_rejects_only_over_capacity_and_recovers_after_dequeue():
    writer = TDSQLTraceWriter(max_queue_items=1, max_queue_bytes=1024)
    writer._started = True
    assert writer.enqueue_span_event(('first',)) == 1
    assert writer.enqueue_span_event(('second',)) == 0
    assert len(writer._queue) == 1
    assert writer._queued_bytes == 5
    assert len(writer._pop_batch()) == 1
    assert writer._queued_bytes == 0
    assert writer.enqueue_span_event(('second',)) == 2


def test_byte_limit_rejects_oversized_record_and_recovers_after_dequeue():
    writer = TDSQLTraceWriter(max_queue_items=10, max_queue_bytes=5)
    writer._started = True
    assert writer.enqueue_span_event(('123456',)) == 0
    assert writer._queued_bytes == 0
    assert writer.enqueue_span_event(('12345',)) == 1
    assert writer.enqueue_span_event(('x',)) == 0
    assert len(writer._pop_batch()) == 1
    assert writer._queued_bytes == 0
    assert writer.enqueue_span_event(('x',)) == 2


@pytest.mark.asyncio
async def test_writer_sharing_preserves_independent_limits(monkeypatch):
    async def ready(self, _db_config):
        self._started = True
        self._conn = object()
        return True

    monkeypatch.setattr(TDSQLTraceWriter, 'ensure_started', ready)
    monkeypatch.setattr(writer_module, '_writers', {})
    monkeypatch.setenv(ITEMS_ENV, '1')
    monkeypatch.setenv(BYTES_ENV, '1024')
    config = {'host': 'mysql.example', 'user': 'runner', 'db': 'traces'}
    first = await writer_module.get_writer(config)
    same = await writer_module.get_writer(config, max_queue_items=1, max_queue_bytes=1024)
    more_items = await writer_module.get_writer(config, max_queue_items=2, max_queue_bytes=1024)
    more_bytes = await writer_module.get_writer(config, max_queue_items=1, max_queue_bytes=2048)
    assert first is same
    assert first is not more_items
    assert first is not more_bytes
    assert more_items is not more_bytes
    assert first.enqueue_span_event(('first',)) == 1
    assert first.enqueue_span_event(('second',)) == 0
    assert more_items.enqueue_span_event(('first',)) == 1
    assert more_items.enqueue_span_event(('second',)) == 2
    assert more_bytes._queued_bytes == 0


@pytest.mark.parametrize('sink_class', [TDSQLTraceSink, TDSQLBoundlessTraceSink])
@pytest.mark.asyncio
async def test_sink_passes_queue_limits_to_writer(sink_class, monkeypatch):
    writer = Mock()
    writer.enqueue_span_event.return_value = 1
    factory = AsyncMock(return_value=writer)
    monkeypatch.setattr(writer_module, 'get_writer', factory)
    sink = sink_class(host='mysql.example', user='runner', db='traces', max_queue_items=25, max_queue_bytes=4096)
    context = RunTraceContext(
        run_id='run-1',
        run_span_id='span-1',
        run_kind='single',
        metadata={
            'storage_scope': 'principal_v1',
            'principal_scope_id': 'a' * 32,
            'parent_created_at': '2026-09-25 00:00:00',
        },
    )
    await sink.start(context)
    await sink.on_event(
        context,
        SpanEventRecord(
            record_id=1,
            event_type='update',
            time_unix_nano=1,
            trace_id='trace-1',
            span_id='span-1',
            parent_span_id='',
            event_data={'events': []},
        ),
    )
    factory.assert_awaited_once_with(
        sink._db_config(),
        dialect=sink.dialect,
        max_queue_items=25,
        max_queue_bytes=4096,
        writer_shards=1,
        run_span_id='span-1',
    )
    assert sink._last_seq == 1
