from __future__ import annotations

import asyncio
import re

import pytest

from tests.unit.distributed.test_trace_read_service import event

from nexus.runner.trace.tdsql.sink_source import (
    RawTraceEventTooLargeError,
    RawTraceRecordInvalidError,
    TDSQLTraceSource,
)


META = {'storage_scope': 'principal_v1', 'principal_scope_id': '0' * 32, 'run_id': 'exec', 'run_span_id': 'span'}


class Cursor:
    def __init__(self, connection, rows):
        self.connection = connection
        self.rows = iter(rows)
        self.calls = []
        self.reads = 0

    async def execute(self, sql, params):
        self.calls.append((sql, params))

    async def fetchone(self):
        self.reads += 1
        return next(self.rows, None)

    async def close(self):
        assert self.connection.closed, 'unbuffered cursor must not drain before connection closure'


class Connection:
    def __init__(self, rows):
        self.closed = False
        self.instance = Cursor(self, rows)

    def cursor(self, cls):
        assert cls.__name__ == 'SSCursor'
        return self.instance

    def thread_id(self):
        return 42

    def close(self):
        self.closed = True


def install(monkeypatch, rows, *, dialect='tdsql'):
    connection = Connection(rows)

    async def connect(**kwargs):
        assert kwargs['autocommit'] is True
        return connection

    monkeypatch.setattr('nexus.runner.trace.tdsql._driver.connect', connect)
    source = TDSQLTraceSource(host='reader', db='traces', user='read', password='secret', dialect=dialect)

    async def terminate(thread_id, query_marker):
        assert thread_id == 42
        assert connection.closed
        assert re.fullmatch(r'nexus_trace_read_[0-9a-f]{32}', query_marker)

    monkeypatch.setattr(source, '_terminate_raw_query', terminate)
    return source, connection


def raw_row(record_id, *, raw_size=10):
    return (*event(record_id).values(), raw_size)


async def test_stream_listing_uses_distinct_without_max_or_payload(monkeypatch):
    source, connection = install(monkeypatch, [('a',), ('B',)])
    assert await source.list_raw_streams(META, limit=2) == ['a', 'B']
    sql, params = connection.instance.calls[0]
    assert re.fullmatch(r'SELECT /\*nexus_trace_read_[0-9a-f]{32}\*/ DISTINCT run_span_id.*', sql)
    assert 'MAX' not in sql
    assert 'event_data' not in sql
    assert '/*slave*/' not in sql
    assert params == ('exec', 2)
    assert connection.closed


async def test_page_uses_streaming_projection_and_returns_only_accepted_prefix(monkeypatch):
    source, connection = install(monkeypatch, [raw_row(1), raw_row(3), raw_row(5)])
    page = await source.read_raw_event_page(META, 0, limit=2, accept_record=lambda row: row['record_id'] < 3)
    assert [r['record_id'] for r in page.items] == [1]
    assert page.has_more
    assert connection.instance.reads == 2
    sql, params = connection.instance.calls[0]
    assert 'CASE WHEN OCTET_LENGTH(event_data) <= %s THEN event_data ELSE NULL END' in sql
    assert 'run_id = %s AND run_span_id = %s AND record_id > %s' in sql
    assert params == (8 * 1024 * 1024, 'exec', 'span', 0, 3)
    assert connection.closed


async def test_oversized_body_is_not_decoded_and_valid_prefix_can_be_returned(monkeypatch):
    oversized = list(raw_row(2, raw_size=9 * 1024 * 1024))
    oversized[10] = None
    source, connection = install(monkeypatch, [raw_row(1), tuple(oversized)])
    page = await source.read_raw_event_page(META, 0, accept_record=lambda row: True)
    assert len(page.items) == 1
    assert page.has_more
    assert connection.closed
    source, connection = install(monkeypatch, [tuple(oversized)])
    with pytest.raises(RawTraceEventTooLargeError) as error:
        await source.read_raw_event_page(META, 1, accept_record=lambda row: pytest.fail('oversized body was forwarded'))
    assert error.value.record_id == 2
    assert connection.closed


async def test_preview_keeps_oversized_record_identity_and_advances_to_tail(monkeypatch):
    oversized = list(raw_row(2, raw_size=20 * 1024 * 1024))
    oversized[10] = None
    source, connection = install(monkeypatch, [tuple(oversized), raw_row(3)])
    page = await source.read_raw_event_page(META, 1, omit_oversized_event_data=True, accept_record=lambda row: True)
    assert [row['record_id'] for row in page.items] == [2, 3]
    assert page.items[0]['event_data'] is None
    assert page.items[0]['event_data_omitted'] is True
    assert page.items[0]['event_data_bytes'] == 20 * 1024 * 1024
    assert page.items[1]['event_data_omitted'] is False
    assert page.items[1]['event_data'] == event(3)['event_data']
    assert page.has_more is False
    assert connection.closed


@pytest.mark.parametrize('size', [None, -1, True, '100'])
async def test_preview_rejects_invalid_length_metadata(monkeypatch, size):
    source, _ = install(monkeypatch, [raw_row(1, raw_size=size)])
    with pytest.raises(RawTraceRecordInvalidError):
        await source.read_raw_event_page(META, 0, omit_oversized_event_data=True, accept_record=lambda row: True)


async def test_preview_rejects_unexplained_null_body(monkeypatch):
    row = list(raw_row(1))
    row[10] = None
    source, _ = install(monkeypatch, [tuple(row)])
    with pytest.raises(RawTraceRecordInvalidError):
        await source.read_raw_event_page(META, 0, omit_oversized_event_data=True, accept_record=lambda row: True)


async def test_cancel_closes_actual_connection_before_cursor_cleanup(monkeypatch):
    source, connection = install(monkeypatch, [])
    entered = asyncio.Event()

    async def fetch():
        entered.set()
        await asyncio.Event().wait()

    connection.instance.fetchone = fetch
    task = asyncio.create_task(source.read_raw_event_page(META, 0, accept_record=lambda row: True))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert connection.closed


@pytest.mark.parametrize('disconnect_code', [None, 1094, 704])
async def test_cleanup_waits_for_query_marker_when_proxy_and_backend_ids_differ(monkeypatch, disconnect_code):
    marker = 'nexus_trace_read_' + 'a' * 32
    calls = []
    snapshots = iter([
        [(9001, f'SELECT /*{marker}*/ SLEEP(20)'), (42, 'unrelated query')],
        [(42, 'unrelated query')],
    ])

    class ControlCursor:
        description = [('Id',), ('Info',)]

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def execute(self, sql):
            calls.append(sql)
            if sql.startswith('KILL') and disconnect_code is not None:
                raise RuntimeError(disconnect_code, 'already disconnected')

        async def fetchall(self):
            return next(snapshots)

    class Controller:
        closed = False

        def cursor(self):
            return ControlCursor()

        def close(self):
            self.closed = True

    controller = Controller()

    async def connect(**kwargs):
        return controller

    monkeypatch.setattr('nexus.runner.trace.tdsql._driver.connect', connect)
    source = TDSQLTraceSource(host='reader', db='traces', user='read', password='secret')
    await source._terminate_raw_query(42, marker)
    assert calls == ['KILL CONNECTION 42', 'SHOW FULL PROCESSLIST', 'SHOW FULL PROCESSLIST']
    assert controller.closed


@pytest.mark.parametrize('operation', ['list', 'events', 'exists'])
async def test_bounded_reads_reject_parent_scope_before_connect(monkeypatch, operation):
    source, connection = install(monkeypatch, [])
    metadata = {**META, 'parent_execution_id': 'parent'}
    if operation == 'list':
        pending = source.list_raw_streams(metadata)
    elif operation == 'events':
        pending = source.read_raw_event_page(metadata, 0, accept_record=lambda row: True)
    else:
        pending = source.raw_stream_exists(metadata)
    with pytest.raises(ValueError, match='require one run_id'):
        await pending
    assert connection.instance.calls == []
