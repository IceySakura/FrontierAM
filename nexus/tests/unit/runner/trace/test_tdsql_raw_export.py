from __future__ import annotations

import logging

from datetime import datetime
from typing import Any

import pytest

from nexus.runner.trace.tdsql import sink_source
from nexus.runner.trace.tdsql.schema import principal_trace_table_name
from nexus.runner.trace.tdsql.sink_source import TDSQLTraceSource
from nexus.trace import SpanEventRecord


PRINCIPAL_SCOPE_ID = '3190e1c7731fff7995861c3221cec9a6'
PARENT_EXECUTION_ID = 'exe_0123456789abcdef0123456789abcdef'
RUN_ID = 'exe_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
RUN_SPAN_ID = 'run-span-Z'


class _Cursor:
    def __init__(self, rows: list[tuple[Any, ...]]) -> None:
        self.rows = rows
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def execute(self, sql: str, params: tuple[Any, ...]) -> None:
        self.calls.append((sql, params))

    async def fetchall(self) -> list[tuple[Any, ...]]:
        return self.rows


class _Connection:
    def __init__(self, rows: list[tuple[Any, ...]]) -> None:
        self.cursor_instance = _Cursor(rows)
        self.closed = False

    def cursor(self) -> _Cursor:
        return self.cursor_instance

    def close(self) -> None:
        self.closed = True


class _DatabaseBoundary:
    def __init__(self, pages: list[list[tuple[Any, ...]]]) -> None:
        self.pages = list(pages)
        self.connections: list[_Connection] = []
        self.configs: list[dict[str, Any]] = []

    async def connect(self, **config: Any) -> _Connection:
        self.configs.append(config)
        connection = _Connection(self.pages.pop(0))
        self.connections.append(connection)
        return connection


def _source() -> TDSQLTraceSource:
    return TDSQLTraceSource(host='mysql.example', user='reader', password='secret', db='traces')


def _batch_metadata() -> dict[str, str]:
    return {
        'storage_scope': 'principal_v1',
        'principal_scope_id': PRINCIPAL_SCOPE_ID,
        'parent_execution_id': PARENT_EXECUTION_ID,
    }


def _stream_metadata() -> dict[str, str]:
    return {**_batch_metadata(), 'run_id': RUN_ID, 'run_span_id': RUN_SPAN_ID}


def _install_database(monkeypatch, pages: list[list[tuple[Any, ...]]]) -> _DatabaseBoundary:
    database = _DatabaseBoundary(pages)
    monkeypatch.setattr('nexus.runner.trace.tdsql._driver.connect', database.connect)
    return database


async def test_read_plan_uses_only_sizes_and_stops_at_byte_boundary(monkeypatch):
    db = _install_database(monkeypatch, [[(1, 80), (3, 50), (5, 1000)]])
    plan = await _source().plan_raw_event_read(_stream_metadata(), 0, target_bytes=100, limit=3)
    assert plan.record_sizes == ((1, 80),)
    assert plan.size_bytes == 80
    assert plan.through_record_id == 1
    assert plan.has_more
    sql, params = db.connections[0].cursor_instance.calls[0]
    assert 'SELECT record_id, OCTET_LENGTH(event_data)' in sql
    assert 'parent_execution_id = %s' in sql
    assert params == (RUN_ID, RUN_SPAN_ID, PARENT_EXECUTION_ID, 0, 4)
    assert db.connections[0].closed


async def test_read_plan_keeps_first_oversized_row_and_captured_upper_bound(monkeypatch):
    db = _install_database(monkeypatch, [[(3, 20 * 1024 * 1024), (8, 10)]])
    plan = await _source().plan_raw_event_read(_stream_metadata(), 1, through_record_id=8)
    assert plan.record_sizes == ((3, 20 * 1024 * 1024),)
    assert plan.has_more
    sql, params = db.connections[0].cursor_instance.calls[0]
    assert 'record_id <= %s' in sql
    assert params[-2:] == (8, 501)


@pytest.mark.parametrize('sizes', [[(1, None)], [(1, True)], [(1, -1)], [(3, 20), (2, 10)]])
async def test_read_plan_rejects_corrupt_metadata(monkeypatch, sizes):
    _install_database(monkeypatch, [sizes])
    with pytest.raises(sink_source.RawTraceRecordInvalidError):
        await _source().plan_raw_event_read(_stream_metadata(), 0)


@pytest.mark.asyncio
async def test_batch_discovery_uses_database_keyset_cursor_and_returns_database_order(monkeypatch) -> None:
    database = _install_database(monkeypatch, [[('child-z', 'a', 9), ('child-z', 'B', 4)]])

    rows = await _source().discover_raw_streams(
        _batch_metadata(),
        after=('child-a', 'Z'),
        limit=2,
    )

    assert rows == [
        {'run_id': 'child-z', 'run_span_id': 'a', 'max_record_id': 9},
        {'run_id': 'child-z', 'run_span_id': 'B', 'max_record_id': 4},
    ]
    sql, params = database.connections[0].cursor_instance.calls[0]
    assert f'FROM {principal_trace_table_name(PRINCIPAL_SCOPE_ID)}' in sql
    assert 'SELECT run_id, run_span_id, MAX(record_id)' in sql
    assert 'parent_execution_id = %s' in sql
    assert '(run_id > %s OR (run_id = %s AND run_span_id > %s))' in sql
    assert 'GROUP BY run_id, run_span_id' in sql
    assert 'ORDER BY run_id, run_span_id' in sql
    assert '/*slave*/' not in sql
    assert params == (PARENT_EXECUTION_ID, 'child-a', 'child-a', 'Z', 2)
    assert database.connections[0].closed is True


@pytest.mark.asyncio
async def test_single_discovery_scopes_by_run_id(monkeypatch) -> None:
    database = _install_database(monkeypatch, [[(RUN_ID, RUN_SPAN_ID, 7)]])
    metadata = {
        'storage_scope': 'principal_v1',
        'principal_scope_id': PRINCIPAL_SCOPE_ID,
        'run_id': RUN_ID,
    }

    rows = await _source().discover_raw_streams(metadata)

    assert rows == [{'run_id': RUN_ID, 'run_span_id': RUN_SPAN_ID, 'max_record_id': 7}]
    sql, params = database.connections[0].cursor_instance.calls[0]
    assert 'run_id = %s' in sql
    assert 'parent_execution_id = %s' not in sql
    assert params == (RUN_ID, 500)


@pytest.mark.asyncio
async def test_fresh_discovery_sweep_can_return_a_later_lower_sorting_stream(monkeypatch) -> None:
    database = _install_database(monkeypatch, [[('child-z', 'Z', 3)], [('child-a', 'a', 1)]])
    source = _source()

    first = await source.discover_raw_streams(_batch_metadata(), after=('child-z', 'B'), limit=1)
    fresh = await source.discover_raw_streams(_batch_metadata(), after=None, limit=1)

    assert first == [{'run_id': 'child-z', 'run_span_id': 'Z', 'max_record_id': 3}]
    assert fresh == [{'run_id': 'child-a', 'run_span_id': 'a', 'max_record_id': 1}]
    _, fresh_params = database.connections[1].cursor_instance.calls[0]
    assert fresh_params == (PARENT_EXECUTION_ID, 1)


@pytest.mark.asyncio
async def test_raw_reader_returns_exact_principal_columns_without_reencoding_event_data(monkeypatch) -> None:
    raw_event_data = ' {\n  "message": "实时 ✓", "spaces": [ 1,  2 ]\n} '
    created_at = datetime(2026, 9, 11, 0, 0, 0, 123456)
    row = (
        101,
        RUN_ID,
        RUN_SPAN_ID,
        PARENT_EXECUTION_ID,
        7,
        'update',
        1_789_084_800_000_000_000,
        'trace-1',
        'span-1',
        None,
        raw_event_data,
        created_at,
    )
    database = _install_database(monkeypatch, [[row]])

    records = await _source().read_raw_events_after(_stream_metadata(), 4, limit=20, through_record_id=9)

    assert records == [
        {
            'id': 101,
            'run_id': RUN_ID,
            'run_span_id': RUN_SPAN_ID,
            'parent_execution_id': PARENT_EXECUTION_ID,
            'record_id': 7,
            'event_type': 'update',
            'time_unix_nano': 1_789_084_800_000_000_000,
            'trace_id': 'trace-1',
            'span_id': 'span-1',
            'parent_span_id': None,
            'event_data': raw_event_data,
            'created_at': '2026-09-11T00:00:00.123456',
        }
    ]
    assert tuple(records[0]) == (
        'id',
        'run_id',
        'run_span_id',
        'parent_execution_id',
        'record_id',
        'event_type',
        'time_unix_nano',
        'trace_id',
        'span_id',
        'parent_span_id',
        'event_data',
        'created_at',
    )
    sql, params = database.connections[0].cursor_instance.calls[0]
    assert 'SELECT id, run_id, run_span_id, parent_execution_id,' in sql
    assert 'run_id = %s' in sql
    assert 'run_span_id = %s' in sql
    assert 'parent_execution_id = %s' in sql
    assert 'record_id > %s' in sql
    assert 'record_id <= %s' in sql
    assert 'ORDER BY record_id' in sql
    assert '/*slave*/' not in sql
    assert params == (RUN_ID, RUN_SPAN_ID, PARENT_EXECUTION_ID, 4, 9, 20)
    assert database.connections[0].closed is True


@pytest.mark.asyncio
async def test_raw_reader_rejects_mismatched_stream_identity(monkeypatch) -> None:
    database = _install_database(
        monkeypatch,
        [[(1, RUN_ID, RUN_SPAN_ID, 'wrong-parent', 1, 'start', 1, 'trace', 'span', None, '{}', None)]],
    )

    with pytest.raises(ValueError, match='identity'):
        await _source().read_raw_events_after(_stream_metadata(), 0)

    assert database.connections[0].closed is True


@pytest.mark.asyncio
async def test_empty_raw_snapshot_interval_returns_without_query(monkeypatch) -> None:
    database = _install_database(monkeypatch, [])

    assert await _source().read_raw_events_after(_stream_metadata(), 8, through_record_id=8) == []
    assert database.connections == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('method_name', 'metadata', 'args', 'kwargs', 'message'),
    [
        ('discover_raw_streams', {}, (), {}, 'storage_scope'),
        (
            'discover_raw_streams',
            {'storage_scope': 'shared', 'principal_scope_id': PRINCIPAL_SCOPE_ID, 'run_id': RUN_ID},
            (),
            {},
            'storage_scope',
        ),
        (
            'discover_raw_streams',
            {'storage_scope': 'principal_v1', 'principal_scope_id': 'unsafe-table', 'run_id': RUN_ID},
            (),
            {},
            'principal_scope_id',
        ),
        (
            'discover_raw_streams',
            {'storage_scope': 'principal_v1', 'principal_scope_id': PRINCIPAL_SCOPE_ID},
            (),
            {},
            'run_id or parent_execution_id',
        ),
        ('discover_raw_streams', _batch_metadata(), (), {'limit': True}, 'limit'),
        ('discover_raw_streams', _batch_metadata(), (), {'after': ('run-only', '')}, 'after'),
        ('read_raw_events_after', _batch_metadata(), (0,), {}, 'run_id'),
        ('read_raw_events_after', _stream_metadata(), (-1,), {}, 'cursor'),
        ('read_raw_events_after', _stream_metadata(), (0,), {'limit': 0}, 'limit'),
        ('read_raw_events_after', _stream_metadata(), (5,), {'through_record_id': 4}, 'through_record_id'),
        ('read_raw_events_after', _stream_metadata(), (False,), {}, 'cursor'),
    ],
)
async def test_raw_queries_reject_invalid_scope_cursor_and_limit_before_connecting(
    monkeypatch,
    method_name: str,
    metadata: dict[str, Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    message: str,
) -> None:
    database = _install_database(monkeypatch, [])
    method = getattr(_source(), method_name)

    with pytest.raises(ValueError, match=message):
        await method(metadata, *args, **kwargs)

    assert database.connections == []


@pytest.mark.asyncio
async def test_slow_discovery_does_not_treat_three_column_rows_as_event_payloads(monkeypatch) -> None:
    _install_database(monkeypatch, [[(RUN_ID, RUN_SPAN_ID, 3)]])
    reports: list[tuple[str, dict[str, Any]]] = []

    def report_monitor(event: str, _message: str, **fields: Any) -> None:
        reports.append((event, fields))

    monkeypatch.setattr('nexus.runner.trace.tdsql.writer.report_monitor', report_monitor)
    monkeypatch.setattr(sink_source, 'SLOW_OPERATION_WARN_SECONDS', -1)

    await _source().discover_raw_streams({
        'storage_scope': 'principal_v1',
        'principal_scope_id': PRINCIPAL_SCOPE_ID,
        'run_id': RUN_ID,
    })

    slow_report = next(fields for event, fields in reports if event == 'read_slow')
    assert slow_report['payload_bytes'] == 0


@pytest.mark.asyncio
async def test_slow_raw_read_accounts_for_the_raw_event_data_column(monkeypatch) -> None:
    event_data = '{"message":"实时"}'
    _install_database(
        monkeypatch,
        [[(1, RUN_ID, RUN_SPAN_ID, PARENT_EXECUTION_ID, 1, 'start', 1, 'trace', 'span', None, event_data, None)]],
    )
    reports: list[tuple[str, dict[str, Any]]] = []

    def report_monitor(event: str, _message: str, **fields: Any) -> None:
        reports.append((event, fields))

    monkeypatch.setattr('nexus.runner.trace.tdsql.writer.report_monitor', report_monitor)
    monkeypatch.setattr(sink_source, 'SLOW_OPERATION_WARN_SECONDS', -1)

    await _source().read_raw_events_after(_stream_metadata(), 0)

    slow_report = next(fields for event, fields in reports if event == 'read_slow')
    assert slow_report['payload_bytes'] == len(event_data)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('method_name', 'args'),
    [('discover_raw_streams', ()), ('read_raw_events_after', (0,))],
)
async def test_raw_query_failures_do_not_log_database_exception_details(
    monkeypatch,
    caplog,
    method_name: str,
    args: tuple[Any, ...],
) -> None:
    sentinel = 'credential-password-SHOULD-NOT-BE-LOGGED'

    async def connect(**_config: Any):
        raise RuntimeError(f'database rejected {sentinel}')

    monkeypatch.setattr('nexus.runner.trace.tdsql._driver.connect', connect)
    caplog.set_level(logging.WARNING)
    method = getattr(_source(), method_name)

    with pytest.raises(RuntimeError, match=sentinel):
        await method(_stream_metadata(), *args)

    assert 'db read_failure' in caplog.text
    assert sentinel not in caplog.text


@pytest.mark.asyncio
async def test_existing_typed_reader_keeps_span_event_contract(monkeypatch) -> None:
    database = _install_database(monkeypatch, [[(2, 'update', 20, 'trace', 'span', None, '{"events": []}')]])

    records = await _source().read_events_after(
        {'run_id': RUN_ID, 'run_span_id': RUN_SPAN_ID},
        1,
        limit=3,
    )

    assert records == [
        SpanEventRecord(
            record_id=2,
            event_type='update',
            time_unix_nano=20,
            trace_id='trace',
            span_id='span',
            parent_span_id=None,
            event_data={'events': []},
        )
    ]
    sql, params = database.connections[0].cursor_instance.calls[0]
    assert '/*slave*/' not in sql
    assert params == (RUN_ID, RUN_SPAN_ID, RUN_ID, 1, 3)
