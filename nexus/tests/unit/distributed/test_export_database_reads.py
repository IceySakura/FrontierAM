from __future__ import annotations

from typing import Any

import aiomysql
import pytest

from nexus.distributed.export import results
from nexus.distributed.export.types import TraceResult
from nexus.runner.trace.tdsql import sink_source


TABLE = 'trajectory_span_events_pr_' + 'a' * 32
PARTITION = '2026-09-25 00:00:00.000000'


def reader(**kwargs):
    return results.TdsqlTraceReader(
        host='db.test', port=3306, user='reader', password='unused', database='trace', table=TABLE, **kwargs
    )


class Database:
    def __init__(self, *, unattributed=False):
        self.calls = []
        self.closed = 0
        self.unattributed = unattributed
        self.rows = []

    async def connect(self, **kwargs):
        return self

    def cursor(self):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def execute(self, sql, params):
        self.calls.append((sql, params))
        if 'MIN(time_unix_nano)' in sql:
            self.rows = [] if self.unattributed and 'parent_execution_id = %s' in sql else [('child', 100, 200)]
        else:
            self.rows = []

    async def fetchall(self):
        return self.rows

    def close(self):
        self.closed += 1


async def test_results_default_driver_is_aiomysql_and_does_not_change_sink_driver(monkeypatch):
    db = Database()
    monkeypatch.setattr(aiomysql, 'connect', db.connect)
    from nexus.runner.trace.tdsql import _driver

    original_writer_connect = _driver.connect
    await reader().read_many(['child'], include_timestamps=False)
    assert db.closed == 1
    assert _driver.connect is original_writer_connect


async def test_control_times_skip_the_database_aggregate():
    db = Database()
    await reader(connect=db.connect).read_many(['child'], parent_execution_id='batch', include_timestamps=False)
    assert db.calls
    assert all('MIN(' not in sql and 'MAX(' not in sql for sql, _ in db.calls)


@pytest.mark.parametrize('unattributed', [False, True])
async def test_missing_control_times_use_covering_parent_index_with_legacy_fallback(unattributed):
    db = Database(unattributed=unattributed)
    actual = await reader(connect=db.connect, parent_created_at=PARTITION).read_many(
        ['child'], parent_execution_id='batch'
    )
    sql, params = db.calls[0]
    assert 'FORCE INDEX (idx_traj_span_events_parent_cursor)' in sql
    assert 'parent_execution_id = %s' in sql
    assert 'parent_created_at = %s' in sql
    assert params == ('batch', 'child', PARTITION)
    assert actual['child'].start_time is not None
    aggregates = [(sql, params) for sql, params in db.calls if 'MIN(' in sql]
    assert len(aggregates) == (2 if unattributed else 1)
    if unattributed:
        sql, params = aggregates[-1]
        assert 'parent_execution_id = %s' not in sql
        assert 'parent_created_at = %s' in sql
        assert params == ('child', PARTITION)


@pytest.mark.parametrize(('start', 'end'), [(True, True), (False, True), (True, False), (False, False)])
async def test_batch_reader_only_requests_missing_lifecycle_times(start, end, monkeypatch):
    calls = []

    class Api:
        async def get_control_events(self, execution_id):
            events = []
            for include, status, when in [(start, 'running', 'control-start'), (end, 'succeeded', 'control-end')]:
                if include:
                    events.append({'type': 'execution.status.changed', 'payload': {'to': status}, 'created_at': when})
            return events

    async def read_many(ids, **kwargs):
        calls.append((ids, kwargs))
        return {'child': TraceResult(output={'success': True}, start_time='trace-start', end_time='trace-end')}

    trace_reader = reader()
    monkeypatch.setattr(trace_reader, 'read_many', read_many)

    value = await results._single_record(
        api=Api(),  # type: ignore[arg-type]
        trace_reader=trace_reader,
        single=results._batch_export(
            {'execution_id': 'batch', 'spec_type': 'nexus_batch'},
            {'item_id': 'item', 'status': 'succeeded', 'current_attempt_no': 0, 'current_child_execution_id': 'child'},
        ),
    )
    assert calls == [
        (
            ['child'],
            {
                'parent_execution_id': 'batch',
                'include_timestamps': not (start and end),
                'scan_out': None,
                'spill_pool': None,
            },
        )
    ]
    assert value['start_time'] == ('control-start' if start else 'trace-start')
    assert value['end_time'] == ('control-end' if end else 'trace-end')


async def test_trace_export_source_uses_aiomysql_and_wraps_its_db_errors(monkeypatch):
    calls = []

    async def fail(**config: Any):
        calls.append(config)
        raise aiomysql.OperationalError(2013, 'test connection lost')

    monkeypatch.setattr(aiomysql, 'connect', fail)
    source = sink_source.TDSQLTraceSource(host='db.test', user='reader', db='trace', read_driver='aiomysql')
    with pytest.raises(sink_source.RunTraceSourceUnavailable):
        await source.discover_raw_streams({
            'storage_scope': 'principal_v1',
            'principal_scope_id': 'a' * 32,
            'run_id': 'child',
        })
    assert len(calls) == 1
