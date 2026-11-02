from __future__ import annotations

import copy
import json

from typing import Any
from unittest.mock import AsyncMock

import pytest

from nexus.runner.trace import RunTraceContext, RunTraceScope, RunTraceScopePreparer, principal_trace_scope_id
from nexus.runner.trace.tdsql import schema as tdsql_schema
from nexus.runner.trace.tdsql import writer as tdsql_writer
from nexus.runner.trace.tdsql.config import build_db_config
from nexus.runner.trace.tdsql.schema import (
    principal_trace_check_constraint_name,
    principal_trace_table_name,
    principal_trace_table_statement,
    principal_trace_table_upgrade_statement,
)
from nexus.runner.trace.tdsql.sink_source import TDSQLTraceSink, TDSQLTraceSource, _build_query
from nexus.runner.trace.tdsql.writer import TDSQLTraceWriter, _Write
from nexus.trace import SpanEventRecord


PARENT_EXECUTION_ID = 'exe_0123456789abcdef0123456789abcdef'
OTHER_PARENT_EXECUTION_ID = 'exe_fedcba9876543210fedcba9876543210'
OWNER_PRINCIPAL_ID = 'principal-owner-a'
OTHER_OWNER_PRINCIPAL_ID = 'principal-owner-b'
PRINCIPAL_SCOPE_ID = '3190e1c7731fff7995861c3221cec9a6'
OTHER_PRINCIPAL_SCOPE_ID = '9ae9114872a4a190d65731e26959427f'


def _record(record_id: int = 1) -> SpanEventRecord:
    return SpanEventRecord(
        record_id=record_id,
        event_type='update',
        time_unix_nano=record_id,
        trace_id='trace-1',
        span_id='run-span-1',
        parent_span_id='',
        event_data={'events': []},
    )


def _event_record(event_name: str, *, record_id: int = 1) -> SpanEventRecord:
    return SpanEventRecord(
        record_id=record_id,
        event_type='update',
        time_unix_nano=record_id,
        trace_id='trace-1',
        span_id='span-1',
        parent_span_id='run-span-1',
        event_data={
            'type': 'UPDATE',
            'span_id': 'span-1',
            'time_unix_nano': record_id,
            'attributes': None,
            'events': [{'name': event_name, 'time_unix_nano': record_id, 'attributes': {}}],
            'status': None,
            'trace_id': 'trace-1',
            'parent_span_id': 'run-span-1',
        },
    )


@pytest.mark.parametrize(
    ('values', 'field_name'),
    [
        ({'host': '', 'user': 'runner', 'db': 'traces'}, 'host'),
        ({'host': 'mysql.example', 'user': '', 'db': 'traces'}, 'user'),
        ({'host': 'mysql.example', 'user': 'runner', 'db': ''}, 'db'),
    ],
)
def test_tdsql_db_config_requires_a_complete_target(values: dict[str, Any], field_name: str) -> None:
    with pytest.raises(ValueError, match=field_name):
        build_db_config(**values)


@pytest.mark.parametrize('port', [False, 0, 65_536])
def test_tdsql_db_config_rejects_invalid_ports(port: int) -> None:
    with pytest.raises(ValueError, match='port'):
        build_db_config(host='mysql.example', port=port, user='runner', db='traces')


@pytest.mark.parametrize('backend_class', [TDSQLTraceSink, TDSQLTraceSource])
def test_tdsql_backends_validate_config_at_construction(backend_class) -> None:
    with pytest.raises(ValueError, match='db'):
        backend_class(host='mysql.example', user='runner')


def test_principal_trace_table_name_is_deterministic_and_bounded() -> None:
    assert principal_trace_scope_id(OWNER_PRINCIPAL_ID) == PRINCIPAL_SCOPE_ID
    assert principal_trace_table_name(PRINCIPAL_SCOPE_ID) == (
        'trajectory_span_events_pr_3190e1c7731fff7995861c3221cec9a6'
    )
    assert len(principal_trace_table_name(PRINCIPAL_SCOPE_ID)) == 58


def test_principal_trace_schema_exports_generic_names() -> None:
    assert tdsql_schema.principal_trace_table_name is principal_trace_table_name
    assert tdsql_schema.principal_trace_check_constraint_name is principal_trace_check_constraint_name
    assert tdsql_schema.principal_trace_table_statement is principal_trace_table_statement


@pytest.mark.parametrize('owner_principal_id', ['', 'owner/unsafe', 'x' * 65, 123])
def test_principal_trace_scope_id_rejects_invalid_owner_identity(owner_principal_id: object) -> None:
    with pytest.raises(ValueError, match='owner_principal_id'):
        principal_trace_scope_id(owner_principal_id)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    'principal_scope_id',
    [
        '',
        'principal-owner-a',
        '3190E1C7731FFF7995861C3221CEC9A6',
        '3190e1c7731fff7995861c3221cec9a',
        '3190e1c7731fff7995861c3221cec9a6;DROP TABLE users',
    ],
)
def test_principal_trace_table_name_rejects_noncanonical_scope_ids(principal_scope_id: str) -> None:
    with pytest.raises(ValueError, match='principal_scope_id'):
        principal_trace_table_name(principal_scope_id)


def test_principal_trace_table_supports_standalone_and_batch_runs() -> None:
    statement = principal_trace_table_statement(PRINCIPAL_SCOPE_ID)
    normalized_statement = ' '.join(statement.split())
    constraint_name = principal_trace_check_constraint_name(PRINCIPAL_SCOPE_ID)

    assert f'CREATE TABLE IF NOT EXISTS {principal_trace_table_name(PRINCIPAL_SCOPE_ID)}' in statement
    assert len(constraint_name) <= 64
    assert f'CONSTRAINT {constraint_name} CHECK' in statement
    assert 'trajectory_group_id' not in statement
    assert 'parent_execution_id VARCHAR(64) NULL' in statement
    assert 'PRIMARY KEY (id, run_id)' in statement
    assert 'UNIQUE KEY uq_traj_span_events_record (run_id, run_span_id, record_id)' in statement
    assert (
        'INDEX idx_traj_span_events_parent_cursor (parent_execution_id, time_unix_nano, run_id, run_span_id, record_id)'
    ) in normalized_statement
    assert 'shardkey=run_id' in statement
    assert principal_trace_table_upgrade_statement(PRINCIPAL_SCOPE_ID) == (
        f'ALTER TABLE {principal_trace_table_name(PRINCIPAL_SCOPE_ID)} '
        'MODIFY COLUMN parent_execution_id VARCHAR(64) NULL'
    )


class _Writer:
    def __init__(self) -> None:
        self.rows: list[tuple[tuple[Any, ...], str, str | None, str | None]] = []
        self.settled_through = 0

    def enqueue_span_event(
        self,
        params: tuple[Any, ...],
        *,
        run_span_id: str = '',
        principal_scope_id: str | None = None,
        parent_execution_id: str | None = None,
    ) -> int:
        self.rows.append((params, run_span_id, principal_scope_id, parent_execution_id))
        self.settled_through += 1
        return self.settled_through

    async def wait_until_settled(self, seq: int, *, timeout: float) -> None:
        del timeout
        assert seq <= self.settled_through

    def consume_run_failure(self, run_span_id: str) -> None:
        del run_span_id
        return


@pytest.mark.parametrize(
    ('artifact_available', 'limit', 'span_id', 'omitted'),
    [
        (True, 700, 'run-span-1', True),
        (False, 700, 'run-span-1', False),
        (True, 0, 'run-span-1', False),
        (True, None, 'run-span-1', False),
        (True, 700, 'child-span', False),
        (True, 100000, 'run-span-1', False),
    ],
)
async def test_tdsql_omits_large_run_output_only_with_artifact_fallback(
    monkeypatch, artifact_available, limit, span_id, omitted
):
    writer = _Writer()
    monkeypatch.setattr('nexus.runner.trace.tdsql.writer.get_writer', AsyncMock(return_value=writer))
    context = RunTraceContext(
        run_id='run-1', run_span_id='run-span-1', run_kind='single', result_artifact_available=artifact_available
    )
    sink_options = {} if limit is None else {'max_run_output_bytes': limit}
    sink = TDSQLTraceSink(host='mysql.example', user='runner', db='traces', **sink_options)
    await sink.start(context)
    record = _record().model_copy(update={'span_id': span_id})
    record.event_data['events'] = [
        {
            'name': 'run.event',
            'attributes': {
                'event_type': 'run.output',
                'event_data': {
                    'run_id': 'run-1',
                    'success': True,
                    'output': {'success': True, 'task_output': {'text': '截图' * (1_500_000 if limit is None else 1000)}},
                },
            },
        }
    ]
    original = copy.deepcopy(record.event_data)
    await sink.on_event(context, record)
    await sink.close(context)
    stored = json.loads(writer.rows[0][0][-1])
    payload = stored['events'][0]['attributes']['event_data']
    assert record.event_data == original
    assert payload['success'] is True
    assert (payload['output'] is None) == omitted
    if omitted:
        assert payload['output_omitted'] is True
        assert payload['output_omitted_reason'] == 'size_limit'
        assert payload['original_event_bytes'] > limit
        assert len(writer.rows[0][0][-1].encode()) < limit
    else:
        assert stored == original


async def test_tdsql_result_reader_does_not_reuse_output_before_an_omission(monkeypatch):
    source = TDSQLTraceSource(host='mysql.example', user='reader', db='traces')
    rows = []
    for record_id, output in [(2, None), (1, {'success': True, 'old': True})]:
        row = _record(record_id)
        row.event_data['events'] = [
            {
                'name': 'run.event',
                'attributes': {'event_type': 'run.output', 'event_data': {'output': output}},
            }
        ]
        rows.append(row)
    monkeypatch.setattr(TDSQLTraceSource, '_read_span_records', AsyncMock(return_value=rows))
    with pytest.raises(FileNotFoundError, match='not available'):
        await source.read_result({'run_span_id': 'run-span-1'})


@pytest.mark.asyncio
async def test_tdsql_sink_skips_transient_stream_updates_without_starting_writer(monkeypatch) -> None:
    async def get_writer(_config, **_kwargs):
        pytest.fail('transient stream updates must not start the TDSQL writer')

    monkeypatch.setattr('nexus.runner.trace.tdsql.writer.get_writer', get_writer)
    context = RunTraceContext(run_id='run-1', run_span_id='run-span-1', run_kind='live')
    sink = TDSQLTraceSink(host='mysql.example', user='runner', db='traces')
    await sink.start(context)

    for record_id, event_name in enumerate(
        (
            'agent.stream',
            'chat.completion.delta',
            'chat.completion.reasoning.delta',
            'chat.completion.tool_call.delta',
        ),
        start=1,
    ):
        await sink.on_event(context, _event_record(event_name, record_id=record_id))

    assert sink.metadata()['trace_event_cursor'] == 4


@pytest.mark.asyncio
async def test_tdsql_sink_skips_update_containing_only_transient_stream_events(monkeypatch) -> None:
    async def get_writer(_config, **_kwargs):
        pytest.fail('transient stream updates must not start the TDSQL writer')

    monkeypatch.setattr('nexus.runner.trace.tdsql.writer.get_writer', get_writer)
    context = RunTraceContext(run_id='run-1', run_span_id='run-span-1', run_kind='live')
    sink = TDSQLTraceSink(host='mysql.example', user='runner', db='traces')
    await sink.start(context)
    record = _event_record('agent.stream', record_id=7)
    record.event_data['events'].append({'name': 'chat.completion.delta', 'time_unix_nano': 7, 'attributes': {}})

    await sink.on_event(context, record)

    assert sink.metadata()['trace_event_cursor'] == 7


@pytest.mark.asyncio
@pytest.mark.parametrize('event_type', ['assistant', 'result'])
async def test_tdsql_sink_persists_terminal_agent_stream_events(monkeypatch, event_type: str) -> None:
    writer = _Writer()

    async def get_writer(_config, **_kwargs):
        return writer

    monkeypatch.setattr('nexus.runner.trace.tdsql.writer.get_writer', get_writer)
    context = RunTraceContext(run_id='run-1', run_span_id='run-span-1', run_kind='live')
    sink = TDSQLTraceSink(host='mysql.example', user='runner', db='traces')
    await sink.start(context)
    record = _event_record('agent.stream')
    record.event_data['events'][0]['attributes'] = {'event_type': event_type}

    await sink.on_event(context, record)

    assert len(writer.rows) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('event_name', ['chat.completion.completed', 'chat.completion.failed', 'run.event'])
async def test_tdsql_sink_persists_terminal_and_run_updates(monkeypatch, event_name: str) -> None:
    writer = _Writer()

    async def get_writer(_config, **_kwargs):
        return writer

    monkeypatch.setattr('nexus.runner.trace.tdsql.writer.get_writer', get_writer)
    context = RunTraceContext(run_id='run-1', run_span_id='run-span-1', run_kind='live')
    sink = TDSQLTraceSink(host='mysql.example', user='runner', db='traces')
    await sink.start(context)

    await sink.on_event(context, _event_record(event_name))

    assert len(writer.rows) == 1


@pytest.mark.asyncio
async def test_tdsql_sink_persists_update_mixing_transient_and_run_events(monkeypatch) -> None:
    writer = _Writer()

    async def get_writer(_config, **_kwargs):
        return writer

    monkeypatch.setattr('nexus.runner.trace.tdsql.writer.get_writer', get_writer)
    context = RunTraceContext(run_id='run-1', run_span_id='run-span-1', run_kind='live')
    sink = TDSQLTraceSink(host='mysql.example', user='runner', db='traces')
    await sink.start(context)
    record = _event_record('agent.stream')
    record.event_data['events'].append({'name': 'run.event', 'time_unix_nano': 1, 'attributes': {}})

    await sink.on_event(context, record)

    assert len(writer.rows) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('field_name', 'value'),
    [
        ('attributes', {'model': 'test'}),
        ('status', {'status_code': 'OK'}),
    ],
)
async def test_tdsql_sink_persists_transient_update_with_span_state(monkeypatch, field_name: str, value: Any) -> None:
    writer = _Writer()

    async def get_writer(_config, **_kwargs):
        return writer

    monkeypatch.setattr('nexus.runner.trace.tdsql.writer.get_writer', get_writer)
    context = RunTraceContext(run_id='run-1', run_span_id='run-span-1', run_kind='live')
    sink = TDSQLTraceSink(host='mysql.example', user='runner', db='traces')
    await sink.start(context)
    record = _event_record('agent.stream')
    record.event_data[field_name] = value

    await sink.on_event(context, record)

    assert len(writer.rows) == 1


@pytest.mark.asyncio
async def test_tdsql_sink_persists_span_end_after_stream_updates(monkeypatch) -> None:
    writer = _Writer()

    async def get_writer(_config, **_kwargs):
        return writer

    monkeypatch.setattr('nexus.runner.trace.tdsql.writer.get_writer', get_writer)
    context = RunTraceContext(run_id='run-1', run_span_id='run-span-1', run_kind='live')
    sink = TDSQLTraceSink(host='mysql.example', user='runner', db='traces')
    await sink.start(context)
    end_record = _record()
    end_record = end_record.model_copy(update={'event_type': 'end', 'event_data': {'type': 'END'}})

    await sink.on_event(context, end_record)

    assert len(writer.rows) == 1


@pytest.mark.asyncio
async def test_principal_sink_routes_batch_rows_and_retains_parent_filter(monkeypatch) -> None:
    writer = _Writer()

    async def get_writer(_config, **_kwargs):
        return writer

    monkeypatch.setattr('nexus.runner.trace.tdsql.writer.get_writer', get_writer)
    context = RunTraceContext(
        run_id='exe_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
        run_span_id='run-span-1',
        run_kind='single',
        metadata={
            'storage_scope': 'principal_v1',
            'principal_scope_id': PRINCIPAL_SCOPE_ID,
            'parent_execution_id': PARENT_EXECUTION_ID,
        },
    )
    sink = TDSQLTraceSink(
        host='mysql.example',
        user='runner',
        db='traces',
        execution_id=context.run_id,
        assignment_id='asn-1',
        attempt_id='attempt-1',
    )

    await sink.start(context)
    await sink.on_event(context, _record())
    await sink.close(context)

    assert writer.rows == [
        (
            (
                context.run_id,
                context.run_span_id,
                PARENT_EXECUTION_ID,
                1,
                'update',
                1,
                'trace-1',
                'run-span-1',
                '',
                '{"events": []}',
            ),
            context.run_span_id,
            PRINCIPAL_SCOPE_ID,
            PARENT_EXECUTION_ID,
        )
    ]
    metadata = sink.metadata()
    assert metadata['storage_scope'] == 'principal_v1'
    assert metadata['principal_scope_id'] == PRINCIPAL_SCOPE_ID
    assert metadata['parent_execution_id'] == PARENT_EXECUTION_ID
    assert 'trajectory_group_id' not in metadata
    assert 'table_name' not in metadata


@pytest.mark.asyncio
async def test_principal_sink_routes_standalone_rows_without_parent(monkeypatch) -> None:
    writer = _Writer()

    async def get_writer(_config, **_kwargs):
        return writer

    monkeypatch.setattr('nexus.runner.trace.tdsql.writer.get_writer', get_writer)
    context = RunTraceContext(
        run_id='exe_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb',
        run_span_id='run-span-live',
        run_kind='live',
        metadata={
            'storage_scope': 'principal_v1',
            'principal_scope_id': PRINCIPAL_SCOPE_ID,
        },
    )
    sink = TDSQLTraceSink(host='mysql.example', user='runner', db='traces')

    await sink.start(context)
    await sink.on_event(context, _record())
    await sink.close(context)

    row, run_span_id, principal_scope_id, parent_execution_id = writer.rows[0]
    assert row[:3] == (context.run_id, context.run_span_id, None)
    assert run_span_id == context.run_span_id
    assert principal_scope_id == PRINCIPAL_SCOPE_ID
    assert parent_execution_id is None
    assert sink.metadata()['storage_scope'] == 'principal_v1'
    assert 'parent_execution_id' not in sink.metadata()
    assert 'trajectory_group_id' not in sink.metadata()


def test_principal_source_derives_table_and_ignores_metadata_identifiers() -> None:
    sql, params = _build_query(
        {
            'storage_scope': 'principal_v1',
            'principal_scope_id': PRINCIPAL_SCOPE_ID,
            'parent_execution_id': PARENT_EXECUTION_ID,
            'table_name': 'attacker_controlled',
            'run_id': 'exe_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
            'run_span_id': 'run-span-1',
        },
        4,
        limit=20,
    )

    assert f'FROM {principal_trace_table_name(PRINCIPAL_SCOPE_ID)}' in sql
    assert 'attacker_controlled' not in sql
    assert 'trajectory_group_id' not in sql
    assert params == (
        'exe_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
        'run-span-1',
        PARENT_EXECUTION_ID,
        4,
        20,
    )


def test_principal_source_queries_standalone_run_without_parent_or_shared_group() -> None:
    sql, params = _build_query(
        {
            'storage_scope': 'principal_v1',
            'principal_scope_id': PRINCIPAL_SCOPE_ID,
            'run_id': 'exe_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb',
            'run_span_id': 'run-span-live',
        },
        2,
    )

    assert f'FROM {principal_trace_table_name(PRINCIPAL_SCOPE_ID)}' in sql
    assert 'parent_execution_id =' not in sql
    assert 'trajectory_group_id' not in sql
    assert params == ('exe_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb', 'run-span-live', 2)


def test_principal_source_reads_legacy_batch_metadata() -> None:
    sql, params = _build_query(
        {
            'storage_scope': 'batch_principal_v1',
            'principal_scope_id': PRINCIPAL_SCOPE_ID,
            'parent_execution_id': PARENT_EXECUTION_ID,
            'run_id': 'exe_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
        },
        0,
    )

    assert f'FROM {principal_trace_table_name(PRINCIPAL_SCOPE_ID)}' in sql
    assert 'parent_execution_id = %s' in sql
    assert params == ('exe_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa', PARENT_EXECUTION_ID, 0)


def test_batch_source_rejects_forged_principal_scope_metadata() -> None:
    with pytest.raises(ValueError, match='principal_scope_id'):
        _build_query(
            {
                'storage_scope': 'batch_principal_v1',
                'principal_scope_id': 'trajectory_span_events; DROP TABLE users',
                'parent_execution_id': PARENT_EXECUTION_ID,
                'run_id': 'exe_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
            },
            0,
        )


def test_one_writer_routes_same_principal_parents_to_one_table_without_per_table_instances() -> None:
    writer = TDSQLTraceWriter()
    writer._started = True

    first_seq = writer.enqueue_span_event(
        ('first',),
        run_span_id='span-1',
        principal_scope_id=PRINCIPAL_SCOPE_ID,
        parent_execution_id=PARENT_EXECUTION_ID,
    )
    second_seq = writer.enqueue_span_event(
        ('second',),
        run_span_id='span-2',
        principal_scope_id=PRINCIPAL_SCOPE_ID,
        parent_execution_id=OTHER_PARENT_EXECUTION_ID,
    )

    assert (first_seq, second_seq) == (1, 2)
    queued = list(writer._queue)
    assert principal_trace_table_name(PRINCIPAL_SCOPE_ID) in queued[0].sql
    assert principal_trace_table_name(PRINCIPAL_SCOPE_ID) in queued[1].sql
    assert queued[0].principal_scope_id == PRINCIPAL_SCOPE_ID
    assert queued[1].principal_scope_id == PRINCIPAL_SCOPE_ID
    assert queued[0].parent_execution_id == PARENT_EXECUTION_ID
    assert queued[1].parent_execution_id == OTHER_PARENT_EXECUTION_ID
    assert queued[0].sql == queued[1].sql


@pytest.mark.asyncio
async def test_writer_registry_shares_by_database_config(monkeypatch) -> None:
    class Writer:
        def __init__(self, **_kwargs) -> None:
            self.configs: list[dict[str, Any] | None] = []
            self.is_ready = True

        async def ensure_started(self, db_config: dict[str, Any] | None) -> bool:
            self.configs.append(db_config)
            return True

    monkeypatch.setattr(tdsql_writer, 'TDSQLTraceWriter', Writer)
    monkeypatch.setattr(tdsql_writer, '_writer', None, raising=False)
    monkeypatch.setattr(tdsql_writer, '_writers', {}, raising=False)
    first_config = {'host': 'mysql-a', 'port': 3306, 'user': 'runner', 'password': 'one', 'db': 'traces'}
    same_config = dict(reversed(first_config.items()))
    second_config = {**first_config, 'db': 'other_traces'}
    rotated_config = {**first_config, 'password': 'two'}

    first = await tdsql_writer.get_writer(first_config)
    same = await tdsql_writer.get_writer(same_config)
    second = await tdsql_writer.get_writer(second_config)
    rotated = await tdsql_writer.get_writer(rotated_config)

    assert first is same
    assert second is not first
    assert rotated is not first
    assert rotated is not second


@pytest.mark.asyncio
async def test_started_writer_rejects_database_config_mismatch() -> None:
    writer = TDSQLTraceWriter()
    writer._db_config = {'host': 'mysql-a', 'db': 'traces'}
    writer._started = True

    with pytest.raises(ValueError, match='already bound to a different database config'):
        await writer.ensure_started({'host': 'mysql-b', 'db': 'traces'})


class _FlushCursor:
    def __init__(self, *, failing_sql: str | None = None) -> None:
        self.failing_sql = failing_sql
        self.calls: list[tuple[str, str, list[tuple[Any, ...]]]] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def execute(self, sql: str, params: tuple[Any, ...]) -> None:
        self.calls.append(('execute', sql, [params]))
        if sql == self.failing_sql:
            raise ValueError('rejected table')

    async def executemany(self, sql: str, rows: list[tuple[Any, ...]]) -> None:
        self.calls.append(('executemany', sql, list(rows)))
        if sql == self.failing_sql:
            raise ValueError('rejected table')


class _FlushConnection:
    def __init__(self, *, failing_sql: str | None = None) -> None:
        self.cursor_instance = _FlushCursor(failing_sql=failing_sql)
        self.commits = 0
        self.rollbacks = 0

    def cursor(self) -> _FlushCursor:
        return self.cursor_instance

    async def commit(self) -> None:
        self.commits += 1

    async def rollback(self) -> None:
        self.rollbacks += 1


def _write(
    sql: str,
    seq: int,
    *,
    run_span_id: str,
    principal_scope_id: str,
    parent_execution_id: str,
) -> _Write:
    return _Write(
        sql=sql,
        params=(
            f'run-{seq}',
            run_span_id,
            parent_execution_id,
            seq,
            'update',
            seq,
            'trace',
            run_span_id,
            '',
            '{}',
        ),
        seq=seq,
        run_span_id=run_span_id,
        principal_scope_id=principal_scope_id,
        parent_execution_id=parent_execution_id,
    )


@pytest.mark.asyncio
async def test_writer_groups_interleaved_principal_rows_by_sql() -> None:
    writer = TDSQLTraceWriter()
    connection = _FlushConnection()
    writer._conn = connection

    await writer._flush_batch(
        [
            _write(
                'insert-a',
                1,
                run_span_id='span-a1',
                principal_scope_id=PRINCIPAL_SCOPE_ID,
                parent_execution_id=PARENT_EXECUTION_ID,
            ),
            _write(
                'insert-b',
                2,
                run_span_id='span-b',
                principal_scope_id=OTHER_PRINCIPAL_SCOPE_ID,
                parent_execution_id=OTHER_PARENT_EXECUTION_ID,
            ),
            _write(
                'insert-a',
                3,
                run_span_id='span-a2',
                principal_scope_id=PRINCIPAL_SCOPE_ID,
                parent_execution_id=OTHER_PARENT_EXECUTION_ID,
            ),
        ]
    )

    assert connection.cursor_instance.calls == [
        (
            'executemany',
            'insert-a',
            [
                (
                    'run-1',
                    'span-a1',
                    PARENT_EXECUTION_ID,
                    1,
                    'update',
                    1,
                    'trace',
                    'span-a1',
                    '',
                    '{}',
                ),
                (
                    'run-3',
                    'span-a2',
                    OTHER_PARENT_EXECUTION_ID,
                    3,
                    'update',
                    3,
                    'trace',
                    'span-a2',
                    '',
                    '{}',
                ),
            ],
        ),
        (
            'execute',
            'insert-b',
            [
                (
                    'run-2',
                    'span-b',
                    OTHER_PARENT_EXECUTION_ID,
                    2,
                    'update',
                    2,
                    'trace',
                    'span-b',
                    '',
                    '{}',
                )
            ],
        ),
    ]
    assert connection.commits == 2
    assert writer.settled_through == 3


@pytest.mark.asyncio
async def test_writer_isolates_principal_table_write_failures() -> None:
    writer = TDSQLTraceWriter()
    connection = _FlushConnection(failing_sql='insert-a')
    writer._conn = connection

    await writer._flush_batch(
        [
            _write(
                'insert-a',
                1,
                run_span_id='span-a',
                principal_scope_id=PRINCIPAL_SCOPE_ID,
                parent_execution_id=PARENT_EXECUTION_ID,
            ),
            _write(
                'insert-b',
                2,
                run_span_id='span-b',
                principal_scope_id=OTHER_PRINCIPAL_SCOPE_ID,
                parent_execution_id=OTHER_PARENT_EXECUTION_ID,
            ),
        ]
    )

    assert [call[1] for call in connection.cursor_instance.calls] == ['insert-a', 'insert-b']
    assert connection.rollbacks == 1
    assert connection.commits == 1
    assert writer.consume_run_failure('span-a') == 1
    assert writer.consume_run_failure('span-b') is None
    assert writer.settled_through == 2


def test_writer_retains_failures_until_each_run_acknowledges_them() -> None:
    writer = TDSQLTraceWriter()
    writes = [
        _write(
            'insert-a',
            index + 1,
            run_span_id=f'span-{index}',
            principal_scope_id=PRINCIPAL_SCOPE_ID,
            parent_execution_id=PARENT_EXECUTION_ID,
        )
        for index in range(501)
    ]

    writer._record_run_failures(writes)

    assert [writer.consume_run_failure(f'span-{index}') for index in range(501)] == list(range(1, 502))


@pytest.mark.asyncio
async def test_tdsql_sink_prepares_principal_trace_table_for_live_runs(monkeypatch) -> None:
    class Cursor:
        def __init__(self) -> None:
            self.calls: list[tuple[str, tuple[Any, ...] | None]] = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def execute(self, statement: str, params: tuple[Any, ...] | None = None) -> None:
            self.calls.append((statement, params))

        async def fetchone(self):
            return None

    class Connection:
        def __init__(self) -> None:
            self.cursor_instance = Cursor()
            self.commits = 0
            self.closed = False

        def cursor(self) -> Cursor:
            return self.cursor_instance

        async def commit(self) -> None:
            self.commits += 1

        def close(self) -> None:
            self.closed = True

    connection = Connection()

    async def connect(**_kwargs):
        return connection

    monkeypatch.setattr('nexus.runner.trace.tdsql._driver.connect', connect)
    sink = TDSQLTraceSink(host='mysql.example', user='writer', db='traces')
    source = TDSQLTraceSource(host='mysql.example', user='reader', db='traces')

    assert isinstance(sink, RunTraceScopePreparer)
    assert not isinstance(source, RunTraceScopePreparer)
    await sink.prepare_scope(RunTraceScope(run_kind='live', scope_id=PRINCIPAL_SCOPE_ID))

    assert connection.cursor_instance.calls == [
        (
            'SELECT IS_NULLABLE FROM information_schema.COLUMNS '
            'WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s AND COLUMN_NAME = %s LIMIT 1',
            ('traces', principal_trace_table_name(PRINCIPAL_SCOPE_ID), 'parent_execution_id'),
        ),
        (principal_trace_table_statement(PRINCIPAL_SCOPE_ID), None),
    ]
    assert connection.commits == 1
    assert connection.closed is True
