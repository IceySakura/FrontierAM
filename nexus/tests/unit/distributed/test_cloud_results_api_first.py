from __future__ import annotations

import asyncio
import copy
import json
import sqlite3
import threading

from collections import Counter
from typing import Any, cast

import pytest

from nexus.distributed.export import results
from nexus.distributed.export.types import ExportError, TraceResult
from nexus.distributed.export.writer import AtomicJsonlOutput


class Api:
    def __init__(self, statuses=('succeeded', 'failed')):
        self.items = [
            {
                'item_id': f'item-{i:03d}',
                'status': status,
                'current_attempt_no': 0,
                'current_child_execution_id': f'child-{i}',
                'metadata': {'row': i},
            }
            for i, status in enumerate(statuses)
        ]
        self.outputs = {'child-0': {'success': True, 'task_output': {'value': 'saved'}}}
        self.executions = {
            'batch': {
                'execution_id': 'batch',
                'spec_type': 'nexus_batch',
                'status': 'failed',
                'attempt_count': 1,
                'current_attempt_id': 'attempt-batch',
                'created_at': '2026-09-21T00:00:00',
            },
            **{
                f'child-{i}': {
                    'execution_id': f'child-{i}',
                    'spec_type': 'nexus_single',
                    'status': 'killed' if status == 'cancelled' else status,
                    'attempt_count': 1,
                    'current_attempt_id': f'attempt-child-{i}',
                    'result_artifact_id': f'artifact-child-{i}',
                    'parent_execution_id': 'batch',
                    'batch_context': {'item_id': f'item-{i:03d}', 'attempt_no': 0},
                    'created_at': '2026-09-21T00:00:01',
                    'error_json': {'type': 'TaskFailed', 'message': 'original failure'}
                    if status != 'succeeded'
                    else None,
                }
                for i, status in enumerate(statuses)
            },
        }
        self.execution_calls = Counter()
        self.result_calls = []
        self.event_calls = []

    async def get_execution(self, execution_id):
        self.execution_calls[execution_id] += 1
        return copy.deepcopy(self.executions[execution_id])

    async def get_batch_summary(self, _execution_id):
        return {
            'generation_status': 'complete',
            'submission_closed': True,
            'summary_version': 1,
            'materialized_count': len(self.items),
            'total_items': len(self.items),
            'terminal_items': len(self.items),
            'item_status_counts': dict(Counter(x['status'] for x in self.items)),
        }

    async def get_batch_items(self, _execution_id, cursor):
        assert cursor is None
        return {'items': self.items, 'next_cursor': None}

    async def get_result(self, execution_id):
        self.result_calls.append(execution_id)
        value = self.outputs.get(execution_id)
        if isinstance(value, Exception):
            raise value
        return value

    async def get_result_to_file(self, execution_id, fileobj, chunk_size=1 << 20, scan=None):
        # Disk-prefetch companion of get_result; result_calls tracking keeps
        # pipeline-level assertions covering both fetch paths.
        self.result_calls.append(execution_id)
        value = self.outputs.get(execution_id)
        if isinstance(value, Exception):
            raise value
        if value is None:
            return None
        data = json.dumps(value).encode()
        fileobj.write(data)
        if scan is not None:
            scan.feed(data)
        return len(data)

    async def get_control_events(self, execution_id):
        self.event_calls.append(execution_id)
        return [
            {'type': 'execution.status.changed', 'payload': {'to': 'running'}, 'created_at': '2026-09-21T01:00:00'},
            {
                'type': 'execution.status.changed',
                'payload': {'to': self.executions[execution_id]['status']},
                'created_at': '2026-09-21T01:00:01',
            },
        ]


async def export(api, path, reader=None, *, allow_partial=False, result_status=None):
    await results.export_execution(
        api=api,
        execution_id='batch',
        output=path,
        force=False,
        trace_reader=reader,
        allow_partial=allow_partial,
        result_status=result_status,
    )
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.fixture
def forbid_file_filter(monkeypatch):
    async def unexpected_filter(self, keep):
        pytest.fail('file filtering must not run after an input read or validation error')

    # The default path filters by index; the parse-per-line path is the strict /
    # index-dropped fallback. Guard both so neither can run after an error.
    monkeypatch.setattr(AtomicJsonlOutput, 'filter_records', unexpected_filter)
    monkeypatch.setattr(AtomicJsonlOutput, 'filter_records_by_index', unexpected_filter)


async def test_batch_results_export_without_database_preserves_missing_failure(tmp_path):
    rows = await export(Api(), tmp_path / 'results.jsonl')
    assert [x['status'] for x in rows] == ['success', 'failed']
    assert rows[0]['task_output'] == {'value': 'saved'}
    assert rows[1]['task_output'] is None
    assert rows[1]['error']['type'] == 'TaskFailed'
    assert rows[0]['start_time'] == '2026-09-21T01:00:00'


@pytest.mark.parametrize('success', [False, True])
async def test_configured_database_takes_priority_without_application_result_reads(tmp_path, success):
    api = Api(('succeeded' if success else 'failed',))

    class Reader:
        async def read_many(self, ids, **kwargs):
            assert ids == ['child-0']
            return {'child-0': TraceResult(output={'success': success, 'task_output': {'value': 'database'}})}

    rows = await export(api, tmp_path / 'results.jsonl', Reader())
    assert rows[0]['task_output'] == {'value': 'database'}
    assert rows[0]['status'] == ('success' if success else 'failed')
    assert api.result_calls == []


@pytest.mark.parametrize('missing_trace', [{}, {'child-0': TraceResult()}])
async def test_only_missing_database_output_uses_application_fallback(tmp_path, missing_trace):
    api = Api(('succeeded', 'succeeded'))

    class Reader:
        async def read_many(self, ids, **kwargs):
            if ids == ['child-0']:
                return missing_trace
            assert ids == ['child-1']
            return {
                'child-1': TraceResult(
                    output={'success': True, 'task_output': {'value': 'recovered'}},
                    start_time='trace-start',
                    end_time='trace-end',
                )
            }

    rows = await export(api, tmp_path / 'results.jsonl', Reader())
    assert rows[0]['task_output'] == {'value': 'saved'}
    assert rows[1]['task_output'] == {'value': 'recovered'}
    assert rows[1]['start_time'] == '2026-09-21T01:00:00'  # Control remains authoritative.
    assert api.result_calls == ['child-0']


@pytest.mark.parametrize('with_database', [False, True])
@pytest.mark.parametrize('selection', [None, 'failed'])
@pytest.mark.usefixtures('forbid_file_filter')
async def test_api_read_error_is_not_treated_as_missing_output(tmp_path, with_database, selection):
    api = Api(('failed' if selection == 'failed' else 'succeeded',))
    api.outputs['child-0'] = ExportError('API unavailable')

    class Reader:
        async def read_many(self, ids, **kwargs):
            return {}

    with pytest.raises(ExportError, match='item item-000, child child-0: API unavailable'):
        await export(
            api,
            tmp_path / 'results.jsonl',
            Reader() if with_database else None,
            allow_partial=selection is not None,
            result_status=selection,
        )
    assert not (tmp_path / 'results.jsonl').exists()
    assert api.result_calls == ['child-0']


@pytest.mark.parametrize('selection', [None, 'failed'])
@pytest.mark.usefixtures('forbid_file_filter')
async def test_database_read_error_is_not_treated_as_missing_output(tmp_path, selection):
    api = Api(('failed' if selection == 'failed' else 'succeeded',))

    class Reader:
        async def read_many(self, ids, **kwargs):
            raise ExportError('database unavailable')

    with pytest.raises(ExportError, match='child child-0: database unavailable'):
        await export(
            api,
            tmp_path / 'results.jsonl',
            Reader(),
            allow_partial=selection is not None,
            result_status=selection,
        )
    assert not (tmp_path / 'results.jsonl').exists()
    assert api.result_calls == []


@pytest.mark.parametrize('with_database', [False, True])
async def test_success_without_any_output_prevents_publication(tmp_path, with_database):
    api = Api(('succeeded',))
    api.outputs.clear()

    class Reader:
        async def read_many(self, ids, **kwargs):
            return {}

    with pytest.raises(ExportError, match='successful execution has no RunOutput'):
        await export(api, tmp_path / 'results.jsonl', Reader() if with_database else None)
    assert not (tmp_path / 'results.jsonl').exists()


@pytest.mark.parametrize('cancel', [False, True])
async def test_ordered_writer_makes_progress_with_bounded_lookahead(cancel):
    gate = asyncio.Event()
    written = asyncio.Event()
    cancelled = asyncio.Event()

    class SlowApi(Api):
        async def get_result(self, execution_id):
            self.result_calls.append(execution_id)
            if execution_id == 'child-1':
                try:
                    await gate.wait()
                except asyncio.CancelledError:
                    cancelled.set()
                    raise
            return {'success': True}

    class Writer:
        rows = []

        async def write_records(self, records):
            assert len(records) == 1
            self.rows.extend(records)
            written.set()

    api = SlowApi(('succeeded',) * 12)
    writer = Writer()
    singles = [results._batch_export({'execution_id': 'batch', 'spec_type': 'nexus_batch'}, item) for item in api.items]
    plans = [results._RecordPlan(single=single) for single in singles]
    task = asyncio.create_task(results._write_records(cast(Any, writer), cast(Any, api), plans))
    try:
        await asyncio.wait_for(written.wait(), 2)
        await asyncio.sleep(0.02)
        assert [x['run_id'] for x in writer.rows] == ['item-000']
        assert len(api.result_calls) <= results._in_ram_concurrency() + 1
        if cancel:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert cancelled.is_set()
        else:
            gate.set()
            await asyncio.wait_for(task, 2)
            assert [x['run_id'] for x in writer.rows] == [f'item-{i:03d}' for i in range(12)]
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize('partitioned', [False, True])
async def test_database_root_paging_ignores_nested_outputs_and_retains_latest(monkeypatch, partitioned):
    db = sqlite3.connect(':memory:')
    table = 'trajectory_span_events_pr_' + 'a' * 32
    db.execute(
        f'CREATE TABLE {table} (run_id TEXT, run_span_id TEXT, record_id INT, '
        'time_unix_nano INT, span_id TEXT, event_type TEXT, event_data TEXT)'
    )

    def output(value):
        return json.dumps({
            'events': [
                {
                    'name': 'run.event',
                    'attributes': {
                        'event_type': 'run.output',
                        'event_data': {'output': {'success': True, 'task_output': value}},
                    },
                }
            ]
        })

    db.executemany(
        f'INSERT INTO {table} VALUES (?,?,?,?,?,?,?)',
        [
            ('child', 'a', 1, 100, 'a', 'update', output('old')),
            ('child', 'a', 2, 200, 'a', 'update', '{}'),
            ('child', 'b', 1, 300, 'b', 'update', output('new')),
            ('child', 'b', 2, 400, 'nested-agent', 'update', output('wrong nested output')),
            ('other', 'a', 1, 900, 'a', 'update', output('wrong child')),
        ],
    )
    partition_key = '2026-09-21 00:00:00.000000' if partitioned else None
    if partitioned:
        db.execute(f'ALTER TABLE {table} ADD COLUMN parent_created_at TEXT')
        db.execute(f'UPDATE {table} SET parent_created_at = ?', (partition_key,))
        db.execute(
            f'INSERT INTO {table} VALUES (?,?,?,?,?,?,?,?)',
            ('child', 'b', 1, 999, 'b', 'update', output('wrong partition'), '2026-09-22 00:00:00.000000'),
        )
    db.execute(f'CREATE INDEX idx_traj_span_events_span_id ON {table}(span_id)')
    reads = []
    decoded = []
    decode = results._decode_event_data

    def track_decode(value):
        decoded.append(value)
        return decode(value)

    monkeypatch.setattr(results, '_decode_event_data', track_decode)

    class Cursor:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def execute(self, sql, params):
            assert 'LIKE' not in sql
            if partitioned:
                assert 'parent_created_at = %s' in sql
                assert partition_key in params
            reads.append(sql)
            translated = sql.replace(
                'FORCE INDEX (idx_traj_span_events_span_id)', 'INDEXED BY idx_traj_span_events_span_id'
            )
            self.rows = db.execute(translated.replace('%s', '?'), params).fetchall()
            if sql.startswith('SELECT event_data'):
                assert 'ORDER BY' not in sql
                assert len(self.rows) == 1

        def fetchall(self):
            return self.rows

    class Connection:
        def cursor(self):
            return Cursor()

        def close(self):
            pass

    async def connect(**kwargs):
        return Connection()

    monkeypatch.setattr(results, '_TDSQL_ROOT_PAGE_SIZE', 2)
    reader = results.TdsqlTraceReader(
        host='unused',
        port=3306,
        user='readonly',
        password='unused',
        database='trace',
        table=table,
        parent_created_at=partition_key,
        connect=connect,
        transient_errors=(),
    )
    try:
        recovered = await reader.read_many(['child'])
        recovered_output = recovered['child'].output
        assert recovered_output is not None
        assert recovered_output['task_output'] == 'new'
        # Reverse mode discovers spans once and reads only older records when
        # the newest record has no output. Two spans still require the same
        # two span pages as the full walk.
        assert sum(sql.startswith('SELECT DISTINCT run_span_id') for sql in reads) == 2
        assert len(decoded) == sum(sql.startswith('SELECT event_data') for sql in reads)
    finally:
        db.close()


async def test_writer_cancellation_waits_for_encoder_before_closing(tmp_path, monkeypatch):
    writer = AtomicJsonlOutput(tmp_path / 'results.jsonl', force=False)
    await writer.open()
    entered = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()
    write = writer._write_records

    def blocked_write(records):
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(5)
        write(records)

    monkeypatch.setattr(writer, '_write_records', blocked_write)
    task = asyncio.create_task(writer.write_records([{'run_id': 'item', 'value': 'kept intact'}]))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await writer.abort()
    assert json.loads(writer.partial.read_text()) == {'run_id': 'item', 'value': 'kept intact'}
    assert not writer.output.exists()


async def test_database_failure_reports_child_and_errno_without_secret_text():
    class DatabaseError(Exception):
        pass

    async def connect(**kwargs):
        raise DatabaseError(1038, 'sentinel-secret')

    async def sleep(_seconds):
        pass

    reader = results.TdsqlTraceReader(
        host='unused',
        port=3306,
        user='readonly',
        password='unused',
        database='trace',
        table='trajectory_span_events_pr_' + 'a' * 32,
        connect=connect,
        transient_errors=(DatabaseError,),
        sleep=sleep,
    )
    with pytest.raises(ExportError, match=r'child-123.*3 attempts.*DatabaseError.*1038') as error:
        await reader.read_many(['child-123'])
    assert 'sentinel-secret' not in str(error.value)


class PartialApi(Api):
    async def get_execution(self, execution_id):
        execution = await super().get_execution(execution_id)
        if execution_id == 'batch':
            execution['status'] = 'stopping'
        return execution

    async def get_batch_summary(self, execution_id):
        summary = await super().get_batch_summary(execution_id)
        summary['terminal_items'] = sum(item['status'] in results._TERMINAL_ITEM_STATUSES for item in self.items)
        return summary

    async def get_batch_items(self, _execution_id, cursor):
        # One item per page includes pages containing only skipped items.
        index = 0 if cursor is None else next(i + 1 for i, item in enumerate(self.items) if item['item_id'] == cursor)
        items = self.items[index : index + 1]
        return {'items': items, 'next_cursor': items[-1]['item_id'] if index + 1 < len(self.items) else None}


@pytest.mark.parametrize('statuses', [('pending', 'succeeded', 'running', 'failed', 'submitted'), ('running',)])
async def test_partial_batch_exports_only_terminal_items_across_skipped_pages(tmp_path, caplog, statuses):
    api = PartialApi(statuses)
    api.outputs = {
        f'child-{i}': {'success': status == 'succeeded', 'task_output': {'value': i}}
        for i, status in enumerate(statuses)
        if status in results._TERMINAL_ITEM_STATUSES
    }
    rows = await export(api, tmp_path / 'results.jsonl', allow_partial=True, result_status='all')
    expected = [i for i, status in enumerate(statuses) if status in results._TERMINAL_ITEM_STATUSES]
    assert [row['run_id'] for row in rows] == [f'item-{i:03d}' for i in expected]
    assert set(api.result_calls) == {f'child-{i}' for i in expected}
    assert f'read {len(expected)} terminal items; skipped {len(statuses) - len(expected)}' in caplog.text


async def test_partial_batch_allows_completion_during_export(tmp_path):
    class CompletingApi(PartialApi):
        async def get_batch_items(self, execution_id, cursor):
            if cursor is not None:
                self.items[1]['status'] = 'failed'
                self.executions['child-1']['status'] = 'failed'
            return await super().get_batch_items(execution_id, cursor)

        async def get_execution(self, execution_id):
            execution = await super().get_execution(execution_id)
            if execution_id == 'batch' and self.result_calls:
                execution.update(status='failed', error_json={'message': 'stopped'}, result_artifact_id='artifact')
            return execution

        async def get_batch_summary(self, execution_id):
            summary = await super().get_batch_summary(execution_id)
            summary['summary_version'] += len(self.result_calls)
            return summary

    api = CompletingApi(('succeeded', 'running'))
    api.outputs['child-1'] = {'success': False, 'task_output': {'value': 'saved before completion'}}
    rows = await export(api, tmp_path / 'results.jsonl', allow_partial=True, result_status='all')
    assert [row['status'] for row in rows] == ['success', 'failed']


@pytest.mark.parametrize('parent_running', [False, True])
async def test_default_export_still_rejects_incomplete_batch(tmp_path, parent_running):
    class IncompleteApi(PartialApi):
        async def get_execution(self, execution_id):
            execution = await super().get_execution(execution_id)
            if not parent_running:
                execution['status'] = 'killed'
            return execution

    message = 'Execution is not terminal' if parent_running else 'Batch is not fully terminal'
    with pytest.raises(ExportError, match=message):
        await export(IncompleteApi(('succeeded', 'running')), tmp_path / 'results.jsonl')
    assert not (tmp_path / 'results.jsonl').exists()


@pytest.mark.parametrize('failure', ['missing_output', 'api_error', 'invalid_cursor'])
async def test_partial_batch_preserves_result_and_consistency_checks(tmp_path, failure):
    class InvalidApi(PartialApi):
        async def get_batch_items(self, execution_id, cursor):
            if failure == 'invalid_cursor':
                return {'items': [], 'next_cursor': 'broken'}
            return await super().get_batch_items(execution_id, cursor)

    api = InvalidApi(('succeeded', 'running'))
    if failure == 'missing_output':
        api.outputs.clear()
    elif failure == 'api_error':
        api.outputs['child-0'] = ExportError('API unavailable')
    with pytest.raises(ExportError):
        await export(api, tmp_path / 'results.jsonl', allow_partial=True)
    assert not (tmp_path / 'results.jsonl').exists()


@pytest.mark.parametrize(('field', 'value'), [('generation_status', 'pending'), ('submission_closed', False)])
async def test_partial_batch_exports_while_generation_or_submission_is_open(tmp_path, field, value):
    class OpenApi(PartialApi):
        async def get_batch_summary(self, execution_id):
            return {**await super().get_batch_summary(execution_id), field: value}

    api = OpenApi(('succeeded', 'running', 'failed', 'cancelled', 'lost', 'expired', 'killed'))
    api.outputs = {
        f'child-{i}': {'success': item['status'] == 'succeeded', 'task_output': {'value': i}}
        for i, item in enumerate(api.items)
    }
    rows = await export(api, tmp_path / 'results.jsonl', allow_partial=True, result_status='all')
    assert [row['run_id'] for row in rows] == ['item-000', 'item-002', 'item-003', 'item-004', 'item-005', 'item-006']


@pytest.mark.parametrize('spec_type', ['nexus_single', 'nexus_live'])
async def test_partial_export_rejects_non_batch_targets(tmp_path, spec_type):
    class SingleApi(Api):
        async def get_execution(self, execution_id):
            return {**await super().get_execution(execution_id), 'spec_type': spec_type}

    with pytest.raises(ExportError, match='--allow-partial is supported only for Batch results'):
        await export(SingleApi(), tmp_path / 'results.jsonl', allow_partial=True)
    assert not (tmp_path / 'results.jsonl').exists()


async def test_partial_ignores_advisory_counts_unknown_states_and_parent_retry(tmp_path):
    class ChangingApi(PartialApi):
        async def get_batch_summary(self, execution_id):
            summary = await super().get_batch_summary(execution_id)
            return {**summary, 'terminal_items': 99, 'materialized_count': 1}

        async def get_execution(self, execution_id):
            execution = await super().get_execution(execution_id)
            if execution_id == 'batch' and self.result_calls:
                execution.update(attempt_count=2, current_attempt_id='next-attempt')
            return execution

    api = ChangingApi(('succeeded', 'future_nonterminal_state'))
    rows = await export(api, tmp_path / 'results.jsonl', allow_partial=True)
    assert [row['run_id'] for row in rows] == ['item-000']
    assert api.result_calls == ['child-0']


async def test_partial_batch_allows_generation_and_counts_to_advance(tmp_path, caplog):
    class GrowingApi(PartialApi):
        async def get_batch_summary(self, execution_id):
            summary = await super().get_batch_summary(execution_id)
            if not self.result_calls:
                return {**summary, 'generation_status': 'pending', 'submission_closed': False, 'total_items': 100}
            return {**summary, 'total_items': 200, 'materialized_count': 150, 'summary_version': 2}

        async def get_batch_items(self, execution_id, cursor):
            if cursor is not None and len(self.items) == 2:
                self.items.append({
                    **self.items[1],
                    'item_id': 'item-002',
                    'status': 'cancelled',
                    'current_child_execution_id': None,
                })
            return await super().get_batch_items(execution_id, cursor)

    api = GrowingApi(('succeeded', 'running'))
    rows = await export(api, tmp_path / 'results.jsonl', allow_partial=True, result_status='all')
    assert [row['run_id'] for row in rows] == ['item-000']
    assert len(api.items) == 3
    assert 'read 2 terminal items; skipped 1 non-terminal items' in caplog.text


class ResultReader:
    def __init__(self, outputs):
        self.outputs = copy.deepcopy(outputs)
        self.calls = []

    async def read_many(self, ids, **kwargs):
        self.calls.extend(ids)
        return {child_id: TraceResult(output=self.outputs.get(child_id)) for child_id in ids}


@pytest.mark.parametrize('source', ['api', 'database', 'fallback'])
@pytest.mark.parametrize(
    ('selection', 'selected', 'expected_statuses'),
    [
        (None, [1], ['success']),
        ('success', [1], ['success']),
        ('failed', [2, 3, 4, 5, 6], ['failed', 'failed', 'timeout', 'cancelled', 'cancelled']),
        ('all', [1, 2, 3, 4, 5, 6], ['success', 'failed', 'failed', 'timeout', 'cancelled', 'cancelled']),
    ],
)
async def test_partial_selection_is_identical_for_api_database_and_fallback(
    tmp_path, source, selection, selected, expected_statuses
):
    api = PartialApi((
        'pending',
        'succeeded',
        'failed',
        'lost',
        'expired',
        'killed',
        'cancelled',
        'running',
        'submitted',
    ))
    api.outputs = {
        f'child-{i}': {
            'success': True,
            'task_output': {'value': i},
            'agent_output': {'main_trajectory': [{'role': 'assistant', 'content': f'saved-{i}'}]},
            'processor_results': {'score': 0},
        }
        for i in range(len(api.items))
    }
    reader = None if source == 'api' else ResultReader(api.outputs if source == 'database' else {})
    rows = await export(api, tmp_path / 'results.jsonl', reader, allow_partial=True, result_status=selection)
    child_ids = [f'child-{i}' for i in selected]
    assert [row['run_id'] for row in rows] == [f'item-{i:03d}' for i in selected]
    assert [row['status'] for row in rows] == expected_statuses
    assert Counter(api.event_calls) == dict.fromkeys(child_ids, 1)
    assert Counter(api.result_calls) == ({} if source == 'database' else dict.fromkeys(child_ids, 1))
    if reader is not None:
        assert Counter(reader.calls) == dict.fromkeys(child_ids, 1)
    for i, row in zip(selected, rows, strict=True):
        child_id = f'child-{i}'
        for field in ('task_output', 'agent_output', 'processor_results'):
            assert row[field] == api.outputs[child_id][field]
        assert row['error'] == api.executions[child_id]['error_json']
        assert row['metadata']['_nexus']['child_execution_id'] == child_id
        assert row['start_time'] == '2026-09-21T01:00:00'
        assert row['end_time'] == '2026-09-21T01:00:01'
    assert not (set(api.execution_calls) - {'batch', *child_ids})


@pytest.mark.parametrize('source', ['api', 'database', 'fallback'])
@pytest.mark.parametrize(
    ('status', 'expected_status'),
    [
        ('failed', 'failed'),
        ('lost', 'failed'),
        ('expired', 'timeout'),
        ('killed', 'cancelled'),
        ('cancelled', 'cancelled'),
    ],
)
@pytest.mark.parametrize('payload', ['missing', 'success_with_trajectory', 'failure_with_error', 'empty_artifacts'])
async def test_failed_selection_filters_empty_results_and_preserves_artifacts_and_error(
    tmp_path, source, status, expected_status, payload
):
    api = Api((status,))
    output = {
        'success': payload == 'success_with_trajectory',
        'error': {'type': 'RunOutputError', 'message': 'saved failure'} if payload == 'failure_with_error' else None,
        'task_output': {},
        'agent_output': {'main_trajectory': [{'role': 'assistant', 'content': 'saved before failure'}]},
        'processor_results': [],
    }
    if payload == 'empty_artifacts':
        output['agent_output'] = {}
    api.outputs = {} if payload == 'missing' else {'child-0': output}
    reader = None if source == 'api' else ResultReader(api.outputs if source == 'database' else {})
    rows = await export(api, tmp_path / 'failed.jsonl', reader, allow_partial=True, result_status='failed')
    assert api.result_calls == ([] if source == 'database' and payload != 'missing' else ['child-0'])
    assert api.event_calls == ['child-0']
    if reader is not None:
        assert reader.calls == ['child-0']
    if payload in {'missing', 'empty_artifacts'}:
        assert rows == []
        return
    assert len(rows) == 1
    row = rows[0]
    assert row['status'] == expected_status
    assert row['error'] == (output['error'] or api.executions['child-0']['error_json'])
    for field in ('task_output', 'agent_output', 'processor_results'):
        assert row[field] == output[field]


@pytest.mark.parametrize('selection', [None, 'failed', 'all'])
@pytest.mark.parametrize(
    ('task_output', 'agent_output', 'processor_results'),
    [(None, {}, []), ({}, [], ''), ([], '', None), ('', None, {})],
)
async def test_partial_failed_and_all_filter_mixed_empty_artifacts_while_full_preserves_them(
    tmp_path, selection, task_output, agent_output, processor_results
):
    api = Api(('failed',))
    output = {
        'success': False,
        'error': {'type': 'SavedError'},
        'task_output': task_output,
        'agent_output': agent_output,
        'processor_results': processor_results,
    }
    api.outputs = {'child-0': output}
    rows = await export(api, tmp_path / 'results.jsonl', allow_partial=selection is not None, result_status=selection)
    assert api.result_calls == ['child-0']
    if selection in {'failed', 'all'}:
        assert rows == []
    else:
        assert len(rows) == 1
        for field in ('error', 'task_output', 'agent_output', 'processor_results'):
            assert rows[0][field] == output[field]


@pytest.mark.parametrize('field', ['task_output', 'agent_output', 'processor_results'])
@pytest.mark.parametrize(
    'value', [0, False, {'nested': {}}, [None]], ids=['zero', 'false', 'nested-dict', 'nested-list']
)
async def test_failed_selection_preserves_falsey_scalars_and_nonempty_containers(tmp_path, field, value):
    api = Api(('failed',))
    output = {'success': False, 'task_output': None, 'agent_output': {}, 'processor_results': []}
    output[field] = value
    api.outputs = {'child-0': output}
    rows = await export(api, tmp_path / 'results.jsonl', allow_partial=True, result_status='failed')
    assert len(rows) == 1
    assert rows[0]['status'] == 'failed'
    assert rows[0]['error'] == api.executions['child-0']['error_json']
    assert type(rows[0][field]) is type(value)
    for name in ('task_output', 'agent_output', 'processor_results'):
        assert rows[0][name] == output[name]


@pytest.mark.parametrize('source', ['api', 'database', 'fallback'])
@pytest.mark.parametrize('output', [{}, {'success': 'true'}, {'success': 0}])
@pytest.mark.usefixtures('forbid_file_filter')
async def test_failed_selection_validates_empty_run_output_before_filtering(tmp_path, source, output):
    api = Api(('failed',))
    api.outputs = {'child-0': output}
    reader = None if source == 'api' else ResultReader(api.outputs if source == 'database' else {})
    path = tmp_path / 'results.jsonl'
    with pytest.raises(ExportError, match='invalid RunOutput success field'):
        await export(api, path, reader, allow_partial=True, result_status='failed')
    assert not path.exists()
    assert api.result_calls == ([] if source == 'database' else ['child-0'])


@pytest.mark.parametrize('failure', ['events', 'error_fallback'])
@pytest.mark.usefixtures('forbid_file_filter')
async def test_failed_selection_does_not_hide_control_read_errors_for_empty_output(tmp_path, failure):
    class FailingApi(Api):
        async def get_control_events(self, execution_id):
            if failure == 'events':
                raise ExportError('events unavailable')
            return await super().get_control_events(execution_id)

        async def get_execution(self, execution_id):
            if failure == 'error_fallback' and execution_id == 'child-0' and self.result_calls:
                raise ExportError('error_fallback unavailable')
            return await super().get_execution(execution_id)

    api = FailingApi(('failed',))
    api.executions['child-0']['error_json'] = None
    api.outputs = {'child-0': {'success': False, 'task_output': {}, 'agent_output': [], 'processor_results': ''}}
    path = tmp_path / 'results.jsonl'
    with pytest.raises(ExportError, match=f'child child-0: {failure} unavailable'):
        await export(api, path, allow_partial=True, result_status='failed')
    assert not path.exists()
    assert api.result_calls == ([] if failure == 'events' else ['child-0'])


@pytest.mark.parametrize('source', ['api', 'database', 'fallback'])
@pytest.mark.parametrize(
    ('output', 'message'),
    [
        (None, 'successful execution has no RunOutput'),
        ({'success': False}, 'successful execution has failed RunOutput'),
        ({'success': 'true'}, 'invalid RunOutput success field'),
        ({}, 'invalid RunOutput success field'),
    ],
)
async def test_selected_success_requires_valid_successful_output(tmp_path, source, output, message):
    api = Api(('succeeded',))
    api.outputs = {'child-0': output}
    reader = None if source == 'api' else ResultReader(api.outputs if source == 'database' else {})
    path = tmp_path / 'success.jsonl'
    with pytest.raises(ExportError, match=message):
        await export(api, path, reader, allow_partial=True)
    assert not path.exists()


@pytest.mark.parametrize('selection', [None, 'success', 'failed', 'all'])
async def test_empty_partial_selection_never_probes_database_or_reads_results(tmp_path, selection):
    statuses = {
        None: ('failed', 'lost', 'expired', 'killed', 'cancelled'),
        'success': ('failed', 'running'),
        'failed': ('succeeded', 'running'),
        'all': ('running', 'pending'),
    }
    api = PartialApi(statuses[selection])

    class NoDatabase:
        def __getattr__(self, name):
            pytest.fail(f'empty selection must not access database {name}')

    path = tmp_path / 'empty.jsonl'
    assert await export(api, path, NoDatabase(), allow_partial=True, result_status=selection) == []
    assert path.read_bytes() == b''
    assert api.execution_calls == {'batch': 2}
    assert api.event_calls == []
    assert api.result_calls == []


@pytest.mark.parametrize('selection', ['success', 'failed'])
@pytest.mark.parametrize(
    ('override', 'message'),
    [
        ({'execution_id': 'another-child'}, 'identity mismatch'),
        ({'spec_type': 'unknown'}, 'unsupported Execution spec_type'),
        ({'spec_type': 'nexus_batch'}, 'Single execution'),
        ({'attempt_count': -1}, 'invalid Execution attempt_count'),
        ({'attempt_count': True}, 'invalid Execution attempt_count'),
        ({'attempt_count': None}, 'invalid Execution attempt_count'),
        ({'status': 'running'}, 'not terminal'),
        ({'parent_execution_id': 'another-batch'}, 'another Batch'),
    ],
)
async def test_selection_checks_every_child_control_identity_before_output(tmp_path, selection, override, message):
    api = Api(('succeeded' if selection == 'success' else 'failed',))
    api.executions['child-0'].update(override)
    path = tmp_path / 'uncreated' / 'results.jsonl'

    class NoDatabase:
        def __getattr__(self, name):
            pytest.fail(f'invalid child must not access database {name}')

    with pytest.raises(ExportError, match=message):
        await export(api, path, NoDatabase(), allow_partial=True, result_status=selection)
    assert not path.parent.exists()
    assert api.result_calls == []
    assert api.event_calls == []


@pytest.mark.parametrize(
    ('selection', 'item_status', 'child_status'),
    [('success', 'succeeded', 'failed'), ('failed', 'failed', 'succeeded')],
)
async def test_selection_rejects_control_status_in_a_different_category(tmp_path, selection, item_status, child_status):
    api = Api((item_status,))
    api.executions['child-0']['status'] = child_status
    path = tmp_path / 'uncreated' / 'results.jsonl'
    with pytest.raises(ExportError, match='does not match.*result-status'):
        await export(api, path, allow_partial=True, result_status=selection)
    assert not path.parent.exists()
    assert api.result_calls == []


@pytest.mark.parametrize('child_status', ['failed', 'lost', 'expired', 'killed'])
async def test_failed_selection_accepts_same_category_without_overwriting_item_status(tmp_path, child_status):
    api = Api(('expired',))
    api.executions['child-0']['status'] = child_status
    rows = await export(api, tmp_path / 'results.jsonl', allow_partial=True, result_status='failed')
    assert rows[0]['status'] == 'timeout'
    assert rows[0]['task_output'] == {'value': 'saved'}


@pytest.mark.parametrize('allow_partial', [False, True])
@pytest.mark.parametrize(
    ('status', 'output', 'needs_control'),
    [
        ('succeeded', {'success': True}, False),
        ('failed', {'success': False, 'error': {'type': 'SavedError'}}, False),
        ('failed', {'success': False, 'error': {}}, False),
        ('failed', {'success': True, 'error': None}, True),
        ('failed', {'success': False}, True),
        ('failed', None, True),
    ],
)
async def test_all_and_full_only_fetch_child_control_when_error_fallback_is_needed(
    tmp_path, allow_partial, status, output, needs_control
):
    class ControlApi(Api):
        async def get_execution(self, execution_id):
            if execution_id != 'batch':
                assert needs_control
                # all/full retain the existing error-only lookup, without selection validation.
                self.execution_calls[execution_id] += 1
                return {'parent_execution_id': 'batch', 'error_json': {'type': 'ControlFailure'}}
            return await super().get_execution(execution_id)

    api = ControlApi((status,))
    api.outputs = {'child-0': output}
    rows = await export(
        api, tmp_path / 'results.jsonl', allow_partial=allow_partial, result_status='all' if allow_partial else None
    )
    assert api.execution_calls['child-0'] == int(needs_control)
    if allow_partial and status != 'succeeded':
        assert rows == []
    else:
        assert rows[0]['error'] == ({'type': 'ControlFailure'} if needs_control else output.get('error'))


@pytest.mark.parametrize('selection', ['success', 'failed'])
@pytest.mark.parametrize(
    'change', ['status', 'attempt_count', 'current_attempt_id', 'result_artifact_id', 'error_json']
)
async def test_selected_child_snapshot_changes_prevent_publication(tmp_path, selection, change):
    class ChangingApi(Api):
        async def get_result(self, execution_id):
            self.executions[execution_id][change] = {
                'status': 'lost',
                'attempt_count': 2,
                'current_attempt_id': 'new-attempt',
                'result_artifact_id': 'new-artifact',
                'error_json': {'type': 'ChangedError'},
            }[change]
            return await super().get_result(execution_id)

    api = ChangingApi(('succeeded' if selection == 'success' else 'failed',))
    if selection == 'failed':
        api.outputs.clear()
    path = tmp_path / 'results.jsonl'
    metadata = path.with_suffix('.jsonl.batch_metadata.json')
    path.write_bytes(b'previous output\n')
    metadata.write_bytes(b'previous metadata\n')
    with pytest.raises(ExportError, match='Execution changed during export: child-0'):
        await results.export_execution(
            api=api,
            execution_id='batch',
            output=path,
            force=True,
            allow_partial=True,
            result_status=selection,
        )
    assert path.read_bytes() == b'previous output\n'
    assert metadata.read_bytes() == b'previous metadata\n'
    assert path.with_suffix('.jsonl.partial').exists()
    assert api.result_calls == ['child-0']
    if selection == 'failed':
        assert path.with_suffix('.jsonl.partial').read_bytes() == b''


def test_loads_json_bytes_prefers_orjson_and_falls_back():
    from nexus.distributed.export import client as client_module

    assert client_module._loads_json_bytes(b'{"a": 1}') == {'a': 1}
    # Lone surrogate escapes: orjson rejects them, stdlib tolerates - the
    # fallback keeps parity with the previous aiohttp response.json() behaviour.
    assert client_module._loads_json_bytes(b'{"note": "\\ud800"}') == {'note': '\ud800'}
    with pytest.raises(json.JSONDecodeError):
        client_module._loads_json_bytes(b'{invalid')
    with pytest.raises(UnicodeDecodeError):
        client_module._loads_json_bytes(b'"\xff"')
