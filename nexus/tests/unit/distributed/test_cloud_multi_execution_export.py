from __future__ import annotations

import asyncio
import copy
import json

from collections import Counter

import pytest

from nexus.distributed.export import results, writer
from nexus.distributed.export.types import ExportError, TraceResult
from nexus.distributed.export.writer import AtomicJsonlOutput


class Api:
    def __init__(self):
        self.executions = {
            execution_id: {
                'execution_id': execution_id,
                'spec_type': 'nexus_live' if execution_id == 'exe-b' else 'nexus_single',
                'status': status,
                'attempt_count': 2,
                'current_attempt_id': 'attempt-2',
                'result_artifact_id': f'artifact-{execution_id}',
                'error_json': {'type': 'TraceDrainFailed'} if status != 'succeeded' else None,
                'parent_execution_id': 'running-parent',
                'created_at': '2026-09-29T00:00:01',
                'batch_context': {'item_id': 'same-item', 'attempt_no': attempt_no},
            }
            for attempt_no, (execution_id, status) in enumerate([
                ('exe-z', 'failed'),
                ('exe-a', 'succeeded'),
                ('exe-b', 'lost'),
                ('exe-empty', 'failed'),
            ])
        }
        self.outputs = {
            execution_id: {
                'success': execution_id != 'exe-b',
                'error': {'type': 'TaskFailed'} if execution_id == 'exe-b' else None,
                'task_output': {'execution_id': execution_id},
                'agent_output': {'messages': [{'role': 'assistant', 'content': 'saved answer'}]},
                'processor_results': {'score': 0},
            }
            for execution_id in self.executions
            if execution_id != 'exe-empty'
        }
        self.execution_calls = Counter()
        self.result_calls = []

    async def get_execution(self, execution_id):
        self.execution_calls[execution_id] += 1
        return copy.deepcopy(self.executions[execution_id])

    async def get_result(self, execution_id):
        self.result_calls.append(execution_id)
        value = self.outputs.get(execution_id)
        if isinstance(value, Exception):
            raise value
        return value

    async def get_result_to_file(self, execution_id, fileobj, chunk_size=1 << 20, scan=None):
        # Disk-prefetch companion of get_result; tracked in result_calls so the
        # pipeline-level assertions keep covering both fetch paths.
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
        return [
            {'type': 'execution.status.changed', 'payload': {'to': 'running'}, 'created_at': '2026-09-29T01:00:00'},
            {
                'type': 'execution.status.changed',
                'payload': {'to': self.executions[execution_id]['status']},
                'created_at': '2026-09-29T01:01:00',
            },
        ]


async def test_selected_executions_deduplicate_preserve_order_outputs_and_exact_identity(tmp_path):
    # The head record (exe-z) blocks in get_result until the prefetch slot
    # (exe-a) has started downloading, proving the two slots overlap.
    second_started = asyncio.Event()

    class OutOfOrderApi(Api):
        async def get_result(self, execution_id):
            if execution_id == 'exe-z':
                await asyncio.wait_for(second_started.wait(), 2)
            elif execution_id == 'exe-a':
                second_started.set()
            return await super().get_result(execution_id)

        async def get_result_to_file(self, execution_id, fileobj, chunk_size=1 << 20, scan=None):
            if execution_id == 'exe-a':
                second_started.set()
            return await super().get_result_to_file(execution_id, fileobj, chunk_size)

    api = OutOfOrderApi()
    path = tmp_path / 'selected.jsonl'
    await results.export_execution(
        api=api, execution_id=['exe-z', 'exe-a', 'exe-z', 'exe-b', 'exe-empty'], output=path, force=False
    )
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row['run_id'] for row in rows] == ['exe-z', 'exe-a', 'exe-b', 'exe-empty']
    assert [row['status'] for row in rows] == ['failed', 'success', 'failed', 'failed']
    assert [row['error'] for row in rows] == [
        {'type': 'TraceDrainFailed'},
        None,
        {'type': 'TaskFailed'},
        {'type': 'TraceDrainFailed'},
    ]
    for row in rows:
        expected = api.outputs.get(row['run_id'], {})
        for field in ('task_output', 'agent_output', 'processor_results'):
            assert row[field] == expected.get(field)
        assert row['retry_count'] == 1
        assert row['metadata'] is None
        assert row['start_time'] == '2026-09-29T01:00:00'
        assert row['end_time'] == '2026-09-29T01:01:00'
    assert api.execution_calls == dict.fromkeys(api.executions, 2)
    assert Counter(api.result_calls) == dict.fromkeys(api.executions, 1)
    assert not path.with_name('selected.jsonl.partial').exists()


@pytest.mark.parametrize('execution_ids', [[], [''], ['exe-z', ' '], b'exe-z'])
async def test_invalid_selection_stops_before_api_or_output(tmp_path, execution_ids):
    api = Api()
    path = tmp_path / 'nested' / 'selected.jsonl'
    with pytest.raises(ExportError, match='non-empty execution IDs'):
        await results.export_execution(api=api, execution_id=execution_ids, output=path, force=False)
    assert not api.execution_calls
    assert not path.parent.exists()


@pytest.mark.parametrize(
    ('override', 'message'),
    [
        ({'spec_type': 'unsupported'}, 'unsupported Execution spec_type'),
        ({'status': 'running'}, 'not terminal'),
        ({'execution_id': 'wrong-id'}, 'identity mismatch'),
    ],
)
async def test_invalid_target_prevents_all_result_reads_and_output(tmp_path, override, message):
    api = Api()
    api.executions['exe-a'].update(override)
    path = tmp_path / 'nested' / 'selected.jsonl'
    with pytest.raises(ExportError, match=message):
        await results.export_execution(api=api, execution_id=['exe-z', 'exe-a'], output=path, force=False)
    assert api.result_calls == []
    assert not path.parent.exists()


@pytest.mark.parametrize('failure', ['api', 'missing_output', 'invalid_output', 'changed_execution'])
async def test_failed_export_preserves_published_output_and_metadata(tmp_path, failure):
    class FailingApi(Api):
        async def get_result(self, execution_id):
            if execution_id == 'exe-a':
                if failure == 'api':
                    raise ExportError('API unavailable')
                if failure == 'changed_execution':
                    self.executions['exe-z']['current_attempt_id'] = 'attempt-3'
            return await super().get_result(execution_id)

        async def get_result_to_file(self, execution_id, fileobj, chunk_size=1 << 20, scan=None):
            if execution_id == 'exe-a':
                if failure == 'api':
                    raise ExportError('API unavailable')
                if failure == 'changed_execution':
                    self.executions['exe-z']['current_attempt_id'] = 'attempt-3'
            return await super().get_result_to_file(execution_id, fileobj, chunk_size)

    api = FailingApi()
    if failure == 'missing_output':
        api.outputs.pop('exe-a')
    elif failure == 'invalid_output':
        api.outputs['exe-a']['success'] = False
    path = tmp_path / 'selected.jsonl'
    metadata = path.with_name(f'{path.name}.batch_metadata.json')
    path.write_text('previous output')
    metadata.write_text('previous metadata')
    with pytest.raises(ExportError):
        await results.export_execution(api=api, execution_id=['exe-z', 'exe-a'], output=path, force=True)
    assert path.read_text() == 'previous output'
    assert metadata.read_text() == 'previous metadata'
    assert path.with_name(f'{path.name}.partial').exists()


async def test_force_recovers_interrupted_export_and_removes_stale_batch_metadata(tmp_path):
    path = tmp_path / 'selected.jsonl'
    partial = path.with_name(f'{path.name}.partial')
    metadata = path.with_name(f'{path.name}.batch_metadata.json')
    path.write_text('previous output')
    partial.write_text('interrupted output')
    metadata.write_text('previous metadata')
    with pytest.raises(ExportError, match='already exists'):
        await results.export_execution(api=Api(), execution_id=['exe-z', 'exe-a'], output=path, force=False)
    assert partial.read_text() == 'interrupted output'
    await results.export_execution(api=Api(), execution_id=['exe-z', 'exe-a'], output=path, force=True)
    assert len(path.read_text().splitlines()) == 2
    assert not partial.exists()
    assert not metadata.exists()


@pytest.mark.parametrize('stop', ['cancel', 'api_error'])
async def test_abort_cancels_inflight_requests_without_reading_the_rest(tmp_path, stop):
    # Both slots must be in flight (head in RAM + prefetch on disk) before the
    # abort fires, proving cancel reaches every outstanding fetch.
    active = set()
    started = asyncio.Event()

    class BlockingApi(Api):
        def _track(self, execution_id):
            self.result_calls.append(execution_id)
            active.add(execution_id)
            if len(active) == 2:
                started.set()

        async def get_result(self, execution_id):
            self._track(execution_id)
            try:
                await started.wait()
                if stop == 'api_error' and execution_id == 'exe-z':
                    raise ExportError('API unavailable')
                await asyncio.Event().wait()
            finally:
                active.remove(execution_id)

        async def get_result_to_file(self, execution_id, fileobj, chunk_size=1 << 20, scan=None):
            self._track(execution_id)
            try:
                await started.wait()
                if stop == 'api_error' and execution_id == 'exe-z':
                    raise ExportError('API unavailable')
                value = self.outputs.get(execution_id)
                if value is None:
                    return None
                data = json.dumps(value).encode()
                fileobj.write(data)
                if scan is not None:
                    scan.feed(data)
                return len(data)
            finally:
                active.remove(execution_id)

    api = BlockingApi()
    path = tmp_path / 'selected.jsonl'
    task = asyncio.create_task(
        results.export_execution(api=api, execution_id=list(api.executions), output=path, force=False)
    )
    try:
        await asyncio.wait_for(started.wait(), 2)
        if stop == 'cancel':
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            with pytest.raises(ExportError, match='execution exe-z: API unavailable'):
                await asyncio.wait_for(task, 2)
        assert active == set()
        # Prefetch depth follows the configured concurrency; the observed reads
        # must be an order-preserving prefix bounded by the in-flight window,
        # and cancellation must stop further reads entirely (race-free: how many
        # fetches managed to start before the cancel is scheduling-dependent).
        lookahead = results._result_concurrency() - 1
        assert 1 <= len(api.result_calls) <= 1 + lookahead
        assert api.result_calls == list(api.executions)[: len(api.result_calls)]
        snapshot = list(api.result_calls)
        await asyncio.sleep(0.05)
        assert api.result_calls == snapshot
        assert not path.exists()
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


class MixedApi(Api):
    def __init__(self):
        super().__init__()
        # Batch children are created from a Single template. Keep the mixed
        # fixture representative; Live's own partition resolution is covered by
        # the Single/Live tests below.
        self.executions['exe-b']['spec_type'] = 'nexus_single'
        self.items = {
            'batch-a': [
                self.item('item-001', 'exe-a', 'succeeded', 'shared-run'),
                self.item('item-002', 'exe-z', 'failed', 'a-failed'),
                self.item('item-003', None, 'cancelled', 'a-cancelled'),
            ],
            'batch-b': [
                self.item('item-001', 'exe-b', 'lost', 'shared-run'),
                self.item('item-002', 'exe-empty', 'failed', 'b-empty'),
            ],
        }
        self.metadata = {'batch-a': {'dataset': 'a'}, 'batch-b': {'dataset': 'b'}}
        self.summary_version = 1
        for day, parent_id in enumerate(self.items, start=28):
            self.executions[parent_id] = {
                'execution_id': parent_id,
                'spec_type': 'nexus_batch',
                'status': 'failed',
                'attempt_count': 1,
                'current_attempt_id': f'attempt-{parent_id}',
                'created_at': f'2026-09-{day}T00:00:00',
            }
            for item in self.items[parent_id]:
                if child_id := item['current_child_execution_id']:
                    self.executions[child_id].update(
                        parent_execution_id=parent_id,
                        batch_context={'item_id': item['item_id'], 'attempt_no': item['current_attempt_no']},
                        status='killed' if item['status'] == 'cancelled' else item['status'],
                    )

    @staticmethod
    def item(item_id, child_id, status, run_id):
        return {
            'item_id': item_id,
            'current_child_execution_id': child_id,
            'current_attempt_no': 0,
            'status': status,
            'metadata': {'_nexus': {'original_run_id': run_id}},
        }

    async def get_batch_summary(self, execution_id):
        items = self.items[execution_id]
        return {
            'generation_status': 'complete',
            'submission_closed': True,
            'summary_version': self.summary_version,
            'materialized_count': len(items),
            'total_items': len(items),
            'terminal_items': sum(item['status'] in results._TERMINAL_ITEM_STATUSES for item in items),
            'item_status_counts': dict(Counter(item['status'] for item in items)),
            'task_list_manifest': {'batch_metadata': copy.deepcopy(self.metadata[execution_id])},
        }

    async def get_batch_items(self, execution_id, cursor):
        items = [item for item in self.items[execution_id] if cursor is None or item['item_id'] > cursor]
        page = copy.deepcopy(items[:2])
        return {'items': page, 'next_cursor': page[-1]['item_id'] if len(items) > 2 else None}


@pytest.mark.parametrize(
    'selection',
    [
        ['batch-a'],
        ['batch-a', 'batch-a'],
        ['batch-a', 'exe-a'],
        ['exe-a', 'batch-a'],
        ['exe-z', 'batch-a', 'exe-a', 'batch-a'],
    ],
)
async def test_redundant_children_do_not_change_batch_bytes_or_metadata(tmp_path, selection):
    reference = tmp_path / 'reference.jsonl'
    candidate = tmp_path / 'candidate.jsonl'
    await results.export_execution(api=MixedApi(), execution_id='batch-a', output=reference, force=False)
    api = MixedApi()
    await results.export_execution(api=api, execution_id=selection, output=candidate, force=False)
    assert candidate.read_bytes() == reference.read_bytes()
    assert (
        candidate.with_suffix('.jsonl.batch_metadata.json').read_bytes()
        == reference.with_suffix('.jsonl.batch_metadata.json').read_bytes()
    )
    assert Counter(api.result_calls) == {'exe-a': 1, 'exe-z': 1}
    rows = [json.loads(line) for line in candidate.read_text().splitlines()]
    assert [row['run_id'] for row in rows] == ['shared-run', 'a-failed', 'a-cancelled']
    assert rows[-1]['error']['type'] == 'BatchItemNoChildExecution'


@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('selection', [None, 'success', 'failed', 'all'])
async def test_multiple_batches_equal_concatenated_batch_exports(tmp_path, reverse, selection):
    batch_ids = ['batch-b', 'batch-a'] if reverse else ['batch-a', 'batch-b']
    options = {'allow_partial': selection is not None, 'result_status': selection}
    expected = b''
    for batch_id in batch_ids:
        path = tmp_path / f'{batch_id}.jsonl'
        await results.export_execution(api=MixedApi(), execution_id=batch_id, output=path, force=False, **options)
        expected += path.read_bytes()
    api = MixedApi()
    candidate = tmp_path / 'combined.jsonl'
    await results.export_execution(api=api, execution_id=batch_ids, output=candidate, force=False, **options)
    assert candidate.read_bytes() == expected
    expected_ids = {
        None: ['exe-a', 'exe-z', 'exe-b', 'exe-empty'],
        'all': ['exe-a', 'exe-z', 'exe-b', 'exe-empty'],
        'success': ['exe-a'],
        'failed': ['exe-z', 'exe-b', 'exe-empty'],
    }[selection]
    assert Counter(api.result_calls) == dict.fromkeys(expected_ids, 1)
    assert json.loads(candidate.with_suffix('.jsonl.batch_metadata.json').read_text()) == {'batches': api.metadata}
    rows = [json.loads(line) for line in candidate.read_text().splitlines()]
    assert len(rows) == {'success': 1, 'failed': 2, 'all': 3, None: 5}[selection]
    assert sum(row['run_id'] == 'shared-run' for row in rows) == (2 if selection in {None, 'all'} else 1)


@pytest.mark.parametrize('selection', ['batch-a', ['batch-a', 'batch-b'], ['exe-z', 'batch-b', 'batch-a', 'exe-a']])
async def test_partial_all_preserves_retained_full_result_bytes_and_metadata(tmp_path, selection):
    full = tmp_path / 'full.jsonl'
    partial = tmp_path / 'partial-all.jsonl'
    await results.export_execution(api=MixedApi(), execution_id=selection, output=full, force=False)
    await results.export_execution(
        api=MixedApi(),
        execution_id=selection,
        output=partial,
        force=False,
        allow_partial=True,
        result_status='all',
    )
    full_lines = full.read_bytes().splitlines(keepends=True)
    placeholders = {'a-cancelled', 'b-empty'}
    assert any(json.loads(line)['run_id'] in placeholders for line in full_lines)
    assert partial.read_bytes() == b''.join(
        line for line in full_lines if json.loads(line)['run_id'] not in placeholders
    )
    assert (
        partial.with_suffix('.jsonl.batch_metadata.json').read_bytes()
        == full.with_suffix('.jsonl.batch_metadata.json').read_bytes()
    )


@pytest.mark.parametrize('mode', ['strict', 'index-dropped'])
async def test_partial_export_validation_paths_publish_identical_bytes(tmp_path, monkeypatch, mode):
    async def export(path, **kwargs):
        await results.export_execution(
            api=MixedApi(),
            execution_id=['batch-a', 'batch-b'],
            output=path,
            force=False,
            allow_partial=True,
            result_status='all',
            **kwargs,
        )

    default = tmp_path / 'default.jsonl'
    await export(default)
    other = tmp_path / f'{mode}.jsonl'
    if mode == 'strict':
        await export(other, strict_validate=True)
    else:
        # Exceeding the cap drops the write-time index; filtering must fall back
        # to the parse-per-line path instead of failing.
        monkeypatch.setattr(writer, '_MAX_INDEX_ENTRIES', 0)
        await export(other)
    assert other.read_bytes() == default.read_bytes()


async def test_mixed_export_appends_explicit_historical_execution_without_changing_batch(tmp_path):
    api = MixedApi()
    api.executions['exe-old'] = {**api.executions['exe-a'], 'execution_id': 'exe-old', 'status': 'failed'}
    api.outputs['exe-old'] = {'success': False, 'agent_output': {'history': 'saved'}}
    path = tmp_path / 'mixed.jsonl'
    await results.export_execution(
        api=api, execution_id=['exe-old', 'batch-a', 'exe-old', 'exe-a'], output=path, force=False
    )
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row['run_id'] for row in rows] == ['shared-run', 'a-failed', 'a-cancelled', 'exe-old']
    assert rows[-1]['status'] == 'failed'
    assert rows[-1]['agent_output'] == {'history': 'saved'}
    assert rows[-1]['metadata'] is None


@pytest.mark.parametrize('allow_partial', [False, True])
async def test_multi_batch_partial_filters_failure_placeholders_and_skips_running(tmp_path, allow_partial):
    api = MixedApi()
    api.executions['batch-a']['status'] = 'stopping'
    api.items['batch-a'].append(api.item('item-004', 'exe-running', 'running', 'running'))
    path = tmp_path / 'partial.jsonl'
    if not allow_partial:
        with pytest.raises(ExportError, match='not terminal'):
            await results.export_execution(api=api, execution_id=['batch-a', 'batch-b'], output=path, force=False)
        assert not api.result_calls
        assert not path.exists()
        return
    await results.export_execution(
        api=api,
        execution_id=['exe-a', 'batch-a', 'batch-b'],
        output=path,
        force=False,
        allow_partial=True,
        result_status='all',
    )
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 3
    assert [row['status'] for row in rows] == ['success', 'failed', 'failed']
    assert Counter(api.result_calls) == dict.fromkeys(['exe-a', 'exe-z', 'exe-b', 'exe-empty'], 1)


@pytest.mark.parametrize('change', ['metadata', 'summary', 'execution'])
async def test_any_batch_change_prevents_publication(tmp_path, change):
    class ChangingApi(MixedApi):
        def _mutate(self, execution_id):
            if execution_id == 'exe-b':
                if change == 'metadata':
                    self.metadata['batch-a']['dataset'] = 'changed'
                elif change == 'summary':
                    self.summary_version += 1
                else:
                    self.executions['batch-a']['attempt_count'] += 1

        async def get_result(self, execution_id):
            self._mutate(execution_id)
            return await super().get_result(execution_id)

        async def get_result_to_file(self, execution_id, fileobj, chunk_size=1 << 20, scan=None):
            self._mutate(execution_id)
            return await super().get_result_to_file(execution_id, fileobj, chunk_size)

    path = tmp_path / 'output.jsonl'
    metadata = path.with_suffix('.jsonl.batch_metadata.json')
    path.write_text('previous output')
    metadata.write_text('previous metadata')
    with pytest.raises(ExportError, match='changed during export'):
        await results.export_execution(api=ChangingApi(), execution_id=['batch-a', 'batch-b'], output=path, force=True)
    assert path.read_text() == 'previous output'
    assert metadata.read_text() == 'previous metadata'


async def test_database_first_matches_batch_projection_and_preserves_key_order(tmp_path):
    api = MixedApi()
    reads = []
    saved = {
        'success': False,
        'task_output': {'z': 1, 'a': 2},
        'agent_output': {'content': 'saved'},
        'processor_results': {'score': 0},
        'error': {'type': 'TaskFailed'},
    }

    class Reader:
        async def read_many(self, ids, **kwargs):
            reads.extend(ids)
            return {ids[0]: TraceResult(output=saved if ids == ['exe-z'] else None)}

    path = tmp_path / 'database.jsonl'
    await results.export_execution(
        api=api, execution_id=['exe-z', 'batch-a', 'batch-b'], output=path, force=False, trace_reader=Reader()
    )
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    row = next(row for row in rows if row['run_id'] == 'a-failed')
    assert list(row['task_output']) == ['z', 'a']
    assert row['agent_output'] == saved['agent_output']
    assert row['processor_results'] == saved['processor_results']
    assert row['status'] == 'failed'
    assert Counter(reads) == dict.fromkeys(['exe-a', 'exe-z', 'exe-b', 'exe-empty'], 1)
    assert Counter(api.result_calls) == dict.fromkeys(['exe-a', 'exe-b', 'exe-empty'], 1)


@pytest.mark.parametrize(
    ('selection', 'allow_partial', 'result_status'),
    [
        ('batch-a', False, None),
        ('exe-a', False, None),
        (['exe-a', 'batch-a', 'batch-b'], False, None),
        ('batch-a', True, None),
        ('batch-a', True, 'success'),
        ('batch-a', True, 'all'),
        (['exe-a', 'batch-a', 'batch-b'], True, 'failed'),
    ],
)
async def test_every_selected_record_uses_single_pipeline_after_expansion(
    tmp_path, monkeypatch, selection, allow_partial, result_status
):
    api = MixedApi()
    expansion_complete = False
    calls = []
    read_records = []
    plans = []
    phases = []
    unfiltered_lines = []
    should_filter = allow_partial and result_status in {'failed', 'all'}
    expand = results._expand_executions
    read_single = results._single_record

    async def tracked_expand(*args, **kwargs):
        nonlocal expansion_complete
        plan = await expand(*args, **kwargs)
        assert not api.result_calls
        plans.append(plan)
        expansion_complete = True
        return plan

    async def tracked_single(*, api, single, trace_reader, **kwargs):
        assert expansion_complete
        assert isinstance(single, results._SingleExport)
        calls.append(single.run_id)
        record = await read_single(api=api, single=single, trace_reader=trace_reader, **kwargs)
        assert isinstance(record, dict)
        read_records.append(record)
        return record

    class Writer(AtomicJsonlOutput):
        async def close_and_validate(self, *, expected_count, require_sorted, strict=False):
            assert expected_count == len(read_records) == len(plans[0].singles)
            await super().close_and_validate(
                expected_count=expected_count, require_sorted=require_sorted, strict=strict
            )
            unfiltered_lines.extend(self.partial.read_bytes().splitlines(keepends=True))
            assert [json.loads(line) for line in unfiltered_lines] == read_records
            phases.append('validated')

        async def filter_records_by_index(self, keep):
            assert phases == ['validated']
            assert len(read_records) == len(plans[0].singles)
            written = await super().filter_records_by_index(keep)
            phases.append('filtered')
            return written

        async def publish(self):
            assert phases == (['validated', 'filtered'] if should_filter else ['validated'])
            await super().publish()
            phases.append('published')

    class Reader:
        table = 'trajectory_span_events_pr_' + 'a' * 32
        parent_created_at = None

        async def has_partition_column(self):
            assert expansion_complete
            return False

        async def read_many(self, ids, **kwargs):
            assert expansion_complete
            return {ids[0]: TraceResult(output=api.outputs.get(ids[0]))}

    monkeypatch.setattr(results, '_expand_executions', tracked_expand)
    monkeypatch.setattr(results, '_single_record', tracked_single)
    monkeypatch.setattr(results, 'AtomicJsonlOutput', Writer)
    path = tmp_path / 'unified.jsonl'
    await results.export_execution(
        api=api,
        execution_id=selection,
        output=path,
        force=False,
        trace_reader=Reader(),
        allow_partial=allow_partial,
        result_status=result_status,
    )
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert calls == [single.run_id for single in plans[0].singles.values()]
    assert len(read_records) == len(calls)
    filtered = {'a-cancelled', 'b-empty'} if should_filter else set()
    assert [record for record in read_records if record['run_id'] not in filtered] == rows
    assert [row['run_id'] for row in rows] == [run_id for run_id in calls if run_id not in filtered]
    assert path.read_bytes() == b''.join(
        line for line in unfiltered_lines if json.loads(line)['run_id'] not in filtered
    )
    assert phases == (['validated', 'filtered', 'published'] if should_filter else ['validated', 'published'])


async def test_file_filter_failure_preserves_complete_partial_and_published_files(tmp_path, monkeypatch):
    api = MixedApi()
    path = tmp_path / 'results.jsonl'
    metadata = path.with_suffix('.jsonl.batch_metadata.json')
    path.write_bytes(b'previous output\n')
    metadata.write_bytes(b'previous metadata\n')
    original = b''
    visited = []
    filter_records_by_index = AtomicJsonlOutput.filter_records_by_index

    async def failing_filter(self, keep):
        nonlocal original
        original = self.partial.read_bytes()
        assert [json.loads(line)['run_id'] for line in original.splitlines()] == ['a-failed', 'a-cancelled']

        def fail_on_second_record(entry):
            visited.append(entry.run_id)
            if len(visited) == 2:
                raise RuntimeError('predicate failed')
            return keep(entry)

        return await filter_records_by_index(self, fail_on_second_record)

    monkeypatch.setattr(AtomicJsonlOutput, 'filter_records_by_index', failing_filter)
    with pytest.raises(RuntimeError, match='predicate failed'):
        await results.export_execution(
            api=api,
            execution_id='batch-a',
            output=path,
            force=True,
            allow_partial=True,
            result_status='failed',
        )
    assert visited == ['a-failed', 'a-cancelled']
    assert api.result_calls == ['exe-z']
    assert path.read_bytes() == b'previous output\n'
    assert metadata.read_bytes() == b'previous metadata\n'
    assert path.with_suffix('.jsonl.partial').read_bytes() == original
    assert not list(tmp_path.glob('*.tmp'))


@pytest.mark.parametrize('failure', ['read_error', 'malformed', 'wrong_id'])
@pytest.mark.parametrize('selection', [None, 'failed'])
async def test_multi_database_read_failures_abort_without_api_fallback(tmp_path, failure, selection):
    api = MixedApi()
    api.outputs.clear()

    class Reader:
        async def read_many(self, ids, **kwargs):
            if failure == 'read_error':
                raise ExportError('database unavailable')
            if failure == 'malformed':
                return {ids[0]: {'output': {}}}
            return {'another-execution': TraceResult()}

    path = tmp_path / 'db.jsonl'
    path.write_text('previous output')
    with pytest.raises(ExportError):
        await results.export_execution(
            api=api,
            execution_id=['batch-b', 'exe-empty'] if selection == 'failed' else ['exe-z', 'exe-a'],
            output=path,
            force=True,
            trace_reader=Reader(),
            allow_partial=selection is not None,
            result_status=selection,
        )
    assert not api.result_calls
    assert path.read_text() == 'previous output'


@pytest.mark.parametrize('partitioned', [False, True])
@pytest.mark.parametrize('batch_ids', [False, True])
async def test_one_database_reader_handles_each_executions_parent_partition(tmp_path, partitioned, batch_ids):
    api = MixedApi()
    calls = []
    probes = []

    class Reader(results.TdsqlTraceReader):
        async def has_partition_column(self):
            probes.append(True)
            return partitioned

        async def read_many(self, ids, **kwargs):
            calls.append((
                ids[0],
                self.parent_created_at,
                {k: v for k, v in kwargs.items() if k not in ('scan_out', 'spill_pool')},
            ))
            return {ids[0]: TraceResult(output=api.outputs.get(ids[0]))}

    reader = Reader(
        host='db.test',
        port=3306,
        user='reader',
        password='unused',
        database='trace',
        table='trajectory_span_events_pr_' + 'a' * 32,
    )
    selection = ['batch-a', 'batch-b', 'exe-z'] if batch_ids else ['exe-z', 'exe-b', 'exe-a', 'exe-empty']
    await results.export_execution(
        api=api, execution_id=selection, output=tmp_path / 'db.jsonl', force=False, trace_reader=reader
    )
    assert probes
    assert reader.parent_created_at is None
    assert len(calls) == 4
    for child_id, partition, kwargs in calls:
        parent_id = api.executions[child_id]['parent_execution_id']
        expected = api.executions[parent_id]['created_at'].replace('T', ' ') + '.000000'
        assert partition == (expected if partitioned else None)
        assert kwargs == {'parent_execution_id': parent_id, 'include_timestamps': False}
    assert api.result_calls == ['exe-empty']
    if partitioned:
        # Selecting the Batch adds its own expansion lookup; the child lookups
        # (one per explicit parent, deduplicated by the partition cache) are
        # unchanged.
        expected_parent_calls = 2 if batch_ids else 1
        assert api.execution_calls['batch-a'] == expected_parent_calls
        assert api.execution_calls['batch-b'] == expected_parent_calls


@pytest.mark.parametrize('spec_type', ['nexus_single', 'nexus_live'])
@pytest.mark.parametrize('has_parent', [False, True])
@pytest.mark.parametrize('probe_error', [OSError, TimeoutError])
async def test_single_and_live_export_without_partition_pruning_when_probe_fails(
    tmp_path, spec_type, has_parent, probe_error
):
    api = Api()
    api.executions['exe-a'].update(spec_type=spec_type, parent_execution_id='running-parent' if has_parent else None)
    calls = []

    class Reader(results.TdsqlTraceReader):
        async def has_partition_column(self):
            calls.append('probe')
            raise probe_error('schema probe unavailable')

        async def read_many(self, ids, **kwargs):
            calls.append((
                ids,
                self.parent_created_at,
                {k: v for k, v in kwargs.items() if k not in ('scan_out', 'spill_pool')},
            ))
            return {ids[0]: TraceResult(output=api.outputs[ids[0]])}

    reader = Reader(
        host='db.test',
        port=3306,
        user='reader',
        password='unused',
        database='trace',
        table='trajectory_span_events_pr_' + 'a' * 32,
    )
    path = tmp_path / 'db.jsonl'
    await results.export_execution(api=api, execution_id='exe-a', output=path, force=False, trace_reader=reader)

    assert calls == [
        'probe',
        (
            ['exe-a'],
            None,
            {'parent_execution_id': 'running-parent' if has_parent else None, 'include_timestamps': False},
        ),
    ]
    assert reader.parent_created_at is None
    assert api.execution_calls == {'exe-a': 2}
    assert not api.result_calls
    record = json.loads(path.read_text())
    assert record['run_id'] == 'exe-a'
    assert record['task_output'] == api.outputs['exe-a']['task_output']
    assert not path.with_suffix('.jsonl.partial').exists()


@pytest.mark.parametrize('explicit_partition', [False, True])
async def test_top_level_single_database_partition_uses_its_creation_time(tmp_path, explicit_partition):
    api = Api()
    api.executions['exe-a'].update(parent_execution_id=None, created_at='2026-09-29T01:00:00')
    reads = []

    class Reader(results.TdsqlTraceReader):
        async def has_partition_column(self):
            # An explicit key is validated, not trusted: the probe still runs
            # once so a classic table is reported instead of failing with 1054.
            return True

        async def read_many(self, ids, **kwargs):
            reads.append((
                self.parent_created_at,
                {k: v for k, v in kwargs.items() if k not in ('scan_out', 'spill_pool')},
            ))
            return {ids[0]: TraceResult(output=api.outputs[ids[0]])}

    reader = Reader(
        host='db.test',
        port=3306,
        user='reader',
        password='unused',
        database='trace',
        table='trajectory_span_events_pr_' + 'a' * 32,
        parent_created_at='2026-09-29T01:00:00' if explicit_partition else None,
    )
    await results.export_execution(
        api=api, execution_id='exe-a', output=tmp_path / 'db.jsonl', force=False, trace_reader=reader
    )
    assert reads == [('2026-09-29 01:00:00.000000', {'parent_execution_id': None, 'include_timestamps': False})]
    assert not api.result_calls


@pytest.mark.parametrize('selection', [None, 'success', 'failed', 'all'])
async def test_batches_expand_first_and_filtered_children_can_be_explicitly_selected(tmp_path, selection):
    api = MixedApi()
    path = tmp_path / 'mixed.jsonl'
    await results.export_execution(
        api=api,
        execution_id=['exe-b', 'exe-z', 'batch-a', 'exe-a', 'batch-b', 'exe-z'],
        output=path,
        force=False,
        allow_partial=True,
        result_status=selection,
    )
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    expected = {
        None: ['shared-run', 'exe-b', 'exe-z'],
        'success': ['shared-run', 'exe-b', 'exe-z'],
        'failed': ['a-failed', 'shared-run', 'exe-a'],
        'all': ['shared-run', 'a-failed', 'shared-run'],
    }[selection]
    assert [row['run_id'] for row in rows] == expected
    for row in rows:
        if row['run_id'].startswith('exe-'):
            assert row['metadata'] is None
            assert row['retry_count'] == 1
        else:
            assert row['metadata']['_nexus']['parent_execution_id'] in {'batch-a', 'batch-b'}
            assert row['retry_count'] == 0
    assert Counter(api.result_calls) == dict.fromkeys(
        ['exe-a', 'exe-b', 'exe-z'] if selection in {None, 'success'} else ['exe-a', 'exe-b', 'exe-z', 'exe-empty'], 1
    )
    assert json.loads(path.with_suffix('.jsonl.batch_metadata.json').read_text()) == {'batches': api.metadata}


@pytest.mark.parametrize('spec_type', ['nexus_single', 'nexus_live'])
@pytest.mark.parametrize(
    ('allow_partial', 'selection'),
    [(False, None), (True, None), (True, 'success'), (True, 'failed'), (True, 'all')],
)
async def test_partial_failed_and_all_filter_explicit_empty_failures_and_keep_explicit_success(
    tmp_path, spec_type, allow_partial, selection
):
    api = MixedApi()
    for execution_id, status in [('exe-failed', 'failed'), ('exe-success', 'succeeded')]:
        api.executions[execution_id] = {
            **api.executions['exe-a'],
            'execution_id': execution_id,
            'spec_type': spec_type,
            'status': status,
        }
        api.outputs[execution_id] = {
            'success': True,
            'error': {'type': 'SavedError'},
            'task_output': {},
            'agent_output': [],
            'processor_results': '',
        }
    path = tmp_path / 'explicit.jsonl'
    await results.export_execution(
        api=api,
        execution_id=['exe-failed', 'batch-a', 'exe-success', 'exe-failed', 'exe-success'],
        output=path,
        force=False,
        allow_partial=allow_partial,
        result_status=selection,
    )
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    batch_ids = (
        ['a-failed']
        if selection == 'failed'
        else ['shared-run', 'a-failed']
        if selection == 'all'
        else ['shared-run']
        if allow_partial and selection in {None, 'success'}
        else ['shared-run', 'a-failed', 'a-cancelled']
    )
    explicit_ids = ['exe-success'] if selection in {'failed', 'all'} else ['exe-failed', 'exe-success']
    assert [row['run_id'] for row in rows] == [*batch_ids, *explicit_ids]
    for row in rows[-len(explicit_ids) :]:
        assert row['status'] == ('success' if row['run_id'] == 'exe-success' else 'failed')
        assert row['metadata'] is None
        assert row['retry_count'] == 1
        assert row['error'] == {'type': 'SavedError'}
        assert (row['task_output'], row['agent_output'], row['processor_results']) == ({}, [], '')
    assert api.result_calls.count('exe-failed') == 1
    assert api.result_calls.count('exe-success') == 1
    assert api.execution_calls['exe-failed'] == 2
    assert api.execution_calls['exe-success'] == 2
    assert json.loads(path.with_suffix('.jsonl.batch_metadata.json').read_text()) == api.metadata['batch-a']


@pytest.mark.parametrize('change', ['metadata', 'batch_identity', 'explicit_execution', 'child_execution'])
async def test_all_filtered_results_still_validate_snapshots_and_preserve_published_files(
    tmp_path, monkeypatch, change
):
    api = MixedApi()
    api.outputs.clear()
    filter_records_by_index = AtomicJsonlOutput.filter_records_by_index
    filtered = False

    async def filter_then_change(self, keep):
        nonlocal filtered
        assert len(self.partial.read_bytes().splitlines()) == 3
        written = await filter_records_by_index(self, keep)
        assert written == 0
        filtered = True
        if change == 'metadata':
            api.metadata['batch-a']['dataset'] = 'changed'
        elif change == 'batch_identity':
            api.executions['batch-a']['spec_type'] = 'nexus_single'
        else:
            child_id = 'exe-empty' if change == 'explicit_execution' else 'exe-z'
            api.executions[child_id]['current_attempt_id'] = 'new-attempt'
        return written

    monkeypatch.setattr(AtomicJsonlOutput, 'filter_records_by_index', filter_then_change)
    path = tmp_path / 'results.jsonl'
    metadata = path.with_suffix('.jsonl.batch_metadata.json')
    path.write_bytes(b'previous output\n')
    metadata.write_bytes(b'previous metadata\n')
    with pytest.raises(ExportError, match='changed during export'):
        await results.export_execution(
            api=api,
            execution_id=['batch-a', 'exe-empty'],
            output=path,
            force=True,
            allow_partial=True,
            result_status='failed',
        )
    assert Counter(api.result_calls) == {'exe-z': 1, 'exe-empty': 1}
    assert filtered
    assert path.read_bytes() == b'previous output\n'
    assert metadata.read_bytes() == b'previous metadata\n'
    assert path.with_suffix('.jsonl.partial').read_bytes() == b''


@pytest.mark.parametrize('selection', ['success', 'failed'])
async def test_expansion_saves_only_selected_children_for_final_snapshot_validation(selection):
    api = MixedApi()
    plan = await results._expand_executions(api, ['batch-a', 'batch-b'], allow_partial=True, result_status=selection)
    children = ['exe-a'] if selection == 'success' else ['exe-z', 'exe-b', 'exe-empty']
    assert set(plan.snapshots) == {'batch-a', 'batch-b', *children}
    for child_id in children:
        assert plan.snapshots[child_id] == results.execution_snapshot(api.executions[child_id])
    assert not api.result_calls


@pytest.mark.parametrize('selection', ['success', 'failed', 'all'])
async def test_filtering_precedes_child_deduplication_and_first_selected_batch_item_wins(tmp_path, selection):
    api = MixedApi()
    first_status = 'failed' if selection == 'success' else 'succeeded'
    selected_status = 'succeeded' if selection == 'success' else 'failed'
    api.items['batch-a'] = [
        api.item('item-001', 'exe-a', first_status, 'filtered-first'),
        api.item('item-002', 'exe-a', selected_status, 'selected-second'),
        api.item('item-003', 'exe-a', selected_status, 'duplicate-third'),
    ]
    api.executions['exe-a']['status'] = selected_status
    api.outputs['exe-a']['error'] = {'type': 'SavedError'}
    path = tmp_path / 'deduplicated.jsonl'
    await results.export_execution(
        api=api,
        execution_id=['exe-a', 'batch-a'],
        output=path,
        force=False,
        allow_partial=True,
        result_status=selection,
    )
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]['run_id'] == ('filtered-first' if selection == 'all' else 'selected-second')
    assert rows[0]['metadata']['_nexus']['parent_execution_id'] == 'batch-a'
    assert api.result_calls == ['exe-a']


@pytest.mark.parametrize('spec_type', ['nexus_single', 'nexus_live'])
@pytest.mark.parametrize('selection', [None, 'success', 'failed', 'all'])
async def test_explicit_nonterminal_execution_is_rejected_even_if_batch_would_filter_it(tmp_path, spec_type, selection):
    api = MixedApi()
    api.executions['exe-z'].update(spec_type=spec_type, status='running')
    path = tmp_path / 'nested' / 'results.jsonl'
    with pytest.raises(ExportError, match='not terminal'):
        await results.export_execution(
            api=api,
            execution_id=['batch-a', 'exe-z'],
            output=path,
            force=False,
            allow_partial=True,
            result_status=selection,
        )
    assert not path.parent.exists()
    assert api.result_calls == []


@pytest.mark.parametrize('selection', [None, 'success', 'failed', 'all'])
async def test_partial_rejects_a_pure_single_live_collection(tmp_path, selection):
    api = Api()
    path = tmp_path / 'nested' / 'results.jsonl'
    with pytest.raises(ExportError, match='only for Batch results'):
        await results.export_execution(
            api=api,
            execution_id=['exe-a', 'exe-b'],
            output=path,
            force=False,
            allow_partial=True,
            result_status=selection,
        )
    assert not path.parent.exists()
    assert api.result_calls == []


async def test_all_failed_batches_default_to_empty_without_database_access(tmp_path):
    api = MixedApi()
    api.items['batch-a'] = api.items['batch-a'][1:]

    class NoDatabase:
        def __getattr__(self, name):
            pytest.fail(f'empty selection must not access database {name}')

    path = tmp_path / 'empty.jsonl'
    await results.export_execution(
        api=api,
        execution_id=['batch-a', 'batch-b'],
        output=path,
        force=False,
        allow_partial=True,
        trace_reader=NoDatabase(),
    )
    assert path.read_bytes() == b''
    assert api.result_calls == []
    assert api.execution_calls == {'batch-a': 2, 'batch-b': 2}
    assert json.loads(path.with_suffix('.jsonl.batch_metadata.json').read_text()) == {'batches': api.metadata}


@pytest.mark.parametrize('selection', ['success', 'failed'])
async def test_filtered_database_reads_keep_each_selected_parent_partition(tmp_path, selection):
    api = MixedApi()
    calls = []

    class Reader(results.TdsqlTraceReader):
        async def has_partition_column(self):
            return True

        async def read_many(self, ids, **kwargs):
            await asyncio.sleep(0)
            calls.append((ids[0], self.parent_created_at, kwargs['parent_execution_id']))
            return {ids[0]: TraceResult(output=api.outputs.get(ids[0]))}

    reader = Reader(
        host='db.test',
        port=3306,
        user='reader',
        password='unused',
        database='trace',
        table='trajectory_span_events_pr_' + 'a' * 32,
    )
    await results.export_execution(
        api=api,
        execution_id=['exe-b', 'batch-a', 'batch-b'],
        output=tmp_path / 'db.jsonl',
        force=False,
        allow_partial=True,
        result_status=selection,
        trace_reader=reader,
    )
    children = ['exe-a', 'exe-b'] if selection == 'success' else ['exe-z', 'exe-b', 'exe-empty']
    assert Counter(child for child, _, _ in calls) == dict.fromkeys(children, 1)
    for child, partition, parent in calls:
        assert parent == api.executions[child]['parent_execution_id']
        assert partition == api.executions[parent]['created_at'].replace('T', ' ') + '.000000'
    assert reader.parent_created_at is None
    assert api.result_calls == ([] if selection == 'success' else ['exe-empty'])


@pytest.mark.parametrize('selection', [None, 'success', 'failed', 'all'])
async def test_main_extra_survives_batch_selection_and_filtering(tmp_path, selection):
    api = MixedApi()
    extra = {'reference': '原始内容 ${unchanged}', 'nested': {'b': 2, 'a': 1}}
    for output in api.outputs.values():
        output['extra'] = copy.deepcopy(extra)
    api.outputs['exe-empty'] = {'success': False, 'extra': {'reference': 'extra is not an artifact'}}
    path = tmp_path / 'extra.jsonl'
    await results.export_execution(
        api=api,
        execution_id=['batch-a', 'batch-b'],
        output=path,
        force=False,
        allow_partial=selection is not None,
        result_status=selection,
    )
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        if row['run_id'] == 'a-cancelled':
            assert 'extra' not in row
        else:
            child_id = row['metadata']['_nexus']['child_execution_id']
            assert row['extra'] == api.outputs[child_id]['extra']
            assert list(row)[-1] == 'extra'
    assert any(row['run_id'] == 'b-empty' for row in rows) is (selection is None)
