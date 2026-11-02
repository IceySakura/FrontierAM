from __future__ import annotations

import json

from collections import Counter

import pytest

from nexus.distributed.export import results
from nexus.distributed.export.types import ExportError, TraceResult
from tests.unit.distributed.test_cloud_multi_execution_export import MixedApi


@pytest.mark.parametrize('item_ids', [[], '', b'item-001', [''], [' '], ['item-001,'], [None], [False]])
async def test_invalid_item_ids_fail_before_control_reads(tmp_path, item_ids):
    api = MixedApi()
    with pytest.raises(ExportError, match='--item-ids must contain non-empty item IDs'):
        await results.export_execution(
            api=api, execution_id='batch-a', item_ids=item_ids, output=tmp_path / 'results.jsonl', force=False
        )
    assert not api.execution_calls


async def test_item_selection_preserves_batch_bytes_metadata_order_and_database_reads(tmp_path):
    reference = tmp_path / 'full.jsonl'
    await results.export_execution(api=MixedApi(), execution_id='batch-a', output=reference, force=False)
    api = MixedApi()
    reads = []

    class Reader:
        async def read_many(self, ids):
            reads.extend(ids)
            return {eid: TraceResult(output=api.outputs[eid]) for eid in ids}

    output = tmp_path / 'selected.jsonl'
    await results.export_execution(
        api=api,
        execution_id='batch-a',
        item_ids=['item-003,item-001', 'item-001'],
        output=output,
        force=False,
        trace_reader=Reader(),
    )
    lines = reference.read_bytes().splitlines(keepends=True)
    assert output.read_bytes() == lines[0] + lines[2]
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert all('item_id' not in row['metadata']['_nexus'] for row in rows)
    assert [row['run_id'] for row in rows] == ['shared-run', 'a-cancelled']
    assert (
        output.with_suffix('.jsonl.batch_metadata.json').read_bytes()
        == reference.with_suffix('.jsonl.batch_metadata.json').read_bytes()
    )
    assert reads == ['exe-a']
    assert api.result_calls == []


@pytest.mark.parametrize(
    ('selection', 'expected', 'reads'),
    [
        (None, ['shared-run'], ['exe-a']),
        ('failed', ['a-failed'], ['exe-z']),
        ('all', ['shared-run', 'a-failed'], ['exe-a', 'exe-z']),
    ],
)
async def test_item_selection_intersects_status_and_artifact_filters(tmp_path, selection, expected, reads):
    api = MixedApi()
    api.executions['batch-a']['status'] = 'running'
    api.items['batch-a'].append(api.item('item-004', None, 'pending', 'pending'))
    # Excluded successful results must not be read even in the all category.
    api.items['batch-a'].append(api.item('item-005', 'unselected', 'succeeded', 'unselected'))
    output = tmp_path / 'results.jsonl'
    await results.export_execution(
        api=api,
        execution_id='batch-a',
        item_ids=['item-004,item-003,item-002,item-001'],
        allow_partial=True,
        result_status=selection,
        output=output,
        force=False,
    )
    assert [json.loads(line)['run_id'] for line in output.read_text().splitlines()] == expected
    assert Counter(api.result_calls) == Counter(reads)


@pytest.mark.parametrize('item_id', ['item-002', 'item-004'])
async def test_existing_filtered_or_pending_item_is_not_reported_missing(tmp_path, item_id):
    api = MixedApi()
    api.executions['batch-a']['status'] = 'running'
    api.items['batch-a'].append(api.item('item-004', None, 'pending', 'pending'))
    output = tmp_path / 'empty.jsonl'
    await results.export_execution(
        api=api, execution_id='batch-a', item_ids=[item_id], allow_partial=True, output=output, force=False
    )
    assert output.read_bytes() == b''
    assert api.result_calls == []


@pytest.mark.parametrize('allow_partial', [False, True])
async def test_unknown_item_fails_before_results_or_output_and_preserves_existing_file(tmp_path, allow_partial):
    api = MixedApi()
    output = tmp_path / 'results.jsonl'
    output.write_bytes(b'previous export\n')

    class NoDatabase:
        def __getattr__(self, name):
            pytest.fail(f'missing item must fail before database access: {name}')

    with pytest.raises(ExportError, match='Item IDs not found in Batch batch-a: missing'):
        await results.export_execution(
            api=api,
            execution_id='batch-a',
            item_ids=['item-001', 'missing'],
            allow_partial=allow_partial,
            trace_reader=NoDatabase(),
            output=output,
            force=True,
        )
    assert api.result_calls == []
    assert output.read_bytes() == b'previous export\n'
    assert not output.with_suffix('.jsonl.partial').exists()


@pytest.mark.parametrize('ids', [['batch-a', 'batch-b'], ['batch-a', 'exe-a']])
async def test_item_selection_rejects_multiple_targets_before_control_reads(tmp_path, ids):
    api = MixedApi()
    with pytest.raises(ExportError, match='exactly one Batch'):
        await results.export_execution(
            api=api, execution_id=ids, item_ids=['item-001'], output=tmp_path / 'results.jsonl', force=False
        )
    assert not api.execution_calls


@pytest.mark.parametrize('execution_id', ['exe-a', 'exe-b'])
async def test_item_selection_requires_batch_instead_of_single_or_live(tmp_path, execution_id):
    api = MixedApi()
    with pytest.raises(ExportError, match='requires a Batch'):
        await results.export_execution(
            api=api, execution_id=execution_id, item_ids=['item-001'], output=tmp_path / 'results.jsonl', force=False
        )
    assert api.result_calls == []


async def test_selected_item_reads_current_retry_and_keeps_batch_identity(tmp_path):
    api = MixedApi()
    item = api.items['batch-a'][0]
    item.update(current_child_execution_id='retry-child', current_attempt_no=2)
    api.executions['retry-child'] = {**api.executions['exe-a'], 'execution_id': 'retry-child'}
    api.outputs['retry-child'] = {'success': True, 'agent_output': {'answer': 'retried'}}
    output = tmp_path / 'retry.jsonl'
    await results.export_execution(api=api, execution_id='batch-a', item_ids=['item-001'], output=output, force=False)
    row = json.loads(output.read_text())
    assert row['run_id'] == 'shared-run'
    assert row['retry_count'] == 2
    assert row['metadata']['_nexus']['child_execution_id'] == 'retry-child'
    assert row['metadata']['_nexus']['original_run_id'] == 'shared-run'
    assert 'item_id' not in row['metadata']['_nexus']
    assert row['agent_output'] == {'answer': 'retried'}
    assert api.result_calls == ['retry-child']


async def test_item_selection_keeps_full_batch_completion_checks(tmp_path):
    api = MixedApi()
    api.items['batch-a'].append(api.item('item-004', None, 'pending', 'pending'))
    with pytest.raises(ExportError, match='Batch is not fully terminal'):
        await results.export_execution(
            api=api, execution_id='batch-a', item_ids=['item-001'], output=tmp_path / 'results.jsonl', force=False
        )
    assert api.result_calls == []


@pytest.mark.parametrize('broken_page', ['duplicate', 'count'])
async def test_item_selection_does_not_hide_invalid_unselected_pages(tmp_path, broken_page):
    class Api(MixedApi):
        async def get_batch_items(self, execution_id, cursor):
            page = await super().get_batch_items(execution_id, cursor)
            if cursor is not None:
                if broken_page == 'duplicate':
                    page['items'][0]['item_id'] = 'item-002'
                else:
                    page['items'] = []
            return page

    api = Api()
    with pytest.raises(ExportError, match='order is not strictly increasing|count mismatch'):
        await results.export_execution(
            api=api, execution_id='batch-a', item_ids=['item-001'], output=tmp_path / 'results.jsonl', force=False
        )
    assert api.result_calls == []
