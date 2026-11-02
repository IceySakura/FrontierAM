import pytest

from nexus.distributed.export import results
from nexus.distributed.export.types import ExportError, resolve_result_status


@pytest.mark.parametrize(
    ('allow_partial', 'requested', 'expected'),
    [
        (False, None, 'all'),
        (True, None, 'success'),
        (True, 'success', 'success'),
        (True, 'failed', 'failed'),
        (True, 'all', 'all'),
    ],
)
def test_result_status_defaults_and_explicit_partial_selection(allow_partial, requested, expected):
    assert resolve_result_status(allow_partial=allow_partial, result_status=requested) == expected


@pytest.mark.parametrize('requested', ['success', 'failed', 'all'])
def test_explicit_result_status_requires_partial(requested):
    with pytest.raises(ExportError, match='--result-status.*--allow-partial'):
        resolve_result_status(allow_partial=False, result_status=requested)


@pytest.mark.parametrize('requested', ['', 'succeeded', 'lost', 'SUCCESS', ' all ', 'unknown', False, 0])
def test_result_status_rejects_unknown_selection_values(requested):
    with pytest.raises(ExportError, match='--result-status'):
        resolve_result_status(allow_partial=True, result_status=requested)


@pytest.mark.parametrize(
    ('status', 'expected'),
    [
        ('succeeded', 'success'),
        ('failed', 'failed'),
        ('lost', 'failed'),
        ('expired', 'timeout'),
        ('killed', 'cancelled'),
        ('cancelled', 'cancelled'),
    ],
)
def test_record_projection_uses_control_status_and_preserves_empty_artifacts(status, expected):
    record = results.make_jsonl_record(
        run_id='run',
        distributed_status=status,
        start_time=None,
        end_time=None,
        retry_count=0,
        fallback_error={'type': 'ControlError'},
        output={'success': True, 'task_output': {}, 'agent_output': [], 'processor_results': ''},
        metadata=None,
    )
    assert record['status'] == expected
    assert record['error'] == (None if status == 'succeeded' else {'type': 'ControlError'})
    assert (record['task_output'], record['agent_output'], record['processor_results']) == ({}, [], '')


@pytest.mark.parametrize('execution_id', ['batch', ['batch-a', 'batch-b']])
@pytest.mark.parametrize(
    ('allow_partial', 'requested'),
    [(False, 'success'), (False, 'failed'), (False, 'all'), (True, ''), (True, 'succeeded')],
)
async def test_invalid_selection_options_fail_before_api_database_or_output(
    tmp_path, execution_id, allow_partial, requested
):
    class NoReads:
        def __getattr__(self, name):
            pytest.fail(f'invalid options must not access {name}')

    path = tmp_path / 'uncreated' / 'results.jsonl'
    with pytest.raises(ExportError, match='--result-status'):
        await results.export_execution(
            api=NoReads(),
            execution_id=execution_id,
            output=path,
            force=False,
            allow_partial=allow_partial,
            result_status=requested,
            trace_reader=NoReads(),
        )
    assert not path.parent.exists()


@pytest.mark.parametrize(
    ('allow_partial', 'requested'),
    [(False, 'success'), (False, 'failed'), (False, 'all'), (True, ''), (True, 'succeeded')],
)
async def test_expansion_also_validates_raw_options_before_reading_control(allow_partial, requested):
    class NoReads:
        def __getattr__(self, name):
            pytest.fail(f'invalid options must not access {name}')

    with pytest.raises(ExportError, match='--result-status'):
        await results._expand_executions(NoReads(), ['batch'], allow_partial=allow_partial, result_status=requested)


@pytest.mark.parametrize('selection', ['success', 'failed', 'all'])
@pytest.mark.parametrize(
    ('status', 'category'),
    [
        ('succeeded', 'success'),
        ('failed', 'failed'),
        ('lost', 'failed'),
        ('expired', 'failed'),
        ('killed', 'failed'),
        ('cancelled', 'failed'),
        ('pending', None),
        ('submitted', None),
        ('running', None),
        ('stopping', None),
        ('future_state', None),
    ],
)
async def test_batch_items_selects_terminal_categories_before_expansion(selection, status, category):
    item = {'item_id': 'item-001', 'status': status, 'current_child_execution_id': 'child'}

    class Api:
        async def get_batch_items(self, parent_id, cursor):
            assert parent_id == 'batch'
            assert cursor is None
            return {'items': [item], 'next_cursor': None}

    selected = [
        item
        async for item in results._batch_items(
            Api(), 'batch', allow_partial=True, expected_count=1, result_status=selection
        )
    ]
    assert selected == ([item] if category is not None and selection in {category, 'all'} else [])


async def test_batch_items_keeps_internal_all_default():
    items = [{'item_id': 'a', 'status': 'succeeded'}, {'item_id': 'b', 'status': 'cancelled'}]

    class Api:
        async def get_batch_items(self, parent_id, cursor):
            return {'items': items, 'next_cursor': None}

    assert [item async for item in results._batch_items(Api(), 'batch', allow_partial=True, expected_count=2)] == items


@pytest.mark.parametrize(
    ('items', 'cursor', 'expected_count', 'message'),
    [
        ([{'item_id': '', 'status': 'failed'}], None, None, 'invalid item_id'),
        (
            [{'item_id': 'b', 'status': 'failed'}, {'item_id': 'a', 'status': 'failed'}],
            None,
            None,
            'order is not strictly increasing',
        ),
        (
            [{'item_id': 'a', 'status': 'failed'}, {'item_id': 'a', 'status': 'failed'}],
            None,
            None,
            'order is not strictly increasing',
        ),
        ([{'item_id': 'a', 'status': 'failed'}], 'wrong-cursor', None, 'cursor did not advance'),
        ([], 'a', None, 'cursor did not advance'),
        ([{'item_id': 'a', 'status': 'failed'}], None, 2, 'count mismatch'),
        (None, None, None, 'invalid Batch items response'),
    ],
)
async def test_filtering_cannot_hide_invalid_pages(items, cursor, expected_count, message):
    class Api:
        async def get_batch_items(self, parent_id, previous):
            return {'items': items, 'next_cursor': cursor}

    with pytest.raises(ExportError, match=message):
        _ = [
            item
            async for item in results._batch_items(
                Api(), 'batch', allow_partial=True, expected_count=expected_count, result_status='success'
            )
        ]
