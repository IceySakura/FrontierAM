from __future__ import annotations

import json

from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from sqlalchemy import select

from nexus.distributed.analytics.backfill import backfill_page, decode_cursor, encode_cursor
from nexus.distributed.analytics.fields import task_name_from_spec
from nexus.distributed.batch.task_store import BatchTaskRecord, TDSQLBatchTaskStore
from nexus.distributed.persistence.orm import BatchItemRow, ExecutionRow
from nexus.distributed.submission.input import normalize_run_input
from nexus.distributed.utils.identity import canonical_sha256
from tests.unit.distributed.analytics.test_statistics import NOW, database, execution, item, seed


@pytest.mark.asyncio
async def test_preview_resume_and_repeated_apply_preserve_existing_names(database):
    _, factory, _ = database
    async with factory() as session:
        session.add_all([
            execution('a', task_name=None, spec_json={'run_input': {'task_name': 'Exact_task_v2'}}),
            execution('b', task_name='preserved', spec_json={'run_input': {'task_name': 'different'}}),
            execution('c', task_name=None, spec_type='nexus_live', spec_json={'task_name': 'live_task'}),
        ])
        await session.commit()
    preview = await backfill_page(factory, environment_id='one', page_size=1)
    assert preview['would_update'] == 1 and preview['updated'] == 0
    async with factory() as session:
        assert (await session.get(ExecutionRow, 'a')).task_name is None
    applied = await backfill_page(factory, environment_id='one', page_size=1, apply=True)
    assert applied['updated'] == 1
    second = await backfill_page(factory, environment_id='one', page_size=1, apply=True, cursor=applied['cursor'])
    assert second['updated'] == 1
    phase = await backfill_page(factory, environment_id='one', apply=True, cursor=second['cursor'])
    assert phase['phase'] == 'executions'
    done = await backfill_page(factory, environment_id='one', apply=True, cursor=phase['cursor'])
    assert done['complete']
    async with factory() as session:
        assert dict((await session.execute(select(ExecutionRow.execution_id, ExecutionRow.task_name))).all()) == {'a': 'Exact_task_v2', 'b': 'preserved', 'c': 'live_task'}
    assert (await backfill_page(factory, environment_id='one', apply=True))['updated'] == 0


@pytest.mark.asyncio
async def test_batch_recovers_first_attempt_time_final_time_and_names_but_does_not_guess_cancel_time(database):
    _, factory, _ = database
    await seed(factory)
    async with factory() as session:
        rows = (await session.scalars(select(BatchItemRow))).all()
        for row in rows:
            row.task_name = row.first_started_at = row.finished_at = None
        await session.commit()
    report = await backfill_page(factory, environment_id='one', apply=True, cursor=encode_cursor('items', None, 'one', None))
    assert report['updated'] == 1
    assert report['unresolved_names'] == 1 and report['unresolved_times'] == 1
    async with factory() as session:
        saved = await session.get(BatchItemRow, ('batch', 'item'))
        assert saved.task_name == 'batch_task'
        assert saved.first_started_at == NOW
        assert saved.finished_at == NOW + timedelta(hours=1)
        assert (await session.get(BatchItemRow, ('batch', 'cancelled'))).finished_at is None


@pytest.mark.asyncio
async def test_backfill_recovers_local_task_list_only_after_identity_verification(database, tmp_path):
    _, factory, _ = database
    payload = {'version': 2, 'item_id': 'record', 'run_input': normalize_run_input({'task_name': 'registered_task_2'}), 'metadata': {}}
    content = (json.dumps(payload) + '\n').encode()
    path = tmp_path / 'tasks.jsonl'
    path.write_bytes(content)
    async with factory() as session:
        session.add(execution('batch', spec_type='nexus_batch'))
        session.add(item('record', task_name=None, status='pending', first_started_at=None, finished_at=None, task_length=len(content), spec_sha256=canonical_sha256(payload)))
        await session.commit()
    cursor = encode_cursor('items', None, 'one', None)
    report = await backfill_page(factory, cursor=cursor, environment_id='one', apply=True, task_lists={'batch': path})
    assert report['updated'] == 1
    async with factory() as session:
        saved = await session.get(BatchItemRow, ('batch', 'record'))
        assert saved.task_name == 'registered_task_2'
        saved.task_name = None
        await session.commit()
    path.write_bytes(content.replace(b'registered_task_2', b'registered_task_3'))
    tampered = await backfill_page(factory, cursor=cursor, environment_id='one', apply=True, task_lists={'batch': path})
    assert tampered['updated'] == 0 and tampered['unresolved_names'] == 1


def test_backfill_cursor_cannot_be_reused_with_a_different_scope():
    cursor = encode_cursor('items', ['batch', 'one'], 'a', None)
    with pytest.raises(ValueError, match='invalid_statistics_backfill_cursor'):
        decode_cursor(cursor, 'b', None)
    assert task_name_from_spec('nexus_single', {'task_name': 'raw_task'}) == 'raw_task'
    assert task_name_from_spec('nexus_batch', {'task_name': 'orchestrator'}) is None


@pytest.mark.asyncio
async def test_read_only_task_store_never_creates_schema_and_bounds_payload():
    record = BatchTaskRecord.from_identity(parent_execution_id='batch', identity={'version': 2, 'item_id': 'one', 'run_input': {'task_name': 'identity'}, 'metadata': {}})
    store = TDSQLBatchTaskStore(host='localhost', user='test', database='test', read_only=True)
    connection, cursor = MagicMock(), MagicMock()
    connection.cursor.return_value.__aenter__ = AsyncMock(return_value=cursor)
    connection.cursor.return_value.__aexit__ = AsyncMock()
    connection.rollback = AsyncMock()
    cursor.execute = AsyncMock()
    cursor.fetchone = AsyncMock(return_value=(2, record.record_sha256, record.record_data))
    store._connection = connection
    result = await store.read_record(parent_execution_id='batch', item_id='one', max_bytes=4096)
    assert result == record
    assert cursor.execute.await_count == 1
    assert 'OCTET_LENGTH(record_data)' in cursor.execute.await_args.args[0]
    assert cursor.execute.await_args.args[1] == (4096, 'batch', 'one')
    with pytest.raises(ValueError, match='read_only'):
        await store.put_records([record])
    cursor.fetchone.return_value = (2, record.record_sha256, None)
    with pytest.raises(ValueError, match='exceeds_byte_limit'):
        await store.read_record(parent_execution_id='batch', item_id='one', max_bytes=1)
    await store.close()


@pytest.mark.asyncio
async def test_backfill_replaces_estimate_with_recovered_real_finish_time(database):
    _, factory, _ = database
    await seed(factory)
    async with factory() as session:
        saved = await session.get(BatchItemRow, ('batch', 'item'))
        saved.finished_at = NOW + timedelta(days=3)
        saved.finished_at_estimated = True
        await session.commit()
    report = await backfill_page(
        factory, environment_id='one', apply=True, cursor=encode_cursor('items', None, 'one', None)
    )
    assert report['updated'] == 1
    async with factory() as session:
        saved = await session.get(BatchItemRow, ('batch', 'item'))
        assert saved.finished_at == NOW + timedelta(hours=1)
        assert not saved.finished_at_estimated
