from __future__ import annotations

import asyncio

from datetime import datetime, timedelta

import pytest

from pydantic import ValidationError
from sqlalchemy import event, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from nexus.distributed.analytics.query import StatisticsQuery, load_statistics
from nexus.distributed.analytics.reader import BusinessStatisticsReader
from nexus.distributed.config import DistributedConfig
from nexus.distributed.persistence.orm import Base, BatchItemAttemptRow, BatchItemRow, ExecutionRow


NOW = datetime(2026, 9, 14)


def execution(name, *, status='succeeded', task_name='echo_task', **overrides):
    values = dict(
        execution_id=name, shard_id=0, mode='job', spec_type='nexus_single', instance_mode='asyncio',
        spec_json={'run_input': {'task_name': task_name}, 'large': 'x' * 10000}, task_name=task_name,
        status=status, route_status='none', environment_id='one', created_at=NOW, updated_at=NOW,
        next_schedule_at=NOW, started_at=NOW, finished_at=NOW if status in {'succeeded', 'failed'} else None,
    )
    return ExecutionRow(**(values | overrides))


def item(name, *, status='succeeded', **overrides):
    values = dict(
        parent_execution_id='batch', item_id=name, task_ordinal=0, task_offset=0, spec_sha256='test',
        task_name='batch_task', status=status, created_at=NOW, updated_at=NOW,
        first_started_at=NOW, finished_at=NOW + timedelta(hours=1),
    )
    return BatchItemRow(**(values | overrides))


def query(**overrides):
    return StatisticsQuery.model_validate({
        'start_at': '2026-09-14T00:00:00+08:00', 'end_at': '2026-09-15T00:00:00+08:00',
        'environment_id': 'one', **overrides,
    })


@pytest.fixture
async def database(tmp_path):
    config = DistributedConfig(database_url=f'sqlite+aiosqlite:///{tmp_path}/statistics.db')
    engine = create_async_engine(config.read_database_url())
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield config, factory, engine
    await engine.dispose()


async def seed(factory):
    async with factory() as session:
        session.add_all([
            execution('single'), execution('failed', status='failed', finished_at=NOW + timedelta(hours=1)),
            execution('live', status='waiting_input', spec_type='nexus_live', task_name='identity'),
            execution('batch', spec_type='nexus_batch', task_name='orchestrator'),
            execution('attempt-0', parent_execution_id='batch', status='failed', task_name='batch_task'),
            execution('attempt-1', parent_execution_id='batch', task_name='batch_task', finished_at=NOW + timedelta(hours=1)),
            execution('other-environment', environment_id='two'),
            item('item', current_child_execution_id='attempt-1', current_attempt_no=1),
            item('cancelled', task_ordinal=1, status='cancelled', first_started_at=None, finished_at=NOW + timedelta(hours=2)),
            BatchItemAttemptRow(parent_execution_id='batch', item_id='item', attempt_no=0, child_execution_id='attempt-0',
                spec_sha256='test', status='failed', created_at=NOW, updated_at=NOW, started_at=NOW, finished_at=NOW),
            BatchItemAttemptRow(parent_execution_id='batch', item_id='item', attempt_no=1, child_execution_id='attempt-1',
                spec_sha256='test', status='succeeded', created_at=NOW, updated_at=NOW, started_at=NOW, finished_at=NOW + timedelta(hours=1)),
        ])
        await session.commit()


@pytest.mark.asyncio
async def test_logical_counts_retry_deduplication_and_all_breakdowns_agree(database):
    config, factory, engine = database
    await seed(factory)
    statements = []
    event.listen(engine.sync_engine, 'before_cursor_execute', lambda c, cu, sql, p, ctx, m: statements.append(sql))
    async with factory() as session:
        result = await load_statistics(session, query(), config.business_statistics)
    assert result['summary']['total'] == 4
    assert [result['summary'][key] for key in ('succeeded', 'failed', 'cancelled', 'unfinished')] == [2, 1, 1, 0]
    assert result['summary']['success_rate'] == 2 / 3
    assert result['daily'][0]['date'] == '2026-09-14'
    assert len(result['hourly']) == 24
    assert result['daily'][0]['total'] == sum(p['total'] for p in result['hourly']) == 4
    assert result['peaks']['hourly']['count'] == 2
    assert result['peaks']['hourly']['start_at'] == '2026-09-14T01:00:00Z'
    assert {p['task_name']: p['total'] for p in result['task_names']['items']} == {'echo_task': 2, 'batch_task': 2}
    assert result['coverage']['count_complete']
    assert all('spec_json' not in sql and 'result_json' not in sql and 'metadata_json' not in sql for sql in statements)


@pytest.mark.asyncio
@pytest.mark.parametrize('basis,total,unfinished', [('created', 5, 1), ('started', 4, 1), ('finished', 4, 0)])
async def test_time_basis_uses_one_cohort_and_current_outcomes(database, basis, total, unfinished):
    config, factory, _ = database
    await seed(factory)
    async with factory() as session:
        value = await load_statistics(session, query(basis=basis), config.business_statistics)
    assert value['summary']['total'] == total
    assert value['summary']['unfinished'] == unfinished
    assert value['coverage']['missing_time_records'] == 0


@pytest.mark.asyncio
async def test_filters_preserve_case_suffixes_and_unknown_name_is_not_literal_unknown(database):
    config, factory, _ = database
    async with factory() as session:
        session.add_all([execution(str(n), task_name=name) for n, name in enumerate(['foo_task', 'foo', 'Foo_task', 'unknown', None])])
        await session.commit()
    async with factory() as session:
        value = await load_statistics(session, query(), config.business_statistics)
        filtered = await load_statistics(session, query(task_name='foo_task'), config.business_statistics)
    assert value['summary']['total'] == 5
    assert len(value['task_names']['items']) == 5
    assert value['task_names']['unknown']['total'] == 1
    assert value['coverage']['name_coverage_ratio'] == 0.8
    assert filtered['summary']['total'] == 1
    assert filtered['coverage']['excluded_unknown_name_records'] == 1
    assert filtered['data_status'] == 'partial'


@pytest.mark.asyncio
async def test_missing_time_is_scoped_outside_window_and_cannot_be_reported_as_zero(database):
    config, factory, _ = database
    async with factory() as session:
        session.add(execution('legacy', finished_at=None, created_at=NOW - timedelta(days=50)))
        await session.commit()
    async with factory() as session:
        value = await load_statistics(session, query(), config.business_statistics)
    assert value['summary']['total'] == 0
    assert value['summary']['success_rate'] is None
    assert value['data_status'] == 'partial'
    assert value['coverage']['missing_time_records'] == 1
    assert all(p['count_complete'] is False for p in value['hourly'])
    assert value['peaks']['hourly']['start_at'] is None


@pytest.mark.asyncio
async def test_task_name_filters_preserve_trailing_spaces_and_unicode(database):
    config, factory, _ = database
    names = ['任务_task', '任务_task ', '任务_Task']
    async with factory() as session:
        session.add_all([execution(str(index), task_name=name) for index, name in enumerate(names)])
        await session.commit()
    for name in names:
        async with factory() as session:
            value = await load_statistics(session, query(task_name=name), config.business_statistics)
        assert value['summary']['total'] == 1
        assert value['task_names']['items'][0]['task_name'] == name


@pytest.mark.asyncio
async def test_half_open_boundaries_partial_periods_and_empty_window(database):
    config, factory, _ = database
    async with factory() as session:
        session.add_all([execution('left', finished_at=NOW), execution('right', finished_at=NOW + timedelta(hours=1))])
        await session.commit()
    async with factory() as session:
        value = await load_statistics(session, query(start_at='2026-09-14T00:00:00Z', end_at='2026-09-14T01:00:00Z'), config.business_statistics)
        empty = await load_statistics(session, query(start_at='2026-09-14T00:01:00Z', end_at='2026-09-14T00:59:00Z'), config.business_statistics)
    assert value['summary']['total'] == 1
    assert value['hourly'][0]['partial_period'] is False
    assert value['daily'][0]['partial_period'] is True
    assert empty['summary']['total'] == 0
    assert empty['summary']['success_rate'] is None
    assert empty['hourly'][0]['partial_period'] is True
    assert empty['peaks']['hourly'] == {'count': 0, 'start_at': None, 'end_at': None, 'partial_period': False, 'tied_periods': 0}


@pytest.mark.parametrize('overrides', [
    {'start_at': '2026-09-14T00:00:00'}, {'end_at': '2026-11-01T00:00:00Z'},
    {'end_at': '2026-09-01T00:00:00Z'}, {'task_kind': 'unknown'}, {'extra': 'ignored'},
])
def test_query_rejects_ambiguous_or_unbounded_input(overrides):
    with pytest.raises(ValidationError):
        query(**overrides)


@pytest.mark.asyncio
async def test_statistics_cache_is_scoped_and_response_mutation_does_not_leak(database):
    config, factory, _ = database
    await seed(factory)
    reader = BusinessStatisticsReader(config, session_factory=factory)
    try:
        first = await reader.get(query(), principal_id='admin-a')
        first['summary']['total'] = 999
        second = await reader.get(query(), principal_id='admin-a')
        assert second['summary']['total'] == 4 and second['cached']
        assert second['generated_at'] == first['generated_at']
        assert not (await reader.get(query(), principal_id='admin-b'))['cached']
    finally:
        await reader.close()


@pytest.mark.asyncio
async def test_shared_query_survives_one_cancel_and_last_waiter_cancels_sql(database, monkeypatch):
    from nexus.distributed.analytics import reader as module

    config, factory, _ = database
    reader = BusinessStatisticsReader(config, session_factory=factory)
    entered, release, stopped = asyncio.Event(), asyncio.Event(), asyncio.Event()
    calls = []

    async def load(*args):
        calls.append(1)
        entered.set()
        try:
            await release.wait()
            return {'data_status': 'ready'}
        finally:
            stopped.set()

    monkeypatch.setattr(module, 'load_statistics', load)
    try:
        first = asyncio.create_task(reader.get(query(), principal_id='a'))
        second = asyncio.create_task(reader.get(query(), principal_id='a'))
        await entered.wait()
        first.cancel()
        await asyncio.gather(first, return_exceptions=True)
        assert not stopped.is_set()
        release.set()
        await second
        assert len(calls) == 1
        entered.clear(); release.clear(); stopped.clear()
        last = asyncio.create_task(reader.get(query(), principal_id='b'))
        await entered.wait()
        last.cancel()
        await asyncio.gather(last, return_exceptions=True)
        assert stopped.is_set()
        assert reader._slots._value == 1
        assert not reader._pending
    finally:
        await reader.close()


@pytest.mark.asyncio
async def test_success_is_start_evidence_even_when_historical_attempts_are_missing(database):
    config, factory, _ = database
    async with factory() as session:
        session.add_all([
            execution('missing-single-start', started_at=None),
            execution('batch', spec_type='nexus_batch'),
            item('missing-item-start', first_started_at=None, child_execution_status=None),
        ])
        await session.commit()
    async with factory() as session:
        result = await load_statistics(session, query(basis='started'), config.business_statistics)
    assert result['summary']['total'] == 0
    assert result['coverage']['missing_time_records'] == 2
    assert not result['coverage']['count_complete']


@pytest.mark.asyncio
async def test_parent_scope_excludes_orphans_and_covers_missing_times_across_history(database):
    config, factory, _ = database
    async with factory() as session:
        session.add_all([
            execution('batch', spec_type='nexus_batch'),
            execution('another-batch', spec_type='nexus_batch', environment_id='two'),
            item('legacy', first_started_at=None, finished_at=None, created_at=NOW - timedelta(days=90)),
            item('other', parent_execution_id='another-batch', first_started_at=None, finished_at=None),
            item('orphan', parent_execution_id='deleted-parent', first_started_at=None, finished_at=None),
        ])
        await session.commit()
    async with factory() as session:
        own = await load_statistics(session, query(task_kind='batch'), config.business_statistics)
        global_result = await load_statistics(session, query(environment_id=None, task_kind='batch'), config.business_statistics)
        empty = await load_statistics(session, query(environment_id='absent', task_kind='batch'), config.business_statistics)
    assert own['coverage']['missing_time_records'] == 1
    assert global_result['coverage']['missing_time_records'] == 2
    assert empty['coverage']['missing_time_records'] == 0
    assert empty['coverage']['count_complete']


@pytest.mark.asyncio
async def test_estimated_finish_times_count_only_in_the_selected_scope_and_remain_approximate(database):
    config, factory, _ = database
    await seed(factory)
    async with factory() as session:
        saved = await session.get(BatchItemRow, ('batch', 'item'))
        saved.finished_at_estimated = True
        await session.commit()
        approximate = await load_statistics(session, query(task_kind='batch'), config.business_statistics)
        exact = await load_statistics(session, query(task_kind='single'), config.business_statistics)
        created = await load_statistics(session, query(basis='created'), config.business_statistics)
        named = await load_statistics(session, query(task_name='absent'), config.business_statistics)
    assert approximate['summary']['total'] == 2
    assert approximate['coverage']['estimated_time_records'] == 1
    assert approximate['coverage']['estimated_time_source'] == 'updated_at'
    assert not approximate['coverage']['count_complete'] and approximate['data_status'] == 'partial'
    assert all(not bucket['count_complete'] for bucket in approximate['hourly'])
    for result in [exact, created, named]:
        assert result['coverage']['estimated_time_records'] == 0
