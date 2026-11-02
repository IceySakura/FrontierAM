from __future__ import annotations

import asyncio

from datetime import datetime

import pytest

from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from nexus.distributed.observability.cache import LiveStatusCache
from nexus.distributed.observability.database import queue_committed_metric
from nexus.utils.platform_metrics import metric_scope


@pytest.mark.asyncio
async def test_live_cache_coalesces_and_cancelled_viewer_does_not_cancel_query():
    cache = LiveStatusCache()
    started, release = asyncio.Event(), asyncio.Event()
    calls = 0

    async def load():
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return {'queried_at': datetime(2026, 9, 14), 'data_status': 'ready', 'managers': {'healthy': 3}}

    first = asyncio.create_task(cache.get(('scope',), load))
    await started.wait()
    second = asyncio.create_task(cache.get(('scope',), load))
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    release.set()
    response = await second
    response['managers']['healthy'] = 0
    cached = await cache.get(('scope',), load)
    assert cached['managers']['healthy'] == 3
    assert calls == 1
    await cache.close()


@pytest.mark.asyncio
async def test_failed_refresh_retains_original_sample_and_is_explicitly_stale():
    cache = LiveStatusCache(ttl=0)
    queried = datetime(2026, 9, 14)

    async def good():
        return {'queried_at': queried, 'data_status': 'ready', 'managers': {'healthy': 4}}

    async def failed():
        raise TimeoutError('private SQL details')

    first = await cache.get(('scope',), good)
    stale = await cache.get(('scope',), failed)
    assert stale['queried_at'] == queried
    assert stale['cache_age_seconds'] >= first['cache_age_seconds']
    assert stale['data_status'] == 'stale'
    assert stale['managers']['healthy'] == 4
    assert stale['error_code'] == 'live_query_failed'
    with pytest.raises(TimeoutError):
        await cache.get(('other-scope',), failed)
    await cache.close()


@pytest.mark.asyncio
async def test_transition_metrics_follow_root_commit_and_savepoint_rollback():
    engine = create_async_engine('sqlite+aiosqlite:///:memory:')
    factory = async_sessionmaker(engine)
    events = []

    class Recorder:
        def record(self, name, value=1, attributes=None):
            events.append((name, value, attributes))

    try:
        with metric_scope(Recorder()):
            async with factory() as session:
                await session.execute(text('SELECT 1'))
                queue_committed_metric(session, 'stage_operations', operation='outer')
                nested = await session.begin_nested()
                queue_committed_metric(session, 'stage_operations', operation='rolled_back')
                await nested.rollback()
                assert not events
                async with session.begin_nested():
                    queue_committed_metric(session, 'stage_operations', operation='saved')
                assert not events
                await session.commit()
                assert [labels['operation'] for _, _, labels in events] == ['outer', 'saved']
                await session.execute(text('SELECT 1'))
                queue_committed_metric(session, 'stage_operations', operation='root_rollback')
                await session.rollback()
                await session.commit()
                assert len(events) == 2
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_bulk_stop_counts_committed_terminal_transitions_once():
    from nexus.distributed.persistence.orm import create_schema
    from nexus.distributed.repositories.execution import CreateExecution, ExecutionRepository

    engine = create_async_engine('sqlite+aiosqlite:///:memory:')
    await create_schema(engine)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    events = []
    now = datetime(2026, 9, 14)

    class Recorder:
        def record(self, name, value=1, attributes=None):
            if name == 'platform_terminal_results':
                events.append((value, attributes))

    try:
        with metric_scope(Recorder()):
            async with factory() as session:
                repository = ExecutionRepository(session, shard_count=1)
                for identifier in ('one', 'two'):
                    await repository.create_execution(
                        CreateExecution(
                            execution_id=identifier,
                            mode='job',
                            spec_type='test',
                            instance_mode='asyncio',
                            spec={},
                            environment_id='test-env',
                        ),
                        now=now,
                    )
                await session.commit()
                await repository.request_stop_many(['one', 'two'], now=now)
                await session.rollback()
                assert events == []
                await repository.request_stop_many(['one', 'two'], now=now)
                assert events == []
                await session.commit()
                await repository.request_stop_many(['one', 'two'], now=now)
                await session.commit()
                assert len(events) == 2
                assert all(value == 1 and labels['environment'] == 'test-env' for value, labels in events)
                assert all(labels['category'] == 'cancelled' for _, labels in events)
    finally:
        await engine.dispose()
