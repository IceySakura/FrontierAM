from __future__ import annotations

import asyncio
import logging

import pytest

from aiohttp.test_utils import TestClient, TestServer
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from nexus.distributed.config import DistributedConfig
from nexus.distributed.gateways.common.platform_metrics import LIVE_CACHE_KEY
from nexus.distributed.gateways.control.app import create_control_app
from nexus.distributed.persistence.orm import create_schema
from nexus.distributed.repositories.environment import EnvironmentInventoryRepository


@pytest.fixture
async def summary_client(tmp_path, request):
    engine = create_async_engine(f'sqlite+aiosqlite:///{tmp_path}/summary.db')
    await create_schema(engine)
    config = DistributedConfig(status_sampling=getattr(request, 'param', {}))
    app = create_control_app(session_factory=async_sessionmaker(engine, expire_on_commit=False), config=config)
    try:
        async with TestClient(TestServer(app)) as client:
            yield client
    finally:
        await engine.dispose()


async def test_summary_can_exceed_live_budget_and_reuses_cache_across_pollers(summary_client, monkeypatch):
    calls = []

    async def load(_self, *, now, **scope):
        calls.append(scope)
        await asyncio.sleep(0.6)
        return {'data_status': 'ready', 'queried_at': now, 'total': 17}

    monkeypatch.setattr(EnvironmentInventoryRepository, 'manager_summary', load)
    url = '/v1/managers/summary?environment_id=large'
    first, second = await asyncio.gather(summary_client.get(url), summary_client.get(url))
    assert first.status == second.status == 200
    data = await first.json()
    assert (await second.json())['queried_at'] == data['queried_at']
    assert data['total'] == 17
    await asyncio.sleep(1.05)
    cached = await summary_client.get(url)
    assert cached.status == 200
    assert (await cached.json())['queried_at'] == data['queried_at']
    assert len(calls) == 1
    other = await summary_client.get('/v1/managers/summary?environment_id=other')
    assert other.status == 200
    assert [scope['environment_id'] for scope in calls] == ['large', 'other']


@pytest.mark.parametrize(
    'summary_client',
    [{'manager_summary_timeout_seconds': 0.05, 'manager_summary_cache_seconds': 0}],
    indirect=True,
)
@pytest.mark.parametrize('failure', ['timeout', 'error'])
async def test_failed_summary_refresh_retains_sample_and_logs_cause(summary_client, monkeypatch, caplog, failure):
    fail = False
    cancelled = asyncio.Event()

    async def load(_self, *, now, **scope):
        if fail:
            if failure == 'error':
                raise RuntimeError('summary database failed')
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        return {'data_status': 'ready', 'queried_at': now, 'total': 9}

    monkeypatch.setattr(EnvironmentInventoryRepository, 'manager_summary', load)
    url = '/v1/managers/summary?environment_id=large'
    first = await (await summary_client.get(url)).json()
    fail = True
    with caplog.at_level(logging.WARNING, logger='nexus.distributed.observability.cache'):
        response = await summary_client.get(url)
    assert response.status == 200
    stale = await response.json()
    assert stale['data_status'] == 'stale'
    assert stale['error_code'] == 'live_query_failed'
    assert stale['total'] == first['total']
    assert stale['queried_at'] == first['queried_at']
    assert 'manager-summary' in caplog.text
    assert 'large' in caplog.text
    assert 'elapsed_seconds=' in caplog.text
    if failure == 'timeout':
        assert cancelled.is_set()
        assert 'timed out' in caplog.text
        assert 'budget_seconds=0.05' in caplog.text
    else:
        assert 'summary database failed' in caplog.text
        assert any(record.exc_info for record in caplog.records)
    missing = await summary_client.get('/v1/managers/summary?environment_id=uncached')
    assert missing.status == 503
    unavailable = await missing.json()
    assert unavailable['error_code'] == 'manager_summary_unavailable'
    assert 'summary database failed' not in str(unavailable)


async def test_summary_loads_are_bounded_and_do_not_block_lightweight_status(summary_client, monkeypatch):
    release, two_started = asyncio.Event(), asyncio.Event()
    active = peak = calls = 0

    async def load(_self, *, now, **scope):
        nonlocal active, peak, calls
        calls += 1
        active += 1
        peak = max(peak, active)
        if active == 2:
            two_started.set()
        try:
            await release.wait()
            return {'data_status': 'ready', 'queried_at': now, 'total': 1}
        finally:
            active -= 1

    monkeypatch.setattr(EnvironmentInventoryRepository, 'manager_summary', load)
    pollers = [
        asyncio.create_task(summary_client.get(f'/v1/managers/summary?environment_id=scope-{index}')) for index in range(8)
    ]

    async def lightweight():
        return {'data_status': 'ready'}

    try:
        await asyncio.wait_for(two_started.wait(), 1)
        light_cache = summary_client.app[LIVE_CACHE_KEY]
        assert light_cache.query_timeout == 0.45
        result = await asyncio.wait_for(light_cache.get(('live',), lightweight), 0.2)
        assert result['data_status'] == 'ready'
        assert calls == 2
    finally:
        release.set()
        responses = await asyncio.gather(*pollers)
    assert all(response.status == 200 for response in responses)
    assert calls == 8
    assert peak == 2


async def test_lightweight_cache_still_cancels_slow_queries(summary_client):
    cancelled = asyncio.Event()

    async def load():
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    with pytest.raises(TimeoutError):
        await summary_client.app[LIVE_CACHE_KEY].get(('slow-live',), load)
    assert cancelled.is_set()
