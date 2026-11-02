from __future__ import annotations

import asyncio

import pytest

from aiohttp.test_utils import TestClient, TestServer

from nexus.distributed.access.auth import AuthContext
from nexus.distributed.gateways.control.app import create_control_app
from nexus.distributed.gateways.control.business_statistics import BUSINESS_STATISTICS_KEY
from tests.unit.distributed.analytics.test_statistics import database, query, seed


@pytest.mark.asyncio
async def test_statistics_permission_is_checked_before_every_cached_read(database, monkeypatch):
    from nexus.distributed.access import auth

    config, factory, _ = database
    await seed(factory)
    roles = {'user', 'admin'}

    async def authenticate(*args, **kwargs):
        return AuthContext(principal_id='test-admin', issuer='test', tenant='test', subject='admin', roles=frozenset(roles), scopes=frozenset({'*'}), auth_type='test')

    monkeypatch.setattr(auth, '_authenticate', authenticate)
    client = TestClient(TestServer(create_control_app(session_factory=factory, config=config)))
    await client.start_server()
    try:
        response = await client.get('/v1/system/business-statistics', params={key: value for key, value in query().filters().items() if value is not None})
        assert response.status == 200
        assert (await response.json())['summary']['total'] == 4
        assert response.headers['Cache-Control'] == 'no-store'
        roles.remove('admin')
        denied = await client.get('/v1/system/business-statistics', params={key: value for key, value in query().filters().items() if value is not None})
        assert denied.status == 403
        roles.add('admin')
        invalid = await client.get('/v1/system/business-statistics', params={'start_at': '2026-09-14T00:00:00', 'end_at': '2026-09-15T00:00:00'})
        assert invalid.status == 400
        repeated = await client.get('/v1/system/business-statistics?basis=created&basis=finished')
        assert repeated.status == 400
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_browser_disconnect_stops_the_last_query_and_pool_remains_usable(database, monkeypatch):
    from nexus.distributed.analytics import reader as module

    config, factory, _ = database
    await seed(factory)
    entered, stopped = asyncio.Event(), asyncio.Event()
    original = module.load_statistics

    async def slow(*args):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    monkeypatch.setattr(module, 'load_statistics', slow)
    app = create_control_app(session_factory=factory, config=config)
    client = TestClient(TestServer(app))
    await client.start_server()
    params = {key: value for key, value in query().filters().items() if value is not None}

    async def get():
        return await client.get('/v1/system/business-statistics', params=params)

    try:
        pending = asyncio.create_task(get())
        await asyncio.wait_for(entered.wait(), 3)
        pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)
        await asyncio.wait_for(stopped.wait(), 3)
        monkeypatch.setattr(module, 'load_statistics', original)
        result = await get()
        assert result.status == 200
        assert (await result.json())['summary']['total'] == 4
        assert app[BUSINESS_STATISTICS_KEY]._slots._value == 1
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_disabled_statistics_does_not_open_a_query_pool(database):
    config, factory, _ = database
    config = config.model_copy(update={'business_statistics': config.business_statistics.model_copy(update={'enabled': False})})
    app = create_control_app(session_factory=factory, config=config)
    assert app[BUSINESS_STATISTICS_KEY].engine is None
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.get('/v1/system/business-statistics')
        assert response.status == 503
        assert await response.json() == {'error_code': 'statistics_disabled'}
    finally:
        await client.close()
