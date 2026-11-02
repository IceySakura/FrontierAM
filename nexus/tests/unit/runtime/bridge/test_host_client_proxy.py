import pytest

from nexus.runtime.bridge.host_client import BridgeHostClient


@pytest.mark.asyncio
async def test_explicit_proxy_reaches_socketio_and_closes_session(monkeypatch) -> None:
    import socketio

    from aiohttp import web
    from aiohttp.test_utils import TestServer

    requests = []

    @web.middleware
    async def record(request, handler):
        if request.raw_path.startswith(('ws://', 'wss://')):
            return web.Response(status=400)
        requests.append((request.host, request.query.get('transport')))
        assert request.raw_path.startswith('http://bridge.invalid:8765/')
        return await handler(request)

    app = web.Application(middlewares=[record])
    server = socketio.AsyncServer(async_mode='aiohttp', cors_allowed_origins=[])
    server.attach(app)

    @server.on('probe')
    async def probe(sid, data):
        return data

    monkeypatch.setenv('NO_PROXY', '*')
    monkeypatch.setenv('no_proxy', '*')
    async with TestServer(app) as proxy:
        client = BridgeHostClient(
            base_url='http://bridge.invalid:8765', proxy_url=str(proxy.make_url('/')), keepalive_interval=None
        )
        try:
            # Reconnect after an explicit disconnect also retains the proxy.
            for _ in range(2):
                await client.ws_connect()
                assert client.ws_connected
                assert await client._sio.call('probe', {'request_id': 'test'}, timeout=2) == {'request_id': 'test'}
                await client.ws_disconnect()
                assert client._sio.eio.http.closed
        finally:
            await client.ws_disconnect()
            await server.shutdown()
    assert requests == [('bridge.invalid', 'websocket')] * 2


@pytest.mark.asyncio
async def test_failed_proxy_handshake_closes_http_session() -> None:
    import socketio

    from aiohttp import web
    from aiohttp.test_utils import TestServer

    async def unavailable(request):
        return web.Response(status=503)

    app = web.Application()
    app.router.add_get('/socket.io/', unavailable)
    async with TestServer(app) as proxy:
        client = BridgeHostClient(base_url='http://bridge.invalid', proxy_url=str(proxy.make_url('/')))
        with pytest.raises(socketio.exceptions.ConnectionError):
            await client.ws_connect()
        assert not client.ws_connected
        assert client._sio.eio.http.closed


@pytest.mark.asyncio
async def test_direct_and_proxy_bridges_connect_concurrently_with_path_prefix(monkeypatch):
    import asyncio

    import socketio

    from aiohttp import web
    from aiohttp.test_utils import TestServer

    seen = []
    servers = []

    def app_for(label):
        @web.middleware
        async def record(request, handler):
            seen.append((label, request.raw_path, request.query.get('transport')))
            assert request.path == '/prefix/socket.io/'
            return await handler(request)

        app = web.Application(middlewares=[record])
        server = socketio.AsyncServer(async_mode='aiohttp', cors_allowed_origins=[])
        server.attach(app, socketio_path='prefix/socket.io')
        servers.append(server)

        @server.on('probe')
        async def probe(sid, data):
            return {'server': label, 'data': data}

        return app

    monkeypatch.setenv('NO_PROXY', '*')
    monkeypatch.setenv('no_proxy', '*')
    async with TestServer(app_for('direct')) as direct, TestServer(app_for('proxy')) as proxy:
        clients = [
            BridgeHostClient(base_url=str(direct.make_url('/prefix')), keepalive_interval=None),
            BridgeHostClient(
                base_url='http://bridge.invalid/prefix', proxy_url=str(proxy.make_url('/')), keepalive_interval=None
            ),
        ]
        try:
            await asyncio.gather(*(client.ws_connect() for client in clients))
            replies = await asyncio.gather(
                *(client._sio.call('probe', label, timeout=3) for client, label in zip(clients, ['direct', 'proxy'], strict=True))
            )
            assert replies == [{'server': label, 'data': label} for label in ['direct', 'proxy']]
            assert clients[0]._proxy_session is None
            assert clients[1]._proxy_session is not None
        finally:
            await asyncio.gather(*(client.ws_disconnect() for client in clients))
            await asyncio.gather(*(server.shutdown() for server in servers))
        assert clients[1]._proxy_session.closed
    assert any(label == 'direct' and transport == 'polling' for label, _, transport in seen)
    assert all(
        raw.startswith('http://bridge.invalid/prefix/') and transport == 'websocket'
        for label, raw, transport in seen
        if label == 'proxy'
    )
