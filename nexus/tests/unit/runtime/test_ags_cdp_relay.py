import asyncio
import json
import time

from contextlib import asynccontextmanager

import aiohttp
import pytest

from aiohttp import web

from nexus.runtime.providers.ags.cdp_relay import CDPRelay


@asynccontextmanager
async def upstream_server(handler):
    app = web.Application()
    app.router.add_route('*', '/{path:.*}', handler)
    runner = web.AppRunner(app, shutdown_timeout=0.1)
    await runner.setup()
    await web.TCPSite(runner, '127.0.0.1', 0).start()
    try:
        yield f'http://127.0.0.1:{runner.addresses[0][1]}'
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['discovery502', 'stale404'])
async def test_handshake_recovers_and_rediscovers_without_command_replay(monkeypatch, failure):
    monkeypatch.setenv('no_proxy', '127.0.0.1')
    discoveries, messages, events = [], [], []

    async def handler(request):
        if request.path == '/json/version':
            discoveries.append(True)
            if failure == 'discovery502' and len(discoveries) == 1:
                return web.Response(status=502, text='sensitive upstream body')
            return web.json_response({'webSocketDebuggerUrl': f'ws://localhost/browser/{len(discoveries)}'})
        if failure == 'stale404' and request.path == '/browser/1':
            return web.Response(status=404)
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for message in ws:
            messages.append(message.data)
            await ws.send_str(message.data)
        return ws

    async with upstream_server(handler) as url:
        relay = CDPRelay(url, {'X-Secret': 'never-log-me'}, diagnostic_callback=events.append)
        try:
            async with aiohttp.ClientSession() as client:
                async with client.ws_connect(await relay.start()) as ws:
                    await ws.send_str('one command')
                    assert (await ws.receive(timeout=2)).data == 'one command'
            assert len(discoveries) == 2
            assert messages == ['one command']
            failures = [row for row in events if row['event'] == 'failure']
            assert failures[0]['stage'] == ('discovery' if failure == 'discovery502' else 'upstream_ws')
            assert failures[0]['retry']
            assert not failures[0]['downstream_prepared']
            if failure == 'stale404':
                assert any(row.get('endpoint_changed') for row in events)
            assert 'never-log-me' not in json.dumps(events)
            assert 'sensitive upstream body' not in json.dumps(events)
        finally:
            await relay.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('status', [401, 403, 400])
async def test_handshake_deterministic_errors_fail_fast_and_are_redacted(monkeypatch, status):
    monkeypatch.setenv('no_proxy', '127.0.0.1')
    calls, events = [], []

    async def handler(request):
        calls.append(True)
        return web.Response(status=status, text='token=secret')

    async with upstream_server(handler) as url:
        relay = CDPRelay(url, {}, diagnostic_callback=events.append)
        try:
            async with aiohttp.ClientSession() as client:
                async with client.get(await relay.start()) as response:
                    assert response.status == 502
                    value = await response.json()
                    assert value['stage'] == 'discovery'
                    assert value['diagnostic_id']
                    assert 'secret' not in json.dumps(value)
            assert len(calls) == 1
            assert next(row for row in events if row['event'] == 'failure')['upstream_status'] == status
        finally:
            await relay.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('payload', [{}, {'webSocketDebuggerUrl': 'file:///tmp/secret'}, ['not-a-mapping']])
async def test_bad_discovery_is_not_retried(monkeypatch, payload):
    monkeypatch.setenv('no_proxy', '127.0.0.1')
    calls = []

    async def handler(request):
        calls.append(True)
        return web.json_response(payload)

    async with upstream_server(handler) as url:
        relay = CDPRelay(url, {})
        try:
            async with aiohttp.ClientSession() as client:
                async with client.get(await relay.start()) as response:
                    assert response.status == 502
            assert len(calls) == 1
        finally:
            await relay.close()


@pytest.mark.asyncio
async def test_persistent_failure_is_bounded_and_not_generic500(monkeypatch):
    monkeypatch.setenv('no_proxy', '127.0.0.1')
    calls = []

    async def handler(request):
        calls.append(True)
        return web.Response(status=503)

    async with upstream_server(handler) as url:
        relay = CDPRelay(url, {})
        try:
            started = time.monotonic()
            async with aiohttp.ClientSession() as client:
                async with client.get(await relay.start()) as response:
                    assert response.status == 502
            assert len(calls) == 3
            assert time.monotonic() - started < 6
        finally:
            await relay.close()


@pytest.mark.asyncio
async def test_hung_discovery_has_shared_deadline(monkeypatch):
    monkeypatch.setenv('no_proxy', '127.0.0.1')

    async def handler(request):
        await asyncio.sleep(30)
        return web.json_response({})

    async with upstream_server(handler) as url:
        relay = CDPRelay(url, {}, connect_timeout=0.1)
        try:
            async with aiohttp.ClientSession() as client:
                started = time.monotonic()
                async with client.get(await relay.start()) as response:
                    assert response.status == 504
                assert time.monotonic() - started < 1
        finally:
            await relay.close()


@pytest.mark.asyncio
async def test_disconnected_caller_cancels_handshake(monkeypatch):
    monkeypatch.setenv('no_proxy', '127.0.0.1')
    entered = asyncio.Event()
    events = []

    async def handler(request):
        entered.set()
        await asyncio.sleep(30)
        return web.json_response({})

    async with upstream_server(handler) as url:
        relay = CDPRelay(url, {}, diagnostic_callback=events.append)
        try:
            async with aiohttp.ClientSession() as client:

                async def connect():
                    return await client.ws_connect(await relay.start())

                job = asyncio.create_task(connect())
                await asyncio.wait_for(entered.wait(), 2)
                job.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await job
                for _ in range(20):
                    if not relay._handshakes:
                        break
                    await asyncio.sleep(0.05)
                assert not relay._handshakes
                assert any(row['event'] == 'cancelled' for row in events)
        finally:
            await relay.close()


@pytest.mark.asyncio
async def test_established_disconnect_never_replays_or_reconnects(monkeypatch):
    monkeypatch.setenv('no_proxy', '127.0.0.1')
    discoveries, messages = [], []

    async def handler(request):
        if request.path == '/json/version':
            discoveries.append(True)
            return web.json_response({'webSocketDebuggerUrl': 'ws://localhost/browser/one'})
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        messages.append((await ws.receive()).data)
        await ws.close()
        return ws

    async with upstream_server(handler) as url:
        relay = CDPRelay(url, {})
        try:
            async with aiohttp.ClientSession() as client:
                async with client.ws_connect(await relay.start()) as ws:
                    await ws.send_str('stateful-command')
                    assert (await ws.receive(timeout=2)).type == aiohttp.WSMsgType.CLOSE
            assert discoveries == [True]
            assert messages == ['stateful-command']
        finally:
            await relay.close()


@pytest.mark.asyncio
async def test_cdp_relay_reuses_listener_refreshes_discovery_and_closes(monkeypatch):
    monkeypatch.setenv('no_proxy', '127.0.0.1')
    discoveries = []
    received_headers = []

    async def version(request):
        discoveries.append(True)
        received_headers.append(request.headers.get('X-Access-Token'))
        return web.json_response({'webSocketDebuggerUrl': f'ws://localhost:9222/browser/{len(discoveries)}'})

    async def echo(request):
        received_headers.append(request.headers.get('X-Access-Token'))
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                await ws.send_str(msg.data)
            elif msg.type == aiohttp.WSMsgType.BINARY:
                await ws.send_bytes(msg.data)
        return ws

    app = web.Application()
    app.router.add_get('/json/version', version)
    app.router.add_get('/browser/{id}', echo)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, '127.0.0.1', 0).start()
    relay = CDPRelay(f'http://127.0.0.1:{runner.addresses[0][1]}', {'X-Access-Token': 'test-token'})
    try:
        urls = await asyncio.gather(*(relay.start() for _ in range(5)))
        assert len(set(urls)) == 1
        async with aiohttp.ClientSession() as client:
            for _ in range(2):
                async with client.ws_connect(urls[0]) as ws:
                    await ws.send_str('{"id":1}')
                    assert (await ws.receive(timeout=2)).data == '{"id":1}'
                    await ws.send_bytes(b'cdp')
                    assert (await ws.receive(timeout=2)).data == b'cdp'
            ws = await client.ws_connect(urls[0])
            await asyncio.wait_for(relay.close(), timeout=3)
            assert (await ws.receive(timeout=2)).type == aiohttp.WSMsgType.CLOSE
            await ws.close()
            with pytest.raises(aiohttp.ClientConnectorError):
                await client.ws_connect(urls[0])
        assert len(discoveries) == 3
        assert received_headers == ['test-token'] * 6
        with pytest.raises(RuntimeError, match='closed'):
            await relay.start()
    finally:
        await relay.close()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_cdp_relay_routes_discovery_and_websocket_through_env_proxy(monkeypatch):
    requests = []

    async def proxy(request):
        requests.append(request.raw_path)
        if request.path == '/json/version':
            return web.json_response({'webSocketDebuggerUrl': 'ws://localhost:9222/devtools/browser/test'})
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.send_str('proxied')
        await ws.close()
        return ws

    app = web.Application()
    app.router.add_route('*', '/{path:.*}', proxy)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, '127.0.0.1', 0).start()
    monkeypatch.setenv('http_proxy', f'http://127.0.0.1:{runner.addresses[0][1]}')
    monkeypatch.setenv('no_proxy', '')
    monkeypatch.setenv('NO_PROXY', '')
    relay = CDPRelay('http://cdp.invalid', {})
    try:
        async with aiohttp.ClientSession() as client:
            async with client.ws_connect(await relay.start()) as ws:
                assert (await ws.receive(timeout=2)).data == 'proxied'
        assert requests == ['http://cdp.invalid/json/version', 'ws://cdp.invalid/devtools/browser/test']
    finally:
        await relay.close()
        await runner.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('kind', ['waa', 'osworld'])
async def test_runtime_cdp_relay_switch(enabled, kind):
    from unittest.mock import AsyncMock

    from nexus.runtime.providers.ags.handle import AGSInstanceHandle
    from nexus.runtime.providers.ags.osworld import AGSRawOSWorldRuntime
    from nexus.runtime.providers.ags.waa import AGSRawWindowsAgentArenaRuntime

    handle = AGSInstanceHandle(instance_id='test', access_token='test-token', domain='example.test')
    runtime_class = AGSRawWindowsAgentArenaRuntime if kind == 'waa' else AGSRawOSWorldRuntime
    runtime = runtime_class(handle, enable_cdp_relay=enabled)
    if enabled:
        runtime._cdp_relay.start = AsyncMock(return_value='ws://127.0.0.1:1234/cdp')
        runtime._cdp_relay.close = AsyncMock()
        assert await runtime.chromium_cdp_url() == 'ws://127.0.0.1:1234/cdp'
        await runtime.close()
        runtime._cdp_relay.close.assert_awaited_once()
    else:
        assert runtime._cdp_relay is None
        assert await runtime.chromium_cdp_url() == runtime.chromium_url
        await runtime.close()
