import asyncio
import base64
import json
import shlex
import subprocess

from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import aiohttp
import pytest

from aiohttp import web

from nexus.extensions.tasks.os_world_v2.adapter.transport import WEBSITE_REQUEST, Transport


@asynccontextmanager
async def server(handler):
    app = web.Application(client_max_size=32 * 1024**2)
    app.router.add_route('*', '/{path:.*}', handler)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, '127.0.0.1', 0).start()
    try:
        yield f'http://127.0.0.1:{runner.addresses[0][1]}'
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_large_website_state_uses_private_upload_not_execute_argv():
    response = 'NEXUS_WEBSITE_RESPONSE=' + json.dumps(
        {
            'status': 200,
            'headers': {},
            'body': base64.b64encode(b'{}').decode(),
        }
    )
    runtime = SimpleNamespace(
        run_command=AsyncMock(return_value=SimpleNamespace(return_code=0, stdout=response)),
        setup_upload_bytes=AsyncMock(return_value=SimpleNamespace(status_code=200)),
    )
    state = b'x' * 300000
    async with Transport(runtime, ['teamchat'], 'localhost') as proxy:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f'http://{proxy.host}:5000/_nexus/website',
                json={
                    'url': 'http://teamchat.localhost/api/state',
                    'method': 'PUT',
                    'headers': {'Cookie': 'private'},
                    'body': base64.b64encode(state).decode(),
                },
            ) as reply:
                assert reply.status == 200
    upload = runtime.setup_upload_bytes.call_args
    assert base64.b64decode(json.loads(upload.args[1])['body']) == state
    assert all(len(call.kwargs['command']) < 8192 for call in runtime.run_command.call_args_list)
    assert runtime.run_command.call_args_list[0].kwargs['command'].startswith('umask 077')
    assert runtime.run_command.call_args_list[-1].kwargs['command'].startswith('rm -f')


@pytest.mark.asyncio
async def test_upload_streams_without_request_read_and_preserves_length(monkeypatch):
    content = b'x' * (8 * 1024**2)
    observed = {}

    async def upload(request):
        observed['length'] = request.headers.get('Content-Length')
        observed['auth'] = request.headers.get('X-Access-Token')
        observed['body'] = await request.read()
        return web.Response(body=b'uploaded')

    async with server(upload) as url:
        runtime = SimpleNamespace(
            handle=SimpleNamespace(
                url=lambda *args: url + '/setup/upload',
                headers={
                    'X-Access-Token': 'fixture',
                },
            )
        )
        async with Transport(runtime, [], 'localhost') as proxy:
            # Detect the old whole-upload buffering path without patching the upstream server.
            original = web.Request.read

            async def guarded_read(request):
                if request.host.startswith(proxy.host):
                    raise AssertionError('Proxy must not buffer upload via request.read()')
                return await original(request)

            monkeypatch.setattr(web.Request, 'read', guarded_read)
            async with aiohttp.ClientSession() as session:
                async with session.post(f'http://{proxy.host}:5000/setup/upload', data=content) as reply:
                    assert reply.status == 200
                    assert await reply.read() == b'uploaded'
    assert observed == {'length': str(len(content)), 'auth': 'fixture', 'body': content}


@pytest.mark.asyncio
async def test_standard_cdp_relay_preserves_auth_and_large_frames():
    large = 'x' * (5 * 1024**2)
    observed = []

    async def chrome(request):
        observed.append(request.headers.get('X-Access-Token'))
        if request.path == '/json/version':
            return web.json_response({'webSocketDebuggerUrl': 'ws://localhost:1337/devtools/browser/current'})
        ws = web.WebSocketResponse(max_msg_size=0)
        await ws.prepare(request)
        await ws.send_str(large)
        await ws.close()
        return ws

    async with server(chrome) as url:
        runtime = SimpleNamespace(
            handle=SimpleNamespace(
                url=lambda *args: url,
                headers={
                    'X-Access-Token': 'fixture',
                },
            )
        )
        async with Transport(runtime, [], 'localhost') as proxy:
            async with aiohttp.ClientSession() as session:
                async with session.get(f'http://{proxy.host}:9222/json/version/') as reply:
                    endpoint = (await reply.json())['webSocketDebuggerUrl']
                assert 'fixture' not in endpoint
                async with session.ws_connect(endpoint, max_msg_size=0) as ws:
                    message = await ws.receive(timeout=10)
                    assert message.data == large
    assert observed == ['fixture', 'fixture']


@pytest.mark.asyncio
async def test_cdp_heartbeat_keeps_idle_proxy_alive():
    from nexus.runtime.providers.ags.cdp_relay import CDPRelay

    pings = []

    async def chrome(request):
        if request.path == '/json/version':
            return web.json_response({'webSocketDebuggerUrl': 'ws://localhost/browser'})
        ws = web.WebSocketResponse(autoping=False)
        await ws.prepare(request)
        try:
            while True:
                message = await ws.receive(timeout=0.5)
                if message.type == aiohttp.WSMsgType.PING:
                    pings.append(True)
                    await ws.pong(message.data)
                elif message.type == aiohttp.WSMsgType.TEXT:
                    await ws.send_str(message.data)
                    break
                else:
                    break
        except asyncio.TimeoutError:
            pass
        await ws.close()
        return ws

    async with server(chrome) as url:
        relay = CDPRelay(url, {}, heartbeat=0.1)
        try:
            endpoint = await relay.start()
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(endpoint) as ws:
                    await asyncio.sleep(1.2)
                    await ws.send_str('still-connected')
                    reply = await ws.receive(timeout=3)
                    assert reply.data == 'still-connected'
        finally:
            await relay.close()
    assert pings


@pytest.mark.asyncio
@pytest.mark.parametrize(('method', 'expected_attempts'), [('GET', 3), ('PUT', 1), ('POST', 1)])
async def test_website_retries_reads_but_never_replays_state_writes(method, expected_attempts):
    calls = []

    async def handler(request):
        calls.append(request.method)
        return web.Response(status=503)

    async with server(handler) as url:
        spec = {'method': method, 'url': url, 'headers': {}, 'body': '', 'timeout': 2}
        script = WEBSITE_REQUEST.replace('time.sleep(attempt+1)', 'time.sleep(0)')
        result = await asyncio.to_thread(
            subprocess.run,
            ['python3', '-c', script],
            input=json.dumps(spec),
            capture_output=True,
            text=True,
            timeout=10,
        )
    assert result.returncode == 0
    assert calls == [method] * expected_attempts


def test_worker_diagnostic_tail_is_bounded_and_redacted():
    from nexus.extensions.tasks.os_world_v2.adapter.adapter import Worker

    worker = Worker(None, 10)
    worker.secret_values = ['secret-value']
    worker.stderr_tail = 'PLAYWRIGHT ENV: secret-value\n' + 'x' * 5000 + '\nsecret-value'
    detail = worker.diagnostic_tail()
    assert len(detail) <= 4000
    assert 'secret-value' not in detail
    assert 'PLAYWRIGHT ENV' not in detail


@pytest.mark.asyncio
async def test_worker_stderr_markers_separate_setup_and_evaluate(tmp_path):
    import asyncio

    from nexus.extensions.tasks.os_world_v2.adapter.adapter import Worker

    worker = Worker(None, 10)
    worker.diagnostic_directory = tmp_path
    worker.secret_values = ['private-key']
    worker.stderr_boundary = 'setup-id'
    reader = asyncio.StreamReader()
    worker.process = SimpleNamespace(stderr=reader)
    reader.feed_data(b'NEXUS_WORKER_STDERR:setup-id:sta')
    reader.feed_data(b'rt\nold setup CDP500 private-key\nNEXUS_WORKER_STDERR:setup-id:end\n')
    reader.feed_eof()
    await worker._read_stderr()
    assert worker.stderr_end.is_set()
    assert 'old setup CDP500' in worker.diagnostic_tail()
    assert 'private-key' not in worker.diagnostic_tail()
    worker.stderr_boundary = 'evaluate-id'
    worker.stderr_end.clear()
    worker._stderr_line('NEXUS_WORKER_STDERR:evaluate-id:start\n')
    worker._stderr_line('current screenshot failure\n')
    worker._stderr_line('NEXUS_WORKER_STDERR:evaluate-id:end\n')
    assert worker.diagnostic_tail() == 'current screenshot failure'
    assert 'old setup' in worker.stderr_tail
    archived = (tmp_path / 'worker-stderr.log').read_text()
    assert 'old setup CDP500' in archived
    assert 'current screenshot failure' in archived
    assert 'private-key' not in archived
    assert 'NEXUS_WORKER_STDERR:' not in archived


@pytest.mark.asyncio
@pytest.mark.parametrize('guest_status', [200, 403])
async def test_large_upload_chunks_verify_and_preserve_guest_permissions(monkeypatch, guest_status):
    from urllib.parse import urlsplit

    from nexus.extensions.tasks.os_world_v2.adapter import uploads

    monkeypatch.setattr(uploads, 'CHUNK_SIZE', 512 * 1024)
    content = b'fixture' * 400000
    received, chunks = [], []

    async def guest_upload(request):
        form = await request.multipart()
        async for part in form:
            received.append((part.name, await part.read()))
        return web.Response(status=guest_status, text='original endpoint result')

    async with server(guest_upload) as url:
        port = urlsplit(url).port

        async def command(**kwargs):
            cmd = kwargs['command'].replace('HTTPConnection("127.0.0.1",5000,', f'HTTPConnection("127.0.0.1",{port},')
            if cmd.startswith('python3 -c '):
                source = shlex.split(cmd)[2].replace(
                    'HTTPConnection("127.0.0.1",5000,', f'HTTPConnection("127.0.0.1",{port},'
                )
                args = ['python3', '-c', source]
            else:
                args = ['bash', '-c', cmd]
            result = await asyncio.to_thread(subprocess.run, args, capture_output=True, text=True, timeout=15)
            return SimpleNamespace(return_code=result.returncode, stdout=result.stdout, stderr=result.stderr)

        async def upload(path, block, **kwargs):
            chunks.append(len(block))
            Path(path).write_bytes(block)
            return SimpleNamespace(status_code=200)

        runtime = SimpleNamespace(run_command=command, setup_upload_bytes=upload)

        async def gateway(request):
            return await uploads.upload_large_multipart(runtime, request)

        async with server(gateway) as endpoint:
            form = aiohttp.FormData()
            form.add_field('file_path', '/home/user/Desktop/fixture.zip')
            form.add_field('file_data', content, filename='fixture.zip')
            async with aiohttp.ClientSession() as session:
                async with session.post(endpoint, data=form) as response:
                    assert response.status == guest_status
                    assert await response.text() == 'original endpoint result'
    assert len(chunks) > 1
    assert max(chunks) <= 512 * 1024
    assert received == [('file_path', b'/home/user/Desktop/fixture.zip'), ('file_data', content)]
