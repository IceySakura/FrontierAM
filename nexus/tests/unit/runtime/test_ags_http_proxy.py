import asyncio

import pytest

from nexus.utils.async_http import http_client_with_trust_env as http_client


@pytest.mark.asyncio
@pytest.mark.parametrize('bypass', [False, True])
async def test_ags_http_respects_proxy_environment(monkeypatch, bypass):
    requests = []

    async def respond(reader, writer):
        requests.append(await reader.readuntil(b'\r\n\r\n'))
        writer.write(b'HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok')
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    async with await asyncio.start_server(respond, '127.0.0.1', 0) as server:
        port = server.sockets[0].getsockname()[1]
        for name in ('HTTP_PROXY', 'HTTPS_PROXY', 'NO_PROXY', 'ALL_PROXY'):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv('http_proxy', 'http://127.0.0.1:1' if bypass else f'http://127.0.0.1:{port}')
        monkeypatch.setenv('no_proxy', '127.0.0.1' if bypass else '')
        url = f'http://127.0.0.1:{port}/probe' if bypass else 'http://ags-proxy-test.invalid/probe'
        try:
            response = await asyncio.wait_for(http_client.get(url), timeout=5)
            assert response.status_code == 200
            assert response.content == b'ok'
            expected = b'GET /probe ' if bypass else b'GET http://ags-proxy-test.invalid/probe '
            assert requests[0].startswith(expected)
        finally:
            await http_client.close()
