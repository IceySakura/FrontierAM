from __future__ import annotations

from typing import Any

import pytest
import socketio

from aiohttp import web

from nexus.runtime.context import SIOBridge
from nexus.runtime.server.services.rsio_llm_bridge import RSIOLLMBridge
from nexus.runtime.server.worker.sdk import WorkerSDK


@pytest.mark.asyncio
@pytest.mark.parametrize('response_bytes', [1024, 5 * 1024 * 1024], ids=['small', 'above-default-websocket-limit'])
async def test_worker_receives_llm_final_without_disconnect_or_resend(
    unused_tcp_port: int, response_bytes: int
) -> None:
    server = socketio.AsyncServer(async_mode='aiohttp')
    app = web.Application()
    server.attach(app)
    runner = web.AppRunner(app)
    sdk = WorkerSDK(f'http://127.0.0.1:{unused_tcp_port}', 'transport-test')
    bridge = RSIOLLMBridge(SIOBridge(sdk._sio, namespace='/worker'))
    content = 'x' * response_bytes
    requests: list[str] = []

    @server.on('llm_request', namespace='/worker')
    async def reply(sid: str, data: dict[str, Any]) -> None:
        requests.append(data['request_id'])
        await server.emit(
            'llm_response',
            {
                'request_id': data['request_id'],
                'success': True,
                'rsio': {'kind': 'final'},
                'response': {'content': content},
            },
            to=sid,
            namespace='/worker',
        )

    await runner.setup()
    try:
        await web.TCPSite(runner, '127.0.0.1', unused_tcp_port).start()
        await sdk._sio.connect(sdk.base_url, namespaces=['/worker'], transports=['websocket'])
        result = await bridge.request({'messages': [{'role': 'user', 'content': 'hello'}]}, timeout=5)

        assert result == {'content': content}
        assert len(requests) == 1
        assert sdk._connected.is_set()
    finally:
        try:
            await sdk._sio.disconnect()
        finally:
            await server.shutdown()
            await runner.cleanup()
