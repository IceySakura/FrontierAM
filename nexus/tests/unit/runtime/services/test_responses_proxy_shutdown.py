import asyncio

import httpx
import pytest

from nexus.runtime.server.services.responses_proxy.server import ResponsesProxy


class _SilentHost:
    def on(self, event):
        return lambda handler: handler

    async def emit(self, event, data):
        pass


@pytest.mark.parametrize('rsio_disabled', [False, True])
async def test_stop_cancels_unbounded_request_after_client_disconnect(unused_tcp_port, monkeypatch, rsio_disabled):
    if rsio_disabled:
        monkeypatch.setenv('NEXUS_DISABLE_LLM_RSIO', '1')
    else:
        monkeypatch.delenv('NEXUS_DISABLE_LLM_RSIO', raising=False)
    proxy = ResponsesProxy(
        sio=_SilentHost(),
        listen_port=unused_tcp_port,
        request_timeout=None,
        responses_to_openai_converter=lambda *_args, **_kwargs: {},
    )
    await proxy.start()
    try:
        async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
            request = asyncio.create_task(client.post(f'http://127.0.0.1:{unused_tcp_port}/v1/responses', json={}))
            try:
                for _ in range(100):
                    if proxy.bridge._pending:
                        break
                    await asyncio.sleep(0.01)
                assert proxy.bridge._pending, 'the proxy must be waiting for the host response'
            finally:
                request.cancel()
                await asyncio.gather(request, return_exceptions=True)

        await proxy.stop()
        assert not proxy.bridge._pending
        assert not proxy._server.server_state.tasks
    finally:
        # Clean up even on the pre-fix implementation, whose server task exits
        # without cancelling the disconnected client's ASGI request.
        pending = list(proxy._server.server_state.tasks)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        if proxy._server_task is not None and not proxy._server_task.done():
            await proxy.stop()
