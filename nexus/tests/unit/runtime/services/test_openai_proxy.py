from __future__ import annotations

import asyncio
import json

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import socketio
import uvicorn

from nexus.runtime.context import SIOBridge
from nexus.runtime.server.services.openai_proxy import LLMBridge, OpenAIProxy, simulate_chat_completions_stream


class _Bridge:
    def __init__(self, response: dict) -> None:
        self.response = response
        self.requests: list[dict] = []

    async def request(self, params: dict, timeout: float) -> dict:
        self.requests.append(params)
        return self.response


async def _post(proxy: OpenAIProxy, body: dict) -> httpx.Response:
    transport = httpx.ASGITransport(app=proxy._create_app())
    async with httpx.AsyncClient(transport=transport, base_url='http://testserver') as client:
        return await client.post('/v1/chat/completions', json=body)


def _sse_payloads(response: httpx.Response) -> list[dict]:
    return [
        json.loads(line.removeprefix('data: ')) for line in response.text.splitlines() if line.startswith('data: {')
    ]


@pytest.mark.asyncio
async def test_simulate_chat_completions_stream_emits_done_marker():
    payload = {
        'id': 'chatcmpl-test',
        'object': 'chat.completion',
        'created': 123,
        'model': 'gpt-5',
        'choices': [
            {
                'index': 0,
                'message': {
                    'role': 'assistant',
                    'content': 'ok',
                },
                'finish_reason': 'stop',
            }
        ],
    }

    chunks = [chunk async for chunk in simulate_chat_completions_stream(payload)]

    assert chunks[-1] == 'data: [DONE]\n\n'

    first_payload = json.loads(chunks[0][len('data: ') : -2])
    assert first_payload['object'] == 'chat.completion.chunk'
    assert first_payload['choices'][0]['delta']['role'] == 'assistant'
    assert first_payload['choices'][0]['delta']['content'] == 'ok'


@pytest.mark.asyncio
async def test_simulate_chat_completions_stream_forwards_thinking_blocks():
    payload = {
        'id': 'chatcmpl-test',
        'object': 'chat.completion',
        'created': 123,
        'model': 'claude-opus-5',
        'choices': [
            {
                'index': 0,
                'message': {
                    'role': 'assistant',
                    'content': 'ok',
                    'reasoning_content': 'plan',
                    'thinking_blocks': [
                        {'type': 'thinking', 'thinking': 'plan', 'signature': 'sig-1'},
                    ],
                },
                'finish_reason': 'stop',
            }
        ],
    }

    chunks = [chunk async for chunk in simulate_chat_completions_stream(payload)]
    first_payload = json.loads(chunks[0][len('data: ') : -2])
    delta = first_payload['choices'][0]['delta']
    assert delta['reasoning_content'] == 'plan'
    assert delta['thinking_blocks'] == [{'type': 'thinking', 'thinking': 'plan', 'signature': 'sig-1'}]

    final_payload = json.loads(chunks[1][len('data: ') : -2])
    assert final_payload['choices'][0]['finish_reason'] == 'stop'


@pytest.mark.asyncio
async def test_simulate_chat_completions_stream_preserves_tool_call_order_and_arguments():
    payload = {
        'id': 'chatcmpl-test',
        'object': 'chat.completion',
        'created': 123,
        'model': 'gpt-5.5',
        'choices': [
            {
                'index': 0,
                'message': {
                    'role': 'assistant',
                    'tool_calls': [
                        {
                            'id': 'call_1',
                            'type': 'function',
                            'function': {
                                'name': 'str_replace_editor',
                                'arguments': '{"command":"str_replace","old_str":"old","new_str":"new"}',
                            },
                        },
                        {
                            'id': 'call_2',
                            'type': 'function',
                            'function': {
                                'name': 'bash',
                                'arguments': '{"command":"pytest -q"}',
                            },
                        },
                    ],
                },
                'finish_reason': 'tool_calls',
            }
        ],
    }

    chunks = [chunk async for chunk in simulate_chat_completions_stream(payload)]

    first_payload = json.loads(chunks[0][len('data: ') : -2])
    tool_calls = first_payload['choices'][0]['delta']['tool_calls']
    assert [tool_call['id'] for tool_call in tool_calls] == ['call_1', 'call_2']
    assert [tool_call['index'] for tool_call in tool_calls] == [0, 1]
    assert tool_calls[0]['function']['arguments'] == '{"command":"str_replace","old_str":"old","new_str":"new"}'
    assert tool_calls[1]['function']['arguments'] == '{"command":"pytest -q"}'


@pytest.mark.asyncio
async def test_simulate_chat_completions_stream_preserves_existing_tool_call_index():
    payload = {
        'choices': [
            {
                'message': {
                    'role': 'assistant',
                    'tool_calls': [
                        {
                            'index': 7,
                            'id': 'call_1',
                            'type': 'function',
                            'function': {'name': 'bash', 'arguments': '{}'},
                        }
                    ],
                },
                'finish_reason': 'tool_calls',
            }
        ]
    }

    chunks = [chunk async for chunk in simulate_chat_completions_stream(payload)]

    first_payload = json.loads(chunks[0][len('data: ') : -2])
    assert first_payload['choices'][0]['delta']['tool_calls'][0]['index'] == 7


@pytest.mark.asyncio
async def test_stream_endpoint_emits_usage_chunk_when_requested():
    completion = {
        'id': 'chatcmpl-usage',
        'created': 123,
        'model': 'gpt-5.5',
        'choices': [
            {
                'index': 0,
                'message': {'role': 'assistant', 'content': 'ok'},
                'finish_reason': 'stop',
            }
        ],
        'usage': {'prompt_tokens': 5, 'completion_tokens': 2, 'total_tokens': 7},
    }
    bridge = _Bridge(completion)
    proxy = OpenAIProxy(sio=object())  # type: ignore[arg-type]
    proxy.bridge = bridge  # type: ignore[assignment]

    response = await _post(
        proxy,
        {
            'model': 'nexus-model',
            'messages': [{'role': 'user', 'content': 'hello'}],
            'stream': True,
            'stream_options': {'include_usage': True},
        },
    )
    payloads = _sse_payloads(response)

    assert response.status_code == 200
    assert bridge.requests[0]['stream'] is False
    assert payloads[-1]['choices'] == []
    assert payloads[-1]['usage'] == completion['usage']
    assert response.text.rstrip().endswith('data: [DONE]')


@pytest.mark.asyncio
async def test_stream_endpoint_omits_usage_chunk_when_not_requested():
    completion = {
        'id': 'chatcmpl-usage',
        'choices': [
            {
                'index': 0,
                'message': {'role': 'assistant', 'content': 'ok'},
                'finish_reason': 'stop',
            }
        ],
        'usage': {'prompt_tokens': 5, 'completion_tokens': 2, 'total_tokens': 7},
    }
    proxy = OpenAIProxy(sio=object())  # type: ignore[arg-type]
    proxy.bridge = _Bridge(completion)  # type: ignore[assignment]

    response = await _post(
        proxy,
        {
            'model': 'nexus-model',
            'messages': [{'role': 'user', 'content': 'hello'}],
            'stream': True,
        },
    )

    assert response.status_code == 200
    assert all(payload.get('choices') != [] for payload in _sse_payloads(response))


@pytest.mark.asyncio
async def test_stop_server_error_clears_references() -> None:
    proxy = OpenAIProxy(SIOBridge(socketio.AsyncClient(), '/worker'))
    bridge = LLMBridge(proxy.sio)
    proxy.bridge = bridge
    proxy._server = uvicorn.Server(uvicorn.Config(proxy._create_app()))

    async def failed_server() -> None:
        raise RuntimeError('server failed')

    server_task = asyncio.create_task(failed_server())
    await asyncio.wait((server_task,))
    proxy._server_task = server_task

    with pytest.raises(RuntimeError, match='server failed'):
        await proxy.stop()

    assert proxy.bridge is None, 'Failed server shutdown must release the bridge'
    assert proxy._server is None, 'Failed server shutdown must release the server'
    assert proxy._server_task is None, 'Failed server shutdown must release the server task'
    with pytest.raises(RuntimeError, match='closed'):
        await bridge.request({'model': 'm'})


@pytest.mark.asyncio
async def test_stop_cancels_own_pending_request() -> None:
    raw_sio = socketio.AsyncClient()
    proxy = OpenAIProxy(SIOBridge(raw_sio, '/worker'))
    bridge = LLMBridge(proxy.sio)
    proxy.bridge = bridge
    emitted = asyncio.Event()

    async def emit(*args: object, **kwargs: object) -> None:
        emitted.set()

    with patch.object(raw_sio, 'emit', side_effect=emit):
        request_task = asyncio.create_task(bridge.request({'model': 'm'}, timeout=None))
        try:
            await asyncio.wait_for(emitted.wait(), timeout=1)
            pending = next(iter(bridge._pending.values()))
            await proxy.stop()
            assert pending.future.cancelled(), 'Stopping the proxy must cancel its outstanding bridge request'
            assert not bridge._pending, 'Stopping the proxy must remove its pending response entry'
            with pytest.raises(asyncio.CancelledError):
                await request_task
            await proxy.stop()
        finally:
            request_task.cancel()
            await asyncio.gather(request_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_stop_external_cancellation_releases_bridge_and_propagates() -> None:
    proxy = OpenAIProxy(SIOBridge(socketio.AsyncClient(), '/worker'))
    bridge = LLMBridge(proxy.sio)
    proxy.bridge = bridge
    proxy._server = uvicorn.Server(uvicorn.Config(proxy._create_app()))
    serving = asyncio.Event()
    stopping = asyncio.Event()

    async def serve() -> None:
        serving.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            stopping.set()
            await asyncio.Event().wait()

    server_task = asyncio.create_task(serve())
    proxy._server_task = server_task
    await asyncio.wait_for(serving.wait(), timeout=1)
    stop_task = asyncio.create_task(proxy.stop())
    try:
        await asyncio.wait_for(stopping.wait(), timeout=1)
        stop_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(stop_task, timeout=1)
        assert proxy.bridge is None, 'Caller cancellation must still release the bridge reference'
        assert proxy._server is None, 'Caller cancellation must release the server'
        assert proxy._server_task is None, 'Caller cancellation must release the server task'
        with pytest.raises(RuntimeError, match='closed'):
            await bridge.request({'model': 'm'})
    finally:
        stop_task.cancel()
        server_task.cancel()
        await asyncio.gather(stop_task, server_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_start_forwards_channel_and_disabled_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    raw_sio = socketio.AsyncClient()
    proxy = OpenAIProxy(SIOBridge(raw_sio, '/worker'), channel_id='session-a', request_timeout=None)
    monkeypatch.setattr(
        'nexus.runtime.server.services.openai_proxy.uvicorn.Server',
        lambda config: SimpleNamespace(serve=AsyncMock(), should_exit=False),
    )
    try:
        await proxy.start()
        assert proxy.bridge is not None, 'Start must create a bridge'
        assert proxy.bridge.channel_id == 'session-a', 'Start must forward the proxy channel to its bridge'
        assert proxy.request_timeout is None, 'A disabled request timeout must be preserved'
    finally:
        await proxy.stop()
