from __future__ import annotations

import json

from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from fastapi.testclient import TestClient

from nexus.runtime.bridge.adapters.ags_osworld import create_bridge_fastapi_app
from nexus.runtime.context import SIOBridge
from nexus.runtime.server.services.anthropic_proxy.server import AnthropicProxy, LLMBridge


class DummySio:
    def __init__(self) -> None:
        self.handlers: dict[str, object] = {}

    def on(self, event: str):
        def decorator(handler: object) -> object:
            self.handlers[event] = handler
            return handler

        return decorator

    async def emit(self, event: str, data: object = None) -> None:
        pass


class DummyBridge:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def request(self, openai_params: dict[str, Any], timeout: float = 300) -> dict[str, Any]:
        self.calls.append({'openai_params': openai_params, 'timeout': timeout})
        return {
            'id': 'chatcmpl-test',
            'object': 'chat.completion',
            'created': 1,
            'model': openai_params.get('model', 'test-model'),
            'choices': [
                {
                    'index': 0,
                    'message': {'role': 'assistant', 'content': 'hello from host'},
                    'finish_reason': 'stop',
                }
            ],
            'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2},
        }


@pytest.mark.parametrize('api_protocol', ['anthropic', 'osworld-anthropic'])
@pytest.mark.parametrize('string_arguments', [False, True])
def test_messages_stream_delivers_tool_calls_without_api_error(monkeypatch, api_protocol, string_arguments) -> None:
    expected = [
        ('call-bash', 'Bash', {'command': 'echo 你好'}),
        ('call-read', 'Read', {'file_path': '/tmp/example.txt'}),
    ]
    backend_response = {
        'choices': [
            {
                'message': {
                    'role': 'assistant',
                    'content': None,
                    'tool_calls': [
                        {
                            'id': call_id,
                            'type': 'function',
                            'function': {
                                'name': name,
                                'arguments': json.dumps(arguments) if string_arguments else arguments,
                            },
                        }
                        for call_id, name, arguments in expected
                    ],
                },
                'finish_reason': 'tool_calls',
            }
        ],
        'usage': {'prompt_tokens': 10, 'completion_tokens': 5},
    }
    app, bridge = create_bridge_fastapi_app(
        api_protocol=api_protocol,
        thinking_config='hunyuan',
    )
    backend = AsyncMock(return_value=backend_response)
    monkeypatch.setattr(bridge, 'request', backend)

    with TestClient(app) as client:
        with client.stream(
            'POST',
            '/v1/messages',
            json={
                'model': 'test-model',
                'max_tokens': 128,
                'stream': True,
                'messages': [{'role': 'user', 'content': 'Run the tools.'}],
            },
        ) as response:
            assert response.status_code == 200
            assert response.headers['content-type'].startswith('text/event-stream')
            events = [
                json.loads(line[6:])
                for line in response.iter_lines()
                if line.startswith('data: ') and line[6:] != '[DONE]'
            ]

    backend.assert_awaited_once()
    assert not any(event['type'] == 'error' for event in events)
    assert events[0]['type'] == 'message_start'
    assert events[-1]['type'] == 'message_stop'
    starts = [event for event in events if event['type'] == 'content_block_start']
    assert [(event['content_block']['id'], event['content_block']['name']) for event in starts] == [
        (call_id, name) for call_id, name, _ in expected
    ]
    for start, (_, _, arguments) in zip(starts, expected, strict=True):
        assert start['content_block']['type'] == 'tool_use'
        fragments = [
            event['delta']['partial_json']
            for event in events
            if event['type'] == 'content_block_delta' and event['index'] == start['index']
        ]
        assert json.loads(''.join(fragments)) == arguments
        assert any(event['type'] == 'content_block_stop' and event['index'] == start['index'] for event in events)
    assert [event['delta']['stop_reason'] for event in events if event['type'] == 'message_delta'] == ['tool_use']


def test_create_app_initializes_bridge_and_exposes_root_health() -> None:
    proxy = AnthropicProxy(sio=cast(SIOBridge, DummySio()), request_timeout=123)

    app = proxy.create_app()
    client = TestClient(app)

    assert proxy.bridge is not None
    assert client.get('/').json() == {'status': 'ok', 'bridge_ready': True}
    assert client.get('/health').status_code == 404


def test_create_app_messages_uses_existing_bridge() -> None:
    bridge = DummyBridge()
    proxy = AnthropicProxy(sio=cast(SIOBridge, DummySio()), request_timeout=123)
    proxy.bridge = cast(LLMBridge, bridge)

    app = proxy.create_app()
    client = TestClient(app)
    response = client.post(
        '/v1/messages',
        json={
            'model': 'claude-sonnet-4-5',
            'max_tokens': 128,
            'messages': [{'role': 'user', 'content': 'say hello'}],
        },
    )

    assert response.status_code == 200
    assert response.json()['content'][0]['text'] == 'hello from host'
    assert bridge.calls[0]['timeout'] == 123
