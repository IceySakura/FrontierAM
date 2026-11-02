from __future__ import annotations

from copy import deepcopy
from typing import Any

import httpx
import pytest

from nexus.runtime.server.services.anthropic_proxy.models import MessagesResponse
from nexus.runtime.server.services.anthropic_proxy.server import AnthropicProxy


class _Bridge:
    async def request(self, params: dict[str, Any], timeout: float) -> dict[str, Any]:
        del params, timeout
        return {
            'id': 'chatcmpl-wire',
            'object': 'chat.completion',
            'created': 1,
            'model': 'openai-wire',
            'choices': [
                {
                    'index': 0,
                    'finish_reason': 'stop',
                    'message': {'role': 'assistant', 'content': 'WIRE_OK'},
                }
            ],
            'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2},
        }


class _CapturingBridge(_Bridge):
    def __init__(self) -> None:
        self.params: dict[str, Any] | None = None

    async def request(self, params: dict[str, Any], timeout: float) -> dict[str, Any]:
        self.params = deepcopy(params)
        return await super().request(params, timeout)


class _FourRoundNativeBridge:
    def __init__(self, *, refuse_final: bool) -> None:
        self.refuse_final = refuse_final
        self.requests: list[dict[str, Any]] = []

    async def request(self, params: dict[str, Any], timeout: float) -> dict[str, Any]:
        del timeout
        self.requests.append(deepcopy(params))
        index = len(self.requests)
        if index < 4:
            content = [
                {'type': 'thinking', 'thinking': f'think-{index}', 'signature': f'sig-{index}'},
                {
                    'type': 'tool_use',
                    'id': f'toolu-{index}',
                    'name': 'shell',
                    'input': {'command': f'echo round-{index}'},
                },
            ]
            stop_reason = 'tool_use'
            stop_details = None
        elif self.refuse_final:
            content = []
            stop_reason = 'refusal'
            stop_details = {
                'type': 'refusal',
                'category': 'integration',
                'explanation': 'native-four-round-refusal',
            }
        else:
            content = [{'type': 'text', 'text': 'PROXY_FOUR_ROUND_OK'}]
            stop_reason = 'end_turn'
            stop_details = None

        body: dict[str, Any] = {
            'id': f'msg-native-{index}',
            'type': 'message',
            'role': 'assistant',
            'model': 'claude-native',
            'content': content,
            'stop_reason': stop_reason,
            'stop_sequence': None,
            'usage': {'input_tokens': index, 'output_tokens': index},
            'provider_extension': {'round': index},
        }
        if stop_details is not None:
            body['stop_details'] = stop_details
        return {
            'id': f'chatcmpl-wrapped-{index}',
            'provider_specific_fields': {
                'native_response': {'protocol': 'anthropic', 'body': body},
            },
        }


@pytest.mark.integration
async def test_proxy_omits_none_stop_details_without_field_exclude_if(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_model_dump = MessagesResponse.model_dump

    def simulate_pydantic_211_model_dump(self: MessagesResponse, *args: Any, **kwargs: Any) -> dict[str, Any]:
        data = original_model_dump(self, *args, **kwargs)
        if not kwargs.get('exclude_none', False):
            data['stop_details'] = None
        return data

    monkeypatch.setattr(MessagesResponse, 'model_dump', simulate_pydantic_211_model_dump)
    proxy = AnthropicProxy(sio=object(), thinking_config='bedrock-claude')  # type: ignore[arg-type]
    proxy.bridge = _Bridge()  # type: ignore[assignment]
    transport = httpx.ASGITransport(app=proxy.create_app())

    async with httpx.AsyncClient(transport=transport, base_url='http://anthropic-proxy.test') as client:
        response = await client.post(
            '/v1/messages',
            json={
                'model': 'claude-wire',
                'max_tokens': 32,
                'messages': [{'role': 'user', 'content': 'hello'}],
            },
        )

    assert response.status_code == 200
    assert response.json()['content'] == [{'type': 'text', 'text': 'WIRE_OK'}]
    assert 'stop_details' not in response.json()


@pytest.mark.integration
async def test_proxy_forwards_hycode_identity_headers_as_request_meta() -> None:
    bridge = _CapturingBridge()
    proxy = AnthropicProxy(sio=object(), thinking_config='bedrock-claude')  # type: ignore[arg-type]
    proxy.bridge = bridge  # type: ignore[assignment]
    transport = httpx.ASGITransport(app=proxy.create_app())

    async with httpx.AsyncClient(transport=transport, base_url='http://anthropic-proxy.test') as client:
        response = await client.post(
            '/v1/messages',
            headers={
                'X-Claude-Code-Agent-Id': 'agent-child-1',
                'X-Claude-Code-Session-Id': 'session-1',
            },
            json={
                'model': 'claude-wire',
                'max_tokens': 32,
                'messages': [{'role': 'user', 'content': 'hello'}],
            },
        )

    assert response.status_code == 200
    assert bridge.params is not None
    assert bridge.params['request_meta'] == {
        'agent_id': 'agent-child-1',
        'session_id': 'session-1',
    }


@pytest.mark.integration
@pytest.mark.parametrize('refuse_final', [False, True], ids=['end_turn', 'refusal'])
async def test_proxy_preserves_four_native_anthropic_rounds(refuse_final: bool) -> None:
    bridge = _FourRoundNativeBridge(refuse_final=refuse_final)
    proxy = AnthropicProxy(sio=object(), thinking_config='bedrock-claude')  # type: ignore[arg-type]
    proxy.bridge = bridge  # type: ignore[assignment]
    transport = httpx.ASGITransport(app=proxy.create_app())
    messages: list[dict[str, Any]] = [{'role': 'user', 'content': 'Start the native four-round sequence.'}]
    responses: list[dict[str, Any]] = []

    async with httpx.AsyncClient(transport=transport, base_url='http://anthropic-proxy.test') as client:
        for index in range(1, 5):
            response = await client.post(
                '/v1/messages',
                json={
                    'model': 'claude-native',
                    'max_tokens': 128,
                    'thinking': {'type': 'enabled', 'budget_tokens': 64},
                    'messages': messages,
                },
            )
            assert response.status_code == 200
            body = response.json()
            responses.append(body)
            assert body['id'] == f'msg-native-{index}'
            assert body['provider_extension'] == {'round': index}
            if index == 4:
                continue
            thinking, tool_use = body['content']
            assert thinking['signature'] == f'sig-{index}'
            messages.extend([
                {'role': 'assistant', 'content': body['content']},
                {
                    'role': 'user',
                    'content': [
                        {
                            'type': 'tool_result',
                            'tool_use_id': tool_use['id'],
                            'content': [
                                {'type': 'text', 'text': f'round-{index}-done'},
                                {
                                    'type': 'image',
                                    'source': {
                                        'type': 'base64',
                                        'media_type': 'image/png',
                                        'data': f'aW1hZ2Ut{index}',
                                    },
                                },
                            ],
                        }
                    ],
                },
            ])

    assert len(bridge.requests) == 4
    for index, request in enumerate(bridge.requests[1:], start=1):
        assistant_messages = [message for message in request['messages'] if message['role'] == 'assistant']
        assert assistant_messages[-1]['thinking_blocks'][0] == {
            'type': 'thinking',
            'thinking': f'think-{index}',
            'signature': f'sig-{index}',
        }
        assert assistant_messages[-1]['tool_calls'][0]['id'] == f'toolu-{index}'

    terminal = responses[-1]
    if refuse_final:
        assert terminal['content'] == []
        assert terminal['stop_reason'] == 'refusal'
        assert terminal['stop_details']['explanation'] == 'native-four-round-refusal'
    else:
        assert terminal['content'] == [{'type': 'text', 'text': 'PROXY_FOUR_ROUND_OK'}]
        assert terminal['stop_reason'] == 'end_turn'
        assert 'stop_details' not in terminal
