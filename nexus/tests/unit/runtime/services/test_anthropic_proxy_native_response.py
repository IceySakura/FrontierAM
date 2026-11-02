import json

import httpx
import pytest

from nexus.runtime.server.services.anthropic_proxy.server import AnthropicProxy
from nexus.runtime.server.services.rsio_llm_bridge import LLMRequestError


class _Bridge:
    def __init__(self, response: dict) -> None:
        self.response = response

    async def request(self, params: dict, timeout: float) -> dict:
        return self.response


class _FailingBridge:
    async def request(self, params: dict, timeout: float) -> dict:
        raise LLMRequestError('over the limit', error_code='context_length_exceeded')


def _request_body(*, stream: bool = False) -> dict:
    return {
        'model': 'claude-test',
        'messages': [{'role': 'user', 'content': 'Hello'}],
        'max_tokens': 32,
        'stream': stream,
    }


def _native_body(*, content: list[dict], stop_reason: str | None = 'end_turn') -> dict:
    return {
        'id': 'msg_native',
        'type': 'message',
        'role': 'assistant',
        'model': 'claude-native',
        'content': content,
        'stop_reason': stop_reason,
        'stop_sequence': None,
        'usage': {
            'input_tokens': 7,
            'output_tokens': 3,
            'cache_creation_input_tokens': 1,
            'cache_read_input_tokens': 2,
        },
        'provider_extension': {'preserved': True},
    }


async def _post(proxy: AnthropicProxy, body: dict) -> httpx.Response:
    transport = httpx.ASGITransport(app=proxy.create_app())
    async with httpx.AsyncClient(transport=transport, base_url='http://testserver') as client:
        return await client.post('/v1/messages', json=body)


@pytest.mark.asyncio
async def test_nonstream_native_anthropic_body_is_returned_unchanged() -> None:
    native_body = _native_body(content=[{'type': 'text', 'text': 'Native answer'}])
    proxy = AnthropicProxy(sio=object())  # type: ignore[arg-type]
    proxy.bridge = _Bridge({
        'id': 'chatcmpl-wrapped',
        'provider_specific_fields': {
            'native_response': {'protocol': 'anthropic', 'body': native_body},
        },
    })  # type: ignore[assignment]

    response = await _post(proxy, _request_body())

    assert response.status_code == 200
    assert response.json() == native_body


@pytest.mark.asyncio
async def test_stream_native_empty_refusal_emits_no_synthesized_text() -> None:
    native_body = _native_body(content=[], stop_reason='refusal')
    native_body['stop_details'] = {
        'type': 'refusal',
        'category': 'cyber',
        'explanation': 'Native refusal details must survive streaming.',
    }
    proxy = AnthropicProxy(sio=object())  # type: ignore[arg-type]
    proxy.bridge = _Bridge({
        'provider_specific_fields': {
            'native_response': {'protocol': 'anthropic', 'body': native_body},
        },
    })  # type: ignore[assignment]

    response = await _post(proxy, _request_body(stream=True))
    payloads = [
        json.loads(line.removeprefix('data: ')) for line in response.text.splitlines() if line.startswith('data: {')
    ]

    assert response.status_code == 200
    assert not [payload for payload in payloads if payload.get('type') == 'content_block_delta']
    message_delta = next(payload for payload in payloads if payload.get('type') == 'message_delta')['delta']
    assert message_delta['stop_reason'] == 'refusal'
    assert message_delta['stop_details'] == native_body['stop_details']


@pytest.mark.asyncio
async def test_stream_native_future_fields_are_preserved_without_model_validation() -> None:
    future_block = {
        'type': 'server_tool_use_20990101',
        'id': 'srvtool_1',
        'name': 'future_search',
        'input': {'query': 'Nexus'},
        'provider_extension': {'mode': 'future'},
    }
    native_body = _native_body(content=[future_block], stop_reason='pause_turn')
    native_body['usage']['future_billable_tokens'] = 11
    native_body['terminal_extension'] = {'reason': 'provider_pause'}
    proxy = AnthropicProxy(sio=object())  # type: ignore[arg-type]
    proxy.bridge = _Bridge({
        'provider_specific_fields': {
            'native_response': {'protocol': 'anthropic', 'body': native_body},
        },
    })  # type: ignore[assignment]

    response = await _post(proxy, _request_body(stream=True))
    payloads = [
        json.loads(line.removeprefix('data: ')) for line in response.text.splitlines() if line.startswith('data: {')
    ]

    assert response.status_code == 200
    assert not [payload for payload in payloads if payload.get('type') == 'error']
    message_start = next(payload for payload in payloads if payload.get('type') == 'message_start')['message']
    assert message_start['terminal_extension'] == {'reason': 'provider_pause'}
    block_start = next(payload for payload in payloads if payload.get('type') == 'content_block_start')
    assert block_start['content_block'] == future_block
    message_delta = next(payload for payload in payloads if payload.get('type') == 'message_delta')
    assert message_delta['delta']['stop_reason'] == 'pause_turn'
    assert message_delta['usage']['future_billable_tokens'] == 11


@pytest.mark.asyncio
async def test_stream_native_known_blocks_keep_standard_sse_semantics() -> None:
    native_body = _native_body(
        content=[
            {'type': 'text', 'text': 'Native answer'},
            {'type': 'thinking', 'thinking': 'Inspect first.', 'signature': 'sig-native'},
            {'type': 'redacted_thinking', 'data': 'redacted-native'},
            {'type': 'tool_use', 'id': 'toolu_1', 'name': 'shell', 'input': {'cmd': 'pwd'}},
        ],
        stop_reason='tool_use',
    )
    native_body['stop_sequence'] = 'END'
    native_body['stop_details'] = {'type': 'provider_detail', 'code': 'complete'}
    native_body['usage'] = {'input_tokens': 7, 'output_tokens': 9}
    proxy = AnthropicProxy(sio=object())  # type: ignore[arg-type]
    proxy.bridge = _Bridge({
        'provider_specific_fields': {
            'native_response': {'protocol': 'anthropic', 'body': native_body},
        },
    })  # type: ignore[assignment]

    response = await _post(proxy, _request_body(stream=True))
    payloads = [
        json.loads(line.removeprefix('data: ')) for line in response.text.splitlines() if line.startswith('data: {')
    ]

    message_start = next(payload for payload in payloads if payload.get('type') == 'message_start')['message']
    assert message_start['usage'] == {
        'input_tokens': 7,
        'output_tokens': 0,
        'cache_creation_input_tokens': 0,
        'cache_read_input_tokens': 0,
    }
    block_starts = [payload['content_block'] for payload in payloads if payload.get('type') == 'content_block_start']
    assert block_starts == [
        {'type': 'text', 'text': ''},
        {'type': 'thinking', 'thinking': '', 'signature': ''},
        {'type': 'redacted_thinking', 'data': 'redacted-native'},
        {'type': 'tool_use', 'id': 'toolu_1', 'name': 'shell', 'input': {}},
    ]
    deltas = [payload['delta'] for payload in payloads if payload.get('type') == 'content_block_delta']
    assert {'type': 'text_delta', 'text': 'Native answer'} in deltas
    assert {'type': 'thinking_delta', 'thinking': 'Inspect first.'} in deltas
    assert {'type': 'signature_delta', 'signature': 'sig-native'} in deltas
    assert {'type': 'input_json_delta', 'partial_json': '{"cmd": "pwd"}'} in deltas
    message_delta = next(payload for payload in payloads if payload.get('type') == 'message_delta')
    assert message_delta == {
        'type': 'message_delta',
        'delta': {
            'stop_reason': 'tool_use',
            'stop_sequence': 'END',
            'stop_details': {'type': 'provider_detail', 'code': 'complete'},
        },
        'usage': {'output_tokens': 9},
    }


@pytest.mark.asyncio
async def test_malformed_present_native_envelope_returns_an_error() -> None:
    proxy = AnthropicProxy(sio=object())  # type: ignore[arg-type]
    proxy.bridge = _Bridge({
        'provider_specific_fields': {
            'native_response': {'protocol': 'anthropic', 'body': ['not', 'a', 'dict']},
        },
    })  # type: ignore[assignment]

    response = await _post(proxy, _request_body())

    assert response.status_code == 500
    assert 'native_response.body must be a dict' in response.json()['detail']


@pytest.mark.asyncio
async def test_nonstream_wrong_native_protocol_returns_an_error() -> None:
    proxy = AnthropicProxy(sio=object())  # type: ignore[arg-type]
    proxy.bridge = _Bridge({
        'provider_specific_fields': {
            'native_response': {'protocol': 'openai', 'body': _native_body(content=[])},
        },
    })  # type: ignore[assignment]

    response = await _post(proxy, _request_body())

    assert response.status_code == 500
    assert "native_response.protocol must be 'anthropic'" in response.json()['detail']


@pytest.mark.asyncio
async def test_stream_wrong_native_protocol_yields_an_sse_error() -> None:
    proxy = AnthropicProxy(sio=object())  # type: ignore[arg-type]
    proxy.bridge = _Bridge({
        'provider_specific_fields': {
            'native_response': {'protocol': 'openai', 'body': _native_body(content=[])},
        },
    })  # type: ignore[assignment]

    response = await _post(proxy, _request_body(stream=True))
    error_payload = next(
        json.loads(line.removeprefix('data: ')) for line in response.text.splitlines() if line.startswith('data: {')
    )

    assert response.status_code == 200
    assert error_payload == {
        'type': 'error',
        'error': {
            'type': 'api_error',
            'message': "native_response.protocol must be 'anthropic'",
        },
    }


@pytest.mark.asyncio
async def test_nonstream_coded_bridge_failure_is_a_client_error() -> None:
    proxy = AnthropicProxy(sio=object())  # type: ignore[arg-type]
    proxy.bridge = _FailingBridge()  # type: ignore[assignment]

    response = await _post(proxy, _request_body())

    assert response.status_code == 400
    assert response.json() == {
        'error': {
            'message': '[context_length_exceeded] over the limit',
            'type': 'invalid_request_error',
            'code': 'context_length_exceeded',
        }
    }


@pytest.mark.asyncio
async def test_stream_coded_bridge_failure_yields_non_retryable_sse_error() -> None:
    proxy = AnthropicProxy(sio=object())  # type: ignore[arg-type]
    proxy.bridge = _FailingBridge()  # type: ignore[assignment]

    response = await _post(proxy, _request_body(stream=True))
    error_payload = next(
        json.loads(line.removeprefix('data: ')) for line in response.text.splitlines() if line.startswith('data: {')
    )

    assert response.status_code == 200
    assert error_payload == {
        'type': 'error',
        'error': {
            'type': 'invalid_request_error',
            'message': '[context_length_exceeded] over the limit',
        },
    }


@pytest.mark.asyncio
async def test_missing_native_envelope_uses_openai_compatibility_fallback() -> None:
    proxy = AnthropicProxy(sio=object())  # type: ignore[arg-type]
    proxy.bridge = _Bridge({
        'id': 'chatcmpl-fallback',
        'choices': [
            {
                'index': 0,
                'finish_reason': 'stop',
                'message': {'role': 'assistant', 'content': 'Converted answer'},
            }
        ],
        'usage': {'prompt_tokens': 5, 'completion_tokens': 2},
    })  # type: ignore[assignment]

    response = await _post(proxy, _request_body())

    assert response.status_code == 200
    assert response.json()['content'] == [{'type': 'text', 'text': 'Converted answer'}]
