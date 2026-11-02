import json

import pytest

from nexus.extensions.openai_clients.vertex import _convert_response
from nexus.runtime.server.services.anthropic_proxy.converter import openai_to_anthropic
from nexus.runtime.server.services.anthropic_proxy.streaming import simulate_streaming


def _openai_refusal_response(*, stop_details: dict | None = None) -> dict:
    choice: dict = {
        'index': 0,
        'finish_reason': 'stop',
        'message': {
            'role': 'assistant',
            'content': None,
            'refusal': 'I cannot help with that request.',
        },
    }
    if stop_details is not None:
        choice['provider_specific_fields'] = {'stop_details': stop_details}
    return {
        'id': 'chatcmpl-refusal',
        'model': 'claude-opus-4-8',
        'choices': [choice],
        'usage': {'prompt_tokens': 10, 'completion_tokens': 2, 'total_tokens': 12},
    }


def test_openai_refusal_uses_legacy_empty_end_turn_fallback() -> None:
    stop_details = {
        'type': 'refusal',
        'category': 'cyber',
        'explanation': 'I cannot help with that request.',
    }

    response = openai_to_anthropic(
        _openai_refusal_response(stop_details=stop_details),
        model='claude-opus-4-8',
    )

    assert [block.model_dump(exclude_none=True) for block in response.content] == [{'type': 'text', 'text': ''}]
    assert response.stop_reason == 'end_turn'
    assert response.stop_details is None


def test_vertex_anthropic_openai_shape_alone_uses_legacy_fallback() -> None:
    stop_details = {
        'type': 'refusal',
        'category': 'cyber',
        'explanation': 'Request refused by the provider.',
    }
    completion = _convert_response(
        {
            'id': 'msg_vertex_refusal',
            'model': 'claude-opus-4-8',
            'content': [],
            'stop_reason': 'refusal',
            'stop_details': stop_details,
            'usage': {'input_tokens': 10, 'output_tokens': 2},
        },
        model='claude-opus-4-8',
    )

    response = openai_to_anthropic(completion, model='claude-opus-4-8')

    assert response.content[0].model_dump() == {'type': 'text', 'text': ''}
    assert response.stop_reason == 'end_turn'
    assert response.stop_details is None


def test_openai_refusal_does_not_synthesize_stop_details() -> None:
    response = openai_to_anthropic(_openai_refusal_response(), model='gpt-test')

    assert response.stop_reason == 'end_turn'
    assert response.stop_details is None


def test_non_refusal_response_does_not_emit_stop_details() -> None:
    response = openai_to_anthropic(
        {
            'id': 'chatcmpl-ok',
            'choices': [
                {
                    'index': 0,
                    'finish_reason': 'stop',
                    'message': {'role': 'assistant', 'content': 'Done.'},
                }
            ],
            'usage': {},
        },
        model='gpt-test',
    )

    assert response.stop_reason == 'end_turn'
    assert response.stop_details is None
    assert 'stop_details' not in response.model_dump(exclude_none=True)


def test_missing_choices_uses_legacy_empty_anthropic_assistant() -> None:
    response = openai_to_anthropic({'id': 'chatcmpl-empty', 'choices': [], 'usage': {}}, model='gpt-test')

    assert response.content[0].model_dump() == {'type': 'text', 'text': ''}
    assert response.stop_reason == 'end_turn'


def test_contentless_choice_uses_legacy_empty_anthropic_assistant() -> None:
    response = openai_to_anthropic(
        {
            'id': 'chatcmpl-empty-message',
            'choices': [
                {
                    'index': 0,
                    'finish_reason': 'stop',
                    'message': {'role': 'assistant', 'content': None},
                }
            ],
            'usage': {},
        },
        model='gpt-test',
    )

    assert response.content[0].model_dump() == {'type': 'text', 'text': ''}
    assert response.stop_reason == 'end_turn'


@pytest.mark.asyncio
async def test_legacy_streaming_refusal_remains_empty_end_turn() -> None:
    stop_details = {
        'type': 'refusal',
        'category': 'cyber',
        'explanation': 'I cannot help with that request.',
    }
    response = openai_to_anthropic(
        _openai_refusal_response(stop_details=stop_details),
        model='claude-opus-4-8',
    )

    events = [event async for event in simulate_streaming(response)]
    payloads = [
        json.loads(line.removeprefix('data: '))
        for event in events
        for line in event.splitlines()
        if line.startswith('data: {')
    ]

    refusal_text = ''.join(
        payload['delta']['text']
        for payload in payloads
        if payload.get('type') == 'content_block_delta' and payload.get('delta', {}).get('type') == 'text_delta'
    )
    message_delta = next(payload for payload in payloads if payload.get('type') == 'message_delta')
    assert refusal_text == ''
    assert message_delta['delta'] == {
        'stop_reason': 'end_turn',
        'stop_sequence': None,
    }
