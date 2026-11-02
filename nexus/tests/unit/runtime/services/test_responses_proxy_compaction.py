from __future__ import annotations

import json

import httpx
import pytest

from nexus.runtime.server.services.codex_compaction import (
    CODEX_TURN_METADATA_HEADER,
    DEFAULT_CODEX_COMPACTION_PROMPT_PREFIX,
    classify_codex_compaction,
    final_user_message_text,
)
from nexus.runtime.server.services.responses_proxy.converter import ResponsesProxyConversionError, responses_to_openai
from nexus.runtime.server.services.responses_proxy.server import ResponsesProxy
from nexus.runtime.server.services.rsio_llm_bridge import (
    NEXUS_AGENT_COMPACTION_FIELD,
    NEXUS_COMPACTION_DETECTOR_FIELD,
)


DEFAULT_PROMPT = (
    'You are performing a CONTEXT CHECKPOINT COMPACTION. '
    'Create a handoff summary for another LLM that will resume the task.'
)


def _message(text: str, *, role: str = 'user', message_type: str = 'message') -> dict:
    return {
        'type': message_type,
        'role': role,
        'content': [{'type': 'input_text', 'text': text}],
    }


def test_metadata_classifies_native_compaction() -> None:
    evidence = classify_codex_compaction(
        {'input': [_message('ordinary turn')]},
        turn_metadata=json.dumps({'request_kind': 'compaction'}),
    )

    assert evidence is not None
    assert evidence.detector == 'turn_metadata'


def test_explicit_non_compaction_metadata_suppresses_prompt_fallback() -> None:
    evidence = classify_codex_compaction(
        {'input': [_message(DEFAULT_PROMPT)]},
        turn_metadata=json.dumps({'request_kind': 'turn'}),
    )

    assert evidence is None


def test_client_metadata_classifies_native_compaction() -> None:
    evidence = classify_codex_compaction({
        'input': [_message('ordinary turn')],
        'client_metadata': {
            CODEX_TURN_METADATA_HEADER: {'request_kind': 'compaction'},
        },
    })

    assert evidence is not None
    assert evidence.detector == 'client_metadata'


def test_configured_compaction_prompt_is_matched_exactly() -> None:
    custom_prompt = 'Compress this history into a precise handoff.'

    evidence = classify_codex_compaction({'input': [_message(custom_prompt)]}, custom_prompt)

    assert evidence is not None
    assert evidence.detector == 'configured_prompt'
    assert classify_codex_compaction({'input': [_message(DEFAULT_PROMPT)]}, custom_prompt) is None


def test_default_prefix_classifies_compaction() -> None:
    evidence = classify_codex_compaction({'input': [_message(DEFAULT_PROMPT)]})

    assert evidence is not None
    assert evidence.detector == 'default_prompt_prefix'
    assert DEFAULT_PROMPT.startswith(DEFAULT_CODEX_COMPACTION_PROMPT_PREFIX)


def test_only_final_user_prompt_can_classify_legacy_compaction() -> None:
    assert classify_codex_compaction({'input': [_message(DEFAULT_PROMPT), _message('continue')]}) is None
    assert classify_codex_compaction({'input': [_message(DEFAULT_PROMPT, role='assistant')]}) is None


def test_final_user_message_text_supports_responses_and_rendered_chat_messages() -> None:
    response_items = [_message('first'), _message('latest')]
    rendered_messages = (
        {'role': 'assistant', 'content': 'first'},
        {'role': 'user', 'content': [{'type': 'text', 'text': 'latest'}, ' suffix']},
    )

    assert final_user_message_text(response_items, require_responses_message_type=True) == 'latest'
    assert final_user_message_text(rendered_messages, require_responses_message_type=False) == 'latest suffix'


def test_malformed_metadata_falls_back_to_prompt_detection() -> None:
    evidence = classify_codex_compaction(
        {'input': [_message(DEFAULT_PROMPT)]},
        turn_metadata='{not json',
    )

    assert evidence is not None
    assert evidence.detector == 'default_prompt_prefix'


def test_ordinary_or_non_message_request_is_not_compaction() -> None:
    assert classify_codex_compaction({'input': [_message('ordinary task')]}) is None
    assert classify_codex_compaction({'input': [_message(DEFAULT_PROMPT, message_type='function_call')]}) is None
    assert classify_codex_compaction({'input': DEFAULT_PROMPT}) is None


def test_converter_rejects_remote_compaction_v2_trigger() -> None:
    with pytest.raises(ResponsesProxyConversionError, match='remote compaction v2') as error:
        responses_to_openai({'input': [_message('ordinary turn'), {'type': 'compaction_trigger'}]})

    assert error.value.code == 'remote_compaction_v2_unsupported'


class _Bridge:
    def __init__(self) -> None:
        self.requests: list[dict] = []

    async def request(self, params: dict, timeout: float) -> dict:
        assert timeout == 300
        self.requests.append(params)
        if params.get('_nexus_operation') == 'responses_compact':
            return {
                'id': 'response-compact',
                'output': [
                    {
                        'type': 'compaction',
                        'id': 'cmp_1',
                        'encrypted_content': 'opaque-compaction-state',
                    }
                ],
                'usage': {'input_tokens': 12, 'output_tokens': 3, 'total_tokens': 15},
            }
        return {'id': 'response-1', 'choices': [{'message': {'role': 'assistant', 'content': 'handoff'}}]}


@pytest.mark.asyncio
@pytest.mark.parametrize('headers', [{}, {CODEX_TURN_METADATA_HEADER: json.dumps({'request_kind': 'compaction'})}])
async def test_proxy_routes_remote_compaction_v2_before_converter(headers: dict[str, str]) -> None:
    proxy = ResponsesProxy(sio=object(), codex_version='0.147.0')
    bridge = _Bridge()
    proxy.bridge = bridge
    transport = httpx.ASGITransport(app=proxy._create_app())

    async with httpx.AsyncClient(transport=transport, base_url='http://test') as client:
        response = await client.post(
            '/v1/responses',
            headers=headers,
            json={
                'model': 'test',
                'input': [_message('ordinary turn'), {'type': 'compaction_trigger'}],
                'instructions': 'system instructions',
                'max_output_tokens': 123,
            },
        )

    assert response.status_code == 200
    assert bridge.requests == [
        {
            '_nexus_operation': 'responses_compact',
            'request_body': {
                'model': 'test',
                'input': [_message('ordinary turn')],
                'instructions': 'system instructions',
            },
        }
    ]
    assert 'event: response.output_item.done' in response.text
    assert 'opaque-compaction-state' in response.text
    assert 'event: response.completed' in response.text


@pytest.mark.asyncio
async def test_proxy_marks_detected_compaction_before_runtime_bridge() -> None:
    proxy = ResponsesProxy(sio=object(), codex_version='0.147.0')
    bridge = _Bridge()
    proxy.bridge = bridge
    transport = httpx.ASGITransport(app=proxy._create_app())

    async with httpx.AsyncClient(transport=transport, base_url='http://test') as client:
        response = await client.post(
            '/v1/responses',
            headers={CODEX_TURN_METADATA_HEADER: json.dumps({'request_kind': 'compaction'})},
            json={'model': 'test', 'input': [_message('ordinary turn')]},
        )

    assert response.status_code == 200
    assert bridge.requests[0][NEXUS_AGENT_COMPACTION_FIELD] is True
    assert bridge.requests[0][NEXUS_COMPACTION_DETECTOR_FIELD] == 'turn_metadata'


@pytest.mark.asyncio
async def test_proxy_does_not_mark_ordinary_request() -> None:
    proxy = ResponsesProxy(sio=object(), codex_version='0.114.0')
    bridge = _Bridge()
    proxy.bridge = bridge
    transport = httpx.ASGITransport(app=proxy._create_app())

    async with httpx.AsyncClient(transport=transport, base_url='http://test') as client:
        response = await client.post(
            '/v1/responses',
            json={'model': 'test', 'input': [_message('ordinary turn')]},
        )

    assert response.status_code == 200
    assert NEXUS_AGENT_COMPACTION_FIELD not in bridge.requests[0]
    assert NEXUS_COMPACTION_DETECTOR_FIELD not in bridge.requests[0]


@pytest.mark.asyncio
@pytest.mark.parametrize('metadata', [False, True])
async def test_successful_native_compaction_marks_only_its_next_thread_request_as_resume(metadata):
    proxy = ResponsesProxy(sio=object(), codex_version='0.147.0')
    bridge = _Bridge()
    proxy.bridge = bridge
    transport = httpx.ASGITransport(app=proxy._create_app())

    def headers(thread, kind):
        if not metadata:
            return {}
        return {
            CODEX_TURN_METADATA_HEADER: json.dumps({'request_kind': kind, 'session_id': 'root', 'thread_id': thread})
        }

    async with httpx.AsyncClient(transport=transport, base_url='http://test') as client:
        response = await client.post(
            '/v1/responses', json={'input': [_message(DEFAULT_PROMPT)]}, headers=headers('root', 'compaction')
        )
        assert response.status_code == 200
        if metadata:
            response = await client.post(
                '/v1/responses', json={'input': [_message('other thread')]}, headers=headers('child', 'turn')
            )
            assert response.status_code == 200
            assert 'invocation_kind' not in bridge.requests[-1]['request_meta']
        for expected in ['compaction_resume', None]:
            response = await client.post(
                '/v1/responses', json={'input': [_message('resume summary')]}, headers=headers('root', 'turn')
            )
            assert response.status_code == 200
            assert bridge.requests[-1]['request_meta'].get('invocation_kind') == expected


@pytest.mark.asyncio
async def test_failed_summary_does_not_mark_a_later_request_as_resume():
    class FailingSummaryBridge(_Bridge):
        async def request(self, params, timeout=None):
            if params.get(NEXUS_AGENT_COMPACTION_FIELD):
                raise RuntimeError('summary failed')
            return await super().request(params, timeout=timeout)

    proxy = ResponsesProxy(sio=object(), codex_version='0.147.0')
    bridge = FailingSummaryBridge()
    proxy.bridge = bridge
    transport = httpx.ASGITransport(app=proxy._create_app())
    async with httpx.AsyncClient(transport=transport, base_url='http://test') as client:
        response = await client.post('/v1/responses', json={'input': [_message(DEFAULT_PROMPT)]})
        assert response.status_code == 500
        response = await client.post('/v1/responses', json={'input': [_message('ordinary turn')]})
        assert response.status_code == 200
    assert 'invocation_kind' not in bridge.requests[0]['request_meta']
