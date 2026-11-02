"""Characterize native Nexus behavior before the FrontierAM policy adapter.

Passing a test named ``gap`` documents behavior the benchmark must guard; it
does not certify that native behavior satisfies the benchmark's contract.
All HTTP is intercepted by httpx.MockTransport, and all credentials are dummy.
"""

from __future__ import annotations

import asyncio
import copy
import json

from typing import Any

import httpx
import openai
import pytest

from tenacity import wait_none

from nexus.core import OpenAIConfig
from nexus.extensions.openai_clients import openai as openai_module
from nexus.runtime.server.services.responses_proxy.converter import openai_to_responses_events
from nexus.utils import common


def completion(
    arguments: Any = '{"command":"printf fixture"}',
    *,
    name: str = 'workspace_exec',
    finish_reason: str = 'tool_calls',
) -> dict[str, Any]:
    return {
        'id': 'chatcmpl-fixture',
        'object': 'chat.completion',
        'created': 1,
        'model': 'fixture',
        'choices': [
            {
                'index': 0,
                'message': {
                    'role': 'assistant',
                    'content': None,
                    'tool_calls': [
                        {
                            'id': 'call_fixture',
                            'type': 'function',
                            'function': {'name': name, 'arguments': arguments},
                        }
                    ],
                },
                'finish_reason': finish_reason,
            }
        ],
        'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2},
    }


def function_items(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [event['item'] for event in events if event.get('item', {}).get('type') == 'function_call']


def test_native_converter_serializes_object_arguments_without_changing_input():
    arguments = {'command': 'printf fixture', 'timeout_seconds': 1}
    payload = completion(arguments)
    original = copy.deepcopy(payload)
    item = function_items(openai_to_responses_events(payload))[0]
    assert isinstance(item['arguments'], str)
    assert json.loads(item['arguments']) == arguments
    assert payload == original


def test_native_converter_generates_missing_call_id():
    payload = completion()
    del payload['choices'][0]['message']['tool_calls'][0]['id']
    item = function_items(openai_to_responses_events(payload))[0]
    assert item['call_id'].startswith('call_')
    assert len(item['call_id']) > len('call_')
    assert 'id' not in payload['choices'][0]['message']['tool_calls'][0]


def test_native_gap_unknown_tool_is_forwarded_for_downstream_resolution():
    items = function_items(openai_to_responses_events(completion(name='unadvertised_tool')))
    assert len(items) == 1
    assert items[0]['name'] == 'unadvertised_tool'


@pytest.mark.parametrize('arguments', ['{"command":"printf fixture"}', '{"command":"unfinished'])
def test_native_gap_length_response_still_emits_tool_and_completed(arguments):
    events = openai_to_responses_events(completion(arguments, finish_reason='length'))
    assert function_items(events)[0]['arguments'] == arguments
    assert events[-1]['type'] == 'response.completed'


def test_native_gap_duplicate_call_ids_are_not_rejected_by_converter():
    payload = completion()
    calls = payload['choices'][0]['message']['tool_calls']
    calls.append(copy.deepcopy(calls[0]))
    items = function_items(openai_to_responses_events(payload))
    assert [item['call_id'] for item in items] == ['call_fixture', 'call_fixture']


def test_native_gap_nonfunction_type_is_not_checked_by_converter():
    payload = completion()
    payload['choices'][0]['message']['tool_calls'][0]['type'] = 'unexpected'
    assert len(function_items(openai_to_responses_events(payload))) == 1


@pytest.mark.parametrize(('status', 'expected'), [(400, False), (401, True), (429, True), (500, True)])
def test_native_retry_is_exception_based_not_aihub_code_policy(status, expected):
    request = httpx.Request('POST', 'http://fixture.invalid/v1/chat/completions')
    response = httpx.Response(status, request=request)
    error = openai.APIStatusError('fixture', response=response, body={'error': {'code': 9999, 'is_retryable': False}})
    assert common._is_retryable_exception(error) is expected


def test_native_retry_does_not_swallow_cancellation():
    assert common._is_retryable_exception(asyncio.CancelledError()) is False


def test_native_optional_retry_deadlines_are_disabled_by_default(monkeypatch):
    monkeypatch.delenv('NEXUS_RETRY_TOTAL_TIMEOUT', raising=False)
    monkeypatch.delenv('NEXUS_CALL_DEADLINE', raising=False)
    assert common._resolve_retry_total_timeout() == 0.0
    assert common._resolve_call_deadline() == 0.0


@pytest.mark.asyncio
async def test_native_openai_default_outer_attempts_and_sdk_setting_are_independent(monkeypatch):
    observed = []

    def record_retry(attempts):
        observed.append(attempts)
        return lambda function: function

    monkeypatch.setattr(openai_module, 'configurable_retry', record_retry)
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=completion()))
    http_client = httpx.AsyncClient(transport=transport)
    client = openai_module.OpenAIClient(
        OpenAIConfig(
            client_args={
                'api_key': 'fixture-not-a-secret',
                'base_url': 'http://fixture.invalid/v1',
                'http_client': http_client,
                'sdk_max_retries': 0,
            },
            request_args={},
        )
    )
    try:
        response = await client._create_completion([{'role': 'user', 'content': 'fixture'}], model='fixture')
        assert response.choices[0].message.tool_calls
        assert observed == [15]
        assert client.client.max_retries == 0
    finally:
        await client.client.close()


@pytest.mark.asyncio
async def test_native_openai_with_sdk_retries_off_retries_500_with_same_payload(monkeypatch):
    bodies = []
    monkeypatch.setattr(common, 'wait_exponential_jitter', lambda **_kwargs: wait_none())
    monkeypatch.delenv('NEXUS_RETRY_TOTAL_TIMEOUT', raising=False)
    monkeypatch.delenv('NEXUS_CALL_DEADLINE', raising=False)

    def respond(request):
        bodies.append(request.content)
        if len(bodies) == 1:
            return httpx.Response(500, json={'error': {'message': 'fixture unavailable', 'code': 3006}})
        return httpx.Response(200, json=completion())

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    client = openai_module.OpenAIClient(
        OpenAIConfig(
            client_args={
                'api_key': 'fixture-not-a-secret',
                'base_url': 'http://fixture.invalid/v1',
                'http_client': http_client,
                'max_retries': 2,
                'sdk_max_retries': 0,
            },
            request_args={},
        )
    )
    try:
        response = await client._create_completion([{'role': 'user', 'content': 'fixture'}], model='fixture')
        assert response.choices[0].message.tool_calls
        assert len(bodies) == 2
        assert bodies[0] == bodies[1]
    finally:
        await client.client.close()
