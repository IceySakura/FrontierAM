from __future__ import annotations

import json

from typing import Any

import httpx
import pytest

from nexus.core import OpenAIConfig
from nexus.extensions.openai_clients.openai import OpenAIClient
from nexus.extensions.openai_clients.openai_responses import OpenAIResponsesClient


def _chat_response() -> dict[str, Any]:
    return {
        'id': 'chatcmpl_test',
        'object': 'chat.completion',
        'created': 1,
        'model': 'forced-model',
        'choices': [
            {
                'index': 0,
                'finish_reason': 'stop',
                'message': {'role': 'assistant', 'content': 'done'},
            }
        ],
    }


def _responses_response() -> dict[str, Any]:
    return {
        'id': 'resp_test',
        'object': 'response',
        'created_at': 1,
        'model': 'forced-model',
        'status': 'completed',
        'output': [
            {
                'id': 'msg_test',
                'type': 'message',
                'status': 'completed',
                'role': 'assistant',
                'content': [{'type': 'output_text', 'text': 'done', 'annotations': []}],
            }
        ],
        'error': None,
        'incomplete_details': None,
        'metadata': {},
        'usage': {
            'input_tokens': 1,
            'input_tokens_details': {'cached_tokens': 0},
            'output_tokens': 1,
            'output_tokens_details': {'reasoning_tokens': 0},
            'total_tokens': 2,
        },
    }


@pytest.mark.asyncio
async def test_chat_http_body_uses_enforced_values_after_per_call_extra_body() -> None:
    captured: list[dict[str, Any]] = []
    captured_headers: list[str | None] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        captured_headers.append(request.headers.get('X-Policy'))
        return httpx.Response(200, json=_chat_response())

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    config_request_args = {
        'model': 'config-model',
        'response_format': {'type': 'text'},
        'extra_body': {'model': 'config-extra-model', 'vendor_mode': 'config'},
    }
    enforced_request_args = {
        'model': 'forced-model',
        'response_format': {'type': 'json_object'},
        'temperature': None,
        'extra_headers': {'X-Policy': 'forced'},
        'extra_body': {'vendor_mode': 'forced'},
    }
    client = OpenAIClient(
        OpenAIConfig(
            client_args={
                'api_key': 'test-key',
                'base_url': 'https://provider.test/v1',
                'http_client': http_client,
                'max_retries': 1,
            },
            request_args=config_request_args,
            enforced_request_args=enforced_request_args,
        )
    )

    await client.create_completion(
        [{'role': 'user', 'content': 'hello'}],
        model='call-model',
        response_format={'type': 'text'},
        extra_headers={'X-Policy': 'call'},
        extra_body={'model': 'per-call-extra-model', 'vendor_mode': 'per-call'},
    )
    await http_client.aclose()

    assert captured[0]['model'] == 'forced-model'
    assert captured[0]['response_format'] == {'type': 'json_object'}
    assert captured[0]['temperature'] is None
    assert captured[0]['vendor_mode'] == 'forced'
    assert 'extra_headers' not in captured[0]
    assert captured_headers == ['forced']
    assert config_request_args['extra_body']['model'] == 'config-extra-model'
    assert enforced_request_args['extra_body'] == {'vendor_mode': 'forced'}


@pytest.mark.asyncio
async def test_chat_http_body_uses_cache_control_messages_and_tools() -> None:
    captured: list[dict[str, Any]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, json=_chat_response())

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = OpenAIClient(
        OpenAIConfig(
            client_args={
                'api_key': 'test-key',
                'base_url': 'https://provider.test/v1',
                'http_client': http_client,
                'max_retries': 1,
                'cache_control_enabled': True,
                'cache_control_strategy': 'claude_code',
            },
            request_args={'model': 'test-model'},
        )
    )
    messages = [{'role': 'user', 'content': 'hello'}]
    tools = [
        {
            'type': 'function',
            'function': {
                'name': 'get_weather',
                'description': 'Get weather',
                'parameters': {'type': 'object', 'properties': {}},
            },
        }
    ]

    await client.create_completion(messages, tools=tools)
    await http_client.aclose()

    assert captured[0]['messages'][0]['content'] == [
        {'type': 'text', 'text': 'hello', 'cache_control': {'type': 'ephemeral'}}
    ]
    assert captured[0]['tools'][0]['cache_control'] == {'type': 'ephemeral'}
    assert messages == [{'role': 'user', 'content': 'hello'}]
    assert 'cache_control' not in tools[0]


@pytest.mark.asyncio
async def test_responses_http_body_uses_enforced_model_input_and_budget() -> None:
    captured: list[dict[str, Any]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, json=_responses_response())

    forced_input = [{'role': 'user', 'content': [{'type': 'input_text', 'text': 'policy input'}]}]
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = OpenAIResponsesClient(
        OpenAIConfig(
            client_args={
                'api_key': 'test-key',
                'base_url': 'https://provider.test/v1',
                'http_client': http_client,
                'max_retries': 1,
            },
            request_args={'model': 'config-model', 'max_output_tokens': 100},
            enforced_request_args={
                'model': 'forced-model',
                'input': forced_input,
                'max_output_tokens': 12,
            },
        )
    )

    await client.create_completion(
        [{'role': 'user', 'content': 'runtime input'}],
        model='call-model',
        max_output_tokens=999,
        extra_body={'model': 'extra-model', 'input': 'extra input'},
    )
    await http_client.aclose()

    assert captured[0]['model'] == 'forced-model'
    assert captured[0]['input'] == forced_input
    assert captured[0]['max_output_tokens'] == 12


@pytest.mark.asyncio
async def test_chat_request_args_override_replaces_configured_request() -> None:
    captured: list[dict[str, Any]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, json=_chat_response())

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = OpenAIClient(
        OpenAIConfig(
            client_args={
                'api_key': 'test-key',
                'base_url': 'https://provider.test/v1',
                'http_client': http_client,
                'max_retries': 1,
            },
            request_args={
                'model': 'config-model',
                'max_tokens': 128000,
                'extra_body': {'thinking': {'type': 'adaptive'}},
            },
        )
    )

    await client.create_completion(
        [{'role': 'user', 'content': 'hello'}],
        _request_args_override={'model': 'no-thinking-model', 'max_tokens': 8000},
    )
    await http_client.aclose()

    assert '_request_args_override' not in captured[0]
    assert captured[0]['model'] == 'no-thinking-model'
    assert captured[0]['max_tokens'] == 8000
    assert 'thinking' not in captured[0]


@pytest.mark.asyncio
async def test_chat_without_override_keeps_configured_request() -> None:
    captured: list[dict[str, Any]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, json=_chat_response())

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = OpenAIClient(
        OpenAIConfig(
            client_args={
                'api_key': 'test-key',
                'base_url': 'https://provider.test/v1',
                'http_client': http_client,
                'max_retries': 1,
            },
            request_args={'model': 'config-model', 'max_tokens': 128000},
        )
    )

    await client.create_completion([{'role': 'user', 'content': 'hello'}])
    await http_client.aclose()

    assert captured[0]['model'] == 'config-model'
    assert captured[0]['max_tokens'] == 128000
