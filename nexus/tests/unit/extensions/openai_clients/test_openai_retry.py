from __future__ import annotations

import httpx
import openai
import pytest

from tenacity import wait_none

from nexus.core import OpenAIConfig
from nexus.extensions.openai_clients.openai import OpenAIClient
from nexus.utils import common


def _completion():
    return {
        'id': 'chatcmpl_test',
        'object': 'chat.completion',
        'created': 1,
        'model': 'test-model',
        'choices': [
            {
                'index': 0,
                'finish_reason': 'stop',
                'message': {'role': 'assistant', 'content': 'done'},
            }
        ],
    }


def _client(http_client):
    return OpenAIClient(
        OpenAIConfig(
            client_args={
                'api_key': 'test-key',
                'base_url': 'https://provider.test/v1',
                'http_client': http_client,
                'max_retries': 2,
            },
            request_args={'model': 'test-model'},
        )
    )


@pytest.mark.parametrize('malformed', [{'choices': None}, {'choices': []}, {'error': {'message': 'temporary failure'}}])
async def test_empty_chat_response_retries_before_parsing(monkeypatch, malformed):
    monkeypatch.setattr(common, 'wait_exponential_jitter', lambda **_kwargs: wait_none())
    requests = []

    async def handler(request):
        requests.append(request)
        return httpx.Response(200, json=malformed if len(requests) == 1 else _completion())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        response = await _client(http_client).create_completion([{'role': 'user', 'content': 'hello'}])

    assert len(requests) == 2
    assert response.choices[0].message.content == 'done'


async def test_exhausted_empty_chat_responses_preserve_request_context(monkeypatch):
    monkeypatch.setattr(common, 'wait_exponential_jitter', lambda **_kwargs: wait_none())
    requests = []

    async def handler(request):
        requests.append(request)
        return httpx.Response(200, json={'choices': []})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        with pytest.raises(openai.APIError, match='empty choices') as caught:
            await _client(http_client).create_completion([{'role': 'user', 'content': 'hello'}])

    assert len(requests) == 2
    assert caught.value.request.method == 'POST'
    assert caught.value.request.url == requests[-1].url


async def test_structured_timeouts_and_sdk_retry_limit_do_not_multiply_nexus_retries(monkeypatch):
    monkeypatch.setattr(common, 'wait_exponential_jitter', lambda **_kwargs: wait_none())
    requests = []

    async def handler(request):
        requests.append(request)
        return httpx.Response(500, json={'error': {'message': 'temporary failure'}})

    timeout = {'connect': 5.0, 'read': 60.0, 'write': 10.0, 'pool': 5.0}
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        config = OpenAIConfig(
            client_args={
                'api_key': 'test-key',
                'base_url': 'https://provider.test/v1',
                'http_client': http_client,
                'timeout': timeout,
                'max_retries': 2,
                'sdk_max_retries': 0,
            },
            request_args={'model': 'test-model'},
        )
        client = OpenAIClient(config)
        assert client.client.timeout == httpx.Timeout(**timeout)
        assert config.client_args['timeout'] == timeout
        with pytest.raises(openai.InternalServerError):
            await client.create_completion([{'role': 'user', 'content': 'hello'}])

    assert len(requests) == 2
