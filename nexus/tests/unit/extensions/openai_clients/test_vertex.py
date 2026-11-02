from __future__ import annotations

import asyncio
import json

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import aiohttp
import httpx
import openai
import pytest

from openai.types.chat import ChatCompletion, ChatCompletionChunk

from nexus.core import ContextLengthExceededError, OpenAIConfig, openai_client_registry
from nexus.core.types import ExtendedChatCompletion
from nexus.trace import Span, SpanEndEvent, SpanProcessor, processor_context


_VERTEX_IMPORT_ERROR: ModuleNotFoundError | None = None
try:
    from nexus.extensions.openai_clients.vertex import (
        VertexAnthropicAdapter,
        VertexAnthropicHTTPError,
        VertexClient,
        VertexRequestError,
        _convert_response,
        _read_stream_response,
    )
except ModuleNotFoundError as exc:
    if exc.name != 'nexus.extensions.openai_clients.vertex':
        raise
    _VERTEX_IMPORT_ERROR = exc
    VertexAnthropicAdapter: Any = None
    VertexAnthropicHTTPError: Any = None
    VertexClient: Any = None
    VertexRequestError: Any = None
    _convert_response: Any = None
    _read_stream_response: Any = None


class CapturingProcessor(SpanProcessor):
    def __init__(self) -> None:
        self.ended: list[Span] = []

    def on_end(self, span: Span, _event: SpanEndEvent) -> None:
        self.ended.append(span)


class FakeAsyncStream:
    def __init__(self, chunks: list[ChatCompletionChunk]) -> None:
        self._chunks = iter(chunks)

    def __aiter__(self) -> FakeAsyncStream:
        return self

    async def __anext__(self) -> ChatCompletionChunk:
        try:
            return next(self._chunks)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


class FailingAsyncStream:
    def __init__(self, chunk: ChatCompletionChunk, error: BaseException) -> None:
        self._chunk = chunk
        self._error = error
        self._yielded = False

    def __aiter__(self) -> FailingAsyncStream:
        return self

    async def __anext__(self) -> ChatCompletionChunk:
        if not self._yielded:
            self._yielded = True
            return self._chunk
        raise self._error


class FailingAsyncBytes:
    def __init__(self, lines: list[bytes], error: BaseException) -> None:
        self._lines = iter(lines)
        self._error = error

    def __aiter__(self) -> FailingAsyncBytes:
        return self

    async def __anext__(self) -> bytes:
        try:
            return next(self._lines)
        except StopIteration:
            raise self._error from None


class FakeAiohttpResponse:
    def __init__(self, status: int, body: str, content: Any = None) -> None:
        self.status = status
        self._body = body
        self.content = content

    async def __aenter__(self) -> FakeAiohttpResponse:
        return self

    async def __aexit__(self, *_args: Any) -> None:
        return None

    async def text(self) -> str:
        return self._body


class FakeAiohttpSession:
    def __init__(self, response: FakeAiohttpResponse) -> None:
        self._response = response

    async def __aenter__(self) -> FakeAiohttpSession:
        return self

    async def __aexit__(self, *_args: Any) -> None:
        return None

    def post(self, *_args: Any, **_kwargs: Any) -> FakeAiohttpResponse:
        return self._response


def _require_vertex() -> None:
    assert _VERTEX_IMPORT_ERROR is None, 'unified nexus.extensions.openai_clients.vertex module is missing'


def _make_chat_completion(
    content: str = 'ok',
    model: str = 'vertex-model',
    provider_specific_fields: dict[str, Any] | None = None,
    prompt_tokens: int = 3,
) -> ChatCompletion:
    return ChatCompletion.model_validate({
        'id': 'chatcmpl-vertex',
        'object': 'chat.completion',
        'created': 123,
        'model': model,
        'provider_specific_fields': provider_specific_fields,
        'choices': [
            {
                'index': 0,
                'finish_reason': 'stop',
                'message': {'role': 'assistant', 'content': content},
            }
        ],
        'usage': {'prompt_tokens': prompt_tokens, 'completion_tokens': 2, 'total_tokens': prompt_tokens + 2},
    })


def _make_chat_completion_chunk(
    content: str | None,
    *,
    finish_reason: str | None = None,
    usage: dict[str, int] | None = None,
) -> ChatCompletionChunk:
    return ChatCompletionChunk.model_validate({
        'id': 'chatcmpl-stream',
        'object': 'chat.completion.chunk',
        'created': 123,
        'model': 'vertex-model',
        'choices': [
            {
                'index': 0,
                'finish_reason': finish_reason,
                'delta': {'role': 'assistant' if content == 'hello' else None, 'content': content},
            }
        ],
        'usage': usage,
    })


def _fake_async_openai(create: AsyncMock, init_calls: list[dict[str, Any]] | None = None) -> type:
    class FakeAsyncOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            if init_calls is not None:
                init_calls.append(kwargs)
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))

    return FakeAsyncOpenAI


def _openai_error(kind: str) -> Exception:
    request = httpx.Request('POST', 'https://vertex.test/v1/chat/completions')
    if kind == 'connection':
        return openai.APIConnectionError(message='connection failed', request=request)
    if kind == 'timeout':
        return openai.APITimeoutError(request)

    status = 429 if kind == 'rate_limit' else 500
    response = httpx.Response(status, request=request)
    if kind == 'rate_limit':
        return openai.RateLimitError('rate limited', response=response, body={'error': 'busy'})
    return openai.APIStatusError('server error', response=response, body={'error': 'unavailable'})


def _anthropic_response(content: str = 'ok') -> dict[str, Any]:
    return {
        'id': 'msg_vertex',
        'model': 'claude-opus-4-8',
        'content': [{'type': 'text', 'text': content}],
        'stop_reason': 'end_turn',
        'usage': {'input_tokens': 3, 'output_tokens': 2},
    }


def _sse_lines(events: list[dict[str, Any]]) -> list[bytes]:
    lines: list[bytes] = []
    for event in events:
        lines.append(f'data: {json.dumps(event, separators=(",", ":"))}\n'.encode())
        lines.append(b'\n')
    return lines


def test_vertex_requires_explicit_protocol() -> None:
    _require_vertex()

    with pytest.raises(ValueError, match='protocol'):
        VertexClient(OpenAIConfig(client_args={}, request_args={}))


def test_vertex_rejects_unknown_protocol() -> None:
    _require_vertex()

    with pytest.raises(ValueError, match='anthropic.*openai'):
        VertexClient(OpenAIConfig(client_args={'protocol': 'auto'}, request_args={}))


def test_vertex_is_registered() -> None:
    _require_vertex()

    assert openai_client_registry.get('vertex') is VertexClient


@pytest.mark.asyncio
async def test_openai_protocol_routes_through_sdk_and_returns_extended_completion(monkeypatch) -> None:
    _require_vertex()
    response = _make_chat_completion()
    create = AsyncMock(return_value=response)
    init_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(openai, 'AsyncOpenAI', _fake_async_openai(create, init_calls))

    client = VertexClient(
        OpenAIConfig(
            client_args={
                'protocol': 'openai',
                'api_key': 'secret',
                'base_url': 'https://vertex.test/v1',
                'timeout': 20,
                'max_retries': 4,
                'capture_raw_io': True,
            },
            request_args={'model': 'configured-model', 'temperature': 0.2},
        )
    )

    result = await client._create_completion(
        [{'role': 'user', 'content': 'hi'}],
        model='vertex-model',
        temperature=0.7,
    )

    assert isinstance(result, ExtendedChatCompletion)
    assert result.choices[0].message.content == 'ok'
    assert init_calls == [
        {
            'api_key': 'secret',
            'base_url': 'https://vertex.test/v1',
            'timeout': 20,
            'max_retries': 0,
        }
    ]
    create.assert_awaited_once_with(
        model='vertex-model',
        temperature=0.7,
        messages=[{'role': 'user', 'content': 'hi'}],
    )


@pytest.mark.asyncio
async def test_openai_gateway_routing_records_cache_ids_without_credentials(monkeypatch) -> None:
    _require_vertex()
    create = AsyncMock(return_value=_make_chat_completion())
    init_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(openai, 'AsyncOpenAI', _fake_async_openai(create, init_calls))
    processor = CapturingProcessor()
    client = VertexClient(
        OpenAIConfig(
            client_args={
                'protocol': 'openai',
                'api_key': 'app-id:app-secret?cache_task_id=task-one',
                'base_url': 'https://vertex.test/v1',
                'gateway_routing': {
                    'provider': 'google',
                    'model': 'fable 5',
                    'api': 'openai_api',
                    'timeout': 600,
                },
                'capture_raw_io': True,
            },
            request_args={'model': 'vertex-model'},
        )
    )

    with processor_context([processor]):
        completion = await client.create_completion([{'role': 'user', 'content': 'hi'}])

    routed_key = 'app-id:app-secret?cache_task_id=task-one&provider=google&model=fable+5&api=openai_api&timeout=600'
    assert init_calls == [
        {
            'api_key': routed_key,
            'base_url': 'https://vertex.test/v1',
            'max_retries': 0,
        }
    ]
    assert 'gateway_routing' not in init_calls[0]
    create.assert_awaited_once()
    assert 'gateway_routing' not in create.await_args.kwargs
    provider_call = completion.provider_specific_fields['provider_call']
    span_attributes = processor.ended[-1].attributes
    assert span_attributes['input_cache_task_id'] == 'task-one'
    assert span_attributes['cache_task_id'] == 'task-one'
    evidence = json.dumps({'completion': provider_call, 'span': span_attributes})
    assert 'app-id' not in evidence
    assert 'app-secret' not in evidence
    assert 'gateway_routing' not in evidence
    assert 'cache_task_id' not in json.dumps({
        'completion': provider_call,
        'span': span_attributes['provider_call'],
    })


@pytest.mark.asyncio
async def test_openai_protocol_preserves_non_stream_provider_specific_fields(monkeypatch) -> None:
    _require_vertex()
    response = ChatCompletion.model_validate({
        'id': 'chatcmpl-provider-fields',
        'object': 'chat.completion',
        'created': 123,
        'model': 'vertex-model',
        'provider_specific_fields': {'top_level_vendor': 'top-value'},
        'choices': [
            {
                'index': 0,
                'finish_reason': 'stop',
                'provider_specific_fields': {'choice_vendor': 'choice-value'},
                'message': {
                    'role': 'assistant',
                    'content': 'ok',
                    'thinking_blocks': [{'type': 'thinking', 'thinking': 'reasoning', 'signature': 'vertex-signature'}],
                    'provider_specific_fields': {'message_vendor': 'message-value'},
                },
            }
        ],
        'usage': {
            'prompt_tokens': 5,
            'completion_tokens': 2,
            'total_tokens': 7,
            'cache_creation_input_tokens': 3,
            'cache_read_input_tokens': 2,
        },
    })
    create = AsyncMock(return_value=response)
    monkeypatch.setattr(openai, 'AsyncOpenAI', _fake_async_openai(create))
    client = VertexClient(
        OpenAIConfig(
            client_args={
                'protocol': 'openai',
                'api_key': 'secret',
                'base_url': 'https://vertex.test/v1',
            },
            request_args={'model': 'vertex-model'},
        )
    )

    completion = await client._create_completion([{'role': 'user', 'content': 'preserve fields'}])

    assert completion.provider_specific_fields is not None
    assert completion.provider_specific_fields['top_level_vendor'] == 'top-value'
    assert completion.choices[0].provider_specific_fields == {'choice_vendor': 'choice-value'}
    assert completion.choices[0].message.provider_specific_fields == {'message_vendor': 'message-value'}
    assert completion.choices[0].message.thinking_blocks[0].signature == 'vertex-signature'
    assert completion.usage.cache_creation_input_tokens == 3
    assert completion.usage.cache_read_input_tokens == 2


@pytest.mark.asyncio
async def test_openai_protocol_records_sanitized_raw_io_on_completion_and_span(monkeypatch) -> None:
    _require_vertex()
    create = AsyncMock(return_value=_make_chat_completion(content='recorded'))
    monkeypatch.setattr(openai, 'AsyncOpenAI', _fake_async_openai(create))
    processor = CapturingProcessor()
    client = VertexClient(
        OpenAIConfig(
            client_args={
                'protocol': 'openai',
                'api_key': 'top-secret',
                'base_url': 'https://user:password@vertex.test/v1?api_key=top-secret',
                'capture_raw_io': True,
            },
            request_args={'model': 'vertex-model'},
        )
    )

    with processor_context([processor]):
        completion = await client.create_completion(
            [
                {
                    'role': 'user',
                    'content': 'inspect https://user:password@files.test/item?token=top-secret',
                }
            ],
            extra_headers={'Authorization': 'Bearer other-secret', 'Cookie': 'session=abc'},
        )

    assert completion.provider_specific_fields is not None
    provider_call = completion.provider_specific_fields['provider_call']
    assert provider_call['provider'] == 'vertex'
    assert provider_call['protocol'] == 'openai'
    assert provider_call['model'] == 'vertex-model'
    assert provider_call['attempts'] == [{'attempt': 1, 'status': 'success'}]
    assert provider_call['request']['messages'] == [
        {'role': 'user', 'content': 'inspect https://files.test/item?token=<redacted>'}
    ]
    assert provider_call['response']['id'] == 'chatcmpl-vertex'
    serialized = json.dumps(provider_call)
    for value in ('top-secret', 'other-secret', 'session=abc', 'user:password'):
        assert value not in serialized
    assert provider_call['endpoint'].endswith('/chat/completions?api_key=<redacted>')
    assert processor.ended[-1].attributes['provider_call'] == provider_call


@pytest.mark.asyncio
async def test_raw_io_redacts_credentials_and_preserves_numeric_values_and_data_urls(monkeypatch) -> None:
    _require_vertex()
    create = AsyncMock(return_value=_make_chat_completion())
    monkeypatch.setattr(openai, 'AsyncOpenAI', _fake_async_openai(create))
    screenshot = 'data:image/png;base64,c2NyZWVuc2hvdA=='
    client = VertexClient(
        OpenAIConfig(
            client_args={
                'protocol': 'openai',
                'api_key': 'configured-key',
                'base_url': 'https://vertex.test/v1',
                'capture_raw_io': True,
            },
            request_args={'model': 'vertex-model'},
        )
    )

    completion = await client._create_completion(
        [
            {
                'role': 'user',
                'content': [
                    {
                        'type': 'text',
                        'text': (
                            'inspect HTTPS://user:password@files.test/item?token=url-secret '
                            'and repeat aws-header-secret bearer-secret cookie-secret'
                        ),
                    },
                    {'type': 'image_url', 'image_url': {'url': screenshot}},
                ],
            }
        ],
        model='HTTPS://model-user:model-password@models.test/path?api_key=model-secret',
        extra_headers=httpx.Headers({
            'AWS_SECRET_ACCESS_KEY': 'aws-header-secret',
            'X-Goog-Api-Key': 'header-api-key',
            'Authorization': 'Bearer bearer-secret',
            'Cookie': 'session=cookie-secret',
            'AccessToken': 'header-token',
            'access_tokens': 'plural-access-token',
            'refresh_tokens': 'plural-refresh-token',
            'SessionCookie': 'header-cookie',
            'ClientPassword': 'header-password',
        }),
        max_tokens=123,
    )

    assert completion.provider_specific_fields is not None
    provider_call = completion.provider_specific_fields['provider_call']
    request = provider_call['request']
    serialized = json.dumps(request)
    for value in (
        'user:password',
        'url-secret',
        'header-api-key',
        'header-token',
        'plural-access-token',
        'plural-refresh-token',
        'header-cookie',
        'header-password',
        'aws-header-secret',
        'bearer-secret',
        'cookie-secret',
    ):
        assert value not in serialized
    assert screenshot in serialized
    assert request['max_tokens'] == 123
    assert provider_call['model'] == 'https://models.test/path?api_key=<redacted>'
    assert provider_call['response']['usage']['prompt_tokens'] == 3
    assert provider_call['response']['usage']['completion_tokens'] == 2
    assert provider_call['response']['usage']['total_tokens'] == 5


@pytest.mark.asyncio
async def test_raw_io_redacts_numeric_credentials_and_preserves_explicit_usage_metrics(monkeypatch) -> None:
    _require_vertex()
    access_token = 11
    api_key = 246810
    create = AsyncMock(
        return_value=_make_chat_completion(
            content=f'provider echoed {access_token} and {api_key}',
            provider_specific_fields={
                'debug_value': access_token,
                'prompt_tokens': access_token,
                'usage': {'prompt_tokens': access_token},
            },
            prompt_tokens=access_token,
        )
    )
    monkeypatch.setattr(openai, 'AsyncOpenAI', _fake_async_openai(create))
    processor = CapturingProcessor()
    client = VertexClient(
        OpenAIConfig(
            client_args={
                'protocol': 'openai',
                'api_key': 'configured-key',
                'base_url': 'https://vertex.test/v1',
                'capture_raw_io': True,
            },
            request_args={'model': 'vertex-model'},
        )
    )

    with processor_context([processor]):
        completion = await client.create_completion(
            [{'role': 'user', 'content': f'repeat {access_token} and {api_key}'}],
            extra_query={
                'access_token': access_token,
                'api_key': api_key,
                'prompt_tokens': access_token,
                'usage': {'prompt_tokens': access_token},
            },
        )

    assert completion.choices[0].message.content == f'provider echoed {access_token} and {api_key}'
    assert completion.provider_specific_fields is not None
    assert completion.provider_specific_fields['debug_value'] == access_token
    assert completion.provider_specific_fields['prompt_tokens'] == access_token
    provider_call = completion.provider_specific_fields['provider_call']
    assert provider_call['request']['extra_query'] == {
        'access_token': '<redacted>',
        'api_key': '<redacted>',
        'prompt_tokens': access_token,
        'usage': {'prompt_tokens': access_token},
    }
    assert (
        provider_call['response']['choices'][0]['message']['content'] == f'provider echoed {access_token} and {api_key}'
    )
    assert provider_call['response']['provider_specific_fields']['debug_value'] == access_token
    assert provider_call['response']['provider_specific_fields']['prompt_tokens'] == access_token
    assert provider_call['response']['provider_specific_fields']['usage']['prompt_tokens'] == access_token
    assert provider_call['response']['usage']['prompt_tokens'] == 11
    assert provider_call['response']['usage']['completion_tokens'] == 2
    assert provider_call['response']['usage']['total_tokens'] == 13
    span = processor.ended[-1]
    assert span.attributes['inputs']['kwargs']['extra_query'] == {
        'access_token': access_token,
        'api_key': api_key,
        'prompt_tokens': access_token,
        'usage': {'prompt_tokens': access_token},
    }
    assert (
        span.attributes['outputs']['choices'][0]['message']['content']
        == f'provider echoed {access_token} and {api_key}'
    )
    assert span.attributes['outputs']['provider_specific_fields']['prompt_tokens'] == access_token
    assert span.attributes['outputs']['provider_specific_fields']['usage']['prompt_tokens'] == access_token
    assert span.attributes['outputs']['usage']['prompt_tokens'] == 11


@pytest.mark.asyncio
async def test_raw_io_is_omitted_by_default(monkeypatch) -> None:
    _require_vertex()
    create = AsyncMock(return_value=_make_chat_completion())
    monkeypatch.setattr(openai, 'AsyncOpenAI', _fake_async_openai(create))
    client = VertexClient(
        OpenAIConfig(
            client_args={
                'protocol': 'openai',
                'api_key': 'secret',
                'base_url': 'https://vertex.test/v1',
            },
            request_args={'model': 'vertex-model'},
        )
    )

    completion = await client._create_completion([{'role': 'user', 'content': 'hello'}])

    assert completion.provider_specific_fields is not None
    provider_call = completion.provider_specific_fields['provider_call']
    assert 'request' not in provider_call
    assert 'response' not in provider_call


@pytest.mark.parametrize('error_kind', ['connection', 'timeout', 'rate_limit', 'server'])
@pytest.mark.asyncio
async def test_openai_protocol_retries_only_retryable_sdk_errors(monkeypatch, error_kind: str) -> None:
    _require_vertex()
    create = AsyncMock(side_effect=[_openai_error(error_kind), _make_chat_completion()])
    sleep = AsyncMock()
    monkeypatch.setattr(openai, 'AsyncOpenAI', _fake_async_openai(create))
    monkeypatch.setattr('nexus.extensions.openai_clients.vertex.asyncio.sleep', sleep)
    client = VertexClient(
        OpenAIConfig(
            client_args={
                'protocol': 'openai',
                'api_key': 'secret',
                'base_url': 'https://vertex.test/v1',
                'max_retries': 2,
            },
            request_args={'model': 'vertex-model'},
        )
    )

    completion = await client._create_completion([{'role': 'user', 'content': 'retry'}])

    assert create.await_count == 2
    sleep.assert_awaited_once()
    assert completion.provider_specific_fields is not None
    attempts = completion.provider_specific_fields['provider_call']['attempts']
    assert [attempt['status'] for attempt in attempts] == ['error', 'success']


@pytest.mark.asyncio
async def test_openai_protocol_does_not_retry_ordinary_4xx_and_records_failure(monkeypatch) -> None:
    _require_vertex()
    request = httpx.Request('POST', 'https://vertex.test/v1/chat/completions')
    response = httpx.Response(400, request=request)
    error = openai.BadRequestError(
        'bad api key top-secret',
        response=response,
        body={'error': {'message': 'top-secret rejected'}},
    )
    create = AsyncMock(side_effect=error)
    monkeypatch.setattr(openai, 'AsyncOpenAI', _fake_async_openai(create))
    processor = CapturingProcessor()
    client = VertexClient(
        OpenAIConfig(
            client_args={
                'protocol': 'openai',
                'api_key': 'top-secret',
                'base_url': 'https://vertex.test/v1?credential=top-secret',
                'max_retries': 3,
            },
            request_args={'model': 'vertex-model'},
        )
    )

    with processor_context([processor]), pytest.raises(RuntimeError) as caught:
        await client.create_completion([{'role': 'user', 'content': 'fail'}])

    assert caught.value.__cause__ is error
    assert create.await_count == 1
    provider_call = processor.ended[-1].attributes['provider_call']
    assert len(provider_call['attempts']) == 1
    assert provider_call['attempts'][0]['status_code'] == 400
    serialized = json.dumps(provider_call)
    assert 'top-secret' not in serialized
    assert provider_call['endpoint'].endswith('/chat/completions?credential=<redacted>')


@pytest.mark.asyncio
async def test_openai_provider_error_is_safe_at_observed_boundary_and_keeps_original_as_cause(monkeypatch) -> None:
    _require_vertex()
    request = httpx.Request('POST', 'https://vertex.test/v1/chat/completions')
    response = httpx.Response(400, request=request)
    original = openai.BadRequestError(
        'credential boundary-secret rejected at HTTPS://user:password@vertex.test/v1?token=boundary-secret',
        response=response,
        body={'error': {'X-Goog-Api-Key': 'unconfigured-secret'}},
    )
    create = AsyncMock(side_effect=original)
    monkeypatch.setattr(openai, 'AsyncOpenAI', _fake_async_openai(create))
    processor = CapturingProcessor()
    client = VertexClient(
        OpenAIConfig(
            client_args={
                'protocol': 'openai',
                'api_key': 'boundary-secret',
                'base_url': 'https://vertex.test/v1',
                'max_retries': 1,
            },
            request_args={'model': 'vertex-model'},
        )
    )

    with processor_context([processor]), pytest.raises(RuntimeError) as caught:
        await client.create_completion([{'role': 'user', 'content': 'fail'}])

    assert caught.value.__cause__ is original
    assert getattr(caught.value, 'original_error', None) is None
    assert 'boundary-secret' not in str(caught.value)
    assert 'unconfigured-secret' not in str(caught.value)
    assert 'user:password' not in str(caught.value)
    assert processor.ended[-1].status.message == str(caught.value)
    provider_error = processor.ended[-1].attributes['provider_call']['error']['message']
    assert 'boundary-secret' not in provider_error
    assert 'user:password' not in provider_error


@pytest.mark.asyncio
async def test_openai_error_body_redacts_request_header_value(monkeypatch) -> None:
    _require_vertex()
    request = httpx.Request('POST', 'https://vertex.test/v1/chat/completions')
    response = httpx.Response(400, request=request)
    original = openai.BadRequestError(
        'request rejected',
        response=response,
        body='aws-header-secret rejected',
    )
    create = AsyncMock(side_effect=original)
    monkeypatch.setattr(openai, 'AsyncOpenAI', _fake_async_openai(create))
    processor = CapturingProcessor()
    client = VertexClient(
        OpenAIConfig(
            client_args={
                'protocol': 'openai',
                'api_key': 'configured-key',
                'base_url': 'https://vertex.test/v1',
                'capture_raw_io': True,
                'max_retries': 1,
            },
            request_args={'model': 'vertex-model'},
        )
    )

    with processor_context([processor]), pytest.raises(RuntimeError):
        await client.create_completion(
            [{'role': 'user', 'content': 'fail'}],
            extra_headers=httpx.Headers({'AWS_SECRET_ACCESS_KEY': 'aws-header-secret'}),
        )

    response_capture = processor.ended[-1].attributes['provider_call']['response']
    assert response_capture['body'] == '<redacted> rejected'


@pytest.mark.asyncio
async def test_provider_call_redacts_short_client_defaults_and_configured_request_credentials(
    monkeypatch,
) -> None:
    _require_vertex()
    request = httpx.Request('POST', 'https://vertex.test/v1/chat/completions')
    response = httpx.Response(400, request=request)
    original = openai.BadRequestError(
        'request rejected',
        response=response,
        body={'error': {'message': 'proxy xy query z header q request pw'}},
    )
    create = AsyncMock(side_effect=original)
    monkeypatch.setattr(openai, 'AsyncOpenAI', _fake_async_openai(create))
    processor = CapturingProcessor()
    client = VertexClient(
        OpenAIConfig(
            client_args={
                'protocol': 'openai',
                'api_key': 'configured-api-key',
                'base_url': 'https://vertex.test/v1',
                'capture_raw_io': True,
                'default_headers': {'X-Short-Credential': 'xy'},
                'default_query': {'sig': 'z'},
                'max_retries': 1,
            },
            request_args={
                'model': 'vertex-model',
                'extra_headers': {'Authorization': 'Bearer q'},
                'metadata': {'client_password': 'pw'},
            },
        )
    )

    with processor_context([processor]), pytest.raises(RuntimeError):
        await client.create_completion(
            [{'role': 'user', 'content': 'proxy xy query z header q request pw'}],
        )

    span = processor.ended[-1]
    provider_body = span.attributes['provider_call']['response']['body']['error']['message']
    assert provider_body == 'proxy <redacted> query z header <redacted> request <redacted>'
    assert span.attributes['inputs']['messages'][0]['content'] == 'proxy xy query z header q request pw'


@pytest.mark.asyncio
async def test_provider_call_redacts_composite_api_key_component_in_response(monkeypatch) -> None:
    _require_vertex()
    create = AsyncMock(return_value=_make_chat_completion(content='provider echoed app-secret and innocent-component'))
    monkeypatch.setattr(openai, 'AsyncOpenAI', _fake_async_openai(create))
    processor = CapturingProcessor()
    client = VertexClient(
        OpenAIConfig(
            client_args={
                'protocol': 'openai',
                'api_key': 'app-id:app-secret?provider=vertex',
                'base_url': 'https://vertex.test/v1',
                'capture_raw_io': True,
            },
            request_args={'model': 'vertex-model'},
        )
    )
    message = 'arbitrary public-id:innocent-component?provider=input'

    with processor_context([processor]):
        completion = await client.create_completion([{'role': 'user', 'content': message}])

    assert completion.choices[0].message.content == 'provider echoed app-secret and innocent-component'
    assert completion.provider_specific_fields is not None
    provider_call = completion.provider_specific_fields['provider_call']
    assert provider_call['request']['messages'][0]['content'] == message
    assert provider_call['response']['choices'][0]['message']['content'] == (
        'provider echoed <redacted> and innocent-component'
    )
    span = processor.ended[-1]
    assert span.attributes['inputs']['messages'][0]['content'] == message
    assert span.attributes['outputs']['choices'][0]['message']['content'] == (
        'provider echoed app-secret and innocent-component'
    )


@pytest.mark.asyncio
async def test_provider_call_redacts_malformed_urls_and_path_credentials(monkeypatch) -> None:
    _require_vertex()
    create = AsyncMock(return_value=_make_chat_completion())
    monkeypatch.setattr(openai, 'AsyncOpenAI', _fake_async_openai(create))
    processor = CapturingProcessor()
    malformed_url = 'HTTPS://user:password@[invalid/path/path-secret?token=query-secret'
    client = VertexClient(
        OpenAIConfig(
            client_args={
                'protocol': 'openai',
                'api_key': 'path-secret',
                'base_url': 'https://vertex.test/v1/path-secret',
                'capture_raw_io': True,
            },
            request_args={'model': 'vertex-model'},
        )
    )

    with processor_context([processor]):
        completion = await client.create_completion(
            [{'role': 'user', 'content': f'inspect {malformed_url}'}],
            model=malformed_url,
        )

    assert completion.provider_specific_fields is not None
    provider_call = completion.provider_specific_fields['provider_call']
    assert provider_call['model'] == 'HTTPS://[invalid/path/<redacted>?token=<redacted>'
    assert provider_call['endpoint'] == 'https://vertex.test/v1/<redacted>/chat/completions'
    assert provider_call['request']['messages'][0]['content'] == (
        'inspect HTTPS://[invalid/path/<redacted>?token=<redacted>'
    )
    assert processor.ended[-1].attributes['inputs']['messages'][0]['content'] == f'inspect {malformed_url}'
    serialized = json.dumps(provider_call)
    for secret in ('user:password', 'path-secret', 'query-secret'):
        assert secret not in serialized


@pytest.mark.asyncio
async def test_openai_protocol_converts_context_length_error_without_retry(monkeypatch) -> None:
    _require_vertex()
    request = httpx.Request('POST', 'https://vertex.test/v1/chat/completions')
    response = httpx.Response(400, request=request)
    create = AsyncMock(
        side_effect=openai.BadRequestError(
            'maximum context length exceeded',
            response=response,
            body={'error': {'code': 'context_length_exceeded'}},
        )
    )
    monkeypatch.setattr(openai, 'AsyncOpenAI', _fake_async_openai(create))
    client = VertexClient(
        OpenAIConfig(
            client_args={
                'protocol': 'openai',
                'api_key': 'secret',
                'base_url': 'https://vertex.test/v1',
                'max_retries': 3,
            },
            request_args={'model': 'vertex-model'},
        )
    )

    with pytest.raises(ContextLengthExceededError, match='context length'):
        await client._create_completion([{'role': 'user', 'content': 'too long'}])

    assert create.await_count == 1


@pytest.mark.asyncio
async def test_openai_protocol_converts_structured_context_code_with_generic_message(monkeypatch) -> None:
    _require_vertex()
    request = httpx.Request('POST', 'https://vertex.test/v1/chat/completions')
    response = httpx.Response(400, request=request)
    original = openai.BadRequestError(
        'request rejected',
        response=response,
        body={'error': {'code': 'context_length_exceeded', 'message': 'request rejected'}},
    )
    create = AsyncMock(side_effect=original)
    monkeypatch.setattr(openai, 'AsyncOpenAI', _fake_async_openai(create))
    client = VertexClient(
        OpenAIConfig(
            client_args={
                'protocol': 'openai',
                'api_key': 'secret',
                'base_url': 'https://vertex.test/v1',
                'max_retries': 3,
            },
            request_args={'model': 'vertex-model'},
        )
    )

    with pytest.raises(ContextLengthExceededError, match='context length') as caught:
        await client._create_completion([{'role': 'user', 'content': 'too long'}])

    assert caught.value.__cause__ is original
    assert create.await_count == 1


@pytest.mark.asyncio
async def test_openai_protocol_aggregates_stream_into_extended_completion(monkeypatch) -> None:
    _require_vertex()
    stream = FakeAsyncStream([
        _make_chat_completion_chunk('hello'),
        _make_chat_completion_chunk(
            ' world',
            finish_reason='stop',
            usage={'prompt_tokens': 3, 'completion_tokens': 2, 'total_tokens': 5},
        ),
    ])
    create = AsyncMock(return_value=stream)
    monkeypatch.setattr(openai, 'AsyncOpenAI', _fake_async_openai(create))
    client = VertexClient(
        OpenAIConfig(
            client_args={
                'protocol': 'openai',
                'api_key': 'secret',
                'base_url': 'https://vertex.test/v1',
                'capture_raw_io': True,
            },
            request_args={'model': 'vertex-model', 'stream': True},
        )
    )

    completion = await client._create_completion([{'role': 'user', 'content': 'stream'}])

    assert isinstance(completion, ExtendedChatCompletion)
    assert completion.id == 'chatcmpl-stream'
    assert completion.choices[0].message.content == 'hello world'
    assert completion.choices[0].finish_reason == 'stop'
    assert completion.usage is not None
    assert completion.usage.total_tokens == 5
    assert completion.provider_specific_fields is not None
    assert completion.provider_specific_fields['provider_call']['response']['id'] == 'chatcmpl-stream'


@pytest.mark.asyncio
async def test_openai_stream_preserves_reasoning_logprobs_and_raw_chunk_fields(monkeypatch) -> None:
    _require_vertex()
    chunks = [
        ChatCompletionChunk.model_validate({
            'id': 'chatcmpl-stream-rich',
            'object': 'chat.completion.chunk',
            'created': 123,
            'model': 'vertex-model',
            'provider_specific_fields': {'top_first': 'one'},
            'choices': [
                {
                    'index': 0,
                    'finish_reason': None,
                    'provider_specific_fields': {'choice_first': 'one'},
                    'delta': {
                        'role': 'assistant',
                        'reasoning_content': 'plan ',
                        'reasoning_items': [{'type': 'reasoning', 'id': 'reason-1'}],
                        'provider_specific_fields': {'reasoning_signature': 'sig-1'},
                    },
                    'logprobs': {
                        'content': [
                            {'token': 'a', 'bytes': [97], 'logprob': -0.1, 'top_logprobs': []},
                        ]
                    },
                }
            ],
            'vendor_chunk_field': {'opaque': 'first'},
        }),
        ChatCompletionChunk.model_validate({
            'id': 'chatcmpl-stream-rich',
            'object': 'chat.completion.chunk',
            'created': 123,
            'model': 'vertex-model',
            'provider_specific_fields': {'top_second': 'two'},
            'choices': [
                {
                    'index': 0,
                    'finish_reason': 'stop',
                    'provider_specific_fields': {'choice_second': 'two'},
                    'delta': {
                        'content': 'answer',
                        'reasoning_content': 'done',
                        'reasoning_items': [{'type': 'reasoning', 'id': 'reason-2'}],
                    },
                    'logprobs': {
                        'content': [
                            {'token': 'b', 'bytes': [98], 'logprob': -0.2, 'top_logprobs': []},
                        ]
                    },
                }
            ],
        }),
    ]
    create = AsyncMock(return_value=FakeAsyncStream(chunks))
    monkeypatch.setattr(openai, 'AsyncOpenAI', _fake_async_openai(create))
    client = VertexClient(
        OpenAIConfig(
            client_args={
                'protocol': 'openai',
                'api_key': 'secret',
                'base_url': 'https://vertex.test/v1',
                'capture_raw_io': True,
            },
            request_args={'model': 'vertex-model', 'stream': True},
        )
    )

    completion = await client._create_completion([{'role': 'user', 'content': 'stream'}])

    assert completion.choices[0].message.reasoning_content == 'plan done'
    assert completion.choices[0].message.reasoning_items == [
        {'type': 'reasoning', 'id': 'reason-1'},
        {'type': 'reasoning', 'id': 'reason-2'},
    ]
    assert completion.choices[0].logprobs is not None
    assert [item.token for item in completion.choices[0].logprobs.content or []] == ['a', 'b']
    assert completion.provider_specific_fields is not None
    assert completion.provider_specific_fields['top_first'] == 'one'
    assert completion.provider_specific_fields['top_second'] == 'two'
    assert completion.choices[0].provider_specific_fields == {
        'choice_first': 'one',
        'choice_second': 'two',
    }
    assert completion.choices[0].message.provider_specific_fields == {'reasoning_signature': 'sig-1'}
    response = completion.provider_specific_fields['provider_call']['response']
    assert response['stream_chunks'][0]['vendor_chunk_field'] == {'opaque': 'first'}
    assert response['stream_chunks'][0]['choices'][0]['delta']['provider_specific_fields'] == {
        'reasoning_signature': 'sig-1'
    }
    assert response['stream_chunks'][0]['choices'][0]['logprobs']['content'][0]['token'] == 'a'


@pytest.mark.asyncio
async def test_openai_stream_without_finish_reason_is_recorded_as_incomplete_error(monkeypatch) -> None:
    _require_vertex()
    create = AsyncMock(return_value=FakeAsyncStream([_make_chat_completion_chunk('partial')]))
    monkeypatch.setattr(openai, 'AsyncOpenAI', _fake_async_openai(create))
    processor = CapturingProcessor()
    client = VertexClient(
        OpenAIConfig(
            client_args={
                'protocol': 'openai',
                'api_key': 'secret',
                'base_url': 'https://vertex.test/v1',
                'max_retries': 1,
            },
            request_args={'model': 'vertex-model', 'stream': True},
        )
    )

    with processor_context([processor]), pytest.raises(RuntimeError) as caught:
        await client.create_completion([{'role': 'user', 'content': 'stream'}])

    assert 'partial' not in str(caught.value)
    assert caught.value.__cause__ is not None
    provider_call = processor.ended[-1].attributes['provider_call']
    assert provider_call['attempts'][0]['status'] == 'error'
    assert not any(attempt['status'] == 'success' for attempt in provider_call['attempts'])


@pytest.mark.asyncio
async def test_openai_clean_eof_without_finish_reason_retries_all_configured_attempts(monkeypatch) -> None:
    _require_vertex()
    create = AsyncMock(
        side_effect=[FakeAsyncStream([_make_chat_completion_chunk(f'partial-{attempt}')]) for attempt in range(1, 4)]
    )
    monkeypatch.setattr(openai, 'AsyncOpenAI', _fake_async_openai(create))
    sleep = AsyncMock()
    monkeypatch.setattr('nexus.extensions.openai_clients.vertex.asyncio.sleep', sleep)
    processor = CapturingProcessor()
    client = VertexClient(
        OpenAIConfig(
            client_args={
                'protocol': 'openai',
                'api_key': 'secret',
                'base_url': 'https://vertex.test/v1',
                'max_retries': 3,
            },
            request_args={'model': 'vertex-model', 'stream': True},
        )
    )

    with processor_context([processor]), pytest.raises(VertexRequestError):
        await client.create_completion([{'role': 'user', 'content': 'stream'}])

    assert create.await_count == 3
    assert sleep.await_count == 2
    provider_call = processor.ended[-1].attributes['provider_call']
    assert [attempt['error_type'] for attempt in provider_call['attempts']] == [
        '_VertexOpenAIStreamError',
        '_VertexOpenAIStreamError',
        '_VertexOpenAIStreamError',
    ]


@pytest.mark.parametrize(
    ('capture_raw_io', 'expected_count', 'expected_first_sequence'),
    [(False, 20, 5), (True, 25, 0)],
)
@pytest.mark.asyncio
async def test_openai_incomplete_stream_history_is_capture_aware(
    monkeypatch,
    capture_raw_io: bool,
    expected_count: int,
    expected_first_sequence: int,
) -> None:
    _require_vertex()
    create = AsyncMock(return_value=FakeAsyncStream([{'sequence': index} for index in range(25)]))
    monkeypatch.setattr(openai, 'AsyncOpenAI', _fake_async_openai(create))
    processor = CapturingProcessor()
    client = VertexClient(
        OpenAIConfig(
            client_args={
                'protocol': 'openai',
                'api_key': 'secret',
                'base_url': 'https://vertex.test/v1',
                'capture_raw_io': capture_raw_io,
                'max_retries': 1,
            },
            request_args={'model': 'vertex-model', 'stream': True},
        )
    )

    with processor_context([processor]), pytest.raises(VertexRequestError) as caught:
        await client.create_completion([{'role': 'user', 'content': 'stream'}])

    cause = caught.value.__cause__
    assert cause is not None
    history = cause.body['stream_chunks']
    assert len(history) == expected_count
    assert history[0]['sequence'] == expected_first_sequence
    assert history[-1]['sequence'] == 24
    provider_call = processor.ended[-1].attributes['provider_call']
    if capture_raw_io:
        recorded_history = provider_call['response']['body']['stream_chunks']
        assert [chunk['sequence'] for chunk in recorded_history] == list(range(25))
    else:
        assert 'response' not in provider_call


@pytest.mark.asyncio
async def test_openai_stream_read_failure_records_chunks_received_before_error(monkeypatch) -> None:
    _require_vertex()
    stream = FailingAsyncStream(
        _make_chat_completion_chunk('partial-stream-data'),
        httpx.ReadError('stream disconnected'),
    )
    create = AsyncMock(return_value=stream)
    monkeypatch.setattr(openai, 'AsyncOpenAI', _fake_async_openai(create))
    processor = CapturingProcessor()
    client = VertexClient(
        OpenAIConfig(
            client_args={
                'protocol': 'openai',
                'api_key': 'secret',
                'base_url': 'https://vertex.test/v1',
                'capture_raw_io': True,
                'max_retries': 1,
            },
            request_args={'model': 'vertex-model', 'stream': True},
        )
    )

    with processor_context([processor]), pytest.raises(RuntimeError):
        await client.create_completion([{'role': 'user', 'content': 'stream'}])

    provider_call = processor.ended[-1].attributes['provider_call']
    assert provider_call['response']['body']['stream_chunks'][0]['choices'][0]['delta']['content'] == (
        'partial-stream-data'
    )


@pytest.mark.parametrize('read_error_type', [httpx.ReadTimeout, httpx.ReadError, httpx.RemoteProtocolError])
@pytest.mark.asyncio
async def test_openai_stream_retries_httpx_read_failures(monkeypatch, read_error_type: type[Exception]) -> None:
    _require_vertex()
    failing_stream = FailingAsyncStream(
        _make_chat_completion_chunk('discarded partial'),
        read_error_type('stream interrupted'),
    )
    successful_stream = FakeAsyncStream([
        _make_chat_completion_chunk('hello'),
        _make_chat_completion_chunk(' again', finish_reason='stop'),
    ])
    create = AsyncMock(side_effect=[failing_stream, successful_stream])
    sleep = AsyncMock()
    monkeypatch.setattr(openai, 'AsyncOpenAI', _fake_async_openai(create))
    monkeypatch.setattr('nexus.extensions.openai_clients.vertex.asyncio.sleep', sleep)
    client = VertexClient(
        OpenAIConfig(
            client_args={
                'protocol': 'openai',
                'api_key': 'secret',
                'base_url': 'https://vertex.test/v1',
                'capture_raw_io': True,
                'max_retries': 2,
            },
            request_args={'model': 'vertex-model', 'stream': True},
        )
    )

    completion = await client._create_completion([{'role': 'user', 'content': 'stream'}])

    assert completion.choices[0].message.content == 'hello again'
    assert create.await_count == 2
    sleep.assert_awaited_once()
    assert completion.provider_specific_fields is not None
    attempts = completion.provider_specific_fields['provider_call']['attempts']
    assert [attempt['status'] for attempt in attempts] == ['error', 'success']
    assert attempts[0]['response']['body']['stream_chunks'][0]['choices'][0]['delta']['content'] == (
        'discarded partial'
    )


@pytest.mark.asyncio
async def test_openai_stream_cancellation_records_partial_call_and_raw_status(monkeypatch) -> None:
    _require_vertex()
    cancellation = asyncio.CancelledError('cancel-secret interrupted the stream')
    stream = FailingAsyncStream(_make_chat_completion_chunk('safe partial'), cancellation)
    create = AsyncMock(return_value=stream)
    monkeypatch.setattr(openai, 'AsyncOpenAI', _fake_async_openai(create))
    processor = CapturingProcessor()
    client = VertexClient(
        OpenAIConfig(
            client_args={
                'protocol': 'openai',
                'api_key': 'cancel-secret',
                'base_url': 'https://vertex.test/v1',
                'capture_raw_io': True,
            },
            request_args={'model': 'vertex-model', 'stream': True},
        )
    )

    with processor_context([processor]), pytest.raises(asyncio.CancelledError) as caught:
        await client.create_completion([{'role': 'user', 'content': 'stream'}])

    assert caught.value is cancellation
    assert str(caught.value) == 'cancel-secret interrupted the stream'
    span = processor.ended[-1]
    assert span.status.message == 'cancel-secret interrupted the stream'
    provider_call = span.attributes['provider_call']
    assert provider_call['attempts'][0]['error_type'] == 'CancelledError'
    assert (
        provider_call['attempts'][0]['response']['body']['stream_chunks'][0]['choices'][0]['delta']['content']
        == 'safe partial'
    )
    assert 'cancel-secret' not in json.dumps(provider_call)


@pytest.mark.asyncio
async def test_anthropic_protocol_preserves_existing_payload_and_response_behavior(monkeypatch) -> None:
    _require_vertex()
    client = VertexClient(
        OpenAIConfig(
            client_args={
                'protocol': 'anthropic',
                'api_key': 'secret',
                'base_url': 'https://vertex.test',
                'capture_raw_io': True,
            },
            request_args={
                'model': 'claude-opus-4-8',
                'thinking': {'type': 'adaptive'},
                'output_config': {'effort': 'xhigh'},
            },
        )
    )
    assert isinstance(client._adapter, VertexAnthropicAdapter)
    post_once = AsyncMock(
        return_value={
            'id': 'msg_vertex',
            'model': 'claude-opus-4-8',
            'content': [
                {'type': 'thinking', 'thinking': 'Inspect.', 'signature': 'sig-vertex'},
                {'type': 'text', 'text': 'done'},
            ],
            'stop_reason': 'end_turn',
            'usage': {
                'input_tokens': 4,
                'output_tokens': 3,
                'cache_creation_input_tokens': 1,
                'cache_read_input_tokens': 2,
            },
        }
    )
    client._adapter._post_once = post_once

    completion = await client._create_completion([
        {'role': 'user', 'content': 'go'},
        {
            'role': 'assistant',
            'content': 'working',
            'thinking_blocks': [{'type': 'thinking', 'thinking': 'Need inspect.', 'signature': 'sig-prior'}],
        },
    ])

    endpoint, payload = post_once.await_args.args
    assert endpoint == 'https://vertex.test/v1/publishers/anthropic/models/claude-opus-4-8:rawPredict'
    assert payload['thinking'] == {'type': 'adaptive'}
    assert payload['output_config'] == {'effort': 'xhigh'}
    assert payload['max_tokens'] == 128000
    assert payload['messages'][1]['content'][0]['signature'] == 'sig-prior'
    assert completion.choices[0].message.content == 'done'
    assert completion.choices[0].message.thinking_blocks is not None
    assert completion.choices[0].message.thinking_blocks[0].signature == 'sig-vertex'
    assert completion.usage is not None
    assert completion.usage.prompt_tokens == 7
    assert completion.usage.completion_tokens == 3
    assert completion.usage.total_tokens == 10
    assert completion.usage.prompt_tokens_details is not None
    assert completion.usage.prompt_tokens_details.cache_creation_tokens == 1
    assert completion.usage.prompt_tokens_details.cached_tokens == 2
    assert completion.usage.cache_creation_input_tokens == 1
    assert completion.usage.cache_read_input_tokens == 2
    assert completion.provider_specific_fields is not None
    provider_call = completion.provider_specific_fields['provider_call']
    assert provider_call['response']['id'] == 'msg_vertex'


@pytest.mark.parametrize(
    ('stream', 'endpoint_suffix'),
    [(False, ':rawPredict'), (True, ':streamRawPredict')],
)
@pytest.mark.asyncio
async def test_anthropic_default_endpoint_matches_effective_stream_mode(stream: bool, endpoint_suffix: str) -> None:
    _require_vertex()
    client = VertexClient(
        OpenAIConfig(
            client_args={
                'protocol': 'anthropic',
                'api_key': 'secret',
                'base_url': 'https://vertex.test',
            },
            request_args={'model': 'claude-opus-4-8', 'stream': stream},
        )
    )
    post_once = AsyncMock(return_value=_anthropic_response())
    client._adapter._post_once = post_once

    await client.create_completion([{'role': 'user', 'content': 'go'}])

    endpoint, payload = post_once.await_args.args
    assert endpoint.endswith(endpoint_suffix)
    assert payload['stream'] is stream


@pytest.mark.asyncio
async def test_anthropic_explicit_endpoint_is_unchanged_for_streaming() -> None:
    _require_vertex()
    client = VertexClient(
        OpenAIConfig(
            client_args={
                'protocol': 'anthropic',
                'api_key': 'secret',
                'base_url': 'https://vertex.test',
                'endpoint': 'https://gateway.test/custom-predict',
            },
            request_args={'model': 'claude-opus-4-8', 'stream': True},
        )
    )
    post_once = AsyncMock(return_value=_anthropic_response())
    client._adapter._post_once = post_once

    await client.create_completion([{'role': 'user', 'content': 'go'}])

    endpoint, _ = post_once.await_args.args
    assert endpoint == 'https://gateway.test/custom-predict'


@pytest.mark.asyncio
async def test_anthropic_protocol_rejects_non_messages_gateway_envelope() -> None:
    _require_vertex()
    processor = CapturingProcessor()
    client = VertexClient(
        OpenAIConfig(
            client_args={
                'protocol': 'anthropic',
                'api_key': 'secret',
                'base_url': 'https://vertex.test',
                'capture_raw_io': True,
                'max_retries': 1,
            },
            request_args={'model': 'claude-opus-4-8'},
        )
    )
    client._adapter._post_once = AsyncMock(return_value={'Output': {'__type': 'AnthropicStream'}, 'Version': '1.0'})

    with (
        processor_context([processor]),
        pytest.raises(
            VertexRequestError,
            match='Vertex Anthropic request failed',
        ),
    ):
        await client.create_completion([{'role': 'user', 'content': 'go'}])

    provider_call = processor.ended[-1].attributes['provider_call']
    assert provider_call['response']['Output']['__type'] == 'AnthropicStream'
    assert provider_call['error']['type'] == 'ValueError'


@pytest.mark.asyncio
async def test_anthropic_gateway_routing_is_used_for_authorization_and_filtered_from_payload(monkeypatch) -> None:
    _require_vertex()
    post_calls: list[dict[str, Any]] = []
    response = FakeAiohttpResponse(200, json.dumps(_anthropic_response()))

    class CapturingSession(FakeAiohttpSession):
        def post(self, *_args: Any, **kwargs: Any) -> FakeAiohttpResponse:
            post_calls.append(kwargs)
            return self._response

    monkeypatch.setattr(
        'nexus.extensions.openai_clients.vertex.aiohttp.ClientSession',
        lambda **_kwargs: CapturingSession(response),
    )
    processor = CapturingProcessor()
    client = VertexClient(
        OpenAIConfig(
            client_args={
                'protocol': 'anthropic',
                'api_key': 'app-id:app-secret?cache_task_id=task-two',
                'base_url': 'https://vertex.test',
                'gateway_routing': {
                    'provider': 'google',
                    'model': 'fable5',
                    'api': 'claude_api',
                    'timeout': 600,
                },
                'capture_raw_io': True,
            },
            request_args={'model': 'claude-opus-4-8'},
        )
    )

    with processor_context([processor]):
        completion = await client.create_completion([{'role': 'user', 'content': 'go'}])

    assert post_calls[0]['headers']['Authorization'] == (
        'Bearer app-id:app-secret?cache_task_id=task-two&provider=google&model=fable5&api=claude_api&timeout=600'
    )
    assert 'gateway_routing' not in post_calls[0]['json']
    span_attributes = processor.ended[-1].attributes
    assert span_attributes['input_cache_task_id'] == 'task-two'
    assert span_attributes['cache_task_id'] == 'task-two'
    evidence = json.dumps({
        'completion': completion.provider_specific_fields['provider_call'],
        'span': span_attributes,
    })
    assert 'app-id' not in evidence
    assert 'app-secret' not in evidence
    assert 'gateway_routing' not in evidence
    assert 'cache_task_id' not in json.dumps({
        'completion': completion.provider_specific_fields['provider_call'],
        'span': span_attributes['provider_call'],
    })


@pytest.mark.asyncio
async def test_anthropic_normalizes_implicit_thinking_before_dropping_forced_tool_choice() -> None:
    _require_vertex()
    client = VertexClient(
        OpenAIConfig(
            client_args={
                'protocol': 'anthropic',
                'api_key': 'secret',
                'base_url': 'https://vertex.test',
            },
            request_args={
                'model': 'claude-opus-4-8',
                'thinking': {'budget_tokens': 1024},
                'tool_choice': 'required',
                'tools': [
                    {
                        'type': 'function',
                        'function': {
                            'name': 'inspect',
                            'description': 'Inspect the current state.',
                            'parameters': {'type': 'object', 'properties': {}},
                        },
                    }
                ],
            },
        )
    )
    post_once = AsyncMock(return_value=_anthropic_response())
    client._adapter._post_once = post_once

    await client._create_completion([{'role': 'user', 'content': 'inspect'}])

    payload = post_once.await_args.args[1]
    assert payload['thinking'] == {'type': 'adaptive', 'budget_tokens': 1024}
    assert 'tool_choice' not in payload


@pytest.mark.asyncio
async def test_anthropic_per_call_none_disables_default_thinking_and_output_config() -> None:
    _require_vertex()
    client = VertexClient(
        OpenAIConfig(
            client_args={
                'protocol': 'anthropic',
                'api_key': 'secret',
                'base_url': 'https://vertex.test',
            },
            request_args={
                'model': 'claude-opus-4-8',
                'thinking': {'type': 'adaptive'},
                'output_config': {'effort': 'max'},
            },
        )
    )
    post_once = AsyncMock(return_value=_anthropic_response())
    client._adapter._post_once = post_once

    await client._create_completion(
        [{'role': 'user', 'content': 'no thinking'}],
        thinking=None,
        output_config=None,
    )

    payload = post_once.await_args.args[1]
    assert 'thinking' not in payload
    assert 'output_config' not in payload


@pytest.mark.asyncio
async def test_anthropic_stream_interruption_retains_raw_sse_and_partial_response() -> None:
    _require_vertex()
    transport_error = aiohttp.ClientPayloadError('stream transport interrupted')
    response = SimpleNamespace(
        content=FailingAsyncBytes(
            [
                (
                    b'data: {"type":"message_start","message":{"id":"msg-partial","model":"claude-opus-4-8",'
                    b'"content":[]}}\n'
                ),
                b'\n',
                b'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n',
                b'\n',
                (
                    b'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta",'
                    b'"text":"partial stream-secret"}}\n'
                ),
                b'\n',
            ],
            transport_error,
        )
    )

    with pytest.raises(RuntimeError, match='stream read failed') as caught:
        await _read_stream_response(response, 'claude-opus-4-8')

    assert caught.value.__cause__ is transport_error
    body = caught.value.body
    assert body['raw_sse_events'][-1].endswith('"text":"partial stream-secret"}}')
    assert body['partial_response']['content'][0]['text'] == 'partial stream-secret'


@pytest.mark.asyncio
async def test_anthropic_stream_usage_preserves_uncached_created_and_read_token_invariant() -> None:
    _require_vertex()
    response = SimpleNamespace(
        content=FailingAsyncBytes(
            [
                (
                    b'data: {"type":"message_start","message":{"id":"msg-stream","model":"claude-fable-5",'
                    b'"content":[],"usage":{"input_tokens":4,"cache_creation_input_tokens":1,'
                    b'"cache_read_input_tokens":2}}}\n'
                ),
                b'\n',
                b'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n',
                b'\n',
                (b'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"done"}}\n'),
                b'\n',
                b'data: {"type":"content_block_stop","index":0}\n',
                b'\n',
                (b'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":3}}\n'),
                b'\n',
                b'data: {"type":"message_stop"}\n',
                b'\n',
            ],
            StopAsyncIteration(),
        )
    )

    raw_response = await _read_stream_response(response, 'claude-fable-5')
    completion = _convert_response(raw_response, 'claude-fable-5')

    assert completion.usage is not None
    assert completion.usage.prompt_tokens == 7
    assert completion.usage.completion_tokens == 3
    assert completion.usage.total_tokens == 10
    assert completion.usage.cache_creation_input_tokens == 1
    assert completion.usage.cache_read_input_tokens == 2


@pytest.mark.asyncio
async def test_anthropic_clean_eof_without_message_stop_records_incomplete_attempt() -> None:
    _require_vertex()
    response = SimpleNamespace(
        content=FailingAsyncBytes(
            [
                (
                    b'data: {"type":"message_start","message":{"id":"msg-incomplete",'
                    b'"model":"claude-opus-4-8","content":[]}}\n'
                ),
                b'\n',
                b'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n',
                b'\n',
                (
                    b'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta",'
                    b'"text":"partial answer"}}\n'
                ),
                b'\n',
                b'data: {"type":"content_block_stop","index":0}\n',
                b'\n',
            ],
            StopAsyncIteration(),
        )
    )
    processor = CapturingProcessor()
    client = VertexClient(
        OpenAIConfig(
            client_args={
                'protocol': 'anthropic',
                'api_key': 'secret',
                'base_url': 'https://vertex.test',
                'capture_raw_io': True,
                'max_retries': 1,
            },
            request_args={'model': 'claude-opus-4-8', 'stream': True},
        )
    )

    async def post_once(_endpoint: str, _payload: dict[str, Any]) -> dict[str, Any]:
        return await _read_stream_response(response, 'claude-opus-4-8')

    client._adapter._post_once = post_once

    with processor_context([processor]), pytest.raises(VertexRequestError, match='Vertex Anthropic request failed'):
        await client.create_completion([{'role': 'user', 'content': 'stream'}])

    provider_call = processor.ended[-1].attributes['provider_call']
    assert provider_call['attempts'][0]['status'] == 'error'
    assert provider_call['attempts'][0]['error_type'] == '_VertexAnthropicStreamError'
    assert provider_call['error']['type'] == '_VertexAnthropicStreamError'
    evidence = provider_call['attempts'][0]['response']['body']
    assert evidence['raw_sse_events'][-1] == 'data: {"type":"content_block_stop","index":0}'
    assert evidence['partial_response']['content'][0]['text'] == 'partial answer'
    assert 'stop_reason' not in evidence['partial_response']


@pytest.mark.asyncio
async def test_anthropic_clean_eof_without_message_stop_retries_all_configured_attempts(monkeypatch) -> None:
    _require_vertex()

    async def clean_eof(_endpoint: str, _payload: dict[str, Any]) -> dict[str, Any]:
        response = SimpleNamespace(content=FailingAsyncBytes([], StopAsyncIteration()))
        return await _read_stream_response(response, 'claude-opus-4-8')

    post_once = AsyncMock(side_effect=clean_eof)
    sleep = AsyncMock()
    monkeypatch.setattr('nexus.extensions.openai_clients.vertex.asyncio.sleep', sleep)
    processor = CapturingProcessor()
    client = VertexClient(
        OpenAIConfig(
            client_args={
                'protocol': 'anthropic',
                'api_key': 'secret',
                'base_url': 'https://vertex.test',
                'max_retries': 3,
            },
            request_args={'model': 'claude-opus-4-8', 'stream': True},
        )
    )
    client._adapter._post_once = post_once

    with processor_context([processor]), pytest.raises(VertexRequestError):
        await client.create_completion([{'role': 'user', 'content': 'stream'}])

    assert post_once.await_count == 3
    assert sleep.await_count == 2
    provider_call = processor.ended[-1].attributes['provider_call']
    assert [attempt['error_type'] for attempt in provider_call['attempts']] == [
        '_VertexAnthropicStreamError',
        '_VertexAnthropicStreamError',
        '_VertexAnthropicStreamError',
    ]


@pytest.mark.parametrize(
    ('capture_raw_io', 'expected_count', 'expected_first_sequence'),
    [(False, 20, 5), (True, 25, 0)],
)
@pytest.mark.asyncio
async def test_anthropic_incomplete_stream_history_is_capture_aware(
    monkeypatch,
    capture_raw_io: bool,
    expected_count: int,
    expected_first_sequence: int,
) -> None:
    _require_vertex()
    events = [{'type': 'ping', 'sequence': index} for index in range(25)]
    response = FakeAiohttpResponse(
        200,
        '',
        content=FailingAsyncBytes(_sse_lines(events), StopAsyncIteration()),
    )
    monkeypatch.setattr(aiohttp, 'ClientSession', lambda **_kwargs: FakeAiohttpSession(response))
    processor = CapturingProcessor()
    client = VertexClient(
        OpenAIConfig(
            client_args={
                'protocol': 'anthropic',
                'api_key': 'secret',
                'base_url': 'https://vertex.test',
                'capture_raw_io': capture_raw_io,
                'max_retries': 1,
            },
            request_args={'model': 'claude-opus-4-8', 'stream': True},
        )
    )

    with processor_context([processor]), pytest.raises(VertexRequestError) as caught:
        await client.create_completion([{'role': 'user', 'content': 'stream'}])

    cause = caught.value.__cause__
    assert cause is not None
    history = cause.body['raw_sse_events']
    assert len(history) == expected_count
    assert f'"sequence":{expected_first_sequence}' in history[0]
    assert '"sequence":24' in history[-1]
    provider_call = processor.ended[-1].attributes['provider_call']
    if capture_raw_io:
        recorded_history = provider_call['response']['body']['raw_sse_events']
        assert len(recorded_history) == 25
        assert '"sequence":0' in recorded_history[0]
        assert '"sequence":24' in recorded_history[-1]
    else:
        assert 'response' not in provider_call


@pytest.mark.asyncio
async def test_anthropic_retry_keeps_complete_failed_attempt_evidence(monkeypatch) -> None:
    _require_vertex()

    class PartialStreamError(aiohttp.ClientPayloadError):
        def __init__(self) -> None:
            super().__init__('partial stream failed')
            self.body = {
                'raw_sse_events': ['data: proxy credential xy'],
                'partial_response': {'content': [{'type': 'text', 'text': 'proxy credential xy'}]},
            }

    client = VertexClient(
        OpenAIConfig(
            client_args={
                'protocol': 'anthropic',
                'api_key': 'secret',
                'base_url': 'https://vertex.test',
                'capture_raw_io': True,
                'default_headers': {'X-Short-Credential': 'xy'},
                'max_retries': 2,
            },
            request_args={'model': 'claude-opus-4-8', 'stream': True},
        )
    )
    post_once = AsyncMock(side_effect=[PartialStreamError(), _anthropic_response('recovered')])
    sleep = AsyncMock()
    client._adapter._post_once = post_once
    monkeypatch.setattr('nexus.extensions.openai_clients.vertex.asyncio.sleep', sleep)

    completion = await client._create_completion([{'role': 'user', 'content': 'retry stream'}])

    assert completion.provider_specific_fields is not None
    provider_call = completion.provider_specific_fields['provider_call']
    assert provider_call['response']['id'] == 'msg_vertex'
    assert [attempt['status'] for attempt in provider_call['attempts']] == ['error', 'success']
    failed_response = provider_call['attempts'][0]['response']
    assert failed_response['body']['raw_sse_events'] == ['data: proxy credential <redacted>']
    assert failed_response['body']['partial_response']['content'][0]['text'] == 'proxy credential <redacted>'


@pytest.mark.parametrize('status_code', [408, 409, 429, 500])
@pytest.mark.asyncio
async def test_anthropic_protocol_retries_retryable_statuses(monkeypatch, status_code: int) -> None:
    _require_vertex()
    client = VertexClient(
        OpenAIConfig(
            client_args={
                'protocol': 'anthropic',
                'api_key': 'secret',
                'base_url': 'https://vertex.test',
                'max_retries': 2,
            },
            request_args={'model': 'claude-opus-4-8'},
        )
    )
    post_once = AsyncMock(side_effect=[VertexAnthropicHTTPError(status_code, 'retry me'), _anthropic_response()])
    sleep = AsyncMock()
    client._adapter._post_once = post_once
    monkeypatch.setattr('nexus.extensions.openai_clients.vertex.asyncio.sleep', sleep)

    completion = await client._create_completion([{'role': 'user', 'content': 'retry'}])

    assert post_once.await_count == 2
    sleep.assert_awaited_once()
    assert completion.provider_specific_fields is not None
    attempts = completion.provider_specific_fields['provider_call']['attempts']
    assert [attempt['status'] for attempt in attempts] == ['error', 'success']
    assert attempts[0]['status_code'] == status_code


@pytest.mark.asyncio
async def test_anthropic_cancellation_records_partial_call_and_raw_status() -> None:
    _require_vertex()
    cancellation = asyncio.CancelledError('anthropic-cancel-secret interrupted the request')
    processor = CapturingProcessor()
    client = VertexClient(
        OpenAIConfig(
            client_args={
                'protocol': 'anthropic',
                'api_key': 'anthropic-cancel-secret',
                'base_url': 'https://vertex.test',
            },
            request_args={'model': 'claude-opus-4-8'},
        )
    )
    post_once = AsyncMock(side_effect=cancellation)
    client._adapter._post_once = post_once

    with processor_context([processor]), pytest.raises(asyncio.CancelledError) as caught:
        await client.create_completion([{'role': 'user', 'content': 'cancel'}])

    assert caught.value is cancellation
    assert str(caught.value) == 'anthropic-cancel-secret interrupted the request'
    span = processor.ended[-1]
    assert span.status.message == 'anthropic-cancel-secret interrupted the request'
    provider_call = span.attributes['provider_call']
    assert provider_call['attempts'] == [
        {
            'attempt': 1,
            'status': 'error',
            'error_type': 'CancelledError',
            'message': '<redacted> interrupted the request',
        }
    ]
    assert provider_call['error']['type'] == 'CancelledError'
    assert provider_call['error']['message'] == '<redacted> interrupted the request'
    assert 'anthropic-cancel-secret' not in json.dumps(provider_call)
    assert post_once.await_count == 1


@pytest.mark.asyncio
async def test_anthropic_protocol_does_not_retry_ordinary_4xx_and_records_complete_failure(monkeypatch) -> None:
    _require_vertex()
    client = VertexClient(
        OpenAIConfig(
            client_args={
                'protocol': 'anthropic',
                'api_key': 'top-secret',
                'endpoint': 'https://user:password@vertex.test/rawPredict?token=top-secret',
                'max_retries': 3,
            },
            request_args={'model': 'claude-opus-4-8'},
        )
    )
    post_once = AsyncMock(
        side_effect=VertexAnthropicHTTPError(
            400,
            'top-secret rejected at https://user:password@vertex.test/rawPredict?token=top-secret',
        )
    )
    client._adapter._post_once = post_once
    processor = CapturingProcessor()

    with processor_context([processor]), pytest.raises(RuntimeError) as caught:
        await client.create_completion([{'role': 'user', 'content': 'fail'}])

    assert caught.value.__cause__ is post_once.side_effect
    assert post_once.await_count == 1
    provider_call = processor.ended[-1].attributes['provider_call']
    assert provider_call['attempts'][0]['status_code'] == 400
    serialized = json.dumps(provider_call)
    assert 'top-secret' not in serialized
    assert 'user:password' not in serialized
    assert provider_call['endpoint'].endswith('?token=<redacted>')


@pytest.mark.asyncio
async def test_anthropic_protocol_converts_413_to_context_length_error() -> None:
    _require_vertex()
    client = VertexClient(
        OpenAIConfig(
            client_args={
                'protocol': 'anthropic',
                'api_key': 'secret',
                'base_url': 'https://vertex.test',
            },
            request_args={'model': 'claude-opus-4-8'},
        )
    )
    post_once = AsyncMock(side_effect=VertexAnthropicHTTPError(413, 'request too large'))
    client._adapter._post_once = post_once

    with pytest.raises(ContextLengthExceededError, match='context length'):
        await client._create_completion([{'role': 'user', 'content': 'too long'}])

    assert post_once.await_count == 1


@pytest.mark.parametrize(
    ('status_code', 'body', 'expected_error'),
    [
        (
            413,
            'payload boundary-secret at HTTPS://user:password@vertex.test/too-large?token=boundary-secret',
            ContextLengthExceededError,
        ),
        (200, 'not-json boundary-secret at HTTPS://user:password@vertex.test/raw?token=boundary-secret', RuntimeError),
    ],
)
@pytest.mark.asyncio
async def test_anthropic_raw_error_response_is_complete_sanitized_and_recorded(
    monkeypatch,
    status_code: int,
    body: str,
    expected_error: type[Exception],
) -> None:
    _require_vertex()
    oversized_body = f'{body} {"x" * 3000}'
    response = FakeAiohttpResponse(status_code, oversized_body)
    monkeypatch.setattr(
        'nexus.extensions.openai_clients.vertex.aiohttp.ClientSession',
        lambda **_kwargs: FakeAiohttpSession(response),
    )
    processor = CapturingProcessor()
    client = VertexClient(
        OpenAIConfig(
            client_args={
                'protocol': 'anthropic',
                'api_key': 'boundary-secret',
                'base_url': 'https://vertex.test',
                'capture_raw_io': True,
                'max_retries': 1,
            },
            request_args={'model': 'claude-opus-4-8'},
        )
    )

    with processor_context([processor]), pytest.raises(expected_error) as caught:
        await client.create_completion([{'role': 'user', 'content': 'fail'}])

    provider_call = processor.ended[-1].attributes['provider_call']
    captured_body = provider_call['response']['body']
    assert provider_call['response']['status_code'] == status_code
    expected_prefix = (
        'payload <redacted> at https://vertex.test/too-large?token=<redacted>'
        if status_code == 413
        else 'not-json <redacted> at https://vertex.test/raw?token=<redacted>'
    )
    assert captured_body == f'{expected_prefix} {"x" * 3000}'
    assert len(captured_body) > 3000
    assert 'boundary-secret' not in captured_body
    assert 'user:password' not in captured_body
    assert '?' in captured_body
    assert 'boundary-secret' not in str(caught.value)
    assert 'user:password' not in str(caught.value)
    assert caught.value.__cause__ is not None


@pytest.mark.asyncio
async def test_missing_model_records_one_partial_provider_call_before_raising(monkeypatch) -> None:
    _require_vertex()
    create = AsyncMock()
    monkeypatch.setattr(openai, 'AsyncOpenAI', _fake_async_openai(create))
    processor = CapturingProcessor()
    client = VertexClient(
        OpenAIConfig(
            client_args={'protocol': 'openai', 'api_key': 'secret', 'base_url': 'https://vertex.test/v1'},
            request_args={},
        )
    )

    with processor_context([processor]), pytest.raises(RuntimeError) as caught:
        await client.create_completion([{'role': 'user', 'content': 'missing'}])

    assert isinstance(caught.value.__cause__, ValueError)
    assert 'model' not in str(caught.value).lower()
    provider_call = processor.ended[-1].attributes['provider_call']
    assert provider_call['protocol'] == 'openai'
    assert provider_call['model'] == ''
    assert provider_call['attempts'] == []
    create.assert_not_awaited()


@pytest.mark.asyncio
async def test_missing_anthropic_base_url_records_one_partial_provider_call_before_raising() -> None:
    _require_vertex()
    processor = CapturingProcessor()
    client = VertexClient(
        OpenAIConfig(
            client_args={'protocol': 'anthropic', 'api_key': 'secret'},
            request_args={'model': 'claude-opus-4-8'},
        )
    )

    with processor_context([processor]), pytest.raises(RuntimeError) as caught:
        await client.create_completion([{'role': 'user', 'content': 'missing'}])

    assert isinstance(caught.value.__cause__, ValueError)
    assert 'base_url' not in str(caught.value)
    provider_call = processor.ended[-1].attributes['provider_call']
    assert provider_call['protocol'] == 'anthropic'
    assert provider_call['model'] == 'claude-opus-4-8'
    assert provider_call['endpoint'] == ''
    assert provider_call['attempts'] == []


@pytest.mark.asyncio
async def test_anthropic_payload_conversion_failure_records_one_partial_provider_call_before_raising() -> None:
    _require_vertex()
    processor = CapturingProcessor()
    client = VertexClient(
        OpenAIConfig(
            client_args={'protocol': 'anthropic', 'api_key': 'secret', 'base_url': 'https://vertex.test'},
            request_args={
                'model': 'claude-opus-4-8',
                'tools': [
                    {
                        'type': 'computer_20241022',
                        'function': {'name': 'computer', 'parameters': {'display_width_px': 1280}},
                    }
                ],
            },
        )
    )

    with processor_context([processor]), pytest.raises(RuntimeError) as caught:
        await client.create_completion([{'role': 'user', 'content': 'convert'}])

    assert isinstance(caught.value.__cause__, ValueError)
    assert 'computer_20241022' not in str(caught.value)
    provider_call = processor.ended[-1].attributes['provider_call']
    assert provider_call['protocol'] == 'anthropic'
    assert provider_call['model'] == 'claude-opus-4-8'
    assert provider_call['attempts'] == []


@pytest.mark.asyncio
async def test_openai_full_endpoint_derives_exact_sdk_base_and_overrides_conflicting_base_url(monkeypatch) -> None:
    _require_vertex()
    create = AsyncMock(return_value=_make_chat_completion())
    init_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(openai, 'AsyncOpenAI', _fake_async_openai(create, init_calls))
    client = VertexClient(
        OpenAIConfig(
            client_args={
                'protocol': 'openai',
                'api_key': 'secret',
                'base_url': 'https://wrong.test/v1',
                'endpoint': 'https://vertex.test/v1/chat/completions',
            },
            request_args={'model': 'vertex-model'},
        )
    )

    completion = await client._create_completion([{'role': 'user', 'content': 'exact'}])

    assert init_calls == [
        {
            'api_key': 'secret',
            'base_url': 'https://vertex.test/v1/',
            'max_retries': 0,
        }
    ]
    assert completion.provider_specific_fields is not None
    assert completion.provider_specific_fields['provider_call']['endpoint'] == 'https://vertex.test/v1/chat/completions'


@pytest.mark.asyncio
async def test_invalid_openai_full_endpoint_records_partial_call_and_fails_before_network(monkeypatch) -> None:
    _require_vertex()
    create = AsyncMock(return_value=_make_chat_completion())
    monkeypatch.setattr(openai, 'AsyncOpenAI', _fake_async_openai(create))
    processor = CapturingProcessor()
    client = VertexClient(
        OpenAIConfig(
            client_args={
                'protocol': 'openai',
                'api_key': 'secret',
                'endpoint': 'https://vertex.test/v1/responses',
            },
            request_args={'model': 'vertex-model'},
        )
    )

    with processor_context([processor]), pytest.raises(RuntimeError) as caught:
        await client.create_completion([{'role': 'user', 'content': 'invalid'}])

    assert isinstance(caught.value.__cause__, ValueError)
    assert 'responses' not in str(caught.value)
    provider_call = processor.ended[-1].attributes['provider_call']
    assert provider_call['protocol'] == 'openai'
    assert provider_call['attempts'] == []
    create.assert_not_awaited()


@pytest.mark.asyncio
async def test_openai_sdk_construction_error_is_safe_and_records_partial_call(monkeypatch) -> None:
    _require_vertex()
    original = ValueError('SDK rejected HTTPS://user:password@vertex.test/v1?api_key=sdk-secret and aws-header-secret')

    class FailingAsyncOpenAI:
        def __init__(self, **_kwargs: Any) -> None:
            raise original

    monkeypatch.setattr(openai, 'AsyncOpenAI', FailingAsyncOpenAI)
    processor = CapturingProcessor()
    client = VertexClient(
        OpenAIConfig(
            client_args={
                'protocol': 'openai',
                'api_key': 'sdk-secret',
                'base_url': 'https://vertex.test/v1',
            },
            request_args={'model': 'vertex-model'},
        )
    )

    with processor_context([processor]), pytest.raises(RuntimeError) as caught:
        await client.create_completion(
            [{'role': 'user', 'content': 'construct'}],
            extra_headers={'AWS_SECRET_ACCESS_KEY': 'aws-header-secret'},
        )

    assert caught.value.__cause__ is original
    assert 'sdk-secret' not in str(caught.value)
    assert 'aws-header-secret' not in str(caught.value)
    assert processor.ended[-1].status.message == str(caught.value)
    provider_call = processor.ended[-1].attributes['provider_call']
    assert provider_call['attempts'] == []
    assert 'sdk-secret' not in provider_call['error']['message']
    assert 'aws-header-secret' not in provider_call['error']['message']


def test_vertex_anthropic_alias_warns_and_forces_anthropic_protocol() -> None:
    _require_vertex()
    from nexus.extensions.openai_clients.vertex_anthropic import VertexAnthropicClient

    original_config = OpenAIConfig(
        client_args={'api_key': 'secret', 'base_url': 'https://vertex.test'},
        request_args={'model': 'claude-opus-4-8'},
    )

    with pytest.warns(DeprecationWarning, match='vertex_anthropic.*deprecated'):
        client = VertexAnthropicClient(original_config)

    assert isinstance(client, VertexClient)
    assert client.protocol == 'anthropic'
    assert isinstance(client._adapter, VertexAnthropicAdapter)
    assert client.config.client_args['protocol'] == 'anthropic'
    assert 'protocol' not in original_config.client_args
    assert openai_client_registry.get('vertex_anthropic') is VertexAnthropicClient
