from __future__ import annotations

import asyncio
import contextvars
import json
import logging

from contextlib import contextmanager
from types import SimpleNamespace

import httpx
import pytest

from nexus.core.openai_client import (
    CONTEXT_LENGTH_EXCEEDED_CODE,
    UNPARSED_TOOL_CALL_FIELD,
    ContextLengthExceededError,
)
from nexus.runtime.server import rpc_replay_guard
from nexus.runtime.server.services import llm_handler
from nexus.runtime.server.services.codex_compaction import CODEX_TURN_METADATA_HEADER
from nexus.runtime.server.services.llm_handler import LLMRequestHandler, sanitize_openai_params
from nexus.runtime.server.services.responses_proxy.server import ResponsesProxy
from nexus.runtime.server.services.rsio_llm_bridge import (
    NEXUS_AGENT_COMPACTION_FIELD,
    NEXUS_COMPACTION_DETECTOR_FIELD,
    RSIOLLMConfig,
)
from nexus.trace import SpanProcessor, processor_context, span_context


class _CompactionTestClient:
    config = SimpleNamespace(request_args={}, client_args={})
    supports_native_responses_compaction = True

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.requests: list[dict] = []

    async def create_completion(self, **kwargs):  # noqa: ANN003, ANN202
        self.requests.append(kwargs)
        if self.fail:
            raise RuntimeError('provider failed')
        return {'choices': [{'message': {'role': 'assistant', 'content': 'summary'}}]}

    async def compact_response(self, request_body: dict) -> dict:
        self.requests.append(request_body)
        return {
            'id': 'resp_compact',
            'output': [{'type': 'compaction', 'id': 'cmp_1', 'encrypted_content': 'opaque-state'}],
        }


class _RecordedSpan:
    def __init__(self, attributes: dict) -> None:
        self.attributes = dict(attributes)

    def update(self, *, attributes: dict) -> None:
        self.attributes.update(attributes)


class _CaptureSpans(SpanProcessor):
    def __init__(self) -> None:
        self.ended = []

    def on_end(self, span, _event) -> None:
        self.ended.append(span)


def _record_span_context(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, _RecordedSpan]]:
    recorded: list[tuple[str, _RecordedSpan]] = []

    @contextmanager
    def fake_span_context(name: str, *, attributes: dict):
        span = _RecordedSpan(attributes)
        recorded.append((name, span))
        yield span

    monkeypatch.setattr(llm_handler, 'span_context', fake_span_context, raising=False)
    return recorded


@pytest.mark.asyncio
async def test_native_responses_compaction_uses_client_compact_method(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded_spans = _record_span_context(monkeypatch)
    client = _CompactionTestClient()
    handler = LLMRequestHandler(runtime=object(), openai_client=client)
    request_body = {'model': 'test', 'input': [{'type': 'message', 'role': 'user', 'content': 'history'}]}

    result = await handler._produce_response(
        'compact-native',
        {
            '_nexus_operation': 'responses_compact',
            'request_body': request_body,
        },
        rsio_mode=False,
    )

    assert result == {
        'id': 'resp_compact',
        'output': [{'type': 'compaction', 'id': 'cmp_1', 'encrypted_content': 'opaque-state'}],
    }
    assert client.requests == [request_body]
    assert recorded_spans[0][0] == 'agent.compaction'
    assert recorded_spans[0][1].attributes == {
        'type': 'agent_compaction',
        'compacted': True,
        'source': 'provider_native',
        'trainable': False,
    }


@pytest.mark.asyncio
async def test_native_responses_compaction_allows_message_items_before_compaction() -> None:
    client = _CompactionTestClient()
    message_items = [
        {
            'type': 'message',
            'id': f'msg_{index}',
            'role': 'assistant',
            'content': [{'type': 'output_text', 'text': f'history-{index}'}],
        }
        for index in range(6)
    ]
    compaction_item = {'type': 'compaction', 'id': 'cmp_1', 'encrypted_content': 'opaque-state'}
    response = {'id': 'resp_compact', 'output': [*message_items, compaction_item]}

    async def compact_response(_request_body: dict) -> dict:
        return response

    client.compact_response = compact_response
    handler = LLMRequestHandler(runtime=object(), openai_client=client)

    result = await handler._produce_response(
        'compact-native-with-history',
        {
            '_nexus_operation': 'responses_compact',
            'request_body': {'model': 'test', 'input': [{'type': 'message', 'role': 'user', 'content': 'history'}]},
        },
        rsio_mode=False,
    )

    assert result == response


@pytest.mark.asyncio
async def test_native_responses_compaction_rejects_malformed_provider_output() -> None:
    client = _CompactionTestClient()

    async def malformed_compact(_request_body: dict) -> dict:
        return {'id': 'resp_compact', 'output': []}

    client.compact_response = malformed_compact
    handler = LLMRequestHandler(runtime=object(), openai_client=client)

    with pytest.raises(ValueError, match='exactly one compaction item'):
        await handler._produce_response(
            'compact-native',
            {'_nexus_operation': 'responses_compact', 'request_body': {'model': 'test', 'input': []}},
            rsio_mode=False,
        )


@pytest.mark.asyncio
async def test_runtime_compaction_marker_is_consumed_and_span_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded_spans = _record_span_context(monkeypatch)
    client = _CompactionTestClient()
    handler = LLMRequestHandler(runtime=object(), openai_client=client)

    await handler._produce_response(
        'compact-success',
        {
            'messages': [{'role': 'user', 'content': 'summary request'}],
            NEXUS_AGENT_COMPACTION_FIELD: True,
            NEXUS_COMPACTION_DETECTOR_FIELD: 'proxy_metadata',
        },
        rsio_mode=False,
    )

    assert client.requests == [{'messages': [{'role': 'user', 'content': 'summary request'}]}]
    assert recorded_spans[0][0] == 'agent.compaction'
    assert recorded_spans[0][1].attributes == {
        'type': 'agent_compaction',
        'compacted': True,
        'source': 'runtime_marker',
        'detector': 'proxy_metadata',
    }


@pytest.mark.asyncio
async def test_registered_handler_preserves_agent_parent_across_socket_task() -> None:
    class Runtime:
        listener = None

        def on(self, _event, listener) -> None:
            self.listener = listener

        def off(self, _event, _listener) -> None:
            self.listener = None

        async def emit(self, _event, _payload) -> None:
            return None

    class TracingClient(_CompactionTestClient):
        async def create_completion(self, **kwargs):  # noqa: ANN003, ANN202
            with span_context('openai_completion', attributes={'type': 'openai_completion'}):
                return await super().create_completion(**kwargs)

    runtime = Runtime()
    handler = LLMRequestHandler(runtime=runtime, openai_client=TracingClient())
    processor = _CaptureSpans()

    with processor_context([processor]):
        socket_context = contextvars.copy_context()
        with span_context('agent.codex', attributes={'type': 'agent'}) as agent_span:
            handler.register()
            assert runtime.listener is not None
            callback_task = socket_context.copy().run(
                asyncio.create_task,
                runtime.listener({
                    'request_id': 'compact-request',
                    'messages': [{'role': 'user', 'content': 'summary request'}],
                    NEXUS_AGENT_COMPACTION_FIELD: True,
                    NEXUS_COMPACTION_DETECTOR_FIELD: 'turn_metadata',
                }),
            )
            await callback_task
            handler.unregister()

    compaction_span = next(span for span in processor.ended if span.name == 'agent.compaction')
    completion_span = next(span for span in processor.ended if span.name == 'openai_completion')
    assert compaction_span.parent_span_id == agent_span.span_id
    assert completion_span.parent_span_id == compaction_span.span_id


@pytest.mark.asyncio
async def test_failed_runtime_compaction_span_remains_unsuccessful(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded_spans = _record_span_context(monkeypatch)
    client = _CompactionTestClient(fail=True)
    handler = LLMRequestHandler(runtime=object(), openai_client=client)

    with pytest.raises(RuntimeError, match='provider failed'):
        await handler._produce_response(
            'compact-failure',
            {
                'messages': [{'role': 'user', 'content': 'summary request'}],
                NEXUS_AGENT_COMPACTION_FIELD: True,
                NEXUS_COMPACTION_DETECTOR_FIELD: 'client_metadata',
            },
            rsio_mode=False,
        )

    assert NEXUS_AGENT_COMPACTION_FIELD not in client.requests[0]
    assert NEXUS_COMPACTION_DETECTOR_FIELD not in client.requests[0]
    assert recorded_spans[0][1].attributes['compacted'] is False


@pytest.mark.asyncio
@pytest.mark.parametrize('invocation_kind', [None, 'compaction_resume'])
async def test_handler_does_not_infer_compaction_from_prompt(
    monkeypatch: pytest.MonkeyPatch, invocation_kind: str | None
) -> None:
    prompt = (
        'You are performing a CONTEXT CHECKPOINT COMPACTION. '
        'Create a handoff summary for another LLM that will resume the task.'
    )
    recorded_spans = _record_span_context(monkeypatch)
    class MetadataClient(_CompactionTestClient):
        supports_request_meta = True

    client = MetadataClient()
    handler = LLMRequestHandler(
        runtime=object(),
        openai_client=client,
    )
    request_meta = {'codex': {'request_kind': 'turn'}}
    if invocation_kind is not None:
        request_meta['invocation_kind'] = invocation_kind
    await handler._produce_response(
        'ordinary-turn',
        {'messages': [{'role': 'user', 'content': prompt}], 'request_meta': request_meta},
        rsio_mode=False,
    )

    assert client.requests[0]['request_meta'] == request_meta
    assert recorded_spans == []


@pytest.mark.asyncio
@pytest.mark.parametrize('metadata_source', ['header', 'body'])
@pytest.mark.parametrize('compact_prompt', [None, 'Compress this history into a precise handoff.'])
@pytest.mark.parametrize('request_kind', ['turn', 'compaction', None])
async def test_handler_preserves_proxy_compaction_classification(
    monkeypatch: pytest.MonkeyPatch,
    metadata_source: str,
    compact_prompt: str | None,
    request_kind: str | None,
) -> None:
    recorded_spans = _record_span_context(monkeypatch)

    class MetadataClient(_CompactionTestClient):
        supports_request_meta = True

    client = MetadataClient()
    handler = LLMRequestHandler(
        runtime=object(),
        openai_client=client,
    )

    class HandlerBridge:
        async def request(self, params: dict, timeout: float) -> dict:
            return await handler._produce_response('proxy-request', params, rsio_mode=False)

    proxy = ResponsesProxy(
        sio=object(), codex_version='0.114.0' if request_kind is None else '0.147.0', compact_prompt=compact_prompt
    )
    proxy.bridge = HandlerBridge()
    prompt = compact_prompt or 'You are performing a CONTEXT CHECKPOINT COMPACTION. Explain this instruction.'
    body = {'model': 'test', 'input': [{'type': 'message', 'role': 'user', 'content': prompt}]}
    headers = {}
    if request_kind is not None:
        metadata = {'request_kind': request_kind}
        if metadata_source == 'header':
            headers[CODEX_TURN_METADATA_HEADER] = json.dumps(metadata)
        else:
            body['client_metadata'] = {CODEX_TURN_METADATA_HEADER: metadata}

    transport = httpx.ASGITransport(app=proxy._create_app())
    async with httpx.AsyncClient(transport=transport, base_url='http://test') as http_client:
        response = await http_client.post('/v1/responses', headers=headers, json=body)

    assert response.status_code == 200
    forwarded_request = client.requests[0]
    assert forwarded_request['request_meta']['codex'].get('request_kind') == request_kind
    expected_invocation_kind = None if request_kind == 'turn' else 'summary_inference'
    assert forwarded_request['request_meta'].get('invocation_kind') == expected_invocation_kind
    assert NEXUS_AGENT_COMPACTION_FIELD not in forwarded_request
    assert NEXUS_COMPACTION_DETECTOR_FIELD not in forwarded_request
    if request_kind == 'turn':
        assert recorded_spans == []
    else:
        assert len(recorded_spans) == 1
        assert recorded_spans[0][0] == 'agent.compaction'
        assert recorded_spans[0][1].attributes['compacted'] is True



def test_sanitize_openai_params_drops_unsupported_fields():
    payload = {
        'model': 'anthropic.claude-sonnet-4-6',
        'messages': [{'role': 'user', 'content': 'hi'}],
        'store': True,
        'metadata': {'source': 'openclaw'},
        'service_tier': 'default',
        'stream_options': {'include_usage': True},
        'cache_retention': 'long',
        'extra_vars': {'mode': 'quick'},
        'tools': [{'type': 'function'}],
    }

    sanitized = sanitize_openai_params(payload)

    assert sanitized == {
        'model': 'anthropic.claude-sonnet-4-6',
        'messages': [{'role': 'user', 'content': 'hi'}],
        'tools': [{'type': 'function'}],
    }
    assert payload['store'] is True


def _handler() -> LLMRequestHandler:
    return LLMRequestHandler(runtime=object(), openai_client=object())


def test_host_request_args_are_defaults_except_pinned_reasoning_effort():
    class FakeRuntime:
        async def emit(self, *_args, **_kwargs):
            return None

    class FakeClient:
        def __init__(self):
            self.config = SimpleNamespace(
                request_args={
                    'model': 'bedrock/anthropic.claude-opus-4-8',
                    'max_tokens': 128000,
                    'reasoning_effort': 'xhigh',
                },
                client_args={},
            )
            self.request = None

        async def create_completion(self, **kwargs):
            self.request = {**self.config.request_args, **kwargs}
            return {'choices': []}

    client = FakeClient()
    handler = LLMRequestHandler(runtime=FakeRuntime(), openai_client=client)

    asyncio.run(
        handler.handle_llm_request(
            {
                'request_id': 'request-1',
                'model': 'claude-default',
                'messages': [{'role': 'user', 'content': 'hi'}],
                'max_tokens': 64000,
                'reasoning_effort': 'medium',
            }
        )
    )

    assert client.request['model'] == 'bedrock/anthropic.claude-opus-4-8'
    assert client.request['max_tokens'] == 64000
    assert client.request['reasoning_effort'] == 'xhigh'


def test_no_thinking_inherits_host_routing_and_drops_thinking_fields():
    class FakeRuntime:
        async def emit(self, *_args, **_kwargs):
            return None

    class FakeClient:
        def __init__(self):
            self.config = SimpleNamespace(
                request_args={
                    'model': 'api_google_claude-opus-5',
                    'custom_llm_provider': 'openai',
                    'max_tokens': 128000,
                    'extra_body': {
                        'thinking': {'type': 'adaptive'},
                        'output_config': {'effort': 'max'},
                    },
                },
                client_args={},
            )
            self.request = None

        async def create_completion(self, **kwargs):
            self.request = kwargs
            return {'choices': []}

    client = FakeClient()
    handler = LLMRequestHandler(
        runtime=FakeRuntime(),
        openai_client=client,
        no_thinking_request_args={
            'model': 'api_google_claude-opus-5',
            'max_tokens': 128000,
        },
        no_thinking_request_args_mode='inherit',
    )

    asyncio.run(
        handler.handle_llm_request(
            {
                'request_id': 'request-no-thinking',
                'model': 'claude-opus-5',
                'messages': [{'role': 'user', 'content': 'hi'}],
                '_client_has_thinking': False,
            }
        )
    )

    assert client.request['_request_args_override'] == {
        'model': 'api_google_claude-opus-5',
        'custom_llm_provider': 'openai',
        'max_tokens': 128000,
    }


def test_no_thinking_replaces_host_request_args_by_default():
    class FakeRuntime:
        async def emit(self, *_args, **_kwargs):
            return None

    class FakeClient:
        def __init__(self):
            self.config = SimpleNamespace(
                request_args={
                    'model': 'host-model',
                    'custom_llm_provider': 'host-provider',
                    'thinking': {'type': 'adaptive'},
                },
                client_args={},
            )
            self.request = None

        async def create_completion(self, **kwargs):
            self.request = kwargs
            return {'choices': []}

    client = FakeClient()
    handler = LLMRequestHandler(
        runtime=FakeRuntime(),
        openai_client=client,
        no_thinking_request_args={'model': 'no-thinking-model', 'max_tokens': 8000},
    )

    asyncio.run(
        handler.handle_llm_request(
            {
                'request_id': 'request-replace',
                'messages': [{'role': 'user', 'content': 'hi'}],
                '_client_has_thinking': False,
            }
        )
    )

    assert client.request['_request_args_override'] == {
        'model': 'no-thinking-model',
        'max_tokens': 8000,
    }


def _response_message(message: dict):
    return {'choices': [{'message': message}]}


def _reasoning_item(**overrides):
    item = {
        'type': 'reasoning',
        'id': 'rs_1',
        'summary': [{'type': 'summary_text', 'text': 'summary'}],
        'encrypted_content': 'encrypted-payload',
        'status': 'completed',
    }
    item.update(overrides)
    return item


def test_preserve_thinking_injects_by_tool_call_id_and_content():
    handler = _handler()
    handler._cache_thinking_from_response(
        _response_message(
            {
                'role': 'assistant',
                'content': 'I will inspect the file.',
                'tool_calls': [{'id': 'call_1', 'function': {'name': 'read'}}],
                'thinking_blocks': [{'type': 'thinking', 'thinking': 'inspect', 'signature': 'sig-1'}],
            }
        )
    )

    messages = [
        {
            'role': 'assistant',
            'content': 'I will inspect the file.',
            'tool_calls': [{'id': 'call_1', 'function': {'name': 'read'}}],
        }
    ]

    assert handler._inject_thinking_blocks(messages) == 1
    assert messages[0]['thinking_blocks'] == [{'type': 'thinking', 'thinking': 'inspect', 'signature': 'sig-1'}]


def test_preserve_thinking_reanchors_ids_after_scaffold_strips_underscores():
    handler = _handler()
    original_id = 'toolu_bdrk_01LjG12rco4YgkPNo6ZNUTPv'
    stripped_id = 'toolubdrk01LjG12rco4YgkPNo6ZNUTPv'
    handler._cache_thinking_from_response(
        _response_message(
            {
                'role': 'assistant',
                'content': None,
                'tool_calls': [{'id': original_id, 'function': {'name': 'exec'}}],
                'thinking_blocks': [{'type': 'thinking', 'thinking': 'inspect', 'signature': 'sig-1'}],
            }
        )
    )
    handler._cache_reasoning_items_from_response(
        _response_message(
            {
                'role': 'assistant',
                'content': None,
                'tool_calls': [{'id': original_id, 'function': {'name': 'exec'}}],
                'reasoning_items': [_reasoning_item()],
            }
        )
    )

    messages = [
        {
            'role': 'assistant',
            'content': None,
            'tool_calls': [{'id': stripped_id, 'function': {'name': 'exec'}}],
        },
        {'role': 'tool', 'tool_call_id': stripped_id, 'content': 'ok'},
    ]
    assert handler._inject_thinking_blocks(messages) == 1
    assert handler._inject_reasoning_items(messages) == 1
    assert messages[0]['thinking_blocks'] == [{'type': 'thinking', 'thinking': 'inspect', 'signature': 'sig-1'}]
    assert messages[0]['reasoning_items'] == [_reasoning_item()]
    assert handler._rewrite_stripped_tool_ids(messages) == 2
    assert messages[0]['tool_calls'][0]['id'] == original_id
    assert messages[1]['tool_call_id'] == original_id


def _cache_tool_call(handler, tool_id):
    handler._cache_thinking_from_response(
        _response_message(
            {
                'role': 'assistant',
                'content': None,
                'tool_calls': [{'id': tool_id, 'function': {'name': 'exec'}}],
                'thinking_blocks': [{'type': 'thinking', 'thinking': 'inspect', 'signature': 'sig-1'}],
            }
        )
    )


def test_preserve_thinking_restores_sanitized_and_truncated_gemini_tool_id():
    handler = _handler()
    original_id = 'call_868dd051180c4cd1bf6cc51e7345__thought__AY89a+19T/mB35Hch=='
    sanitized_id = 'call868dd051180c4cd1bf6cc51e7345thoughtA'
    _cache_tool_call(handler, original_id)

    messages = [
        {'role': 'assistant', 'content': None, 'tool_calls': [{'id': sanitized_id, 'function': {'name': 'exec'}}]},
        {'role': 'tool', 'tool_call_id': sanitized_id, 'content': 'ok'},
    ]

    assert handler._rewrite_stripped_tool_ids(messages) == 2
    assert messages[0]['tool_calls'][0]['id'] == original_id
    assert messages[1]['tool_call_id'] == original_id
    assert handler._inject_thinking_blocks(messages) == 1


def test_preserve_thinking_restores_underscore_stripped_gemini_tool_id_with_base64_chars():
    handler = _handler()
    original_id = 'call_868dd051180c4cd1bf6cc51e7345__thought__AY89a+19T/mB35Hch=='
    _cache_tool_call(handler, original_id)
    messages = [{'role': 'tool', 'tool_call_id': original_id.replace('_', ''), 'content': 'ok'}]

    assert handler._rewrite_stripped_tool_ids(messages) == 1
    assert messages[0]['tool_call_id'] == original_id


def test_preserve_thinking_keeps_exact_and_unknown_tool_ids():
    handler = _handler()
    _cache_tool_call(handler, 'toolu_01AbCdEfGhIjKlMnOp')
    messages = [
        {'role': 'tool', 'tool_call_id': 'toolu_01AbCdEfGhIjKlMnOp', 'content': 'ok'},
        {'role': 'tool', 'tool_call_id': 'toolu_99ZzZzZzZzZzZzZzZz', 'content': 'ok'},
        {'role': 'tool', 'tool_call_id': 'toolu01AbC', 'content': 'ok'},
    ]

    assert handler._rewrite_stripped_tool_ids(messages) == 0


def test_preserve_thinking_leaves_ambiguous_truncated_tool_id_untouched():
    handler = _handler()
    _cache_tool_call(handler, 'call_0123456789abcdef__thought__AAA')
    _cache_tool_call(handler, 'call_0123456789abcdef__thought__BBB')
    messages = [{'role': 'tool', 'tool_call_id': 'call0123456789abcdefthought', 'content': 'ok'}]

    assert handler._rewrite_stripped_tool_ids(messages) == 0
    assert messages[0]['tool_call_id'] == 'call0123456789abcdefthought'


def test_preserve_thinking_skips_tool_call_when_content_differs():
    handler = _handler()
    handler._cache_thinking_from_response(
        _response_message(
            {
                'role': 'assistant',
                'content': 'I will inspect the file.',
                'tool_calls': [{'id': 'call_1', 'function': {'name': 'read'}}],
                'thinking_blocks': [{'type': 'thinking', 'thinking': 'inspect', 'signature': 'sig-1'}],
            }
        )
    )

    messages = [
        {
            'role': 'assistant',
            'content': 'rewritten by scaffold',
            'tool_calls': [{'id': 'call_1', 'function': {'name': 'read'}}],
        }
    ]

    assert handler._inject_thinking_blocks(messages) == 0
    assert 'thinking_blocks' not in messages[0]


def test_preserve_thinking_skips_tool_call_when_tool_call_id_differs():
    handler = _handler()
    handler._cache_thinking_from_response(
        _response_message(
            {
                'role': 'assistant',
                'content': 'I will inspect the file.',
                'tool_calls': [{'id': 'call_1', 'function': {'name': 'read'}}],
                'thinking_blocks': [{'type': 'thinking', 'thinking': 'inspect', 'signature': 'sig-1'}],
            }
        )
    )

    messages = [
        {
            'role': 'assistant',
            'content': 'I will inspect the file.',
            'tool_calls': [{'id': 'call_2', 'function': {'name': 'read'}}],
        }
    ]

    assert handler._inject_thinking_blocks(messages) == 0
    assert 'thinking_blocks' not in messages[0]


def test_preserve_thinking_injects_tool_call_when_both_contents_are_missing():
    handler = _handler()
    handler._cache_thinking_from_response(
        _response_message(
            {
                'role': 'assistant',
                'content': '',
                'tool_calls': [{'id': 'call_1', 'function': {'name': 'read'}}],
                'thinking_blocks': [{'type': 'thinking', 'thinking': 'inspect', 'signature': 'sig-1'}],
            }
        )
    )

    messages = [
        {
            'role': 'assistant',
            'tool_calls': [{'id': 'call_1', 'function': {'name': 'read'}}],
        }
    ]

    assert handler._inject_thinking_blocks(messages) == 1
    assert messages[0]['thinking_blocks'] == [{'type': 'thinking', 'thinking': 'inspect', 'signature': 'sig-1'}]


def test_preserve_thinking_injects_by_unique_plaintext_content():
    handler = _handler()
    handler._cache_thinking_from_response(
        _response_message(
            {
                'role': 'assistant',
                'content': 'Plain assistant text.',
                'thinking_blocks': [{'type': 'thinking', 'thinking': 'plain', 'signature': 'sig-plain'}],
            }
        )
    )

    messages = [{'role': 'assistant', 'content': 'Plain assistant text.'}]

    assert handler._inject_thinking_blocks(messages) == 1
    assert messages[0]['thinking_blocks'] == [{'type': 'thinking', 'thinking': 'plain', 'signature': 'sig-plain'}]


def test_preserve_thinking_does_not_fall_back_to_position():
    handler = _handler()
    handler._cache_thinking_from_response(
        _response_message(
            {
                'role': 'assistant',
                'content': 'original assistant text',
                'thinking_blocks': [{'type': 'thinking', 'thinking': 'original', 'signature': 'sig-original'}],
            }
        )
    )

    messages = [{'role': 'assistant', 'content': 'different assistant text'}]

    assert handler._inject_thinking_blocks(messages) == 0
    assert 'thinking_blocks' not in messages[0]


def test_preserve_thinking_skips_ambiguous_content_matches():
    handler = _handler()
    for signature in ('sig-a', 'sig-b'):
        handler._cache_thinking_from_response(
            _response_message(
                {
                    'role': 'assistant',
                    'content': 'same text',
                    'thinking_blocks': [{'type': 'thinking', 'thinking': signature, 'signature': signature}],
                }
            )
        )

    messages = [{'role': 'assistant', 'content': 'same text'}]

    assert handler._inject_thinking_blocks(messages) == 0
    assert 'thinking_blocks' not in messages[0]


def test_preserve_thinking_replays_unsigned_blocks_with_message_thought_signatures():
    handler = _handler()
    cached_n, key = handler._cache_thinking_from_response(
        _response_message(
            {
                'role': 'assistant',
                'content': 'I will inspect the file.',
                'tool_calls': [{'id': 'call_1', 'function': {'name': 'read'}}],
                'thinking_blocks': [{'type': 'thinking', 'thinking': 'inspect', 'signature': None}],
                'provider_specific_fields': {'thought_signatures': ['sig-thought-1']},
            }
        )
    )
    assert (cached_n, key) == (1, 'call_1')

    messages = [
        {
            'role': 'assistant',
            'content': 'I will inspect the file.',
            'tool_calls': [{'id': 'call_1', 'function': {'name': 'read'}}],
        }
    ]

    assert handler._inject_thinking_blocks(messages) == 1
    assert messages[0]['thinking_blocks'] == [{'type': 'thinking', 'thinking': 'inspect', 'signature': None}]
    assert messages[0]['provider_specific_fields'] == {'thought_signatures': ['sig-thought-1']}


def test_preserve_thinking_replays_thought_signatures_without_blocks():
    handler = _handler()
    handler._cache_thinking_from_response(
        _response_message(
            {
                'role': 'assistant',
                'content': 'no summary emitted',
                'provider_specific_fields': {'thought_signatures': ['sig-thought-1']},
            }
        )
    )

    messages = [{'role': 'assistant', 'content': 'no summary emitted'}]

    assert handler._inject_thinking_blocks(messages) == 1
    assert 'thinking_blocks' not in messages[0]
    assert messages[0]['provider_specific_fields'] == {'thought_signatures': ['sig-thought-1']}


def test_preserve_thinking_ignores_unsigned_blocks_without_thought_signatures():
    handler = _handler()
    cached_n, key = handler._cache_thinking_from_response(
        _response_message(
            {
                'role': 'assistant',
                'content': 'unsigned only',
                'thinking_blocks': [{'type': 'thinking', 'thinking': 'inspect', 'signature': None}],
            }
        )
    )
    assert (cached_n, key) == (0, None)

    messages = [{'role': 'assistant', 'content': 'unsigned only'}]

    assert handler._inject_thinking_blocks(messages) == 0
    assert 'thinking_blocks' not in messages[0]


def test_preserve_thinking_does_not_overwrite_existing_provider_specific_fields():
    handler = _handler()
    handler._cache_thinking_from_response(
        _response_message(
            {
                'role': 'assistant',
                'content': 'keep mine',
                'thinking_blocks': [{'type': 'thinking', 'thinking': 'inspect', 'signature': None}],
                'provider_specific_fields': {'thought_signatures': ['sig-cached']},
            }
        )
    )

    messages = [
        {
            'role': 'assistant',
            'content': 'keep mine',
            'provider_specific_fields': {'thought_signatures': ['sig-live']},
        }
    ]

    assert handler._inject_thinking_blocks(messages) == 1
    assert messages[0]['provider_specific_fields'] == {'thought_signatures': ['sig-live']}


def test_preserve_thinking_restores_signatures_when_scaffold_echoes_empty_signature_block():
    """bedrock-claude preset: Claude Code echoes ``{thinking, signature: ''}`` back.

    That block carries no state, so it must not stop the cached Gemini
    ``thought_signatures`` from being restored.
    """
    handler = _handler()
    handler._cache_thinking_from_response(
        _response_message({
            'role': 'assistant',
            'content': 'I will inspect the file.',
            'tool_calls': [{'id': 'call_1', 'function': {'name': 'read'}}],
            'reasoning_content': 'summary',
            'thinking_blocks': [{'type': 'thinking', 'thinking': 'summary'}],
            'provider_specific_fields': {'thought_signatures': ['sig-text']},
        })
    )

    messages = [
        {
            'role': 'assistant',
            'content': 'I will inspect the file.',
            'tool_calls': [{'id': 'call_1', 'function': {'name': 'read'}}],
            'thinking_blocks': [{'type': 'thinking', 'thinking': 'summary', 'signature': ''}],
        }
    ]

    assert handler._inject_thinking_blocks(messages) == 1
    assert messages[0]['provider_specific_fields'] == {'thought_signatures': ['sig-text']}
    assert messages[0]['thinking_blocks'] == [{'type': 'thinking', 'thinking': 'summary'}]


def test_preserve_thinking_keeps_scaffold_signed_blocks():
    handler = _handler()
    handler._cache_thinking_from_response(
        _response_message({
            'role': 'assistant',
            'content': 'keep mine',
            'thinking_blocks': [{'type': 'thinking', 'thinking': 'cached', 'signature': 'sig-cached'}],
        })
    )
    live = [{'type': 'thinking', 'thinking': 'live', 'signature': 'sig-live'}]
    messages = [{'role': 'assistant', 'content': 'keep mine', 'thinking_blocks': list(live)}]

    assert handler._inject_thinking_blocks(messages) == 0
    assert messages[0]['thinking_blocks'] == live


def test_preserve_thinking_keeps_scaffold_redacted_blocks():
    handler = _handler()
    handler._cache_thinking_from_response(
        _response_message({
            'role': 'assistant',
            'content': 'redacted',
            'thinking_blocks': [{'type': 'thinking', 'thinking': 'cached', 'signature': 'sig-cached'}],
        })
    )
    live = [{'type': 'redacted_thinking', 'data': 'opaque'}]
    messages = [{'role': 'assistant', 'content': 'redacted', 'thinking_blocks': list(live)}]

    assert handler._inject_thinking_blocks(messages) == 0
    assert messages[0]['thinking_blocks'] == live


def test_drop_unreplayable_thinking_blocks():
    messages = [
        {'role': 'user', 'content': 'hi', 'thinking_blocks': [{'type': 'thinking', 'thinking': 'x', 'signature': ''}]},
        {
            'role': 'assistant',
            'content': 'a',
            'thinking_blocks': [
                {'type': 'thinking', 'thinking': 'empty', 'signature': ''},
                {'type': 'thinking', 'thinking': 'none', 'signature': None},
                {'type': 'thinking', 'thinking': 'missing'},
                {'type': 'thinking', 'thinking': 'signed', 'signature': 'sig'},
                {'type': 'redacted_thinking', 'data': 'opaque'},
                {'type': 'redacted_thinking', 'data': ''},
            ],
        },
        {
            'role': 'assistant',
            'content': 'b',
            'thinking_blocks': [{'type': 'thinking', 'thinking': 'summary', 'signature': ''}],
            'provider_specific_fields': {'thought_signatures': ['sig-text']},
        },
        {'role': 'assistant', 'content': 'c'},
    ]

    assert LLMRequestHandler._drop_unreplayable_thinking_blocks(messages) == 5
    # Non-assistant messages are left alone.
    assert messages[0]['thinking_blocks'] == [{'type': 'thinking', 'thinking': 'x', 'signature': ''}]
    assert messages[1]['thinking_blocks'] == [
        {'type': 'thinking', 'thinking': 'signed', 'signature': 'sig'},
        {'type': 'redacted_thinking', 'data': 'opaque'},
    ]
    assert 'thinking_blocks' not in messages[2]
    assert messages[2]['provider_specific_fields'] == {'thought_signatures': ['sig-text']}
    assert messages[3] == {'role': 'assistant', 'content': 'c'}


def test_drop_unreplayable_thinking_blocks_is_noop_for_signed_history():
    messages = [
        {
            'role': 'assistant',
            'content': 'a',
            'thinking_blocks': [{'type': 'thinking', 'thinking': 't', 'signature': 'sig'}],
        }
    ]
    snapshot = json.loads(json.dumps(messages))

    assert LLMRequestHandler._drop_unreplayable_thinking_blocks(messages) == 0
    assert messages == snapshot


def test_preserve_reasoning_injects_by_tool_call_ids_and_content():
    handler = _handler()
    cached_n, key = handler._cache_reasoning_items_from_response(
        _response_message(
            {
                'role': 'assistant',
                'content': 'I will inspect the file.',
                'tool_calls': [{'id': 'call_1', 'function': {'name': 'read'}}],
                'reasoning_items': [_reasoning_item()],
            }
        )
    )
    assert cached_n == 1
    assert key == 'call_1'

    messages = [
        {
            'role': 'assistant',
            'content': 'I will inspect the file.',
            'tool_calls': [{'id': 'call_1', 'function': {'name': 'read'}}],
        }
    ]

    assert handler._inject_reasoning_items(messages) == 1
    assert messages[0]['reasoning_items'] == [_reasoning_item()]


def test_preserve_reasoning_skips_when_content_differs():
    handler = _handler()
    handler._cache_reasoning_items_from_response(
        _response_message(
            {
                'role': 'assistant',
                'content': 'I will inspect the file.',
                'tool_calls': [{'id': 'call_1', 'function': {'name': 'read'}}],
                'reasoning_items': [_reasoning_item()],
            }
        )
    )

    messages = [
        {
            'role': 'assistant',
            'content': 'rewritten by scaffold',
            'tool_calls': [{'id': 'call_1', 'function': {'name': 'read'}}],
        }
    ]

    assert handler._inject_reasoning_items(messages) == 0
    assert 'reasoning_items' not in messages[0]


def test_preserve_reasoning_skips_when_tool_call_ids_differ():
    handler = _handler()
    handler._cache_reasoning_items_from_response(
        _response_message(
            {
                'role': 'assistant',
                'content': 'I will inspect the file.',
                'tool_calls': [{'id': 'call_1', 'function': {'name': 'read'}}],
                'reasoning_items': [_reasoning_item()],
            }
        )
    )

    messages = [
        {
            'role': 'assistant',
            'content': 'I will inspect the file.',
            'tool_calls': [{'id': 'call_2', 'function': {'name': 'read'}}],
        }
    ]

    assert handler._inject_reasoning_items(messages) == 0
    assert 'reasoning_items' not in messages[0]


def test_preserve_reasoning_skips_when_tool_call_order_differs():
    handler = _handler()
    handler._cache_reasoning_items_from_response(
        _response_message(
            {
                'role': 'assistant',
                'content': 'I will inspect the files.',
                'tool_calls': [
                    {'id': 'call_1', 'type': 'function', 'function': {'name': 'read', 'arguments': '{"path":"a"}'}},
                    {'id': 'call_2', 'type': 'function', 'function': {'name': 'read', 'arguments': '{"path":"b"}'}},
                ],
                'reasoning_items': [_reasoning_item()],
            }
        )
    )

    messages = [
        {
            'role': 'assistant',
            'content': 'I will inspect the files.',
            'tool_calls': [
                {'id': 'call_2', 'type': 'function', 'function': {'name': 'read', 'arguments': '{"path":"b"}'}},
                {'id': 'call_1', 'type': 'function', 'function': {'name': 'read', 'arguments': '{"path":"a"}'}},
            ],
        }
    ]

    assert handler._inject_reasoning_items(messages) == 0
    assert 'reasoning_items' not in messages[0]


def test_preserve_reasoning_skips_when_tool_call_payload_differs():
    handler = _handler()
    handler._cache_reasoning_items_from_response(
        _response_message(
            {
                'role': 'assistant',
                'content': 'I will inspect the file.',
                'tool_calls': [
                    {'id': 'call_1', 'type': 'function', 'function': {'name': 'read', 'arguments': '{"path":"a"}'}},
                ],
                'reasoning_items': [_reasoning_item()],
            }
        )
    )

    messages = [
        {
            'role': 'assistant',
            'content': 'I will inspect the file.',
            'tool_calls': [
                {'id': 'call_1', 'type': 'function', 'function': {'name': 'read', 'arguments': '{"path":"b"}'}},
            ],
        }
    ]

    assert handler._inject_reasoning_items(messages) == 0
    assert 'reasoning_items' not in messages[0]


def test_preserve_reasoning_does_not_overwrite_existing_items():
    handler = _handler()
    handler._cache_reasoning_items_from_response(
        _response_message(
            {
                'role': 'assistant',
                'content': 'Plain assistant text.',
                'reasoning_items': [_reasoning_item()],
            }
        )
    )

    existing = [_reasoning_item(id='rs_existing', encrypted_content='existing')]
    messages = [{'role': 'assistant', 'content': 'Plain assistant text.', 'reasoning_items': existing}]

    assert handler._inject_reasoning_items(messages) == 0
    assert messages[0]['reasoning_items'] == existing


def test_preserve_reasoning_restores_provider_item_id_rewritten_by_codex():
    handler = _handler()
    original = _reasoning_item(id='rs_provider', encrypted_content='bound-to-provider-id')
    handler._cache_reasoning_items_from_response(
        _response_message({
            'role': 'assistant',
            'content': '',
            'tool_calls': [{'id': 'call_1', 'function': {'name': 'read'}}],
            'reasoning_items': [original],
        })
    )

    messages = [{
        'role': 'assistant',
        'content': '',
        'tool_calls': [{'id': 'call_1', 'function': {'name': 'read'}}],
        'reasoning_items': [
            _reasoning_item(id='rs_codex_rewritten', encrypted_content='bound-to-provider-id')
        ],
    }]

    assert handler._inject_reasoning_items(messages) == 1
    assert messages[0]['reasoning_items'] == [original]


def test_preserve_reasoning_restores_content_only_item_id_by_encrypted_content():
    handler = _handler()
    original = _reasoning_item(id='rs_provider', encrypted_content='bound-to-provider-id')
    handler._cache_reasoning_items_from_response(
        _response_message({
            'role': 'assistant',
            'content': 'Here is the plan.',
            'reasoning_items': [original],
        })
    )

    messages = [{
        'role': 'assistant',
        'content': 'Here is the plan.',
        'reasoning_items': [
            _reasoning_item(id='rs_codex_rewritten', encrypted_content='bound-to-provider-id'),
        ],
    }]

    assert handler._inject_reasoning_items(messages) == 1
    assert messages[0]['reasoning_items'] == [original]


def test_preserve_reasoning_keeps_content_only_items_when_blob_does_not_match():
    handler = _handler()
    handler._cache_reasoning_items_from_response(
        _response_message({
            'role': 'assistant',
            'content': 'Plain assistant text.',
            'reasoning_items': [_reasoning_item()],
        })
    )

    existing = [_reasoning_item(id='rs_existing', encrypted_content='unrelated-blob')]
    messages = [{
        'role': 'assistant',
        'content': 'Plain assistant text.',
        'reasoning_items': existing,
    }]

    assert handler._inject_reasoning_items(messages) == 0
    assert messages[0]['reasoning_items'] == existing


def test_preserve_reasoning_restores_polarist_output_items_for_sft():
    handler = _handler()
    original = _reasoning_item(id='rs_provider', encrypted_content='enc-1')
    output_items = [
        original,
        {
            'type': 'message',
            'id': 'msg_1',
            'role': 'assistant',
            'status': 'completed',
            'content': [{'type': 'output_text', 'text': 'I will inspect the file.'}],
        },
        {
            'type': 'function_call',
            'id': 'fc_1',
            'call_id': 'call_1',
            'name': 'bash',
            'arguments': '{"command":"ls"}',
            'status': 'completed',
        },
    ]
    handler._cache_reasoning_items_from_response(
        _response_message({
            'role': 'assistant',
            'content': 'I will inspect the file.',
            'tool_calls': [{'id': 'call_1', 'function': {'name': 'bash'}}],
            'reasoning_items': [original],
            'provider_specific_fields': {'responses_output_items': output_items},
        })
    )

    messages = [{
        'role': 'assistant',
        'content': 'I will inspect the file.',
        'tool_calls': [{'id': 'call_1', 'function': {'name': 'bash'}}],
        'reasoning_items': [_reasoning_item(id='rs_codex', encrypted_content='enc-1')],
        'provider_specific_fields': {'thought_signatures': ['keep']},
    }]

    assert handler._inject_reasoning_items(messages) == 1
    assert messages[0]['reasoning_items'] == [original]
    psf = messages[0]['provider_specific_fields']
    assert psf['thought_signatures'] == ['keep']
    assert psf['responses_output_items'] == output_items
    assert psf['responses_output_items'][0]['id'] == 'rs_provider'


def test_preserve_reasoning_injects_by_unique_plaintext_content():
    handler = _handler()
    handler._cache_reasoning_items_from_response(
        _response_message(
            {
                'role': 'assistant',
                'content': 'Plain assistant text.',
                'reasoning_items': [_reasoning_item()],
            }
        )
    )

    messages = [{'role': 'assistant', 'content': 'Plain assistant text.'}]

    assert handler._inject_reasoning_items(messages) == 1
    assert messages[0]['reasoning_items'] == [_reasoning_item()]


def test_preserve_reasoning_skips_ambiguous_content_matches():
    handler = _handler()
    for item_id in ('rs_a', 'rs_b'):
        handler._cache_reasoning_items_from_response(
            _response_message(
                {
                    'role': 'assistant',
                    'content': 'same text',
                    'reasoning_items': [_reasoning_item(id=item_id, encrypted_content=f'enc-{item_id}')],
                }
            )
        )

    messages = [{'role': 'assistant', 'content': 'same text'}]

    assert handler._inject_reasoning_items(messages) == 0
    assert 'reasoning_items' not in messages[0]


def test_preserve_reasoning_skips_encrypted_item_without_id():
    handler = _handler()

    cached_n, _ = handler._cache_reasoning_items_from_response(
        _response_message(
            {
                'role': 'assistant',
                'content': 'Plain assistant text.',
                'reasoning_items': [_reasoning_item(id=None)],
            }
        )
    )

    assert cached_n == 0
    assert handler._reasoning_items_cache == []


def test_preserve_reasoning_injects_content_entry_onto_merged_message():
    # opencode collapses a multi-choice Responses reply (text choice + function_call
    # choice) into one assistant message carrying both content and tool_calls. The
    # reasoning rode on the text (content-only) choice, so it must be matched by
    # content even though the merged message also has tool_calls.
    handler = _handler()
    handler._cache_reasoning_items_from_response(
        _response_message(
            {
                'role': 'assistant',
                'content': 'I will inspect the repository structure.',
                'reasoning_items': [_reasoning_item()],
            }
        )
    )
    # The tool-call choice carried no reasoning of its own.
    handler._cache_reasoning_items_from_response(
        _response_message(
            {
                'role': 'assistant',
                'tool_calls': [{'id': 'call_1', 'type': 'function', 'function': {'name': 'glob'}}],
            }
        )
    )

    messages = [
        {
            'role': 'assistant',
            'content': 'I will inspect the repository structure.',
            'tool_calls': [
                {'id': 'call_1', 'type': 'function', 'function': {'name': 'glob'}, 'index': 0},
            ],
        }
    ]

    assert handler._inject_reasoning_items(messages) == 1
    assert messages[0]['reasoning_items'] == [_reasoning_item()]


def test_preserve_reasoning_merges_content_then_tool_entries_in_order():
    # Both the text choice and the tool-call choice carried reasoning. They must be
    # injected onto the single merged message in temporal order: content reasoning
    # first, then tool-call reasoning, mirroring litellm's replay order.
    handler = _handler()
    handler._cache_reasoning_items_from_response(
        _response_message(
            {
                'role': 'assistant',
                'content': 'Planning the edit.',
                'reasoning_items': [_reasoning_item(id='rs_content', encrypted_content='enc-content')],
            }
        )
    )
    handler._cache_reasoning_items_from_response(
        _response_message(
            {
                'role': 'assistant',
                'tool_calls': [{'id': 'call_1', 'type': 'function', 'function': {'name': 'edit'}}],
                'reasoning_items': [_reasoning_item(id='rs_tool', encrypted_content='enc-tool')],
            }
        )
    )

    messages = [
        {
            'role': 'assistant',
            'content': 'Planning the edit.',
            'tool_calls': [{'id': 'call_1', 'type': 'function', 'function': {'name': 'edit'}, 'index': 0}],
        }
    ]

    assert handler._inject_reasoning_items(messages) == 1
    assert [item['id'] for item in messages[0]['reasoning_items']] == ['rs_content', 'rs_tool']


def test_preserve_reasoning_matches_tool_call_with_reordered_argument_keys():
    # opencode re-serializes tool_call arguments, reordering JSON object keys. The
    # canonical comparison must still match on the semantically identical payload.
    handler = _handler()
    handler._cache_reasoning_items_from_response(
        _response_message(
            {
                'role': 'assistant',
                'tool_calls': [
                    {
                        'id': 'call_1',
                        'type': 'function',
                        'function': {'name': 'grep', 'arguments': '{"path":"/x","pattern":"foo"}'},
                        'index': 0,
                    }
                ],
                'reasoning_items': [_reasoning_item()],
            }
        )
    )

    messages = [
        {
            'role': 'assistant',
            'tool_calls': [
                {
                    'id': 'call_1',
                    'type': 'function',
                    'function': {'name': 'grep', 'arguments': '{"pattern":"foo","path":"/x"}'},
                }
            ],
        }
    ]

    assert handler._inject_reasoning_items(messages) == 1
    assert messages[0]['reasoning_items'] == [_reasoning_item()]


def test_preserve_reasoning_single_choice_content_and_tools_not_double_counted():
    # HY-style single choice: reasoning, content and tool_calls all in one message
    # cached as one entry. A merged-message injection must use that one entry only,
    # not also pull a coincidental same-text content-only entry (there is none here).
    handler = _handler()
    handler._cache_reasoning_items_from_response(
        _response_message(
            {
                'role': 'assistant',
                'content': 'Inspecting now.',
                'tool_calls': [{'id': 'call_1', 'type': 'function', 'function': {'name': 'read'}}],
                'reasoning_items': [_reasoning_item()],
            }
        )
    )

    messages = [
        {
            'role': 'assistant',
            'content': 'Inspecting now.',
            'tool_calls': [{'id': 'call_1', 'type': 'function', 'function': {'name': 'read'}}],
        }
    ]

    assert handler._inject_reasoning_items(messages) == 1
    assert messages[0]['reasoning_items'] == [_reasoning_item()]


def test_preserve_reasoning_omits_null_encrypted_content():
    handler = _handler()

    cached_n, _ = handler._cache_reasoning_items_from_response(
        _response_message(
            {
                'role': 'assistant',
                'content': 'Plain assistant text.',
                'reasoning_items': [_reasoning_item(encrypted_content=None)],
            }
        )
    )

    assert cached_n == 1
    assert 'encrypted_content' not in handler._reasoning_items_cache[0]['items'][0]


class FakeRuntime:
    def __init__(self) -> None:
        self.emitted = []
        self.registered = {}

    async def emit(self, event, data):  # noqa: ANN001, ANN202
        self.emitted.append((event, data))

    def on(self, event, handler):  # noqa: ANN001, ANN202
        self.registered[event] = handler

    def off(self, event, handler):  # noqa: ANN001, ANN202, ARG002
        self.registered.pop(event, None)


class FakeOpenAIClient:
    class Config:
        request_args = {}
        client_args = {}

    config = Config()

    def __init__(self) -> None:
        self.calls = 0
        self.last_kwargs: dict | None = None

    async def create_completion(self, **kwargs):  # noqa: ANN003, ANN202
        self.calls += 1
        self.last_kwargs = kwargs
        await asyncio.sleep(0.02)
        return {'choices': [{'message': {'content': kwargs['messages'][0]['content']}}]}


def _finals(runtime: FakeRuntime) -> list[dict]:
    return [
        data
        for (event, data) in runtime.emitted
        if event == 'llm_response' and (data.get('rsio') or {}).get('kind') == 'final'
    ]


def _runnings(runtime: FakeRuntime) -> list[dict]:
    return [
        data
        for (event, data) in runtime.emitted
        if event == 'llm_response' and (data.get('rsio') or {}).get('kind') == 'running'
    ]


@pytest.mark.asyncio
async def test_heartbeat_stops_after_five_minutes_of_consecutive_emit_failures(monkeypatch) -> None:
    handler = _handler()
    handler._report_interval = 0
    outcomes = iter([False, False, True, False, False, False])
    failure_times = iter([0.0, 100.0, 200.0, 400.0, 500.0])
    calls = 0

    async def emit_running(request_id: str) -> bool:
        nonlocal calls
        assert request_id == 'R5'
        calls += 1
        return next(outcomes)

    monkeypatch.setattr(handler, '_emit_running', emit_running)
    monkeypatch.setattr(llm_handler, 'time', SimpleNamespace(monotonic=lambda: next(failure_times)))

    await handler._run_heartbeat('R5')

    assert calls == 6


@pytest.mark.asyncio
async def test_handler_emits_running_ack_before_final(caplog) -> None:
    runtime = FakeRuntime()
    client = FakeOpenAIClient()
    handler = LLMRequestHandler(runtime=runtime, openai_client=client, preserve_thinking=False)

    with caplog.at_level(logging.DEBUG, logger=llm_handler.__name__):
        await handler.handle_llm_request(
            {
                'request_id': 'RA',
                'messages': [{'role': 'user', 'content': 'hi'}],
                'rsio': {'attempt': 1},
            }
        )

    # A running ack is emitted immediately (submit-ack), before the final.
    kinds = [(e, (d.get('rsio') or {}).get('kind')) for (e, d) in runtime.emitted if e == 'llm_response']
    assert kinds[0] == ('llm_response', 'running')
    assert kinds[-1] == ('llm_response', 'final')
    assert len(_runnings(runtime)) >= 1
    rsio_records = [record for record in caplog.records if record.getMessage().startswith('rsio.host:')]
    assert rsio_records
    assert all(record.levelno == logging.DEBUG for record in rsio_records)
    handling_record = next(
        record for record in caplog.records if record.getMessage().startswith('Handling LLM request:')
    )
    assert handling_record.levelno == logging.DEBUG


@pytest.mark.asyncio
async def test_handler_delays_replay_discard_after_successful_final_emit() -> None:
    runtime = FakeRuntime()
    client = FakeOpenAIClient()
    handler = LLMRequestHandler(runtime=runtime, openai_client=client, preserve_thinking=False)
    env = {'request_id': 'RB', 'messages': [{'role': 'user', 'content': 'hi'}], 'rsio': {'attempt': 1}}

    # A resend already in flight when the first final arrives must replay the
    # completed response instead of starting another billed model call.
    await handler.handle_llm_request(dict(env))
    await handler.handle_llm_request(dict(env))

    assert client.calls == 1
    assert len(_runnings(runtime)) == 1
    assert len(_finals(runtime)) == 2
    assert 'llm_request:RB' in handler._guard._entries


@pytest.mark.asyncio
async def test_handler_keeps_replay_when_final_emit_fails() -> None:
    class _FailingFinalRuntime(FakeRuntime):
        async def emit(self, event, data):  # noqa: ANN001, ANN202
            self.emitted.append((event, data))
            if event == 'llm_response' and (data.get('rsio') or {}).get('kind') == 'final':
                raise RuntimeError('socket disconnected')

    runtime = _FailingFinalRuntime()
    client = FakeOpenAIClient()
    handler = LLMRequestHandler(runtime=runtime, openai_client=client, preserve_thinking=False)
    env = {'request_id': 'RC', 'messages': [{'role': 'user', 'content': 'hi'}], 'rsio': {'attempt': 1}}

    await handler.handle_llm_request(dict(env))
    await handler.handle_llm_request(dict(env))

    assert client.calls == 1
    assert len(_runnings(runtime)) == 1
    assert len(_finals(runtime)) == 2
    assert 'llm_request:RC' in handler._guard._entries


@pytest.mark.asyncio
async def test_handler_replays_a_lost_final_for_a_call_longer_than_the_replay_ttl(monkeypatch) -> None:
    clock = [0.0]
    monkeypatch.setattr(rpc_replay_guard, 'time', SimpleNamespace(monotonic=lambda: clock[0]))

    class _FailingFinalRuntime(FakeRuntime):
        async def emit(self, event, data):  # noqa: ANN001, ANN202
            self.emitted.append((event, data))
            if event == 'llm_response' and (data.get('rsio') or {}).get('kind') == 'final':
                raise RuntimeError('socket disconnected')

    class _SlowClient(FakeOpenAIClient):
        async def create_completion(self, **kwargs):  # noqa: ANN003, ANN202
            clock[0] += 310.0
            return await super().create_completion(**kwargs)

    client = _SlowClient()
    handler = LLMRequestHandler(runtime=_FailingFinalRuntime(), openai_client=client, preserve_thinking=False)
    env = {'request_id': 'RD', 'messages': [{'role': 'user', 'content': 'hi'}], 'rsio': {'attempt': 1}}

    await handler.handle_llm_request(dict(env))
    clock[0] += 45.0
    await handler.handle_llm_request({**env, 'rsio': {'attempt': 2}})

    assert client.calls == 1


def test_final_replay_grace_outlasts_the_worker_resend_gap() -> None:
    assert llm_handler._FINAL_REPLAY_GRACE_SECONDS > RSIOLLMConfig().report_timeout


@pytest.mark.asyncio
async def test_handler_legacy_when_request_has_no_rsio() -> None:
    # An old worker sends llm_request WITHOUT an rsio sub-field. The host must
    # not emit running heartbeats (old worker mistakes them for a failed final)
    # and must emit a plain final with no rsio field.
    runtime = FakeRuntime()
    client = FakeOpenAIClient()
    handler = LLMRequestHandler(runtime=runtime, openai_client=client, preserve_thinking=False)

    await handler.handle_llm_request(
        {
            'request_id': 'OLD1',
            'messages': [{'role': 'user', 'content': 'hi'}],
        }
    )  # no 'rsio' key

    assert len(_runnings(runtime)) == 0  # no heartbeat/ack in legacy mode
    finals = [d for (e, d) in runtime.emitted if e == 'llm_response']
    assert len(finals) == 1
    assert finals[0]['success'] is True
    assert 'rsio' not in finals[0]  # legacy final carries no rsio
    assert finals[0]['response'] == {'choices': [{'message': {'content': 'hi'}}]}
    assert client.calls == 1


@pytest.mark.asyncio
async def test_handler_emits_final_with_rsio() -> None:
    runtime = FakeRuntime()
    client = FakeOpenAIClient()
    handler = LLMRequestHandler(runtime=runtime, openai_client=client, preserve_thinking=False)

    result = await handler.handle_llm_request(
        {
            'request_id': 'R1',
            'messages': [{'role': 'user', 'content': 'hi'}],
            'rsio': {'request_id': 'R1', 'attempt': 1},
        }
    )

    assert result is None
    finals = _finals(runtime)
    assert len(finals) == 1
    assert finals[0]['success'] is True
    assert finals[0]['request_id'] == 'R1'
    assert finals[0]['response'] == {'choices': [{'message': {'content': 'hi'}}]}
    # Response rsio carries only `kind`; request_id lives at the top level.
    assert finals[0]['rsio'] == {'kind': 'final'}
    assert client.calls == 1


@pytest.mark.asyncio
async def test_handler_dedups_concurrent_same_request_id() -> None:
    runtime = FakeRuntime()
    client = FakeOpenAIClient()
    handler = LLMRequestHandler(runtime=runtime, openai_client=client, preserve_thinking=False)
    env = {
        'request_id': 'R2',
        'messages': [{'role': 'user', 'content': 'hi'}],
        'rsio': {'request_id': 'R2', 'attempt': 1},
    }

    await asyncio.gather(handler.handle_llm_request(dict(env)), handler.handle_llm_request(dict(env)))

    assert client.calls == 1  # guard dedups; create_completion runs once
    assert len(_runnings(runtime)) == 1
    assert len(_finals(runtime)) == 2
    assert 'llm_request:R2' in handler._guard._entries


@pytest.mark.asyncio
async def test_handler_does_not_leak_rsio_to_client() -> None:
    runtime = FakeRuntime()
    client = FakeOpenAIClient()
    handler = LLMRequestHandler(runtime=runtime, openai_client=client, preserve_thinking=False)

    await handler.handle_llm_request(
        {
            'request_id': 'R3',
            'messages': [{'role': 'user', 'content': 'hi'}],
            'rsio': {'request_id': 'R3', 'attempt': 1},
        }
    )

    assert 'rsio' not in client.last_kwargs
    assert 'request_id' not in client.last_kwargs


@pytest.mark.asyncio
async def test_handler_error_emits_final_failure() -> None:
    runtime = FakeRuntime()

    class _BoomClient(FakeOpenAIClient):
        async def create_completion(self, **kwargs):  # noqa: ANN003, ANN202
            raise ValueError('boom')

    handler = LLMRequestHandler(runtime=runtime, openai_client=_BoomClient(), preserve_thinking=False)

    result = await handler.handle_llm_request(
        {
            'request_id': 'R4',
            'messages': [{'role': 'user', 'content': 'hi'}],
            'rsio': {'request_id': 'R4', 'attempt': 1},
        }
    )

    assert result is None
    finals = _finals(runtime)
    assert len(finals) == 1
    assert finals[0]['success'] is False
    assert 'boom' in finals[0]['error']
    assert 'error_code' not in finals[0]
    assert 'llm_request:R4' in handler._guard._entries


class _BudgetClient(FakeOpenAIClient):
    """Refuses the requests named in ``refuse`` on token budget; serves the rest normally."""

    def __init__(self, refuse: set[str]) -> None:
        super().__init__()
        self.refuse = refuse

    async def create_completion(self, **kwargs):  # noqa: ANN003, ANN202
        if kwargs['messages'][0]['content'] in self.refuse:
            raise ContextLengthExceededError('prompt is 200k tokens over the limit')
        return await super().create_completion(**kwargs)


async def _request(handler: LLMRequestHandler, request_id: str) -> None:
    await handler.handle_llm_request(
        {
            'request_id': request_id,
            'messages': [{'role': 'user', 'content': request_id}],
            'rsio': {'request_id': request_id, 'attempt': 1},
        }
    )


@pytest.mark.asyncio
async def test_budget_refusal_is_attributed_to_its_own_request() -> None:
    runtime = FakeRuntime()
    handler = LLMRequestHandler(runtime=runtime, openai_client=_BudgetClient({'r1'}), preserve_thinking=False)

    await _request(handler, 'r1')

    assert handler.request_outcomes == {'r1': CONTEXT_LENGTH_EXCEEDED_CODE}
    assert handler.last_request_error_code == CONTEXT_LENGTH_EXCEEDED_CODE
    assert _finals(runtime)[-1]['error_code'] == CONTEXT_LENGTH_EXCEEDED_CODE


@pytest.mark.asyncio
async def test_later_success_is_not_reported_as_a_budget_refusal() -> None:
    """The regression this attribution exists for: one refused call must not taint the whole run."""
    runtime = FakeRuntime()
    handler = LLMRequestHandler(runtime=runtime, openai_client=_BudgetClient({'r1'}), preserve_thinking=False)

    await _request(handler, 'r1')
    await _request(handler, 'r2')

    assert handler.request_outcomes == {'r1': CONTEXT_LENGTH_EXCEEDED_CODE, 'r2': None}
    assert handler.last_request_error_code is None


@pytest.mark.asyncio
async def test_unrelated_failure_after_a_refusal_is_not_labelled_a_refusal() -> None:
    runtime = FakeRuntime()

    class _FlakyClient(_BudgetClient):
        async def create_completion(self, **kwargs):  # noqa: ANN003, ANN202
            if kwargs['messages'][0]['content'] == 'r2':
                raise ValueError('connection reset')
            return await super().create_completion(**kwargs)

    handler = LLMRequestHandler(runtime=runtime, openai_client=_FlakyClient({'r1'}), preserve_thinking=False)

    await _request(handler, 'r1')
    await _request(handler, 'r2')

    assert handler.request_outcomes == {'r1': CONTEXT_LENGTH_EXCEEDED_CODE, 'r2': None}
    assert handler.last_request_error_code is None


@pytest.mark.asyncio
async def test_resend_of_a_settled_request_keeps_the_original_outcome_order() -> None:
    runtime = FakeRuntime()
    handler = LLMRequestHandler(runtime=runtime, openai_client=_BudgetClient({'r2'}), preserve_thinking=False)

    await _request(handler, 'r1')
    await _request(handler, 'r2')
    await _request(handler, 'r1')  # worker resend of an already-settled call

    assert list(handler.request_outcomes) == ['r1', 'r2']
    assert handler.last_request_error_code == CONTEXT_LENGTH_EXCEEDED_CODE


@pytest.mark.asyncio
async def test_no_requests_yet_reports_no_error_code() -> None:
    handler = LLMRequestHandler(runtime=FakeRuntime(), openai_client=FakeOpenAIClient(), preserve_thinking=False)

    assert handler.request_outcomes == {}
    assert handler.last_request_error_code is None
    assert handler.last_request_finish_reason is None
    assert handler.last_request_left_tool_call_unparsed is False


class _ChoiceShapeClient(_BudgetClient):
    """Serves each request with the first-choice fields named for it in ``choices``."""

    def __init__(self, choices: dict[str, dict], refuse: set[str] = frozenset()) -> None:
        super().__init__(set(refuse))
        self.choices = choices

    async def create_completion(self, **kwargs):  # noqa: ANN003, ANN202
        response = await super().create_completion(**kwargs)
        response['choices'][0].update(self.choices.get(kwargs['messages'][0]['content'], {}))
        return response


@pytest.mark.asyncio
async def test_last_finish_reason_follows_the_last_settled_request() -> None:
    client = _ChoiceShapeClient({'r1': {'finish_reason': 'stop'}, 'r2': {'finish_reason': 'length'}}, refuse={'r3'})
    handler = LLMRequestHandler(runtime=FakeRuntime(), openai_client=client, preserve_thinking=False)

    await _request(handler, 'r1')
    assert handler.last_request_finish_reason == 'stop'
    await _request(handler, 'r2')
    assert handler.last_request_finish_reason == 'length'
    await _request(handler, 'r3')
    assert handler.last_request_finish_reason is None


@pytest.mark.asyncio
async def test_unparsed_tool_call_flag_follows_the_last_settled_request() -> None:
    client = _ChoiceShapeClient(
        {
            'r1': {'finish_reason': 'stop', 'provider_specific_fields': {UNPARSED_TOOL_CALL_FIELD: True}},
            'r2': {'finish_reason': 'tool_calls', 'provider_specific_fields': {'other': 1}},
        },
        refuse={'r4'},
    )
    handler = LLMRequestHandler(runtime=FakeRuntime(), openai_client=client, preserve_thinking=False)

    await _request(handler, 'r1')
    assert handler.last_request_left_tool_call_unparsed is True
    await _request(handler, 'r2')
    assert handler.last_request_left_tool_call_unparsed is False
    await _request(handler, 'r1')  # a resend of a settled call must not move it back to last
    assert handler.last_request_left_tool_call_unparsed is False
    await _request(handler, 'r3')
    assert handler.last_request_left_tool_call_unparsed is False
    await _request(handler, 'r4')
    assert handler.last_request_left_tool_call_unparsed is False
