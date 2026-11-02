"""Task policy is explicit; shared Infer Proxy retains its own default contract."""

import asyncio
import copy
import json

from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import openai
import pytest

from tenacity import wait_none

from nexus.core import ContextLengthExceededError, OpenAIConfig, openai_client_registry
from nexus.extensions.openai_clients import infer_proxy as standard
from nexus.runtime.bridge.adapters import ags_osworld as bridge_adapter


def response(content='ok', calls=None):
    return openai.types.chat.ChatCompletion.model_validate(
        {
            'id': 'reply',
            'object': 'chat.completion',
            'created': 1,
            'model': 'service',
            'choices': [
                {
                    'index': 0,
                    'finish_reason': 'stop',
                    'message': {
                        'role': 'assistant',
                        'content': content,
                        **({'tool_calls': calls} if calls else {}),
                    },
                }
            ],
        }
    )


def make(monkeypatch, cls=None, args=None, request=None, enforced=None):
    create = AsyncMock(return_value=response())
    initialized = []

    class SDK:
        def __init__(self, **kwargs):
            initialized.append(kwargs)
            self.max_retries = kwargs.get('max_retries', 2)
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))

        def with_options(self, **kwargs):
            copied = copy.copy(self)
            copied.max_retries = kwargs['max_retries']
            return copied

    monkeypatch.setattr(openai, 'AsyncOpenAI', SDK)
    monkeypatch.setattr(bridge_adapter, '_infer_proxy_retry_wait', wait_none)
    # Test overrides are adapter attributes, never public SDK/client_args fields.
    policy = {key: value for key, value in (args or {}).items() if key != 'max_retries'}
    config = OpenAIConfig(
        client_args={'api_key': 'fixture', 'max_retries': (args or {}).get('max_retries', 3)},
        request_args={'model': 'service', **(request or {})},
        enforced_request_args=enforced or {},
    )
    source = standard.InferProxyClient(config)
    if cls is standard.InferProxyClient:
        return source, create, initialized
    adapter = bridge_adapter.adapt_infer_proxy_client(source)
    for key, value in policy.items():
        setattr(adapter, key, value)
    assert source.client.max_retries == 2
    assert adapter.client is not source.client
    return adapter, create, initialized


def bad_request():
    return openai.BadRequestError(
        'maximum context length is 1000 tokens; request has 800 input tokens',
        response=httpx.Response(400, request=httpx.Request('POST', 'https://test')),
        body={'error': {'code': 'context_length_exceeded'}},
    )


@pytest.mark.asyncio
async def test_standard_identity_sdk_retries_and_response_unchanged(monkeypatch):
    client, create, initialized = make(monkeypatch, standard.InferProxyClient)
    create.return_value = response('<think>reason</think>visible')
    result = await client._create_completion([], temperature=0.3)
    assert result.choices[0].message.content == '<think>reason</think>visible'
    assert 'max_retries' not in initialized[0]  # main retains SDK defaults.
    assert client.client.max_retries == 2
    assert 'request_id' not in create.call_args.kwargs.get('extra_body', {})
    assert openai_client_registry.get('infer_proxy') is standard.InferProxyClient
    assert not hasattr(standard, 'OSWorldInferProxyClient')


@pytest.mark.asyncio
async def test_adapter_is_handler_local_with_real_sdk_and_does_not_mutate_source(monkeypatch):
    from nexus.runtime.server.services.llm_handler import LLMRequestHandler as DefaultHandler

    sent = []

    async def transport(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=json.loads(response('<think>reason</think>visible').model_dump_json()))

    http = httpx.AsyncClient(transport=httpx.MockTransport(transport))
    config = OpenAIConfig(
        client_args={'api_key': 'test', 'base_url': 'https://fixture/v1', 'http_client': http},
        request_args={'model': 'service'},
        enforced_request_args={'extra_headers': {'SessionID': 'fixed'}},
    )
    original = standard.InferProxyClient(config)
    original._client_key = 'main'
    source_retries = original.client.max_retries
    specialized = bridge_adapter.create_handler(runtime=object(), openai_client=original)
    default = DefaultHandler(runtime=object(), openai_client=original)
    try:
        assert default.openai_client is original
        assert specialized.openai_client is not original
        adapted = specialized.openai_client
        assert adapted.client is not original.client
        assert adapted.client._client is original.client._client
        assert adapted.client.max_retries == 0
        assert original.client.max_retries == source_retries
        assert adapted._client_key == 'main'
        assert adapted.config is not config
        adapted.config.request_args['temperature'] = 0.3
        assert 'temperature' not in config.request_args
        a, b = await asyncio.gather(adapted._create_completion([]), original._create_completion([]))
        assert a.choices[0].message.content == 'visible'
        assert b.choices[0].message.content == '<think>reason</think>visible'
        assert sum('request_id' in request for request in sent) == 1
        await specialized.aclose()
        assert not http.is_closed  # The borrowed transport remains owned by the source.
        await original._create_completion([])
    finally:
        await original.client.close()


def test_adapter_does_not_wrap_custom_infer_proxy_subclasses():
    class CustomClient(standard.InferProxyClient):
        pass

    client = object.__new__(CustomClient)
    assert bridge_adapter.adapt_infer_proxy_client(client) is client


def test_six_mode_infer_config_has_no_task_policy_sdk_arguments(monkeypatch):
    from nexus.extensions.tasks.os_world.data.task_source import OSWorldBatchTask
    from nexus.extensions.tasks.os_world.generators.clients import _hunyuan_openai_clients

    monkeypatch.setenv('HUNYUAN_BASE_URL', 'https://fixture/v1')
    monkeypatch.setenv('HUNYUAN_MODEL', 'service')
    config = _hunyuan_openai_clients(
        OSWorldBatchTask(
            domain='os',
            task_id='test',
            instruction='test',
            task_file='unused',
        )
    )['main']
    assert config['name'] == 'infer_proxy'
    assert set(config['config']['client_args']) == {'api_key', 'base_url', 'timeout', 'max_retries'}
    assert 'extra_headers' in config['config']['enforced_request_args']


@pytest.mark.asyncio
async def test_standard_keeps_existing_tool_forwarding_and_retry_policy(monkeypatch):
    from nexus.utils import common

    monkeypatch.setattr(common, 'wait_exponential_jitter', lambda **kwargs: wait_none())
    client, create, _ = make(monkeypatch, standard.InferProxyClient)
    error = openai.AuthenticationError(
        'rejected',
        response=httpx.Response(401, request=httpx.Request('POST', 'https://test')),
        body={},
    )
    create.side_effect = error
    tools = [{'type': 'provider_specific_tool'}]
    with pytest.raises(openai.AuthenticationError):
        await client._create_completion([], tools=tools)
    assert create.await_count == 3
    assert create.call_args.kwargs['tools'] == tools


@pytest.mark.asyncio
async def test_task_normalizes_only_explicit_flags(monkeypatch):
    client, create, _ = make(monkeypatch, args={'split_think_content': True, 'normalize_tool_finish_reason': True})
    create.return_value = response(
        '<think>reason</think>visible',
        [
            {'id': 'call1', 'type': 'function', 'function': {'name': 'Bash', 'arguments': '{}'}},
        ],
    )
    result = await client._create_completion([])
    assert client.client.max_retries == 0
    assert result.choices[0].message.content == 'visible'
    assert result.choices[0].message.reasoning_content == 'reason'
    assert result.choices[0].finish_reason == 'tool_calls'


@pytest.mark.asyncio
async def test_enforced_overrides_runtime_and_extra_body_without_mutation(monkeypatch):
    enforced = {'temperature': 0.1, 'extra_headers': {'SessionID': 'fixed'}}
    client, create, _ = make(monkeypatch, enforced=enforced)
    before = copy.deepcopy(enforced)
    await client._create_completion(
        [], temperature=0.9, extra_headers={'SessionID': 'runtime'}, extra_body={'temperature': 0.8}
    )
    sent = create.call_args.kwargs
    assert sent['temperature'] == 0.1
    assert sent['extra_headers'] == {'SessionID': 'fixed'}
    assert 'temperature' not in sent.get('extra_body', {})
    assert enforced == before


@pytest.mark.asyncio
async def test_transport_retry_reuses_id_but_distinct_calls_do_not(monkeypatch):
    client, create, _ = make(monkeypatch, args={'inject_request_id': True})
    ids = []

    async def send(**kwargs):
        ids.append(kwargs['extra_body']['request_id'])
        if len(ids) == 1:
            raise openai.APIConnectionError(request=httpx.Request('POST', 'https://test'))
        return response()

    create.side_effect = send
    await client._create_completion([])
    await client._create_completion([])
    assert ids[0] == ids[1]
    assert ids[1] != ids[2]


@pytest.mark.asyncio
@pytest.mark.parametrize('pin', [None, 'typed', 'extra_body'])
async def test_context_adjustment_respects_enforced_output_budget(monkeypatch, pin):
    enforced = (
        {'max_tokens': 500} if pin == 'typed' else ({'extra_body': {'max_tokens': 500}} if pin == 'extra_body' else {})
    )
    client, create, _ = make(
        monkeypatch,
        args={'adaptive_context_max_tokens': True, 'inject_request_id': True, 'context_token_margin': 50},
        request={'max_tokens': 500},
        enforced=enforced,
    )
    create.side_effect = [bad_request(), response()]
    if pin:
        with pytest.raises(ContextLengthExceededError):
            await client._create_completion([])
        assert create.await_count == 1
    else:
        await client._create_completion([])
        first, second = [call.kwargs for call in create.call_args_list]
        assert second['max_tokens'] == 150
        assert first['extra_body']['request_id'] != second['extra_body']['request_id']


@pytest.mark.asyncio
async def test_adaptive_retry_replaces_extra_body_budget_and_shares_attempt_limit(monkeypatch):
    client, create, _ = make(
        monkeypatch,
        args={
            'adaptive_context_max_tokens': True,
            'context_token_margin': 50,
            'max_retries': 2,
        },
        request={'max_tokens': 500, 'extra_body': {'max_tokens': 600}},
    )
    create.side_effect = [bad_request(), response()]
    await client._create_completion([])
    final = create.call_args.kwargs
    assert final['max_tokens'] == 150
    assert 'max_tokens' not in final['extra_body']
    create.reset_mock()
    create.side_effect = [openai.APIConnectionError(request=httpx.Request('POST', 'https://test')), bad_request()]
    with pytest.raises(ContextLengthExceededError):
        await client._create_completion([])
    assert create.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize('status', [400, 401, 403])
async def test_task_deterministic_errors_do_not_retry(monkeypatch, status):
    client, create, _ = make(monkeypatch)
    error = openai.APIStatusError(
        'rejected', response=httpx.Response(status, request=httpx.Request('POST', 'https://test')), body={}
    )
    create.side_effect = error
    with pytest.raises(openai.APIStatusError):
        await client._create_completion([])
    assert create.await_count == 1


def test_external_form_does_not_drop_enforced_policy():
    from nexus.extensions.tasks.os_world.scripts.export_platform_dataset import _client_form

    item = SimpleNamespace(
        config=SimpleNamespace(
            openai_clients={
                'main': {
                    'name': 'infer_proxy',
                    'config': {'request_args': {'model': 'service'}, 'enforced_request_args': {'temperature': 0.1}},
                }
            }
        )
    )
    with pytest.raises(ValueError, match='cannot represent'):
        _client_form(item)


@pytest.mark.asyncio
async def test_task_rejects_bad_tools_before_http_and_cancellation_is_not_retried(monkeypatch):
    client, create, _ = make(monkeypatch)
    with pytest.raises(ValueError, match='function tools'):
        await client._create_completion([], tools=[{'type': 'web_search'}])
    create.assert_not_awaited()
    create.side_effect = asyncio.CancelledError
    with pytest.raises(asyncio.CancelledError):
        await client._create_completion([])
    assert create.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('harness', ['cc', 'codex'])
@pytest.mark.parametrize('platform', [False, True])
@pytest.mark.parametrize('stream', [False, True])
async def test_real_client_through_harness_proxy_two_tool_rounds(monkeypatch, harness, platform, stream):
    from nexus.runtime.bridge.adapters import ags_osworld
    from nexus.runtime.bridge.adapters.ags_osworld import LLMRequestHandler

    backend_calls = []
    calls = [{'id': 'call1', 'type': 'function', 'function': {'name': 'Bash', 'arguments': '{"command":"echo ok"}'}}]
    if platform:
        import importlib.util

        if importlib.util.find_spec('nexus_bridge') is None:
            pytest.skip('Set PYTHONPATH to the actual nexus-bridge src for platform compatibility tests')
        import litellm

        from nexus_bridge.modules.litellm_client import LiteLLMClient

        async def complete(**kwargs):
            backend_calls.append(kwargs)
            return litellm.ModelResponse(
                **json.loads(
                    response(
                        'ok' if len(backend_calls) > 1 else None, calls if len(backend_calls) == 1 else None
                    ).model_dump_json()
                )
            )

        monkeypatch.setattr(litellm, 'acompletion', complete)
        client = LiteLLMClient(
            OpenAIConfig(
                client_args={'api_key': 'fixture', 'base_url': 'http://fixture', 'max_retries': 1},
                request_args={'model': 'openai/service'},
            )
        )
    else:
        client, create, _ = make(monkeypatch, standard.InferProxyClient)

        async def complete(**kwargs):
            backend_calls.append(kwargs)
            return response('ok' if len(backend_calls) > 1 else None, calls if len(backend_calls) == 1 else None)

        create.side_effect = complete

    handler = LLMRequestHandler(runtime=object(), openai_client=client)
    if platform:
        assert handler.openai_client is client
    else:
        assert handler.openai_client is not client
        assert client.client.max_retries == 2

    class Bridge:
        async def request(self, body, **kwargs):
            return await handler._produce_response('round', body, rsio_mode=False)

    protocol = 'anthropic' if harness == 'cc' else 'responses-chat'
    app, bridge = ags_osworld.create_bridge_fastapi_app(api_protocol=protocol)
    monkeypatch.setattr(bridge, 'request', Bridge().request)
    tool = {'name': 'Bash', 'description': 'shell', 'input_schema': {'type': 'object'}}
    if harness == 'cc':
        first = {
            'model': 'facade',
            'max_tokens': 100,
            'messages': [{'role': 'user', 'content': 'test'}],
            'tools': [tool],
        }
        url = '/v1/messages'
    else:
        first = {
            'model': 'facade',
            'input': 'test',
            'tools': [
                {'type': 'function', 'name': 'Bash', 'description': 'shell', 'parameters': {'type': 'object'}},
            ],
        }
        url = '/v1/responses'
    first['stream'] = stream
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as session:
            result = await session.post(url, json=first)
            assert result.status_code == 200, result.text
            assert 'call1' in result.text
            if harness == 'cc':
                first['messages'] += [
                    {
                        'role': 'assistant',
                        'content': [
                            {'type': 'tool_use', 'id': 'call1', 'name': 'Bash', 'input': {'command': 'echo ok'}}
                        ],
                    },
                    {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 'call1', 'content': 'ok'}]},
                ]
            else:
                first['input'] = [
                    {'type': 'message', 'role': 'user', 'content': 'test'},
                    {'type': 'function_call', 'call_id': 'call1', 'name': 'Bash', 'arguments': '{"command":"echo ok"}'},
                    {'type': 'function_call_output', 'call_id': 'call1', 'output': 'ok'},
                ]
            result = await session.post(url, json=first)
            assert result.status_code == 200, result.text
            assert len(backend_calls) == 2
            assert backend_calls[-1]['messages'][-1]['role'] == 'tool'
            assert backend_calls[-1]['model'] == ('openai/service' if platform else 'service')
            assert 'adaptive_context_max_tokens' not in backend_calls[-1]
            assert 'fixed_request_args' not in backend_calls[-1]
    finally:
        bridge.close()
        await handler.aclose()
