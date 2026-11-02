from __future__ import annotations

import asyncio
import hashlib
import json
import uuid

from functools import wraps
from types import SimpleNamespace
from urllib.parse import parse_qsl

import httpx
import pytest

from nexus.core import OpenAICompatibleClient, OpenAIConfig, TaskRunInput, TaskRunOutput, task_registry
from nexus.core.context import require_context
from nexus.extensions.openai_clients.azure_openai import AzureOpenAIClient
from nexus.extensions.openai_clients.cache_task_id import observe_client_init, prepare_cache_task_id
from nexus.extensions.openai_clients.infer_proxy import InferProxyClient
from nexus.extensions.openai_clients.openai import OpenAIClient
from nexus.extensions.openai_clients.openai_responses import OpenAIResponsesClient
from nexus.runner.single import RunHooks, RunInput, RunOutput, run
from nexus.trace import SpanProcessor, processor_context, span_context
from nexus.utils.cache import cache_context


class CaptureSpans(SpanProcessor):
    def __init__(self):
        self.spans = []

    def on_end(self, span, _event):
        self.spans.append(span)


@pytest.mark.parametrize('explicit', [False, True])
async def test_openai_reports_the_sent_cache_id_on_initialization_and_calls(explicit):
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={
                'id': 'test',
                'object': 'chat.completion',
                'created': 0,
                'model': 'test',
                'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'ok'}, 'finish_reason': 'stop'}],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http_client:
        args = {'api_key': 'credential?provider=test&cache_task_id=generator', 'http_client': http_client}
        if explicit:
            args['cache_task_id'] = 'task/a + b'
        config = OpenAIConfig(client_args=args, request_args={'model': 'test'})
        client = OpenAIClient(config)
        original = client._create_completion
        capture = CaptureSpans()
        expected = 'task/a + b' if explicit else 'generator'
        with processor_context([capture]):
            assert observe_client_init(client) == expected
            await client.create_completion([])
            await client.create_completion([])

    assert client._create_completion == original
    assert client.config.client_args is args
    assert args['api_key'].endswith('cache_task_id=generator')
    assert [span.name for span in capture.spans] == ['openai_client_init', 'openai_completion', 'openai_completion']
    assert capture.spans[0].attributes['cache_task_id'] == expected
    assert not {'api_key', 'base_url', '_client_key'} & capture.spans[0].attributes.keys()
    assert 'credential' not in capture.spans[0].model_dump_json()
    assert all(span.attributes['cache_task_id'] == expected for span in capture.spans[1:])
    for request in requests:
        assert dict(parse_qsl(request.headers['authorization'].partition('?')[2])) == {
            'provider': 'test',
            'cache_task_id': expected,
        }
        assert 'cache_task_id' not in json.loads(request.content)


@pytest.mark.parametrize('api_key', [None, 'credential', 'credential?cache_task_id='])
def test_init_reports_null_without_a_cache_id(api_key):
    capture = CaptureSpans()
    with processor_context([capture]):
        assert observe_client_init(SimpleNamespace(client=SimpleNamespace(api_key=api_key))) is None
    assert capture.spans[0].name == 'openai_client_init'
    assert json.loads(capture.spans[0].model_dump_json())['attributes']['cache_task_id'] is None


@pytest.mark.parametrize('failure', [False, True])
async def test_reused_client_records_current_id_on_calls_and_replay(failure):
    calls = []
    error = RuntimeError('provider failed')

    class Client(OpenAICompatibleClient):
        async def _create_completion(self, messages, model=None, **kwargs):
            calls.append(messages)
            if failure:
                raise error
            return

    client = Client(OpenAIConfig(client_args={'api_key': 'credential?cache_task_id=old'}, request_args={}))
    original = client._create_completion
    observe_client_init(client)
    client.config.client_args['api_key'] = 'credential?cache_task_id=current'
    observe_client_init(client)
    assert client._create_completion == original
    capture = CaptureSpans()
    cache = {}
    with processor_context([capture]), cache_context(cache):
        if failure:
            with pytest.raises(RuntimeError) as caught:
                await client.create_completion([])
            assert caught.value is error
        else:
            await client.create_completion([])
    assert len(calls) == 1
    assert capture.spans[0].attributes['cache_task_id'] == 'current'
    if not failure:
        with processor_context([capture]), cache_context(cache):
            await client.create_completion([])
        assert len(calls) == 1
        assert capture.spans[-1].attributes['cache_task_id'] == 'current'
        with processor_context([capture]), span_context('unrelated'):
            await client._create_completion([])
        assert 'cache_task_id' not in capture.spans[-1].attributes


@pytest.mark.parametrize('cache_task_id', [None, 'configured'])
async def test_self_attrs_preserves_existing_injection_and_omits_missing_id(cache_task_id):
    calls = []

    class Client(OpenAICompatibleClient):
        async def _create_completion(self, messages, model=None, **kwargs):
            calls.append('provider')

    client = Client(
        OpenAIConfig(
            client_args={'api_key': f'credential?cache_task_id={cache_task_id}' if cache_task_id else 'credential'},
            request_args={},
        )
    )
    original = client._create_completion

    @wraps(original)
    async def injected(*args, **kwargs):
        calls.append('injection')
        return await original(*args, **kwargs)

    client._create_completion = injected
    observe_client_init(client)
    observe_client_init(client)
    assert client._create_completion is injected
    capture = CaptureSpans()
    with processor_context([capture]):
        await client.create_completion([])
    assert calls == ['injection', 'provider']
    if cache_task_id:
        assert capture.spans[0].attributes['cache_task_id'] == cache_task_id
    else:
        assert 'cache_task_id' not in capture.spans[0].attributes


async def test_cache_id_isolated_between_concurrent_calls_without_affecting_child_spans():
    class Client(OpenAICompatibleClient):
        async def _create_completion(self, messages, model=None, **kwargs):
            await asyncio.sleep(0)
            with span_context(messages[0]['content']):
                await asyncio.sleep(0)

    clients = [
        Client(OpenAIConfig(client_args={'api_key': f'credential?cache_task_id={key}'}, request_args={}))
        for key in ('main', 'judge')
    ]
    capture = CaptureSpans()
    with processor_context([capture]):
        await asyncio.gather(
            *(
                client.create_completion([{'role': 'user', 'content': key}])
                for client, key in zip(clients, ('main', 'judge'), strict=True)
            )
        )
        with span_context('after_calls'):
            pass
    for span in capture.spans:
        if span.name == 'openai_completion':
            assert span.attributes['cache_task_id'] == span.attributes['inputs']['messages'][0]['content']
        else:
            assert 'cache_task_id' not in span.attributes


async def test_responses_records_input_and_hashed_final_id():
    client = OpenAIResponsesClient(
        OpenAIConfig(
            client_args={'api_key': 'credential', 'cache_task_id': 'explicit'},
            request_args={},
        )
    )
    try:
        assert client.input_cache_task_id == 'explicit'
        assert observe_client_init(client) == hashlib.md5(b'explicit').hexdigest()
        assert client.gateway_cache_task_id == hashlib.md5(b'explicit').hexdigest()
    finally:
        await client.client.close()


@pytest.mark.parametrize(
    ('client_cls', 'options'),
    [
        (AzureOpenAIClient, {'azure_endpoint': 'https://example.invalid', 'api_version': '2024-10-21'}),
        (InferProxyClient, {}),
    ],
)
async def test_other_sdk_clients_send_and_record_explicit_id(client_cls, options):
    client = client_cls(
        OpenAIConfig(
            client_args={'api_key': 'credential?provider=test', 'cache_task_id': 'explicit', **options},
            request_args={},
        )
    )
    try:
        assert observe_client_init(client) == 'explicit'
    finally:
        if getattr(client, 'client', None) is not None:
            await client.client.close()


def test_resolved_client_args_take_precedence_over_original_configuration():
    client = SimpleNamespace(
        client_args={'api_key': 'credential?cache_task_id=actual'},
        config=OpenAIConfig(client_args={'api_key': 'credential?cache_task_id=old'}, request_args={}),
    )
    assert observe_client_init(client) == 'actual'
    assert observe_client_init(SimpleNamespace()) is None


@pytest.mark.parametrize('args', [{'cache_task_id': 'id'}, {'api_key': 'credential', 'cache_task_id': 123}])
def test_invalid_explicit_configuration_is_rejected(args):
    with pytest.raises((TypeError, ValueError)):
        prepare_cache_task_id(args)


@pytest.mark.parametrize('failure', [False, True])
@pytest.mark.parametrize('ids', [[None], ['one'], ['one', 'two']])
async def test_runner_records_init_before_calls_and_persists_ids(ids, failure, monkeypatch):
    name = f'cache_task_init_{uuid.uuid4().hex}'
    capture = CaptureSpans()

    async def complete(self, messages, model=None, **kwargs):
        return None

    monkeypatch.setattr(OpenAIClient, '_create_completion', complete)

    @task_registry.register(name)
    async def task(_input: TaskRunInput) -> TaskRunOutput:
        assert [span.attributes['cache_task_id'] for span in capture.spans] == ids
        for client in require_context().openai_clients.values():
            await client.create_completion([])
        if failure:
            raise RuntimeError('task failed')
        return TaskRunOutput()

    clients = {
        str(index): {
            'name': 'openai',
            'config': {
                'client_args': {'api_key': 'credential', 'cache_task_id': value},
                'request_args': {},
            },
        }
        for index, value in enumerate(ids)
    }
    result = await run(RunInput(task_name=name, openai_clients=clients), hooks=RunHooks(span_processors=[capture]))
    assert result.success is not failure
    if failure:
        assert result.error.message == 'task failed'
    init_spans = [span for span in capture.spans if span.name == 'openai_client_init']
    assert [span.attributes['cache_task_id'] for span in init_spans] == ids
    call_spans = [span for span in capture.spans if span.name == 'openai_completion']
    assert [span.attributes.get('cache_task_id') for span in call_spans] == ids
    assert all(
        'cache_task_id' not in span.attributes for span, value in zip(call_spans, ids, strict=True) if value is None
    )
    payload = result.model_dump(mode='json')
    expected = (
        {}
        if ids == [None]
        else (
            {'cache_task_id': ids[0]} if len(ids) == 1 else {'cache_task_ids': {str(k): v for k, v in enumerate(ids)}}
        )
    )
    assert {key: value for key, value in payload.items() if key in {'cache_task_id', 'cache_task_ids'}} == expected
    assert {key: value for key, value in payload.items() if key in {'input_cache_task_id', 'input_cache_task_ids'}} == {
        'input_' + key: value for key, value in expected.items()
    }
    assert RunOutput.model_validate(payload).model_dump(mode='json') == payload


@pytest.mark.parametrize('source', ['request_args', 'call', 'enforced_request_args', '_request_args_override'])
@pytest.mark.parametrize('effective_id', ['actual/a + b', None])
@pytest.mark.parametrize('failure', [False, True])
async def test_litellm_records_final_request_id_in_span_client_and_output(source, effective_id, failure, monkeypatch):
    import litellm

    from nexus.distributed.export.results import _project_cache_task_ids
    from nexus.extensions.openai_clients import litellm as litellm_client

    requests = []

    async def complete(**kwargs):
        requests.append(kwargs)
        if failure:
            raise RuntimeError('provider failed')
        return litellm.ModelResponse(
            choices=[{'index': 0, 'message': {'role': 'assistant', 'content': 'ok'}, 'finish_reason': 'stop'}]
        )

    monkeypatch.setattr(litellm_client.litellm, 'acompletion', complete)
    override = {'api_key': 'override?cache_task_id=actual%2Fa%20%2B%20b' if effective_id else 'override'}
    config = {
        'client_args': {'api_key': 'credential?cache_task_id=configured', 'max_retries': 1},
        'request_args': {'model': 'openai/test'},
    }
    call_args = {}
    if source in ('request_args', 'enforced_request_args'):
        config.setdefault(source, {}).update(override)
    elif source == '_request_args_override':
        call_args[source] = {'model': 'openai/test', **override}
    else:
        call_args = override
    name = f'litellm_effective_cache_id_{uuid.uuid4().hex}'
    capture = CaptureSpans()

    @task_registry.register(name)
    async def task(_input: TaskRunInput) -> TaskRunOutput:
        client = require_context().openai_clients['main']
        try:
            await client.create_completion([], **call_args)
        finally:
            assert client.cache_task_id == effective_id
            assert client.input_cache_task_id == 'configured'
        return TaskRunOutput()

    result = await run(
        RunInput(task_name=name, openai_clients={'main': {'name': 'litellm', 'config': config}}),
        hooks=RunHooks(span_processors=[capture]),
    )
    assert result.success is not failure
    if failure:
        assert result.error.message == 'provider failed'
    assert requests[0]['api_key'] == override['api_key']
    init = next(span for span in capture.spans if span.name == 'openai_client_init')
    completion = next(span for span in capture.spans if span.name == 'openai_completion')
    assert init.attributes['cache_task_id'] == 'configured'
    assert completion.attributes['cache_task_id'] == effective_id
    assert result.cache_task_id == effective_id
    assert result.input_cache_task_id == 'configured'
    metadata = _project_cache_task_ids(output=result.model_dump(mode='json'), metadata={}, run_id='test')
    assert metadata['_nexus'].get('cache_task_id') == effective_id


@pytest.mark.parametrize('keys', [['main'], ['main', 'judge']])
@pytest.mark.parametrize('capture_spans', [False, True])
async def test_responses_input_and_final_ids_survive_result_serialization(keys, capture_spans, monkeypatch):
    from nexus.distributed.artifacts.run_result import decode_run_result_content, encode_run_result_content

    async def complete(self, messages, model=None, **kwargs):
        return None

    monkeypatch.setattr(OpenAIResponsesClient, '_create_completion', complete)
    name = f'cache_id_pairs_{uuid.uuid4().hex}'

    @task_registry.register(name)
    async def task(_input: TaskRunInput) -> TaskRunOutput:
        for client in require_context().openai_clients.values():
            await client.create_completion([])
        return TaskRunOutput()

    inputs = {key: f'{key}-input' for key in keys}
    final = {key: hashlib.md5(value.encode()).hexdigest() for key, value in inputs.items()}
    capture = CaptureSpans()
    result = await run(
        RunInput(
            task_name=name,
            openai_clients={
                key: {
                    'name': 'openai_responses',
                    'config': {
                        'client_args': {'api_key': f'credential?cache_task_id=generator-{key}', 'cache_task_id': value},
                        'request_args': {},
                    },
                }
                for key, value in inputs.items()
            },
        ),
        hooks=RunHooks(span_processors=[capture] if capture_spans else []),
    )
    assert result.success
    payload = result.model_dump(mode='json')
    if len(keys) == 1:
        assert payload['input_cache_task_id'] == inputs['main']
        assert payload['cache_task_id'] == final['main']
    else:
        assert payload['input_cache_task_ids'] == inputs
        assert payload['cache_task_ids'] == final
    assert (
        decode_run_result_content(encode_run_result_content(status='succeeded', output=payload, error=None))['output']
        == payload
    )
    if capture_spans:
        init = [span for span in capture.spans if span.name == 'openai_client_init']
        calls = [span for span in capture.spans if span.name == 'openai_completion']
        assert [span.attributes['input_cache_task_id'] for span in init] == list(inputs.values())
        assert [span.attributes['cache_task_id'] for span in init] == list(final.values())
        assert {span.attributes['_client_key']: span.attributes['input_cache_task_id'] for span in calls} == inputs
        assert {span.attributes['_client_key']: span.attributes['cache_task_id'] for span in calls} == final
