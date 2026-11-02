import hashlib
import importlib

from types import SimpleNamespace
from urllib.parse import parse_qsl, quote

import openai
import pytest

from nexus.core import OpenAIConfig, openai_client_registry
from nexus.extensions.openai_clients.cache_task_id import prepare_cache_task_id


@pytest.mark.parametrize(
    'name', ['openai', 'azure_openai', 'infer_proxy', 'litellm', 'openai_responses', 'aihub_responses']
)
@pytest.mark.parametrize('value', ['task/a + b', 'd1adaf6f1eef4c2a86ac8a56bf5ecfa2'])
def test_explicit_cache_id_replaces_all_query_values_without_rewriting_other_parameters(name, value, monkeypatch):
    importlib.import_module(f'nexus.extensions.openai_clients.{name}')
    sdk_args = []

    def sdk(**kwargs):
        sdk_args.append(kwargs)
        return SimpleNamespace(api_key=kwargs['api_key'])

    monkeypatch.setattr(openai, 'AsyncOpenAI', sdk)
    monkeypatch.setattr(openai, 'AsyncAzureOpenAI', sdk)
    api_key = 'credential?provider=one&cache_task_id=old&flag&provider=two&cache%5Ftask%5Fid=encoded&opaque=a%2fb+z&cache_task_id=older'
    args = {'api_key': api_key, 'cache_task_id': value}
    config = OpenAIConfig(client_args=args, request_args={})
    client = openai_client_registry.get(name)(config)
    if name == 'litellm':
        actual_key = client.client_args['api_key']
    else:
        actual_key = sdk_args[0]['api_key']
        assert 'cache_task_id' not in sdk_args[0]
    expected = hashlib.md5(value.encode()).hexdigest() if name.endswith('responses') else value
    assert (
        actual_key
        == f'credential?provider=one&flag&provider=two&opaque=a%2fb+z&cache_task_id={quote(expected, safe="")}'
    )
    assert [v for k, v in parse_qsl(actual_key.partition('?')[2]) if k == 'cache_task_id'] == [expected]
    assert client.input_cache_task_id == value
    assert client.cache_task_id == expected
    assert config.client_args['api_key'] == api_key
    assert config.client_args['cache_task_id'] == value


@pytest.mark.parametrize('value', [None, ''])
def test_missing_explicit_id_preserves_generator_query(value):
    api_key = 'credential?provider=a&cache_task_id=from%2Fgenerator&provider=b'
    args = {'api_key': api_key, 'cache_task_id': value}
    assert prepare_cache_task_id(args) == {'api_key': api_key}
    assert args == {'api_key': api_key, 'cache_task_id': value}


@pytest.mark.parametrize('source', ['request_args', 'call'])
async def test_explicit_cache_id_wins_when_litellm_request_overrides_api_key(source, monkeypatch):
    import litellm

    from nexus.extensions.openai_clients.litellm import LiteLLMClient

    requests = []

    async def complete(**kwargs):
        requests.append(kwargs)
        return litellm.ModelResponse(
            choices=[
                {
                    'index': 0,
                    'message': {'role': 'assistant', 'content': 'ok'},
                    'finish_reason': 'stop',
                }
            ]
        )

    monkeypatch.setattr(litellm, 'acompletion', complete)
    request_args = {'model': 'openai/test'}
    override = {'api_key': 'override?provider=a&cache_task_id=implicit&provider=b&cache%5Ftask%5Fid=old'}
    if source == 'request_args':
        request_args.update(override)
    client = LiteLLMClient(
        OpenAIConfig(
            client_args={'api_key': 'original', 'cache_task_id': 'explicit', 'max_retries': 1},
            request_args=request_args,
        )
    )
    await client.create_completion([], **(override if source == 'call' else {}))
    assert requests[0]['api_key'] == 'override?provider=a&provider=b&cache_task_id=explicit'
    assert client.cache_task_id == 'explicit'
