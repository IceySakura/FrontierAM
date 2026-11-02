import json

import httpx
import pytest

from nexus.core import OpenAIConfig
from nexus.extensions.tasks.mac_os_world.clients import MacOSWorldOpenAIClient


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


@pytest.mark.parametrize('request_id_key', [None, 'x-request-id'])
async def test_platform_request_id_metadata_does_not_leak_to_sdk_or_request(request_id_key):
    requests = []

    async def handler(request):
        requests.append(request)
        return httpx.Response(200, json=_completion())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        config = OpenAIConfig(
            client_args={
                'api_key': 'test-key',
                'base_url': 'https://provider.test/v1',
                'http_client': http_client,
                'request_id_key': request_id_key,
                'sdk_max_retries': 0,
            },
            request_args={'model': 'test-model', 'temperature': 1, 'top_p': 0.9},
        )
        client = MacOSWorldOpenAIClient(config)
        response = await client.create_completion([{'role': 'user', 'content': 'hello'}])

    assert response.choices[0].message.content == 'done'
    assert config.client_args['request_id_key'] == request_id_key
    assert len(requests) == 1
    assert requests[0].url == 'https://provider.test/v1/chat/completions'
    assert json.loads(requests[0].content) == {
        'model': 'test-model',
        'messages': [{'role': 'user', 'content': 'hello'}],
        'temperature': 1,
        'top_p': 0.9,
    }


@pytest.mark.parametrize('thinking', [None, {'type': 'enabled', 'budget_tokens': 1024}])
def test_anthropic_local_tool_conversion_preserves_payload_and_input(thinking):
    import copy

    from nexus.extensions.openai_clients.vertex import _build_payload
    from nexus.extensions.tasks.mac_os_world.clients import MacOSWorldAnthropicClient

    client = object.__new__(MacOSWorldAnthropicClient)
    args = {
        'model': 'claude-3-7-sonnet',
        'max_tokens': 4096,
        'thinking': thinking,
        'tool_choice': {'type': 'function', 'function': {'name': 'computer'}},
        'tools': [
            {
                'type': 'computer_20251124',
                'function': {'name': 'computer', 'parameters': {'display_width_px': 1024, 'display_height_px': 768}},
                'cache_control': {'type': 'ephemeral'},
            }
        ],
        'temperature': 1,
        'top_p': 0.9,
    }
    messages = [{'role': 'user', 'content': 'task'}]
    original = copy.deepcopy(args)
    assert client._build_request_payload(messages=messages, request_args=args) == _build_payload(
        messages=messages, request_args=args
    )
    assert args == original
    args['tools'] = [{'type': 'computer_toolset_20260801', 'cache_control': {'type': 'ephemeral'}}]
    original = copy.deepcopy(args)
    result = client._build_request_payload(messages=messages, request_args=args)
    assert result['tools'] == args['tools']
    assert result['tools'] is not args['tools']
    assert args == original
    with pytest.raises(ValueError, match='Unsupported Anthropic native tool'):
        _build_payload(messages=messages, request_args=args)
