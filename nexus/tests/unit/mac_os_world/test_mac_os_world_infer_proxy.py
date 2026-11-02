import copy
import json

import httpx
import pytest

from nexus.core import OpenAIConfig
from nexus.extensions.tasks.mac_os_world.clients import MacOSWorldInferProxyClient as InferProxyClient


@pytest.mark.asyncio
@pytest.mark.parametrize('normalize', [False, True])
async def test_original_detail_compatibility_preserves_tool_images_and_source(normalize):
    sent = []

    async def transport(request):
        sent.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                'id': 'reply',
                'object': 'chat.completion',
                'created': 1,
                'model': 'hy',
                'choices': [{'index': 0, 'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': 'ok'}}],
            },
        )

    http = httpx.AsyncClient(transport=httpx.MockTransport(transport))
    client = InferProxyClient(
        OpenAIConfig(
            client_args={
                'api_key': 'test',
                'base_url': 'https://fixture/v1',
                'http_client': http,
                'normalize_image_detail': normalize,
            },
            request_args={'model': 'hy'},
        )
    )
    messages = [
        {
            'role': 'tool',
            'tool_call_id': 'image-call',
            'content': [
                {'type': 'text', 'text': 'screenshot'},
                {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AAAA', 'detail': 'original'}},
                {'type': 'image_url', 'image_url': {'url': 'https://fixture/image', 'detail': 'high'}},
            ],
        }
    ]
    original = copy.deepcopy(messages)
    try:
        await client._create_completion(messages)
    finally:
        await http.aclose()
    assert messages == original
    expected = copy.deepcopy(original)
    if normalize:
        expected[0]['content'][1]['image_url'].pop('detail')
    assert sent[0]['messages'] == expected
    assert 'normalize_image_detail' not in sent[0]
