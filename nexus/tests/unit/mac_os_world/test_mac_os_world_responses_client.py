import openai
import pytest

from nexus.core import OpenAIConfig
from nexus.extensions.tasks.mac_os_world.clients import MacOSWorldResponsesClient


@pytest.mark.asyncio
@pytest.mark.parametrize('request_id_key', [None, 'request_id'])
async def test_platform_request_id_key_does_not_break_sdk_initialization(request_id_key: str | None) -> None:
    config = OpenAIConfig(
        client_args={
            'api_key': 'test-key',
            'base_url': 'https://api.openai.test/v1',
            'request_id_key': request_id_key,
        },
        request_args={'model': 'gpt-cua'},
    )
    client = MacOSWorldResponsesClient(config)
    try:
        assert isinstance(client.client, openai.AsyncOpenAI)
        assert client.client.api_key == 'test-key'
        assert str(client.client.base_url) == 'https://api.openai.test/v1/'
        assert config.client_args['request_id_key'] == request_id_key
    finally:
        await client.client.close()
