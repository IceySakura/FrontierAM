"""No-Auth OpenAI Client for internal LLM proxies.

Some internal LLM proxies (e.g., http://llm-api.model-eval.woa.com/v1) require
that NO Authorization header is sent — they use their own backend keys internally.
The standard OpenAI SDK and litellm always inject "Bearer <key>" headers, which
causes these proxies to fail.

This client wraps AsyncOpenAI with an httpx event hook that strips the auth header,
making it compatible with these internal proxies while keeping the standard nexus
OpenAICompatibleClient interface.

Usage in generator config:
    'openai_clients': {
        'main': {
            'name': 'noauth_openai',
            'config': {
                'client_args': {
                    'base_url': 'http://llm-api.model-eval.woa.com/v1',
                },
                'request_args': {
                    'model': 'api_azure_openai_gpt-5.4-2026-03-05',
                },
            },
        },
    }
"""

from typing import Any

import httpx

from nexus.core import OpenAICompatibleClient, OpenAIConfig, openai_client_registry
from nexus.core.types import ExtendedChatCompletion
from nexus.utils.common import configurable_retry


async def _strip_auth_header(request: httpx.Request) -> None:
    """Remove Authorization header before sending to proxy."""
    if 'authorization' in request.headers:
        # httpx Headers are case-insensitive
        del request.headers['authorization']


@openai_client_registry.register('noauth_openai')
class NoAuthOpenAIClient(OpenAICompatibleClient):
    """OpenAI-compatible client that does NOT send Authorization headers.

    For internal proxies that manage their own backend API keys and reject
    client-supplied auth.
    """

    def __init__(self, config: OpenAIConfig):
        super().__init__(config)
        self._http_client: httpx.AsyncClient | None = None
        self._openai_client: Any = None

    def _get_client(self) -> Any:
        if self._openai_client is None:
            from openai import AsyncOpenAI

            self._http_client = httpx.AsyncClient(
                event_hooks={'request': [_strip_auth_header]},
                timeout=httpx.Timeout(300.0, connect=60.0),
            )
            self._openai_client = AsyncOpenAI(
                api_key='not-needed',
                base_url=self.config.client_args.get('base_url', 'http://llm-api.model-eval.woa.com/v1'),
                http_client=self._http_client,
                max_retries=0,  # We handle retries ourselves
            )
        return self._openai_client

    async def _create_completion(self, messages, model=None, **kwargs):
        client = self._get_client()

        request_kwargs = {
            **self.config.request_args,
            **kwargs,
            'messages': messages,
        }
        if model:
            request_kwargs['model'] = model
        request_kwargs = self._apply_enforced_request_args(request_kwargs)

        max_retries = self.config.client_args.get('max_retries', 10)

        async def _call():
            return await client.chat.completions.create(**request_kwargs)

        response = await configurable_retry(max_retries)(_call)()
        return ExtendedChatCompletion.model_validate(response.model_dump())
