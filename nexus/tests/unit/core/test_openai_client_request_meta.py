"""request_meta forwarding gate on the OpenAI-compatible client base class.

Identity metadata (agent_id/session_id lifted by the anthropic proxy) rides
create_completion as a named kwarg. Only clients that declare
``supports_request_meta`` receive it in ``_create_completion``; for every
other provider client the base class swallows it so SDKs never see an
unknown field.
"""

from __future__ import annotations

from typing import Any

import pytest

from nexus.core.openai_client import OpenAICompatibleClient, OpenAIConfig


class _RecordingClient(OpenAICompatibleClient):
    def __init__(self):
        super().__init__(OpenAIConfig(client_args={}, request_args={}))
        self.seen_kwargs: dict[str, Any] | None = None

    async def _create_completion(self, messages, model=None, **kwargs):
        self.seen_kwargs = kwargs
        return {'id': 'x'}

    async def _count_tokens(self, **kwargs):  # pragma: no cover - abstract filler
        raise NotImplementedError


class _MetaAwareClient(_RecordingClient):
    supports_request_meta = True


@pytest.mark.asyncio
async def test_default_client_never_sees_request_meta():
    client = _RecordingClient()
    await client.create_completion([{'role': 'user', 'content': 'hi'}], request_meta={'agent_id': 'a1'})
    assert 'request_meta' not in client.seen_kwargs


@pytest.mark.asyncio
async def test_opted_in_client_receives_request_meta():
    client = _MetaAwareClient()
    await client.create_completion([{'role': 'user', 'content': 'hi'}], request_meta={'agent_id': 'a1'})
    assert client.seen_kwargs.get('request_meta') == {'agent_id': 'a1'}


@pytest.mark.asyncio
async def test_absent_request_meta_is_not_injected():
    client = _MetaAwareClient()
    await client.create_completion([{'role': 'user', 'content': 'hi'}])
    assert 'request_meta' not in client.seen_kwargs
