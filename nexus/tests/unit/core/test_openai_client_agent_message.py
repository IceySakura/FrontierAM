"""Codex ``agent_message`` items on the OpenAI-compatible client base class.

Clients that do not declare ``supports_agent_message`` receive them as user turns.
The caller's list, which the trace records, keeps the native item.
"""

from __future__ import annotations

import copy

import pytest

from nexus.core.openai_client import OpenAICompatibleClient, OpenAIConfig
from nexus.core.types import TokenCountResult


AGENT_MESSAGE = {
    'type': 'agent_message',
    'id': 'amsg_child_task',
    'author': '/root',
    'recipient': '/root/child',
    'content': [
        {'type': 'input_text', 'text': 'Message Type: NEW_TASK\nTask name: /root/child\nSender: /root\nPayload:\n'},
        {'type': 'encrypted_content', 'encrypted_content': 'Fix the failing test.'},
    ],
}


class _RecordingClient(OpenAICompatibleClient):
    def __init__(self):
        super().__init__(OpenAIConfig(client_args={}, request_args={}))
        self.seen_messages = None

    async def _create_completion(self, messages, model=None, **kwargs):
        self.seen_messages = messages
        return {'id': 'x'}

    async def _count_tokens(self, *, messages, tools=None, add_generation_prompt=True):
        del tools, add_generation_prompt
        self.seen_messages = messages
        return TokenCountResult(input_tokens=0)


class _NativeClient(_RecordingClient):
    supports_agent_message = True


@pytest.mark.asyncio
async def test_agent_message_reaches_a_chat_client_as_a_user_turn_and_the_input_stays_native():
    messages = [{'role': 'user', 'content': '<environment_context/>'}, copy.deepcopy(AGENT_MESSAGE)]
    recorded = copy.deepcopy(messages)
    client = _RecordingClient()

    await client.create_completion(messages)

    assert client.seen_messages == [
        messages[0],
        {
            'role': 'user',
            'content': 'Message Type: NEW_TASK\nTask name: /root/child\nSender: /root\nPayload:\nFix the failing test.',
        },
    ]
    assert messages == recorded

    await client.count_tokens(messages=messages)
    assert client.seen_messages == [
        messages[0],
        {
            'role': 'user',
            'content': 'Message Type: NEW_TASK\nTask name: /root/child\nSender: /root\nPayload:\nFix the failing test.',
        },
    ]
    assert messages == recorded


@pytest.mark.asyncio
async def test_agent_message_passes_through_to_a_native_client():
    client = _NativeClient()
    messages = [copy.deepcopy(AGENT_MESSAGE)]

    await client.create_completion(messages)
    assert client.seen_messages == [AGENT_MESSAGE]

    await client.count_tokens(messages=messages)
    assert client.seen_messages == [AGENT_MESSAGE]
    assert messages == [AGENT_MESSAGE]
