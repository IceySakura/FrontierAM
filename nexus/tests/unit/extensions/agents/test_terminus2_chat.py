from __future__ import annotations

import asyncio

from types import SimpleNamespace

import pytest

from nexus.extensions.agents.terminus2 import chat as chat_module
from nexus.extensions.agents.terminus2.agent import _query_llm
from nexus.extensions.agents.terminus2.chat import LLMResponse, OutputLengthExceededError


class _Message:
    def __init__(self, *, content: str | None, reasoning_content: str | None = None):
        self.role = 'assistant'
        self.content = content
        self.reasoning_content = reasoning_content

    def model_dump(self, *, exclude_none: bool = False) -> dict:
        result = {
            'role': self.role,
            'content': self.content,
            'reasoning_content': self.reasoning_content,
        }
        if exclude_none:
            return {key: value for key, value in result.items() if value is not None}
        return result


class _Client:
    def __init__(self, message: _Message, *, finish_reason: str):
        self.message = message
        self.finish_reason = finish_reason

    async def create_completion(self, **_kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=self.message,
                    finish_reason=self.finish_reason,
                )
            ],
            usage=None,
        )


def test_lite_llm_call_uses_reasoning_without_rewriting_raw_message(monkeypatch) -> None:
    message = _Message(content=None, reasoning_content='terminal action')
    client = _Client(message, finish_reason='stop')
    monkeypatch.setattr(chat_module, 'require_context', lambda: SimpleNamespace(openai_client=client))

    result = asyncio.run(chat_module.lite_llm_call('next'))

    assert result.content == 'terminal action'
    assert result.raw_message == {
        'role': 'assistant',
        'reasoning_content': 'terminal action',
    }
    assert message.content is None


def test_lite_llm_call_length_error_preserves_complete_raw_message(monkeypatch) -> None:
    message = _Message(content='partial response', reasoning_content='private reasoning')
    client = _Client(message, finish_reason='length')
    monkeypatch.setattr(chat_module, 'require_context', lambda: SimpleNamespace(openai_client=client))

    with pytest.raises(OutputLengthExceededError) as exc_info:
        asyncio.run(chat_module.lite_llm_call('next'))

    assert exc_info.value.truncated_response == 'partial response'
    assert exc_info.value.raw_message == {
        'role': 'assistant',
        'content': 'partial response',
        'reasoning_content': 'private reasoning',
    }


def test_output_length_retry_replays_exact_assistant_message() -> None:
    raw_message = {
        'role': 'assistant',
        'content': 'partial response',
        'reasoning_content': 'private reasoning',
        'provider_specific_fields': {'stable': 'metadata'},
    }

    class _RetryChat:
        def __init__(self):
            self.messages: list[dict] = []
            self.calls = 0

        async def chat(self, _prompt: str) -> LLMResponse:
            self.calls += 1
            if self.calls == 1:
                raise OutputLengthExceededError(
                    'length',
                    truncated_response='partial response',
                    raw_message=raw_message,
                )
            assert self.messages == [
                {'role': 'user', 'content': 'original prompt'},
                raw_message,
            ]
            return LLMResponse(content='{"analysis":"done","plan":"done","commands":[]}')

    chat = _RetryChat()
    result = asyncio.run(_query_llm(chat, 'original prompt', parser_name='json'))

    assert result.content == '{"analysis":"done","plan":"done","commands":[]}'
    assert chat.calls == 2
