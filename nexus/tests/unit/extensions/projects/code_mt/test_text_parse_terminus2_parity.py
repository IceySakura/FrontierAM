from __future__ import annotations

import asyncio

from types import SimpleNamespace

import pytest

from nexus.extensions.agents.terminus2 import chat as terminus_chat
from nexus.extensions.openai_clients.hytuner import HyTunerClient
from nexus.extensions.projects.code_mt.agent import text_parse_loop
from nexus.extensions.projects.code_mt.agent.text_parse_helpers import chat as code_mt_chat
from nexus.utils.timeout import InferenceTimeoutError, TimeBudget


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


@pytest.mark.parametrize('chat_module', [terminus_chat, code_mt_chat])
def test_reasoning_only_response_preserves_hytuner_replay_parity(monkeypatch, chat_module) -> None:
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

    cached_history = [
        {'role': 'user', 'content': 'next'},
        result.raw_message,
    ]
    replayed_history = cached_history + [{'role': 'user', 'content': 'terminal output'}]
    assert HyTunerClient._find_common_prefix(replayed_history, cached_history) == len(cached_history)


def test_code_mt_output_length_retry_replays_exact_assistant_message() -> None:
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

        async def chat(self, _prompt: str) -> code_mt_chat.LLMResponse:
            self.calls += 1
            if self.calls == 1:
                raise code_mt_chat.OutputLengthExceededError(
                    'length',
                    truncated_response='partial response',
                    raw_message=raw_message,
                )
            assert self.messages == [
                {'role': 'user', 'content': 'original prompt'},
                raw_message,
            ]
            return code_mt_chat.LLMResponse(content='{"analysis":"done","plan":"done","commands":[]}')

    chat = _RetryChat()
    result = asyncio.run(text_parse_loop._query_llm(chat, 'original prompt', parser_name='json'))

    assert result.content == '{"analysis":"done","plan":"done","commands":[]}'
    assert chat.calls == 2


def test_code_mt_chat_inference_budget_interrupts_unreturned_call(monkeypatch) -> None:
    class _HangingClient:
        async def create_completion(self, **_kwargs):
            await asyncio.Event().wait()

    monkeypatch.setattr(code_mt_chat, 'require_context', lambda: SimpleNamespace(openai_client=_HangingClient()))

    with pytest.raises(InferenceTimeoutError):
        asyncio.run(
            code_mt_chat.lite_llm_call(
                'next',
                inference_budget=TimeBudget(0.01),
            )
        )
