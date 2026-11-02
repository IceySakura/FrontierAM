from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from nexus.core import OpenAIConfig
from nexus.extensions.openai_clients.hytuner import HyTunerClient


class CharacterTokenizer:
    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> str:
        del tokenize, tools, kwargs
        rendered = ''.join(f'<{message["role"]}>{message.get("content") or ""}' for message in messages)
        return rendered + ('<assistant>' if add_generation_prompt else '')

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(character) for character in text]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        del token_ids, skip_special_tokens
        return 'answer'


class ToolRenderingTokenizer(CharacterTokenizer):
    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> str:
        rendered = super().apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
            **kwargs,
        )
        tool_names = ','.join(str(tool.get('name')) for tool in (tools or []))
        return f'<tools>{tool_names}</tools>{rendered}'


class ServerManager:
    async def generate(self, **kwargs: Any) -> SimpleNamespace:
        del kwargs
        return SimpleNamespace(token_ids=[10, 11], log_probs=None, stop_reason='stop', extra_fields={})


def _build_client(token_budget_include: list[str] | None = None) -> HyTunerClient:
    client_args: dict[str, Any] = {'tokenizer': CharacterTokenizer()}
    if token_budget_include is not None:
        client_args['token_budget_include'] = token_budget_include
    return HyTunerClient(OpenAIConfig(client_args=client_args, request_args={}))


@pytest.mark.asyncio
async def test_selected_components_affect_budget_but_not_segment_counter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observation_client = _build_client(['agent_output'])
    observation_client.last_messages = [{'role': 'user', 'content': 'question'}]
    observation_client.token_consumed = 7
    observation_client.curr_segment_consumed_tokens = 7

    messages = [*observation_client.last_messages, {'role': 'tool', 'content': 'result'}]
    await observation_client.update_token_consumption(messages)

    # Tool observations still consume the current segment, but are excluded
    # from this trajectory's output-only budget.
    assert observation_client.curr_segment_consumed_tokens > 7
    assert observation_client.token_consumed == 7

    async def parse_tool_calls(token_ids: list[int], request: Any = None) -> tuple[str, str, list]:
        del token_ids, request
        return '', 'answer', []

    agent_client = _build_client(['agent_output'])
    agent_client.server_manager = ServerManager()
    monkeypatch.setattr(agent_client, '_parse_tool_calls', parse_tool_calls)
    await agent_client._create_completion([{'role': 'user', 'content': 'question'}])

    assert agent_client.token_consumed == 2
    assert agent_client.curr_segment_consumed_tokens == 2

    summary_client = _build_client(['agent_output'])
    summary_client.server_manager = ServerManager()
    monkeypatch.setattr(summary_client, '_parse_tool_calls', parse_tool_calls)
    await summary_client.create_completion(
        [{'role': 'user', 'content': 'summarize'}],
        request_meta={'invocation_kind': 'summary_inference'},
    )

    # A compaction output is excluded from the trajectory budget but remains
    # visible to the segment-level length contract.
    assert summary_client.token_consumed == 0
    assert summary_client.curr_segment_consumed_tokens == 2


@pytest.mark.asyncio
@pytest.mark.parametrize('invocation_kind', ['summary_inference', 'compaction_resume'])
async def test_summary_inference_retokenizes_when_tools_change(
    monkeypatch: pytest.MonkeyPatch,
    invocation_kind: str,
) -> None:
    client = HyTunerClient(OpenAIConfig(client_args={'tokenizer': ToolRenderingTokenizer()}, request_args={}))
    client.server_manager = ServerManager()

    async def parse_tool_calls(token_ids: list[int], request: Any = None) -> tuple[str, str, list]:
        del token_ids, request
        return '', 'answer', []

    monkeypatch.setattr(client, '_parse_tool_calls', parse_tool_calls)
    await client.create_completion(
        [{'role': 'user', 'content': 'question'}],
        tools=[{'name': 'exec_command'}],
    )
    summary_messages = [*client.last_messages, {'role': 'user', 'content': 'summarize'}]

    completion = await client.create_completion(
        summary_messages,
        tools=[],
        request_meta={'invocation_kind': invocation_kind},
    )

    assert completion.usage.prompt_tokens_details.cached_tokens == 0
