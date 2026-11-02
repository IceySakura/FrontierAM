from __future__ import annotations

from types import SimpleNamespace

import pytest

from nexus.extensions.compaction.compaction_strategy import (
    SelfSummarizeCompactionStrategy,
    SelfSummarizeV2CompactionStrategy,
)


async def _token_count(*_args) -> int:
    return 10


@pytest.mark.asyncio
async def test_self_summary_v2_append_control_preserves_history_prefix_and_tools() -> None:
    request: dict = {}

    async def completion_fn(**kwargs):
        request.update(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content='short summary'),
                    finish_reason='stop',
                )
            ],
            usage=SimpleNamespace(prompt_tokens=30, completion_tokens=3, completion_tokens_details=None),
        )

    messages = [
        {'role': 'system', 'content': 'system'},
        {'role': 'user', 'content': 'original task'},
        {'role': 'assistant', 'tool_calls': [{'id': 'call-1', 'function': {'name': 'search'}}]},
        {'role': 'tool', 'tool_call_id': 'call-1', 'content': 'tool result'},
    ]
    tools = [{'type': 'function', 'function': {'name': 'search'}}]
    strategy = SelfSummarizeV2CompactionStrategy()

    result = await strategy.compact(
        messages,
        tools=tools,
        target_tokens=400,
        token_count_fn=_token_count,
        completion_fn=completion_fn,
        self_summary_append_control=True,
        self_summary_system_prompt='custom system',
        self_summary_prompt='summarize within {target_tokens} tokens: {text}',
        self_summary_append_control_prompt='{system_prompt}\n{summary_prompt}',
        self_summary_handoff_prefix='custom handoff: ',
    )

    assert request['messages'][:-1] == messages
    assert request['messages'][-1]['content'].startswith('custom system\nsummarize within 390 tokens:')
    assert request['tools'] == tools
    assert 'count_toward_token_budget' not in request
    assert result[-1]['content'] == 'custom handoff: short summary'


@pytest.mark.asyncio
async def test_legacy_self_summary_preserves_request_and_failure_behavior() -> None:
    requests: list[dict] = []

    async def completion_fn(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='legacy summary'))],
        )

    messages = [
        {'role': 'system', 'content': 'system'},
        {'role': 'user', 'content': 'task'},
        {'role': 'assistant', 'content': 'work'},
    ]
    strategy = SelfSummarizeCompactionStrategy()
    result = await strategy.compact(messages, None, 400, _token_count, completion_fn=completion_fn)

    assert requests[0]['messages'] == [
        {
            'role': 'user',
            'content': '[assistant]: work\n\nPlease summarize the above conversation history concisely. '
            'Preserve key facts, decisions, tool call results, and context needed for continuing the conversation. '
            'Only output the compacted context.',
        }
    ]
    assert result[-1]['content'] == 'legacy summary'

    async def failing_completion_fn(**_kwargs):
        raise RuntimeError('expected failure')

    assert await strategy.compact(messages, None, 400, _token_count, completion_fn=failing_completion_fn) == messages
