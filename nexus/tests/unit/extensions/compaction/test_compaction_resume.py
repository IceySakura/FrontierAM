from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from nexus.extensions.compaction.compaction_manager import CompactionManager


@pytest.mark.asyncio
@pytest.mark.parametrize('inject_budget_state', [False, True])
@pytest.mark.parametrize(
    ('strategy_name', 'expects_resume'),
    [('self_summarize', False), ('self_summarize_v2', True)],
)
async def test_compacted_assistant_summary_adds_resume_only_for_self_summarize_v2(
    monkeypatch: pytest.MonkeyPatch,
    inject_budget_state: bool,
    strategy_name: str,
    expects_resume: bool,
) -> None:
    async def compact(messages: list[dict[str, Any]], *_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        return [
            messages[0],
            {'role': 'assistant', 'content': 'handoff', '_is_compacted_summary': True},
        ]

    async def token_count(messages: list[dict[str, Any]], _tools: Any) -> int:
        return len(messages)

    client = SimpleNamespace(
        token_budget=1000,
        token_consumed=400,
        get_effective_request_args=lambda: {'sampling_params': {}},
    )
    monkeypatch.setattr(
        'nexus.extensions.compaction.compaction_manager.require_context',
        lambda: SimpleNamespace(openai_client=client),
    )
    manager = CompactionManager(
        strategy_name=strategy_name,
        condition_name='always',
        token_count_fn=token_count,
        completion_fn=lambda: None,
        inject_trajectory_budget_state=inject_budget_state,
    )
    manager._strategy = SimpleNamespace(compact=compact)

    result = await manager.maybe_compact(
        [
            {'role': 'system', 'content': 'system'},
            {'role': 'assistant', 'content': 'continued work'},
        ],
        tools=[],
    )

    has_resume = any('<compaction_resume>' in message.get('content', '') for message in result.messages)
    assert has_resume is expects_resume
    assert result.prompt_token_count == len(result.messages)
    has_budget_state = any('<trajectory_budget_state>' in message.get('content', '') for message in result.messages)
    assert has_budget_state is inject_budget_state
