from __future__ import annotations

import asyncio

from types import SimpleNamespace
from typing import Any

import pytest

from nexus.core import OpenAIConfig


pytest.importorskip('numpy')

import nexus.extensions.agents.cc_search_agent_whitebox.compaction  # noqa: E402, F401

from nexus.extensions.compaction.compaction_manager import CompactionManager  # noqa: E402
from nexus.extensions.openai_clients.hytuner import HyTunerClient  # noqa: E402
from nexus.utils.timeout import InferenceTimeoutError, TimeBudget  # noqa: E402


def test_hytuner_effective_args_use_enforced_sampling_budget() -> None:
    client = HyTunerClient(
        OpenAIConfig(
            client_args={},
            request_args={'sampling_params': {'max_tokens': 1024, 'temperature': 1.0}},
            enforced_request_args={'sampling_params': {'max_tokens': 64, 'temperature': 0.2}},
        )
    )

    assert client.get_effective_request_args()['sampling_params'] == {
        'max_tokens': 64,
        'temperature': 0.2,
    }


@pytest.mark.asyncio
async def test_self_summarize_reads_effective_sampling_params(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class EffectiveClient:
        def get_effective_request_args(self) -> dict[str, Any]:
            return {'sampling_params': {'max_tokens': 64, 'temperature': 0.2}}

    class CapturingStrategy:
        async def compact(self, messages, tools, target_tokens, token_count_fn, **kwargs):
            del tools, target_tokens, token_count_fn
            captured.update(kwargs)
            return messages[1:]

    async def token_count(messages, tools):
        del messages, tools
        return 10

    monkeypatch.setattr(
        'nexus.extensions.compaction.compaction_manager.require_context',
        lambda: SimpleNamespace(openai_client=EffectiveClient()),
    )
    manager = CompactionManager(
        strategy_name='self_summarize',
        condition_name='always',
        token_count_fn=token_count,
        completion_fn=lambda: None,
        sampling_params={'max_tokens': 999},
    )
    manager._strategy = CapturingStrategy()

    await manager.maybe_compact(
        [{'role': 'system', 'content': 'system'}, {'role': 'user', 'content': 'hello'}],
        tools=[],
    )

    assert captured['sampling_params'] == {'max_tokens': 64, 'temperature': 0.2}


@pytest.mark.asyncio
@pytest.mark.parametrize('strategy_name', ['self_summarize', 'search/self_summarize'])
async def test_self_summarize_completion_uses_inference_budget(
    monkeypatch: pytest.MonkeyPatch,
    strategy_name: str,
) -> None:
    class EffectiveClient:
        def get_effective_request_args(self) -> dict[str, Any]:
            return {'sampling_params': {}}

    class CallingStrategy:
        async def compact(self, messages, tools, target_tokens, token_count_fn, **kwargs):
            del tools, target_tokens, token_count_fn
            await kwargs['completion_fn'](messages=messages)
            return messages

    async def token_count(messages, tools):
        del messages, tools
        return 10

    async def hanging_completion(**kwargs):
        del kwargs
        await asyncio.Event().wait()

    monkeypatch.setattr(
        'nexus.extensions.compaction.compaction_manager.require_context',
        lambda: SimpleNamespace(openai_client=EffectiveClient()),
    )
    manager = CompactionManager(
        strategy_name=strategy_name,
        condition_name='always',
        token_count_fn=token_count,
        completion_fn=hanging_completion,
    )
    manager._strategy = CallingStrategy()

    with pytest.raises(InferenceTimeoutError):
        await manager.maybe_compact(
            [{'role': 'user', 'content': 'hello'}],
            tools=[],
            inference_budget=TimeBudget(0.01),
        )
