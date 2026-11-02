from __future__ import annotations

import asyncio
import time

from typing import Any

import pytest

from nexus.extensions.tasks.aidata.engine.models import AidataTaskInput
from nexus.extensions.tasks.aidata.engine.task import _close_mcp_tools_with_agent_timeout, build_agent_kwargs


async def _never_called_agent(**_: Any) -> Any:  # pragma: no cover
    raise AssertionError('run_agent must not be invoked')


class _SlowCloseContext:
    def __init__(self) -> None:
        self.close_calls = 0

    async def close_mcp_tools(self) -> None:
        self.close_calls += 1
        await asyncio.sleep(1)


def test_agent_timeout_does_not_implicitly_set_observation_timeout() -> None:
    task_input = AidataTaskInput(
        run_agent=_never_called_agent,
        agent_timeout=123,
        max_iterations=7,
    )

    kwargs = build_agent_kwargs(
        task_input,
        user_message='do work',
        system_prompt='system',
        tools=['base/finish'],
    )

    assert kwargs['timeout'] == 123
    assert 'observation_time_timeout' not in kwargs
    assert kwargs['max_iterations'] == 7


def test_explicit_observation_timeout_is_forwarded_for_main_style_agents() -> None:
    task_input = AidataTaskInput(
        run_agent=_never_called_agent,
        agent_timeout=1800,
        observation_time_timeout=45.5,
    )

    kwargs = build_agent_kwargs(
        task_input,
        user_message='do work',
        system_prompt='system',
        tools=['base/finish'],
    )

    assert kwargs['timeout'] == 1800
    assert kwargs['observation_time_timeout'] == 45.5


@pytest.mark.asyncio
async def test_mcp_close_uses_remaining_agent_timeout_budget() -> None:
    task_input = AidataTaskInput(
        run_agent=_never_called_agent,
        agent_timeout=1,
    )
    ctx = _SlowCloseContext()

    with pytest.raises(TimeoutError):
        await _close_mcp_tools_with_agent_timeout(
            ctx,
            task_input=task_input,
            task_id='timeout-close',
            agent_timeout_started_at=time.monotonic() - 2,
            propagate_timeout=True,
        )

    assert ctx.close_calls == 1


@pytest.mark.asyncio
async def test_mcp_close_timeout_is_suppressed_for_final_cleanup() -> None:
    task_input = AidataTaskInput(
        run_agent=_never_called_agent,
        agent_timeout=1,
    )
    ctx = _SlowCloseContext()

    await _close_mcp_tools_with_agent_timeout(
        ctx,
        task_input=task_input,
        task_id='final-cleanup',
        agent_timeout_started_at=time.monotonic() - 2,
        propagate_timeout=False,
    )

    assert ctx.close_calls == 1
