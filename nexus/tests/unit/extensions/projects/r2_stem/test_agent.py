import asyncio

from types import SimpleNamespace

import pytest

from nexus.core import ExecutionContext, ToolExecutor
from nexus.core.openai_client import OpenAICompatibleClient, OpenAIConfig
from nexus.extensions.projects.r2_stem.agent import (
    EXIT_STATUS_TO_TERMINATION_REASON,
    MiniSWEAgentStemExitStatus,
    MiniSWEAgentStemInput,
    _parse_tool_calls,
    run_mini_swe_agent_stem,
)
from nexus.extensions.projects.r2_stem.tools import bash_tool


def test_every_exit_status_maps_to_a_termination_reason() -> None:
    unmapped = [
        status.value for status in MiniSWEAgentStemExitStatus if status.value not in EXIT_STATUS_TO_TERMINATION_REASON
    ]
    assert unmapped == []


def test_output_limit_and_context_limit_report_distinct_termination_reasons() -> None:
    output_limit = EXIT_STATUS_TO_TERMINATION_REASON[MiniSWEAgentStemExitStatus.OUTPUT_LENGTH_EXCEEDED.value]
    context_limit = EXIT_STATUS_TO_TERMINATION_REASON[MiniSWEAgentStemExitStatus.TRUNCATION.value]

    assert output_limit == 'OUTPUT_LENGTH_EXCEEDED'
    assert context_limit == 'TRUNCATION'


class _HangingClient(OpenAICompatibleClient):
    """Stands in for a sampling call that never comes back within the budget."""

    def __init__(self) -> None:
        super().__init__(OpenAIConfig(client_args={}, request_args={}))

    async def _create_completion(self, messages, model=None, **kwargs):
        raise NotImplementedError

    async def create_completion(self, messages, **kwargs):
        await asyncio.sleep(60)


@pytest.mark.asyncio
async def test_wall_time_limit_stops_a_sampling_call_that_outlives_the_budget() -> None:
    """One sampling call must not be able to consume the whole wall-clock budget.

    Checking the deadline between turns is not enough on its own: a single call has been
    seen to run for hours, and then it is the caller's outer ``asyncio.wait_for`` that ends
    the run — by cancelling this coroutine, which loses the trajectory and the termination
    reason with it. Stopping here keeps both, so the rollout stays explainable.
    """
    ctx = ExecutionContext(openai_clients={'main': _HangingClient()})

    async with ctx:
        output = await asyncio.wait_for(
            run_mini_swe_agent_stem(MiniSWEAgentStemInput(user_prompt='solve it', wall_time_limit_seconds=0.05)),
            timeout=10,
        )

    assert output.exit_status is MiniSWEAgentStemExitStatus.TIMEOUT
    assert EXIT_STATUS_TO_TERMINATION_REASON[output.exit_status.value] == 'TASK_TIMEOUT'
    assert output.error_type == 'WallTimeLimit'
    # The prompt the agent was working on survives, which is the point of stopping here.
    assert output.main_trajectory


@pytest.mark.asyncio
async def test_no_wall_time_limit_leaves_the_sampling_call_unbounded() -> None:
    ctx = ExecutionContext(openai_clients={'main': _HangingClient()})

    async with ctx:
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                run_mini_swe_agent_stem(MiniSWEAgentStemInput(user_prompt='solve it')),
                timeout=0.2,
            )


def _tool_call_message(name: str, arguments: str) -> SimpleNamespace:
    function = SimpleNamespace(name=name, arguments=arguments)
    tool_call = SimpleNamespace(id='call-1', function=function)
    return SimpleNamespace(tool_calls=[tool_call])


def test_parse_tool_calls_rejects_unknown_tool_as_format_error() -> None:
    executor = ToolExecutor([bash_tool])
    message = _tool_call_message('unknown', '{"command": "echo ok"}')

    parsed, failed = _parse_tool_calls(message, 300.0, executor)

    assert parsed == []
    assert len(failed) == 1
    assert "Tool 'unknown' not found" in failed[0][1]


def test_parse_tool_calls_rejects_invalid_tool_model_arguments_as_format_error() -> None:
    executor = ToolExecutor([bash_tool])
    message = _tool_call_message('bash', '{"command": "echo ok", "timeout": "not-a-number"}')

    parsed, failed = _parse_tool_calls(message, 300.0, executor)

    assert parsed == []
    assert len(failed) == 1
    assert "Invalid arguments for tool 'bash'" in failed[0][1]
    assert 'timeout' in failed[0][1]
