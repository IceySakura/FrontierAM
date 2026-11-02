"""A tool call the client flagged as unparsed scores 0 without calling the judge."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from nexus.core.context import ExecutionContext
from nexus.core.openai_client import UNPARSED_TOOL_CALL_FIELD, OpenAICompatibleClient, OpenAIConfig
from nexus.core.types import ExtendedChatCompletion, ExtendedChoice, ExtendedMessage
from nexus.extensions.agents.main.agent import MainAgentInput, MainExitStatus, run_main_agent
from nexus.extensions.tasks.mm_agent.task import SearchTaskInput, mm_agent_task


class _ScriptedClient(OpenAICompatibleClient):
    def __init__(self, completion: ExtendedChatCompletion) -> None:
        super().__init__(OpenAIConfig(client_args={}, request_args={}))
        self.completion = completion

    async def _create_completion(self, messages, model=None, **kwargs) -> ExtendedChatCompletion:
        raise NotImplementedError

    async def create_completion(self, messages, **kwargs) -> ExtendedChatCompletion:
        return self.completion


def _unparsed_tool_call_completion() -> ExtendedChatCompletion:
    return ExtendedChatCompletion(
        id='unparsed-tool-call',
        created=0,
        choices=[
            ExtendedChoice(
                finish_reason='stop',
                index=0,
                message=ExtendedMessage(
                    role='assistant',
                    content='The answer is 4.<tool_calls><tool_call>code<arg_key>code<arg_key></tool_call></tool_calls>',
                ),
                provider_specific_fields={UNPARSED_TOOL_CALL_FIELD: True},
            )
        ],
    )


async def _run_task(completion: ExtendedChatCompletion) -> tuple[Any, AsyncMock]:
    async def run_agent(**kwargs: Any):
        return await run_main_agent(MainAgentInput(tools=[], **kwargs))

    task_input = SearchTaskInput(
        question='How many bottles?',
        answer='4',
        instruction_template='{question}',
        reward_func='exact_match',
        run_agent=run_agent,
    )
    judge = AsyncMock(return_value=(1.0, {}))
    async with ExecutionContext(openai_clients={'main': _ScriptedClient(completion)}):
        with patch('nexus.extensions.tasks.mm_agent.task.vl_reward_system', judge):
            output = await mm_agent_task(task_input)
    return output, judge


@pytest.mark.asyncio
async def test_unparsed_tool_call_scores_zero_without_calling_the_judge() -> None:
    output, judge = await _run_task(_unparsed_tool_call_completion())

    assert output.exit_status == MainExitStatus.TOOL_ERROR.value
    assert MainExitStatus.termination_reason_for(output.exit_status) == 'FORMAT_ERROR'
    assert output.score == 0.0
    judge.assert_not_called()
    assert output.trajectory[-1]['role'] == 'assistant'
