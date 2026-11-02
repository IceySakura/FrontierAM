from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from harbor.models.agent.context import AgentContext
from harbor.trial.trial import Trial

from nexus.core import AgentRunOutput, ExecutionContext
from nexus.extensions.harbor_trial.agent import NexusAgent
from nexus.extensions.harbor_trial.context import HarborTrialBinding, harbor_trial_binding


class Output(AgentRunOutput):
    final_answer: str = 'done'


def _environment():
    return SimpleNamespace(task_env_config=SimpleNamespace(workdir='/workspace'), default_user=None)


async def test_generic_agent_defers_output_until_native_post_run(tmp_path) -> None:
    run_agent = AsyncMock(return_value=Output(), nexus_meta={})
    binding = HarborTrialBinding(execution_context=ExecutionContext(), run_agent=run_agent)
    context = AgentContext()
    async with harbor_trial_binding(binding):
        agent = NexusAgent(logs_dir=tmp_path / 'logs')
        await agent.run('test', _environment(), context)
    assert context.is_empty()
    run_agent.assert_awaited_once_with(user_prompt='test', working_dir='/workspace')
    trial = SimpleNamespace(agent=agent, user_agent=None)
    Trial._populate_agent_context(trial, context)
    assert context.metadata == {'nexus_agent_output': {'final_answer': 'done'}}
    Trial._populate_agent_context(trial, context)
    assert context.metadata == {'nexus_agent_output': {'final_answer': 'done'}}


async def test_post_run_merges_existing_metadata(tmp_path) -> None:
    run_agent = AsyncMock(return_value=Output(), nexus_meta={})
    binding = HarborTrialBinding(execution_context=ExecutionContext(), run_agent=run_agent)
    context = AgentContext()
    async with harbor_trial_binding(binding):
        agent = NexusAgent(logs_dir=tmp_path / 'logs')
        await agent.run('test', _environment(), context)
    context.metadata = {'existing': True}
    agent.populate_context_post_run(context)
    assert context.metadata == {'existing': True, 'nexus_agent_output': {'final_answer': 'done'}}


@pytest.mark.parametrize('fail_before_agent', [False, True])
async def test_new_run_resets_pending_output_before_workdir_resolution(tmp_path, fail_before_agent) -> None:
    run_agent = AsyncMock(side_effect=[Output(), RuntimeError('agent failed')], nexus_meta={})
    binding = HarborTrialBinding(execution_context=ExecutionContext(), run_agent=run_agent)
    first_context, second_context = AgentContext(), AgentContext()
    async with harbor_trial_binding(binding):
        agent = NexusAgent(logs_dir=tmp_path / 'logs')
        await agent.run('first', _environment(), first_context)
        environment = _environment()
        if fail_before_agent:
            environment.task_env_config.workdir = None
            environment.exec = AsyncMock(side_effect=RuntimeError('pwd failed'))
        with pytest.raises(RuntimeError, match='pwd failed' if fail_before_agent else 'agent failed'):
            await agent.run('second', environment, second_context)
        agent.populate_context_post_run(second_context)
        agent.populate_context_post_run(first_context)
    assert first_context.is_empty()
    assert second_context.is_empty()
    assert run_agent.await_count == (1 if fail_before_agent else 2)
