from __future__ import annotations

from typing import Any

import pytest

from pydantic import ValidationError

from nexus.extensions.projects.hy_live import agent as agent_module
from nexus.extensions.projects.hy_live import health as health_module
from nexus.extensions.projects.hy_live import task as task_module
from nexus.extensions.projects.hy_live.coordination.models import AgentTask, TaskError, TaskResult
from nexus.extensions.projects.hy_live.user_simulator import (
    CerebellumResponse,
    SimulatedConversation,
    SimulatedTurn,
)
from nexus.runner.single import _build_input


async def _unused_run_agent(**kwargs):
    raise AssertionError(f'unexpected run_agent call: {kwargs}')


def test_task_input_accepts_user_prompt_and_rejects_conflicting_prompt() -> None:
    task_input = task_module.LiveAgentTaskInput.model_validate({
        'user_prompt': 'inspect',
        'run_agent': _unused_run_agent,
    })
    assert task_input.prompt == 'inspect'

    runner_input = _build_input(
        task_module.LiveAgentTaskInput,
        {'user_prompt': 'from runner', 'run_agent': _unused_run_agent},
    )
    assert runner_input.prompt == 'from runner'

    with pytest.raises(ValueError, match='must match'):
        task_module.LiveAgentTaskInput.model_validate({
            'prompt': 'one',
            'user_prompt': 'two',
            'run_agent': _unused_run_agent,
        })


def test_task_input_accepts_scripted_user_config_without_prompt() -> None:
    task_input = task_module.LiveAgentTaskInput.model_validate({
        'user_simulator': {'type': 'scripted', 'turns': ['先做 A', '再做 B']},
        'run_agent': _unused_run_agent,
    })

    assert task_input.prompt is None
    assert task_input.user_simulator is not None
    assert task_input.user_simulator.type == 'scripted'


def test_task_input_ignores_retired_legacy_keys_in_old_rows() -> None:
    old_row = {
        'prompt': 'inspect',
        'summary_instruction': 'old instruction',
        'enabled_work_agents': ['codex'],
        'max_policy_steps_per_turn': 1,
        'delivery_batch_window_s': 1.0,
        'channel': 'full_duplex_voice',
        'client_working_directory': '/srv',
        'default_budget': {'timeout_s': 900, 'max_steps': 20, 'max_observation_tokens': 4096},
        'run_agent': _unused_run_agent,
    }

    for task_input in (
        task_module.LiveAgentTaskInput.model_validate(old_row),
        _build_input(task_module.LiveAgentTaskInput, old_row),
    ):
        dumped = task_input.model_dump()
        assert task_input.prompt == 'inspect'
        assert task_input.backend_adapter == 'workbuddy'
        assert task_input.default_budget.model_dump() == {'timeout_s': 900}
        assert not {'summary_instruction', 'enabled_work_agents', 'max_policy_steps_per_turn', 'channel'} & set(dumped)


def _agent_task(status: str, result: TaskResult) -> AgentTask:
    return AgentTask(
        task_id='task-1',
        owner_id='owner-1',
        objective='work',
        created_sequence=1,
        coordinator_session_id='coordinator-1',
        status=status,
        result=result,
    )


def _agent_output(**fields: Any) -> agent_module.LiveAgentAgentOutput:
    return agent_module.LiveAgentAgentOutput(**{
        'main_trajectory': [],
        'owner_id': 'owner-1',
        'summary': 'done',
        'conversation': SimulatedConversation(turns=[], ended_by_user=True),
        'tasks': [],
        **fields,
    })


async def test_live_agent_task_maps_interactive_output_and_leaves_the_score_to_the_reward_loop() -> None:
    task = _agent_task('done', TaskResult(final_output='A done', metadata={'score': 0.5}))
    conversation = SimulatedConversation(
        turns=[SimulatedTurn(index=1, user_message='do A', cerebellum_response=CerebellumResponse(content='accepted'))],
        ended_by_user=True,
    )
    calls: list[dict] = []

    async def run_agent(**kwargs):
        calls.append(kwargs)
        return _agent_output(summary='A done', conversation=conversation, tasks=[task])

    result = await task_module.live_agent_task(task_module.LiveAgentTaskInput(prompt='do A', run_agent=run_agent))

    assert calls[0]['prompt'] == 'do A'
    assert result.owner_id == 'owner-1'
    assert result.summary == 'A done'
    assert result.tasks == [task]
    assert result.conversation == conversation
    assert result.score is None, 'A None score lets the HyTuner reward loop run'


@pytest.mark.parametrize('adapter', ['codebuddy', 'workbuddy'])
async def test_backend_adapter_defaults_to_workbuddy_and_each_supported_one_reaches_the_agent(adapter: str) -> None:
    calls = []

    async def run_agent(**kwargs):
        calls.append(kwargs)
        return _agent_output()

    default = task_module.LiveAgentTaskInput.model_validate({'prompt': 'task', 'run_agent': _unused_run_agent})
    assert default.backend_adapter == 'workbuddy'
    assert agent_module.LiveAgentAgentInput(prompt='task').backend_adapter == 'workbuddy'
    with pytest.raises(ValidationError):
        task_module.LiveAgentTaskInput.model_validate({
            'prompt': 'task',
            'backend_adapter': 'codex',
            'run_agent': _unused_run_agent,
        })
    task_input = task_module.LiveAgentTaskInput.model_validate({
        'prompt': 'task',
        'backend_adapter': adapter,
        'run_agent': run_agent,
    })

    await task_module.live_agent_task(task_input)

    assert calls[0]['backend_adapter'] == adapter
    assert calls[0]['summarize_outcomes'] is False, 'finished work is told without a midbrain summary by default'
    assert set(calls[0]) <= set(agent_module.LiveAgentAgentInput.model_fields), 'Every forwarded key is an agent field'
    assert agent_module.LiveAgentAgentInput.model_validate(calls[0]).backend_adapter == adapter


async def test_summarize_outcomes_reaches_the_agent() -> None:
    calls = []

    async def run_agent(**kwargs):
        calls.append(kwargs)
        return _agent_output()

    task_input = task_module.LiveAgentTaskInput.model_validate({
        'prompt': 'task',
        'summarize_outcomes': True,
        'run_agent': run_agent,
    })

    await task_module.live_agent_task(task_input)

    assert calls[0]['summarize_outcomes'] is True


async def test_live_agent_task_forwards_layer_inputs_and_exposes_both_trajectories() -> None:
    calls = []
    cerebellum_record = {
        'tools': [],
        'messages': [{'role': 'system', 'content': '小脑', 'at': '2026-10-09T11:31:06.995019+08:00'}],
    }
    midbrain_record = {
        'tools': [],
        'messages': [{'role': 'system', 'content': '中脑', 'at': '2026-10-09T11:31:07.000001+08:00'}],
    }

    async def run_agent(**kwargs):
        calls.append(kwargs)
        return _agent_output(
            main_trajectory=[{'role': 'system', 'content': '小脑'}],
            cerebellum_trajectory=cerebellum_record,
            midbrain_trajectory=midbrain_record,
        )

    result = await task_module.live_agent_task(
        task_module.LiveAgentTaskInput.model_validate({
            'prompt': 'task',
            'midbrain_client_name': 'midbrain',
            'user_place': {'country': '中国', 'city': '北京'},
            'user_now': '2026-10-09T11:31:06+08:00',
            'run_agent': run_agent,
        })
    )

    assert calls[0]['midbrain_client_name'] == 'midbrain'
    assert calls[0]['user_place'] == {'country': '中国', 'city': '北京'}
    assert calls[0]['user_now'].isoformat() == '2026-10-09T11:31:06+08:00'
    assert result.cerebellum_trajectory == cerebellum_record
    assert result.midbrain_trajectory == midbrain_record
    assert result.policy_messages == [{'role': 'system', 'content': '小脑'}], 'policy_messages stays main_trajectory'


def test_example_run_input_gives_cerebellum_and_midbrain_their_own_sampling() -> None:
    from nexus.extensions.projects.hy_live.example.cases import get_case
    from nexus.extensions.projects.hy_live.example.harness import LayerSampling, build_run_input

    def request_args(run_input, name: str) -> dict:
        return run_input.openai_clients[name]['config']['request_args']

    run_input = build_run_input(
        get_case('single_snake'),
        run_id='test-sampling',
        api_base_url='http://example.invalid/v1',
        model='dummy',
        reasoning_effort='high',
        temperature=0.0,
        sandbox_api_url='http://example.invalid',
        runtime_image='current',
        workspace_image='img',
        timeout=60.0,
    )

    assert run_input.task_config['midbrain_client_name'] == 'midbrain'
    assert request_args(run_input, 'main') == {
        'model': 'dummy',
        'reasoning_effort': 'none',
        'temperature': 1.0,
        'top_p': 1.0,
        'max_completion_tokens': 4096,
    }, 'The cerebellum runs with thinking off'
    assert request_args(run_input, 'midbrain')['reasoning_effort'] == 'low', 'The midbrain keeps light thinking'
    assert (request_args(run_input, 'midbrain')['temperature'], request_args(run_input, 'midbrain')['top_p']) == (
        1.0,
        1.0,
    )
    assert request_args(run_input, 'external')['reasoning_effort'] == 'high', 'Backend flags are unchanged'
    assert request_args(run_input, 'external')['temperature'] == 0.0

    overridden = build_run_input(
        get_case('single_snake'),
        run_id='test-sampling-override',
        api_base_url='http://example.invalid/v1',
        model='dummy',
        reasoning_effort='high',
        temperature=0.0,
        sandbox_api_url='http://example.invalid',
        runtime_image='current',
        workspace_image='img',
        timeout=60.0,
        cerebellum_sampling=LayerSampling(reasoning_effort='', temperature=0.7, max_tokens=1024),
    )
    assert request_args(overridden, 'main') == {
        'model': 'dummy',
        'temperature': 0.7,
        'top_p': 1.0,
        'max_completion_tokens': 1024,
    }, 'An empty reasoning effort is left out of the request'


def test_task_input_layer_inputs_default_to_none() -> None:
    task_input = task_module.LiveAgentTaskInput.model_validate({'prompt': 'task', 'run_agent': _unused_run_agent})

    assert (task_input.midbrain_client_name, task_input.user_place, task_input.user_now) == (None, None, None)


_INFRASTRUCTURE_FAILURE = TaskResult(
    execution_status='failed',
    failure_kind='infrastructure',
    error=TaskError(code='BACKEND_UNAVAILABLE', message='backend down'),
)
_BACKGROUND_TIMEOUT = TaskResult(
    execution_status='timeout',
    failure_kind='timeout',
    error=TaskError(code='TASK_TIMEOUT', message='project timed out'),
)


@pytest.mark.parametrize(
    ('tasks', 'episode_termination', 'expected'),
    [
        ([], health_module.FORMAT_ERROR_TERMINATION, ('format_error', [], health_module.FORMAT_ERROR_TERMINATION)),
        ([], 'ENV_ERROR', ('error', [], 'ENV_ERROR')),
        ([_agent_task('done', TaskResult(final_output='done'))], 'ENV_DONE', ('healthy', [], 'ENV_DONE')),
        ([_agent_task('failed', _INFRASTRUCTURE_FAILURE)], None, ('error', ['infrastructure'], 'ENV_ERROR')),
        # The policy was told the background task failed, so reporting it is scored rather than masked.
        ([_agent_task('failed', _BACKGROUND_TIMEOUT)], None, ('healthy', ['timeout'], 'ENV_DONE')),
        ([], 'TASK_TIMEOUT', ('timeout', [], 'TASK_TIMEOUT')),
        # The trainer drops it by this name: the harness answered for a user who had left.
        (
            [],
            health_module.USER_LEFT_EARLY_TERMINATION,
            ('user_left', [], health_module.USER_LEFT_EARLY_TERMINATION),
        ),
    ],
)
def test_aggregate_episode_health(
    tasks: list[AgentTask], episode_termination: str | None, expected: tuple[str, list[str], str]
) -> None:
    health = health_module.aggregate_episode_health(tasks=tasks, episode_termination=episode_termination)

    assert (health['health_status'], health['failure_kinds'], health['termination_reason']) == expected


async def test_live_agent_task_metadata_reports_env_error_without_failing() -> None:
    async def run_agent(**kwargs):
        return _agent_output(summary='partial summary', tasks=[_agent_task('failed', _INFRASTRUCTURE_FAILURE)])

    result = await task_module.live_agent_task(task_module.LiveAgentTaskInput(prompt='task', run_agent=run_agent))

    assert result.summary == 'partial summary'
    assert result.metadata['termination_reason'] == 'ENV_ERROR'
    assert result.metadata['failure_kinds'] == ['infrastructure']
    assert result.metadata['health_status'] == 'error'
