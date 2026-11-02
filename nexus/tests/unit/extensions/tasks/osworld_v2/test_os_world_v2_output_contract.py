"""V2 adopts V1's stages without losing scored metadata or marker policy."""

import json

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nexus.core import ExecutionContext
from nexus.extensions.tasks.os_world_v2.adapter.claude_multiturn import OSWorldV2MultiturnProfile
from nexus.extensions.tasks.os_world_v2.adapter.outputs import V2OutputContractHook, normalize_agent, summarize
from nexus.extensions.tasks.os_world_v2.outputs import OSWorldOutputContractHook, record_agent_output
from nexus.extensions.tasks.os_world_v2.task import OSWorldV2TaskOutput


@pytest.mark.asyncio
@pytest.mark.parametrize('hook_type', [OSWorldOutputContractHook, V2OutputContractHook])
@pytest.mark.parametrize('success', [False, True])
@pytest.mark.parametrize('atif_enabled', [False, True])
async def test_hook_retains_bridge_trajectory_until_postprocessing(hook_type, success, atif_enabled):
    from nexus.runner.single import ErrorInfo, RunOutput

    chat = {'trajectories': [[{'role': 'assistant', 'content': '完整对话' * 4000}]], 'tools': []}
    atif = {'atif': {'steps': [{'role': 'agent', 'message': '完整轨迹'}], 'extra': {}}}
    counters = {'main': {'completion_tokens': 591, 'prompt_tokens': 60448}}
    processors = {'_default_chat_trace': chat, 'token_counter': counters, 'chat': {'large': 'drop'}}
    if atif_enabled:
        processors['_default_atif_trace'] = atif
    output = RunOutput(
        success=success,
        task_output={
            'status': 'success' if success else 'agent_error',
            'task_id': '070',
            'score': 0 if success else None,
        },
        processor_results=processors,
        error=None if success else ErrorInfo(type='RuntimeError', message='original failure'),
    )
    archived = []

    async def publish(value):
        archived.append(value.model_dump(mode='json'))
        return {'status': 'published', 'uri': 'artifact:///fixture/details/'}

    ctx = SimpleNamespace(
        task_name=hook_type.task_name,
        tool_state={'osworld_output_publisher': SimpleNamespace(publish_run_output_details=publish)},
    )
    await hook_type().on_run_output(ctx, output)
    assert output.success is success
    assert output.processor_results['_default_chat_trace'] is chat
    assert output.processor_results['token_counter'] == counters
    assert archived[0]['processor_results']['_default_chat_trace'] == chat
    assert 'chat' not in output.processor_results
    if atif_enabled:
        assert output.processor_results['_default_atif_trace'] is atif
    else:
        assert '_default_atif_trace' not in output.processor_results


def cc_result(subtype='success', *, error=False, text='TASK_DONE'):
    return '\n'.join(
        json.dumps(event)
        for event in [
            {'type': 'system', 'subtype': 'init', 'session_id': 'parent'},
            {'type': 'result', 'session_id': 'parent', 'subtype': subtype, 'is_error': error, 'result': text},
        ]
    )


@pytest.mark.parametrize(
    ('raw', 'status', 'code'),
    [
        ({'success': True, 'exit_code': 0, 'output': cc_result()}, 'finished', 0),
        ({'success': True, 'exit_code': 1, 'output': cc_result('error_max_turns')}, 'max_iterations', 0),
        (
            {
                'success': False,
                'exit_code': 1,
                'output': cc_result(
                    'error_during_execution',
                    error=True,
                    text="API Error: Claude's response exceeded the 64000 output token maximum.",
                ),
            },
            'output_length_exceeded',
            0,
        ),
        (
            {
                'success': False,
                'exit_code': 1,
                'output': cc_result(
                    'error_during_execution',
                    error=True,
                    text='maximum context length exceeded',
                ),
            },
            'context_length_exceeded',
            0,
        ),
        (
            {'success': False, 'exit_code': -1, 'stderr': 'Claude Code exited without writing an exit code'},
            'runtime_error_by_env',
            -1,
        ),
        (
            {'success': False, 'exit_code': -1, 'failure_stage': 'install', 'error': 'npm unavailable'},
            'environment_error',
            -1,
        ),
    ],
)
def test_cc_semantics(raw, status, code):
    result = normalize_agent(raw, harness='cc')
    assert result.exit_status == status
    assert result.exit_code == code
    assert result.process_exit_code == raw['exit_code']


@pytest.mark.parametrize('typed', [False, True])
@pytest.mark.parametrize('cli_signal', [False, True])
def test_limit_evidence_overrides_finished_without_mutating_input(typed, cli_signal):
    from nexus.extensions.tasks.os_world_v2.outputs import OSWorldAgentOutput, normalize_agent_output

    data = {
        'success': True,
        'exit_status': 'finished',
        'exit_code': 0,
        'process_exit_code': 1 if cli_signal else 0,
        'conclusion': 'done',
        'output': cc_result('error_max_turns', error=True) if cli_signal else cc_result(text='Continue work'),
        'reached_max_turns': not cli_signal,
    }
    raw = OSWorldAgentOutput.model_validate(data) if typed else data
    before = raw.model_dump() if typed else dict(raw)
    result = normalize_agent(raw, harness='cc')
    assert result.exit_status == 'max_iterations'
    assert result.exit_code == 0
    assert result.conclusion is None
    assert result.process_exit_code == (data['process_exit_code'] if typed else 0)
    assert (raw.model_dump() if typed else raw) == before
    assert normalize_agent_output(result, harness='cc').model_dump() == result.model_dump()


def test_cli_limit_is_evaluable_even_when_cli_reports_error():
    result = normalize_agent(
        {'success': False, 'exit_code': 1, 'output': cc_result('error_max_turns', error=True)}, harness='cc'
    )
    assert result.exit_status == 'max_iterations'
    assert result.exit_code == 0
    assert result.process_exit_code == 1


@pytest.mark.parametrize(
    ('extra', 'expected'),
    [
        ({'max_iterations': 1, 'max_turns': 1, 'iterations': 1}, 'finished'),
        ({'reached_max_turns': True, 'ended_by_user': True}, 'finished'),
        ({'reached_max_turns': True, 'output': cc_result(text='TASK_DONE')}, 'finished'),
        ({'reached_max_turns': True, 'terminal_api_error': 'HTTP 503'}, 'inference_error'),
        ({'reached_max_turns': True, 'failure_kind': 'infrastructure'}, 'runtime_error_by_env'),
        ({'reached_max_turns': True, 'termination_reason': 'cc_timeout'}, 'agent_timeout'),
        ({'reached_max_turns': True, 'failure_stage': 'install'}, 'environment_error'),
        ({'reached_max_turns': True, 'terminal_finish_reason': 'length'}, 'output_length_exceeded'),
        (
            {
                'reached_max_turns': True,
                'output': cc_result('error_during_execution', error=True, text='API Error: 503'),
            },
            'inference_error',
        ),
    ],
)
def test_limits_do_not_override_actual_completion_or_failure(extra, expected):
    result = normalize_agent(
        {
            'success': True,
            'exit_code': 0,
            'exit_status': 'finished',
            'output': cc_result(text='Continue work'),
            **extra,
        },
        harness='cc',
    )
    assert result.exit_status == expected


@pytest.mark.parametrize('reason', ['max_turns', 'max_iterations'])
def test_structured_harness_limit_reason_overrides_generic_finished(reason):
    result = normalize_agent(
        {
            'success': True,
            'exit_status': 'finished',
            'exit_code': 0,
            'termination_reason': reason,
        },
        harness='codex',
    )
    assert result.exit_status == 'max_iterations'
    assert result.exit_code == 0
    assert result.termination_reason == reason


def test_child_limit_does_not_stop_successful_parent():
    raw = (
        cc_result(text='TASK_DONE')
        + '\n'
        + json.dumps({
            'type': 'result',
            'session_id': 'child',
            'parent_tool_use_id': 'tool-child',
            'subtype': 'error_max_turns',
            'is_error': True,
        })
    )
    assert (
        normalize_agent(
            {'success': True, 'exit_code': 0, 'exit_status': 'finished', 'output': raw}, harness='cc'
        ).exit_status
        == 'finished'
    )


@pytest.mark.asyncio
async def test_first_normalization_preserves_limit_through_task_and_output_hook(monkeypatch):
    from nexus.extensions.tasks.os_world_v2.adapter.artifacts import RunArtifacts
    from nexus.extensions.tasks.os_world_v2.outputs import OSWorldJudgeOutput, run_task_agent, task_status
    from nexus.runner.single import RunOutput

    monkeypatch.setattr(
        RunArtifacts,
        'publish_run_output_details',
        AsyncMock(return_value={'status': 'published', 'uri': 'artifact:///fixture/details/', 'upload_failed': False}),
    )
    async with ExecutionContext(task_name='os_world_v2', agent_name='cua_gpt_agent') as ctx:
        agent = await run_task_agent(
            AsyncMock(
                return_value={
                    'success': True,
                    'exit_code': 0,
                    'exit_status': 'finished',
                    'reached_max_turns': True,
                    'turns': 1,
                }
            )
        )
        assert agent.exit_status == 'max_iterations'
        judge = OSWorldJudgeOutput(exit_status='finished', exit_code=0)
        task = OSWorldV2TaskOutput(
            completed=True,
            status=task_status(agent, judge),
            task_id='020',
            instruction='fixture',
            score=0.25,
            evaluator_result={'scored': True},
            agent_output=agent,
            judge_output=judge,
        )
        output = RunOutput(success=True, task_output=task, agent_output=agent)
        await V2OutputContractHook().on_run_output(ctx, output)
    assert output.success
    assert output.task_output['score'] == 0.25
    assert output.task_output['evaluator_result']['scored']
    assert output.agent_output['exit_status'] == 'max_iterations'
    assert output.agent_output['exit_code'] == 0


def test_v2_does_not_add_infeasible_marker():
    result = normalize_agent(
        {'success': True, 'exit_code': 0, 'output': cc_result(text='TASK_INFEASIBLE')}, harness='cc'
    )
    assert result.exit_status == 'finished'
    assert result.conclusion != 'infeasible'


@pytest.mark.asyncio
async def test_v2_profile_evidence_is_recorded_but_ordinary_tasks_are_not():
    async with ExecutionContext(task_name='os_world_v2') as ctx:
        result = record_agent_output(
            {'success': True, 'exit_code': 1, 'exit_status': 'max_iterations'}, harness='codex'
        )
        assert result is not None
        assert result.exit_code == 0
        assert ctx.tool_state['osworld_agent_output'] is result
    async with ExecutionContext(task_name='unrelated'):
        assert record_agent_output({'exit_status': 'finished'}, harness='codex') is None


def test_projection_keeps_v2_scored_flag_without_mutating_full_output():
    task = OSWorldV2TaskOutput(
        completed=True,
        task_id='104',
        instruction='private full task',
        score=0.2,
        agent_output=normalize_agent({'exit_status': 'finished', 'exit_code': 0}),
        evaluator_result={'score': 0.2, 'scored': True, 'phases': [{'large': 'evidence'}]},
    )
    small = summarize(task)
    assert small['evaluator_result'] == {'score': 0.2, 'scored': True}
    assert 'setup_results' not in small
    assert task.instruction == 'private full task'
    assert task.evaluator_result is not None
    assert task.evaluator_result['phases']


@pytest.mark.asyncio
async def test_failure_hook_attaches_summary_before_archival_and_compacts():
    partial = {
        'status': 'judge_error',
        'task_id': '104',
        'completed': False,
        'score': None,
        'agent_output': {'exit_status': 'finished', 'exit_code': 0},
        'judge_output': {'exit_status': 'judge_error', 'exit_code': -1, 'error': 'CAD unavailable'},
        'artifacts': {'status': 'published', 'uri': 'artifact:///owner/run/', 'uris': []},
    }
    archived = []

    async def publish(output):
        archived.append(output.task_output)
        return {'status': 'published', 'uri': 'artifact:///owner/details/'}

    ctx = SimpleNamespace(
        task_name='os_world_v2',
        tool_state={
            'osworld_partial_task_output': partial,
            'osworld_output_publisher': SimpleNamespace(publish_run_output_details=publish),
        },
    )
    output = SimpleNamespace(
        success=False,
        task_output=None,
        agent_output=None,
        model_dump=lambda: {},
        processor_results={'chat': 'large'},
        error=None,
    )
    hook = V2OutputContractHook()
    await hook.on_run_output(ctx, output)
    assert archived[0] is partial
    assert output.success is False
    assert output.agent_output is not None
    assert output.task_output['status'] == 'judge_error'
    assert output.agent_output['exit_status'] == 'finished'
    assert output.task_output['artifacts']['runner_details']['status'] == 'published'
    assert output.processor_results == {}
    await hook.on_context_closing(ctx)
    assert ctx.tool_state == {}


@pytest.mark.parametrize('override', [None, 7200])
def test_multiturn_timeout_override_retains_distinct_cli_and_user_turn_limits(override):
    from nexus.extensions.agents.claude_code.agent import ClaudeCodeMultiturnInput

    data = {
        'execution_profile': 'osworld_v2',
        'execution_options': {'cc_timeout': 1800, 'max_turns': 500},
        'max_turns': 20,
        'working_dir': '/home/user',
        'user_prompt': 'fixture',
        'get_user_input': AsyncMock(return_value=None),
    }
    if override is not None:
        data['timeout'] = override
    inp = ClaudeCodeMultiturnInput.model_validate(data)
    profile = OSWorldV2MultiturnProfile(inp)
    assert profile.options.timeout == (override or 1800)
    assert profile.options.max_turns == 500
    assert profile.input.max_turns == 20


@pytest.mark.asyncio
async def test_real_runner_failure_hook_retains_v2_summary(tmp_path, monkeypatch):
    from nexus import RunInput
    from nexus.extensions.tasks.os_world_v2 import task
    from nexus.extensions.tasks.os_world_v2.data.artifacts import RunArtifacts
    from nexus.runner.single import run

    async def failed(_):
        raise RuntimeError('fixture setup failure')

    monkeypatch.setattr(task, 'run', failed)
    monkeypatch.setattr(
        RunArtifacts,
        'publish',
        AsyncMock(
            return_value={
                'status': 'published',
                'uri': 'artifact:///fixture/task/',
                'uris': [],
            }
        ),
    )
    archive = AsyncMock(return_value={'status': 'published', 'uri': 'artifact:///fixture/details/'})
    monkeypatch.setattr(RunArtifacts, 'publish_run_output_details', archive)
    output = await run(
        RunInput.model_validate({
            'task_name': 'os_world_v2',
            'task_config': {
                'task_id': '041',
                'v21': {},
            },
            'runner': {'hooks': {'output_contract': {'name': 'osworld_v2_output_contract', 'args': {}}}},
        })
    )
    assert output.success is False
    assert output.agent_output is not None
    assert output.error is not None
    assert output.task_output['status'] == 'agent_error'
    assert output.agent_output['exit_status'] == 'environment_error'
    assert output.task_output['artifacts']['runner_details']['status'] == 'published'
    assert output.error.message == 'fixture setup failure'
    archive.assert_awaited_once()
