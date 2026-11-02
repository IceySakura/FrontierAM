"""Exercise platform statuses at the agent and native runner output boundaries."""

import asyncio
import copy
import uuid

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from openai.types.chat import ChatCompletion, ChatCompletionMessage

from nexus.core import ContextLengthExceededError, TaskRunInput, TaskRunOutput, task_registry
from nexus.core.context import require_context
from nexus.extensions.agents.swe_agent import agent
from nexus.extensions.agents.swe_agent.platform_status import (
    CONTRACT_ID,
    EXIT_CODES,
    PlatformExitStatus,
    exception_details,
    runtime_failure_status,
)
from nexus.extensions.tasks.swe.platform import STATE_KEY, SuperSWEPlatformHook
from nexus.runner.single import RunInput, RunOutput, run
from nexus.runtime.client.base import RuntimeUnhealthyError
from nexus.runtime.client.remote import RemoteRuntime
from nexus.runtime.providers.gongfeng import SandboxApiError


def completion(*, content=None, finish='tool_calls', arguments='{"command": "true"}'):
    message = ChatCompletionMessage(
        role='assistant',
        content=content,
        tool_calls=(
            [{'id': 'call_fixture', 'type': 'function', 'function': {'name': 'bash', 'arguments': arguments}}]
            if arguments is not None
            else None
        ),
    )
    response = ChatCompletion(
        id='fixture',
        created=0,
        model='fixture',
        object='chat.completion',
        choices=[{'index': 0, 'message': message, 'finish_reason': 'length' if finish == 'max_tokens' else finish}],
    )
    response.choices[0].finish_reason = finish
    return response


@pytest.fixture
def harness(monkeypatch):
    client = SimpleNamespace(create_completion=AsyncMock(return_value=completion()))
    ctx = SimpleNamespace(
        task_name='superswe_task',
        openai_client=client,
        tool_state={},
        task_config={'problem_statement': 'Fix the behavior.', 'working_dir': '/testbed', 'language': 'py'},
    )
    executor = SimpleNamespace(
        get_tool_schemas=Mock(return_value=[]),
        call=AsyncMock(return_value=SimpleNamespace(output=[{'text': 'ok'}])),
    )
    monkeypatch.setattr(agent, 'require_context', lambda: ctx)
    monkeypatch.setattr(agent, 'ToolExecutor', lambda tools: executor)
    request = agent.SWEAgentInput(
        user_prompt='Fix the behavior.',
        working_dir='/testbed',
        max_iterations=1,
        platform_exit_status=True,
    )
    return ctx, client, executor, request


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('response', 'status'),
    [
        (completion(content='Done.', finish='stop', arguments=None), 'finished'),
        (completion(finish='stop', arguments=None), 'inference_error'),
        (completion(content='cut off', finish='length', arguments=None), 'output_length_exceeded'),
        (completion(finish='length'), 'output_length_exceeded'),
        (completion(finish='max_tokens'), 'output_length_exceeded'),
        (completion(arguments='{broken'), 'unknown_error'),
        (completion(), 'max_iterations'),
    ],
)
async def test_response_and_budget_statuses(harness, response, status):
    _, client, _, request = harness
    client.create_completion.return_value = response
    result = await agent.run_swe_agent(request)
    assert result.exit_status.value == status
    assert result.exit_code == EXIT_CODES[PlatformExitStatus(status)]
    assert result.iterations == 1
    if status not in ('finished',):
        assert result.error


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('error', 'status'),
    [
        (TimeoutError('provider timeout'), 'inference_error'),
        (ContextLengthExceededError('input is too long'), 'context_length_exceeded'),
        (RuntimeError('upstream rejected request'), 'inference_error'),
    ],
)
async def test_inference_failures_keep_the_real_error(harness, error, status):
    _, client, _, request = harness
    client.create_completion.side_effect = error
    result = await agent.run_swe_agent(request)
    assert result.exit_status.value == status
    assert str(error) in result.error


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('reason', 'code', 'status'),
    [
        ('Process.OOMKilled', 260422, 'runtime_error_by_agent'),
        ('Process.NonZeroExit', 260422, 'runtime_error_by_agent'),
        ('Process.CrashLoopBackOff', 260422, 'runtime_error_by_agent'),
        ('Process.Killed', 260500, 'runtime_error_by_env'),
        ('Process.Killed', 260422, 'runtime_error_by_env'),
        ('Process.ExceedStorageLimit', 260422, 'runtime_error_by_env'),
        ('Platform.UnknownError', 260500, 'runtime_error_by_env'),
        ('Runtime.NotFound', 260404, 'runtime_error_by_env'),
        ('Schedule.NodeNotReady', 260503, 'runtime_error_by_env'),
        ('Queue.Timeout', 260422, 'runtime_error_by_env'),
        ('Queue.GiveUp', 260422, 'runtime_error_by_env'),
        ('Process.OOMKilled', 260500, 'runtime_error_by_env'),
        ('Process.OOMKilled', '260422', 'runtime_error_by_env'),
        ('Process.FutureReason', 260422, 'runtime_error_by_env'),
        (None, 260422, 'runtime_error_by_env'),
        (None, None, 'runtime_error_by_env'),
    ],
)
async def test_runtime_attribution_needs_structured_evidence(harness, reason, code, status):
    _, _, executor, request = harness
    runtime = RemoteRuntime(base_url='http://runtime.example.com', auto_connect_ws=False)
    runtime.mark_unhealthy(
        'diagnostic text includes OOMKilled and exit 137',
        details={'reason': reason, 'code': code, 'exit_code': 137},
    )
    with pytest.raises(RuntimeUnhealthyError) as captured:
        runtime._raise_if_unhealthy()
    executor.call.side_effect = captured.value
    result = await agent.run_swe_agent(request)
    assert result.exit_status.value == status
    assert result.exit_code == EXIT_CODES[PlatformExitStatus(status)]
    assert result.failure_kind == ('agent' if status == 'runtime_error_by_agent' else 'infrastructure')
    assert result.sandbox['reason'] == reason


@pytest.mark.asyncio
@pytest.mark.parametrize('error_type', [SandboxApiError, RuntimeUnhealthyError])
@pytest.mark.parametrize('wrapper', [None, RuntimeError, TimeoutError, agent.ToolInfrastructureError])
async def test_direct_and_wrapped_sandbox_failures_stop_the_tool_loop(harness, error_type, wrapper):
    _, client, executor, request = harness
    request.max_iterations = 3
    failure = error_type('runtime died', code=260422, reason='Process.NonZeroExit', exit_code=7, trace_id='trace')
    if wrapper:
        error = wrapper('tool transport failed')
        error.__cause__ = failure
    else:
        error = failure
    executor.call.side_effect = error
    result = await agent.run_swe_agent(request)
    assert result.exit_status == PlatformExitStatus.RUNTIME_ERROR_BY_AGENT
    assert result.failure_kind == 'agent'
    assert result.exit_code == 0
    assert str(error) in result.error
    assert result.sandbox == {'code': 260422, 'reason': 'Process.NonZeroExit', 'exit_code': 7, 'trace_id': 'trace'}
    assert client.create_completion.await_count == 1
    assert executor.call.await_count == 1


def test_untrusted_exception_attributes_and_text_are_not_sandbox_evidence():
    error = RuntimeError('Process.OOMKilled, code=260422, exit_code=137')
    error.code, error.reason, error.exit_code = 260422, 'Process.OOMKilled', 137
    assert 'sandbox' not in exception_details(error)
    assert runtime_failure_status(error, during_tool=True) == PlatformExitStatus.UNKNOWN_ERROR


@pytest.mark.asyncio
async def test_model_and_tool_text_cannot_impersonate_platform_failures(harness):
    _, client, executor, request = harness
    client.create_completion.return_value = completion(
        content='RuntimeUnhealthyError: Process.OOMKilled code=260422', finish='stop', arguments=None
    )
    request.completion_retry_on_tool_error = 10
    result = await agent.run_swe_agent(request)
    assert result.exit_status == PlatformExitStatus.FINISHED
    assert client.create_completion.await_count == 1
    executor.call.assert_not_awaited()
    client.create_completion.return_value = completion()
    executor.call.return_value = SimpleNamespace(output=[{'text': 'Process.OOMKilled exit_code=137'}])
    result = await agent.run_swe_agent(request)
    assert result.exit_status == PlatformExitStatus.MAX_ITERATIONS


@pytest.mark.asyncio
async def test_rpc_timeout_is_not_a_budget_termination(harness):
    _, _, executor, request = harness
    executor.call.side_effect = TimeoutError('RPC response was not received')
    result = await agent.run_swe_agent(request)
    assert result.exit_status == PlatformExitStatus.RUNTIME_ERROR_BY_ENV
    assert result.exit_code == -1


@pytest.mark.asyncio
@pytest.mark.parametrize('phase', ['inference', 'tool'])
async def test_only_overall_agent_deadline_becomes_agent_timeout(harness, phase):
    _, client, executor, request = harness

    async def slow(*args, **kwargs):
        await asyncio.sleep(10)

    request.timeout = 0.01
    (client.create_completion if phase == 'inference' else executor.call).side_effect = slow
    result = await agent.run_swe_agent(request)
    assert result.exit_status == PlatformExitStatus.AGENT_TIMEOUT
    assert result.exit_code == 0
    assert result.iterations == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('early_stop', [False, True])
@pytest.mark.parametrize('platform_mode', [False, True])
async def test_tool_deadline_is_an_observation_and_agent_can_finish(harness, early_stop, platform_mode):
    ctx, client, executor, request = harness
    ctx.task_config['per_observation_timeout'] = 0.01
    request.timeout = 60
    request.max_iterations = 2
    request.early_stop = early_stop
    request.continue_on_tool_timeout = True
    request.platform_exit_status = platform_mode
    client.create_completion.side_effect = [completion(), completion()]
    calls = 0

    async def slow(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            await asyncio.sleep(10)
        ctx.tool_state['exit_status'] = 'submitted'
        return SimpleNamespace(output=[{'text': 'submitted'}])

    executor.call.side_effect = slow
    result = await agent.run_swe_agent(request)
    assert result.exit_status.value == ('finished' if platform_mode else 'submitted')
    assert result.exit_code == (0 if platform_mode else None)
    assert result.error is None
    assert result.iterations == 2
    assert client.create_completion.await_count == 2
    history = client.create_completion.call_args.kwargs['messages']
    assert any(item.get('role') == 'tool' and 'per_observation_timeout=0.01s' in item['content'] for item in history)


@pytest.mark.asyncio
async def test_tool_deadline_with_no_more_turns_is_max_iterations(harness):
    ctx, _, executor, request = harness
    ctx.task_config['per_observation_timeout'] = 0.01
    request.continue_on_tool_timeout = True

    async def slow(*args, **kwargs):
        await asyncio.sleep(10)

    executor.call.side_effect = slow
    result = await agent.run_swe_agent(request)
    assert result.exit_status == PlatformExitStatus.MAX_ITERATIONS
    assert result.exit_code == 0


@pytest.mark.asyncio
@pytest.mark.parametrize('platform_mode', [False, True])
async def test_owned_tool_timeout_does_not_continue_without_explicit_policy(harness, platform_mode):
    ctx, client, executor, request = harness
    request.platform_exit_status = platform_mode
    request.max_iterations = 2
    ctx.task_config['per_observation_timeout'] = 0.01

    async def slow(*args, **kwargs):
        # Native mode delegates its deadline to ToolExecutor; platform mode owns the wrapper.
        async with asyncio.timeout(kwargs.get('timeout')):
            await asyncio.sleep(10)

    executor.call.side_effect = slow
    result = await agent.run_swe_agent(request)
    assert request.continue_on_tool_timeout is False
    assert client.create_completion.await_count == 1
    assert result.failure_stage == ('tool' if platform_mode else None)
    assert result.exit_status.value == ('runtime_error_by_env' if platform_mode else 'timeout')


@pytest.mark.asyncio
async def test_inference_budget_expiry_is_inference_error_without_an_agent_deadline(harness):
    ctx, client, _, request = harness
    ctx.task_config['inference_time_timeout'] = 0.01

    async def slow(**kwargs):
        await asyncio.sleep(10)

    client.create_completion.side_effect = slow
    result = await agent.run_swe_agent(request)
    assert result.exit_status == PlatformExitStatus.INFERENCE_ERROR
    assert result.exit_code == -1
    assert 'Inference time budget exhausted' in result.error
    assert request.timeout is None


@pytest.mark.asyncio
async def test_continuing_owned_timeout_still_respects_native_total_tool_budget(harness):
    ctx, client, executor, request = harness
    request.platform_exit_status = False
    request.continue_on_tool_timeout = True
    request.max_iterations = 2
    ctx.task_config.update(per_observation_timeout=0.01, observation_time_timeout=0.001)

    async def slow(*args, **kwargs):
        await asyncio.sleep(10)

    executor.call.side_effect = slow
    result = await agent.run_swe_agent(request)
    assert result.exit_status.value == 'timeout'
    assert client.create_completion.await_count == 1


@pytest.mark.asyncio
async def test_platform_mode_rejects_an_unrepresentable_total_tool_budget(harness):
    ctx, client, _, request = harness
    ctx.task_config['observation_time_timeout'] = 1
    result = await agent.run_swe_agent(request)
    assert result.exit_status == PlatformExitStatus.ENVIRONMENT_ERROR
    assert result.exit_code == -1
    assert result.iterations == 0
    assert 'agent_config.timeout' in result.error
    client.create_completion.assert_not_awaited()


@pytest.mark.asyncio
async def test_legacy_total_tool_budget_is_still_available(harness):
    ctx, _, _, request = harness
    request.platform_exit_status = False
    ctx.task_config['observation_time_timeout'] = 0.000001
    result = await agent.run_swe_agent(request)
    assert result.exit_status == agent.ExitStatus.TIMEOUT
    assert result.exit_code is None


@pytest.mark.asyncio
async def test_setup_failure_becomes_environment_error(harness, monkeypatch):
    _, _, _, request = harness
    monkeypatch.setattr(agent, 'ToolExecutor', Mock(side_effect=ImportError('missing harness dependency')))
    result = await agent.run_swe_agent(request)
    assert result.exit_status == PlatformExitStatus.ENVIRONMENT_ERROR
    assert result.exit_code == -1
    assert 'missing harness dependency' in result.error


@pytest.mark.asyncio
@pytest.mark.parametrize('wrapper', [None, RuntimeError, TimeoutError])
async def test_sandbox_death_during_agent_setup_is_environment_failure(harness, monkeypatch, wrapper):
    _, _, _, request = harness
    error = RuntimeUnhealthyError('died during setup', code=260422, reason='Process.OOMKilled')
    if wrapper:
        wrapped = wrapper('setup transport failed')
        wrapped.__cause__ = error
        error = wrapped
    monkeypatch.setattr(agent, 'ToolExecutor', Mock(side_effect=error))
    result = await agent.run_swe_agent(request)
    assert result.exit_status == PlatformExitStatus.RUNTIME_ERROR_BY_ENV
    assert result.exit_code == -1
    assert result.failure_stage == 'setup'
    assert result.iterations == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('agent_status', 'agent_code'),
    [
        ('finished', 0),
        ('agent_timeout', 0),
        ('max_iterations', 0),
        ('context_length_exceeded', 0),
        ('output_length_exceeded', 0),
        ('runtime_error_by_agent', 0),
        ('inference_error', -1),
        ('runtime_error_by_env', -1),
        ('environment_error', -1),
        ('unknown_error', -1),
    ],
)
@pytest.mark.parametrize('judge_code', [0, -1])
async def test_agent_judge_priority_preserves_actual_score_and_error(agent_status, agent_code, judge_code):
    ctx = SimpleNamespace(tool_state={})
    hook = SuperSWEPlatformHook()
    await hook.on_context_ready(ctx)
    ctx.tool_state[STATE_KEY].update(agent_started=True, judge_started=True, stage='judge.evaluate')
    output = RunOutput(
        extra={'superswe': {'mode': 'model'}},
        agent_output={'exit_status': agent_status, 'exit_code': agent_code, 'error': 'agent detail'},
        judge_output={
            'exit_status': 'finished' if judge_code == 0 else 'judge_error',
            'exit_code': judge_code,
            'error': None if judge_code == 0 else 'judge detail',
        },
        task_output={'completed': judge_code == 0, 'score': 0.25, 'patch_text': 'saved'},
    )
    await hook.on_run_output(ctx, output)
    assert output.task_output['status'] == ('agent_error' if agent_code else 'judge_error' if judge_code else 'success')
    assert output.task_output['error'] == ('judge detail' if agent_code == 0 and judge_code else 'agent detail')
    assert output.task_output['score'] == 0.25
    assert output.success is (agent_code == 0 and judge_code == 0)
    assert output.task_output['patch_text'] == 'saved'
    assert output.model_dump(mode='json')['judge_output']['exit_code'] == judge_code


@pytest.mark.asyncio
async def test_platform_mode_keeps_the_existing_swe_prompt(harness):
    ctx, client, _, request = harness
    await agent.run_swe_agent(request.model_copy(deep=True))
    superswe_prompt = copy.deepcopy(client.create_completion.call_args.kwargs['messages'][1])
    ctx.task_name = 'swe_task'
    await agent.run_swe_agent(request.model_copy(deep=True))
    assert client.create_completion.call_args.kwargs['messages'][1] == superswe_prompt
    assert '/testbed' in superswe_prompt['content']
    assert '<pr_description>' in superswe_prompt['content']


@pytest.mark.asyncio
async def test_legacy_agent_status_remains_available(harness):
    _, _, _, request = harness
    request.platform_exit_status = False
    result = await agent.run_swe_agent(request)
    assert result.exit_status == agent.ExitStatus.MAX_ITERATIONS
    assert result.exit_code is None


@pytest.mark.asyncio
@pytest.mark.parametrize(('resolved', 'agent_code', 'status'), [(False, 0, 'success'), (True, -1, 'agent_error')])
async def test_valid_grading_is_kept_when_setting_task_status(resolved, agent_code, status):
    ctx = SimpleNamespace(tool_state={})
    hook = SuperSWEPlatformHook()
    await hook.on_context_ready(ctx)
    ctx.tool_state[STATE_KEY].update(agent_started=True, judge_started=True, stage='judge.evaluate')
    output = RunOutput(
        extra={'superswe': {'mode': 'model'}},
        agent_output={
            'exit_status': 'finished' if agent_code == 0 else 'inference_error',
            'exit_code': agent_code,
            'error': None if agent_code == 0 else 'real upstream failure',
        },
        task_output={'completed': True, 'resolved': resolved, 'score': float(resolved), 'patch_text': 'saved patch'},
    )
    await hook.on_run_output(ctx, output)
    assert output.task_output['status'] == status
    assert output.task_output['score'] == float(resolved)
    assert output.task_output['patch_text'] == 'saved patch'
    assert hook.get_result()['contract'] == CONTRACT_ID


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('stage', 'status'), [('runtime.create', 'runtime_error_by_env'), ('environment.initialize', 'environment_error')]
)
async def test_failure_before_agent_does_not_claim_model_execution(stage, status):
    ctx = SimpleNamespace(tool_state={})
    hook = SuperSWEPlatformHook()
    await hook.on_context_ready(ctx)
    ctx.tool_state[STATE_KEY]['stage'] = stage
    output = RunOutput(
        extra={'superswe': {'mode': 'model'}},
        task_output={'completed': False, 'resolved': False, 'score': 0, 'error': 'initialization failed'},
    )
    await hook.on_run_output(ctx, output)
    assert output.agent_output['agent_started'] is False
    assert output.agent_output['exit_status'] == status
    assert output.task_output['status'] == 'agent_error'


@pytest.mark.asyncio
@pytest.mark.parametrize('error_type', [RuntimeUnhealthyError, SandboxApiError, RuntimeError])
@pytest.mark.parametrize('wrapped', [False, True])
async def test_output_hook_attributes_uncaught_errors_from_original_exception(
    register_test_component, error_type, wrapped
):
    evidence = {'code': 260422, 'reason': 'Process.OOMKilled', 'exit_code': 137, 'trace_id': 'fixture'}
    if error_type is RuntimeError:
        failure = RuntimeError('Process.OOMKilled code=260422')
        failure.code, failure.reason = 260422, 'Process.OOMKilled'
    else:
        failure = error_type('runtime failed', **evidence)
    if wrapped:
        wrapper = TimeoutError('transport failed')
        wrapper.__cause__ = failure
        failure = wrapper
    name = f'uncaught_platform_error_{uuid.uuid4().hex}'

    @register_test_component(task_registry, name)
    async def fail(_input: TaskRunInput) -> TaskRunOutput:
        require_context().tool_state[STATE_KEY]['patch_text'] = 'saved patch'
        raise failure

    output = await run(
        RunInput(
            task_name=name,
            extra={'superswe': {'mode': 'model'}},
            runner={'hooks': {'superswe_platform': {'name': 'superswe_platform'}}},
        )
    )

    expected = 'unknown_error' if error_type is RuntimeError else 'runtime_error_by_env'
    assert output.agent_output['exit_status'] == expected
    assert output.agent_output['agent_started'] is False
    assert output.agent_output['sandbox'] == (None if error_type is RuntimeError else evidence)
    assert output.error.exception is failure
    assert not output.success
    assert output.task_output['patch_text'] == 'saved patch'
    assert output.task_output['completed'] is False
    assert output.hook_results['superswe_platform']['failure']['type'] == type(failure).__name__
    assert 'exception' not in output.model_dump(mode='json')['error']


@pytest.mark.asyncio
@pytest.mark.parametrize('patch', ['', ' \n', 'saved patch'])
@pytest.mark.parametrize('resolved', [False, True])
async def test_explicit_patch_does_not_require_an_agent_result(patch, resolved):
    ctx = SimpleNamespace(tool_state={}, task_config={'patch_text': patch})
    hook = SuperSWEPlatformHook()
    await hook.on_context_ready(ctx)
    ctx.tool_state[STATE_KEY].update(judge_started=True, stage='judge.evaluate')
    output = RunOutput(
        extra={'superswe': {'mode': 'model'}},
        task_output={'completed': True, 'resolved': resolved, 'score': float(resolved), 'patch_text': patch},
    )

    await hook.on_run_output(ctx, output)

    assert output.agent_output is None
    assert output.task_output['status'] == 'success'
    assert output.task_output['error'] is None
    assert output.task_output['resolved'] is resolved
    assert output.task_output['score'] == float(resolved)
    assert hook.get_result()['agent_started'] is False


@pytest.mark.asyncio
@pytest.mark.parametrize('patch', ['', 'saved patch'])
async def test_direct_grading_failure_keeps_the_judge_error(patch):
    ctx = SimpleNamespace(tool_state={}, task_config={'patch_text': patch})
    hook = SuperSWEPlatformHook()
    await hook.on_context_ready(ctx)
    ctx.tool_state[STATE_KEY].update(judge_started=True, stage='judge.evaluate')
    output = RunOutput(
        extra={'superswe': {'mode': 'model'}},
        task_output={'completed': False, 'resolved': False, 'score': 0, 'patch_text': patch, 'error': 'judge failed'},
    )

    await hook.on_run_output(ctx, output)

    assert output.agent_output is None
    assert output.task_output['status'] == 'judge_error'
    assert output.task_output['error'] == 'judge failed'
    assert output.task_output['patch_text'] == patch


@pytest.mark.asyncio
@pytest.mark.parametrize('task_config', [{}, {'patch_text': None}])
async def test_absent_patch_still_requires_the_model_result(task_config):
    ctx = SimpleNamespace(tool_state={}, task_config=task_config)
    hook = SuperSWEPlatformHook()
    await hook.on_context_ready(ctx)
    output = RunOutput(extra={'superswe': {'mode': 'model'}})

    await hook.on_run_output(ctx, output)

    assert output.agent_output['agent_started'] is False
    assert output.agent_output['exit_code'] == -1
    assert output.task_output['status'] == 'agent_error'


@pytest.mark.asyncio
async def test_explicit_patch_does_not_erase_a_real_agent_error():
    ctx = SimpleNamespace(tool_state={}, task_config={'patch_text': 'saved patch'})
    hook = SuperSWEPlatformHook()
    await hook.on_context_ready(ctx)
    output = RunOutput(
        extra={'superswe': {'mode': 'model'}},
        agent_output={'exit_status': 'inference_error', 'exit_code': -1, 'error': 'upstream failure'},
        task_output={'completed': True, 'resolved': False, 'score': 0, 'patch_text': 'saved patch'},
    )

    await hook.on_run_output(ctx, output)

    assert output.task_output['status'] == 'agent_error'
    assert output.task_output['error'] == 'upstream failure'
    assert output.agent_output['error'] == 'upstream failure'


@pytest.mark.asyncio
async def test_oracle_control_does_not_require_an_agent_in_model_mode():
    ctx = SimpleNamespace(tool_state={}, task_config={'run_oracle': True})
    hook = SuperSWEPlatformHook()
    await hook.on_context_ready(ctx)
    output = RunOutput(
        extra={'superswe': {'mode': 'model'}},
        task_output={'completed': True, 'resolved': True, 'score': 1, 'patch_text': 'reference'},
    )

    await hook.on_run_output(ctx, output)

    assert output.agent_output is None
    assert output.task_output['status'] == 'success'


@pytest.mark.asyncio
async def test_judge_failure_keeps_saved_patch_and_original_agent_status():
    ctx = SimpleNamespace(tool_state={})
    hook = SuperSWEPlatformHook()
    await hook.on_context_ready(ctx)
    ctx.tool_state[STATE_KEY].update(agent_started=True, judge_started=True, stage='judge.evaluate', patch_text='saved')
    output = RunOutput(
        extra={'superswe': {'mode': 'model'}},
        success=False,
        error={'type': 'TimeoutError', 'message': 'judge timed out'},
        agent_output={'exit_status': 'finished', 'exit_code': 0, 'error': None},
    )
    await hook.on_run_output(ctx, output)
    assert output.agent_output['exit_status'] == 'finished'
    assert output.task_output['status'] == 'judge_error'
    assert output.task_output['completed'] is False
    assert output.task_output['patch_text'] == 'saved'


def test_first_runtime_failure_keeps_an_independent_evidence_copy():
    runtime = RemoteRuntime(base_url='http://runtime.example.com', auto_connect_ws=False)
    evidence = {'code': 260422, 'reason': 'Process.OOMKilled', 'exit_code': 137, 'trace_id': 'first'}
    runtime.mark_unhealthy('first failure', details=evidence)
    evidence['reason'] = 'Process.NonZeroExit'
    runtime.mark_unhealthy('later failure', details={'reason': 'Runtime.Stopped'})
    with pytest.raises(RuntimeUnhealthyError) as captured:
        runtime._raise_if_unhealthy()
    assert str(captured.value) == 'first failure'
    assert captured.value.reason == 'Process.OOMKilled'
    assert captured.value.trace_id == 'first'
    assert runtime._unhealthy_event.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('error', 'expected'),
    [
        (ModuleNotFoundError("No module named 'pyte'"), PlatformExitStatus.ENVIRONMENT_ERROR),
        (agent.ToolInfrastructureError('command executor failed'), PlatformExitStatus.RUNTIME_ERROR_BY_ENV),
    ],
)
async def test_missing_tool_environment_stops_before_model_repairs_it(harness, error, expected):
    _, client, executor, request = harness
    request.max_iterations = 3
    executor.call.side_effect = error
    result = await agent.run_swe_agent(request)
    assert result.exit_status == expected
    assert result.exit_code == -1
    assert result.failure_stage == 'tool'
    assert str(error) in result.error
    assert client.create_completion.await_count == 1


@pytest.mark.asyncio
async def test_stateless_backend_reaches_tools_and_prompt(harness):
    ctx, client, _, request = harness
    request.bash_backend = 'stateless'
    await agent.run_swe_agent(request)
    assert ctx.tool_state['bash_backend'] == 'stateless'
    assert 'fresh shell' in client.create_completion.call_args.kwargs['messages'][0]['content']


@pytest.mark.asyncio
async def test_stateless_bash_runs_without_terminal_dependencies(tmp_path, monkeypatch):
    import builtins

    from nexus.extensions.agents.swe_agent.tools import BashInput, bash_tool
    from nexus.runtime.executors import command

    create_process = command._create_command_process

    async def isolated_process(command_text, *, cwd, env):
        # Keep the real subprocess; unrelated host shell startup is outside this fixture.
        return await create_process(command_text, cwd=cwd, env={**env, 'BASH_ENV': ''})

    monkeypatch.setattr(command, '_create_command_process', isolated_process)
    original_import = builtins.__import__

    def checked_import(name, *args, **kwargs):
        if name.startswith('nexus.runtime.executors.terminal') or name == 'pyte':
            raise AssertionError('Stateless commands must not import terminal capture')
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', checked_import)
    ctx = SimpleNamespace(tool_state={'working_dir': str(tmp_path), 'bash_backend': 'stateless'})
    first = await bash_tool(BashInput(command='export SUPERSWE_TEST_STATE=private; pwd; exit 7'), ctx)
    second = await bash_tool(BashInput(command='printf "%s/%s" "$PAGER" "${SUPERSWE_TEST_STATE-unset}"'), ctx)
    assert str(tmp_path) in first.output[0]['text']
    assert first.meta['exit_code'] == 7
    assert first.meta['status'] == 'Finished'
    assert second.output[0]['text'] == 'cat/unset'


@pytest.mark.asyncio
@pytest.mark.parametrize('status', ['TimeLimitExceeded', 'Error'])
async def test_stateless_command_timeout_and_executor_failure_are_distinct(monkeypatch, status):
    from nexus.extensions.agents.swe_agent.tools import BashInput, ToolInfrastructureError, bash_tool
    from nexus.runtime.executors import command
    from nexus.runtime.executors.models.command import CommandRunStatus, RunCommandResponse

    monkeypatch.setattr(
        command,
        'run_command',
        AsyncMock(return_value=RunCommandResponse(status=CommandRunStatus(status), stderr='execution evidence')),
    )
    ctx = SimpleNamespace(tool_state={'working_dir': '/testbed', 'bash_backend': 'stateless'})
    if status == 'Error':
        with pytest.raises(ToolInfrastructureError, match='execution evidence'):
            await bash_tool(BashInput(command='fixture'), ctx)
    else:
        result = await bash_tool(BashInput(command='fixture'), ctx)
        assert 'execution evidence' in result.output[0]['text']
        assert 'timed out' in result.output[0]['text']
        assert result.meta['status'] == status
