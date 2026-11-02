"""Keep the opt-in AgentBench contract available before runtime hooks can run."""

import asyncio
import uuid

from unittest.mock import AsyncMock, Mock

import pytest

from nexus.core import TaskRunInput, TaskRunOutput, WithAgentRunner, task_registry
from nexus.runner import single
from nexus.runtime.client.base import RuntimeUnhealthyError
from nexus.runtime.providers.gongfeng import SandboxApiError


class SetupTaskInput(TaskRunInput, WithAgentRunner):
    pass


@pytest.fixture
def request_config(register_test_component):
    name = f'platform_setup_{uuid.uuid4().hex}'

    @register_test_component(task_registry, name)
    async def never_execute(_task_input: SetupTaskInput) -> TaskRunOutput:
        raise AssertionError('Setup failure must precede task execution')

    return single.RunInput(
        task_name=name,
        agent_name='swe_agent',
        agent_config={'platform_exit_status': True},
        runner={'hooks': {'superswe_platform': {'name': 'superswe_platform'}}},
        run_id='fixture-setup',
        extra={'case': 'retained'},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'phase',
    [
        'import',
        'configured_hook',
        'provider',
        'client',
        'agent_factory',
        'task_input',
        'processors',
        'observers',
        'cache',
        'task_files',
        'context_create',
        'context_enter',
        'processor_enter',
        'ready_hook',
        'host_materialization',
    ],
)
async def test_setup_failures_return_standard_agent_and_task_outputs(monkeypatch, tmp_path, request_config, phase):
    failure = ImportError(f'fixture {phase} setup failed')
    fail = Mock(side_effect=failure)
    hooks = None
    if phase == 'import':
        request_config.imports = ['missing_platform_setup_fixture_module']
    elif phase == 'configured_hook':
        request_config.runner.hooks['missing'] = single.HookConfig(name='missing_platform_setup_fixture_hook')
    elif phase == 'client':
        request_config.openai_clients = {'main': {'name': 'fixture'}}
        monkeypatch.setattr(single.openai_client_registry, 'get', lambda _: fail)
    elif phase == 'context_enter':
        monkeypatch.setattr(single.ExecutionContext, '__aenter__', AsyncMock(side_effect=failure))
    elif phase == 'processor_enter':
        manager = AsyncMock()
        manager.__aenter__.side_effect = failure
        monkeypatch.setattr(single, '_processor_scope', lambda _: manager)
    elif phase == 'ready_hook':
        hooks = single.RunHooks(on_context_ready=AsyncMock(side_effect=failure))
    elif phase == 'host_materialization':
        request_config.runner.task_files_materialization = type(request_config.runner.task_files_materialization)(
            mode='cwd', target='host'
        )
    else:
        target = {
            'provider': '_create_runtime_provider',
            'agent_factory': '_build_run_agent_factory',
            'task_input': '_build_input',
            'processors': '_create_processors',
            'observers': 'create_trajectory_observers',
            'cache': 'load_cache',
            'task_files': 'task_files_from_spec',
            'context_create': 'ExecutionContext',
        }[phase]
        monkeypatch.setattr(single, target, fail)
        if phase == 'cache':
            request_config.runner.cache = single.CacheConfig(file_path=str(tmp_path / 'fixture.pkl'))

    result = await single.run(request_config, hooks=hooks)
    assert result.success is False
    assert result.extra == {'case': 'retained'}
    assert result.agent_output['exit_status'] == 'environment_error'
    assert result.agent_output['exit_code'] == -1
    assert result.agent_output['iterations'] == 0
    assert result.agent_output['agent_started'] is False
    assert result.task_output['status'] == 'agent_error'
    assert result.task_output['error'] == result.agent_output['error']
    assert result.error.message in result.task_output['error']
    assert result.task_output['score'] == 0
    assert result.judge_output is None  # No grading has started.
    assert result.model_dump(mode='json')['task_output']['status'] == 'agent_error'


@pytest.mark.asyncio
async def test_context_exposes_effective_agent_configuration_before_task_execution(register_test_component):
    from nexus.core.context import require_context

    observed = {}
    name = f'context_agent_config_{uuid.uuid4().hex}'

    @register_test_component(task_registry, name)
    async def inspect_context(_task_input: TaskRunInput) -> TaskRunOutput:
        observed.update(require_context().agent_config)
        return TaskRunOutput()

    result = await single.run(
        single.RunInput(task_name=name, agent_config={'bash_backend': 'terminal', 'max_iterations': 2}),
        hooks=single.RunHooks(agent_input_overrides={'bash_backend': 'stateless'}),
    )
    assert result.success
    assert observed == {'bash_backend': 'stateless', 'max_iterations': 2}


@pytest.mark.asyncio
@pytest.mark.parametrize('error_type', [RuntimeUnhealthyError, SandboxApiError])
@pytest.mark.parametrize('wrapped', [False, True])
async def test_typed_sandbox_failure_before_agent_is_environment_attributed(
    monkeypatch,
    request_config,
    error_type,
    wrapped,
):
    evidence = {'code': 260422, 'reason': 'Process.OOMKilled', 'exit_code': 137, 'trace_id': 'fixture'}
    failure = error_type('sandbox failed before agent execution', **evidence)
    if wrapped:
        wrapper = RuntimeError('setup transport failed')
        wrapper.__cause__ = failure
        failure = wrapper
    monkeypatch.setattr(single, '_create_runtime_provider', Mock(side_effect=failure))
    result = await single.run(request_config)
    assert result.agent_output['exit_status'] == 'runtime_error_by_env'
    assert result.agent_output['exit_code'] == -1
    assert result.agent_output['sandbox'] == evidence
    assert result.task_output['status'] == 'agent_error'


@pytest.mark.asyncio
async def test_untrusted_error_text_and_attributes_do_not_impersonate_a_sandbox(monkeypatch, request_config):
    failure = RuntimeError('Process.OOMKilled code=260422')
    failure.code, failure.reason = 260422, 'Process.OOMKilled'
    monkeypatch.setattr(single, '_create_runtime_provider', Mock(side_effect=failure))
    result = await single.run(request_config)
    assert result.agent_output['exit_status'] == 'environment_error'
    assert result.agent_output['sandbox'] is None


@pytest.mark.asyncio
@pytest.mark.parametrize('task_config', [{'patch_text': ''}, {'patch_text': 'saved patch'}, {'run_oracle': True}])
async def test_direct_grading_setup_failure_reports_judge_error_without_an_agent(
    monkeypatch,
    request_config,
    task_config,
):
    request_config.task_config = task_config
    monkeypatch.setattr(single, '_create_runtime_provider', Mock(side_effect=ImportError('setup unavailable')))
    result = await single.run(request_config)
    assert result.agent_output is None
    assert result.judge_output['exit_status'] == 'judge_error'
    assert result.judge_output['exit_code'] == -1
    assert result.judge_output['tests_executed'] is False
    assert result.task_output['status'] == 'judge_error'
    assert result.task_output['error'] == result.judge_output['error']
    assert result.task_output['patch_text'] == task_config.get('patch_text', '')


@pytest.mark.asyncio
async def test_legacy_client_setup_still_raises_without_an_initialization_error_hook(monkeypatch, request_config):
    request_config.agent_config = {}
    request_config.runner.hooks = {}
    request_config.openai_clients = {'main': {'name': 'fixture'}}
    fail = Mock(side_effect=ValueError('legacy client setup failed'))
    monkeypatch.setattr(single.openai_client_registry, 'get', lambda _: fail)
    with pytest.raises(ValueError, match='legacy client setup failed'):
        await single.run(request_config)


@pytest.mark.asyncio
async def test_setup_cancellation_is_not_converted_to_a_completed_failure(monkeypatch, request_config):
    monkeypatch.setattr(single.ExecutionContext, '__aenter__', AsyncMock(side_effect=asyncio.CancelledError))
    with pytest.raises(asyncio.CancelledError):
        await single.run(request_config)
