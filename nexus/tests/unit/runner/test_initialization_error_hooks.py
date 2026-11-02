"""Preserve task diagnostics while reporting setup failures through hooks."""

import asyncio
import json
import uuid

from unittest.mock import Mock

import pytest

from nexus.core import TaskRunInput, TaskRunOutput, task_registry
from nexus.runner import single
from nexus.runner.hooks import RunHook, run_hook_registry
from nexus.runner.single import ErrorInfo, RunHookError, RunHooks, RunInput, RunOutput, run


@pytest.mark.asyncio
@pytest.mark.parametrize('error_type', [ValueError, TimeoutError])
async def test_output_hook_receives_original_exception_without_changing_json(register_test_component, error_type):
    failure = error_type('task failed')
    failure.__cause__ = RuntimeError('original cause')
    name = f'exception_output_{uuid.uuid4().hex}'
    seen = []

    @register_test_component(task_registry, name)
    async def fail(_input: TaskRunInput) -> TaskRunOutput:
        raise failure

    def inspect(_ctx, output):
        seen.append(output.error.exception)

    output = await run(RunInput(task_name=name), hooks=RunHooks(on_run_output=inspect))

    assert not output.success
    assert seen == [failure]
    assert output.error.exception is failure
    assert output.error.exception.__cause__ is failure.__cause__
    encoded = output.model_dump_json()
    assert set(json.loads(encoded)['error']) == {'type', 'message', 'traceback'}
    assert RunOutput.model_validate_json(encoded).error.exception is None
    assert 'exception' not in ErrorInfo.model_json_schema()['properties']


@pytest.mark.asyncio
async def test_initialization_hook_receives_configuration_original_error_and_generic_output(monkeypatch):
    failure = ImportError('module setup failed')
    monkeypatch.setattr(single.Registry, 'import_module', Mock(side_effect=failure))
    config = RunInput(task_name='unused', imports=['test_import'], extra={'case': 'retained'})
    seen = []

    def inspect(run_input, error, output):
        seen.append((run_input, error, output))
        assert output.task_output is None
        assert output.agent_output is None
        assert output.error.exception is error
        output.hook_results['diagnostic'] = {'stage': 'initialization'}

    output = await run(config, hooks=RunHooks(on_initialization_error=inspect))

    assert seen == [(config, failure, output)]
    assert not output.success
    assert output.error.exception is failure
    assert output.extra == {'case': 'retained'}
    assert output.hook_results['diagnostic'] == {'stage': 'initialization'}
    assert RunOutput.model_validate_json(output.model_dump_json()).error.exception is None


@pytest.mark.asyncio
async def test_initialization_hook_failure_does_not_replace_the_original_error():
    original = []

    def report(_run_input, error, output):
        original.append(error)
        output.success = True
        output.error = ErrorInfo(type='HookError', message='replacement')
        raise RuntimeError('reporting failed')

    hooks = RunHooks(on_initialization_error=report)
    output = await run(RunInput(task_name='unused', imports=['missing_setup_module']), hooks=hooks)

    assert not output.success
    assert output.error.exception is original[0]
    assert output.error.type == 'ImportError'
    assert output.hook_errors['runner.on_initialization_error'] == 'RuntimeError: reporting failed'
    assert hooks.hook_failures[0].stage == 'initialization_error'
    with pytest.raises(RunHookError, match='reporting failed'):
        hooks.raise_if_failed()


@pytest.mark.asyncio
@pytest.mark.parametrize('failure_stage', ['callback', 'get_result'])
async def test_failed_initialization_hook_does_not_skip_other_error_reporters(register_test_component, failure_stage):
    events = []
    name = f'initialization_reporter_{uuid.uuid4().hex}'

    @register_test_component(run_hook_registry, name)
    class Reporter(RunHook):
        def __init__(self, label):
            self.label = label

        async def on_initialization_error(self, run_input, error, output):
            assert output.error.exception is error
            events.append(self.label)
            if self.label == 'bad' and failure_stage == 'callback':
                raise RuntimeError('reporter failed')

        def get_result(self):
            if self.label == 'bad' and failure_stage == 'get_result':
                raise RuntimeError('result failed')
            return {'label': self.label}

    hooks = RunHooks(on_initialization_error=lambda *_: events.append('embedding'))
    config = RunInput(
        task_name='unused',
        imports=['missing_setup_module'],
        runner={'hooks': {label: {'name': name, 'args': {'label': label}} for label in ['bad', 'good']}},
    )
    output = await run(config, hooks=hooks)

    assert events == ['bad', 'good', 'embedding']
    assert not output.success
    assert output.error.type == 'ImportError'
    assert output.hook_results['good'] == {'label': 'good'}
    stage = 'on_initialization_error' if failure_stage == 'callback' else 'get_result'
    assert f'bad.{stage}' in output.hook_errors
    assert hooks.hook_failures[0].stage == 'initialization_error'


@pytest.mark.asyncio
async def test_failed_hook_constructor_is_not_retried_and_other_hooks_report_its_error(register_test_component):
    failure = RuntimeError('hook constructor failed')
    calls = []
    name = f'initialization_constructor_{uuid.uuid4().hex}'
    task_name = f'initialization_task_{uuid.uuid4().hex}'

    @register_test_component(task_registry, task_name)
    async def unused(_input: TaskRunInput) -> TaskRunOutput:
        raise AssertionError('The task must not start')

    @register_test_component(run_hook_registry, name)
    class Reporter(RunHook):
        def __init__(self, label):
            calls.append(label)
            if label == 'bad':
                raise failure

        async def on_initialization_error(self, run_input, error, output):
            assert error is failure
            output.hook_results['reported'] = True

    config = RunInput(
        task_name=task_name,
        runner={'hooks': {label: {'name': name, 'args': {'label': label}} for label in ['bad', 'good']}},
    )
    output = await run(config)

    assert calls == ['bad', 'good']
    assert output.error.exception is failure
    assert output.hook_results['reported'] is True
    assert output.hook_errors['bad.initialize'] == 'RuntimeError: hook constructor failed'


@pytest.mark.asyncio
async def test_plain_configured_hook_preserves_legacy_setup_exception(register_test_component, monkeypatch):
    name = f'plain_hook_{uuid.uuid4().hex}'
    task_name = f'plain_hook_task_{uuid.uuid4().hex}'

    @register_test_component(task_registry, task_name)
    async def unused(_input: TaskRunInput) -> TaskRunOutput:
        raise AssertionError('The task must not start')

    @register_test_component(run_hook_registry, name)
    class PlainHook(RunHook):
        pass

    monkeypatch.setattr(single, '_build_input', Mock(side_effect=ValueError('invalid task input')))
    with pytest.raises(ValueError, match='invalid task input'):
        await run(RunInput(task_name=task_name, runner={'hooks': {'plain': {'name': name}}}))


@pytest.mark.asyncio
async def test_import_failure_does_not_construct_unrelated_hooks(register_test_component):
    name = f'unrelated_hook_{uuid.uuid4().hex}'
    constructed = []

    @register_test_component(run_hook_registry, name)
    class PlainHook(RunHook):
        def __init__(self):
            constructed.append(True)

    output = await run(
        RunInput(
            task_name='unused',
            imports=['missing_setup_module'],
            runner={'hooks': {'plain': {'name': name}}},
        )
    )

    assert output.error.type == 'ImportError'
    assert not constructed
    assert not output.hook_errors


@pytest.mark.asyncio
async def test_initialization_callback_cancellation_is_propagated():
    def cancel(*_args):
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await run(
            RunInput(task_name='unused', imports=['missing_setup_module']),
            hooks=RunHooks(on_initialization_error=cancel),
        )
