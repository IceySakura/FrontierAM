from __future__ import annotations

import asyncio

from unittest.mock import AsyncMock

import pytest

from tests.unit.harbor_trial.test_compose_build import environment_factory as environment_factory
from tests.unit.harbor_trial.test_harbor_trial import FakeRuntime

from nexus.runtime.executors.models.command import CommandRunStatus, RunCommandResponse


@pytest.mark.parametrize('compose', [False, True], ids=['direct', 'compose'])
@pytest.mark.parametrize('return_code', [0, 1, 127, -15])
async def test_exec_preserves_finished_exit_code_and_output(environment_factory, compose, return_code):
    create, _ = environment_factory
    environment = create(compose=compose, docker_image='example/task:v1')
    runtime = FakeRuntime()
    environment._runtime = runtime
    original = runtime.run_command

    async def run_command(**kwargs):
        if 'run-test-command' in kwargs['command']:
            return RunCommandResponse(
                status=CommandRunStatus.Finished,
                return_code=return_code,
                stdout='command output\n',
                stderr='command diagnostic\n',
            )
        return await original(**kwargs)

    runtime.run_command = AsyncMock(side_effect=run_command)
    result = await environment.exec('run-test-command', timeout_sec=120)

    assert result.return_code == return_code
    assert result.stdout == 'command output\n'
    assert result.stderr == 'command diagnostic\n'


@pytest.mark.parametrize('compose', [False, True], ids=['direct', 'compose'])
@pytest.mark.parametrize(
    ('status', 'return_code', 'error_type', 'message'),
    [
        (CommandRunStatus.TimeLimitExceeded, None, asyncio.TimeoutError, 'timed out after 120s'),
        (CommandRunStatus.TimeLimitExceeded, 0, asyncio.TimeoutError, 'timed out after 120s'),
        (CommandRunStatus.Error, None, RuntimeError, 'execution failed'),
        (CommandRunStatus.Error, 0, RuntimeError, 'execution failed'),
        (CommandRunStatus.Finished, None, RuntimeError, 'Invalid Nexus command result'),
    ],
)
async def test_exec_raises_for_runtime_failure_and_keeps_output(
    environment_factory, compose, status, return_code, error_type, message
):
    create, _ = environment_factory
    environment = create(compose=compose, docker_image='example/task:v1')
    runtime = FakeRuntime()
    environment._runtime = runtime
    original = runtime.run_command

    async def run_command(**kwargs):
        if 'run-test-command' in kwargs['command']:
            assert kwargs['timeout'] == 120
            return RunCommandResponse(
                status=status,
                return_code=return_code,
                stdout='partial output',
                stderr='runtime diagnostic',
            )
        return await original(**kwargs)

    runtime.run_command = AsyncMock(side_effect=run_command)
    with pytest.raises(error_type, match=message) as caught:
        await environment.exec('run-test-command', env={'TEST_VALUE': 'value'}, timeout_sec=120)

    assert 'run-test-command' in str(caught.value)
    assert 'stdout:\npartial output' in str(caught.value)
    assert 'stderr:\nruntime diagnostic' in str(caught.value)
    assert runtime.run_command.await_args.kwargs['command'].startswith('rm -rf /tmp/.harbor_exec_env_')


async def test_exec_normalizes_absent_output(environment_factory):
    create, _ = environment_factory
    environment = create(compose=False, docker_image='example/task:v1')
    runtime = FakeRuntime()
    environment._runtime = runtime
    runtime.run_command = AsyncMock(return_value=RunCommandResponse(status=CommandRunStatus.Finished, return_code=0))

    result = await environment.exec('true')

    assert result.stdout == result.stderr == ''
    assert result.return_code == 0
    runtime.run_command.assert_awaited_once_with(command='true', timeout=None)


@pytest.mark.parametrize('compose', [False, True], ids=['direct', 'compose'])
@pytest.mark.parametrize('timeout_kwargs', [{}, {'timeout_sec': None}, {'timeout_sec': 120}])
async def test_exec_preserves_optional_timeout(environment_factory, compose, timeout_kwargs):
    create, _ = environment_factory
    environment = create(compose=compose, docker_image='example/task:v1')
    runtime = FakeRuntime()
    environment._runtime = runtime
    runtime.run_command = AsyncMock(wraps=runtime.run_command)

    await environment.exec('run-test-command', **timeout_kwargs)

    command_calls = [c for c in runtime.run_command.await_args_list if 'run-test-command' in c.kwargs['command']]
    assert len(command_calls) == 1
    assert command_calls[0].kwargs['timeout'] == timeout_kwargs.get('timeout_sec')
