from __future__ import annotations

import asyncio

from types import SimpleNamespace
from unittest.mock import AsyncMock, call

import pytest

from harbor.environments.base import ExecResult
from tests.unit.harbor_trial.test_compose_build import environment_factory as environment_factory
from tests.unit.harbor_trial.test_harbor_trial import FakeRuntime

from nexus.runtime.executors.models.command import CommandRunStatus, RunCommandResponse


@pytest.mark.parametrize('probe', ['_wait_for_docker_daemon', '_wait_for_main_container'])
async def test_readiness_can_succeed_after_more_than_thirty_attempts(environment_factory, monkeypatch, probe):
    create, _ = environment_factory
    environment = create(docker_image='example/task:v1')
    dind = environment._dind
    execute = AsyncMock(side_effect=[ExecResult(return_code=1)] * 35 + [ExecResult(return_code=0)])
    sleep = AsyncMock()
    monkeypatch.setattr(environment, '_outer_exec', execute)
    monkeypatch.setattr('nexus.extensions.harbor_trial.environment.asyncio', SimpleNamespace(sleep=sleep))

    await getattr(dind, probe)()

    assert execute.await_count == 36
    assert sleep.await_args_list == [call(2)] * 35
    for invocation in execute.await_args_list:
        assert invocation.kwargs['timeout_sec'] == 10
        command = invocation.args[0]
        assert 'while ' not in command
        if probe == '_wait_for_docker_daemon':
            assert command == 'docker info'
            assert invocation.kwargs['env'] is None
        else:
            assert command.endswith('exec -T main true')
            assert invocation.kwargs['env'] == dind._compose_env_vars()


@pytest.mark.parametrize('probe', ['_wait_for_docker_daemon', '_wait_for_main_container'])
@pytest.mark.parametrize('timeout_error', [asyncio.TimeoutError, TimeoutError])
async def test_readiness_retries_single_probe_timeout(environment_factory, monkeypatch, probe, timeout_error):
    create, _ = environment_factory
    environment = create(docker_image='example/task:v1')
    execute = AsyncMock(side_effect=[timeout_error('probe timed out'), ExecResult(return_code=0)])
    sleep = AsyncMock()
    monkeypatch.setattr(environment, '_outer_exec', execute)
    monkeypatch.setattr(
        'nexus.extensions.harbor_trial.environment.asyncio',
        SimpleNamespace(sleep=sleep, TimeoutError=asyncio.TimeoutError),
    )

    await getattr(environment._dind, probe)()

    assert execute.await_count == 2
    sleep.assert_awaited_once_with(2)


@pytest.mark.parametrize('probe', ['_wait_for_docker_daemon', '_wait_for_main_container'])
@pytest.mark.parametrize('cancel_during', ['probe', 'sleep'])
async def test_readiness_propagates_outer_cancellation(environment_factory, monkeypatch, probe, cancel_during):
    create, _ = environment_factory
    environment = create(docker_image='example/task:v1')
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    async def block(*args, **kwargs):
        entered.set()
        try:
            await asyncio.Future()
        finally:
            cancelled.set()

    execute = (
        AsyncMock(side_effect=block) if cancel_during == 'probe' else AsyncMock(return_value=ExecResult(return_code=1))
    )
    sleep = AsyncMock(side_effect=block)
    monkeypatch.setattr(environment, '_outer_exec', execute)
    monkeypatch.setattr(
        'nexus.extensions.harbor_trial.environment.asyncio',
        SimpleNamespace(sleep=sleep, TimeoutError=asyncio.TimeoutError),
    )
    task = asyncio.create_task(getattr(environment._dind, probe)())
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert cancelled.is_set()
    assert execute.await_count == 1
    assert sleep.await_count == (1 if cancel_during == 'sleep' else 0)


@pytest.mark.parametrize('probe', ['_wait_for_docker_daemon', '_wait_for_main_container'])
async def test_readiness_does_not_swallow_execution_failure(environment_factory, monkeypatch, probe):
    create, _ = environment_factory
    environment = create(docker_image='example/task:v1')
    execute = AsyncMock(side_effect=RuntimeError('runtime unavailable'))
    monkeypatch.setattr(environment, '_outer_exec', execute)

    with pytest.raises(RuntimeError, match='runtime unavailable'):
        await getattr(environment._dind, probe)()

    assert execute.await_count == 1


async def test_build_and_up_reach_runtime_without_independent_timeouts(environment_factory, monkeypatch):
    create, provider = environment_factory
    environment = create(docker_image='example/task:v1')
    original = FakeRuntime.run_command
    timeouts = {}

    async def run_command(runtime, **kwargs):
        for stage in ('build', 'up -d'):
            if kwargs['command'].endswith(f' {stage}'):
                timeouts[stage] = kwargs['timeout']
        return await original(runtime, **kwargs)

    monkeypatch.setattr(FakeRuntime, 'run_command', run_command)
    await environment.start(force_build=False)

    assert timeouts == {'build': None, 'up -d': None}
    assert any(command.endswith('exec -T main true') for command in provider.runtimes['main'].commands)
    await environment.stop(delete=True)


@pytest.mark.parametrize('stage', ['build', 'up -d', 'exec -T main true'])
async def test_outer_startup_timeout_cancels_current_stage(environment_factory, monkeypatch, stage):
    create, provider = environment_factory
    environment = create(docker_image='example/task:v1')
    original = FakeRuntime.run_command
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    async def run_command(runtime, **kwargs):
        if kwargs['command'].endswith(f' {stage}'):
            entered.set()
            try:
                await asyncio.Future()
            finally:
                cancelled.set()
        return await original(runtime, **kwargs)

    monkeypatch.setattr(FakeRuntime, 'run_command', run_command)
    try:
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(environment.start(force_build=False), timeout=0.2)
        assert entered.is_set()
        assert cancelled.is_set()
        assert not any('while ' in command for command in provider.runtimes['main'].commands)
    finally:
        await environment.stop(delete=True)


async def test_startup_budget_is_shared_across_build_up_and_readiness(environment_factory, monkeypatch):
    create, _ = environment_factory
    environment = create(docker_image='example/task:v1')
    original = FakeRuntime.run_command
    stages = []

    async def run_command(runtime, **kwargs):
        for stage in ('build', 'up -d', 'exec -T main true'):
            if kwargs['command'].endswith(f' {stage}'):
                stages.append(stage)
                await asyncio.sleep(0.1)
        return await original(runtime, **kwargs)

    monkeypatch.setattr(FakeRuntime, 'run_command', run_command)
    try:
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(environment.start(force_build=False), timeout=0.25)
        assert stages == ['build', 'up -d', 'exec -T main true']
    finally:
        await environment.stop(delete=True)


async def test_compose_up_preserves_runtime_timeout_diagnostics(environment_factory, monkeypatch):
    create, provider = environment_factory
    environment = create(docker_image='example/task:v1')
    original = FakeRuntime.run_command

    async def run_command(runtime, **kwargs):
        if kwargs['command'].endswith(' up -d'):
            assert kwargs['timeout'] is None
            return RunCommandResponse(
                status=CommandRunStatus.TimeLimitExceeded,
                stdout='service starting',
                stderr='service not ready',
            )
        return await original(runtime, **kwargs)

    monkeypatch.setattr(FakeRuntime, 'run_command', run_command)
    try:
        with pytest.raises(asyncio.TimeoutError, match='Nexus command timed out') as caught:
            await environment.start(force_build=False)
        message = str(caught.value)
        assert ' up -d' in message
        assert 'service starting' in message
        assert 'service not ready' in message
        assert 'Nones' not in message
        assert not any('exec -T main true' in command for command in provider.runtimes['main'].commands)
    finally:
        await environment.stop(delete=True)
