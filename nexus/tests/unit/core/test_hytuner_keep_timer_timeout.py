"""Configured sandbox timer RPC timeouts bound control calls without changing unconfigured callers."""

import asyncio
import logging

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nexus.core.context import ExecutionContext
from nexus.extensions.openai_clients.hytuner import capture_hytuner_execution_timeouts
from nexus.runtime.client.remote import RemoteRuntime, http_client
from nexus.runtime.models import InvokeJobInfo, JobStatus, pack_result
from nexus.utils import timeout as timeout_utils
from nexus.utils.async_http import HTTPResponse
from nexus.utils.timeout import async_timeout, with_timeout


def _hanging_runtime(*hanging_actions: str) -> tuple[SimpleNamespace, list[str]]:
    calls: list[str] = []

    async def invoke(target: str, *args: object, **kwargs: object) -> bool:
        action = target.rsplit('::', 1)[-1]
        calls.append(action)
        if action in hanging_actions:
            await asyncio.Event().wait()
        return True

    return SimpleNamespace(default_venv_id='worker', invoke=invoke, drain_trace=AsyncMock()), calls


def _configured_context(runtime: object, timeout_seconds: float) -> ExecutionContext:
    context = ExecutionContext(
        runtime_command_timeout_id='command',
        runtime_configs={'main': {'sandbox_timer_rpc_timeout_s': timeout_seconds}},
    )
    context._runtimes['main'] = runtime
    return context


async def test_configured_pause_timeout_compensates_restores_budgets_and_raises() -> None:
    runtime, calls = _hanging_runtime('pause_command_timeout')
    context = _configured_context(runtime, 0.02)
    async with context:
        async with async_timeout(1.0) as runner:
            context.timeout_handle = runner
            async with with_timeout(1.0):
                inference = timeout_utils._ACTIVE_TIMEOUT_SCOPES.get()[-1]
                controller = capture_hytuner_execution_timeouts('request', track_shared_pause=True)
                assert controller is not None, 'Keep generation must capture its execution budgets'
                with pytest.raises(TimeoutError, match='action=pause'):
                    await controller.pause()
                assert runner.when() is not None, 'A timed-out pause must restore the runner deadline'
                assert inference.handle.when() is not None, 'A timed-out pause must restore inference budgets'
                assert inference.holders == 0, 'A timed-out pause must not leak inference leases'
    assert calls == ['pause_command_timeout', 'resume_command_timeout'], (
        'A timed-out pause must send exactly one compensating resume'
    )


async def test_configured_resume_timeout_restores_budgets_and_raises() -> None:
    runtime, calls = _hanging_runtime('resume_command_timeout')
    context = _configured_context(runtime, 0.02)
    async with context:
        async with async_timeout(1.0) as runner:
            context.timeout_handle = runner
            async with with_timeout(1.0):
                inference = timeout_utils._ACTIVE_TIMEOUT_SCOPES.get()[-1]
                controller = capture_hytuner_execution_timeouts('request', track_shared_pause=True)
                assert controller is not None, 'Keep generation must capture its execution budgets'
                await controller.pause()
                assert runner.when() is None, 'A successful pause must suspend the runner deadline'
                with pytest.raises(TimeoutError, match='action=resume'):
                    await controller.resume()
                assert runner.when() is not None, 'A timed-out resume must restore the runner deadline'
                assert inference.handle.when() is not None, 'A timed-out resume must restore inference budgets'
    assert calls == ['pause_command_timeout', 'resume_command_timeout'], 'A timed-out resume must not be retried'


async def test_compensating_resume_shares_the_timer_rpc_timeout() -> None:
    runtime, calls = _hanging_runtime('pause_command_timeout', 'resume_command_timeout')
    context = _configured_context(runtime, 0.02)
    async with context:
        async with async_timeout(1.0) as runner:
            context.timeout_handle = runner
            controller = capture_hytuner_execution_timeouts('request', track_shared_pause=True)
            assert controller is not None, 'Keep generation must capture its execution budget'
            started_at = asyncio.get_running_loop().time()
            with pytest.raises(TimeoutError):
                await controller.pause()
            elapsed = asyncio.get_running_loop().time() - started_at
            assert elapsed < 0.2, 'A pause and its compensation must each stop at the timer RPC timeout'
            assert runner.when() is not None, 'A failed compensation must still restore the runner deadline'
    assert calls == ['pause_command_timeout', 'resume_command_timeout'], 'Compensation must be attempted once'


@pytest.mark.parametrize('configured', [True, False], ids=['keep-compensation-timeout', 'abort-compensation-error'])
async def test_failed_compensation_keeps_the_original_pause_error(
    caplog: pytest.LogCaptureFixture, configured: bool
) -> None:
    pause_failure = ConnectionError('pause failed')

    async def invoke(target: str, *args: object, **kwargs: object) -> bool:
        if target.endswith('::pause_command_timeout'):
            raise pause_failure
        if configured:
            await asyncio.Event().wait()
        raise ConnectionError('compensation failed')

    runtime = SimpleNamespace(default_venv_id='worker', invoke=invoke, drain_trace=AsyncMock())
    if configured:
        context = _configured_context(runtime, 0.02)
    else:
        context = ExecutionContext(runtime_command_timeout_id='command')
        context._runtimes['main'] = runtime
    async with context:
        async with async_timeout(1.0) as runner:
            context.timeout_handle = runner
            controller = capture_hytuner_execution_timeouts('request', track_shared_pause=configured)
            assert controller is not None, 'A live Nexus context must provide an execution budget'
            with caplog.at_level(logging.WARNING, logger='nexus.extensions.openai_clients.hytuner'):
                with pytest.raises(ConnectionError) as caught:
                    await controller.pause()
            assert caught.value is pause_failure, 'A failed compensation must not replace the pause failure'
            assert runner.when() is not None, 'A failed compensation must still restore the runner deadline'
    assert any('Timer pause compensation failed' in message for message in caplog.messages), (
        'A failed compensation must be logged instead of silently dropped'
    )


async def test_failed_compensation_keeps_the_original_pause_timeout() -> None:
    async def invoke(target: str, *args: object, **kwargs: object) -> bool:
        if target.endswith('::pause_command_timeout'):
            await asyncio.Event().wait()
        raise ConnectionError('compensation failed')

    runtime = SimpleNamespace(default_venv_id='worker', invoke=invoke, drain_trace=AsyncMock())
    context = _configured_context(runtime, 0.02)
    async with context:
        async with async_timeout(1.0) as runner:
            context.timeout_handle = runner
            controller = capture_hytuner_execution_timeouts('request', track_shared_pause=True)
            assert controller is not None, 'Keep generation must capture its execution budget'
            with pytest.raises(TimeoutError, match='action=pause'):
                await controller.pause()
            assert runner.when() is not None, 'A failed compensation must still restore the runner deadline'


@pytest.mark.parametrize('async_result', [False, True])
async def test_timer_rpc_timeout_covers_post_and_result_wait(
    monkeypatch: pytest.MonkeyPatch, async_result: bool
) -> None:
    result_wait_cancelled = asyncio.Event()
    release_pause_post = asyncio.Event()
    runtime = RemoteRuntime(base_url='http://sandbox:8000', auto_connect_ws=False)
    runtime.default_venv_id = 'worker'
    runtime.config.poll_interval = 0.001
    monkeypatch.setattr(runtime, 'drain_trace', AsyncMock())

    async def post(url: str, **kwargs: object) -> HTTPResponse:
        request = kwargs['json']
        assert request['request_timeout'] == 0.05, 'Timer controls must send the configured RPC timeout'
        assert kwargs['timeout'].total == 0.05, 'The HTTP transport must use the configured RPC timeout'
        if request['import_path'].endswith('::pause_command_timeout'):
            if async_result:
                await asyncio.sleep(0.03)
                payload = InvokeJobInfo(request_id=request['request_id'], status=JobStatus.RUNNING).model_dump_json()
                return HTTPResponse(202, {}, payload, payload.encode(), url, 'utf-8')
            await release_pause_post.wait()
        payload = pack_result(True, request_id=request['request_id']).model_dump_json()
        return HTTPResponse(200, {}, payload, payload.encode(), url, 'utf-8')

    async def get(url: str, **kwargs: object) -> HTTPResponse:
        try:
            await asyncio.Event().wait()
        finally:
            result_wait_cancelled.set()

    monkeypatch.setattr(http_client, 'post', post)
    monkeypatch.setattr(http_client, 'get', get)
    context = _configured_context(runtime, 0.05)
    async with context:
        controller = capture_hytuner_execution_timeouts('request', track_shared_pause=True)
        assert controller is not None, 'Keep generation must capture its command budget'
        started_at = asyncio.get_running_loop().time()
        with pytest.raises(TimeoutError, match='action=pause'):
            await controller.pause()
        elapsed = asyncio.get_running_loop().time() - started_at
        assert 0.04 <= elapsed < 0.2, 'POST and result waiting must share the same total RPC budget'
        if async_result:
            assert result_wait_cancelled.is_set(), 'A timed-out RPC must stop waiting for its job result'
    release_pause_post.set()
    await asyncio.sleep(0)


async def test_unconfigured_timer_rpc_keeps_its_original_failure() -> None:
    failure = TimeoutError('timer transport timeout')

    async def invoke(target: str, *args: object, **kwargs: object) -> bool:
        assert 'request_timeout' not in kwargs, 'Unconfigured timer RPCs must keep the runtime request timeout'
        if target.endswith('::pause_command_timeout'):
            raise failure
        return True

    runtime = SimpleNamespace(default_venv_id='worker', invoke=invoke, drain_trace=AsyncMock())
    context = ExecutionContext(runtime_command_timeout_id='command')
    context._runtimes['main'] = runtime
    async with context:
        async with async_timeout(1.0) as runner:
            context.timeout_handle = runner
            controller = capture_hytuner_execution_timeouts('request', track_shared_pause=True)
            assert controller is not None, 'A live Nexus context must provide an execution budget'
            with pytest.raises(TimeoutError) as caught:
                await controller.pause()
            assert caught.value is failure, 'Unconfigured timer RPCs must propagate their original exception'
            assert runner.when() is not None, 'A failed pause must restore the runner deadline'
