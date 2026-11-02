import asyncio

from typing import Any

import pytest

from nexus.core.context import ExecutionContext
from nexus.extensions.openai_clients import hytuner
from nexus.extensions.openai_clients.hytuner import _wait_for_abort_resume
from nexus.utils.timeout import async_timeout


_ACTIVE_TIMEOUT_SECONDS = 0.5
_ABORT_PARK_SECONDS = 0.75


class _RecordingRuntime:
    default_venv_id = 'worker-venv'

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    async def invoke(self, target: str, /, *args: Any, **kwargs: Any) -> bool:
        self.calls.append((target, args, kwargs))
        return True

    async def drain_trace(self) -> None:
        pass


class _ControllableRuntime(_RecordingRuntime):
    def __init__(self, *, block_pause: bool = False, block_resume: bool = False) -> None:
        super().__init__()
        self.paused = False
        self.pause_started = asyncio.Event()
        self.resume_started = asyncio.Event()
        self.allow_pause = asyncio.Event()
        self.allow_resume = asyncio.Event()
        self.remote_tasks: list[asyncio.Task[bool]] = []
        if not block_pause:
            self.allow_pause.set()
        if not block_resume:
            self.allow_resume.set()

    async def invoke(self, target: str, /, *args: Any, **kwargs: Any) -> bool:
        self.calls.append((target, args, kwargs))

        async def remote_update() -> bool:
            if 'pause_command_timeout' in target:
                self.paused = True
                self.pause_started.set()
                await self.allow_pause.wait()
            else:
                self.resume_started.set()
                await self.allow_resume.wait()
                self.paused = False
            return True

        remote_task = asyncio.create_task(remote_update())
        self.remote_tasks.append(remote_task)
        return await asyncio.shield(remote_task)


async def _wait_for_pause_holders(context: ExecutionContext, expected: int) -> None:
    for _ in range(100):
        group = context.runtime_shared.get_optional(
            hytuner._ABORT_TIMEOUT_PAUSE_GROUP_KEY,
            hytuner._AbortTimeoutPauseGroup,
        )
        if group is not None and group._holder_count == expected:
            return
        await asyncio.sleep(0.005)
    raise AssertionError(f'abort timeout pause group did not reach {expected} holder(s)')


async def test_abort_wait_pauses_and_resumes_codex_runtime_timeout() -> None:
    runtime = _RecordingRuntime()
    context = ExecutionContext(runtime_command_timeout_id='codex-timeout')
    context._runtimes['main'] = runtime  # type: ignore[assignment]
    abort_event = asyncio.Event()

    async def resume_rollout() -> None:
        await asyncio.sleep(_ABORT_PARK_SECONDS)
        abort_event.set()

    resume_task = asyncio.create_task(resume_rollout())
    async with context:
        async with async_timeout(_ACTIVE_TIMEOUT_SECONDS) as timeout_handle:
            context.timeout_handle = timeout_handle
            await _wait_for_abort_resume(abort_event, 'request-1')
    await resume_task

    assert runtime.calls == [
        (
            'nexus.runtime.executors.command::pause_command_timeout',
            ('codex-timeout',),
            {'venv_id': 'worker-venv'},
        ),
        (
            'nexus.runtime.executors.command::resume_command_timeout',
            ('codex-timeout',),
            {'venv_id': 'worker-venv'},
        ),
    ]


async def test_cancellation_waits_for_pause_rpc_then_resumes_runtime_timeout() -> None:
    runtime = _ControllableRuntime(block_pause=True)
    context = ExecutionContext(runtime_command_timeout_id='codex-timeout')
    context._runtimes['main'] = runtime  # type: ignore[assignment]
    abort_event = asyncio.Event()

    async with context:
        wait_task = asyncio.create_task(_wait_for_abort_resume(abort_event, 'request-cancel-pause'))
        await asyncio.wait_for(runtime.pause_started.wait(), timeout=1.0)
        wait_task.cancel()
        await asyncio.sleep(0.01)
        assert wait_task.done() is False

        runtime.allow_pause.set()
        with pytest.raises(asyncio.CancelledError):
            await wait_task

    await asyncio.gather(*runtime.remote_tasks)
    assert runtime.paused is False
    assert runtime.resume_started.is_set()


async def test_cancellation_waits_for_resume_rpc_to_settle() -> None:
    runtime = _ControllableRuntime(block_resume=True)
    context = ExecutionContext(runtime_command_timeout_id='codex-timeout')
    context._runtimes['main'] = runtime  # type: ignore[assignment]
    abort_event = asyncio.Event()

    async with context:
        wait_task = asyncio.create_task(_wait_for_abort_resume(abort_event, 'request-cancel-resume'))
        await asyncio.wait_for(runtime.pause_started.wait(), timeout=1.0)
        abort_event.set()
        await asyncio.wait_for(runtime.resume_started.wait(), timeout=1.0)
        wait_task.cancel()
        await asyncio.sleep(0.01)
        assert wait_task.done() is False

        runtime.allow_resume.set()
        with pytest.raises(asyncio.CancelledError):
            await wait_task

    await asyncio.gather(*runtime.remote_tasks)
    assert runtime.paused is False


async def test_cancelled_first_waiter_does_not_resume_shared_timeouts() -> None:
    runtime = _RecordingRuntime()
    context = ExecutionContext(runtime_command_timeout_id='codex-timeout')
    context._runtimes['main'] = runtime  # type: ignore[assignment]
    abort_event = asyncio.Event()

    async with context:
        async with async_timeout(_ACTIVE_TIMEOUT_SECONDS) as timeout_handle:
            context.timeout_handle = timeout_handle
            first_waiter = asyncio.create_task(_wait_for_abort_resume(abort_event, 'request-1'))
            await _wait_for_pause_holders(context, 1)
            second_waiter = asyncio.create_task(_wait_for_abort_resume(abort_event, 'request-2'))
            await _wait_for_pause_holders(context, 2)

            first_waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first_waiter
            assert timeout_handle.when() is None

            await asyncio.sleep(_ABORT_PARK_SECONDS)
            abort_event.set()
            await second_waiter

    assert [target.rsplit('::', 1)[-1] for target, _, _ in runtime.calls] == [
        'pause_command_timeout',
        'resume_command_timeout',
    ]


async def test_cancelling_all_waiters_restores_shared_timeouts_once() -> None:
    runtime = _RecordingRuntime()
    context = ExecutionContext(runtime_command_timeout_id='codex-timeout')
    context._runtimes['main'] = runtime  # type: ignore[assignment]
    abort_event = asyncio.Event()

    async with context:
        async with async_timeout(1.0) as timeout_handle:
            context.timeout_handle = timeout_handle
            first_waiter = asyncio.create_task(_wait_for_abort_resume(abort_event, 'request-1'))
            second_waiter = asyncio.create_task(_wait_for_abort_resume(abort_event, 'request-2'))
            await _wait_for_pause_holders(context, 2)

            first_waiter.cancel()
            second_waiter.cancel()
            results = await asyncio.gather(first_waiter, second_waiter, return_exceptions=True)

            assert all(isinstance(result, asyncio.CancelledError) for result in results)
            assert timeout_handle.when() is not None

    assert [target.rsplit('::', 1)[-1] for target, _, _ in runtime.calls] == [
        'pause_command_timeout',
        'resume_command_timeout',
    ]


async def test_abort_wait_rejects_missing_codex_runtime_timeout() -> None:
    class _MissingTimeoutRuntime(_RecordingRuntime):
        async def invoke(self, target: str, /, *args: Any, **kwargs: Any) -> bool:
            await super().invoke(target, *args, **kwargs)
            return False

    runtime = _MissingTimeoutRuntime()
    context = ExecutionContext(runtime_command_timeout_id='missing-timeout')
    context._runtimes['main'] = runtime  # type: ignore[assignment]
    abort_event = asyncio.Event()

    async def resume_rollout() -> None:
        await asyncio.sleep(0.01)
        abort_event.set()

    resume_task = asyncio.create_task(resume_rollout())
    async with context:
        with pytest.raises(RuntimeError, match='missing-timeout'):
            await _wait_for_abort_resume(abort_event, 'request-2')
    await resume_task


async def test_running_gate_does_not_touch_runtime_or_task_timeout() -> None:
    runtime = _RecordingRuntime()
    context = ExecutionContext(runtime_command_timeout_id='codex-timeout')
    context._runtimes['main'] = runtime  # type: ignore[assignment]
    abort_event = asyncio.Event()
    abort_event.set()

    async with context:
        async with async_timeout(1.0) as timeout_handle:
            context.timeout_handle = timeout_handle
            original_deadline = timeout_handle.when()
            assert await _wait_for_abort_resume(abort_event, 'running-request') == 0.0
            assert timeout_handle.when() == original_deadline
    assert runtime.calls == []


async def test_abort_gate_pauses_the_real_command_and_task_deadlines(monkeypatch):
    from nexus.runtime.executors import command as command_executor
    from nexus.runtime.executors.models.command import CommandRunStatus, RunCommandRequest

    class LocalCommandRuntime(_RecordingRuntime):
        async def invoke(self, target, identifier, *, venv_id):
            assert venv_id == self.default_venv_id
            return await getattr(command_executor, target.rsplit('::', 1)[-1])(identifier)

    monkeypatch.setattr(command_executor, 'EXECUTABLE_BASH', '/bin/sh')
    process_created = asyncio.Event()
    original_spawn = asyncio.create_subprocess_shell

    async def spawn(*args, **kwargs):
        process = await original_spawn(*args, **kwargs)
        process_created.set()
        return process

    monkeypatch.setattr(command_executor.asyncio, 'create_subprocess_shell', spawn)
    context = ExecutionContext(runtime_command_timeout_id='real-command')
    context._runtimes['main'] = LocalCommandRuntime()
    gate = asyncio.Event()
    command_task = asyncio.create_task(
        command_executor.run_command(
            RunCommandRequest(command='exec sleep 0.5', timeout=0.25, timeout_control_id='real-command')
        )
    )
    resume = None
    try:
        await asyncio.wait_for(process_created.wait(), timeout=1.0)
        resume = asyncio.get_running_loop().call_later(0.4, gate.set)
        async with context:
            async with async_timeout(0.25) as handle:
                context.timeout_handle = handle
                await _wait_for_abort_resume(gate, 'real-command-request')
                response = await command_task
        assert response.status == CommandRunStatus.Finished
        assert await command_executor.pause_command_timeout('real-command') is False
    finally:
        if resume is not None:
            resume.cancel()
        gate.set()
        await asyncio.wait_for(command_task, timeout=1.0)
