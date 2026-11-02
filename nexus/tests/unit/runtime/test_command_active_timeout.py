import asyncio
import sys

from collections.abc import Awaitable, Callable

import pytest

from nexus.extensions.agents.installed_common import build_runtime_command
from nexus.runtime.executors import command as command_executor
from nexus.runtime.executors.models.command import CommandRunStatus, RunCommandRequest


@pytest.fixture(autouse=True)
def _shell_start_up_is_not_charged_to_the_deadline(isolated_shell_home) -> None:
    """Every command below runs under a sub-second deadline.

    Without an empty ``HOME`` the spawned bash sources the developer's ``~/.bashrc``, and
    that start-up cost is charged against the command deadline under test.
    """


class _Python310TimeoutHandle:
    def __init__(self, deadline: float) -> None:
        self.deadline = deadline
        self.rejected = False
        self.updated_deadline: float | None = None

    def reject(self) -> None:
        self.rejected = True

    def update(self, deadline: float) -> None:
        self.updated_deadline = deadline


async def _pause_registered_timeout(
    pause_timeout: Callable[[str], Awaitable[bool]],
    timeout_control_id: str,
) -> None:
    for _ in range(100):
        if await pause_timeout(timeout_control_id):
            return
        await asyncio.sleep(0.005)
    raise AssertionError(f'Command timeout {timeout_control_id!r} was not registered')


async def test_command_timeout_supports_python_310_async_timeout_handle() -> None:
    handle = _Python310TimeoutHandle(deadline=10.0)
    command_timeout = command_executor._CommandTimeout(handle=handle, deadline=handle.deadline)

    command_timeout.pause()
    assert handle.rejected is True
    assert command_timeout.pause_started_at is not None
    command_timeout.pause_started_at -= 0.25
    command_timeout.resume()

    # The resume path adds the real wall time spent between pause() and resume(), so the
    # comparison needs a tolerance larger than the loop clock drift of those two calls.
    assert handle.updated_deadline == pytest.approx(10.25, abs=0.05)


async def test_duplicate_timeout_control_id_does_not_spawn_second_process(monkeypatch: pytest.MonkeyPatch) -> None:
    original_create_command_process = command_executor._create_command_process
    spawned_commands: list[str] = []

    async def create_command_process(command: str, *args, **kwargs):
        spawned_commands.append(command)
        return await original_create_command_process(command, *args, **kwargs)

    monkeypatch.setattr(command_executor, '_create_command_process', create_command_process)
    timeout_control_id = 'duplicate-timeout-control'
    command = build_runtime_command([sys.executable, '-c', 'import time; time.sleep(0.2)'])
    first_task = asyncio.create_task(
        command_executor.run_command(
            RunCommandRequest(command=command, timeout=5.0, timeout_control_id=timeout_control_id)
        )
    )

    for _ in range(100):
        if timeout_control_id in command_executor._COMMAND_TIMEOUTS:
            break
        await asyncio.sleep(0.005)
    else:
        raise AssertionError(f'Command timeout {timeout_control_id!r} was not registered')

    duplicate_response = await command_executor.run_command(
        RunCommandRequest(command='echo duplicate', timeout=1.0, timeout_control_id=timeout_control_id)
    )
    first_response = await first_task

    assert duplicate_response.status == CommandRunStatus.Error
    assert first_response.status == CommandRunStatus.Finished
    assert spawned_commands == [command]


async def test_command_timeout_excludes_paused_wall_time(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    pause_timeout = getattr(command_executor, 'pause_command_timeout', None)
    resume_timeout = getattr(command_executor, 'resume_command_timeout', None)
    assert callable(pause_timeout), 'run_command must expose pause_command_timeout'
    assert callable(resume_timeout), 'run_command must expose resume_command_timeout'

    timeout_control_id = 'pause-excludes-wall-time'
    release_file = tmp_path / 'release'
    command_task = asyncio.create_task(
        command_executor.run_command(
            RunCommandRequest(
                command=build_runtime_command([
                    sys.executable,
                    '-c',
                    f'import pathlib, time\nwhile not pathlib.Path({str(release_file)!r}).exists(): time.sleep(0.01)\n'
                    "print('completed')",
                ]),
                timeout=0.3,
                timeout_control_id=timeout_control_id,
            )
        )
    )

    try:
        await _pause_registered_timeout(pause_timeout, timeout_control_id)
        await asyncio.sleep(0.45)
        assert not command_task.done(), 'the wall deadline must not fire while paused'
        release_file.touch()
        assert await resume_timeout(timeout_control_id) is True
        response = await asyncio.wait_for(command_task, timeout=1.0)
    finally:
        release_file.touch()
        await asyncio.wait_for(command_task, timeout=1.0)

    assert response.status == CommandRunStatus.Finished
    assert response.return_code == 0
    assert (response.stdout or '').strip() == 'completed'


async def test_command_timeout_still_expires_after_active_budget_is_used(monkeypatch: pytest.MonkeyPatch) -> None:
    pause_timeout = getattr(command_executor, 'pause_command_timeout', None)
    resume_timeout = getattr(command_executor, 'resume_command_timeout', None)
    assert callable(pause_timeout), 'run_command must expose pause_command_timeout'
    assert callable(resume_timeout), 'run_command must expose resume_command_timeout'

    timeout_control_id = 'active-budget-still-expires'
    command_task = asyncio.create_task(
        command_executor.run_command(
            RunCommandRequest(
                command=build_runtime_command([sys.executable, '-c', 'import time; time.sleep(0.8)']),
                timeout=0.12,
                timeout_control_id=timeout_control_id,
            )
        )
    )

    await _pause_registered_timeout(pause_timeout, timeout_control_id)
    await asyncio.sleep(0.15)
    assert await resume_timeout(timeout_control_id) is True
    response = await asyncio.wait_for(command_task, timeout=1.0)

    assert response.status == CommandRunStatus.TimeLimitExceeded
