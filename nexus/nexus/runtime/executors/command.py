# /// script
# dependencies = []
#
# [project]
# name = "command"
# authors = [
#     { name = "tarschen" },
# ]
#
# [tool.nexus]
# executors = { run_command = "nexus.runtime.executors.command" }
# ///

import asyncio
import base64
import contextlib
import logging
import os
import shlex
import subprocess
import time
import traceback

from asyncio import StreamReader
from dataclasses import dataclass
from typing import Any

from nexus.core import executor_registry
from nexus.runtime.context import get_execution_user
from nexus.runtime.executors.models.command import *  # noqa: F401,F403
from nexus.runtime.executors.models.command import CommandRunStatus, RunCommandRequest, RunCommandResponse
from nexus.runtime.server.platform import subprocess_creation_flags
from nexus.utils.platform import is_windows
from nexus.utils.shell import (
    get_default_pwd,
    get_os_env_without_ld_library_path,
    kill_all_child_processes,
    resolve_first_existing_path,
)
from nexus.utils.timeout import async_timeout


logger = logging.getLogger(__name__)


# ==================== Implementation ====================


EXECUTABLE_BASH = (
    'bash' if is_windows() else resolve_first_existing_path(['/nix/runtime/nix-env/bin/bash', '/bin/bash', 'bash'])
)

EXECUTABLE_SHELL = resolve_first_existing_path(['powershell.exe', 'powershell']) if is_windows() else EXECUTABLE_BASH


def _powershell_args(command: str) -> list[str]:
    prelude = (
        "$ProgressPreference = 'SilentlyContinue'; "
        '$utf8 = [System.Text.UTF8Encoding]::new($false); '
        '[Console]::InputEncoding = $utf8; '
        '[Console]::OutputEncoding = $utf8; $OutputEncoding = $utf8; '
    )
    encoded = base64.b64encode(f'{prelude}{command}'.encode('utf-16-le')).decode('ascii')
    return [
        '-NoLogo',
        '-NoProfile',
        '-NonInteractive',
        '-ExecutionPolicy',
        'Bypass',
        '-EncodedCommand',
        encoded,
    ]


async def _create_command_process(
    command: str,
    *,
    cwd: str,
    env: dict[str, str],
) -> asyncio.subprocess.Process:
    if is_windows():
        return await asyncio.create_subprocess_exec(
            EXECUTABLE_SHELL,
            *_powershell_args(command),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            env=env,
            creationflags=subprocess_creation_flags(),
        )
    return await asyncio.create_subprocess_shell(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd,
        env=env,
        executable=EXECUTABLE_BASH,
    )


async def _terminate_command_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return

    if is_windows():
        await kill_all_child_processes(process.pid)
        with contextlib.suppress(ProcessLookupError):
            process.terminate()
    else:
        import signal

        with contextlib.suppress(ProcessLookupError):
            process.send_signal(signal.SIGTERM)

    try:
        await asyncio.wait_for(process.wait(), timeout=5.0)
    except _TIMEOUT_ERRORS:
        with contextlib.suppress(ProcessLookupError):
            process.kill()
        try:
            await asyncio.wait_for(process.wait(), timeout=3.0)
        except _TIMEOUT_ERRORS:
            logger.warning('Process %s did not exit after force kill', process.pid)


# COMPAT: asyncio.TimeoutError only became an alias of builtins.TimeoutError in Python 3.11.
_TIMEOUT_ERRORS = (asyncio.TimeoutError, TimeoutError)


def _get_timeout_deadline(handle: Any) -> float | None:
    """Return the absolute deadline across supported timeout implementations."""
    when = getattr(handle, 'when', None)
    if callable(when):
        return when()
    return handle.deadline


def _reschedule_timeout(handle: Any, deadline: float | None) -> None:
    """Reschedule or disarm a timeout across supported Python versions."""
    reschedule = getattr(handle, 'reschedule', None)
    if callable(reschedule):
        reschedule(deadline)
        return

    # COMPAT: async-timeout 4 on Python 3.10 predates asyncio.Timeout's reschedule API.
    if deadline is None:
        handle.reject()
    else:
        handle.update(deadline)


@dataclass
class _CommandTimeout:
    handle: Any
    deadline: float
    pause_started_at: float | None = None

    def pause(self) -> None:
        if self.pause_started_at is not None:
            return
        self.pause_started_at = asyncio.get_running_loop().time()
        _reschedule_timeout(self.handle, None)

    def resume(self) -> None:
        if self.pause_started_at is None:
            return
        self.deadline += asyncio.get_running_loop().time() - self.pause_started_at
        self.pause_started_at = None
        _reschedule_timeout(self.handle, self.deadline)


_COMMAND_TIMEOUTS: dict[str, _CommandTimeout | None] = {}


@contextlib.contextmanager
def registered_command_timeout(timeout_control_id: str | None, timeout_handle: Any):
    """Expose a live subprocess timeout to the gateway pause/resume controls."""
    if timeout_control_id is None:
        yield
        return
    if timeout_control_id in _COMMAND_TIMEOUTS:
        raise RuntimeError(f'Command timeout control ID already registered: {timeout_control_id}')
    deadline = _get_timeout_deadline(timeout_handle)
    if deadline is None:
        raise RuntimeError(f'Command timeout has no deadline: {timeout_control_id}')
    _COMMAND_TIMEOUTS[timeout_control_id] = _CommandTimeout(timeout_handle, deadline)
    try:
        yield
    finally:
        _COMMAND_TIMEOUTS.pop(timeout_control_id, None)


@contextlib.asynccontextmanager
async def active_command_timeout(timeout: float, timeout_control_id: str | None = None):
    """Apply a deadline controllable through the command pause/resume RPCs."""
    async with async_timeout(timeout) as handle:
        with registered_command_timeout(timeout_control_id, handle):
            yield


async def pause_command_timeout(timeout_control_id: str) -> bool:
    """Pause a registered command timeout while preserving its remaining budget."""
    command_timeout = _COMMAND_TIMEOUTS.get(timeout_control_id)
    if command_timeout is None:
        return False
    command_timeout.pause()
    return True


async def resume_command_timeout(timeout_control_id: str) -> bool:
    """Resume a registered command timeout after extending it by the paused duration."""
    command_timeout = _COMMAND_TIMEOUTS.get(timeout_control_id)
    if command_timeout is None:
        return False
    command_timeout.resume()
    return True


def _command_for_execution_user(command: str, execution_user: str | int | None) -> str:
    if execution_user is None:
        return command

    if is_windows():
        raise NotImplementedError('execution_user is not supported on Windows')

    import pwd

    if isinstance(execution_user, int):
        if execution_user < 0:
            raise ValueError(f'Invalid execution UID: {execution_user}')
        username = pwd.getpwuid(execution_user).pw_name  # ty: ignore[unresolved-attribute] -- guarded Unix-only API
    else:
        username = execution_user
    current_username = pwd.getpwuid(os.geteuid()).pw_name  # ty: ignore[unresolved-attribute] -- guarded Unix-only APIs
    if username in {'root', current_username}:
        return command
    return f'su {shlex.quote(username)} -s {shlex.quote(EXECUTABLE_BASH)} -c {shlex.quote(command)}'


@executor_registry.register('run_command')
async def run_command(args: RunCommandRequest) -> RunCommandResponse:
    """Execute a shell command. Importable standalone function for backward compat."""
    timeout_control_id = args.timeout_control_id
    owns_timeout_registration = False
    try:
        logger.debug(f'running command {args.command}')
        if timeout_control_id is not None:
            if timeout_control_id in _COMMAND_TIMEOUTS:
                raise RuntimeError(f'Command timeout control ID already registered: {timeout_control_id}')
            _COMMAND_TIMEOUTS[timeout_control_id] = None
            owns_timeout_registration = True

        os_envs = get_os_env_without_ld_library_path()

        if not is_windows():
            # simulate login shell to load user envs
            os_envs['BASH_ENV'] = '~/.bashrc'

        cwd = args.cwd or get_default_pwd()
        command = _command_for_execution_user(args.command, get_execution_user())

        p = await _create_command_process(
            command,
            cwd=cwd,
            env={**os_envs, **(args.extra_env or {})},
        )

        # stdin/stdout/stderr are guaranteed to be non-None because we passed subprocess.PIPE
        assert p.stdin is not None
        assert p.stdout is not None
        assert p.stderr is not None

        if args.stdin is not None:
            p.stdin.write(args.stdin.encode())
        p.stdin.close()

        start_time = time.time()
        stdout_reader = await get_output_reader(p.stdout)
        stderr_reader = await get_output_reader(p.stderr)

        try:
            async with async_timeout(args.timeout) as timeout_handle:
                if timeout_control_id is not None:
                    deadline = _get_timeout_deadline(timeout_handle)
                    if deadline is not None:
                        _COMMAND_TIMEOUTS[timeout_control_id] = _CommandTimeout(timeout_handle, deadline)
                await p.wait()
            execution_time = time.time() - start_time
            logger.debug(f'stop running command {args.command}')
        except _TIMEOUT_ERRORS:
            await _terminate_command_process(p)
            return RunCommandResponse(
                status=CommandRunStatus.TimeLimitExceeded,
                execution_time=time.time() - start_time,
                stdout=await stdout_reader(),
                stderr=await stderr_reader(),
            )
        return RunCommandResponse(
            status=CommandRunStatus.Finished,
            execution_time=execution_time,
            return_code=p.returncode,
            stdout=await stdout_reader(),
            stderr=await stderr_reader(),
        )
    except Exception as e:
        tb = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
        message = f'exception on running command {args.command}: {type(e).__name__}: {e}\n{tb}'
        logger.warning(message)
        return RunCommandResponse(status=CommandRunStatus.Error, stderr=message)
    finally:
        if owns_timeout_registration and timeout_control_id is not None:
            _COMMAND_TIMEOUTS.pop(timeout_control_id, None)


# ==================== Utils for stdio processing ====================


def try_decode(s: bytes) -> str:
    try:
        r = s.decode(encoding='utf-8', errors='replace')
    except Exception as e:
        r = f'[DecodeError] {e}'
    return r


async def get_output_reader(fd: StreamReader, max_out_bytes: int | None = None):
    res = b''

    async def reader():
        nonlocal res
        while True:
            chunk = await fd.read(1024 * 1024)
            if not chunk:
                break
            if not max_out_bytes or len(res) <= max_out_bytes:
                res += chunk
            del chunk

    task = asyncio.create_task(reader())

    async def read():
        try:
            await asyncio.wait_for(task, timeout=0.001)
        except _TIMEOUT_ERRORS:
            task.cancel()
        return try_decode(res)

    return read
