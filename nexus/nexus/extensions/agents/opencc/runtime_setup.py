"""Runtime-side setup and execution functions for OpenCC.

Executed on the runtime side via ``runtime.invoke``. Responsible for:
  1. installing the OpenCC standalone binary,
  2. starting the Anthropic proxy on a free port,
  3. running any pre-start scripts,
  4. spawning ``opencc-bin --print`` and streaming stdout/stderr.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import shlex
import subprocess
import time

from pathlib import Path
from typing import Any

from nexus.extensions.agents.installed_common import (
    DEFAULT_PROXY_API_KEY,
    build_runtime_path,
    find_free_port,
    find_runtime_binary,
)


logger = logging.getLogger(__name__)


async def _pipe_to_logger(
    stream: asyncio.StreamReader,
    prefix: str,
    level: int,
    buffer: list[str],
) -> None:
    """Read lines from *stream*, log each line, and append to *buffer*."""
    while True:
        line = await stream.readline()
        if not line:
            break
        text = line.decode('utf-8', errors='replace').rstrip('\n')
        buffer.append(text)
        logger.log(level, '%s%s', prefix, text)


async def _run_opencc_streaming(
    cmd: list[str],
    cwd: str,
    env: dict[str, str],
    timeout: float,
) -> tuple[int, str, str]:
    """Spawn OpenCC with line-buffered stdout/stderr piped into the logger."""
    from nexus.utils.shell import get_os_env_without_ld_library_path

    base_env = get_os_env_without_ld_library_path()
    base_env['BASH_ENV'] = '~/.bashrc'
    merged_env = {**base_env, **env}

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd,
        env=merged_env,
    )

    assert proc.stdout is not None
    assert proc.stderr is not None

    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    stdout_task = asyncio.create_task(_pipe_to_logger(proc.stdout, '[opencc] ', logging.DEBUG, stdout_lines))
    stderr_task = asyncio.create_task(_pipe_to_logger(proc.stderr, '[opencc:err] ', logging.WARNING, stderr_lines))

    try:
        await asyncio.wait_for(
            asyncio.gather(stdout_task, stderr_task, return_exceptions=True),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        proc.kill()
        with contextlib.suppress(Exception):
            await proc.wait()
        stdout_task.cancel()
        stderr_task.cancel()
        return -1, '\n'.join(stdout_lines), '\n'.join(stderr_lines) + '\n[timeout]'

    try:
        exit_code = await asyncio.wait_for(proc.wait(), timeout=30)
    except asyncio.TimeoutError:
        proc.kill()
        with contextlib.suppress(Exception):
            exit_code = await proc.wait()

    return (exit_code or 0), '\n'.join(stdout_lines), '\n'.join(stderr_lines)


async def _run_pre_start_script(
    script: dict[str, Any],
    working_dir: str,
    base_env: dict[str, str],
) -> None:
    """Execute a single pre-start script."""
    from nexus.runtime.executors.command import CommandRunStatus, RunCommandRequest, run_command

    interpreter = script.get('interpreter', 'bash')
    code = script.get('code', '')
    timeout = float(script.get('timeout', 300.0))
    cwd = script.get('cwd') or working_dir
    fail_on_error = bool(script.get('fail_on_error', True))

    if interpreter == 'bash':
        cmd = f'bash -c {shlex.quote(code)}'
    elif interpreter == 'python':
        py = find_runtime_binary('python') or find_runtime_binary('python3') or 'python3'
        cmd = f'{shlex.quote(py)} -c {shlex.quote(code)}'
    else:
        raise ValueError(f'Unsupported pre-start interpreter: {interpreter}')

    script_env = dict(base_env)
    if script.get('env'):
        script_env.update(script['env'])

    logger.debug('Running pre-start (%s) in %s', interpreter, cwd)
    result = await run_command(RunCommandRequest(command=cmd, cwd=cwd, timeout=timeout, extra_env=script_env))
    if result.stdout:
        logger.debug('[pre-start stdout]\n%s', result.stdout)
    if result.stderr:
        logger.debug('[pre-start stderr]\n%s', result.stderr)

    ok = result.status == CommandRunStatus.Finished and (result.return_code or 0) == 0
    if not ok and fail_on_error:
        raise RuntimeError(
            f'Pre-start script failed (status={result.status}, rc={result.return_code}): '
            f'{result.stderr or result.stdout or ""}'
        )


async def setup_and_run_opencc(
    task: str,
    working_dir: str = '/workspace',
    timeout: float = 3600,
    remote_glibc_path: str = '',
    remote_musl_path: str = '',
    extra_env: dict[str, str] | None = None,
    extra_cli_args: list[str] | None = None,
    pre_start_scripts: list[dict[str, Any]] | None = None,
    conversion_log_level: str = 'INFO',
    thinking_preset: str = 'hunyuan',
) -> dict[str, Any]:
    """Set up and run OpenCC on the runtime side."""
    from nexus.extensions.agents.opencc.installer import (
        DEFAULT_INSTALL_DIR,
        install_opencc_binary,
        resolve_opencc_launch_argv,
    )
    from nexus.runtime.context import require_runtime_control
    from nexus.runtime.server.services.anthropic_proxy import AnthropicProxy

    rt = require_runtime_control()
    start_time = time.time()

    # 1) Install the correct binary variant (glibc or musl, auto-detected).
    try:
        binary = await install_opencc_binary(
            install_dir=DEFAULT_INSTALL_DIR,
            remote_glibc_path=remote_glibc_path,
            remote_musl_path=remote_musl_path,
        )
        logger.debug('OpenCC binary at: %s', binary)
    except Exception as e:
        logger.error('Failed to install OpenCC: %s', e)
        return {
            'success': False,
            'output': '',
            'stderr': f'Failed to install OpenCC: {e}',
            'exit_code': -1,
            'execution_time': time.time() - start_time,
        }

    # 2) Start the Anthropic proxy.
    proxy_port = find_free_port()
    logger.debug('Starting Anthropic Proxy on port %d...', proxy_port)
    proxy = AnthropicProxy(
        sio=rt.sio,
        listen_port=proxy_port,
        listen_host='127.0.0.1',
        request_timeout=timeout,
        conversion_log_level=conversion_log_level,
        thinking_config=thinking_preset,
    )

    try:
        await proxy.start()
        logger.debug('Anthropic Proxy started')
    except Exception as e:
        logger.error('Failed to start Anthropic Proxy: %s', e)
        return {
            'success': False,
            'output': '',
            'stderr': f'Failed to start Anthropic Proxy: {e}',
            'exit_code': -1,
            'execution_time': time.time() - start_time,
        }

    try:
        work_path = Path(working_dir)
        work_path.mkdir(parents=True, exist_ok=True)

        home = os.environ.get('HOME', '/root')
        opencc_env: dict[str, str] = {
            'ANTHROPIC_BASE_URL': f'http://127.0.0.1:{proxy_port}',
            'ANTHROPIC_API_KEY': DEFAULT_PROXY_API_KEY,
            'HOME': home,
            'PATH': build_runtime_path(DEFAULT_INSTALL_DIR),
            'IS_SANDBOX': '1',
        }
        if extra_env:
            opencc_env.update(extra_env)

        # 3) Pre-start scripts.
        for script in pre_start_scripts or []:
            try:
                await _run_pre_start_script(script, working_dir, opencc_env)
            except Exception as e:
                return {
                    'success': False,
                    'output': '',
                    'stderr': f'Pre-start script failed: {e}',
                    'exit_code': -1,
                    'execution_time': time.time() - start_time,
                }

        # 4) Build and run the OpenCC command.
        launch_argv = resolve_opencc_launch_argv(DEFAULT_INSTALL_DIR)
        cmd: list[str] = [*launch_argv, '--dangerously-skip-permissions']
        if extra_cli_args:
            cmd.extend(extra_cli_args)
        cmd.extend(['--', task])

        logger.debug('Running OpenCC: %s', ' '.join(shlex.quote(c) for c in cmd))

        exit_code, stdout, stderr = await _run_opencc_streaming(
            cmd=cmd,
            cwd=working_dir,
            env=opencc_env,
            timeout=timeout,
        )

        execution_time = time.time() - start_time
        logger.info('OpenCC finished with exit code %d in %.2fs', exit_code, execution_time)

        return {
            'success': exit_code == 0,
            'output': stdout,
            'stderr': stderr,
            'exit_code': exit_code,
            'execution_time': execution_time,
        }

    finally:
        logger.debug('Stopping Anthropic Proxy...')
        await proxy.stop()


__all__ = ['setup_and_run_opencc']
