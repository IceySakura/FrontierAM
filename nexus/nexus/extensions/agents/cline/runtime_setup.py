"""Runtime-side setup and execution for Cline CLI.

Executed on the runtime side via ``runtime.invoke``. Responsible for:
  1. Installing the Cline CLI (npm),
  2. Starting an OpenAI-compatible proxy on a free port,
  3. Configuring Cline via ``cline auth`` to route LLM calls through the proxy,
  4. Running pre-start scripts,
  5. Spawning ``cline -y`` in headless mode and streaming stdout/stderr.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import shlex
import subprocess
import tempfile
import time

from pathlib import Path
from typing import Any

from nexus.extensions.agents.installed_common import (
    DEFAULT_AGENT_INSTALL_ROOT,
    DEFAULT_NPM_REGISTRY,
    DEFAULT_PROXY_API_KEY,
    build_runtime_path,
    find_free_port,
    find_runtime_binary,
)


logger = logging.getLogger(__name__)


DEFAULT_INSTALL_DIR = f'{DEFAULT_AGENT_INSTALL_ROOT}/cline'
DEFAULT_MODEL_ID = 'sonnet'

NOOP_MAX_ATTEMPTS = 3
NOOP_FAILURE_STAGE = 'noop_exit'


def _is_silent_noop(exit_code: int, stdout: str) -> bool:
    """Report whether Cline claimed success without doing anything.

    Cline sometimes exits 0 a couple of seconds after launch having written no
    stdout and without ever reaching the LLM proxy. The same prompt succeeds on
    a rerun, so the trigger is not in the input and cannot be screened for; the
    shape of the result is the only reliable signal. Left unchecked this lands
    in the dataset as a legitimate run that solved nothing.
    """
    return exit_code == 0 and not stdout.strip()


async def _run_pre_start_script(
    script: dict[str, Any],
    working_dir: str,
    base_env: dict[str, str],
) -> None:
    """Execute a single pre-start script. Raises if fail_on_error and exit != 0."""
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

    logger.debug(f'Running pre-start ({interpreter}) in {cwd}')
    result = await run_command(RunCommandRequest(command=cmd, cwd=cwd, timeout=timeout, extra_env=script_env))
    if result.stdout:
        logger.debug(f'[pre-start stdout]\n{result.stdout}')
    if result.stderr:
        logger.debug(f'[pre-start stderr]\n{result.stderr}')

    ok = result.status == CommandRunStatus.Finished and (result.return_code or 0) == 0
    if not ok and fail_on_error:
        raise RuntimeError(
            f'Pre-start script failed (status={result.status}, rc={result.return_code}): '
            f'{result.stderr or result.stdout or ""}'
        )


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


async def _run_streaming(
    cmd: list[str],
    cwd: str,
    env: dict[str, str],
    timeout: float,
    log_prefix: str = '[cline]',
) -> tuple[int, str, str]:
    """Spawn a command with line-buffered stdout/stderr piped into the logger."""
    from nexus.utils.shell import get_os_env_without_ld_library_path

    base_env = get_os_env_without_ld_library_path()
    base_env['BASH_ENV'] = '~/.bashrc'
    merged_env = {**base_env, **env}

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd,
        env=merged_env,
    )

    assert proc.stdout is not None
    assert proc.stderr is not None

    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    stdout_task = asyncio.create_task(_pipe_to_logger(proc.stdout, f'{log_prefix} ', logging.DEBUG, stdout_lines))
    stderr_task = asyncio.create_task(_pipe_to_logger(proc.stderr, f'{log_prefix}:err ', logging.WARNING, stderr_lines))

    try:
        exit_code = await asyncio.wait_for(proc.wait(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        with contextlib.suppress(Exception):
            await proc.wait()
        stdout_task.cancel()
        stderr_task.cancel()
        return -1, '\n'.join(stdout_lines), '\n'.join(stderr_lines) + '\n[timeout]'

    await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
    return exit_code or 0, '\n'.join(stdout_lines), '\n'.join(stderr_lines)


async def setup_and_run_cline(
    task: str,
    working_dir: str = '/workspace',
    timeout: float = 3600,
    version: str = 'latest',
    npm_registry: str | None = DEFAULT_NPM_REGISTRY,
    model_id: str = DEFAULT_MODEL_ID,
    thinking_budget_tokens: int = 0,
    extra_env: dict[str, str] | None = None,
    extra_cli_args: list[str] | None = None,
    pre_start_scripts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Set up and run Cline CLI on the runtime side."""
    from nexus.extensions.agents.cline.installer import install_cline_via_npm
    from nexus.runtime.context import require_runtime_control
    from nexus.runtime.server.services.openai_proxy import OpenAIProxy

    rt = require_runtime_control()
    start_time = time.time()

    # 1) Install Cline.
    try:
        cline_path = await install_cline_via_npm(
            install_dir=DEFAULT_INSTALL_DIR,
            version=version,
            npm_registry=npm_registry,
        )
        logger.debug(f'Cline binary at: {cline_path}')
    except Exception as e:
        logger.error(f'Failed to install Cline: {e}')
        return {
            'success': False,
            'output': '',
            'stderr': f'Failed to install Cline: {e}',
            'exit_code': -1,
            'execution_time': time.time() - start_time,
        }

    node_path = find_runtime_binary('node')
    if not node_path:
        return {
            'success': False,
            'output': '',
            'stderr': 'Node.js is required but not found in runtime',
            'exit_code': -1,
            'execution_time': time.time() - start_time,
        }

    # 2) Start the OpenAI proxy.
    proxy_port = find_free_port()
    logger.debug(f'Starting OpenAI Proxy on port {proxy_port}...')
    proxy = OpenAIProxy(
        sio=rt.sio,
        listen_port=proxy_port,
        listen_host='127.0.0.1',
        request_timeout=timeout,
        models=[model_id],
    )

    try:
        await proxy.start()
        logger.debug('OpenAI Proxy started')
    except Exception as e:
        logger.error(f'Failed to start OpenAI Proxy: {e}')
        return {
            'success': False,
            'output': '',
            'stderr': f'Failed to start OpenAI Proxy: {e}',
            'exit_code': -1,
            'execution_time': time.time() - start_time,
        }

    try:
        work_path = Path(working_dir)
        work_path.mkdir(parents=True, exist_ok=True)

        home = os.environ.get('HOME', '/root')
        proxy_url = f'http://127.0.0.1:{proxy_port}'

        # Isolated config dir per run so concurrent tasks don't conflict.
        cline_dir = tempfile.mkdtemp(prefix='cline_config_')

        cline_env: dict[str, str] = {
            'HOME': home,
            'PATH': build_runtime_path(DEFAULT_INSTALL_DIR),
            'CLINE_DIR': cline_dir,
            # Disable telemetry / update checks in CI
            'NO_COLOR': '1',
            'CI': '1',
            # Increase AI SDK request timeout for long-running thinking models
            'OPENAI_TIMEOUT': str(int(timeout)),
            'AI_SDK_TIMEOUT': str(int(timeout * 1000)),  # AI SDK uses ms
        }
        if extra_env:
            cline_env.update(extra_env)

        # 3) Configure Cline auth to use the local OpenAI-compatible proxy.
        auth_cmd: list[str] = [
            node_path,
            str(cline_path),
            'auth',
            '-p',
            'openai',
            '-k',
            DEFAULT_PROXY_API_KEY,
            '-b',
            proxy_url,
            '-m',
            model_id,
        ]
        logger.debug('Configuring Cline auth: %s', ' '.join(shlex.quote(c) for c in auth_cmd))
        auth_exit, auth_out, auth_err = await _run_streaming(
            cmd=auth_cmd,
            cwd=working_dir,
            env=cline_env,
            timeout=30,
            log_prefix='[cline:auth]',
        )
        if auth_exit != 0:
            logger.warning(f'cline auth exited {auth_exit}: {auth_err or auth_out}')
        else:
            logger.debug('Cline auth configured successfully')

        # 4) Pre-start scripts.
        for script in pre_start_scripts or []:
            try:
                await _run_pre_start_script(script, working_dir, cline_env)
            except Exception as e:
                return {
                    'success': False,
                    'output': '',
                    'stderr': f'Pre-start script failed: {e}',
                    'exit_code': -1,
                    'execution_time': time.time() - start_time,
                }

        # 5) Run Cline in headless YOLO mode.
        cmd: list[str] = [
            node_path,
            str(cline_path),
            '-y',  # YOLO: auto-approve all actions, plain-text output, exit on completion
            '-c',
            working_dir,
        ]
        if thinking_budget_tokens > 0:
            # Use --reasoning-effort for OpenAI-compatible models.
            # --thinking is Anthropic-only (extended thinking blocks).
            reasoning_effort = os.environ.get('OPENAI_REASONING_EFFORT', 'high')
            cmd.extend(['--reasoning-effort', reasoning_effort])
        if extra_cli_args:
            cmd.extend(extra_cli_args)
        # Cline treats argv ``@token`` as a file mention and can exit 0 with no LLM
        # call. Prefix ``@`` so GraphQL/Java annotations in the prompt are not parsed.
        cmd.append(task.replace('@', r'\@'))

        logger.debug('Running Cline: %s', ' '.join(shlex.quote(c) for c in cmd))

        for attempt in range(1, NOOP_MAX_ATTEMPTS + 1):
            exit_code, stdout, stderr = await _run_streaming(
                cmd=cmd,
                cwd=working_dir,
                env=cline_env,
                timeout=timeout,
                log_prefix='[cline]',
            )
            if not _is_silent_noop(exit_code, stdout):
                break
            logger.warning(
                'Cline exited 0 with no output on attempt %d/%d (silent no-op); relaunching',
                attempt,
                NOOP_MAX_ATTEMPTS,
            )

        execution_time = time.time() - start_time
        logger.info(f'Cline finished with exit code {exit_code} in {execution_time:.2f}s')

        if _is_silent_noop(exit_code, stdout):
            logger.error('Cline produced no output after %d attempts; reporting failure', NOOP_MAX_ATTEMPTS)
            return {
                'success': False,
                'output': stdout,
                'stderr': stderr,
                'exit_code': exit_code,
                'execution_time': execution_time,
                'failure_kind': 'infrastructure',
                'failure_stage': NOOP_FAILURE_STAGE,
            }

        return {
            'success': exit_code == 0,
            'output': stdout,
            'stderr': stderr,
            'exit_code': exit_code,
            'execution_time': execution_time,
        }

    finally:
        logger.debug('Stopping OpenAI Proxy...')
        await proxy.stop()


__all__ = ['setup_and_run_cline']
