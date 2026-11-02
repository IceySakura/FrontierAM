"""Runtime-side setup and execution for OpenCode CLI.

Executed on the runtime side via ``runtime.invoke``.  Responsible for:
  1. Installing the OpenCode CLI (npm),
  2. Starting an OpenAI-compatible proxy on a free port,
  3. Writing an opencode config that routes LLM calls through the proxy,
  4. Running pre-start scripts,
  5. Spawning ``opencode run`` in headless mode and streaming stdout/stderr.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import shlex
import subprocess
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


DEFAULT_INSTALL_DIR = f'{DEFAULT_AGENT_INSTALL_ROOT}/opencode'
DEFAULT_MODEL_ID = 'nexus-proxy-model'
OPENCODE_PROVIDER_ID = 'nexus-proxy'
STDIN_WRITE_TIMEOUT = 120.0


def _build_opencode_config(proxy_url: str, model_id: str, api_key: str) -> dict[str, Any]:
    """Build an opencode config dict that routes LLM calls through the local proxy."""
    return {
        'provider': {
            OPENCODE_PROVIDER_ID: {
                'name': 'Nexus Proxy',
                'api': proxy_url,
                'npm': '@ai-sdk/openai-compatible',
                'env': [],
                'options': {
                    'apiKey': api_key,
                    'baseURL': proxy_url,
                },
                'models': {
                    model_id: {
                        'id': model_id,
                        'name': model_id,
                        'tool_call': True,
                        'temperature': True,
                        'reasoning': True,
                        'limit': {'context': 200000, 'output': 16384},
                    },
                },
            },
        },
        'model': f'{OPENCODE_PROVIDER_ID}/{model_id}',
        'experimental': {
            'openTelemetry': False,
        },
    }


async def _run_pre_start_script(
    script: dict[str, Any],
    working_dir: str,
    base_env: dict[str, str],
) -> None:
    """Execute a single pre-start script.  Raises if fail_on_error and exit != 0."""
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


async def _write_task_to_stdin(proc: asyncio.subprocess.Process, task: str) -> bool:
    """Feed the task to OpenCode through stdin, then close it so the CLI sees EOF and starts.

    Returns False only when the handover stalled, i.e. the task may have arrived
    truncated; a broken pipe means OpenCode already exited and returns True so the
    caller reports its real exit code and stderr instead.
    """
    assert proc.stdin is not None
    try:
        # surrogateescape: what argv encoding used to do with an undecodable byte in the task
        proc.stdin.write(task.encode('utf-8', errors='surrogateescape'))
        await asyncio.wait_for(proc.stdin.drain(), timeout=STDIN_WRITE_TIMEOUT)
    except asyncio.TimeoutError:
        logger.error('[opencode] Timed out after %.0fs writing the task to stdin', STDIN_WRITE_TIMEOUT)
        return False
    except (BrokenPipeError, ConnectionResetError):
        logger.warning('[opencode] OpenCode closed stdin before the task was written')
    finally:
        with contextlib.suppress(Exception):
            proc.stdin.close()
    return True


async def _run_opencode_streaming(
    cmd: list[str],
    task: str,
    cwd: str,
    env: dict[str, str],
    timeout: float,
) -> tuple[int, str, str]:
    """Spawn OpenCode with *task* on stdin and line-buffered stdout/stderr piped into the logger."""
    from nexus.utils.shell import get_os_env_without_ld_library_path

    base_env = get_os_env_without_ld_library_path()
    base_env['BASH_ENV'] = '~/.bashrc'
    merged_env = {**base_env, **env}

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd,
        env=merged_env,
    )

    assert proc.stdout is not None
    assert proc.stderr is not None

    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    stdout_task = asyncio.create_task(_pipe_to_logger(proc.stdout, '[opencode] ', logging.DEBUG, stdout_lines))
    stderr_task = asyncio.create_task(_pipe_to_logger(proc.stderr, '[opencode:err] ', logging.WARNING, stderr_lines))

    # after the readers are up: a task larger than the pipe buffer would otherwise deadlock against OpenCode's own output
    if not await _write_task_to_stdin(proc, task):
        proc.kill()
        with contextlib.suppress(Exception):
            await proc.wait()
        stdout_task.cancel()
        stderr_task.cancel()
        return -1, '\n'.join(stdout_lines), '\n'.join(stderr_lines) + '\n[stdin write failed]'

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


async def setup_and_run_opencode(
    task: str,
    working_dir: str = '/workspace',
    timeout: float = 3600,
    version: str = 'latest',
    npm_registry: str | None = DEFAULT_NPM_REGISTRY,
    model_id: str = DEFAULT_MODEL_ID,
    extra_env: dict[str, str] | None = None,
    extra_cli_args: list[str] | None = None,
    pre_start_scripts: list[dict[str, Any]] | None = None,
    output_format: str = 'default',
) -> dict[str, Any]:
    """Set up and run OpenCode CLI on the runtime side."""
    from nexus.extensions.agents.opencode.installer import install_opencode_via_npm
    from nexus.runtime.context import require_runtime_control
    from nexus.runtime.server.services.openai_proxy import OpenAIProxy

    rt = require_runtime_control()
    start_time = time.time()

    # 1) Install OpenCode.
    try:
        opencode_path = await install_opencode_via_npm(
            install_dir=DEFAULT_INSTALL_DIR,
            version=version,
            npm_registry=npm_registry,
        )
        logger.debug(f'OpenCode binary at: {opencode_path}')
    except Exception as e:
        logger.error(f'Failed to install OpenCode: {e}')
        return {
            'success': False,
            'output': '',
            'stderr': f'Failed to install OpenCode: {e}',
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

        # 3) Build config routing LLM calls through the proxy.
        proxy_url = f'http://127.0.0.1:{proxy_port}'
        opencode_config = _build_opencode_config(proxy_url, model_id, DEFAULT_PROXY_API_KEY)
        opencode_config_json = json.dumps(opencode_config, ensure_ascii=False)

        oc_env: dict[str, str] = {
            'HOME': home,
            'PATH': build_runtime_path(DEFAULT_INSTALL_DIR),
            'OPENCODE_CONFIG_CONTENT': opencode_config_json,
            'OPENCODE_DISABLE_MODELS_FETCH': '1',
        }
        if extra_env:
            oc_env.update(extra_env)

        # 4) Pre-start scripts.
        for script in pre_start_scripts or []:
            try:
                await _run_pre_start_script(script, working_dir, oc_env)
            except Exception as e:
                return {
                    'success': False,
                    'output': '',
                    'stderr': f'Pre-start script failed: {e}',
                    'exit_code': -1,
                    'execution_time': time.time() - start_time,
                }

        # 5) Run OpenCode CLI in headless mode.
        # Newer opencode-ai npm packages ship a native ELF binary as the bin
        # entry instead of a JS launcher, so we exec the binary directly. The
        # bin path resolves via npm's node_modules/.bin/<name> symlink, which
        # works whether the underlying file is ELF or a JS script with a
        # `#!/usr/bin/env node` shebang.
        cmd: list[str] = [
            str(opencode_path),
            'run',
            '--dangerously-skip-permissions',
            '--dir',
            working_dir,
            '--format',
            output_format,
            '--title',
            'nexus-task',
        ]
        if extra_cli_args:
            cmd.extend(extra_cli_args)

        # the task goes over stdin: `opencode run <task>` quotes any positional holding a space and escapes the quotes inside it
        logger.debug('Running OpenCode: %s <task on stdin, %d chars>', ' '.join(shlex.quote(c) for c in cmd), len(task))

        exit_code, stdout, stderr = await _run_opencode_streaming(
            cmd=cmd,
            task=task,
            cwd=working_dir,
            env=oc_env,
            timeout=timeout,
        )

        execution_time = time.time() - start_time
        logger.info(f'OpenCode finished with exit code {exit_code} in {execution_time:.2f}s')

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


__all__ = ['setup_and_run_opencode']
