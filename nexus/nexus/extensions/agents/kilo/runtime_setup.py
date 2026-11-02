"""Runtime-side setup and execution for Kilo Code CLI.

Executed on the runtime side via ``runtime.invoke``.  Responsible for:
  1. Installing the Kilo CLI (npm),
  2. Starting an OpenAI-compatible proxy on a free port,
  3. Writing a kilo config that routes LLM calls through the proxy,
  4. Running pre-start scripts,
  5. Spawning ``kilo run`` in headless mode and streaming stdout/stderr.
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
    find_nexus_runtime_binary,
    find_runtime_binary,
)


# NOTE: Do NOT import OpenAIProxy at module level!
# It has fastapi/starlette dependencies that are runtime-only.
# Import it inside functions that run on the runtime side.

logger = logging.getLogger(__name__)


DEFAULT_INSTALL_DIR = f'{DEFAULT_AGENT_INSTALL_ROOT}/kilo'
DEFAULT_MODEL_ID = 'nexus-proxy-model'
KILO_PROVIDER_ID = 'nexus-proxy'
STDIN_WRITE_TIMEOUT = 120.0


def _build_kilo_config(
    proxy_url: str,
    model_id: str,
    api_key: str,
    thinking_budget_tokens: int = 0,
) -> dict[str, Any]:
    """Build a kilo config dict that routes LLM calls through the local proxy.

    Uses a custom provider backed by ``@ai-sdk/openai-compatible`` so that
    all requests hit our OpenAI-compatible proxy.

    Args:
        thinking_budget_tokens: When > 0, enable extended thinking with this
            token budget.  Set to 0 (default) to disable thinking.
    """
    model_config: dict[str, Any] = {
        'id': model_id,
        'name': model_id,
        'tool_call': True,
        'temperature': True,
        'reasoning': True,
        'limit': {'context': 200000, 'output': 16384},
    }
    if thinking_budget_tokens > 0:
        model_config['options'] = {
            'thinking': {
                'type': 'enabled',
                'budgetTokens': thinking_budget_tokens,
            },
        }
    return {
        'provider': {
            KILO_PROVIDER_ID: {
                'name': 'Nexus Proxy',
                'api': proxy_url,
                'npm': '@ai-sdk/openai-compatible',
                'env': [],
                'options': {
                    'apiKey': api_key,
                    'baseURL': proxy_url,
                },
                'models': {
                    model_id: model_config,
                },
            },
        },
        'model': f'{KILO_PROVIDER_ID}/{model_id}',
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


# KiloOpenAIProxy is defined as a function to avoid importing OpenAIProxy at module level
def _get_kilo_proxy_class():
    """Get KiloOpenAIProxy class with late-bound OpenAIProxy import."""
    from nexus.runtime.server.services.openai_proxy import OpenAIProxy

    class KiloOpenAIProxy(OpenAIProxy):
        """OpenAI proxy subclass that detects 'suggest' tool calls in LLM responses.

        When Kilo calls the 'suggest' tool, it means the agent considers its work done
        and is waiting for user interaction — which will never arrive in headless mode.
        Setting the stop event allows the runner to terminate Kilo early.
        """

        def __init__(self, *args: Any, suggest_stop_event: asyncio.Event, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._suggest_stop_event = suggest_stop_event

        def _inspect_response_payload(self, payload: dict) -> None:
            """Detect 'suggest' tool_call in the LLM response and signal termination."""
            if not isinstance(payload, dict):
                return
            for choice in payload.get('choices', []) or []:
                message = (choice or {}).get('message') or {}
                for tc in message.get('tool_calls') or []:
                    fn = (tc or {}).get('function') or {}
                    fn_name = (fn.get('name') or '').lower()
                    if 'suggest' in fn_name:
                        logger.debug('[kilo] Suggest tool_call detected (%s) — will terminate', fn.get('name'))
                        self._suggest_stop_event.set()
                        return

    return KiloOpenAIProxy


async def _write_task_to_stdin(proc: asyncio.subprocess.Process, task: str) -> bool:
    """Feed the task to Kilo through stdin, then close it so the CLI sees EOF and starts.

    Returns False only when the handover stalled, i.e. the task may have arrived
    truncated; a broken pipe means Kilo already exited and returns True so the
    caller reports its real exit code and stderr instead.
    """
    assert proc.stdin is not None
    try:
        # surrogateescape: what argv encoding used to do with an undecodable byte in the task
        proc.stdin.write(task.encode('utf-8', errors='surrogateescape'))
        await asyncio.wait_for(proc.stdin.drain(), timeout=STDIN_WRITE_TIMEOUT)
    except asyncio.TimeoutError:
        logger.error('[kilo] Timed out after %.0fs writing the task to stdin', STDIN_WRITE_TIMEOUT)
        return False
    except (BrokenPipeError, ConnectionResetError):
        logger.warning('[kilo] Kilo closed stdin before the task was written')
    finally:
        with contextlib.suppress(Exception):
            proc.stdin.close()
    return True


async def _run_kilo_streaming(
    cmd: list[str],
    task: str,
    cwd: str,
    env: dict[str, str],
    timeout: float,
    suggest_stop_event: asyncio.Event | None = None,
) -> tuple[int, str, str]:
    """Spawn Kilo with *task* on stdin and line-buffered stdout/stderr piped into the logger.

    If *suggest_stop_event* is provided and gets set (by the OpenAI proxy
    detecting a 'suggest' tool call), the process is terminated gracefully
    and treated as a successful exit.
    """
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

    stdout_task = asyncio.create_task(_pipe_to_logger(proc.stdout, '[kilo] ', logging.DEBUG, stdout_lines))
    stderr_task = asyncio.create_task(_pipe_to_logger(proc.stderr, '[kilo:err] ', logging.WARNING, stderr_lines))

    # after the readers are up: a task larger than the pipe buffer would otherwise deadlock against Kilo's own output
    if not await _write_task_to_stdin(proc, task):
        proc.kill()
        with contextlib.suppress(Exception):
            await proc.wait()
        stdout_task.cancel()
        stderr_task.cancel()
        return -1, '\n'.join(stdout_lines), '\n'.join(stderr_lines) + '\n[stdin write failed]'

    try:
        wait_proc = asyncio.create_task(proc.wait())
        tasks_to_wait = [wait_proc]

        if suggest_stop_event is not None:
            wait_suggest = asyncio.create_task(suggest_stop_event.wait())
            tasks_to_wait.append(wait_suggest)

        done, pending = await asyncio.wait(
            tasks_to_wait,
            timeout=timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )

        if not done:
            # Timeout
            proc.kill()
            with contextlib.suppress(Exception):
                await proc.wait()
            stdout_task.cancel()
            stderr_task.cancel()
            for t in pending:
                t.cancel()
            return -1, '\n'.join(stdout_lines), '\n'.join(stderr_lines) + '\n[timeout]'

        if suggest_stop_event is not None and wait_suggest in done:
            # Suggest detected — give Kilo a moment to flush, then terminate
            logger.debug('[kilo] Suggest detected, terminating gracefully...')
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=10.0)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
            for t in pending:
                t.cancel()
            await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
            return 0, '\n'.join(stdout_lines), '\n'.join(stderr_lines)

        # Normal process exit
        exit_code = wait_proc.result()
        for t in pending:
            t.cancel()

    except asyncio.TimeoutError:
        proc.kill()
        with contextlib.suppress(Exception):
            await proc.wait()
        stdout_task.cancel()
        stderr_task.cancel()
        return -1, '\n'.join(stdout_lines), '\n'.join(stderr_lines) + '\n[timeout]'

    await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
    return exit_code or 0, '\n'.join(stdout_lines), '\n'.join(stderr_lines)


async def setup_and_run_kilo(
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
    thinking_budget_tokens: int = 0,
) -> dict[str, Any]:
    """Set up and run Kilo Code CLI on the runtime side."""
    from nexus.extensions.agents.kilo.installer import install_kilo_via_npm
    from nexus.runtime.context import require_runtime_control

    rt = require_runtime_control()
    start_time = time.time()

    # 1) Install Kilo.
    try:
        kilo_path = await install_kilo_via_npm(
            install_dir=DEFAULT_INSTALL_DIR,
            version=version,
            npm_registry=npm_registry,
        )
        logger.debug(f'Kilo binary at: {kilo_path}')
    except Exception as e:
        logger.error(f'Failed to install Kilo: {e}')
        return {
            'success': False,
            'output': '',
            'stderr': f'Failed to install Kilo: {e}',
            'exit_code': -1,
            'execution_time': time.time() - start_time,
        }

    # 2) Start the OpenAI proxy with suggest detection.
    proxy_port = find_free_port()
    suggest_stop_event = asyncio.Event()

    logger.debug(f'Starting OpenAI Proxy on port {proxy_port}...')
    kilo_proxy_class = _get_kilo_proxy_class()
    proxy = kilo_proxy_class(
        sio=rt.sio,
        listen_port=proxy_port,
        listen_host='127.0.0.1',
        request_timeout=timeout,
        models=[model_id],
        suggest_stop_event=suggest_stop_event,
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
        kilo_config = _build_kilo_config(proxy_url, model_id, DEFAULT_PROXY_API_KEY, thinking_budget_tokens)
        kilo_config_json = json.dumps(kilo_config, ensure_ascii=False)

        kilo_env: dict[str, str] = {
            'HOME': home,
            'PATH': build_runtime_path(DEFAULT_INSTALL_DIR),
            'KILO_CONFIG_CONTENT': kilo_config_json,
            'KILO_DISABLE_MODELS_FETCH': '1',
        }
        if extra_env:
            kilo_env.update(extra_env)

        # 4) Pre-start scripts.
        for script in pre_start_scripts or []:
            try:
                await _run_pre_start_script(script, working_dir, kilo_env)
            except Exception as e:
                return {
                    'success': False,
                    'output': '',
                    'stderr': f'Pre-start script failed: {e}',
                    'exit_code': -1,
                    'execution_time': time.time() - start_time,
                }

        # 5) Run Kilo CLI in headless mode.
        node_path = find_nexus_runtime_binary('node')
        if not node_path:
            raise RuntimeError('Node.js is required but not found in runtime')

        cmd: list[str] = [
            node_path,
            str(kilo_path),
            'run',
            '--auto',
            '--dir',
            working_dir,
            '--format',
            output_format,
        ]
        if extra_cli_args:
            cmd.extend(extra_cli_args)

        # the task goes over stdin: `kilo run <task>` quotes any positional holding a space and escapes the quotes inside it
        logger.debug('Running Kilo: %s <task on stdin, %d chars>', ' '.join(shlex.quote(c) for c in cmd), len(task))

        exit_code, stdout, stderr = await _run_kilo_streaming(
            cmd=cmd,
            task=task,
            cwd=working_dir,
            env=kilo_env,
            timeout=timeout,
            suggest_stop_event=suggest_stop_event,
        )

        execution_time = time.time() - start_time
        logger.info(f'Kilo finished with exit code {exit_code} in {execution_time:.2f}s')

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


__all__ = ['setup_and_run_kilo']
