"""Runtime-side setup and execution for CodeBuddy CLI.

Executed on the runtime side via ``runtime.invoke``.  Responsible for:
  1. Installing the CodeBuddy CLI (npm),
  2. Starting an OpenAI-compatible proxy on a free port,
  3. Generating ``~/.codebuddy/models.json`` pointing at the proxy,
  4. Optionally patching ``~/.codebuddy/settings.json``,
  5. Running pre-start scripts,
  6. Spawning ``codebuddy -p`` and streaming stdout/stderr back via logs.
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


logger = logging.getLogger(__name__)


DEFAULT_INSTALL_DIR = f'{DEFAULT_AGENT_INSTALL_ROOT}/codebuddy'
DEFAULT_MODEL_ID = 'nexus-proxy-model'
SKIP_PERMISSIONS_FLAG = '--dangerously-skip-permissions'


def build_codebuddy_permission_cli_args(
    *, skip_permissions: bool, extra_cli_args: list[str] | None = None
) -> list[str]:
    """Build CLI arguments with the dedicated permission setting taking precedence over extras."""
    extras = [arg for arg in extra_cli_args or [] if arg != SKIP_PERMISSIONS_FLAG]
    return [SKIP_PERMISSIONS_FLAG, *extras] if skip_permissions else extras


def _deep_merge(target: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *patch* into *target*, preferring *patch* for scalars."""
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = value
    return target


def _resolve_config_path(raw: str, *, home: str | Path | None = None) -> Path:
    if home is not None and (raw == '~' or raw.startswith('~/')):
        suffix = raw[2:] if raw.startswith('~/') else ''
        return (Path(home) / suffix).resolve()
    return Path(os.path.expanduser(raw)).resolve()


def _apply_json_patch(raw_path: str, patch: dict[str, Any], *, home: str | Path | None = None) -> None:
    """Deep-merge *patch* into the JSON file at *raw_path*, creating it if missing."""
    path = _resolve_config_path(raw_path, home=home)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, Any] = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text()) or {}
        except Exception as exc:
            logger.warning(f'Failed to parse existing {path}: {exc}; overwriting.')
            existing = {}
    merged = _deep_merge(existing, patch)
    path.write_text(json.dumps(merged, indent=2, ensure_ascii=False))
    logger.debug(f'Patched {path}')


def _write_models_json(
    codebuddy_home: Path,
    proxy_url: str,
    model_id: str,
    api_key: str,
    *,
    max_output_tokens: int = 16384,
) -> None:
    """Generate ``~/.codebuddy/models.json`` so the CLI routes requests to our proxy.

    ``max_output_tokens`` caps one completion; WorkBuddy sessions raise it, CodeBuddy keeps the default.
    """
    models_json = codebuddy_home / 'models.json'
    models_config = {
        'models': [
            {
                'id': model_id,
                'name': model_id,
                'vendor': 'nexus',
                'apiKey': api_key,
                'maxInputTokens': 200000,
                'maxOutputTokens': max_output_tokens,
                'url': proxy_url,
                'supportsToolCall': True,
                'supportsImages': True,
            }
        ],
        'availableModels': [model_id],
    }
    models_json.write_text(json.dumps(models_config, indent=2, ensure_ascii=False))
    logger.debug(f'Wrote models.json at {models_json}')


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


async def _run_codebuddy_streaming(
    cmd: list[str],
    cwd: str,
    env: dict[str, str],
    timeout: float,
) -> tuple[int, str, str]:
    """Spawn CodeBuddy with line-buffered stdout/stderr piped into the logger."""
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

    stdout_task = asyncio.create_task(_pipe_to_logger(proc.stdout, '[codebuddy] ', logging.DEBUG, stdout_lines))
    stderr_task = asyncio.create_task(_pipe_to_logger(proc.stderr, '[codebuddy:err] ', logging.WARNING, stderr_lines))

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


async def setup_and_run_codebuddy(
    task: str,
    working_dir: str = '/workspace',
    timeout: float = 3600,
    version: str = 'latest',
    npm_registry: str | None = DEFAULT_NPM_REGISTRY,
    model_id: str = DEFAULT_MODEL_ID,
    extra_env: dict[str, str] | None = None,
    extra_cli_args: list[str] | None = None,
    config_file_patches: list[dict[str, Any]] | None = None,
    pre_start_scripts: list[dict[str, Any]] | None = None,
    *,
    skip_permissions: bool = True,
) -> dict[str, Any]:
    """Set up and run CodeBuddy CLI on the runtime side."""
    from nexus.extensions.agents.codebuddy.installer import install_codebuddy_via_npm
    from nexus.runtime.context import require_runtime_control
    from nexus.runtime.server.services.openai_proxy import OpenAIProxy

    rt = require_runtime_control()
    start_time = time.time()

    # 1) Install CodeBuddy.
    try:
        codebuddy_path = await install_codebuddy_via_npm(
            install_dir=DEFAULT_INSTALL_DIR,
            version=version,
            npm_registry=npm_registry,
        )
        logger.debug(f'CodeBuddy binary at: {codebuddy_path}')
    except Exception as e:
        logger.error(f'Failed to install CodeBuddy: {e}')
        return {
            'success': False,
            'output': '',
            'stderr': f'Failed to install CodeBuddy: {e}',
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
        codebuddy_home = Path(home) / '.codebuddy'
        codebuddy_home.mkdir(parents=True, exist_ok=True)

        # 3) Generate models.json pointing at the local proxy.
        proxy_url = f'http://127.0.0.1:{proxy_port}/v1/chat/completions'
        _write_models_json(codebuddy_home, proxy_url, model_id, DEFAULT_PROXY_API_KEY)

        cb_env: dict[str, str] = {
            'HOME': home,
            'PATH': build_runtime_path(DEFAULT_INSTALL_DIR),
            'IS_SANDBOX': '1',
        }
        if extra_env:
            cb_env.update(extra_env)

        # 4) Apply config patches.
        for patch in config_file_patches or []:
            _apply_json_patch(patch['path'], patch.get('content') or {})

        # 5) Pre-start scripts.
        for script in pre_start_scripts or []:
            try:
                await _run_pre_start_script(script, working_dir, cb_env)
            except Exception as e:
                return {
                    'success': False,
                    'output': '',
                    'stderr': f'Pre-start script failed: {e}',
                    'exit_code': -1,
                    'execution_time': time.time() - start_time,
                }

        # 6) Run CodeBuddy CLI.
        node_path = find_nexus_runtime_binary('node')
        if not node_path:
            raise RuntimeError('Node.js is required but not found in runtime')
        logger.debug('Using Node.js binary for CodeBuddy CLI: %s', node_path)

        cmd: list[str] = [
            node_path,
            str(codebuddy_path),
            *build_codebuddy_permission_cli_args(
                skip_permissions=skip_permissions,
                extra_cli_args=['--model', model_id, '-p', *(extra_cli_args or [])],
            ),
            task,
        ]

        logger.debug('Running CodeBuddy: %s', ' '.join(shlex.quote(c) for c in cmd))

        exit_code, stdout, stderr = await _run_codebuddy_streaming(
            cmd=cmd,
            cwd=working_dir,
            env=cb_env,
            timeout=timeout,
        )

        execution_time = time.time() - start_time
        logger.info(f'CodeBuddy finished with exit code {exit_code} in {execution_time:.2f}s')

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


__all__ = ['SKIP_PERMISSIONS_FLAG', 'build_codebuddy_permission_cli_args', 'setup_and_run_codebuddy']
