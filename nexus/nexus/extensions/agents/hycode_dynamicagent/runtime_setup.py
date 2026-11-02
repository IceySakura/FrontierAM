"""Runtime-side setup and execution for HyCode DynamicAgent."""

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
from typing import IO, Any

from nexus.extensions.agents.installed_common import (
    DEFAULT_PROXY_API_KEY,
    DEFAULT_RUNTIME_BIN_DIR,
    build_npm_env,
    build_runtime_path,
    close_tee_file,
    encode_stdin_prompt,
    find_free_port,
    find_runtime_binary,
    open_cli_tee_files,
    write_and_close_stdin,
    write_tee_line,
)

from .installer import DEFAULT_HYCODE_DYNAMICAGENT_VERSION


logger = logging.getLogger(__name__)


_TOOL_COMPAT_PRE_START_SCRIPT: dict[str, Any] = {
    'interpreter': 'bash',
    'timeout': 30.0,
    'fail_on_error': False,
    'code': (
        'set -u\n'
        'link_binary() {\n'
        '  name="$1"\n'
        '  shift\n'
        '  target="$(command -v "${name}" 2>/dev/null || true)"\n'
        '  if [ -z "${target}" ]; then\n'
        '    return 0\n'
        '  fi\n'
        '  for dest in "$@"; do\n'
        '    if [ ! -x "${dest}" ]; then\n'
        '      mkdir -p "$(dirname "${dest}")" 2>/dev/null || true\n'
        '      ln -sf "${target}" "${dest}" 2>/dev/null || true\n'
        '    fi\n'
        '  done\n'
        '}\n'
        'link_binary bash /bin/bash /usr/local/bin/bash /usr/bin/bash\n'
        'link_binary node /usr/local/bin/node /usr/bin/node /bin/node\n'
        'link_binary npm /usr/local/bin/npm /usr/bin/npm /bin/npm\n'
        'link_binary npx /usr/local/bin/npx /usr/bin/npx /bin/npx\n'
    ),
}


def _with_tool_compat_script(pre_start_scripts: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    return [_TOOL_COMPAT_PRE_START_SCRIPT, *(pre_start_scripts or [])]


def _build_cli_npm_env(npm_registry: str | None) -> dict[str, str]:
    env = build_npm_env(npm_registry=npm_registry)
    env.pop('PATH', None)
    env.update({
        'NPM_CONFIG_STRICT_SSL': 'false',
        'npm_config_strict_ssl': 'false',
    })
    return env


def _merge_cli_extra_env(extra_env: dict[str, str] | None, npm_registry: str | None) -> dict[str, str]:
    merged = _build_cli_npm_env(npm_registry)
    if extra_env:
        merged.update(extra_env)
    return merged


def _deep_merge(target: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = value
    return target


def _resolve_config_path(raw: str) -> Path:
    return Path(os.path.expanduser(raw)).resolve()


def _apply_json_patch(raw_path: str, patch: dict[str, Any]) -> None:
    path = _resolve_config_path(raw_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, Any] = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text()) or {}
        except Exception as exc:
            logger.warning('Failed to parse existing %s: %s; overwriting.', path, exc)
            existing = {}
    path.write_text(json.dumps(_deep_merge(existing, patch), indent=2, ensure_ascii=False))
    logger.debug('Patched %s', path)


async def _run_pre_start_script(
    script: dict[str, Any],
    working_dir: str,
    base_env: dict[str, str],
) -> None:
    from nexus.runtime.executors.command import CommandRunStatus, RunCommandRequest, run_command

    interpreter = script.get('interpreter', 'bash')
    code = script.get('code', '')
    timeout = float(script.get('timeout', 300.0))
    cwd = script.get('cwd') or working_dir
    fail_on_error = bool(script.get('fail_on_error', True))

    if interpreter == 'bash':
        shell = find_runtime_binary('bash') or find_runtime_binary('sh') or 'bash'
        cmd = f'{shlex.quote(shell)} -c {shlex.quote(code)}'
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


async def _pipe_to_logger(
    stream: asyncio.StreamReader,
    prefix: str,
    level: int,
    buffer: list[str],
    tee_fh: IO[str] | None = None,
) -> None:
    while True:
        line = await stream.readline()
        if not line:
            break
        text = line.decode('utf-8', errors='replace').rstrip('\n')
        buffer.append(text)
        logger.log(level, '%s%s', prefix, text)
        write_tee_line(tee_fh, text)


async def _run_hycode_dynamicagent_streaming(
    cmd: list[str],
    cwd: str,
    env: dict[str, str],
    timeout: float,
    tee_log_dir: str | None = None,
    stdin_input: str | None = None,
) -> tuple[int, str, str]:
    from nexus.utils.shell import get_os_env_without_ld_library_path

    base_env = get_os_env_without_ld_library_path()
    base_env['BASH_ENV'] = '~/.bashrc'
    merged_env = {**base_env, **env}
    stdin_bytes = encode_stdin_prompt(stdin_input) if stdin_input is not None else None

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=subprocess.PIPE if stdin_bytes is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd,
        env=merged_env,
    )

    assert proc.stdout is not None
    assert proc.stderr is not None

    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    tee_stdout_fh, tee_stderr_fh = open_cli_tee_files(tee_log_dir, agent_key='hycode_dynamicagent')

    try:
        stdout_task = asyncio.create_task(
            _pipe_to_logger(proc.stdout, '[hycode-dynamicagent] ', logging.DEBUG, stdout_lines, tee_stdout_fh)
        )
        stderr_task = asyncio.create_task(
            _pipe_to_logger(proc.stderr, '[hycode-dynamicagent:err] ', logging.WARNING, stderr_lines, tee_stderr_fh)
        )
        io_tasks = [stdout_task, stderr_task]
        if stdin_bytes is not None:
            assert proc.stdin is not None
            io_tasks.append(asyncio.create_task(write_and_close_stdin(proc.stdin, stdin_bytes)))

        try:
            await asyncio.wait_for(
                asyncio.gather(*io_tasks, return_exceptions=True),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            proc.kill()
            with contextlib.suppress(Exception):
                await proc.wait()
            for io_task in io_tasks:
                io_task.cancel()
            return -1, '\n'.join(stdout_lines), '\n'.join(stderr_lines) + '\n[timeout]'

        try:
            exit_code = await asyncio.wait_for(proc.wait(), timeout=30)
        except asyncio.TimeoutError:
            proc.kill()
            with contextlib.suppress(Exception):
                exit_code = await proc.wait()

        return (exit_code or 0), '\n'.join(stdout_lines), '\n'.join(stderr_lines)
    finally:
        close_tee_file(tee_stdout_fh)
        close_tee_file(tee_stderr_fh)


async def setup_and_run_hycode_dynamicagent(
    task: str,
    working_dir: str = '/workspace',
    timeout: float = 3600,
    version: str = DEFAULT_HYCODE_DYNAMICAGENT_VERSION,
    npm_registry: str | None = None,
    extra_env: dict[str, str] | None = None,
    extra_cli_args: list[str] | None = None,
    config_file_patches: list[dict[str, Any]] | None = None,
    pre_start_scripts: list[dict[str, Any]] | None = None,
    conversion_log_level: str = 'INFO',
    thinking_preset: str = 'hunyuan',
    tee_log_dir: str | None = None,
) -> dict[str, Any]:
    """Install and run ``@tencent/hycode-dynamicagent`` on the runtime side."""
    from nexus.extensions.agents.hycode_dynamicagent.installer import (
        DEFAULT_INSTALL_DIR,
        install_hycode_dynamicagent_via_npm,
    )
    from nexus.runtime.context import require_runtime_control
    from nexus.runtime.server.services.anthropic_proxy import AnthropicProxy

    rt = require_runtime_control()
    start_time = time.time()

    try:
        binary = await install_hycode_dynamicagent_via_npm(
            install_dir=DEFAULT_INSTALL_DIR,
            version=version,
            npm_registry=npm_registry,
        )
        logger.debug('HyCode DynamicAgent CLI at: %s', binary)
    except Exception as e:
        logger.error('Failed to install HyCode DynamicAgent: %s', e)
        return {
            'success': False,
            'output': '',
            'stderr': f'Failed to install HyCode DynamicAgent: {e}',
            'exit_code': -1,
            'execution_time': time.time() - start_time,
        }

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
        launch_env: dict[str, str] = {
            'ANTHROPIC_BASE_URL': f'http://127.0.0.1:{proxy_port}',
            'ANTHROPIC_API_KEY': DEFAULT_PROXY_API_KEY,
            'HOME': home,
            'PATH': build_runtime_path(DEFAULT_INSTALL_DIR, DEFAULT_RUNTIME_BIN_DIR),
            'IS_SANDBOX': '1',
        }
        launch_env.update(_build_cli_npm_env(npm_registry))
        if extra_env:
            launch_env.update(extra_env)

        for patch in config_file_patches or []:
            _apply_json_patch(patch['path'], patch.get('content') or {})

        for script in _with_tool_compat_script(pre_start_scripts):
            try:
                await _run_pre_start_script(script, working_dir, launch_env)
            except Exception as e:
                return {
                    'success': False,
                    'output': '',
                    'stderr': f'Pre-start script failed: {e}',
                    'exit_code': -1,
                    'execution_time': time.time() - start_time,
                }

        cmd: list[str] = [str(binary), '--dangerously-skip-permissions']
        if extra_cli_args:
            cmd.extend(extra_cli_args)
        # The prompt goes through stdin: argv is visible in /proc/<pid>/cmdline, so an
        # agent's `pkill -f <name>` would match any name the prompt mentions and kill
        # the CLI itself. The wrapper appends bundled --agents/--plugin-dir flags after
        # these arguments, so no positional argument may be passed here.
        cmd.append('-p')

        logger.debug('Running HyCode DynamicAgent: %s', ' '.join(shlex.quote(c) for c in cmd))

        exit_code, stdout, stderr = await _run_hycode_dynamicagent_streaming(
            cmd=cmd,
            cwd=working_dir,
            env=launch_env,
            timeout=timeout,
            tee_log_dir=tee_log_dir,
            stdin_input=task,
        )

        execution_time = time.time() - start_time
        logger.info('HyCode DynamicAgent finished with exit code %d in %.2fs', exit_code, execution_time)

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


async def run_hycode_dynamicagent_session_turn(
    *,
    session_id: str,
    prompt: str,
    parent_tool_use_id: str | None = None,
    working_dir: str = '/workspace',
    timeout: float = 3600,
    version: str = DEFAULT_HYCODE_DYNAMICAGENT_VERSION,
    npm_registry: str | None = None,
    extra_env: dict[str, str] | None = None,
    extra_cli_args: list[str] | None = None,
    config_file_patches: list[dict[str, Any]] | None = None,
    pre_start_scripts: list[dict[str, Any]] | None = None,
    conversion_log_level: str = 'INFO',
    thinking_preset: str = 'hunyuan',
    include_partial_messages: bool = False,
    include_hook_events: bool = False,
    enable_user_input_hooks: bool = False,
    record_trace_events: bool = True,
) -> dict[str, Any]:
    """Start or reuse a runtime-side HyCode DynamicAgent stream-json session for one turn."""
    from nexus.extensions.agents.hycode_dynamicagent.installer import (
        DEFAULT_INSTALL_DIR,
        install_hycode_dynamicagent_via_npm,
    )
    from nexus.extensions.agents.installed_common.multiturn import run_installed_cli_session_turn

    return await run_installed_cli_session_turn(
        agent_key='hycode_dynamicagent',
        agent_label='HyCode DynamicAgent',
        session_id=session_id,
        prompt=prompt,
        parent_tool_use_id=parent_tool_use_id,
        working_dir=working_dir,
        timeout=timeout,
        install_cli=install_hycode_dynamicagent_via_npm,
        install_dir=DEFAULT_INSTALL_DIR,
        install_kwargs={'version': version, 'npm_registry': npm_registry},
        path_prepend=[DEFAULT_INSTALL_DIR, DEFAULT_RUNTIME_BIN_DIR],
        extra_env=_merge_cli_extra_env(extra_env, npm_registry),
        extra_cli_args=extra_cli_args,
        config_file_patches=config_file_patches,
        pre_start_scripts=_with_tool_compat_script(pre_start_scripts),
        conversion_log_level=conversion_log_level,
        thinking_preset=thinking_preset,
        include_partial_messages=include_partial_messages,
        include_hook_events=include_hook_events,
        enable_user_input_hooks=enable_user_input_hooks,
        record_trace_events=record_trace_events,
    )


async def close_hycode_dynamicagent_session(*, session_id: str) -> dict[str, Any]:
    """Close one runtime-side HyCode DynamicAgent stream-json session."""
    from nexus.extensions.agents.installed_common.multiturn import close_installed_cli_session

    return await close_installed_cli_session(agent_key='hycode_dynamicagent', session_id=session_id)


__all__ = [
    'close_hycode_dynamicagent_session',
    'run_hycode_dynamicagent_session_turn',
    'setup_and_run_hycode_dynamicagent',
]
