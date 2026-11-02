"""Runtime-side setup and execution functions for HyCode.

Executed on the runtime side via ``runtime.invoke``. Responsible for:
  1. installing HyCode via npm (@tencent/hycode),
  2. patching config files under ~/.claude/,
  3. running user-provided pre-start scripts,
  4. starting the Anthropic proxy on a free port,
  5. spawning ``opencc-bin --print`` and streaming stdout/stderr back via logs.
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
import uuid

from pathlib import Path
from typing import Any

from nexus.extensions.agents.hycode.installer import DEFAULT_HYCODE_VERSION
from nexus.extensions.agents.installed_common import (
    DEFAULT_PROXY_API_KEY,
    build_runtime_path,
    encode_stdin_prompt,
    find_free_port,
    find_runtime_binary,
    write_and_close_stdin,
)
from nexus.extensions.agents.installed_common.execution import AgentExecutionTransport as ExecutionTransport
from nexus.extensions.agents.installed_common.execution import ExecutionTransportError
from nexus.extensions.agents.installed_common.execution import LegacyAgentExecutionTransport as AgentExecutionTransport


logger = logging.getLogger(__name__)


def _deep_merge(target: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *patch* into *target*, preferring *patch* for scalars."""
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = value
    return target


def _resolve_config_path(raw: str, *, home: str | None = None) -> Path:
    """Expand ``~`` against the effective runtime HOME."""
    if home is not None and raw.startswith('~/'):
        return (Path(home) / raw[2:]).resolve()
    return Path(os.path.expanduser(raw)).resolve()


def _apply_json_patch(raw_path: str, patch: dict[str, Any], *, home: str | None = None) -> None:
    """Deep-merge *patch* into the JSON file at *raw_path*, creating it if missing."""
    path = _resolve_config_path(raw_path, home=home)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, Any] = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text()) or {}
        except Exception as exc:
            logger.warning('Failed to parse existing %s: %s; overwriting.', path, exc)
            existing = {}
    merged = _deep_merge(existing, patch)
    path.write_text(json.dumps(merged, indent=2, ensure_ascii=False))
    logger.debug('Patched %s', path)


async def _target_is_musl(transport: AgentExecutionTransport) -> bool | None:
    """Ask a remote execution target which HyCode binary it can execute.

    The package is installed here, on the Controller, but exec'd inside the service, and a
    Nix-based Controller reads as musl while the Agent's image is Debian. Reading this
    machine's libc picks a build whose loader is missing there -- measured as
    ``opencc-bin-musl: cannot execute: required file not found`` against a Debian service.
    """
    from nexus.extensions.agents.claude_code.installer import LINUX_LIBC_PROBE
    from nexus.runtime.executors.command import RunCommandRequest, run_command

    if transport.is_local:
        return None

    argv = transport.probe_command(LINUX_LIBC_PROBE)
    result = await run_command(RunCommandRequest(command=shlex.join(argv), timeout=60))
    if result.return_code != 0:
        details = (result.stderr or result.stdout or 'no output').strip()
        raise RuntimeError(f'Could not probe libc in Agent execution target: {details}')

    fields = (result.stdout or '').strip().splitlines()[-1].split()
    if len(fields) != 2:
        raise RuntimeError(f'unexpected libc probe output: {result.stdout!r}')
    is_musl = fields[1] == 'musl'
    logger.debug('Agent execution target reports libc %s', fields[1])
    return is_musl


async def _run_pre_start_script(
    script: dict[str, Any],
    working_dir: str,
    base_env: dict[str, str],
    *,
    transport: ExecutionTransport | None = None,
    invocation_id: str | None = None,
) -> None:
    """Execute a single pre-start script."""
    from nexus.runtime.executors.command import CommandRunStatus, RunCommandRequest, run_command

    interpreter = script.get('interpreter', 'bash')
    code = script.get('code', '')
    timeout = float(script.get('timeout', 300.0))
    cwd = script.get('cwd') or working_dir
    fail_on_error = bool(script.get('fail_on_error', True))
    resolved_transport = transport or AgentExecutionTransport()
    resolved_invocation_id = invocation_id or uuid.uuid4().hex

    if interpreter not in {'bash', 'python'}:
        raise ValueError(f'Unsupported pre-start interpreter: {interpreter}')
    if not resolved_transport.is_local:
        binary = 'bash' if interpreter == 'bash' else 'python3'
        cmd = [f'/nix/runtime/nix-env/bin/{binary}', '-c', code]
    elif interpreter == 'bash':
        bash = find_runtime_binary('bash')
        if bash is None:
            raise RuntimeError('No bash binary available for the HyCode pre-start script')
        cmd = [bash, '-c', code]
    else:
        py = find_runtime_binary('python') or find_runtime_binary('python3') or 'python3'
        cmd = [py, '-c', code]

    script_env = dict(base_env)
    if script.get('env'):
        script_env.update(script['env'])

    logger.debug('Running pre-start (%s) in %s', interpreter, cwd)
    if resolved_transport.is_local:
        result = await run_command(
            RunCommandRequest(command=shlex.join(cmd), cwd=cwd, timeout=timeout, extra_env=script_env)
        )
    else:
        async with resolved_transport.prepare(
            cmd,
            working_dir=cwd,
            invocation_id=resolved_invocation_id,
            env=script_env,
        ) as prepared:
            result = await run_command(
                RunCommandRequest(
                    command=shlex.join(prepared.command),
                    cwd=prepared.cwd,
                    timeout=timeout,
                    extra_env=prepared.env,
                )
            )
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
) -> None:
    """Read newline-delimited output without imposing asyncio's line limit."""
    pending = b''
    while True:
        try:
            chunk = await stream.read(65536)
        except Exception:
            logger.exception('%sstream read failed; stopping reader', prefix)
            break
        if not chunk:
            if pending:
                text = pending.decode('utf-8', errors='replace')
                buffer.append(text)
                logger.log(level, '%s%s', prefix, text)
            break
        pending += chunk
        while b'\n' in pending:
            line, pending = pending.split(b'\n', 1)
            text = line.decode('utf-8', errors='replace')
            buffer.append(text)
            logger.log(level, '%s%s', prefix, text)


async def _run_hycode_streaming(
    cmd: list[str],
    cwd: str,
    env: dict[str, str],
    timeout: float,
    timeout_control_id: str | None = None,
    stdin_input: str | None = None,
) -> tuple[int, str, str, bool]:
    """Spawn HyCode with line-buffered stdout/stderr piped into the logger.

    The fourth element reports whether the task wall clock ran out. A timeout kills the
    process, so its exit code comes from the kernel and the CLI never gets to describe what
    happened -- only the host can tell run_hycode that this was a timeout rather than a crash.

    ``stdin_input`` is written to the child's stdin, then closed. The stdin write shares the
    task timeout so a child that never reads a large prompt cannot hang ``drain()``.
    """
    from nexus.runtime.executors.command import registered_command_timeout
    from nexus.utils.shell import get_os_env_without_ld_library_path
    from nexus.utils.timeout import async_timeout

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
        limit=16 * 1024 * 1024,
    )

    assert proc.stdout is not None
    assert proc.stderr is not None

    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    stdout_task = asyncio.create_task(_pipe_to_logger(proc.stdout, '[hycode] ', logging.DEBUG, stdout_lines))
    stderr_task = asyncio.create_task(_pipe_to_logger(proc.stderr, '[hycode:err] ', logging.WARNING, stderr_lines))
    io_tasks = [stdout_task, stderr_task]
    if stdin_bytes is not None:
        assert proc.stdin is not None
        io_tasks.append(asyncio.create_task(write_and_close_stdin(proc.stdin, stdin_bytes)))

    try:
        async with async_timeout(timeout) as timeout_handle:
            with registered_command_timeout(timeout_control_id, timeout_handle):
                await asyncio.gather(*io_tasks, return_exceptions=True)
    except (asyncio.TimeoutError, TimeoutError):
        proc.kill()
        with contextlib.suppress(Exception):
            await proc.wait()
        for io_task in io_tasks:
            io_task.cancel()
        return -1, '\n'.join(stdout_lines), '\n'.join(stderr_lines) + '\n[timeout]', True

    try:
        exit_code = await asyncio.wait_for(proc.wait(), timeout=30)
    except asyncio.TimeoutError:
        # Reaping a process that already exited, not the task timeout above: the pipes closed,
        # so HyCode is done and only wait() is slow. Not reported as timed_out.
        proc.kill()
        with contextlib.suppress(Exception):
            exit_code = await proc.wait()

    return (exit_code or 0), '\n'.join(stdout_lines), '\n'.join(stderr_lines), False


def _parse_last_result(stdout: str) -> dict[str, Any]:
    """Extract the last final result from HyCode's newline-delimited stream."""
    result_subtype: str | None = None
    num_turns: int | None = None
    result_is_error: bool | None = None
    result_count = 0
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict) or event.get('type') != 'result':
            continue
        result_count += 1
        result_subtype = event.get('subtype')
        num_turns = event.get('num_turns')
        result_is_error = event.get('is_error')
    return {
        'result_subtype': result_subtype,
        'num_turns': num_turns,
        'result_is_error': result_is_error,
        'result_count': result_count,
    }


async def setup_and_run_hycode(
    task: str,
    working_dir: str = '/workspace',
    timeout: float = 3600,
    version: str = DEFAULT_HYCODE_VERSION,
    npm_registry: str | None = None,
    extra_env: dict[str, str] | None = None,
    extra_cli_args: list[str] | None = None,
    config_file_patches: list[dict[str, Any]] | None = None,
    pre_start_scripts: list[dict[str, Any]] | None = None,
    conversion_log_level: str = 'INFO',
    thinking_preset: str = 'hunyuan',
    execution_target: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Set up and run HyCode on the runtime side."""
    from nexus.extensions.agents.hycode.installer import DEFAULT_INSTALL_DIR, install_hycode_via_npm
    from nexus.runtime.context import require_runtime_control
    from nexus.runtime.server.services.anthropic_proxy import AnthropicProxy
    from nexus.trace import get_current_span_id

    rt = require_runtime_control()
    start_time = time.time()
    transport = AgentExecutionTransport(execution_target)
    transport.validate()
    invocation_id = uuid.uuid4().hex
    install_dir = transport.install_dir(agent_name='hycode', default=DEFAULT_INSTALL_DIR)

    # 1) Install HyCode via npm.
    try:
        binary = await install_hycode_via_npm(
            install_dir=install_dir,
            version=version,
            npm_registry=npm_registry,
            target_is_musl=await _target_is_musl(transport),
        )
        if not transport.is_local:
            binary = Path(binary).resolve(strict=True)
            try:
                binary.relative_to(install_dir)
            except ValueError as exc:
                raise RuntimeError(f'HyCode executable escaped the Agent install directory: {binary}') from exc
        logger.debug('HyCode binary at: %s', binary)
    except Exception as e:
        logger.error('Failed to install HyCode: %s', e)
        return {
            'success': False,
            'output': '',
            'stderr': f'Failed to install HyCode: {e}',
            'exit_code': -1,
            'execution_time': time.time() - start_time,
            'failure_kind': 'infrastructure',
            'failure_stage': 'install',
        }

    # 2) Start the Anthropic proxy.
    proxy_port = find_free_port()
    logger.debug('Starting Anthropic Proxy on port %d...', proxy_port)
    proxy = AnthropicProxy(
        sio=rt.sio,
        listen_port=proxy_port,
        listen_host='127.0.0.1',
        # The subprocess owns the active-time budget. A second wall-clock deadline
        # would still expire while partial rollout is parked for training or eval.
        request_timeout=None,
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
            'failure_kind': 'infrastructure',
            'failure_stage': 'proxy',
        }

    try:
        work_path = Path(working_dir)
        work_path.mkdir(parents=True, exist_ok=True)

        home_path = transport.home_dir(
            invocation_id=invocation_id,
            default=os.environ.get('HOME', '/root'),
        )
        home_path.mkdir(parents=True, exist_ok=True)
        home = str(home_path)
        hycode_env: dict[str, str] = {
            'ANTHROPIC_BASE_URL': f'http://127.0.0.1:{proxy_port}',
            'NEXUS_AGENT_PROXY_URL': f'http://127.0.0.1:{proxy_port}',
            'ANTHROPIC_API_KEY': DEFAULT_PROXY_API_KEY,
            'HOME': home,
            'PATH': build_runtime_path(install_dir),
            'IS_SANDBOX': '1',
        }
        if extra_env:
            hycode_env.update(extra_env)

        # 3) Apply config patches (settings under ~/.claude/).
        for patch in config_file_patches or []:
            _apply_json_patch(patch['path'], patch.get('content') or {}, home=home)

        # 4) Pre-start scripts.
        for script in pre_start_scripts or []:
            try:
                await _run_pre_start_script(
                    script,
                    working_dir,
                    hycode_env,
                    transport=transport,
                    invocation_id=invocation_id,
                )
            except Exception as e:
                return {
                    'success': False,
                    'output': '',
                    'stderr': f'Pre-start script failed: {e}',
                    'exit_code': -1,
                    'execution_time': time.time() - start_time,
                    'failure_kind': 'infrastructure',
                    'failure_stage': 'pre_start',
                }

        # 5) Run the installed binary. The prompt goes through stdin: argv is visible in
        # /proc/<pid>/cmdline, so an agent's `pkill -f <name>` would match any name the
        # prompt mentions and kill HyCode itself.
        cmd: list[str] = [str(binary), '--dangerously-skip-permissions']
        if extra_cli_args:
            cmd.extend(extra_cli_args)
        async with transport.prepare(
            cmd,
            working_dir=working_dir,
            invocation_id=invocation_id,
            env=hycode_env,
        ) as prepared:
            logger.debug('Running HyCode: %s', ' '.join(shlex.quote(c) for c in prepared.command))
            exit_code, stdout, stderr, timed_out = await _run_hycode_streaming(
                cmd=prepared.command,
                cwd=prepared.cwd,
                env=prepared.env,
                timeout=timeout,
                timeout_control_id=get_current_span_id(),
                stdin_input=task,
            )

        execution_time = time.time() - start_time
        logger.info('HyCode finished with exit code %d in %.2fs', exit_code, execution_time)

        parsed_result = _parse_last_result(stdout)
        failure_kind, failure_stage = transport.classify_failure(exit_code, stderr)

        return {
            'success': exit_code == 0,
            'output': stdout,
            'stderr': stderr,
            'exit_code': exit_code,
            'execution_time': execution_time,
            'timed_out': timed_out,
            'result_subtype': parsed_result['result_subtype'],
            'num_turns': parsed_result['num_turns'],
            'result_is_error': parsed_result['result_is_error'],
            'result_count': parsed_result['result_count'],
            'failure_kind': failure_kind,
            'failure_stage': failure_stage,
        }

    except ExecutionTransportError as exc:
        logger.error('HyCode execution transport failed: %s', exc)
        return {
            'success': False,
            'output': '',
            'stderr': str(exc),
            'exit_code': -1,
            'execution_time': time.time() - start_time,
            'timed_out': False,
            'failure_kind': 'infrastructure',
            'failure_stage': 'execution_transport',
        }
    finally:
        logger.debug('Stopping Anthropic Proxy...')
        await proxy.stop()


__all__ = ['setup_and_run_hycode']
