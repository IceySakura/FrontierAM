"""Runtime-side setup and execution functions for Claude Code.

Executed on the runtime side via ``runtime.invoke``. Responsible for:
  1. installing the Claude Code CLI (npm by default, optional binary URL),
  2. patching Claude's config files,
  3. running user-provided pre-start scripts,
  4. starting the Anthropic proxy on a free port,
  5. registering MCP servers,
  6. spawning ``claude --print`` and streaming stdout/stderr back via logs.
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

from nexus.extensions.agents.installed_common import (
    DEFAULT_AGENT_INSTALL_ROOT,
    DEFAULT_CLI_STREAM_READER_LIMIT,
    DEFAULT_NPM_REGISTRY,
    DEFAULT_PROXY_API_KEY,
    build_runtime_command,
    build_runtime_path,
    find_free_port,
    find_runtime_binary,
)
from nexus.extensions.agents.installed_common.execution import AgentExecutionTransport, ExecutionTransportError
from nexus.utils.common import format_exception, terminate_process


logger = logging.getLogger(__name__)


DEFAULT_INSTALL_DIR = f'{DEFAULT_AGENT_INSTALL_ROOT}/claude-code'


def _deep_merge(target: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *patch* into *target*, preferring *patch* for scalars."""
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = value
    return target


def _resolve_config_path(raw: str, *, home: str | None = None) -> Path:
    """Expand ``~`` (against the runtime's HOME) and return an absolute path."""
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
            logger.warning(f'Failed to parse existing {path}: {exc}; overwriting.')
            existing = {}
    merged = _deep_merge(existing, patch)
    path.write_text(json.dumps(merged, indent=2, ensure_ascii=False))
    logger.debug(f'Patched {path}')


async def _run_pre_start_script(
    script: dict[str, Any],
    working_dir: str,
    base_env: dict[str, str],
    *,
    transport: AgentExecutionTransport | None = None,
    invocation_id: str | None = None,
) -> None:
    """Execute a single pre-start script. Raises if fail_on_error and exit != 0."""
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
        binary = await resolved_transport.find_binary('bash' if interpreter == 'bash' else 'python3')
        cmd = shlex.join([binary, '-c', code])
    elif interpreter == 'bash':
        cmd = build_runtime_command(['bash', '-c', code])
    else:
        py = find_runtime_binary('python') or find_runtime_binary('python3') or 'python3'
        cmd = build_runtime_command([py, '-c', code])

    script_env = dict(base_env)
    if script.get('env'):
        script_env.update(script['env'])

    logger.debug(f'Running pre-start ({interpreter}) in {cwd}')
    if resolved_transport.is_local:
        result = await run_command(
            RunCommandRequest(
                command=cmd,
                cwd=cwd,
                timeout=timeout,
                extra_env=script_env,
            )
        )
    else:
        async with resolved_transport.prepare(
            shlex.split(cmd),
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


async def _run_claude_streaming(
    cmd: list[str],
    cwd: str,
    env: dict[str, str],
    timeout: float,
    stdin_input: str | None = None,
) -> tuple[int, str, str]:
    """Spawn Claude with line-buffered stdout/stderr piped into the logger.

    When ``stdin_input`` is set, the prompt is fed through the child's stdin rather
    than as an argv element, keeping the (large) prompt out of claude's cmdline so an
    agent-issued ``pkill -f <substring-of-prompt>`` cannot match and kill the claude
    parent process (which surfaced as ``exit_code=-9``).
    """
    from nexus.utils.shell import get_os_env_without_ld_library_path, kill_all_child_processes

    base_env = get_os_env_without_ld_library_path()
    base_env['BASH_ENV'] = '~/.bashrc'
    merged_env = {**base_env, **env}

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=subprocess.PIPE if stdin_input is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd,
        env=merged_env,
        limit=DEFAULT_CLI_STREAM_READER_LIMIT,
    )

    assert proc.stdout is not None
    assert proc.stderr is not None

    async def stop_process() -> None:
        await kill_all_child_processes(proc.pid)
        await terminate_process(proc)

    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    stdout_task = asyncio.create_task(_pipe_to_logger(proc.stdout, '[claude] ', logging.DEBUG, stdout_lines))
    stderr_task = asyncio.create_task(_pipe_to_logger(proc.stderr, '[claude:err] ', logging.WARNING, stderr_lines))

    # `claude --print` blocks on stdin until EOF when given no positional prompt, so
    # write the whole prompt and close the pipe. The readers above are already
    # draining stdout/stderr, so this can't deadlock even for a large prompt.
    if stdin_input is not None and proc.stdin is not None:
        try:
            proc.stdin.write(stdin_input.encode('utf-8'))
            await proc.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            with contextlib.suppress(Exception):
                proc.stdin.close()

    # Wait for pipes to close (EOF) rather than proc.wait().
    # asyncio's SubprocessTransport unregisters pipe FDs from the event loop
    # when proc.wait() returns, which orphans any unread pipe data.
    # By waiting for pipe EOF first, we ensure all output is consumed.
    try:
        reader_results = await asyncio.wait_for(
            asyncio.gather(stdout_task, stderr_task, return_exceptions=True),
            timeout=timeout,
        )
    except asyncio.CancelledError:
        await asyncio.shield(stop_process())
        stdout_task.cancel()
        stderr_task.cancel()
        raise
    except asyncio.TimeoutError:
        await stop_process()
        stdout_task.cancel()
        stderr_task.cancel()
        return -1, '\n'.join(stdout_lines), '\n'.join(stderr_lines) + '\n[timeout]'

    reader_errors = [
        f'{name} reader failed: {format_exception(result)}'
        for name, result in (('stdout', reader_results[0]), ('stderr', reader_results[1]))
        if isinstance(result, Exception)
    ]
    if reader_errors:
        for error in reader_errors:
            logger.error('[claude] %s', error)
        await stop_process()
        return -1, '\n'.join(stdout_lines), '\n'.join([*stderr_lines, *reader_errors])

    # Pipes closed — process should be dead. Reap it.
    try:
        exit_code = await asyncio.wait_for(proc.wait(), timeout=30)
    except asyncio.TimeoutError:
        proc.kill()
        with contextlib.suppress(Exception):
            exit_code = await proc.wait()

    return (exit_code or 0), '\n'.join(stdout_lines), '\n'.join(stderr_lines)


def _build_single_run_command(
    claude_path: str | Path,
    *,
    extra_cli_args: list[str] | None,
) -> list[str]:
    # The prompt is NOT appended to argv — it is fed via the child's stdin (see
    # `_run_claude_streaming`'s ``stdin_input``). A command-line prompt lands in
    # claude's /proc/<pid>/cmdline, so an agent running e.g. `pkill -9 -f
    # './executable'` (a substring of the prompt) would match and SIGKILL the claude
    # parent itself → exit_code -9.
    extra_args = list(extra_cli_args or [])
    has_print = any(arg in {'--print', '-p'} or arg.startswith('--print=') for arg in extra_args)
    cmd: list[str] = [str(claude_path), '--dangerously-skip-permissions']
    if not has_print:
        cmd.append('--print')
    cmd.extend(extra_args)
    return cmd


async def setup_and_run_claude_code(
    task: str,
    working_dir: str = '/workspace',
    timeout: float = 3600,
    version: str | None = None,
    npm_registry: str | None = DEFAULT_NPM_REGISTRY,
    binary_url: str | None = None,
    binary_checksum: str | None = None,
    mcp_servers: list[dict[str, Any]] | None = None,
    extra_env: dict[str, str] | None = None,
    extra_cli_args: list[str] | None = None,
    config_file_patches: list[dict[str, Any]] | None = None,
    pre_start_scripts: list[dict[str, Any]] | None = None,
    skills_dir: str | None = None,
    execution_target: dict[str, Any] | None = None,
    uploaded_files: list[str] | None = None,
    conversion_log_level: str = 'INFO',
    thinking_preset: str = 'hunyuan',
) -> dict[str, Any]:
    """Set up and run Claude Code on the runtime side."""
    from nexus.extensions.agents.claude_code.installer import (
        install_claude_code_from_url,
        install_claude_code_via_npm,
    )
    from nexus.runtime.context import require_runtime_control
    from nexus.runtime.server.services.anthropic_proxy import AnthropicProxy

    rt = require_runtime_control()
    start_time = time.time()
    transport = AgentExecutionTransport(execution_target)
    transport.validate()
    invocation_id = uuid.uuid4().hex
    # 1) Install Claude Code in the same account and filesystem used for execution.
    try:
        await transport.initialize()
        install_dir = transport.install_dir(agent_name='claude-code', default=DEFAULT_INSTALL_DIR)
        install_kwargs: dict[str, Any] = {'transport': transport} if not transport.is_local else {}
        if binary_url:
            claude_path = await install_claude_code_from_url(
                url=binary_url,
                install_dir=install_dir,
                checksum=binary_checksum,
                **install_kwargs,
            )
        else:
            claude_path = await install_claude_code_via_npm(
                install_dir=install_dir,
                version=version,
                npm_registry=npm_registry,
                **install_kwargs,
            )
        logger.debug(f'Claude Code binary at: {claude_path}')
    except Exception as e:
        logger.error(f'Failed to install Claude Code: {e}')
        return {
            'success': False,
            'output': '',
            'stderr': f'Failed to install Claude Code: {e}',
            'exit_code': -1,
            'execution_time': time.time() - start_time,
            'failure_kind': 'infrastructure',
            'failure_stage': 'install',
        }

    # 2) Start the Anthropic proxy.
    proxy_port = find_free_port()
    logger.debug(f'Starting Anthropic Proxy on port {proxy_port}...')
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
        logger.error(f'Failed to start Anthropic Proxy: {e}')
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
        # Ensure working directory exists.
        work_path = Path(working_dir)
        await transport.mkdir(work_path)

        # Environment used for pre-start scripts and Claude.
        home_path = transport.home_dir(
            invocation_id=invocation_id,
            default=os.environ.get('HOME', '/root'),
        )
        await transport.mkdir(home_path)
        home = str(home_path)
        claude_env: dict[str, str] = {
            'ANTHROPIC_BASE_URL': f'http://127.0.0.1:{proxy_port}',
            'NEXUS_AGENT_PROXY_URL': f'http://127.0.0.1:{proxy_port}',
            'ANTHROPIC_API_KEY': DEFAULT_PROXY_API_KEY,
            'HOME': home,
            'PATH': (
                build_runtime_path(Path(claude_path).parent, install_dir)
                if transport.is_local
                else f'{Path(claude_path).parent}:{install_dir}:{transport.agent_env().get("PATH", "")}'
            ),
            'IS_SANDBOX': '1',
            # Claude Code >= 2.1.196 enables an event-level streaming idle watchdog for
            # all providers: it aborts and retries a stream that emits no *events* for
            # 5 minutes. Our proxy is non-streaming toward the backend, so while waiting
            # it can only emit SSE *comments* (`: keepalive`) — those keep the byte-level
            # timers happy but are dropped by the SDK decoder, so the watchdog sees zero
            # events and kills any turn that takes >300s. Disable it by default; callers
            # can still re-enable via extra_env (applied after this dict).
            'CLAUDE_ENABLE_STREAM_WATCHDOG': '0',
        }
        if extra_env:
            claude_env.update(extra_env)

        claude_env = transport.agent_env(claude_env)
        extra_cli_args = list(extra_cli_args or [])
        pre_start_scripts = [dict(script) for script in pre_start_scripts or []]
        for source in uploaded_files or []:
            destination = home_path / '.local/share/nexus/runs' / invocation_id / Path(source).name
            await transport.deliver_file(source, destination)
            extra_cli_args = [arg.replace(source, str(destination)) for arg in extra_cli_args]
            for script in pre_start_scripts:
                script['code'] = script.get('code', '').replace(source, str(destination))

        # 3) Generate config in the Worker, write it through the Agent account.
        for patch in config_file_patches or []:
            if transport.is_local:
                _apply_json_patch(patch['path'], patch.get('content') or {}, home=home)
            else:
                raw = patch['path']
                path = home_path / raw[2:] if raw.startswith('~/') else Path(raw)
                if not path.is_absolute():
                    path = work_path / path
                content = await transport.read_text(path)
                try:
                    existing = json.loads(content) if content else {}
                except ValueError:
                    existing = {}
                merged = _deep_merge(existing or {}, patch.get('content') or {})
                await transport.write_text(path, json.dumps(merged, indent=2, ensure_ascii=False))

        # 4) Pre-start scripts (before MCP registration so they can prep auth / files).
        runtime_pre_start_scripts = list(pre_start_scripts or [])
        if skills_dir:
            runtime_pre_start_scripts.append({
                'interpreter': 'bash',
                'code': (
                    'mkdir -p "$HOME/.claude/skills" && '
                    f'cp -R {shlex.quote(skills_dir.rstrip("/") + "/.")} "$HOME/.claude/skills/"'
                ),
                'fail_on_error': True,
            })
        for script in runtime_pre_start_scripts:
            try:
                await _run_pre_start_script(
                    script,
                    working_dir,
                    claude_env,
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

        # 5) Add task-provided MCP servers without replacing image/project config.
        from nexus.extensions.agents.claude_code.mcp import register_mcp_servers

        await register_mcp_servers(
            claude_path,
            working_dir=working_dir,
            claude_env=claude_env,
            mcp_servers=mcp_servers,
            scope='local',
            transport=transport,
            invocation_id=invocation_id,
        )

        # 6) Run the installed binary through the selected transport. Local execution
        # preserves the existing command, cwd, environment, streaming, and cancellation path.
        command = _build_single_run_command(claude_path, extra_cli_args=extra_cli_args)
        async with transport.prepare(
            command,
            working_dir=working_dir,
            invocation_id=invocation_id,
            env=claude_env,
        ) as prepared:
            logger.debug('Running Claude Code: %s', ' '.join(shlex.quote(c) for c in prepared.command))
            exit_code, stdout, stderr = await _run_claude_streaming(
                cmd=prepared.command,
                cwd=prepared.cwd,
                env=prepared.env,
                timeout=timeout,
                stdin_input=task,
            )

        execution_time = time.time() - start_time
        logger.info(f'Claude Code finished with exit code {exit_code} in {execution_time:.2f}s')
        failure_kind, failure_stage = transport.classify_failure(exit_code, stderr)

        return {
            'success': exit_code == 0,
            'output': stdout,
            'stderr': stderr,
            'exit_code': exit_code,
            'execution_time': execution_time,
            **(
                {
                    'failure_kind': failure_kind,
                    'failure_stage': failure_stage,
                }
                if failure_kind is not None
                else {}
            ),
        }
    except ExecutionTransportError as exc:
        logger.error('Claude execution transport failed: %s', exc)
        return {
            'success': False,
            'output': '',
            'stderr': str(exc),
            'exit_code': -1,
            'execution_time': time.time() - start_time,
            'failure_kind': 'infrastructure',
            'failure_stage': 'execution_transport',
        }
    finally:
        logger.debug('Stopping Anthropic Proxy...')
        await proxy.stop()


__all__ = ['setup_and_run_claude_code']
