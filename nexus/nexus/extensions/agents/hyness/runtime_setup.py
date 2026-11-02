"""Launch Hyness with the handoff execution protocol and existing Nexus services."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import time
import uuid

from pathlib import Path
from typing import Any, Literal

import yaml

from nexus.extensions.agents.hycode.runtime_setup import _deep_merge, _parse_last_result, _run_pre_start_script
from nexus.extensions.agents.installed_common import DEFAULT_PROXY_API_KEY, build_runtime_path, find_free_port
from nexus.extensions.agents.installed_common.execution import AgentExecutionTransport
from nexus.runtime.executors.command import active_command_timeout
from nexus.utils.shell import get_clean_sandbox_env

from .installer import install_hyness, resolve_app_config, verify_app_engine
from .protocol import convert_execution_output
from .proxy import budget_translation


_ROUTING_ENV = {
    'ANTHROPIC_AUTH_TOKEN',
    'CLAUDE_CODE_OAUTH_TOKEN',
    'ANTHROPIC_API_PROTOCOL',
    'CLAUDE_CODE_USE_BEDROCK',
    'CLAUDE_CODE_USE_VERTEX',
    'CLAUDE_CODE_USE_FOUNDRY',
    'HYNESS_APP_CONFIG',
    'HYNESS_DEFAULT_APP_CONFIG',
    'HYNESS_APP_CONFIG_OVERLAY',
    'HYNESS_HARNESS_ID',
    'HYNESS_DISTRIBUTION_ROOT',
    'HYNESS_PLUGIN_REGISTRY',
}
_RESERVED_FLAGS = {'--', '--app-config', '--app-config-overlay', '--result-protocol', '--resume', '--continue'}


def normalize_hycode_cli_args(
    args: list[str],
    max_turns: int | None,
    stop_on_tool_error: bool,
    input_format: Literal['text', 'stream-json'] = 'text',
) -> tuple[list[str], int | None, bool]:
    """Accept the old headless flags while retaining one authoritative result protocol."""
    extra: list[str] = []
    turns = None
    index = 0
    while index < len(args):
        flag, separator, inline = args[index].partition('=')
        if flag in _RESERVED_FLAGS:
            raise ValueError(f'Nexus controls the Hyness launch flag {flag}')
        if flag in {'-p', '--print', '--verbose', '--dangerously-skip-permissions', '--no-session-persistence'}:
            index += 1
            continue
        if flag == '--stop-on-tool-error':
            stop_on_tool_error = True
            index += 1
            continue
        if flag in {'--max-turns', '--output-format', '--input-format'}:
            if separator:
                value = inline
            else:
                index += 1
                if index >= len(args):
                    raise ValueError(f'Missing value for {flag}')
                value = args[index]
            if flag == '--max-turns':
                turns = int(value)
                if turns < 0:
                    raise ValueError('max_turns must be nonnegative')
            elif value != {'--output-format': 'stream-json', '--input-format': input_format}[flag]:
                raise ValueError(f'Unsupported Nexus Hyness {flag}: {value}')
        else:
            extra.append(args[index])
        index += 1
    return extra, max_turns if max_turns is not None else turns, stop_on_tool_error


async def _capture_process(
    cmd: list[str],
    cwd: str,
    env: dict[str, str],
    timeout: float,
    timeout_control_id: str | None,
    *,
    stdin_input: str | None = None,
) -> tuple[int, str, str, bool]:
    base_env = get_clean_sandbox_env()
    for name in list(base_env):
        if name in _ROUTING_ENV or name.startswith(('HYNESS_SUBAGENT_', 'ANTHROPIC_SMALL_FAST_', 'ANTHROPIC_DEFAULT_')):
            base_env.pop(name, None)
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=cwd,
        env={**base_env, **env},
        stdin=asyncio.subprocess.PIPE if stdin_input is not None else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    capture = asyncio.create_task(proc.communicate(stdin_input.encode('utf-8') if stdin_input is not None else None))
    timed_out = False

    async def cleanup() -> None:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGTERM)
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(asyncio.shield(capture), 3)
        # A detached tool can close its pipes before it exits. Reap the group
        # even when communicate() has already returned.
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
        await capture

    try:
        async with active_command_timeout(timeout, timeout_control_id):
            await asyncio.shield(capture)
    except TimeoutError:
        timed_out = True
    finally:
        cleanup_task = asyncio.create_task(cleanup())
        try:
            await asyncio.shield(cleanup_task)
        except asyncio.CancelledError:
            await cleanup_task
            raise
    stdout, stderr = capture.result()
    return (
        (-1 if timed_out else int(proc.returncode or 0)),
        stdout.decode(errors='replace'),
        stderr.decode(errors='replace'),
        timed_out,
    )


async def setup_and_run_hyness(
    task: str,
    packages: list[str],
    app_config: str,
    preserve_tool_schema: bool = False,
    app_config_overlay: str | None = None,
    harness_id: str | None = None,
    working_dir: str = '/workspace',
    timeout: float = 3600,
    npm_registry: str | None = None,
    extra_env: dict[str, str] | None = None,
    extra_cli_args: list[str] | None = None,
    max_turns: int | None = None,
    stop_on_tool_error: bool = False,
    session_persistence: bool = False,
    mcp_config_servers: dict[str, Any] | None = None,
    config_file_patches: list[dict[str, Any]] | None = None,
    pre_start_scripts: list[dict[str, Any]] | None = None,
    thinking_preset: str = 'hunyuan',
    conversion_log_level: str = 'INFO',
    execution_target: dict[str, Any] | None = None,
    input_format: Literal['text', 'stream-json'] = 'text',
) -> dict[str, Any]:
    from nexus.runtime.context import require_runtime_control
    from nexus.runtime.server.services.anthropic_proxy import AnthropicProxy
    from nexus.trace import get_current_span_id

    class HynessProxy(AnthropicProxy):
        def create_app(self):
            return budget_translation(super().create_app())

    started = time.monotonic()
    transport = AgentExecutionTransport(execution_target)
    proxy = None
    session_dir = None
    raw_code = None
    raw_output = None
    raw_stderr = None
    stage = 'install'
    try:
        installed = await install_hyness(transport, packages, npm_registry)
        app = await resolve_app_config(transport, installed, app_config)
        overlay = await resolve_app_config(transport, installed, app_config_overlay)
        assert app is not None
        await verify_app_engine(transport, installed, app)
        stage = 'proxy'
        proxy = HynessProxy(
            sio=require_runtime_control().sio,
            listen_port=find_free_port(),
            request_timeout=None,
            conversion_log_level=conversion_log_level,
            thinking_config=thinking_preset,
            preserve_tool_schema=preserve_tool_schema,
        )
        await proxy.start()
        invocation = uuid.uuid4().hex
        home = str(transport.home_dir(invocation_id=invocation, default=os.environ.get('HOME', '/root')))
        session_dir = str(Path(home) / '.hyness' / 'nexus' / invocation)
        await transport.mkdir(session_dir)
        env = {
            key: value
            for key, value in (extra_env or {}).items()
            if key not in _ROUTING_ENV and not key.startswith(('HYNESS_SUBAGENT_', 'ANTHROPIC_SMALL_FAST_'))
        }
        task_path = env.get('PATH', transport.agent_env().get('PATH', build_runtime_path()))
        env.update({
            'HOME': home,
            'CLAUDE_CONFIG_DIR': session_dir,
            'BASH_ENV': '/dev/null',
            'HYNESS_PLUGIN_REGISTRY': str(Path(session_dir) / 'plugins.json'),
            'ANTHROPIC_API_KEY': DEFAULT_PROXY_API_KEY,
            'ANTHROPIC_BASE_URL': f'http://127.0.0.1:{proxy.port}',
            'NEXUS_AGENT_PROXY_URL': f'http://127.0.0.1:{proxy.port}',
            'NEXUS_HYNESS_APP_CONFIG': app,
            'IS_SANDBOX': '1',
            'PATH': f'{Path(installed["node"]).parent}:{task_path}',
        })
        if harness_id is not None:
            env['HYNESS_HARNESS_ID'] = harness_id
        env.setdefault('ANTHROPIC_MODEL', 'claude-opus-4-6')
        env.setdefault('CLAUDE_CODE_ATTRIBUTION_HEADER', '0')
        env.setdefault('CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC', '1')
        env.setdefault('USE_BUILTIN_RIPGREP', '0')
        extra, max_turns, stop_on_tool_error = normalize_hycode_cli_args(
            extra_cli_args or [], max_turns, stop_on_tool_error, input_format
        )
        if stop_on_tool_error:
            content = yaml.safe_load(await transport.read_text(overlay) or '{}') if overlay else {}
            content.setdefault('runtime', {})['stopOnToolError'] = True
            overlay = str(Path(session_dir) / 'training.yaml')
            await transport.write_text(overlay, yaml.safe_dump(content))
        stage = 'pre_start'
        for patch in config_file_patches or []:
            path = patch['path']
            for prefix in ['~/.openclaude/', '~/.claude/', '~/.hyness/']:
                if path.startswith(prefix):
                    path = str(Path(session_dir) / path[len(prefix) :])
                    break
            else:
                path = str(Path(home) / path[2:]) if path.startswith('~/') else str(Path(working_dir) / path)
            previous = json.loads(await transport.read_text(path) or '{}')
            await transport.write_text(path, json.dumps(_deep_merge(previous, patch['content'])))
        for script in pre_start_scripts or []:
            await _run_pre_start_script(script, working_dir, env, transport=transport, invocation_id=invocation)
        command = [
            installed['node'],
            installed['entry'],
            '--app-config',
            app,
            '--result-protocol',
            'execution',
            '--print',
            '--input-format',
            input_format,
            '--output-format',
            'stream-json',
            '--verbose',
        ]
        if not session_persistence:
            command.append('--no-session-persistence')
        command.append('--dangerously-skip-permissions')
        if overlay:
            command.extend(['--app-config-overlay', overlay])
        if max_turns is not None:
            command.extend(['--max-turns', str(max_turns)])
        if mcp_config_servers:
            command.extend(['--mcp-config', json.dumps({'mcpServers': mcp_config_servers}), '--strict-mcp-config'])
        # Text and encoded SDK input stay on stdin, outside OS argument-size limits.
        command.extend([*extra, '--'])
        stage = 'process'
        async with transport.prepare(command, working_dir=working_dir, invocation_id=invocation, env=env) as prepared:
            code, stdout, stderr, timed_out = await _capture_process(
                prepared.command,
                prepared.cwd,
                prepared.env,
                timeout,
                get_current_span_id(),
                stdin_input=task,
            )
        raw_code, raw_output, raw_stderr = code, stdout, stderr
        terminal = None
        if not timed_out and code in (0, 1):
            stage = 'runtime_protocol'
            code, stdout, terminal = convert_execution_output(stdout, code)
        kind, failure_stage = transport.classify_failure(code, stderr)
        return {
            'success': code == 0,
            'exit_code': code,
            'output': stdout,
            'stderr': stderr,
            'timed_out': timed_out,
            'process_exit_code': raw_code,
            'process_output': raw_output,
            'process_stderr': raw_stderr,
            'execution_termination': terminal,
            'execution_time': time.monotonic() - started,
            'failure_kind': kind,
            'failure_stage': failure_stage,
            **_parse_last_result(stdout),
        }
    except Exception as error:
        return {
            'success': False,
            'exit_code': -1,
            'output': '',
            'stderr': '\n'.join(part for part in (raw_stderr, str(error)) if part),
            'failure_kind': 'infrastructure',
            'failure_stage': stage,
            'execution_time': time.monotonic() - started,
            'process_exit_code': raw_code,
            'process_output': raw_output,
            'process_stderr': raw_stderr,
        }
    finally:
        if proxy is not None:
            await proxy.stop()
        if session_dir is not None:
            await transport.exec_as_agent(
                [
                    transport.python_binary,
                    '-c',
                    'import shutil,sys;shutil.rmtree(sys.argv[1],ignore_errors=True)',
                    session_dir,
                ],
                timeout=30,
                check=True,
            )
