"""Runtime-side Claude Code stream-json session manager."""

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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nexus.extensions.agents.claude_code.runtime_setup import (
    DEFAULT_INSTALL_DIR,
    _apply_json_patch,
    _run_pre_start_script,
)
from nexus.extensions.agents.claude_code.stream_json import (
    encode_user_message,
    parse_stream_line,
    summarize_turn_events,
)
from nexus.extensions.agents.installed_common import (
    DEFAULT_CLI_STREAM_READER_LIMIT,
    DEFAULT_NPM_REGISTRY,
    DEFAULT_PROXY_API_KEY,
    build_runtime_path,
    find_free_port,
)
from nexus.extensions.agents.installed_common.stream_trace import (
    flush_stream_trace_events,
    record_stream_trace_event,
)
from nexus.utils.common import format_exception, get_process_returncode, terminate_process


logger = logging.getLogger(__name__)

_SESSIONS: dict[str, _ClaudeCodeSession] = {}
_ASK_USER_HOOK_CALLBACK_ID = 'nexus_ask_user_question'
_EXIT_PLAN_HOOK_CALLBACK_ID = 'nexus_exit_plan_mode'


@dataclass
class _ClaudeCodeSession:
    session_id: str
    proc: asyncio.subprocess.Process
    proxy: Any
    cwd: str
    started_at: float
    stdout_task: asyncio.Task | None = None
    stderr_task: asyncio.Task | None = None
    events: asyncio.Queue[dict[str, Any]] = field(default_factory=asyncio.Queue)
    control_responses: dict[str, asyncio.Future[dict[str, Any]]] = field(default_factory=dict)
    stderr_lines: list[str] = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    record_trace_events: bool = True

    @property
    def is_running(self) -> bool:
        return self.proc.returncode is None

    async def send_user_message(
        self,
        prompt: str,
        *,
        timeout: float,
        parent_tool_use_id: str | None = None,
    ) -> dict[str, Any]:
        async with self.lock:
            if not self.is_running:
                raise RuntimeError(f'Claude Code session {self.session_id} is not running')
            if self.proc.stdin is None:
                raise RuntimeError(f'Claude Code session {self.session_id} has no stdin')

            self.proc.stdin.write(
                encode_user_message(prompt, session_id=self.session_id, parent_tool_use_id=parent_tool_use_id)
            )
            await self.proc.stdin.drain()

            start = time.time()
            events: list[dict[str, Any]] = []
            while True:
                remaining = timeout - (time.time() - start)
                if remaining <= 0:
                    raise TimeoutError(f'Claude Code session {self.session_id} timed out waiting for result')

                event = await asyncio.wait_for(self.events.get(), timeout=remaining)
                event_type = event.get('type')
                if event_type == '_eof':
                    stderr = '\n'.join(self.stderr_lines)
                    raise RuntimeError(
                        f'Claude Code session {self.session_id} exited before result '
                        f'(return_code={event.get("return_code")}): {stderr}'
                    )
                if event_type == '_protocol_error':
                    stderr = '\n'.join(self.stderr_lines)
                    raise RuntimeError(
                        f'Claude Code session {self.session_id} emitted invalid stream-json: '
                        f'{event.get("error")}: {stderr}'
                    )
                if event_type in {'_stdout_read_error', '_stderr_read_error', '_control_write_error'}:
                    stderr = '\n'.join(self.stderr_lines)
                    detail = str(event.get('error') or 'unknown error')
                    error_name = {
                        '_stdout_read_error': 'stdout reader',
                        '_stderr_read_error': 'stderr reader',
                        '_control_write_error': 'control response write',
                    }[event_type]
                    raise RuntimeError(
                        f'Claude Code session {self.session_id} {error_name} failed: {detail}'
                        + (f': {stderr}' if stderr else '')
                    )

                events.append(event)
                if event_type == 'result':
                    # Match Claude Agent SDK receive_response(): a wire-level
                    # result frame completes the current response/turn, while
                    # the stdio process can stay open for the next user input.
                    summary = summarize_turn_events(events)
                    output = summary['assistant_text']
                    return {
                        'success': not summary['is_error'],
                        'session_id': summary['session_id'] or self.session_id,
                        'output': output,
                        'stderr': '\n'.join(self.stderr_lines),
                        'exit_code': self.proc.returncode or 0,
                        'execution_time': time.time() - start,
                        'events': events,
                        'result_event': summary['result_event'],
                    }

    async def send_control_request(self, request: dict[str, Any], *, timeout: float = 60.0) -> dict[str, Any]:
        if self.proc.stdin is None:
            raise RuntimeError(f'Claude Code session {self.session_id} has no stdin')

        request_id = f'req_{uuid.uuid4().hex}'
        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        self.control_responses[request_id] = future
        payload = {
            'type': 'control_request',
            'request_id': request_id,
            'request': request,
        }
        self.proc.stdin.write((json_dumps(payload) + '\n').encode('utf-8'))
        await self.proc.stdin.drain()

        try:
            event = await asyncio.wait_for(future, timeout=timeout)
        finally:
            self.control_responses.pop(request_id, None)

        response = event.get('response') or {}
        if response.get('subtype') == 'error':
            raise RuntimeError(str(response.get('error') or 'Claude Code control request failed'))
        data = response.get('response')
        return data if isinstance(data, dict) else {}

    async def initialize(self, *, hooks: dict[str, Any] | None = None, timeout: float = 60.0) -> dict[str, Any]:
        return await self.send_control_request({'subtype': 'initialize', 'hooks': hooks}, timeout=timeout)

    async def close(self) -> dict[str, Any]:
        if self.proc.stdin:
            with contextlib.suppress(Exception):
                self.proc.stdin.close()
                await self.proc.stdin.wait_closed()

        if self.proc.returncode is None:
            try:
                await asyncio.wait_for(self.proc.wait(), timeout=2)
            except asyncio.TimeoutError:
                self.proc.terminate()
                try:
                    await asyncio.wait_for(self.proc.wait(), timeout=5)
                except asyncio.TimeoutError:
                    self.proc.kill()
                    with contextlib.suppress(Exception):
                        await self.proc.wait()

        for task in (self.stdout_task, self.stderr_task):
            if task is None:
                continue
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        with contextlib.suppress(Exception):
            await self.proxy.stop()

        return {
            'success': True,
            'session_id': self.session_id,
            'exit_code': self.proc.returncode if self.proc.returncode is not None else 0,
            'stderr': '\n'.join(self.stderr_lines),
        }


async def _read_stdout(session: _ClaudeCodeSession) -> None:
    assert session.proc.stdout is not None
    try:
        while True:
            line = await session.proc.stdout.readline()
            if not line:
                break
            text = line.decode('utf-8', errors='replace').rstrip('\n')
            logger.debug('[claude:%s] %s', session.session_id, text)
            try:
                event = parse_stream_line(line)
            except Exception as exc:
                await session.events.put({'type': '_protocol_error', 'error': str(exc), 'line': text})
                continue

            event_type = event.get('type')
            if event_type == 'control_response':
                response = event.get('response') or {}
                request_id = response.get('request_id')
                if not isinstance(request_id, str):
                    continue
                future = session.control_responses.get(request_id)
                if future is not None and not future.done():
                    future.set_result(event)
                continue
            if event_type == 'control_request':
                asyncio.create_task(_handle_control_request(session, event))
                continue
            if event_type == 'control_cancel_request':
                continue
            if event_type == 'transcript_mirror':
                continue

            _record_trace_event(session, event, enabled=session.record_trace_events)
            await session.events.put(event)
    except Exception as exc:
        error = format_exception(exc)
        session.stderr_lines.append(f'stdout reader failed: {error}')
        logger.exception('[claude:%s] stdout reader failed', session.session_id)
        await session.events.put({'type': '_stdout_read_error', 'error': error})
        await terminate_process(session.proc)
    finally:
        flush_stream_trace_events(session, agent_key='claude_code')
        return_code = await get_process_returncode(session.proc)
        await session.events.put({'type': '_eof', 'return_code': return_code})


async def _read_stderr(session: _ClaudeCodeSession) -> None:
    assert session.proc.stderr is not None
    try:
        while True:
            line = await session.proc.stderr.readline()
            if not line:
                break
            text = line.decode('utf-8', errors='replace').rstrip('\n')
            session.stderr_lines.append(text)
            logger.warning('[claude:%s:err] %s', session.session_id, text)
    except Exception as exc:
        error = format_exception(exc)
        session.stderr_lines.append(f'stderr reader failed: {error}')
        logger.exception('[claude:%s:err] stderr reader failed', session.session_id)
        await session.events.put({'type': '_stderr_read_error', 'error': error})
        await terminate_process(session.proc)


def _record_trace_event(session: _ClaudeCodeSession, event: dict[str, Any], *, enabled: bool = True) -> None:
    """Best-effort trace event for processors that consume Claude Code streams."""
    record_stream_trace_event(session, event, agent_key='claude_code', enabled=enabled)


async def _handle_control_request(session: _ClaudeCodeSession, event: dict[str, Any]) -> None:
    if session.proc.stdin is None:
        return

    request_id = event.get('request_id')
    request = event.get('request') or {}
    subtype = request.get('subtype')
    response: dict[str, Any]
    try:
        if subtype == 'can_use_tool':
            response_data = await _handle_can_use_tool(session, request)
            response = {
                'type': 'control_response',
                'response': {
                    'subtype': 'success',
                    'request_id': request_id,
                    'response': response_data,
                },
            }
        elif subtype == 'hook_callback':
            response = {
                'type': 'control_response',
                'response': {
                    'subtype': 'success',
                    'request_id': request_id,
                    'response': await _handle_hook_callback(session, request),
                },
            }
        else:
            response = {
                'type': 'control_response',
                'response': {
                    'subtype': 'error',
                    'request_id': request_id,
                    'error': f'Unsupported control request subtype: {subtype}',
                },
            }
    except Exception as exc:
        logger.exception('Claude Code control request failed: %s', exc)
        response = {
            'type': 'control_response',
            'response': {
                'subtype': 'error',
                'request_id': request_id,
                'error': str(exc),
            },
        }
    try:
        session.proc.stdin.write((json_dumps(response) + '\n').encode('utf-8'))
        await session.proc.stdin.drain()
    except Exception as exc:
        error = format_exception(exc)
        session.stderr_lines.append(f'control response write failed: {error}')
        logger.exception('Claude Code control response write failed: %s', exc)
        await session.events.put({'type': '_control_write_error', 'error': error})
        await terminate_process(session.proc)


async def _handle_can_use_tool(_session: _ClaudeCodeSession, request: dict[str, Any]) -> dict[str, Any]:
    raw_tool_input = request.get('input')
    tool_input: dict[str, Any] = raw_tool_input if isinstance(raw_tool_input, dict) else {}

    return {
        'behavior': 'allow',
        'updatedInput': tool_input,
    }


def _build_interaction_hint(tool_name: str, tool_input: dict[str, Any]) -> str:
    if tool_name == 'AskUserQuestion':
        lines = ['Claude Code asks for user input.']
        questions = tool_input.get('questions') if isinstance(tool_input, dict) else None
        if isinstance(questions, list):
            for index, question in enumerate(questions, start=1):
                if not isinstance(question, dict):
                    continue
                header = question.get('header')
                prompt = question.get('question')
                prefix = f'{index}. '
                if header:
                    prefix += f'{header}: '
                lines.append(prefix + str(prompt or ''))
                options = question.get('options')
                if isinstance(options, list):
                    labels = [str(option.get('label')) for option in options if isinstance(option, dict)]
                    if labels:
                        lines.append('Options: ' + ', '.join(labels))
        return '\n'.join(lines)
    if tool_name == 'ExitPlanMode':
        plan = tool_input.get('plan') if isinstance(tool_input, dict) else None
        return 'Claude Code wants to exit plan mode with this plan:\n' + str(plan or '')
    return f'Claude Code requests user input for {tool_name}.'


def _build_hook_output(
    _tool_name: str,
    _answer: str | None,
    _tool_input: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        'hookSpecificOutput': {
            'hookEventName': 'PreToolUse',
            'permissionDecision': 'defer',
            'permissionDecisionReason': 'Nexus will collect the user response and continue the session.',
        }
    }


def _build_user_input_hooks() -> dict[str, Any]:
    return {
        'PreToolUse': [
            {
                'matcher': 'ExitPlanMode',
                'hookCallbackIds': [_EXIT_PLAN_HOOK_CALLBACK_ID],
            },
            {
                'matcher': 'AskUserQuestion',
                'hookCallbackIds': [_ASK_USER_HOOK_CALLBACK_ID],
            },
        ],
    }


def _tool_name_from_hook_callback(callback_id: str) -> str:
    if callback_id == _ASK_USER_HOOK_CALLBACK_ID:
        return 'AskUserQuestion'
    if callback_id == _EXIT_PLAN_HOOK_CALLBACK_ID:
        return 'ExitPlanMode'
    return callback_id


def _extract_hook_tool_input(hook_input: Any) -> dict[str, Any]:
    if not isinstance(hook_input, dict):
        return {}
    for key in ('tool_input', 'toolInput', 'input'):
        value = hook_input.get(key)
        if isinstance(value, dict):
            return value
    return {}


async def _handle_hook_callback(_session: _ClaudeCodeSession, request: dict[str, Any]) -> dict[str, Any]:
    callback_id = str(request.get('callback_id') or '')
    tool_name = _tool_name_from_hook_callback(callback_id)
    tool_input = _extract_hook_tool_input(request.get('input'))
    return _build_hook_output(tool_name, None, tool_input)


def json_dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(',', ':'))


def _build_session_command(
    claude_path: str | Path,
    *,
    session_id: str,
    extra_cli_args: list[str] | None,
    include_partial_messages: bool,
    include_hook_events: bool,
) -> list[str]:
    extra_args = list(extra_cli_args or [])
    resume_or_continue = any(
        arg in {'--resume', '-r', '--continue', '-c'} or arg.startswith('--resume=') for arg in extra_args
    )
    fork_session = any(arg == '--fork-session' for arg in extra_args)
    cmd = [
        str(claude_path),
        '--output-format',
        'stream-json',
        '--verbose',
    ]
    if not resume_or_continue or fork_session:
        cmd.extend(['--session-id', session_id])
    cmd.append('--dangerously-skip-permissions')
    if include_partial_messages:
        cmd.append('--include-partial-messages')
    if include_hook_events:
        cmd.append('--include-hook-events')
    if extra_args:
        cmd.extend(extra_args)
    cmd.extend(['--input-format', 'stream-json'])
    return cmd


async def _register_mcp_servers(
    claude_path: str | Path,
    *,
    working_dir: str,
    claude_env: dict[str, str],
    mcp_servers: list[dict[str, Any]] | None,
) -> None:
    from nexus.extensions.agents.claude_code.mcp import register_mcp_servers

    await register_mcp_servers(
        claude_path,
        working_dir=working_dir,
        claude_env=claude_env,
        mcp_servers=mcp_servers,
    )


async def _start_session(
    *,
    session_id: str,
    working_dir: str,
    timeout: float,
    version: str | None,
    npm_registry: str | None,
    binary_url: str | None,
    binary_checksum: str | None,
    mcp_servers: list[dict[str, Any]] | None,
    extra_env: dict[str, str] | None,
    extra_cli_args: list[str] | None,
    config_file_patches: list[dict[str, Any]] | None,
    pre_start_scripts: list[dict[str, Any]] | None,
    conversion_log_level: str,
    thinking_preset: str,
    include_partial_messages: bool,
    include_hook_events: bool,
    enable_user_input_hooks: bool,
    record_trace_events: bool,
) -> _ClaudeCodeSession:
    from nexus.extensions.agents.claude_code.installer import (
        install_claude_code_from_url,
        install_claude_code_via_npm,
    )
    from nexus.runtime.context import require_runtime_control
    from nexus.runtime.server.services.anthropic_proxy import AnthropicProxy
    from nexus.utils.shell import get_os_env_without_ld_library_path

    rt = require_runtime_control()

    if binary_url:
        claude_path = await install_claude_code_from_url(
            url=binary_url,
            install_dir=DEFAULT_INSTALL_DIR,
            checksum=binary_checksum,
        )
    else:
        claude_path = await install_claude_code_via_npm(
            install_dir=DEFAULT_INSTALL_DIR,
            version=version,
            npm_registry=npm_registry,
        )

    proxy_port = find_free_port()
    proxy = AnthropicProxy(
        sio=rt.sio,
        listen_port=proxy_port,
        listen_host='127.0.0.1',
        request_timeout=timeout,
        conversion_log_level=conversion_log_level,
        thinking_config=thinking_preset,
    )
    await proxy.start()

    session: _ClaudeCodeSession | None = None
    try:
        work_path = Path(working_dir)
        work_path.mkdir(parents=True, exist_ok=True)

        home = os.environ.get('HOME', '/root')
        claude_env: dict[str, str] = {
            'ANTHROPIC_BASE_URL': f'http://127.0.0.1:{proxy_port}',
            'ANTHROPIC_API_KEY': DEFAULT_PROXY_API_KEY,
            'HOME': home,
            'PATH': build_runtime_path(Path(claude_path).parent, DEFAULT_INSTALL_DIR),
            'IS_SANDBOX': '1',
            # See runtime_setup.py: CC >= 2.1.196's event-level stream watchdog aborts
            # turns >300s because our keepalive SSE comments carry no decodable events.
            'CLAUDE_ENABLE_STREAM_WATCHDOG': '0',
            'CLAUDE_CODE_ENTRYPOINT': 'nexus-stream-json',
            'PWD': str(work_path),
        }
        if extra_env:
            claude_env.update(extra_env)

        for patch in config_file_patches or []:
            _apply_json_patch(patch['path'], patch.get('content') or {})

        for script in pre_start_scripts or []:
            await _run_pre_start_script(script, working_dir, claude_env)

        await _register_mcp_servers(
            claude_path,
            working_dir=working_dir,
            claude_env=claude_env,
            mcp_servers=mcp_servers,
        )

        base_env = get_os_env_without_ld_library_path()
        base_env['BASH_ENV'] = '~/.bashrc'
        merged_env = {**base_env, **claude_env}
        cmd = _build_session_command(
            claude_path,
            session_id=session_id,
            extra_cli_args=extra_cli_args,
            include_partial_messages=include_partial_messages,
            include_hook_events=include_hook_events,
        )
        logger.debug('Starting Claude Code session %s: %s', session_id, ' '.join(shlex.quote(c) for c in cmd))

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=working_dir,
            env=merged_env,
            limit=DEFAULT_CLI_STREAM_READER_LIMIT,
        )
        session = _ClaudeCodeSession(
            session_id=session_id,
            proc=proc,
            proxy=proxy,
            cwd=working_dir,
            started_at=time.time(),
            record_trace_events=record_trace_events,
        )
        session.stdout_task = asyncio.create_task(_read_stdout(session))
        session.stderr_task = asyncio.create_task(_read_stderr(session))
        await session.initialize(
            hooks=_build_user_input_hooks() if enable_user_input_hooks else None,
            timeout=min(max(timeout, 1.0), 60.0),
        )
        return session
    except Exception:
        if session is not None:
            with contextlib.suppress(Exception):
                await session.close()
        else:
            with contextlib.suppress(Exception):
                await proxy.stop()
        raise


async def run_claude_code_session_turn(
    *,
    session_id: str,
    prompt: str,
    parent_tool_use_id: str | None = None,
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
    conversion_log_level: str = 'INFO',
    thinking_preset: str = 'hunyuan',
    include_partial_messages: bool = False,
    include_hook_events: bool = False,
    enable_user_input_hooks: bool = False,
    record_trace_events: bool = True,
) -> dict[str, Any]:
    """Start or reuse a Claude Code session and run one user turn."""
    start_time = time.time()
    session = _SESSIONS.get(session_id)
    if session is not None and not session.is_running:
        await close_claude_code_session(session_id=session_id)
        return {
            'success': False,
            'session_id': session_id,
            'output': '',
            'stderr': f'Claude Code session {session_id} exited before the next turn',
            'exit_code': -1,
            'execution_time': time.time() - start_time,
            'events': [],
        }

    try:
        if session is None:
            session = await _start_session(
                session_id=session_id,
                working_dir=working_dir,
                timeout=timeout,
                version=version,
                npm_registry=npm_registry,
                binary_url=binary_url,
                binary_checksum=binary_checksum,
                mcp_servers=mcp_servers,
                extra_env=extra_env,
                extra_cli_args=extra_cli_args,
                config_file_patches=config_file_patches,
                pre_start_scripts=pre_start_scripts,
                conversion_log_level=conversion_log_level,
                thinking_preset=thinking_preset,
                include_partial_messages=include_partial_messages,
                include_hook_events=include_hook_events,
                enable_user_input_hooks=enable_user_input_hooks,
                record_trace_events=record_trace_events,
            )
            _SESSIONS[session_id] = session
        return await session.send_user_message(prompt, timeout=timeout, parent_tool_use_id=parent_tool_use_id)
    except Exception as exc:
        logger.exception('Claude Code session turn failed: %s', exc)
        stderr = ''
        if session is not None:
            stderr = '\n'.join(session.stderr_lines)
        with contextlib.suppress(Exception):
            await close_claude_code_session(session_id=session_id)
        return {
            'success': False,
            'session_id': session_id,
            'output': '',
            'stderr': '\n'.join(part for part in (str(exc), stderr) if part),
            'exit_code': -1,
            'execution_time': time.time() - start_time,
            'events': [],
        }
    finally:
        if session is not None:
            flush_stream_trace_events(session, agent_key='claude_code')


async def close_claude_code_session(*, session_id: str) -> dict[str, Any]:
    """Close one runtime-side Claude Code session."""
    session = _SESSIONS.pop(session_id, None)
    if session is None:
        return {'success': True, 'session_id': session_id, 'exit_code': 0, 'stderr': ''}
    return await session.close()


async def close_all_claude_code_sessions() -> dict[str, Any]:
    """Close all runtime-side Claude Code sessions in this runtime process."""
    session_ids = list(_SESSIONS)
    results = [await close_claude_code_session(session_id=session_id) for session_id in session_ids]
    return {'success': True, 'closed': results}


__all__ = [
    'close_all_claude_code_sessions',
    'close_claude_code_session',
    'run_claude_code_session_turn',
]
