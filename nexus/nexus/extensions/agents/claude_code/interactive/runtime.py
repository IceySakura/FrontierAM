"""Worker-side glue for the Claude Code interactive agent.

Runs inside the runtime sandbox worker. Responsibilities:
  1. install Claude Code (latest) — reuses claude_code.installer (stable helper),
  2. start the AnthropicProxy (LLM routing over ``rt.sio`` — nexus-owned),
  3. write a Stop-hook settings.json so Claude signals per-turn completion,
  4. drive the framework-agnostic ``ClaudeInteractiveSession`` in-process,
  5. keep the session + proxy alive across turns in ``_PTY_SESSIONS`` so the same
     PTY serves multiple user messages, and tear both down on close.

The PTY mechanics themselves live entirely in ``session.py`` (framework
agnostic, swappable for a node-pty implementation). This module is the only
part that touches nexus internals, and it wraps the framework-free session in
``_InteractiveRuntimeSession`` so the session never holds a proxy handle.

Per-turn completion is isolated per session via environment variables injected
into each session's Claude process (``NEXUS_PTY_DONE_FILE`` /
``NEXUS_PTY_FINAL_FILE`` / ``NEXUS_PTY_TURN_FILE`` / ``NEXUS_PTY_SESSION_ID``).
The Stop hook (a static command in the shared settings.json) reads those env
vars — which it inherits from the triggering Claude process — so concurrent
sessions never cross wires, and emits a structured JSON completion event
carrying the ``turn_id``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import os
import re
import shlex
import shutil
import sys
import time

from dataclasses import dataclass, field
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

from .hooks import (
    DEFAULT_HOOK_COMMAND_TIMEOUT,
    DEFAULT_HOOK_WAIT_TIMEOUT,
    FileHookInteractionBridge,
)
from .session import ClaudeInteractiveSession


logger = logging.getLogger(__name__)

DEFAULT_INSTALL_DIR = f'{DEFAULT_AGENT_INSTALL_ROOT}/claude-code'

# Bypass per-tool permission prompts so tool-using turns run unattended in the
# sandbox (parity with the claude_code / claude_code_multiturn agents).
_SKIP_PERMISSIONS_FLAG = '--dangerously-skip-permissions'

# Runtime-side session registry. Keyed by nexus session_id, persists across the
# per-turn ``call_runtime_agent`` invocations that run in this worker process.
_PTY_SESSIONS: dict[str, _InteractiveRuntimeSession] = {}
# Serialises first-create per session so two concurrent turns can't both install
# Claude / start a proxy for the same session_id.
_CREATE_LOCK = asyncio.Lock()


@dataclass
class _InteractiveRuntimeSession:
    """Nexus-side wrapper around a framework-free ClaudeInteractiveSession.

    Holds the nexus proxy alongside the PTY session (the session itself stays
    proxy-agnostic) plus a per-session lock so turns on one session serialise.
    """

    pty_session: ClaudeInteractiveSession
    proxy: Any
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    last_activity: float = 0.0
    closed: bool = False

    async def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        with contextlib.suppress(Exception):
            await self.pty_session.close()
        with contextlib.suppress(Exception):
            await self.proxy.stop()


def _deep_merge(target: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = value
    return target


def _apply_json_patch(raw_path: str, patch: dict[str, Any]) -> None:
    """Deep-merge *patch* into the JSON file at *raw_path*, creating it if missing."""
    path = Path(os.path.expanduser(raw_path)).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, Any] = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text()) or {}
        except Exception as exc:  # noqa: BLE001
            logger.warning('Failed to parse existing %s: %s; overwriting.', path, exc)
            existing = {}
    path.write_text(json.dumps(_deep_merge(existing, patch), indent=2, ensure_ascii=False))
    logger.debug('Patched %s', path)


def _append_required_hooks(raw_path: str, required_settings: dict[str, Any]) -> None:
    """Append Nexus-owned hooks without discarding user-defined hook entries."""
    path = Path(os.path.expanduser(raw_path)).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(path.read_text()) if path.exists() else {}
    if not isinstance(existing, dict):
        existing = {}
    hooks = existing.get('hooks')
    if not isinstance(hooks, dict):
        hooks = {}
    required_hooks = required_settings.get('hooks')
    if not isinstance(required_hooks, dict):
        required_hooks = {}

    for event_name, required_entries in required_hooks.items():
        current_entries = hooks.get(event_name)
        merged_entries = list(current_entries) if isinstance(current_entries, list) else []
        for entry in required_entries if isinstance(required_entries, list) else []:
            if entry not in merged_entries:
                merged_entries.append(entry)
        hooks[event_name] = merged_entries

    existing['hooks'] = hooks
    path.write_text(json.dumps(existing, indent=2, ensure_ascii=False))
    logger.debug('Ensured Nexus hooks in %s', path)


# Stop-hook body: reads the per-session sentinel paths + current turn_id from the
# env it inherits from the Claude process, then writes a structured completion
# event. Static command → one shared settings.json serves every session.
_STOP_HOOK_CODE = (
    'import sys,json,os,pathlib\n'
    'try:\n'
    '    d=json.load(sys.stdin)\n'
    'except Exception:\n'
    '    d={}\n'
    'msg=d.get("last_assistant_message") or ""\n'
    'done=os.environ.get("NEXUS_PTY_DONE_FILE")\n'
    'final=os.environ.get("NEXUS_PTY_FINAL_FILE")\n'
    'turnf=os.environ.get("NEXUS_PTY_TURN_FILE")\n'
    'sid=os.environ.get("NEXUS_PTY_SESSION_ID","")\n'
    'turn_id=0\n'
    'if turnf and os.path.exists(turnf):\n'
    '    try:\n'
    '        turn_id=int((pathlib.Path(turnf).read_text() or "0").strip() or 0)\n'
    '    except Exception:\n'
    '        turn_id=0\n'
    'if final:\n'
    '    pathlib.Path(final).write_text(str(msg), encoding="utf-8")\n'
    'if done:\n'
    '    pathlib.Path(done).write_text('
    'json.dumps({"session_id":sid,"turn_id":turn_id,"last_assistant_message":str(msg)}), encoding="utf-8")\n'
)


def _resolve_hook_python() -> str:
    """Absolute python interpreter for the Stop hook.

    The sandbox image often has no ``python3`` on PATH (Claude ships as a native
    binary, and hooks run via ``/bin/sh`` which then fails with
    ``python3: not found``). ``sys.executable`` is the worker's own interpreter —
    it lives in the same container filesystem the hook runs in, so an absolute
    path always resolves regardless of PATH.
    """
    # Prefer the worker interpreter because the interaction hook executes this
    # package as ``python -m ...hooks``. A system python may exist but not
    # have the uploaded Nexus wheel installed.
    return sys.executable or find_runtime_binary('python3') or 'python3'


def _coerce_hook_timeouts(hook_wait_timeout: float, hook_command_timeout: float) -> tuple[float, float]:
    wait_timeout = float(hook_wait_timeout)
    command_timeout = float(hook_command_timeout)
    if not math.isfinite(wait_timeout) or wait_timeout <= 0:
        raise ValueError(f'hook_wait_timeout must be a positive finite number, got {hook_wait_timeout!r}')
    if not math.isfinite(command_timeout) or command_timeout <= 0:
        raise ValueError(f'hook_command_timeout must be a positive finite number, got {hook_command_timeout!r}')
    if command_timeout <= wait_timeout:
        raise ValueError(
            f'hook_command_timeout must be greater than hook_wait_timeout (got {command_timeout} <= {wait_timeout})'
        )
    return wait_timeout, command_timeout


def _interactive_hook_settings(
    python_bin: str,
    *,
    hook_wait_timeout: float = DEFAULT_HOOK_WAIT_TIMEOUT,
    hook_command_timeout: float = DEFAULT_HOOK_COMMAND_TIMEOUT,
) -> dict[str, Any]:
    """Build static hooks; per-session env decides bridge versus unattended."""
    _, hook_command_timeout = _coerce_hook_timeouts(hook_wait_timeout, hook_command_timeout)
    hook_cmd = shlex.quote(python_bin) + ' -c ' + shlex.quote(_STOP_HOOK_CODE)
    module = 'nexus.extensions.agents.claude_code.interactive.hooks'

    def bridge_command(tool_name: str) -> str:
        return f'{shlex.quote(python_bin)} -m {shlex.quote(module)} {shlex.quote(tool_name)}'

    return {
        'hooks': {
            'Stop': [{'hooks': [{'type': 'command', 'command': hook_cmd}]}],
            'PreToolUse': [
                {
                    'matcher': tool_name,
                    'hooks': [
                        {'type': 'command', 'command': bridge_command(tool_name), 'timeout': hook_command_timeout}
                    ],
                }
                for tool_name in ('ExitPlanMode', 'AskUserQuestion')
            ],
        }
    }


def _session_token(session_id: str) -> str:
    """Filesystem-safe token derived from session_id + pid for sentinel paths."""
    safe = re.sub(r'[^A-Za-z0-9_.-]', '_', session_id)[:64]
    return f'{safe}_{os.getpid()}'


def _ensure_skip_permissions(extra_cli_args: list[str] | None) -> list[str]:
    """Prepend the bypass-permissions flag unless the caller already passed it."""
    args = list(extra_cli_args or [])
    if _SKIP_PERMISSIONS_FLAG not in args:
        args.insert(0, _SKIP_PERMISSIONS_FLAG)
    return args


async def _prepare_runtime_session(
    session_id: str,
    *,
    working_dir: str,
    timeout: float,
    version: str,
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
    enable_input_bridge: bool,
    rows: int,
    cols: int,
    prompt_settle_seconds: float,
    log_frames: bool,
    snapshot_seconds: float | None = None,
    hook_wait_timeout: float = DEFAULT_HOOK_WAIT_TIMEOUT,
    hook_command_timeout: float = DEFAULT_HOOK_COMMAND_TIMEOUT,
) -> _InteractiveRuntimeSession:
    """Install Claude, start the proxy, patch config, and build a session.

    Does NOT spawn Claude — the TUI is launched lazily on the first send_turn so
    the first turn can use the proven ``claude <prompt>`` argv path. On any
    failure the proxy is stopped so nothing leaks.
    """
    from nexus.extensions.agents.claude_code.installer import (
        install_claude_code_from_url,
        install_claude_code_via_npm,
    )
    from nexus.runtime.context import require_runtime_control
    from nexus.utils.shell import get_os_env_without_ld_library_path

    from .proxy import PtyAnthropicProxy

    hook_wait_timeout, hook_command_timeout = _coerce_hook_timeouts(
        hook_wait_timeout,
        hook_command_timeout,
    )
    rt = require_runtime_control()

    # 1) Install Claude Code (bounded so a stalled npm download surfaces clearly).
    if binary_url:
        claude_path = await asyncio.wait_for(
            install_claude_code_from_url(url=binary_url, install_dir=DEFAULT_INSTALL_DIR, checksum=binary_checksum),
            timeout=600,
        )
    else:
        claude_path = await asyncio.wait_for(
            install_claude_code_via_npm(install_dir=DEFAULT_INSTALL_DIR, version=version, npm_registry=npm_registry),
            timeout=600,
        )
    logger.debug('[pty %s] claude binary: %s', session_id, claude_path)

    # 2) Start the Anthropic proxy (LLM routing over rt.sio).
    proxy_port = find_free_port()
    proxy = PtyAnthropicProxy(
        sio=rt.sio,
        listen_port=proxy_port,
        listen_host='127.0.0.1',
        request_timeout=timeout,
        conversion_log_level=conversion_log_level,
        thinking_config=thinking_preset,
    )
    try:
        await asyncio.wait_for(proxy.start(), timeout=60)
    except BaseException:
        with contextlib.suppress(Exception):
            await proxy.stop()
        raise
    logger.debug('[pty %s] proxy on %d', session_id, proxy_port)

    try:
        work_path = Path(working_dir)
        work_path.mkdir(parents=True, exist_ok=True)

        home = os.environ.get('HOME', '/root')
        token = _session_token(session_id)
        done_file = f'/tmp/.cc_pty_{token}.done'
        final_file = f'/tmp/.cc_pty_{token}.final'
        turn_file = f'/tmp/.cc_pty_{token}.turn'
        interaction_dir = f'/tmp/.cc_pty_{token}.interactions'
        for stale in (done_file, final_file, turn_file):
            with contextlib.suppress(OSError):
                os.unlink(stale)
        shutil.rmtree(interaction_dir, ignore_errors=True)

        claude_env: dict[str, str] = {
            'ANTHROPIC_BASE_URL': f'http://127.0.0.1:{proxy_port}',
            'ANTHROPIC_API_KEY': DEFAULT_PROXY_API_KEY,
            'HOME': home,
            'PATH': build_runtime_path(DEFAULT_INSTALL_DIR),
            'IS_SANDBOX': '1',
            'CLAUDE_ENABLE_STREAM_WATCHDOG': '0',
            'TERM': 'xterm-256color',
            'COLORTERM': 'truecolor',
            'CLAUDE_CODE_ENTRYPOINT': 'nexus-pty',
            'PWD': str(work_path),
            'FORCE_HYPERLINK': '1',
        }
        if extra_env:
            claude_env.update(extra_env)
        # Per-session sentinel paths — inherited by the Stop hook subprocess.
        # Written LAST so a caller's extra_env can never override them and break
        # session/turn isolation.
        claude_env.update({
            'NEXUS_PTY_DONE_FILE': done_file,
            'NEXUS_PTY_FINAL_FILE': final_file,
            'NEXUS_PTY_TURN_FILE': turn_file,
            'NEXUS_PTY_SESSION_ID': session_id,
            'NEXUS_PTY_HOOK_WAIT_TIMEOUT': str(hook_wait_timeout),
        })
        claude_env.pop('NEXUS_PTY_INTERACTION_DIR', None)
        if enable_input_bridge:
            claude_env['NEXUS_PTY_INTERACTION_DIR'] = interaction_dir

        # 3a) Seed ~/.claude.json so the interactive TUI skips first-run gates:
        #     onboarding, workspace-trust, and the custom-API-key prompt.
        proxy_api_key = claude_env.get('ANTHROPIC_API_KEY', DEFAULT_PROXY_API_KEY)
        _apply_json_patch(
            os.path.join(home, '.claude.json'),
            {
                'hasCompletedOnboarding': True,
                'hasTrustDialogAccepted': True,
                'bypassPermissionsModeAccepted': True,
                'theme': 'dark',
                'customApiKeyResponses': {'approved': [proxy_api_key[-20:]], 'rejected': []},
                'projects': {
                    working_dir: {
                        'hasTrustDialogAccepted': True,
                        'hasCompletedProjectOnboarding': True,
                        'allowedTools': [],
                    }
                },
            },
        )

        # 3b) Apply user config patches first, then force-inject Nexus hooks
        # so that a user patch containing hooks.Stop or hooks.PreToolUse cannot
        # silently override the completion/interaction hooks.
        settings_path = os.path.join(home, '.claude', 'settings.json')
        for patch in config_file_patches or []:
            _apply_json_patch(patch['path'], patch.get('content') or {})
        _append_required_hooks(
            settings_path,
            _interactive_hook_settings(
                _resolve_hook_python(),
                hook_wait_timeout=hook_wait_timeout,
                hook_command_timeout=hook_command_timeout,
            ),
        )

        # 4) Pre-start scripts / MCP (reuse claude_code helpers when present).
        if pre_start_scripts:
            from nexus.extensions.agents.claude_code.runtime_setup import _run_pre_start_script

            for script in pre_start_scripts:
                await _run_pre_start_script(script, working_dir, claude_env)
        if mcp_servers:
            from nexus.extensions.agents.claude_code.session_runtime import _register_mcp_servers

            await _register_mcp_servers(
                claude_path, working_dir=working_dir, claude_env=claude_env, mcp_servers=mcp_servers
            )

        # 5) Build the framework-free session (Claude spawned on first send_turn).
        base_env = get_os_env_without_ld_library_path()
        base_env['BASH_ENV'] = '~/.bashrc'
        session_env = {**base_env, **claude_env}

        # Run unattended: the interactive TUI otherwise blocks on a per-tool
        # permission prompt (the runner only auto-answers the trust-folder
        # dialog), so tool-using turns hang until timeout. Bypass permissions to
        # match the claude_code / claude_code_multiturn agents (all run inside an
        # isolated sandbox with IS_SANDBOX=1). Deduplicated so a caller that
        # already passed the flag in extra_cli_args does not get it twice.
        session_args = _ensure_skip_permissions(extra_cli_args)

        frames_path = f'/tmp/.cc_pty_frames_{token}.log' if log_frames else None
        interaction_bridge = FileHookInteractionBridge(interaction_dir) if enable_input_bridge else None
        pty_session = ClaudeInteractiveSession(
            claude_path=str(claude_path),
            cwd=working_dir,
            env=session_env,
            done_file=done_file,
            final_file=final_file,
            turn_file=turn_file,
            rows=rows,
            cols=cols,
            settle_seconds=prompt_settle_seconds,
            per_turn_timeout=timeout,
            snapshot_seconds=snapshot_seconds,
            log_frames=log_frames,
            frames_path=frames_path,
            extra_args=session_args,
            interaction_bridge=interaction_bridge,
        )
        await pty_session.start()
        return _InteractiveRuntimeSession(pty_session=pty_session, proxy=proxy, last_activity=time.time())
    except Exception:
        # Nothing is registered yet; make sure the proxy does not leak.
        with contextlib.suppress(Exception):
            await proxy.stop()
        raise


def _format_runtime_turn_result(
    *,
    session_id: str,
    result: dict[str, Any],
    start_time: float,
    prefix_events: list[dict[str, Any]] | None = None,
    recent_events: list[dict[str, Any]] | None = None,
    event_cursor: dict[str, Any] | None = None,
) -> dict[str, Any]:
    status = str(result.get('status') or ('completed' if result.get('completed') else result.get('reason') or 'error'))
    turn_id = result.get('turn_id')
    claude_pid = result.get('claude_pid')
    latest_screen = result.get('latest_screen', '')
    events = list(prefix_events or [])
    response_metadata = {}
    if 'interaction_response_status' in result:
        response_metadata['interaction_response_status'] = result['interaction_response_status']
    if result.get('session_reusable'):
        response_metadata['session_reusable'] = True

    if status == 'needs_user_input':
        interaction = result.get('interaction')
        if not isinstance(interaction, dict):
            raise RuntimeError('PTY returned needs_user_input without an interaction payload')
        events.append({
            'type': 'pty_interaction_requested',
            'session_id': session_id,
            'turn_id': turn_id,
            'interaction_id': interaction.get('interaction_id'),
            'tool_name': interaction.get('tool_name'),
            'claude_pid': claude_pid,
        })
        return {
            'success': True,
            'status': status,
            'session_id': session_id,
            'output': '',
            'stderr': '',
            'exit_code': 0,
            'execution_time': time.time() - start_time,
            'events': events,
            'interaction': interaction,
            'claude_pid': claude_pid,
            'turn_id': turn_id,
            'latest_screen': latest_screen,
            'recent_events': recent_events or [],
            'event_cursor': event_cursor or {},
            **response_metadata,
        }

    completed = status == 'completed' and bool(result.get('completed'))
    successful_boundary = completed or bool(result.get('session_reusable'))
    events.append({
        'type': 'pty_turn',
        'session_id': session_id,
        'turn_id': turn_id,
        'claude_pid': claude_pid,
        'completed': completed,
    })
    return {
        'success': successful_boundary,
        'status': status,
        'session_id': session_id,
        'output': result.get('final_message') or result.get('screen_tail', ''),
        'stderr': '' if successful_boundary else f'PTY turn not completed (reason={result.get("reason")})',
        'exit_code': 0 if successful_boundary else result.get('exit_code', -1),
        'execution_time': time.time() - start_time,
        'events': events,
        'claude_pid': claude_pid,
        'turn_id': turn_id,
        'latest_screen': latest_screen,
        'recent_events': recent_events or [],
        'event_cursor': event_cursor or {},
        **response_metadata,
    }


def _format_and_commit_boundary(
    *,
    session: _InteractiveRuntimeSession,
    session_id: str,
    result: dict[str, Any],
    start_time: float,
    prefix_events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Attach the session event delta to one PTY boundary and advance its cursor."""
    recent_events, event_cursor = session.proxy.get_event_delta()
    formatted = _format_runtime_turn_result(
        session_id=session_id,
        result=result,
        start_time=start_time,
        prefix_events=prefix_events,
        recent_events=recent_events,
        event_cursor=event_cursor,
    )
    session.proxy.commit_event_cursor(event_cursor['latest_seq'])
    return formatted


def _empty_event_cursor() -> dict[str, Any]:
    return {
        'after_seq': 0,
        'latest_seq': 0,
        'count': 0,
        'truncated': False,
        'dropped_count': 0,
    }


def _should_close_after_boundary(result: dict[str, Any]) -> bool:
    status = str(result.get('status') or result.get('reason') or '')
    return status != 'needs_user_input' and not result.get('completed') and not result.get('session_reusable')


async def run_claude_code_interactive_session_turn(
    *,
    session_id: str,
    prompt: str,
    parent_tool_use_id: str | None = None,
    working_dir: str = '/testbed',
    timeout: float = 3600,
    version: str = 'latest',
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
    enable_input_bridge: bool = False,
    rows: int = 50,
    cols: int = 200,
    prompt_settle_seconds: float = 4.0,
    log_frames: bool = False,
    snapshot_seconds: float | None = None,
    hook_wait_timeout: float = DEFAULT_HOOK_WAIT_TIMEOUT,
    hook_command_timeout: float = DEFAULT_HOOK_COMMAND_TIMEOUT,
) -> dict[str, Any]:
    """Start or reuse an interactive PTY session and run one user turn."""
    start_time = time.time()

    session = _PTY_SESSIONS.get(session_id)
    if session is None:
        async with _CREATE_LOCK:
            session = _PTY_SESSIONS.get(session_id)  # re-check under lock
            if session is None:
                try:
                    session = await _prepare_runtime_session(
                        session_id,
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
                        enable_input_bridge=enable_input_bridge,
                        rows=rows,
                        cols=cols,
                        prompt_settle_seconds=prompt_settle_seconds,
                        log_frames=log_frames,
                        snapshot_seconds=snapshot_seconds,
                        hook_wait_timeout=hook_wait_timeout,
                        hook_command_timeout=hook_command_timeout,
                    )
                    _PTY_SESSIONS[session_id] = session
                except Exception as exc:  # noqa: BLE001
                    import traceback

                    logger.error('[pty %s] session prepare failed: %s', session_id, exc)
                    return {
                        'success': False,
                        'status': 'session_prepare_error',
                        'session_id': session_id,
                        'output': '',
                        'stderr': f'session prepare failed: {exc}\n{traceback.format_exc()}',
                        'exit_code': -1,
                        'execution_time': time.time() - start_time,
                        'events': [],
                        'latest_screen': '',
                        'recent_events': [],
                        'event_cursor': _empty_event_cursor(),
                    }

    try:
        async with session.lock:
            session.proxy.mark_turn_start()
            result = await session.pty_session.send_turn(prompt, timeout=timeout)
            session.last_activity = time.time()
    except Exception as exc:  # noqa: BLE001
        logger.exception('[pty %s] turn failed: %s', session_id, exc)
        pty_session = session.pty_session
        boundary_result = {
            'status': 'error',
            'completed': False,
            'turn_id': getattr(pty_session, '_active_turn_id', None) or getattr(pty_session, '_turn', None),
            'final_message': '',
            'screen_tail': pty_session.screen_tail(),
            'latest_screen': pty_session.latest_screen(),
            'reason': 'turn_error',
            'claude_pid': pty_session.claude_pid,
            'exit_code': -1,
        }
        formatted = _format_and_commit_boundary(
            session=session,
            session_id=session_id,
            result=boundary_result,
            start_time=start_time,
        )
        formatted['stderr'] = f'PTY turn failed: {exc}'
        # Tear the session down so a broken PTY/proxy does not leak or serve the
        # next turn in a bad state.
        with contextlib.suppress(Exception):
            await close_claude_code_interactive_session(session_id=session_id)
        return formatted

    status = str(result.get('status') or result.get('reason') or '')
    logger.info(
        '[pty %s] turn %s state=%s completed=%s reason=%s pid=%s',
        session_id,
        result.get('turn_id'),
        status,
        result.get('completed'),
        result.get('reason'),
        result.get('claude_pid'),
    )
    formatted = _format_and_commit_boundary(
        session=session,
        session_id=session_id,
        result=result,
        start_time=start_time,
    )

    # A turn that did not complete (timeout / process_exit) leaves the PTY in an
    # unknown state — hung or already exited. Drop it from the registry and close
    # it so the next turn on this session_id starts a fresh PTY rather than typing
    # into a dead one (this must happen even when close_on_finish is False).
    if _should_close_after_boundary(result):
        with contextlib.suppress(Exception):
            await close_claude_code_interactive_session(session_id=session_id)

    return formatted


async def respond_and_wait_claude_code_interactive_session(
    *,
    session_id: str,
    interaction_id: str,
    answer: str | None,
    timeout: float = 3600,
) -> dict[str, Any]:
    """Answer one Hook request and keep waiting inside the same PTY turn."""
    del timeout  # The original send_turn deadline remains authoritative.
    start_time = time.time()
    session = _PTY_SESSIONS.get(session_id)
    if session is None:
        return {
            'success': False,
            'status': 'session_missing',
            'session_id': session_id,
            'output': '',
            'stderr': f'Interactive PTY session does not exist: {session_id}',
            'exit_code': -1,
            'execution_time': time.time() - start_time,
            'events': [],
            'latest_screen': '',
            'recent_events': [],
            'event_cursor': _empty_event_cursor(),
        }

    resolved_event = {
        'type': 'pty_interaction_resolved',
        'session_id': session_id,
        'interaction_id': interaction_id,
        'resolution': 'user' if answer is not None and answer.strip() else 'unattended',
    }
    try:
        async with session.lock:
            result = await session.pty_session.respond_and_wait(interaction_id, answer)
            session.last_activity = time.time()
    except Exception as exc:  # noqa: BLE001
        logger.exception('[pty %s] interaction %s failed: %s', session_id, interaction_id, exc)
        pty_session = session.pty_session
        boundary_result = {
            'status': 'interaction_error',
            'completed': False,
            'turn_id': getattr(pty_session, '_active_turn_id', None) or getattr(pty_session, '_turn', None),
            'final_message': '',
            'screen_tail': pty_session.screen_tail(),
            'latest_screen': pty_session.latest_screen(),
            'reason': 'interaction_error',
            'claude_pid': pty_session.claude_pid,
            'exit_code': -1,
        }
        formatted = _format_and_commit_boundary(
            session=session,
            session_id=session_id,
            result=boundary_result,
            start_time=start_time,
            prefix_events=[resolved_event],
        )
        formatted['stderr'] = f'PTY interaction failed: {exc}'
        with contextlib.suppress(Exception):
            await close_claude_code_interactive_session(session_id=session_id)
        return formatted

    resolved_event['turn_id'] = result.get('turn_id')
    resolved_event['claude_pid'] = result.get('claude_pid')
    response_status = str(result.get('interaction_response_status') or 'written')
    resolved_event['submitted'] = response_status == 'written'
    if response_status != 'written':
        resolved_event['resolution'] = response_status
    formatted = _format_and_commit_boundary(
        session=session,
        session_id=session_id,
        result=result,
        start_time=start_time,
        prefix_events=[resolved_event],
    )
    if _should_close_after_boundary(result):
        with contextlib.suppress(Exception):
            await close_claude_code_interactive_session(session_id=session_id)
    return formatted


async def close_claude_code_interactive_session(*, session_id: str) -> dict[str, Any]:
    """Close one runtime-side interactive PTY session. Idempotent."""
    session = _PTY_SESSIONS.pop(session_id, None)
    if session is None:
        return {'success': True, 'session_id': session_id, 'exit_code': 0, 'stderr': ''}
    await session.close()
    return {'success': True, 'session_id': session_id, 'exit_code': 0, 'stderr': ''}


async def close_all_claude_code_interactive_sessions() -> dict[str, Any]:
    """Close all runtime-side interactive PTY sessions in this worker process."""
    session_ids = list(_PTY_SESSIONS)
    results = [await close_claude_code_interactive_session(session_id=sid) for sid in session_ids]
    return {'success': True, 'closed': results}


async def setup_and_run_claude_code_interactive(
    task: str,
    working_dir: str = '/testbed',
    timeout: float = 3600,
    version: str = 'latest',
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
    enable_input_bridge: bool = False,
    rows: int = 50,
    cols: int = 200,
    prompt_settle_seconds: float = 4.0,
    log_frames: bool = False,
    snapshot_seconds: float | None = None,
    hook_wait_timeout: float = DEFAULT_HOOK_WAIT_TIMEOUT,
    hook_command_timeout: float = DEFAULT_HOOK_COMMAND_TIMEOUT,
) -> dict[str, Any]:
    """Single-turn in-process path (compat): drive one turn on a throwaway session."""
    start_time = time.time()
    session_id = f'oneshot_{int(start_time * 1000)}_{os.getpid()}'
    logger.debug(
        '[pty %s] ENTER (single-turn) workdir=%s timeout=%s version=%s', session_id, working_dir, timeout, version
    )

    session = await _prepare_runtime_session(
        session_id,
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
        enable_input_bridge=enable_input_bridge,
        rows=rows,
        cols=cols,
        prompt_settle_seconds=prompt_settle_seconds,
        log_frames=log_frames,
        snapshot_seconds=snapshot_seconds,
        hook_wait_timeout=hook_wait_timeout,
        hook_command_timeout=hook_command_timeout,
    )
    try:
        session.proxy.mark_turn_start()
        result = await session.pty_session.send_turn(task, timeout=timeout)
        formatted = _format_and_commit_boundary(
            session=session,
            session_id=session_id,
            result=result,
            start_time=start_time,
        )
    finally:
        await session.close()

    formatted['mode'] = 'pty'
    return formatted


__all__ = [
    'close_all_claude_code_interactive_sessions',
    'close_claude_code_interactive_session',
    'respond_and_wait_claude_code_interactive_session',
    'run_claude_code_interactive_session_turn',
    'setup_and_run_claude_code_interactive',
]
