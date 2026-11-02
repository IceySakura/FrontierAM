"""Runtime-side lifecycle for persistent DeepSeek Harness ACP sessions."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import shutil
import subprocess
import time

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from nexus.extensions.agents.dsh.installer import (
    DEFAULT_DSH_ACP_VERSION,
    DEFAULT_DSH_PREBUILT_BASE_URL,
    DEFAULT_DSH_VERSION,
    DEFAULT_INSTALL_DIR,
    get_installation_root,
    install_dsh,
)
from nexus.extensions.agents.dsh.profile import PROFILE_NAME, DshProfile, prepare_dsh_profile, validate_extra_env
from nexus.extensions.agents.dsh.session_log import DshSessionLog, summarize_turn
from nexus.extensions.agents.installed_common import (
    DEFAULT_CLI_STREAM_READER_LIMIT,
    DEFAULT_NPM_REGISTRY,
    DEFAULT_PROXY_API_KEY,
    DEFAULT_RUNTIME_BIN_DIR,
    build_runtime_path,
    find_free_port,
    find_nexus_runtime_binary,
)
from nexus.extensions.agents.installed_common.acp import AcpConnection, assistant_text_from_updates
from nexus.extensions.agents.installed_common.stream_trace import (
    flush_stream_trace_events,
    record_stream_trace_event,
)
from nexus.utils.common import terminate_process


logger = logging.getLogger(__name__)

_SESSIONS: dict[str, _DshAcpSession] = {}
_SESSION_LOCKS: dict[str, asyncio.Lock] = {}
_MAX_STDERR_LINES = 200


@dataclass
class _DshAcpSession:
    session_id: str
    remote_session_id: str
    proc: asyncio.subprocess.Process
    proxy: Any
    connection: AcpConnection
    session_log: DshSessionLog
    profile: DshProfile
    cwd: str
    started_at: float
    stderr_lines: list[str]
    stderr_task: asyncio.Task[None]
    retain_session_artifacts: bool
    record_trace_events: bool
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @property
    def is_running(self) -> bool:
        return self.proc.returncode is None

    async def send_user_message(
        self,
        prompt: str,
        *,
        timeout: float,
        transcript_timeout: float,
    ) -> dict[str, Any]:
        async with self.lock:
            if not self.is_running:
                raise RuntimeError(f'DeepSeek Harness session {self.session_id} is not running')

            self.connection.drain_updates()
            start = time.time()
            try:
                response = await self.connection.prompt(
                    session_id=self.remote_session_id,
                    text=prompt,
                    timeout=timeout,
                )
            except asyncio.TimeoutError as exc:
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(
                        self.connection.cancel(session_id=self.remote_session_id),
                        timeout=2.0,
                    )
                raise TimeoutError(f'DeepSeek Harness session {self.session_id} timed out') from exc
            except asyncio.CancelledError:
                with contextlib.suppress(Exception):
                    await asyncio.shield(
                        asyncio.wait_for(
                            self.connection.cancel(session_id=self.remote_session_id),
                            timeout=2.0,
                        )
                    )
                raise

            events = await self.session_log.wait_for_turn_end(timeout=transcript_timeout)
            # DSH schedules committed-message notifications without awaiting the
            # transport promise. The JSONL turn/end is authoritative and gives
            # that final notification a chance to reach the reader before drain.
            await asyncio.sleep(0)
            updates = self.connection.drain_updates(session_id=self.remote_session_id)
            for event in events:
                record_stream_trace_event(
                    self,
                    event,
                    agent_key='dsh',
                    enabled=self.record_trace_events,
                )
            summary = summarize_turn(events, acp_text=assistant_text_from_updates(updates))
            stderr = ''
            if not summary['success']:
                stderr = f'DeepSeek Harness turn ended with {summary["status"]}: '
                stderr += json.dumps(summary['result_event'], ensure_ascii=False)
            success = bool(summary['success'])
            return {
                'success': success,
                'status': summary['status'],
                'session_id': self.session_id,
                'dsh_session_id': self.remote_session_id,
                'output': summary['output'],
                'stderr': stderr,
                'exit_code': (self.proc.returncode or 0) if success else -1,
                'execution_time': time.time() - start,
                'events': events,
                'result_event': summary['result_event'],
                'acp_stop_reason': response['stopReason'],
            }

    async def close(self) -> dict[str, Any]:
        async with self.lock:
            if self.proc.stdin is not None:
                with contextlib.suppress(Exception):
                    self.proc.stdin.close()
                    await self.proc.stdin.wait_closed()

            if self.proc.returncode is None:
                try:
                    await asyncio.wait_for(self.proc.wait(), timeout=6.0)
                except asyncio.TimeoutError:
                    await terminate_process(self.proc, timeout=3.0)

            await self.connection.close()
            self.stderr_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.stderr_task
            with contextlib.suppress(Exception):
                await self.proxy.stop()
            if not self.retain_session_artifacts:
                with contextlib.suppress(Exception):
                    await asyncio.to_thread(shutil.rmtree, self.profile.home)
            flush_stream_trace_events(self, agent_key='dsh')
            return {
                'success': True,
                'session_id': self.session_id,
                'exit_code': self.proc.returncode if self.proc.returncode is not None else 0,
                'stderr': '\n'.join(self.stderr_lines),
            }


async def _read_stderr(
    stream: asyncio.StreamReader,
    *,
    session_id: str,
    buffer: list[str],
) -> None:
    try:
        while True:
            line = await stream.readline()
            if not line:
                return
            text = line.decode('utf-8', errors='replace').rstrip('\n')
            buffer.append(text)
            if len(buffer) > _MAX_STDERR_LINES:
                del buffer[: len(buffer) - _MAX_STDERR_LINES]
            logger.debug('[dsh:%s] %s', session_id, text)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception('[dsh:%s] stderr reader failed', session_id)


async def _cleanup_failed_start(
    *,
    proc: asyncio.subprocess.Process | None,
    connection: AcpConnection | None,
    stderr_task: asyncio.Task[None] | None,
    proxy: Any,
    profile: DshProfile | None,
    retain_session_artifacts: bool,
) -> None:
    if proc is not None:
        if proc.stdin is not None:
            with contextlib.suppress(Exception):
                proc.stdin.close()
        await terminate_process(proc, timeout=3.0)
    if connection is not None:
        with contextlib.suppress(Exception):
            await connection.close()
    if stderr_task is not None:
        stderr_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await stderr_task
    with contextlib.suppress(Exception):
        await proxy.stop()
    if profile is not None and not retain_session_artifacts:
        with contextlib.suppress(Exception):
            await asyncio.to_thread(shutil.rmtree, profile.home)


async def _start_session(
    *,
    session_id: str,
    working_dir: str,
    timeout: float,
    version: str,
    acp_version: str,
    npm_registry: str | None,
    prebuilt_base_url: str | None,
    extra_env: dict[str, str] | None,
    permission_mode: Literal['read-only', 'workspace-write', 'danger-full-access'],
    permission_policy: Literal['allow', 'reject'],
    context_window: int,
    max_output_tokens: int,
    transcript_timeout: float,
    record_trace_events: bool,
    retain_session_artifacts: bool,
) -> _DshAcpSession:
    from nexus.runtime.context import require_runtime_control
    from nexus.runtime.server.services.openai_proxy import OpenAIProxy
    from nexus.utils.shell import get_os_env_without_ld_library_path

    validate_extra_env(extra_env)
    rt = require_runtime_control()
    dsh_entry = await install_dsh(
        DEFAULT_INSTALL_DIR,
        version=version,
        acp_version=acp_version,
        npm_registry=npm_registry,
        prebuilt_base_url=prebuilt_base_url,
    )
    dsh_installation = get_installation_root(dsh_entry)
    # Must match the binary the installer validated, and must be the runtime's own: the argv below launches DSH directly, bypassing the PATH set in managed_env.
    node = find_nexus_runtime_binary('node')
    if node is None:
        raise RuntimeError('DeepSeek Harness requires Node.js, but node was not found after installation')

    work_path = Path(working_dir).resolve()
    work_path.mkdir(parents=True, exist_ok=True)
    proxy_port = find_free_port()
    proxy = OpenAIProxy(
        sio=rt.sio,
        listen_port=proxy_port,
        listen_host='127.0.0.1',
        request_timeout=timeout,
        models=['nexus-model'],
    )
    await proxy.start()

    profile: DshProfile | None = None
    proc: asyncio.subprocess.Process | None = None
    connection: AcpConnection | None = None
    stderr_task: asyncio.Task[None] | None = None
    stderr_lines: list[str] = []
    try:
        profile = prepare_dsh_profile(
            install_dir=DEFAULT_INSTALL_DIR,
            runtime_installation=dsh_installation,
            session_id=session_id,
            acp_version=acp_version,
            proxy_base_url=f'http://127.0.0.1:{proxy_port}/v1',
            context_window=context_window,
            max_output_tokens=max_output_tokens,
        )
        managed_env = {
            'DSH_HOME': str(profile.home),
            'DSH_PERMISSION_MODE': permission_mode,
            'DSH_TELEMETRY_DISABLED': '1',
            'NEXUS_PROXY_API_KEY': DEFAULT_PROXY_API_KEY,
            'PATH': build_runtime_path(
                dsh_installation / 'node_modules' / '.bin',
                DEFAULT_RUNTIME_BIN_DIR,
            ),
            'PWD': str(work_path),
            'CI': '1',
            'NO_COLOR': '1',
        }
        launch_env = get_os_env_without_ld_library_path()
        launch_env['BASH_ENV'] = '~/.bashrc'
        launch_env.update(extra_env or {})
        launch_env.update(managed_env)
        # Cordis first tries a native addon to access Node's internal ESM
        # loader. That fallback cannot load reliably when node comes from the
        # Nexus Nix closure and node-pty/addons are built against the user
        # image, so expose the same internals explicitly to this DSH process.
        cmd = [node, '--expose-internals', str(dsh_entry), '--profile', PROFILE_NAME]
        logger.debug('Starting DeepSeek Harness ACP session %s', session_id)
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(work_path),
            env=launch_env,
            limit=DEFAULT_CLI_STREAM_READER_LIMIT,
        )
        if proc.stdin is None or proc.stdout is None or proc.stderr is None:
            raise RuntimeError('DeepSeek Harness process was created without stdio pipes')
        stderr_task = asyncio.create_task(_read_stderr(proc.stderr, session_id=session_id, buffer=stderr_lines))
        connection = AcpConnection(proc.stdout, proc.stdin, permission_policy=permission_policy)
        connection.start()
        rpc_timeout = min(max(timeout, 1.0), 60.0)
        await connection.initialize(timeout=rpc_timeout)
        remote_session_id = await connection.new_session(cwd=str(work_path), timeout=rpc_timeout)
        session_log = await DshSessionLog.wait_for_session(
            root=profile.session_root,
            session_id=remote_session_id,
            cwd=str(work_path),
            timeout=min(max(transcript_timeout, 1.0), 10.0),
        )
        return _DshAcpSession(
            session_id=session_id,
            remote_session_id=remote_session_id,
            proc=proc,
            proxy=proxy,
            connection=connection,
            session_log=session_log,
            profile=profile,
            cwd=str(work_path),
            started_at=time.time(),
            stderr_lines=stderr_lines,
            stderr_task=stderr_task,
            retain_session_artifacts=retain_session_artifacts,
            record_trace_events=record_trace_events,
        )
    except BaseException:
        await _cleanup_failed_start(
            proc=proc,
            connection=connection,
            stderr_task=stderr_task,
            proxy=proxy,
            profile=profile,
            retain_session_artifacts=retain_session_artifacts,
        )
        raise


async def run_dsh_session_turn(
    *,
    session_id: str,
    prompt: str,
    parent_tool_use_id: str | None = None,
    working_dir: str = '/workspace',
    timeout: float = 3600.0,
    version: str = DEFAULT_DSH_VERSION,
    acp_version: str = DEFAULT_DSH_ACP_VERSION,
    npm_registry: str | None = DEFAULT_NPM_REGISTRY,
    prebuilt_base_url: str | None = DEFAULT_DSH_PREBUILT_BASE_URL,
    extra_env: dict[str, str] | None = None,
    permission_mode: Literal['read-only', 'workspace-write', 'danger-full-access'] = 'workspace-write',
    permission_policy: Literal['allow', 'reject'] = 'allow',
    context_window: int = 131072,
    max_output_tokens: int = 32768,
    transcript_timeout: float = 5.0,
    record_trace_events: bool = True,
    retain_session_artifacts: bool = False,
) -> dict[str, Any]:
    """Start or reuse one DSH ACP process and run a user turn."""
    del parent_tool_use_id  # ACP follow-ups are ordinary user messages in the same session.
    start = time.time()
    session = _SESSIONS.get(session_id)
    if session is not None and not session.is_running:
        await close_dsh_session(session_id=session_id)
        return {
            'success': False,
            'session_id': session_id,
            'output': '',
            'stderr': f'DeepSeek Harness session {session_id} exited before the next turn',
            'exit_code': -1,
            'execution_time': time.time() - start,
            'events': [],
        }

    try:
        if session is None:
            start_lock = _SESSION_LOCKS.setdefault(session_id, asyncio.Lock())
            async with start_lock:
                session = _SESSIONS.get(session_id)
                if session is None:
                    session = await _start_session(
                        session_id=session_id,
                        working_dir=working_dir,
                        timeout=timeout,
                        version=version,
                        acp_version=acp_version,
                        npm_registry=npm_registry,
                        prebuilt_base_url=prebuilt_base_url,
                        extra_env=extra_env,
                        permission_mode=permission_mode,
                        permission_policy=permission_policy,
                        context_window=context_window,
                        max_output_tokens=max_output_tokens,
                        transcript_timeout=transcript_timeout,
                        record_trace_events=record_trace_events,
                        retain_session_artifacts=retain_session_artifacts,
                    )
                    _SESSIONS[session_id] = session
        expected_cwd = str(Path(working_dir).resolve())
        if session.cwd != expected_cwd:
            raise ValueError(
                f'DeepSeek Harness session {session_id} already uses cwd {session.cwd}, not {expected_cwd}'
            )
        result = await session.send_user_message(
            prompt,
            timeout=timeout,
            transcript_timeout=transcript_timeout,
        )
        if not result.get('success', False):
            await close_dsh_session(session_id=session_id)
        return result
    except asyncio.CancelledError:
        with contextlib.suppress(Exception):
            await asyncio.shield(close_dsh_session(session_id=session_id))
        raise
    except Exception as exc:
        logger.exception('DeepSeek Harness session turn failed: %s', exc)
        stderr = '\n'.join(session.stderr_lines) if session is not None else ''
        with contextlib.suppress(Exception):
            await close_dsh_session(session_id=session_id)
        return {
            'success': False,
            'session_id': session_id,
            'output': '',
            'stderr': '\n'.join(part for part in (str(exc), stderr) if part),
            'exit_code': -1,
            'execution_time': time.time() - start,
            'events': [],
        }
    finally:
        if session is not None:
            flush_stream_trace_events(session, agent_key='dsh')


async def close_dsh_session(*, session_id: str) -> dict[str, Any]:
    """Idempotently close one runtime-side DSH process and proxy."""
    start_lock = _SESSION_LOCKS.setdefault(session_id, asyncio.Lock())
    try:
        async with start_lock:
            session = _SESSIONS.pop(session_id, None)
            if session is None:
                return {'success': True, 'session_id': session_id, 'exit_code': 0, 'stderr': ''}
            return await session.close()
    finally:
        if _SESSION_LOCKS.get(session_id) is start_lock:
            _SESSION_LOCKS.pop(session_id, None)


__all__ = ['close_dsh_session', 'run_dsh_session_turn']
