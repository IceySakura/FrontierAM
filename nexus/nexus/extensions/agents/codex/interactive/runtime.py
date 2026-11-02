"""Worker-side glue: install Codex, start the proxy, own the session registry.

One session holds a Codex TUI, its private app-server + relay, a Responses proxy,
and a stable ``CODEX_HOME``. All of it is keyed by the Nexus ``session_id`` so the
same PTY serves every turn of a run, and so two concurrent sessions never share a
socket, a home, or a thread.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import os
import re
import shutil
import time

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nexus.extensions.agents.installed_common import (
    DEFAULT_NPM_REGISTRY,
    DEFAULT_PROXY_API_KEY,
    build_runtime_path,
    find_free_port,
)

from . import INTERACTIVE_DEFAULT_CODEX_VERSION, validate_interactive_version
from .events import empty_event_cursor
from .guards import (
    apply_protected_env,
    assert_env_not_overriding,
    validate_extra_cli_args,
    validate_extra_config_toml,
)
from .session import CodexInteractiveSession


logger = logging.getLogger(__name__)

# Keyed by Nexus session_id; persists across the per-turn runtime invocations that
# run inside this worker process.
_CODEX_PTY_SESSIONS: dict[str, _InteractiveRuntimeSession] = {}
# Serialises first-create so two concurrent turns cannot both install Codex or
# start a second app-server for one session_id.
_CREATE_LOCK = asyncio.Lock()

_SESSION_ROOT_NAME = '.nexus-codex-pty'


@dataclass
class _InteractiveRuntimeSession:
    """Nexus-side wrapper: PTY session + proxy + per-session turn lock."""

    session: CodexInteractiveSession
    proxy: Any
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    close_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    closed: bool = False

    async def close(self) -> None:
        async with self.close_lock:
            if self.closed:
                return
            errors: list[tuple[str, Exception]] = []
            try:
                await self.session.close()
            except Exception as error:
                errors.append(('session', error))
            try:
                await self.proxy.stop()
            except Exception as error:
                errors.append(('proxy', error))
            if errors:
                details = '; '.join(f'{stage}: {error}' for stage, error in errors)
                raise RuntimeError(f'Codex runtime cleanup failed: {details}') from errors[0][1]
            self.closed = True


def _session_token(session_id: str) -> str:
    """Filesystem-safe, collision-free directory token for one session.

    Sanitising alone is not injective: ``a/b`` and ``a?b`` both sanitise to
    ``a_b``, and truncation merges any two ids sharing a long prefix. Two sessions
    mapping to one token would share CODEX_HOME and sockets, and closing one would
    delete the other's directory. A digest of the *original* id is appended so the
    readable prefix stays useful while the token stays unique.
    """
    readable = re.sub(r'[^A-Za-z0-9_.-]', '_', session_id)[:48]
    digest = hashlib.sha256(session_id.encode('utf-8')).hexdigest()[:16]
    return f'{readable}_{digest}_{os.getpid()}'


def _session_paths(session_id: str) -> tuple[str, str]:
    """Return (session_dir, codex_home) — both private to this session."""
    home = os.environ.get('HOME', '/root')
    session_dir = os.path.join(home, _SESSION_ROOT_NAME, _session_token(session_id))
    return session_dir, os.path.join(session_dir, 'codex-home')


async def _cleanup_failed_prepare(
    *,
    session: CodexInteractiveSession | None,
    proxy: Any | None,
    session_dir: str,
) -> list[tuple[str, Exception]]:
    """Release resources created before a session could enter the registry."""
    errors: list[tuple[str, Exception]] = []
    if session is not None:
        try:
            await asyncio.shield(session.close())
        except Exception as error:
            errors.append(('session', error))
    if proxy is not None:
        try:
            await asyncio.shield(proxy.stop())
        except Exception as error:
            errors.append(('proxy', error))
    try:
        shutil.rmtree(session_dir)
    except FileNotFoundError:
        pass
    except OSError as error:
        errors.append(('session directory', error))
    return errors


async def _prepare_session(
    session_id: str,
    *,
    working_dir: str,
    timeout: float,
    version: str,
    npm_registry: str | None,
    binary_url: str | None,
    binary_checksum: str | None,
    model: str | None,
    model_catalog_json: dict[str, Any] | None,
    mcp_servers: list[dict[str, Any]] | None,
    extra_env: dict[str, str] | None,
    extra_cli_args: list[str] | None,
    extra_config_toml: str | None,
    model_instructions_file: str | None = None,
    use_azure_provider: bool,
    rows: int,
    cols: int,
) -> _InteractiveRuntimeSession:
    """Install Codex, start the proxy, write config, and boot the TUI."""
    from nexus.extensions.agents.codex.runtime_setup import (
        DEFAULT_INSTALL_DIR,
        build_codex_config_lines,
        install_codex_binary,
        resolve_codex_launch_argv,
    )
    from nexus.runtime.context import require_runtime_control
    from nexus.runtime.server.services.responses_proxy import ResponsesProxy

    validate_interactive_version(version)
    # Reject unsafe overrides before anything is installed or started, so a bad
    # configuration fails fast instead of after a proxy and a TUI are running.
    validate_extra_cli_args(extra_cli_args)
    validate_extra_config_toml(extra_config_toml)
    assert_env_not_overriding(extra_env)
    rt = require_runtime_control()

    session_dir, codex_home = _session_paths(session_id)
    proxy: Any | None = None
    session: CodexInteractiveSession | None = None
    try:
        Path(codex_home).mkdir(parents=True, exist_ok=True)
        work_path = Path(working_dir).resolve()
        work_path.mkdir(parents=True, exist_ok=True)
        working_dir = str(work_path)

        install_dir = f'{DEFAULT_INSTALL_DIR}-interactive'
        codex_path = await asyncio.wait_for(
            install_codex_binary(
                install_dir=install_dir,
                version=version,
                npm_registry=npm_registry,
                binary_url=binary_url,
                binary_checksum=binary_checksum,
            ),
            timeout=1800,
        )
        logger.info('[codex-pty %s] codex binary: %s', session_id, codex_path)

        proxy_port = find_free_port()
        proxy = ResponsesProxy(
            sio=rt.sio,
            listen_port=proxy_port,
            listen_host='127.0.0.1',
            request_timeout=None,
            codex_version=version,
        )
        await asyncio.wait_for(proxy.start(), timeout=60)
        logger.info('[codex-pty %s] responses proxy on %d', session_id, proxy_port)

        config_lines = build_codex_config_lines(
            codex_home=Path(codex_home),
            proxy_base_url=f'http://127.0.0.1:{proxy_port}/v1',
            version=version,
            model=model,
            model_catalog_json=model_catalog_json,
            default_mode_request_user_input=True,
            trusted_project=working_dir,
            use_azure_provider=use_azure_provider,
            mcp_servers=mcp_servers,
            extra_config_toml=extra_config_toml,
            model_instructions_file=model_instructions_file,
        )
        (Path(codex_home) / 'config.toml').write_text('\n'.join(config_lines) + '\n', encoding='utf-8')

        codex_env: dict[str, str] = {
            **os.environ,
            'CODEX_HOME': codex_home,
            'NEXUS_AGENT_PROXY_URL': f'http://127.0.0.1:{proxy_port}',
            'NEXUS_PROXY_API_KEY': DEFAULT_PROXY_API_KEY,
            'PATH': build_runtime_path(install_dir),
            'TERM': 'xterm-256color',
        }
        owned_env = {
            'CODEX_HOME': codex_home,
            'NEXUS_AGENT_PROXY_URL': f'http://127.0.0.1:{proxy_port}',
            'NEXUS_PROXY_API_KEY': DEFAULT_PROXY_API_KEY,
        }
        if extra_env:
            codex_env.update(extra_env)
        # Re-applied AFTER the merge: a caller's extra_env must not be able to
        # break session isolation or redirect the proxy. PATH is intentionally not
        # protected here -- that stays the existing installed-agent contract.
        codex_env = apply_protected_env(codex_env, owned_env)

        session = CodexInteractiveSession(
            codex_argv=resolve_codex_launch_argv(codex_path=codex_path, install_dir=install_dir, binary_url=binary_url),
            env=codex_env,
            working_dir=working_dir,
            session_dir=session_dir,
            rows=rows,
            cols=cols,
            per_turn_timeout=timeout,
            extra_cli_args=extra_cli_args,
        )
        await session.start()
        return _InteractiveRuntimeSession(session=session, proxy=proxy)
    except BaseException as error:
        # BaseException, not Exception: a cancellation here happens before the
        # session is registered, so nothing else would ever reclaim the proxy,
        # the app-server, or the PTY. Shielded so cleanup completes.
        cleanup_errors = await _cleanup_failed_prepare(session=session, proxy=proxy, session_dir=session_dir)
        if cleanup_errors:
            details = '; '.join(f'{stage}: {cleanup_error}' for stage, cleanup_error in cleanup_errors)
            logger.error('[codex-pty %s] session prepare cleanup failed: %s', session_id, details)
            if isinstance(error, Exception):
                raise RuntimeError(f'{error}; session prepare cleanup failed: {details}') from error
        raise


def _format_turn(
    *,
    session_id: str,
    entry: _InteractiveRuntimeSession,
    report: dict[str, Any],
    start_time: float,
) -> dict[str, Any]:
    """Render one boundary into the runtime contract and advance the cursor.

    Truth table (``success`` is "did this call do what was asked", NOT "is the
    session still usable" — conflating them reported errors as successes):

    | status               | success | completed | reusable |
    |----------------------|---------|-----------|----------|
    | completed            | True    | True      | True     |
    | needs_user_input     | True    | False     | True     |
    | interaction_not_found| False   | False     | True     |
    | approval_protocol_error | False | False   | False    |
    | interaction_unanswered  | False | False   | False    |
    | timeout              | False   | False     | False    |
    | process_exit         | False   | False     | False    |
    | event_source_exit    | False   | False     | False    |
    | failed / interrupted | False   | False     | False    |
    | error                | False   | False     | False    |
    """
    recent_events, cursor = entry.session.events.get_delta()
    status = str(report.get('status') or 'error')
    completed = bool(report.get('completed'))
    reusable = bool(report.get('session_reusable', True))
    # Waiting for the caller is the only successful non-terminal outcome.
    success = completed or status == 'needs_user_input'

    events = [
        {
            'type': 'pty_turn',
            'session_id': session_id,
            'thread_id': report.get('thread_id'),
            'turn_id': report.get('turn_id'),
            'status': status,
            'completed': completed,
            'codex_pid': entry.session.codex_pid,
        }
    ]
    formatted: dict[str, Any] = {
        'success': success,
        'status': status,
        'session_id': session_id,
        'thread_id': report.get('thread_id'),
        'turn_id': report.get('turn_id'),
        'output': report.get('output') or '',
        'stderr': report.get('stderr') or '',
        'exit_code': report.get('exit_code', 0 if success else -1),
        'execution_time': time.time() - start_time,
        'events': events,
        'latest_screen': report.get('latest_screen', ''),
        'recent_events': recent_events,
        'event_cursor': cursor,
        'session_reusable': reusable,
        'codex_pid': entry.session.codex_pid,
    }
    if report.get('interaction') is not None:
        formatted['interaction'] = report['interaction']
    entry.session.events.commit(cursor['latest_seq'])
    return formatted


def _error_result(session_id: str, status: str, stderr: str, start_time: float) -> dict[str, Any]:
    return {
        'success': False,
        'status': status,
        'session_id': session_id,
        'thread_id': None,
        'turn_id': None,
        'output': '',
        'stderr': stderr,
        'exit_code': -1,
        'execution_time': time.time() - start_time,
        'events': [],
        'latest_screen': '',
        'recent_events': [],
        'event_cursor': empty_event_cursor(),
        'session_reusable': False,
    }


async def run_codex_interactive_session_turn(
    *,
    session_id: str,
    prompt: str,
    parent_tool_use_id: str | None = None,
    working_dir: str = '/workspace',
    timeout: float = 3600,
    version: str = INTERACTIVE_DEFAULT_CODEX_VERSION,
    npm_registry: str | None = DEFAULT_NPM_REGISTRY,
    binary_url: str | None = None,
    binary_checksum: str | None = None,
    model: str | None = None,
    model_catalog_json: dict[str, Any] | None = None,
    mcp_servers: list[dict[str, Any]] | None = None,
    extra_env: dict[str, str] | None = None,
    extra_cli_args: list[str] | None = None,
    extra_config_toml: str | None = None,
    model_instructions_file: str | None = None,
    use_azure_provider: bool = True,
    rows: int = 50,
    cols: int = 200,
) -> dict[str, Any]:
    """Start or reuse an interactive session and run one user turn."""
    del parent_tool_use_id  # Codex correlates by thread/turn id, not tool-use id.
    start_time = time.time()

    entry = _CODEX_PTY_SESSIONS.get(session_id)
    if entry is None:
        async with _CREATE_LOCK:
            entry = _CODEX_PTY_SESSIONS.get(session_id)  # re-check under the lock
            if entry is None:
                try:
                    entry = await _prepare_session(
                        session_id,
                        working_dir=working_dir,
                        timeout=timeout,
                        version=version,
                        npm_registry=npm_registry,
                        binary_url=binary_url,
                        binary_checksum=binary_checksum,
                        model=model,
                        model_catalog_json=model_catalog_json,
                        mcp_servers=mcp_servers,
                        extra_env=extra_env,
                        extra_cli_args=extra_cli_args,
                        extra_config_toml=extra_config_toml,
                        model_instructions_file=model_instructions_file,
                        use_azure_provider=use_azure_provider,
                        rows=rows,
                        cols=cols,
                    )
                    _CODEX_PTY_SESSIONS[session_id] = entry
                except asyncio.CancelledError:
                    # _prepare_session already released everything it had built;
                    # never swallow cancellation into a result dict.
                    logger.warning('[codex-pty %s] cancelled during session prepare', session_id)
                    raise
                except Exception as exc:  # noqa: BLE001
                    import traceback

                    logger.error('[codex-pty %s] session prepare failed: %s', session_id, exc)
                    return _error_result(
                        session_id,
                        'session_prepare_error',
                        f'session prepare failed: {exc}\n{traceback.format_exc()}',
                        start_time,
                    )

    try:
        async with entry.lock:
            report = (await entry.session.send_turn(prompt, timeout=timeout)).to_dict()
    except asyncio.CancelledError:
        # The Host is going away mid-turn. Nothing else will close this session,
        # so reclaim it here, then let the cancellation propagate.
        logger.warning('[codex-pty %s] turn cancelled; tearing the session down', session_id)
        with contextlib.suppress(Exception):
            await asyncio.shield(close_codex_interactive_session(session_id=session_id))
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception('[codex-pty %s] turn failed: %s', session_id, exc)
        report = {
            'status': 'error',
            'completed': False,
            'reason': 'turn_error',
            'stderr': f'PTY turn failed: {exc}',
            'latest_screen': await entry.session.screen.latest_screen(),
            'session_reusable': False,
        }
        formatted = _format_turn(session_id=session_id, entry=entry, report=report, start_time=start_time)
        with contextlib.suppress(Exception):
            await close_codex_interactive_session(session_id=session_id)
        return formatted

    formatted = _format_turn(session_id=session_id, entry=entry, report=report, start_time=start_time)
    if not report.get('session_reusable', True):
        # A session whose state cannot be proven reusable must not serve the next
        # turn: drop it so the caller gets a fresh PTY rather than a dead one.
        with contextlib.suppress(Exception):
            await close_codex_interactive_session(session_id=session_id)
    return formatted


async def respond_and_wait_codex_interactive_session(
    *,
    session_id: str,
    interaction_id: str,
    answer: str | None,
    timeout: float = 3600,
) -> dict[str, Any]:
    """Answer a pending question and keep waiting on the same turn."""
    start_time = time.time()
    entry = _CODEX_PTY_SESSIONS.get(session_id)
    if entry is None:
        return _error_result(
            session_id, 'session_missing', f'interactive Codex session does not exist: {session_id}', start_time
        )
    try:
        async with entry.lock:
            report = (await entry.session.respond(interaction_id, answer, timeout=timeout)).to_dict()
    except asyncio.CancelledError:
        logger.warning('[codex-pty %s] interaction cancelled; tearing the session down', session_id)
        with contextlib.suppress(Exception):
            await asyncio.shield(close_codex_interactive_session(session_id=session_id))
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception('[codex-pty %s] interaction %s failed: %s', session_id, interaction_id, exc)
        report = {
            'status': 'interaction_error',
            'completed': False,
            'reason': 'interaction_error',
            'stderr': f'PTY interaction failed: {exc}',
            'latest_screen': await entry.session.screen.latest_screen(),
            'session_reusable': False,
        }
        formatted = _format_turn(session_id=session_id, entry=entry, report=report, start_time=start_time)
        with contextlib.suppress(Exception):
            await close_codex_interactive_session(session_id=session_id)
        return formatted

    formatted = _format_turn(session_id=session_id, entry=entry, report=report, start_time=start_time)
    if not report.get('session_reusable', True):
        with contextlib.suppress(Exception):
            await close_codex_interactive_session(session_id=session_id)
    return formatted


async def close_codex_interactive_session(*, session_id: str) -> dict[str, Any]:
    """Close one runtime-side session. Idempotent."""
    entry = _CODEX_PTY_SESSIONS.get(session_id)
    if entry is None:
        return {'success': True, 'session_id': session_id, 'exit_code': 0, 'stderr': ''}
    await entry.close()
    if _CODEX_PTY_SESSIONS.get(session_id) is entry:
        _CODEX_PTY_SESSIONS.pop(session_id)
    return {'success': True, 'session_id': session_id, 'exit_code': 0, 'stderr': ''}


__all__ = [
    'close_codex_interactive_session',
    'respond_and_wait_codex_interactive_session',
    'run_codex_interactive_session_turn',
]
