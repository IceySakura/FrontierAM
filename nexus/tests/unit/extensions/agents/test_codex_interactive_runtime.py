"""Runtime-layer tests: session registry, result contract, cleanup, isolation.

These exercise ``interactive/runtime.py`` with a stub session so registry
behaviour (double-checked create, per-session lock, drop-on-unusable, idempotent
close) is tested without booting a PTY; the PTY paths are covered in
``test_codex_interactive.py``.
"""

from __future__ import annotations

import asyncio

from types import SimpleNamespace
from typing import Any

import pytest

from nexus.extensions.agents.codex.interactive import runtime as rt
from nexus.extensions.agents.codex.interactive.events import EventBuffer
from nexus.extensions.agents.codex.interactive.session import SessionState, TurnReport


asyncio_test = pytest.mark.asyncio


class _StubScreen:
    async def latest_screen(self) -> str:
        return 'VIEWPORT'


class _StubSession:
    """Stands in for CodexInteractiveSession at the registry boundary."""

    def __init__(self, reports: list[TurnReport] | None = None) -> None:
        self.reports = list(reports or [])
        self.events = EventBuffer()
        self.screen = _StubScreen()
        self.state = SessionState.READY
        self.codex_pid = 4242
        self.closed = 0
        self.prompts: list[str] = []
        self.responses: list[tuple[str, str | None]] = []
        self.concurrent_peak = 0
        self._active = 0

    async def send_turn(self, prompt: str, *, timeout: float | None = None) -> TurnReport:
        self._active += 1
        self.concurrent_peak = max(self.concurrent_peak, self._active)
        try:
            self.prompts.append(prompt)
            await asyncio.sleep(0.05)
            return self.reports.pop(0) if self.reports else _ok_report()
        finally:
            self._active -= 1

    async def respond(self, interaction_id: str, answer: str | None, *, timeout: float | None = None) -> TurnReport:
        self.responses.append((interaction_id, answer))
        return self.reports.pop(0) if self.reports else _ok_report()

    async def close(self) -> None:
        self.closed += 1


class _StubProxy:
    def __init__(self) -> None:
        self.stopped = 0

    async def stop(self) -> None:
        self.stopped += 1


def _ok_report(turn_id: str = 'turn-1') -> TurnReport:
    return TurnReport(
        status='completed',
        completed=True,
        reason='turn_completed',
        thread_id='thread-1',
        turn_id=turn_id,
        latest_screen='VIEWPORT',
    )


def _prepare_kwargs(tmp_path) -> dict[str, Any]:
    return {
        'session_id': 'fresh',
        'working_dir': str(tmp_path / 'workspace'),
        'timeout': 30,
        'version': rt.INTERACTIVE_DEFAULT_CODEX_VERSION,
        'npm_registry': None,
        'binary_url': None,
        'binary_checksum': None,
        'model': None,
        'model_catalog_json': None,
        'mcp_servers': None,
        'extra_env': None,
        'extra_cli_args': None,
        'extra_config_toml': None,
        'use_azure_provider': True,
        'rows': 24,
        'cols': 80,
    }


def _register(
    session_id: str,
    session: _StubSession,
    tmp_path,
) -> rt._InteractiveRuntimeSession:
    entry = rt._InteractiveRuntimeSession(
        session=session,  # type: ignore[arg-type]
        proxy=_StubProxy(),
    )
    rt._CODEX_PTY_SESSIONS[session_id] = entry
    return entry


@pytest.fixture(autouse=True)
def _clear_registry():
    rt._CODEX_PTY_SESSIONS.clear()
    yield
    rt._CODEX_PTY_SESSIONS.clear()


@asyncio_test
async def test_turn_result_carries_the_full_contract(tmp_path):
    session = _StubSession()
    _register('s1', session, tmp_path)
    session.events.append_message({
        'method': 'item/completed',
        'params': {'item': {'itemType': 'commandExecution', 'command': 'ls'}},
    })

    result = await rt.run_codex_interactive_session_turn(session_id='s1', prompt='hello')

    for key in (
        'success',
        'status',
        'session_id',
        'thread_id',
        'turn_id',
        'output',
        'stderr',
        'exit_code',
        'execution_time',
        'events',
        'latest_screen',
        'recent_events',
        'event_cursor',
        'session_reusable',
    ):
        assert key in result, f'missing contract field: {key}'
    assert result['success'] is True
    assert result['turn_id'] == 'turn-1'
    assert result['latest_screen'] == 'VIEWPORT'
    assert len(result['recent_events']) == 1
    assert session.prompts == ['hello']


@asyncio_test
async def test_event_cursor_advances_between_turns(tmp_path):
    session = _StubSession(reports=[_ok_report('t1'), _ok_report('t2')])
    _register('s1', session, tmp_path)
    session.events.append_message({'method': 'item/completed', 'params': {'item': {'itemType': 'a'}}})
    first = await rt.run_codex_interactive_session_turn(session_id='s1', prompt='one')
    assert len(first['recent_events']) == 1

    second = await rt.run_codex_interactive_session_turn(session_id='s1', prompt='two')
    assert second['recent_events'] == []  # already committed, not re-sent


@asyncio_test
async def test_unusable_session_is_dropped_from_registry(tmp_path):
    bad = TurnReport(status='timeout', completed=False, reason='timeout', latest_screen='V', session_reusable=False)
    session = _StubSession(reports=[bad])
    _register('s1', session, tmp_path)

    result = await rt.run_codex_interactive_session_turn(session_id='s1', prompt='hangs')
    assert result['status'] == 'timeout'
    assert result['session_reusable'] is False
    assert 's1' not in rt._CODEX_PTY_SESSIONS
    assert session.closed == 1


@asyncio_test
async def test_needs_input_keeps_session_and_exposes_interaction(tmp_path):
    waiting = TurnReport(
        status='needs_user_input',
        completed=False,
        reason='user_input',
        latest_screen='V',
        interaction={'interaction_id': 'req-1', 'kind': 'user_input', 'method': 'item/tool/requestUserInput'},
    )
    session = _StubSession(reports=[waiting, _ok_report('t9')])
    _register('s1', session, tmp_path)

    first = await rt.run_codex_interactive_session_turn(session_id='s1', prompt='risky')
    assert first['status'] == 'needs_user_input'
    assert first['interaction']['kind'] == 'user_input'
    assert first['interaction']['interaction_id'] == 'req-1'
    assert 's1' in rt._CODEX_PTY_SESSIONS  # must survive to be answered

    second = await rt.respond_and_wait_codex_interactive_session(session_id='s1', interaction_id='req-1', answer='yes')
    assert second['success'] is True
    assert session.responses == [('req-1', 'yes')]


@asyncio_test
async def test_same_session_turns_serialize(tmp_path):
    session = _StubSession(reports=[_ok_report('t1'), _ok_report('t2')])
    _register('s1', session, tmp_path)
    await asyncio.gather(
        rt.run_codex_interactive_session_turn(session_id='s1', prompt='a'),
        rt.run_codex_interactive_session_turn(session_id='s1', prompt='b'),
    )
    # The per-session lock must prevent interleaved writes into one PTY.
    assert session.concurrent_peak == 1
    assert sorted(session.prompts) == ['a', 'b']


@asyncio_test
async def test_distinct_sessions_are_isolated(tmp_path):
    s1, s2 = _StubSession(), _StubSession()
    _register('one', s1, tmp_path)
    _register('two', s2, tmp_path)
    r1, r2 = await asyncio.gather(
        rt.run_codex_interactive_session_turn(session_id='one', prompt='first'),
        rt.run_codex_interactive_session_turn(session_id='two', prompt='second'),
    )
    assert r1['session_id'] == 'one'
    assert r2['session_id'] == 'two'
    assert s1.prompts == ['first']
    assert s2.prompts == ['second']


@asyncio_test
async def test_missing_session_reports_cleanly():
    result = await rt.respond_and_wait_codex_interactive_session(session_id='nope', interaction_id='x', answer='y')
    assert result['success'] is False
    assert result['status'] == 'session_missing'
    assert result['event_cursor']['count'] == 0


@asyncio_test
async def test_close_is_idempotent_and_stops_proxy(tmp_path):
    session = _StubSession()
    entry = _register('s1', session, tmp_path)

    assert (await rt.close_codex_interactive_session(session_id='s1'))['success'] is True
    assert (await rt.close_codex_interactive_session(session_id='s1'))['success'] is True
    assert session.closed == 1
    assert entry.proxy.stopped == 1
    assert 's1' not in rt._CODEX_PTY_SESSIONS


@asyncio_test
async def test_close_failure_keeps_session_registered_for_retry(tmp_path):
    from nexus.extensions.agents.codex.interactive.session import CodexInteractiveSession
    from nexus.runtime.server.services.responses_proxy import ResponsesProxy

    session = CodexInteractiveSession(
        codex_argv=['codex'],
        env={},
        working_dir=str(tmp_path),
        session_dir=str(tmp_path / 'session'),
    )
    invalid_socket_dir = tmp_path / 'socket-file'
    invalid_socket_dir.write_text('not a directory')
    session._socket_dir = str(invalid_socket_dir)
    entry = rt._InteractiveRuntimeSession(session=session, proxy=ResponsesProxy(sio=object()))  # type: ignore[arg-type]
    rt._CODEX_PTY_SESSIONS['retry'] = entry

    with pytest.raises(RuntimeError, match='cleanup failed'):
        await rt.close_codex_interactive_session(session_id='retry')

    assert rt._CODEX_PTY_SESSIONS['retry'] is entry
    invalid_socket_dir.unlink()
    result = await rt.close_codex_interactive_session(session_id='retry')
    assert result['success'] is True
    assert 'retry' not in rt._CODEX_PTY_SESSIONS


@asyncio_test
async def test_failed_prepare_cleanup_reports_session_cleanup_error(tmp_path):
    from nexus.extensions.agents.codex.interactive.session import CodexInteractiveSession

    session_dir = tmp_path / 'session'
    session_dir.mkdir()
    invalid_socket_dir = session_dir / 'socket-file'
    invalid_socket_dir.write_text('not a directory')
    session = CodexInteractiveSession(
        codex_argv=['codex'],
        env={},
        working_dir=str(tmp_path),
        session_dir=str(session_dir),
    )
    session._socket_dir = str(invalid_socket_dir)

    errors = await rt._cleanup_failed_prepare(session=session, proxy=None, session_dir=str(session_dir))

    assert len(errors) == 1
    assert errors[0][0] == 'session'
    assert 'socket directory' in str(errors[0][1])
    assert not session_dir.exists()


@asyncio_test
async def test_turn_exception_closes_session_and_reports(tmp_path):
    class _Boom(_StubSession):
        async def send_turn(self, prompt: str, *, timeout: float | None = None) -> TurnReport:
            raise RuntimeError('pty exploded')

    session = _Boom()
    _register('s1', session, tmp_path)
    result = await rt.run_codex_interactive_session_turn(session_id='s1', prompt='x')
    assert result['status'] == 'error'
    assert 'pty exploded' in result['stderr']
    assert 's1' not in rt._CODEX_PTY_SESSIONS


@asyncio_test
async def test_prepare_failure_is_reported_not_raised(monkeypatch):
    async def _fail(*args: Any, **kwargs: Any):
        raise RuntimeError('install blew up')

    monkeypatch.setattr(rt, '_prepare_session', _fail)
    result = await rt.run_codex_interactive_session_turn(session_id='fresh', prompt='x')
    assert result['success'] is False
    assert result['status'] == 'session_prepare_error'
    assert 'install blew up' in result['stderr']
    assert 'fresh' not in rt._CODEX_PTY_SESSIONS


@asyncio_test
async def test_proxy_start_failure_removes_private_session_dir(monkeypatch, tmp_path):
    from nexus.extensions.agents.codex import runtime_setup
    from nexus.runtime import context as runtime_context
    from nexus.runtime.server.services import responses_proxy

    session_dir = tmp_path / 'session'
    stopped = 0

    async def _install(**kwargs: Any):
        return tmp_path / 'codex'

    class _FailingProxy:
        def __init__(self, **kwargs: Any) -> None:
            pass

        async def start(self) -> None:
            raise RuntimeError('proxy failed')

        async def stop(self) -> None:
            nonlocal stopped
            stopped += 1

    monkeypatch.setattr(rt, '_session_paths', lambda session_id: (str(session_dir), str(session_dir / 'codex-home')))
    monkeypatch.setattr(rt, 'find_free_port', lambda: 12345)
    monkeypatch.setattr(runtime_context, 'require_runtime_control', lambda: SimpleNamespace(sio=object()))
    monkeypatch.setattr(runtime_setup, 'install_codex_binary', _install)
    monkeypatch.setattr(responses_proxy, 'ResponsesProxy', _FailingProxy)

    with pytest.raises(RuntimeError, match='proxy failed'):
        await rt._prepare_session(**_prepare_kwargs(tmp_path))

    assert stopped == 1
    assert not session_dir.exists()


@asyncio_test
async def test_concurrent_first_turns_create_one_session(monkeypatch, tmp_path):
    created = []

    async def _prepare(session_id: str, **kwargs: Any):
        created.append(session_id)
        await asyncio.sleep(0.05)  # widen the race window
        return rt._InteractiveRuntimeSession(
            session=_StubSession(reports=[_ok_report('t1'), _ok_report('t2')]),  # type: ignore[arg-type]
            proxy=_StubProxy(),
        )

    monkeypatch.setattr(rt, '_prepare_session', _prepare)
    await asyncio.gather(
        rt.run_codex_interactive_session_turn(session_id='race', prompt='a'),
        rt.run_codex_interactive_session_turn(session_id='race', prompt='b'),
    )
    # Double-checked locking: exactly one app-server/proxy per session_id.
    assert created == ['race']


# --------------------------------------------------------------------------- #
# Success/completed/reusable truth table
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ('status', 'completed', 'reusable', 'expect_success'),
    [
        ('completed', True, True, True),
        ('needs_user_input', False, True, True),
        # Errors must NOT be reported as success just because the session lives.
        ('interaction_not_found', False, True, False),
        ('timeout', False, False, False),
        ('process_exit', False, False, False),
        ('event_source_exit', False, False, False),
        ('failed', False, False, False),
        ('interrupted', False, False, False),
        ('error', False, False, False),
    ],
)
@asyncio_test
async def test_success_semantics_truth_table(tmp_path, status, completed, reusable, expect_success):
    report = TurnReport(
        status=status,
        completed=completed,
        reason=status,
        latest_screen='V',
        session_reusable=reusable,
    )
    session = _StubSession(reports=[report])
    _register('s1', session, tmp_path)
    result = await rt.run_codex_interactive_session_turn(session_id='s1', prompt='x')
    assert result['success'] is expect_success, f'{status} should report success={expect_success}'
    assert result['status'] == status


@asyncio_test
async def test_interaction_not_found_is_not_success(tmp_path):
    """Regression: a reusable session used to mask this error as success."""
    report = TurnReport(
        status='interaction_not_found',
        completed=False,
        reason='interaction_not_found',
        latest_screen='V',
        session_reusable=True,
    )
    session = _StubSession(reports=[report])
    _register('s1', session, tmp_path)
    result = await rt.respond_and_wait_codex_interactive_session(session_id='s1', interaction_id='gone', answer='y')
    assert result['success'] is False
    assert 's1' in rt._CODEX_PTY_SESSIONS  # still reusable, just not successful


@asyncio_test
async def test_completed_turn_propagates_output(tmp_path):
    report = TurnReport(
        status='completed',
        completed=True,
        reason='turn_completed',
        thread_id='th',
        turn_id='t1',
        output='FINAL TEXT',
        latest_screen='V',
    )
    session = _StubSession(reports=[report])
    _register('s1', session, tmp_path)
    result = await rt.run_codex_interactive_session_turn(session_id='s1', prompt='x')
    assert result['output'] == 'FINAL TEXT'


# --------------------------------------------------------------------------- #
# Cancellation
# --------------------------------------------------------------------------- #


@asyncio_test
async def test_cancelled_turn_closes_the_session_and_reraises(tmp_path):
    class _Hanging(_StubSession):
        async def send_turn(self, prompt: str, *, timeout: float | None = None) -> TurnReport:
            await asyncio.sleep(30)
            raise AssertionError('should have been cancelled')

    session = _Hanging()
    entry = _register('s1', session, tmp_path)
    task = asyncio.create_task(rt.run_codex_interactive_session_turn(session_id='s1', prompt='x'))
    await asyncio.sleep(0.1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    # No leak: session closed, proxy stopped, registry entry gone.
    assert session.closed == 1
    assert entry.proxy.stopped == 1
    assert 's1' not in rt._CODEX_PTY_SESSIONS


@asyncio_test
async def test_cancelled_interaction_closes_the_session_and_reraises(tmp_path):
    class _Hanging(_StubSession):
        async def respond(self, interaction_id, answer, *, timeout=None) -> TurnReport:
            await asyncio.sleep(30)
            raise AssertionError('should have been cancelled')

    session = _Hanging()
    entry = _register('s1', session, tmp_path)
    task = asyncio.create_task(
        rt.respond_and_wait_codex_interactive_session(session_id='s1', interaction_id='r', answer='y')
    )
    await asyncio.sleep(0.1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert session.closed == 1
    assert entry.proxy.stopped == 1
    assert 's1' not in rt._CODEX_PTY_SESSIONS


@asyncio_test
async def test_cancel_during_install_removes_private_session_dir(monkeypatch, tmp_path):
    """Cancellation before registration must not leave a private CODEX_HOME."""
    from nexus.extensions.agents.codex import runtime_setup
    from nexus.runtime import context as runtime_context

    session_dir = tmp_path / 'session'
    installing = asyncio.Event()

    async def _install(**kwargs: Any):
        installing.set()
        await asyncio.sleep(30)

    monkeypatch.setattr(rt, '_session_paths', lambda session_id: (str(session_dir), str(session_dir / 'codex-home')))
    monkeypatch.setattr(runtime_context, 'require_runtime_control', lambda: SimpleNamespace(sio=object()))
    monkeypatch.setattr(runtime_setup, 'install_codex_binary', _install)

    task = asyncio.create_task(rt._prepare_session(**_prepare_kwargs(tmp_path)))
    await installing.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not session_dir.exists()


@asyncio_test
async def test_cancellation_is_never_swallowed_into_a_result(tmp_path):
    """CancelledError must propagate, not become an 'error' status dict."""

    class _Cancelling(_StubSession):
        async def send_turn(self, prompt: str, *, timeout: float | None = None) -> TurnReport:
            raise asyncio.CancelledError

    _register('s1', _Cancelling(), tmp_path)
    with pytest.raises(asyncio.CancelledError):
        await rt.run_codex_interactive_session_turn(session_id='s1', prompt='x')
