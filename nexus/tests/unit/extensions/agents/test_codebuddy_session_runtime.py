from __future__ import annotations

import asyncio
import inspect
import json

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import anyio
import pytest

from nexus.extensions.agents.codebuddy import session_runtime
from nexus.extensions.agents.codebuddy.session_runtime import (
    CodeBuddySessionExtension,
    _CodeBuddyAcpSession,
    close_codebuddy_session,
    interrupt_codebuddy_session,
    run_codebuddy_session_turn,
    steer_codebuddy_session,
)
from nexus.extensions.agents.installed_common.acp import AcpRemoteError


def _message_chunk(text: str) -> dict[str, Any]:
    return {
        'sessionId': 'remote-1',
        'update': {'sessionUpdate': 'agent_message_chunk', 'content': {'type': 'text', 'text': text}},
    }


class FakeProc:
    """Record process shutdown without launching a CLI."""

    def __init__(self) -> None:
        self.returncode: int | None = None
        self.launch_env: dict[str, str] = {}
        self.terminated = False
        self.killed = False
        self.stdin = object()
        self.stdout = asyncio.StreamReader()
        self.stderr = asyncio.StreamReader()
        self.stderr.feed_eof()

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    async def wait(self) -> int:
        return self.returncode or 0


class FakeProxy:
    """Record proxy ownership and repeated shutdown calls."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.stopped = 0

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        self.stopped += 1


class FakeConnection:
    """Provide controllable prompt terminals and cancel notification drains."""

    def __init__(self, *_args: Any, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.prompts: list[str] = []
        self.cancelled: list[str] = []
        self.closed = 0
        self.started = False
        self.deferred_permission_handler = kwargs.get('deferred_permission_handler')
        self.updates: list[dict[str, Any]] = []
        self.prompt_entered = asyncio.Event()
        self.prompt_release = asyncio.Event()
        self.cancel_entered = asyncio.Event()
        self.cancel_release = asyncio.Event()
        self.cancel_release.set()
        self.block_prompt = False
        self.ack_cancel = True
        self.stop_reason = 'end_turn'
        self.mcp_servers: list[dict[str, Any]] | None = None
        self.capabilities = {'agentCapabilities': {'mcpCapabilities': {'http': True}}}
        self.elicitation_handler: Any = None
        self.client_capabilities: dict[str, Any] | None = None

    def start(self) -> None:
        self.started = True

    async def initialize(self, *, timeout: float, client_capabilities: dict[str, Any] | None = None) -> dict[str, Any]:
        self.client_capabilities = client_capabilities
        return self.capabilities

    async def new_session(self, *, cwd: str, timeout: float, mcp_servers: list[dict[str, Any]]) -> str:
        self.mcp_servers = mcp_servers
        return 'remote-1'

    def drain_updates(self, *, session_id: str | None = None) -> list[dict[str, Any]]:
        updates, self.updates = self.updates, []
        return updates

    async def prompt(self, *, session_id: str, text: str, timeout: float) -> dict[str, Any]:
        self.prompts.append(text)
        self.updates.append(_message_chunk(f'reply:{text}'))
        self.prompt_entered.set()
        if self.block_prompt:
            await self.prompt_release.wait()
        return {'stopReason': self.stop_reason}

    async def cancel(self, *, session_id: str) -> None:
        self.cancelled.append(session_id)
        self.cancel_entered.set()
        await self.cancel_release.wait()
        if self.ack_cancel:
            self.stop_reason = 'cancelled'
            self.prompt_release.set()

    async def close(self) -> None:
        self.closed += 1


def _default_start_config() -> dict[str, Any]:
    return {
        name: parameter.default
        for name, parameter in inspect.signature(run_codebuddy_session_turn).parameters.items()
        if name not in {'session_id', 'prompt', 'parent_tool_use_id', 'timeout', 'relay_turn_id'}
    }


def _make_session(*, register: bool = True, **overrides: Any) -> _CodeBuddyAcpSession:
    session = _CodeBuddyAcpSession(
        session_id='s1',
        remote_session_id='remote-1',
        proc=overrides.pop('proc', FakeProc()),
        proxy=overrides.pop('proxy', FakeProxy()),
        connection=overrides.pop('connection', FakeConnection()),
        cwd='/workspace',
        started_at=0.0,
        stderr_lines=[],
        stderr_task=asyncio.create_task(asyncio.sleep(3600)),
        model_id='nexus-proxy-model',
        start_config=overrides.pop('start_config', _default_start_config()),
        **overrides,
    )
    if register:
        session_runtime._SESSIONS['s1'] = session
    return session


@pytest.fixture(autouse=True)
async def _clear_sessions() -> AsyncIterator[None]:
    session_runtime._SESSIONS.clear()
    session_runtime._SESSION_STARTS.clear()
    session_runtime._SESSION_LOCKS.clear()
    session_runtime._SESSION_CLEANUPS.clear()
    yield
    for session in list(session_runtime._SESSIONS.values()):
        await session.close()
    session_runtime._SESSIONS.clear()
    session_runtime._SESSION_STARTS.clear()
    session_runtime._SESSION_LOCKS.clear()
    session_runtime._SESSION_CLEANUPS.clear()


def _patch_startup_dependencies(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, connection_type: type[FakeConnection] = FakeConnection
) -> dict[str, Any]:
    resources: dict[str, Any] = {'processes': [], 'proxies': [], 'connections': [], 'installs': 0}

    async def fake_install(**_kwargs: Any) -> Path:
        resources['installs'] += 1
        return tmp_path / 'codebuddy'

    async def fake_spawn(*args: Any, **kwargs: Any) -> FakeProc:
        proc = FakeProc()
        proc.launch_env = kwargs['env']
        resources['processes'].append(proc)
        resources['cmd'] = args
        resources['spawn_kwargs'] = kwargs
        return proc

    def make_proxy(**kwargs: Any) -> FakeProxy:
        proxy = FakeProxy(**kwargs)
        resources['proxies'].append(proxy)
        return proxy

    def make_connection(*args: Any, **kwargs: Any) -> FakeConnection:
        connection = connection_type(*args, **kwargs)
        resources['connections'].append(connection)
        return connection

    monkeypatch.setattr(session_runtime, '_SESSION_STATE_ROOT', tmp_path / 'sessions')
    monkeypatch.setattr(session_runtime, 'install_codebuddy_via_npm', fake_install)
    monkeypatch.setattr(session_runtime, 'find_free_port', lambda: 18001)
    monkeypatch.setattr(session_runtime, 'find_nexus_runtime_binary', lambda _name: '/usr/bin/node')
    monkeypatch.setattr(session_runtime, 'AcpConnection', make_connection)
    monkeypatch.setattr(session_runtime.asyncio, 'create_subprocess_exec', fake_spawn)
    monkeypatch.setattr('nexus.runtime.context.require_runtime_control', lambda: SimpleNamespace(sio=object()))
    monkeypatch.setattr('nexus.runtime.server.services.openai_proxy.OpenAIProxy', make_proxy)
    monkeypatch.setattr('nexus.utils.shell.get_os_env_without_ld_library_path', lambda: {})
    return resources


def _startup_kwargs(tmp_path: Path, **overrides: Any) -> dict[str, Any]:
    return {**_default_start_config(), 'session_id': 's1', 'working_dir': str(tmp_path / 'workspace'), **overrides}


@asynccontextmanager
async def extension_factory(
    *, session_id: str, sio: Any, connection: Any, config: dict[str, Any]
) -> AsyncIterator[CodeBuddySessionExtension]:
    assert not connection.started, 'Extension must bind before the ACP reader starts'
    assert session_id, 'Factory must receive the logical session identity'
    assert sio is not None, 'Factory must receive the runtime transport'
    events = config['events']
    events.append('enter')
    try:
        if config.get('enter_error'):
            raise RuntimeError('extension enter failed')

        async def permission_handler(*_args: Any, **_kwargs: Any) -> dict[str, Any] | None:
            return None

        async def elicitation_handler(_params: dict[str, Any]) -> dict[str, Any]:
            return {'action': 'cancel'}

        yield CodeBuddySessionExtension(
            permission_handler=None if config.get('no_handler') else permission_handler,
            mcp_servers=tuple(config.get('mcp_servers', ())),
            client_capabilities=config.get('client_capabilities'),
            elicitation_handler=elicitation_handler if config.get('elicitation') else None,
            on_turn_end=config.get('on_turn_end'),
        )
    finally:
        events.append('exit')
        if config.get('exit_error'):
            raise RuntimeError('extension exit failed')


class FakeTurnRelay:
    """Record prompt boundaries and count the updates published inside them."""

    def __init__(self) -> None:
        self.events: list[str] = []
        self.frames = 0
        self.session_ids: list[str | None] = []

    def begin_turn(self, turn_id: str, *, session_id: str | None = None) -> None:
        self.events.append(f'begin:{turn_id}')
        self.session_ids.append(session_id)
        self.frames = 3

    def end_turn(self) -> int:
        self.events.append('end')
        return self.frames

    async def flush(self) -> None:
        self.events.append('flush')


async def test_a_relayed_turn_reports_its_pushed_frames() -> None:
    relay = FakeTurnRelay()
    session = _make_session(update_relay=relay)

    relayed = await run_codebuddy_session_turn(session_id='s1', prompt='one', relay_turn_id='turn-1')
    plain = await run_codebuddy_session_turn(session_id='s1', prompt='two')

    assert relay.events == ['begin:turn-1', 'end', 'flush'], 'Frames must be flushed before the result returns'
    assert relayed['relay_frames'] == 3
    assert plain['relay_frames'] is None, 'A prompt without a turn id must not be relayed'
    assert cast(FakeConnection, session.connection).prompts == ['one', 'two']


async def test_turn_end_hook_runs_after_each_prompt_has_finished_including_a_timed_out_one() -> None:
    connection = FakeConnection()
    finished: list[bool] = []

    def on_turn_end() -> None:
        assert session.active_turn is not None
        finished.append(session.active_turn.prompt_task.done())

    session = _make_session(connection=connection, on_turn_end=on_turn_end)
    await session.send_user_message('one', timeout=1)
    connection.block_prompt = True
    timed_out = await session.send_user_message('two', timeout=0.01)

    assert timed_out['status'] == 'timeout'
    assert finished == [True, True], 'The prompt, and so its deferred permissions, must be over first'


async def test_turns_reuse_one_session_without_leaking_updates() -> None:
    session = _make_session()
    first = await run_codebuddy_session_turn(session_id='s1', prompt='one')
    second = await run_codebuddy_session_turn(session_id='s1', prompt='two')
    assert cast(FakeConnection, session.connection).prompts == ['one', 'two'], (
        'Both prompts must use the original process'
    )
    assert first['output'] == 'reply:one', 'First turn must collect its own updates'
    assert second['output'] == 'reply:two', 'Updates must not leak into the next turn'
    assert second['success'], 'A normal terminal is successful'
    assert not second['partial'], 'A normal terminal is complete'


async def test_cancelled_peer_terminal_remains_successful() -> None:
    session = _make_session()
    cast(FakeConnection, session.connection).stop_reason = 'cancelled'
    result = await session.send_user_message('one', timeout=1)
    assert result['success'], 'Soft cancel is controlled'
    assert result['status'] == 'cancelled', 'Peer cancellation must retain its status'
    assert result['partial'], 'Cancellation retains a partial trajectory'


async def test_idle_interrupt_never_sends_a_cancel_for_a_future_turn() -> None:
    session = _make_session()
    result = await session.interrupt()
    assert result['success'], 'An idle interrupt is a no-op'
    assert not result['stopped'], 'An idle session has no target turn'
    assert cast(FakeConnection, session.connection).cancelled == [], 'An unscoped idle cancel could hit a future prompt'
    await session.send_user_message('next', timeout=1)
    assert session.reusable, 'Idle interruption must not quarantine the session'


async def test_interrupt_waits_for_original_terminal_and_preserves_partial() -> None:
    session = _make_session()
    connection = cast(FakeConnection, session.connection)
    connection.block_prompt = True
    turn_task = asyncio.create_task(session.send_user_message('one', timeout=1))
    await connection.prompt_entered.wait()
    result = await session.interrupt(grace_timeout=0.2)
    turn_result = await turn_task
    assert result['stopped'], 'Interrupt must confirm the target terminal'
    assert result['output'] == 'reply:one', 'Interrupt must return the target partial trajectory'
    assert turn_result['status'] == 'cancelled', 'The peer terminal must remain visible'
    assert session.reusable, 'Terminal cancellation permits reuse'
    assert not cast(FakeProc, session.proc).terminated, 'Soft cancellation must preserve the CLI context'


async def test_double_task_cancel_cannot_interrupt_prompt_cleanup() -> None:
    session = _make_session()
    connection = cast(FakeConnection, session.connection)
    connection.block_prompt = True
    connection.cancel_release.clear()
    turn_task = asyncio.create_task(session.send_user_message('one', timeout=1))
    await connection.prompt_entered.wait()
    turn = session.active_turn
    assert turn is not None, 'The blocked prompt must retain its active turn'
    turn_task.cancel()
    await connection.cancel_entered.wait()
    turn_task.cancel()
    await asyncio.sleep(0)
    assert not turn_task.done(), 'Repeated cancellation must not orphan the ACP prompt'
    connection.cancel_release.set()
    with pytest.raises(asyncio.CancelledError):
        await turn_task
    assert turn.done.done(), 'Every exit must settle the active turn'
    assert session.active_turn is None, 'Every exit must clear the active turn'
    assert connection.cancelled == ['remote-1'], 'Repeated cancellation shares one handshake'
    assert session.reusable, 'A confirmed terminal allows reuse'
    await session.send_user_message('next', timeout=1)


async def test_interrupt_snapshot_blocks_next_turn_until_cancel_drain() -> None:
    session = _make_session()
    connection = cast(FakeConnection, session.connection)
    connection.block_prompt = True
    connection.cancel_release.clear()
    first = asyncio.create_task(session.send_user_message('one', timeout=1))
    await connection.prompt_entered.wait()
    interrupt = asyncio.create_task(session.interrupt(grace_timeout=0.5))
    await connection.cancel_entered.wait()
    connection.prompt_release.set()
    second = asyncio.create_task(session.send_user_message('two', timeout=1))
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert connection.prompts == ['one'], 'A terminal alone cannot release a pending cancel write into the next turn'
    connection.cancel_release.set()
    result = await interrupt
    await asyncio.gather(first, second)
    assert result['output'] == 'reply:one', 'Interrupt must retain its original target'
    assert connection.cancelled == ['remote-1'], 'Exactly one notification belongs to the first turn'


async def test_cancel_write_timeout_quarantines_even_after_prompt_terminal() -> None:
    session = _make_session()
    connection = cast(FakeConnection, session.connection)
    connection.block_prompt = True
    connection.cancel_release.clear()
    turn_task = asyncio.create_task(session.send_user_message('one', timeout=1))
    await connection.prompt_entered.wait()
    interrupt_task = asyncio.create_task(session.interrupt(grace_timeout=0.01))
    await connection.cancel_entered.wait()
    connection.prompt_release.set()
    interrupt_result, turn_result = await asyncio.gather(interrupt_task, turn_task)
    assert not interrupt_result['success'], 'A terminal cannot make a pending cancel write safe for the next turn'
    assert not session.reusable, 'Unknown cancel write state must quarantine the transport'
    assert turn_result['status'] == 'failed', 'A timed-out cancel write is a protocol failure'
    assert turn_result['partial'], 'An unsafe cancellation must be reported as a partial trajectory'


async def test_repeated_interrupts_share_one_deadline_and_cancel_task(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(session_runtime, '_CANCEL_GRACE_TIMEOUT', 0.03)
    session = _make_session()
    connection = cast(FakeConnection, session.connection)
    connection.block_prompt = True
    connection.ack_cancel = False
    turn_task = asyncio.create_task(session.send_user_message('one', timeout=1))
    await connection.prompt_entered.wait()
    first = asyncio.create_task(session.interrupt(grace_timeout=0.03))
    await connection.cancel_entered.wait()
    second = asyncio.create_task(session.interrupt(grace_timeout=10))
    results = await asyncio.wait_for(asyncio.gather(first, second), timeout=0.3)
    await asyncio.gather(turn_task, return_exceptions=True)
    assert all(not result['success'] and result['quarantined'] for result in results), 'No terminal means no reuse'
    assert connection.cancelled == ['remote-1'], 'A later interrupt must reuse the existing cancellation budget'
    assert not session.reusable, 'Unconfirmed cancellation must quarantine the session'
    assert session.active_turn is None, 'Timed-out cancellation must settle the turn'


async def test_timeout_preserves_partial_and_quarantines_without_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(session_runtime, '_CANCEL_GRACE_TIMEOUT', 0.01)
    session = _make_session()
    connection = cast(FakeConnection, session.connection)
    connection.block_prompt = True
    connection.ack_cancel = False
    result = await session.send_user_message('halfway', timeout=0.01)
    assert result['status'] == 'timeout', 'The prompt budget must report timeout'
    assert not result['success'], 'The prompt budget is a business failure'
    assert result['output'] == 'reply:halfway', 'Partial trajectory must survive timeout'
    assert result['partial'], 'Timeout output is incomplete'
    with pytest.raises(RuntimeError, match='quarantined'):
        await run_codebuddy_session_turn(session_id='s1', prompt='must not restart')


async def test_timeout_with_terminal_allows_context_reuse(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(session_runtime, '_CANCEL_GRACE_TIMEOUT', 0.1)
    session = _make_session()
    connection = cast(FakeConnection, session.connection)
    connection.block_prompt = True
    result = await session.send_user_message('one', timeout=0.01)
    assert result['status'] == 'timeout', 'A late terminal does not change the business timeout'
    assert not result['success'], 'Business timeout remains a failed turn'
    assert session.reusable, 'Terminal permits later reuse'
    await session.send_user_message('two', timeout=1)
    assert connection.prompts == ['one', 'two'], 'Timeout must not discard confirmed CLI context'


async def test_protocol_failure_always_settles_turn_and_quarantines(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _make_session()

    async def fail_prompt(**_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError('protocol broke')

    monkeypatch.setattr(session.connection, 'prompt', fail_prompt)
    with pytest.raises(RuntimeError, match='protocol broke'):
        await session.send_user_message('one', timeout=1)
    assert session.active_turn is None, 'Exceptional exits must clear the active turn'
    assert not session.reusable, 'Exceptional exits must not leave a reusable session'


async def test_startup_config_mismatch_never_restarts_existing_session() -> None:
    session = _make_session()
    with pytest.raises(ValueError, match='configuration'):
        await run_codebuddy_session_turn(session_id='s1', prompt='one', model_id='different')
    assert cast(FakeConnection, session.connection).prompts == [], 'Config drift must not start another prompt'
    assert not cast(FakeProc, session.proc).terminated, 'Reject drift without losing context'


async def test_dead_session_requires_explicit_close() -> None:
    session = _make_session()
    cast(FakeProc, session.proc).returncode = 1
    with pytest.raises(RuntimeError, match='not running'):
        await run_codebuddy_session_turn(session_id='s1', prompt='one')
    result = await interrupt_codebuddy_session(session_id='s1')
    assert not result['success'], 'Interrupting a dead session must fail'
    assert cast(FakeConnection, session.connection).cancelled == [], 'A dead session has no cancellable peer'


async def test_close_rejects_active_turn_without_harming_session() -> None:
    session = _make_session()
    connection = cast(FakeConnection, session.connection)
    connection.block_prompt = True
    task = asyncio.create_task(session.send_user_message('one', timeout=1))
    await connection.prompt_entered.wait()
    result = await close_codebuddy_session(session_id='s1')
    assert not result['success'], 'Busy-close must be rejected'
    assert not session.closing, 'Busy-close is non-destructive'
    connection.prompt_release.set()
    await task


async def test_close_is_idempotent_and_keeps_stable_creation_lock() -> None:
    session = _make_session()
    first = await close_codebuddy_session(session_id='s1')
    creation_lock = session_runtime._SESSION_LOCKS['s1']
    second = await close_codebuddy_session(session_id='s1')
    assert first['success'], 'First close must clean up the session'
    assert second['success'], 'Closing an absent session is a no-op'
    assert cast(FakeConnection, session.connection).closed == 1, 'ACP must be closed exactly once'
    assert session.proxy.stopped == 1, 'Proxy must be stopped exactly once'
    assert session_runtime._SESSION_LOCKS['s1'] is creation_lock, 'Waiters must never split across creation locks'


async def test_close_race_rejects_turn_that_already_looked_up_session() -> None:
    session = _make_session()
    await session.close()
    with pytest.raises(RuntimeError, match='closing|closed'):
        await session.send_user_message('stale lookup', timeout=1)
    assert cast(FakeConnection, session.connection).prompts == [], (
        'A stale lookup must not start a new turn after close'
    )


async def test_repeated_close_and_cancellation_share_one_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _make_session()
    connection = cast(FakeConnection, session.connection)
    entered, release = asyncio.Event(), asyncio.Event()

    async def blocking_close() -> None:
        entered.set()
        await release.wait()
        connection.closed += 1

    monkeypatch.setattr(connection, 'close', blocking_close)
    first = asyncio.create_task(session.close())
    await entered.wait()
    second = asyncio.create_task(session.close())
    first.cancel()
    await asyncio.sleep(0)
    first.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert (await second)['success'], 'Other close callers must observe the same successful cleanup'
    assert connection.closed == 1, 'Double cancellation must not double-close ACP'
    assert session.proxy.stopped == 1, 'Double cancellation must not double-close the proxy'


async def test_interrupt_during_startup_prevents_first_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    entered, release = asyncio.Event(), asyncio.Event()
    session = _make_session(register=False)

    async def blocking_start(**_kwargs: Any) -> _CodeBuddyAcpSession:
        entered.set()
        await release.wait()
        return session

    monkeypatch.setattr(session_runtime, '_start_session', blocking_start)
    task = asyncio.create_task(run_codebuddy_session_turn(session_id='s1', prompt='one'))
    await entered.wait()
    first = await interrupt_codebuddy_session(session_id='s1')
    second = await interrupt_codebuddy_session(session_id='s1')
    release.set()
    result = await task
    assert first == second, 'Startup interruption must be idempotent'
    assert first['pending'], 'Startup interruption cannot claim an ACP terminal yet'
    assert result['status'] == 'cancelled', 'Pending cancellation must reach the turn caller'
    assert cast(FakeConnection, session.connection).prompts == [], 'No first prompt after interrupt'
    assert cast(FakeProc, session.proc).terminated, 'Unpublished startup process must be closed'
    assert session.proxy.stopped == 1, 'Unpublished startup proxy must be closed'


async def test_extension_binds_before_reader_and_closes_once(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    resources = _patch_startup_dependencies(monkeypatch, tmp_path)
    events: list[str] = []
    server = {'type': 'http', 'name': 'tools', 'url': 'http://127.0.0.1:1234/mcp', 'headers': []}
    session = await session_runtime._start_session(
        **_startup_kwargs(
            tmp_path,
            extension_factory=f'{__name__}::extension_factory',
            extension_config={'events': events, 'mcp_servers': [server]},
            permission_policy='defer',
            skip_permissions=False,
        )
    )
    assert resources['connections'][0].deferred_permission_handler is not None, 'The extension owns deferred policy'
    assert cast(FakeConnection, session.connection).mcp_servers == [server], (
        'Extension MCP servers must reach session/new'
    )
    await asyncio.gather(session.close(), session.close())
    assert events == ['enter', 'exit'], 'Extension lifetime belongs to the session, not an individual invoke'
    assert resources['proxies'][0].stopped == 1, 'Concurrent close must share cleanup'


async def test_extension_turn_end_hook_reaches_the_session(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_startup_dependencies(monkeypatch, tmp_path)
    ended: list[str] = []
    session = await session_runtime._start_session(
        **_startup_kwargs(
            tmp_path,
            extension_factory=f'{__name__}::extension_factory',
            extension_config={'events': [], 'on_turn_end': lambda: ended.append('end')},
            permission_policy='defer',
            skip_permissions=False,
        )
    )
    await session.send_user_message('one', timeout=1)
    assert ended == ['end']
    await session.close()


@pytest.mark.parametrize('declares', [False, True])
async def test_extension_client_capabilities_and_elicitation_handler_reach_acp(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, declares: bool
) -> None:
    resources = _patch_startup_dependencies(monkeypatch, tmp_path)
    capabilities = {'elicitation': {'form': {}}, '_meta': {'codebuddy.ai': {'question': True}}}
    config: dict[str, Any] = {'events': []}
    if declares:
        config.update(client_capabilities=capabilities, elicitation=True)
    session = await session_runtime._start_session(
        **_startup_kwargs(
            tmp_path,
            extension_factory=f'{__name__}::extension_factory',
            extension_config=config,
            permission_policy='defer',
        )
    )
    connection = resources['connections'][0]
    assert connection.client_capabilities == (capabilities if declares else None), (
        'Only an extension that opts in may change the initialize handshake'
    )
    assert (connection.elicitation_handler is not None) is declares
    await session.close()


@pytest.mark.parametrize('failure', ['initialize', 'new_session', 'enter', 'missing_handler'])
async def test_start_failure_rolls_back_extension_and_resources(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: str
) -> None:
    resources = _patch_startup_dependencies(monkeypatch, tmp_path)
    events: list[str] = []

    async def fail(**_kwargs: Any) -> Any:
        raise RuntimeError('handshake failed')

    if failure in {'initialize', 'new_session'}:
        monkeypatch.setattr(FakeConnection, failure, lambda _self, **kwargs: fail(**kwargs))
    kwargs = _startup_kwargs(
        tmp_path,
        extension_factory=f'{__name__}::extension_factory',
        extension_config={
            'events': events,
            'enter_error': failure == 'enter',
            'no_handler': failure == 'missing_handler',
        },
        permission_policy='defer',
    )
    with pytest.raises((RuntimeError, ValueError)):
        await session_runtime._start_session(**kwargs)
    assert events == ['enter', 'exit'], 'Failed __aenter__ rolls itself back; successful enter exits through the stack'
    assert resources['processes'][0].terminated, 'CLI must be terminated after failed startup'
    assert resources['connections'][0].closed == 1, 'ACP must be torn down'
    assert resources['proxies'][0].stopped == 1, 'Failure rollback must include the proxy'


async def test_extension_exit_failure_does_not_skip_proxy_cleanup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    resources = _patch_startup_dependencies(monkeypatch, tmp_path)
    events: list[str] = []
    session = await session_runtime._start_session(
        **_startup_kwargs(
            tmp_path,
            extension_factory=f'{__name__}::extension_factory',
            extension_config={'events': events, 'exit_error': True},
        )
    )
    result = await session.close()
    assert not result['success'], 'Cleanup failures must remain observable'
    assert 'extension' in result['stderr'], 'Cleanup failure must name its resource'
    assert events == ['enter', 'exit'], 'Failed cleanup still exits the extension exactly once'
    assert resources['proxies'][0].stopped == 1, 'Later cleanup cannot be skipped'


@pytest.mark.parametrize('timeout', [None, 900.0])
async def test_deferred_permission_timeout_reaches_acp_connection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, timeout: float | None
) -> None:
    resources = _patch_startup_dependencies(monkeypatch, tmp_path)
    session = await session_runtime._start_session(**_startup_kwargs(tmp_path, deferred_permission_timeout=timeout))
    assert resources['connections'][0].kwargs['deferred_permission_timeout'] == timeout, (
        'Hosts waiting on humans must control the approval budget, including disabling it'
    )
    await session.close()


@pytest.mark.parametrize('factory', [None, 'bare_registry_name', 'module:factory', 'module::', '::factory'])
async def test_invalid_defer_factory_fails_before_spawning(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, factory: str | None
) -> None:
    resources = _patch_startup_dependencies(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match='extension_factory'):
        await session_runtime._start_session(
            **_startup_kwargs(tmp_path, permission_policy='defer', extension_factory=factory)
        )
    assert resources['processes'] == [], 'Invalid policy configuration must fail before acquiring resources'


@pytest.mark.parametrize('http_supported', [False, True])
async def test_standalone_http_mcp_requires_advertised_capability(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, http_supported: bool
) -> None:
    resources = _patch_startup_dependencies(monkeypatch, tmp_path)

    async def initialize(_self: Any, **_kwargs: Any) -> dict[str, Any]:
        return {'agentCapabilities': {'mcpCapabilities': {'http': http_supported}}}

    monkeypatch.setattr(FakeConnection, 'initialize', initialize)
    servers = [{'type': 'http', 'name': 'standalone', 'url': 'http://127.0.0.1:1234/mcp', 'headers': []}]
    if http_supported:
        session = await session_runtime._start_session(**_startup_kwargs(tmp_path, mcp_servers=servers))
        assert cast(FakeConnection, session.connection).mcp_servers == servers, (
            'Standalone configuration must not require an extension'
        )
        await session.close()
    else:
        with pytest.raises(RuntimeError, match='HTTP MCP'):
            await session_runtime._start_session(**_startup_kwargs(tmp_path, mcp_servers=servers))
        assert resources['proxies'][0].stopped == 1, 'Capability rejection must roll back startup'


async def test_startup_double_cancellation_finishes_cleanup_before_propagating(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    resources = _patch_startup_dependencies(monkeypatch, tmp_path)
    initialize_entered, close_entered, release_close = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def blocking_initialize(_self: Any, **_kwargs: Any) -> dict[str, Any]:
        initialize_entered.set()
        await asyncio.Event().wait()
        raise AssertionError('unreachable')

    async def blocking_close(connection: FakeConnection) -> None:
        close_entered.set()
        await release_close.wait()
        connection.closed += 1

    monkeypatch.setattr(FakeConnection, 'initialize', blocking_initialize)
    monkeypatch.setattr(FakeConnection, 'close', blocking_close)
    task = asyncio.create_task(run_codebuddy_session_turn(session_id='s1', prompt='one', working_dir=str(tmp_path)))
    await initialize_entered.wait()
    task.cancel()
    await close_entered.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done(), 'Cancellation must await the in-progress startup rollback'
    release_close.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert resources['connections'][0].closed == 1, 'No orphaned ACP connection'
    assert resources['proxies'][0].stopped == 1, 'No orphaned proxy'
    assert session_runtime._SESSION_STARTS == {}, 'Cancelled startup must clear its record'
    assert session_runtime._SESSIONS == {}, 'Cancelled startup cannot publish'


async def test_concurrent_sessions_isolate_home_and_proxy_channel(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    resources = _patch_startup_dependencies(monkeypatch, tmp_path)
    common = _startup_kwargs(
        tmp_path,
        extra_env={'HOME': '/must-not-win'},
        config_file_patches=[{'path': '~/.codebuddy/settings.json', 'content': {'enabled': True}}],
    )
    common.pop('session_id')
    first, second = await asyncio.gather(
        session_runtime._start_session(session_id='a', **common),
        session_runtime._start_session(session_id='b', **common),
    )
    assert first.home != second.home, 'Separate sessions require isolated startup settings'
    assert [proxy.kwargs['channel_id'] for proxy in resources['proxies']] == ['a', 'b'], (
        'Proxy routing uses logical identity'
    )
    for session in (first, second):
        home = Path(session.home)
        settings = json.loads((home / '.codebuddy/settings.json').read_text())
        models = json.loads((home / '.codebuddy/models.json').read_text())
        assert settings == {'enabled': True}, 'Home-relative patches must land in the isolated home'
        assert models['models'][0]['url'] == 'http://127.0.0.1:18001/v1/chat/completions', 'CLI must use its own proxy'
        assert cast(FakeProc, session.proc).launch_env['HOME'] == session.home, (
            'extra_env cannot override session isolation'
        )
    await asyncio.gather(first.close(), second.close())


@pytest.mark.parametrize('skip_permissions', [False, True])
async def test_dedicated_permission_flag_overrides_extra_args(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, skip_permissions: bool
) -> None:
    resources = _patch_startup_dependencies(monkeypatch, tmp_path)
    session = await session_runtime._start_session(
        **_startup_kwargs(
            tmp_path, skip_permissions=skip_permissions, extra_cli_args=['--dangerously-skip-permissions', '--verbose']
        )
    )
    assert ('--dangerously-skip-permissions' in resources['cmd']) == skip_permissions, 'Dedicated policy must win'
    assert '--verbose' in resources['cmd'], 'Unrelated CLI arguments must survive'
    await session.close()


def test_http_mcp_capability_detection_accepts_both_peer_shapes() -> None:
    assert session_runtime._supports_http_mcp({'agentCapabilities': {'mcpCapabilities': {'http': True}}}), 'ACP v1'
    assert session_runtime._supports_http_mcp({'agentCapabilities': {'session': {'mcp': {'http': True}}}}), 'ACP v2'
    assert not session_runtime._supports_http_mcp({'agentCapabilities': {}}), 'Missing capability cannot imply support'


@asynccontextmanager
async def owner_task_extension_factory(
    *, session_id: str, sio: Any, connection: Any, config: dict[str, Any]
) -> AsyncIterator[CodeBuddySessionExtension]:
    ready, stop = asyncio.Event(), asyncio.Event()

    async def own_task_group() -> None:
        async with anyio.create_task_group():
            config['events'].append('owner-enter')
            ready.set()
            await stop.wait()
        config['events'].append('owner-exit')

    owner_task = asyncio.create_task(own_task_group())
    try:
        await ready.wait()
        yield CodeBuddySessionExtension()
    finally:
        stop.set()
        await owner_task


async def test_task_affine_extension_uses_owner_task_across_invokes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_startup_dependencies(monkeypatch, tmp_path)
    events: list[str] = []
    session = await asyncio.create_task(
        session_runtime._start_session(
            **_startup_kwargs(
                tmp_path,
                extension_factory=f'{__name__}::owner_task_extension_factory',
                extension_config={'events': events},
            )
        )
    )
    result = await asyncio.create_task(session.close())
    assert result['success'], 'The owner task must exit its own AnyIO task group'
    assert events == ['owner-enter', 'owner-exit'], 'Cross-invoke extension shutdown must join its owner task'


async def test_close_and_get_waiters_share_the_original_creation_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    previous = _make_session()
    replacement = _make_session(register=False)
    close_entered, release_close = asyncio.Event(), asyncio.Event()
    starts = 0

    async def blocking_proxy_stop() -> None:
        close_entered.set()
        await release_close.wait()
        previous.proxy.stopped += 1

    async def start_replacement(**_kwargs: Any) -> _CodeBuddyAcpSession:
        nonlocal starts
        starts += 1
        return replacement

    monkeypatch.setattr(previous.proxy, 'stop', blocking_proxy_stop)
    monkeypatch.setattr(session_runtime, '_start_session', start_replacement)
    close_task = asyncio.create_task(close_codebuddy_session(session_id='s1'))
    await close_entered.wait()
    creation_lock = session_runtime._SESSION_LOCKS['s1']
    first_get = asyncio.create_task(session_runtime._get_or_create_session(session_id='s1', **_default_start_config()))
    second_get = asyncio.create_task(session_runtime._get_or_create_session(session_id='s1', **_default_start_config()))
    repeated_close = asyncio.create_task(close_codebuddy_session(session_id='s1'))
    await asyncio.sleep(0)
    assert not first_get.done(), 'Lookup must wait until the previous session is fully closed'
    assert not repeated_close.done(), 'Concurrent close must join cleanup instead of returning busy'
    release_close.set()
    first, second = await asyncio.gather(first_get, second_get)
    assert (await close_task)['success'], 'Original close must finish normally'
    assert (await repeated_close)['success'], 'Repeated close must share the original cleanup result'
    assert first is replacement, 'The first waiter must publish the replacement'
    assert second is replacement, 'All waiters must observe one replacement'
    assert starts == 1, 'Waiters must never split across detached creation locks'
    assert session_runtime._SESSION_LOCKS['s1'] is creation_lock, 'Creation lock identity must survive close'


async def test_turn_waiter_rechecks_closing_after_acquiring_lock() -> None:
    session = _make_session()
    await session.lock.acquire()
    waiter = asyncio.create_task(session.send_user_message('stale lookup', timeout=1))
    await asyncio.sleep(0)
    await session.close()
    session.lock.release()
    with pytest.raises(RuntimeError, match='closing|closed'):
        await waiter
    assert cast(FakeConnection, session.connection).prompts == [], 'A waiting turn must not enter a closed connection'


async def test_startup_failure_after_soft_interrupt_returns_cancelled(monkeypatch: pytest.MonkeyPatch) -> None:
    entered, release = asyncio.Event(), asyncio.Event()

    async def failing_start(**_kwargs: Any) -> _CodeBuddyAcpSession:
        entered.set()
        await release.wait()
        raise RuntimeError('startup failed')

    monkeypatch.setattr(session_runtime, '_start_session', failing_start)
    task = asyncio.create_task(run_codebuddy_session_turn(session_id='s1', prompt='one'))
    await entered.wait()
    interrupt = await interrupt_codebuddy_session(session_id='s1')
    release.set()
    assert interrupt['pending'], 'Startup interrupt must record its intent'
    assert (await task)['status'] == 'cancelled', 'Controlled startup interruption must retain its status'
    assert session_runtime._SESSION_STARTS == {}, 'Failed interrupted startup cannot leave a pending record'


async def test_startup_has_an_independent_bounded_deadline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    resources = _patch_startup_dependencies(monkeypatch, tmp_path)
    monkeypatch.setattr(session_runtime, '_STARTUP_TIMEOUT', 0.01)

    async def blocking_initialize(_self: Any, **_kwargs: Any) -> dict[str, Any]:
        await asyncio.Event().wait()
        raise AssertionError('unreachable')

    monkeypatch.setattr(FakeConnection, 'initialize', blocking_initialize)
    with pytest.raises(asyncio.TimeoutError):
        await run_codebuddy_session_turn(session_id='s1', prompt='one', working_dir=str(tmp_path), timeout=3600)
    assert resources['processes'][0].terminated, 'Startup timeout must clean up the process'
    assert resources['proxies'][0].stopped == 1, 'Startup timeout must clean up the proxy'
    assert session_runtime._SESSION_STARTS == {}, 'Startup timeout must release the pending record'


async def test_pending_acp_close_retains_extension_and_failed_session_until_joined(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(session_runtime, '_RESOURCE_CLOSE_TIMEOUT', 0.01)
    session = _make_session()
    connection = cast(FakeConnection, session.connection)
    entered, release = asyncio.Event(), asyncio.Event()
    events: list[str] = []
    await session.extension_stack.enter_async_context(
        extension_factory(session_id='s1', sio=object(), connection=connection, config={'events': events})
    )

    async def blocking_close() -> None:
        entered.set()
        await release.wait()
        connection.closed += 1

    monkeypatch.setattr(connection, 'close', blocking_close)
    first = await close_codebuddy_session(session_id='s1')
    assert entered.is_set(), 'Close must start ACP shutdown'
    assert not first['success'], 'An unfinished callback join cannot be reported as closed'
    assert 'pending' in first['stderr'], 'Pending resource cleanup must remain observable'
    assert session_runtime._SESSIONS['s1'] is session, 'Pending close must retain its session tombstone'
    assert events == ['enter'], 'Extension transport must remain alive until ACP callbacks stop'
    second = await close_codebuddy_session(session_id='s1')
    assert not second['success'], 'A second close must not wash an unfinished cleanup green'
    with pytest.raises(RuntimeError, match='closing'):
        await run_codebuddy_session_turn(session_id='s1', prompt='must not replace')
    release.set()
    cleanup_task = session._cleanup_task
    assert cleanup_task is not None, 'Pending close must retain its original cleanup task'
    await cleanup_task
    final = await close_codebuddy_session(session_id='s1')
    assert final['success'], 'Close can succeed once the original cleanup finishes'
    assert events == ['enter', 'exit'], 'The original cleanup must exit the extension exactly once'
    assert connection.closed == 1, 'Pending close must not create a second ACP shutdown'
    assert session.proxy.stopped == 1, 'Proxy shutdown follows extension teardown'
    assert session_runtime._SESSION_CLEANUPS == {}, 'Finished cleanup must release retained ownership'


async def test_cleanup_failure_remains_visible_on_repeated_public_close(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _make_session()

    async def fail_proxy_stop() -> None:
        session.proxy.stopped += 1
        raise RuntimeError('proxy shutdown failed')

    monkeypatch.setattr(session.proxy, 'stop', fail_proxy_stop)
    first = await close_codebuddy_session(session_id='s1')
    second = await close_codebuddy_session(session_id='s1')
    assert not first['success'], 'Cleanup failure must reach the first close caller'
    assert second == first, 'Repeated close cannot hide a previous resource failure'
    assert session_runtime._SESSIONS['s1'] is session, 'Failed resource ownership must stay discoverable'
    assert session.proxy.stopped == 1, 'Repeated close must not reenter a failed cleanup'


async def test_failed_startup_retains_pending_cleanup_until_explicit_close(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    resources = _patch_startup_dependencies(monkeypatch, tmp_path)
    monkeypatch.setattr(session_runtime, '_RESOURCE_CLOSE_TIMEOUT', 0.01)
    release = asyncio.Event()

    async def failing_initialize(_self: Any, **_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError('initialize failed')

    async def blocking_close(connection: FakeConnection) -> None:
        await release.wait()
        connection.closed += 1

    monkeypatch.setattr(FakeConnection, 'initialize', failing_initialize)
    monkeypatch.setattr(FakeConnection, 'close', blocking_close)
    with pytest.raises(RuntimeError, match='initialize failed'):
        await run_codebuddy_session_turn(session_id='s1', prompt='one', working_dir=str(tmp_path))
    assert 's1' in session_runtime._SESSION_CLEANUPS, 'Failed startup must retain unfinished cleanup resources'
    with pytest.raises(RuntimeError, match='cleanup'):
        await run_codebuddy_session_turn(session_id='s1', prompt='cannot overwrite', working_dir=str(tmp_path))
    assert not (await close_codebuddy_session(session_id='s1'))['success'], (
        'Unknown-close must not hide pending startup'
    )
    release.set()
    await session_runtime._SESSION_CLEANUPS['s1']
    assert (await close_codebuddy_session(session_id='s1'))['success'], (
        'Explicit close releases completed startup cleanup'
    )
    assert resources['proxies'][0].stopped == 1, 'The retained startup task must finish proxy cleanup'


@pytest.mark.parametrize('extra_env', [None, {'CODEBUDDY_API_KEY': 'explicit-test-key'}])
async def test_startup_sets_proxy_auth_default_and_preserves_explicit_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, extra_env: dict[str, str] | None
) -> None:
    resources = _patch_startup_dependencies(monkeypatch, tmp_path)
    session = await session_runtime._start_session(**_startup_kwargs(tmp_path, extra_env=extra_env))
    try:
        expected_key = extra_env['CODEBUDDY_API_KEY'] if extra_env else session_runtime.DEFAULT_PROXY_API_KEY
        assert resources['processes'][0].launch_env['CODEBUDDY_API_KEY'] == expected_key, (
            'The isolated CLI must start with local proxy auth unless explicitly overridden'
        )
    finally:
        await session.close()


async def test_unknown_session_interrupt_reports_failure() -> None:
    result = await interrupt_codebuddy_session(session_id='missing')
    assert not result['success'], 'Unknown session interruption must fail'
    assert 'Unknown' in result['stderr'], 'Unknown session interruption must be observable'


_STEER_CAPABILITIES = {
    'protocolVersion': 2,
    'capabilities': {'session': {'prompt': {'_meta': {'codebuddy.ai/prompt': {'modes': ['steer']}}}, 'fork': {}}},
}
_PERMISSION_MODES = ['default', 'acceptEdits', 'bypassPermissions', 'fullAccess']


class FakeV2Connection(FakeConnection):
    """A WorkBuddy-shaped peer: a prompt is accepted at once and its turn ends with an ``idle`` update."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.permission_policy = kwargs.get('permission_policy', 'allow')
        self.protocol_version = 2
        self.capabilities = deepcopy(_STEER_CAPABILITIES)
        self.initialize_versions: list[int] = []
        self.listeners: list[Any] = []
        self.config_options: dict[str, list[dict[str, Any]]] = {}
        self.permission_modes: list[str] | None = list(_PERMISSION_MODES)
        self.config_calls: list[tuple[str, str, Any]] = []
        self.forked: list[tuple[str, str]] = []
        self.closed_sessions: list[str] = []
        self.bound: dict[str, tuple[Any, Any]] = {}
        self.finished: list[str] = []
        self.metas: list[dict[str, Any] | None] = []
        self.running: dict[str, str] = {}
        self.requests = 0
        self.steer_error: dict[str, Any] | None = None
        self.idle_before_result = False
        self.idle_stop_reason = 'end_turn'
        self.idle_error: dict[str, Any] | None = None

    async def initialize(
        self, *, timeout: float, client_capabilities: dict[str, Any] | None = None, protocol_version: int = 1
    ) -> dict[str, Any]:
        self.initialize_versions.append(protocol_version)
        self.client_capabilities = client_capabilities
        return self.capabilities

    def _options(self) -> list[dict[str, Any]]:
        options: list[dict[str, Any]] = [
            {'configId': 'mode', 'currentValue': 'craft', 'options': [{'value': 'craft'}, {'value': 'ask'}]}
        ]
        if self.permission_modes is not None:
            options.append({
                'configId': 'permission_mode',
                'currentValue': 'bypassPermissions',
                'options': [{'value': value, 'name': value} for value in self.permission_modes],
            })
        return options

    async def new_session(self, *, cwd: str, timeout: float, mcp_servers: list[dict[str, Any]]) -> str:
        self.mcp_servers = mcp_servers
        self.config_options['remote-1'] = self._options()
        return 'remote-1'

    async def fork_session(self, *, session_id: str, cwd: str, timeout: float = 60.0) -> str:
        self.forked.append((session_id, cwd))
        remote = f'remote-fork-{len(self.forked)}'
        self.config_options[remote] = self._options()
        return remote

    async def close_session(self, *, session_id: str, timeout: float = 60.0) -> None:
        self.closed_sessions.append(session_id)

    async def set_config_option(
        self, *, session_id: str, config_id: str, value: Any, value_type: str = 'id', timeout: float = 60.0
    ) -> list[dict[str, Any]]:
        self.config_calls.append((session_id, config_id, value))
        return self.config_options.get(session_id, [])

    def bind_session_handlers(
        self, remote_session_id: str, *, permission_handler: Any = None, elicitation_handler: Any = None
    ) -> Any:
        self.bound[remote_session_id] = (permission_handler, elicitation_handler)
        return lambda: self.bound.pop(remote_session_id, None)

    def add_update_listener(self, listener: Any) -> Any:
        self.listeners.append(listener)

        def remove() -> None:
            if listener in self.listeners:
                self.listeners.remove(listener)

        return remove

    def finish_prompt(self, session_id: str) -> None:
        self.finished.append(session_id)

    def publish(self, session_id: str, update: dict[str, Any]) -> None:
        for listener in list(self.listeners):
            listener({'sessionId': session_id, 'update': update})

    def idle(self, session_id: str, request_id: str | None, *, stop_reason: str | None = None) -> None:
        meta: dict[str, Any] = {} if request_id is None else {'codebuddy.ai/requestId': request_id}
        if self.idle_error is not None:
            meta['codebuddy.ai/error'] = self.idle_error
        self.publish(
            session_id,
            {
                'sessionUpdate': 'state_update',
                'state': 'idle',
                'stopReason': stop_reason or self.idle_stop_reason,
                '_meta': meta,
            },
        )

    def finish(self, request_id: str, *, stop_reason: str | None = None) -> None:
        self.idle(self.running.pop(request_id), request_id, stop_reason=stop_reason)

    async def prompt(
        self, *, session_id: str, text: str, timeout: float, meta: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        self.metas.append(meta)
        if meta is not None:
            if self.steer_error is not None:
                raise AcpRemoteError(self.steer_error)
            expected = meta['codebuddy.ai/prompt']['expectedRequestId']
            return {'messageId': 'steer-message', '_meta': {'codebuddy.ai/requestId': expected}}
        self.prompts.append(text)
        self.requests += 1
        request_id = f'req-{self.requests}'
        self.updates.append(_message_chunk(f'reply:{text}'))
        self.prompt_entered.set()
        if self.idle_before_result:
            self.idle(session_id, request_id)
        elif self.block_prompt:
            self.running[request_id] = session_id
        else:
            asyncio.get_running_loop().call_soon(self.idle, session_id, request_id)
        return {'messageId': request_id, '_meta': {'codebuddy.ai/requestId': request_id}}

    async def cancel(self, *, session_id: str) -> None:
        self.cancelled.append(session_id)
        self.cancel_entered.set()
        await self.cancel_release.wait()
        if self.ack_cancel:
            for request_id, running_session in list(self.running.items()):
                if running_session == session_id:
                    self.finish(request_id, stop_reason='cancelled')


def _v2_session(**overrides: Any) -> _CodeBuddyAcpSession:
    connection = overrides.pop('connection', FakeV2Connection())
    overrides.setdefault('initialize_result', connection.capabilities)
    return _make_session(connection=connection, acp_cli='workbuddy', protocol_version=2, **overrides)


async def _wait_until(predicate: Any) -> None:
    for _ in range(200):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError('condition never held')


async def test_v2_turn_waits_for_the_idle_of_its_own_request() -> None:
    session = _v2_session()
    connection = cast(FakeV2Connection, session.connection)
    connection.block_prompt = True
    turn = asyncio.create_task(session.send_user_message('build tetris', timeout=1))
    await _wait_until(lambda: session.active_request_id == 'req-1')
    connection.idle('remote-other', 'req-1')
    connection.idle('remote-1', 'req-old')
    await asyncio.sleep(0)
    assert not turn.done(), 'Neither another session nor another request may end this turn'
    assert connection.finished == [], 'Nothing is withdrawn while the turn still runs'
    connection.finish('req-1')
    result = await turn
    assert result['status'] == 'ok'
    assert result['acp_stop_reason'] == 'end_turn'
    assert result['output'] == 'reply:build tetris'
    assert connection.finished == ['remote-1'], 'The idle turn withdraws what it left open'
    assert session.active_request_id is None
    assert connection.listeners == [], 'The idle listener lives only as long as its turn'


async def test_v2_idle_that_arrives_before_the_prompt_result_ends_the_turn() -> None:
    session = _v2_session()
    connection = cast(FakeV2Connection, session.connection)
    connection.idle_before_result = True
    result = await asyncio.wait_for(session.send_user_message('one', timeout=1), timeout=1)
    assert result['status'] == 'ok'
    assert connection.finished == ['remote-1']


async def test_v2_error_stop_reason_fails_the_turn_and_keeps_the_session() -> None:
    session = _v2_session()
    connection = cast(FakeV2Connection, session.connection)
    connection.idle_stop_reason = '_codebuddy.ai/error'
    connection.idle_error = {'code': 'auth', 'message': 'model key rejected'}
    failed = await session.send_user_message('one', timeout=1)
    assert failed['status'] == 'failed'
    assert not failed['success'], 'A failed turn is not a successful result'
    assert 'model key rejected' in failed['stderr']
    assert 'auth' in failed['stderr']
    assert session.reusable, 'A turn that failed inside the agent leaves the session usable'
    connection.idle_stop_reason, connection.idle_error = 'end_turn', None
    assert (await session.send_user_message('two', timeout=1))['status'] == 'ok'


async def test_v2_interrupt_cancels_the_turn_and_waits_for_its_idle() -> None:
    session = _v2_session()
    connection = cast(FakeV2Connection, session.connection)
    connection.block_prompt = True
    turn = asyncio.create_task(session.send_user_message('one', timeout=1))
    await _wait_until(lambda: session.active_request_id is not None)
    interrupted = await session.interrupt(grace_timeout=0.5)
    result = await turn
    assert interrupted['stopped'], 'The cancelled idle confirms the interrupt'
    assert result['status'] == 'cancelled'
    assert session.reusable


async def test_v2_relay_turn_names_its_acp_session() -> None:
    relay = FakeTurnRelay()
    session = _v2_session(update_relay=relay)
    await session.send_user_message('one', timeout=1, relay_turn_id='turn-1')
    assert relay.session_ids == ['remote-1'], 'A relay on a shared process must keep only this session'
    plain = FakeTurnRelay()
    await _make_session(register=False, update_relay=plain).send_user_message('one', timeout=1, relay_turn_id='t')
    assert plain.session_ids == [None], 'A CodeBuddy relay is told the turn id only, as before'


async def test_steer_names_the_running_request() -> None:
    session = _v2_session()
    connection = cast(FakeV2Connection, session.connection)
    connection.block_prompt = True
    turn = asyncio.create_task(session.send_user_message('build tetris', timeout=1))
    await _wait_until(lambda: session.active_request_id == 'req-1')
    steered = await steer_codebuddy_session(session_id='s1', prompt='make it smaller')
    assert steered == {'success': True, 'steered': True, 'session_id': 's1', 'stderr': '', 'error_code': None}
    assert connection.metas[-1] == {'codebuddy.ai/prompt': {'mode': 'steer', 'expectedRequestId': 'req-1'}}
    assert connection.prompts == ['build tetris'], 'A steer joins the running turn instead of starting one'
    connection.finish('req-1')
    assert (await turn)['status'] == 'ok'


async def test_steer_without_a_running_turn_inserts_nothing() -> None:
    session = _v2_session()
    result = await steer_codebuddy_session(session_id='s1', prompt='late')
    assert not result['steered']
    assert result['error_code'] is None
    assert cast(FakeV2Connection, session.connection).metas == [], 'Nothing is sent without a running turn'
    unknown = await steer_codebuddy_session(session_id='missing', prompt='x')
    assert not unknown['steered']
    assert 'Unknown' in unknown['stderr']


async def test_steer_reports_the_peer_refusal_code() -> None:
    session = _v2_session()
    connection = cast(FakeV2Connection, session.connection)
    connection.block_prompt = True
    connection.steer_error = {'code': -32600, 'message': 'prompt cannot take steer input right now'}
    turn = asyncio.create_task(session.send_user_message('one', timeout=1))
    await _wait_until(lambda: session.active_request_id is not None)
    refused = await session.steer('too early', timeout=1)
    assert not refused['steered']
    assert refused['error_code'] == -32600, 'The caller retries or sends a new turn by this code'
    connection.finish('req-1')
    await turn


@pytest.mark.parametrize('kind', ['codebuddy', 'unadvertised'])
async def test_steer_needs_a_peer_that_advertises_it(kind: str) -> None:
    if kind == 'codebuddy':
        session = _make_session()
    else:
        session = _v2_session(initialize_result={'protocolVersion': 2, 'capabilities': {'session': {}}})
    session.active_request_id = 'req-1'
    result = await session.steer('x', timeout=1)
    assert not result['steered']
    assert 'steering' in result['stderr']


_CHANNEL_LOG: dict[str, Any] = {}
"""What channel_extension_factory saw; the session runtime deep-copies extension_config, so it lives here."""


@asynccontextmanager
async def channel_extension_factory(
    *, session_id: str, sio: Any, connection: Any, config: dict[str, Any]
) -> AsyncIterator[CodeBuddySessionExtension]:
    """Yield handlers and a relay tagged with the session id the extension was entered for."""
    _CHANNEL_LOG['entered'].append(session_id)

    async def permission_handler(*_args: Any) -> None:
        return None

    async def elicitation_handler(_params: dict[str, Any]) -> dict[str, Any]:
        return {'action': 'cancel'}

    permission_handler.channel = session_id  # type: ignore[attr-defined]
    relay = FakeTurnRelay()
    _CHANNEL_LOG['relays'][session_id] = relay
    try:
        yield CodeBuddySessionExtension(
            permission_handler=permission_handler,
            elicitation_handler=elicitation_handler,
            update_relay=relay,
            mcp_servers=tuple(config.get('mcp_servers', ())),
        )
    finally:
        _CHANNEL_LOG['exited'].append(session_id)


def _patch_workbuddy(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    from nexus.extensions.agents.codebuddy import workbuddy_acp

    record: dict[str, Any] = {'ensured': [], 'stopped': []}
    _CHANNEL_LOG.clear()
    _CHANNEL_LOG.update(entered=[], exited=[], relays={})

    async def ensure_workbuddy_cli(options: dict[str, Any], *, node_path: str) -> Path:
        record['ensured'].append((options, node_path))
        return tmp_path / 'wb' / 'package' / 'bin' / 'wb.js'

    async def stop_workbuddy_process(proc: FakeProc) -> None:
        record['stopped'].append(proc)
        proc.returncode = 0

    monkeypatch.setattr(workbuddy_acp, 'ensure_workbuddy_cli', ensure_workbuddy_cli)
    monkeypatch.setattr(session_runtime, '_stop_workbuddy_process', stop_workbuddy_process)
    return record


def _workbuddy_kwargs(tmp_path: Path, **overrides: Any) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        'acp_cli': 'workbuddy',
        'workbuddy_options': {'version': '5.11.0', 'fork_mode': 'ask'},
        'extension_factory': f'{__name__}::channel_extension_factory',
        'extension_config': {},
        'permission_policy': 'defer',
        'skip_permissions': False,
    }
    return _startup_kwargs(tmp_path, **{**defaults, **overrides})


async def test_workbuddy_start_launches_wb_acp_on_protocol_version_2(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    resources = _patch_startup_dependencies(monkeypatch, tmp_path, connection_type=FakeV2Connection)
    record = _patch_workbuddy(monkeypatch, tmp_path)
    session = await session_runtime._start_session(
        **_workbuddy_kwargs(
            tmp_path, extra_cli_args=['--verbose'], extra_env={'CODEBUDDY_API_KEY': 'explicit-search-key'}
        )
    )
    connection = cast(FakeV2Connection, session.connection)
    cmd = resources['cmd']
    assert resources['installs'] == 0, 'A WorkBuddy session never installs the CodeBuddy CLI'
    assert record['ensured'][0][0]['version'] == '5.11.0'
    assert cmd[:3] == ('/usr/bin/node', str(tmp_path / 'wb/package/bin/wb.js'), '--acp')
    assert '--acp-transport' not in cmd
    assert '--dangerously-skip-permissions' not in cmd
    assert cmd[-1] == '--verbose', 'Caller CLI arguments still reach wb'
    assert resources['spawn_kwargs']['start_new_session'] is True, 'wb and its agents share a process group'
    env = resources['processes'][0].launch_env
    config_dir = Path(session.home) / 'workbuddy'
    assert env['WORKBUDDY_CONFIG_DIR'] == str(config_dir)
    assert env['WORKBUDDY_USER_DATA_DIR'] == str(config_dir / 'app')
    assert env['CODEBUDDY_BASE_URL'] == 'http://127.0.0.1:18001/v1', 'wb model traffic stays on the session proxy'
    assert env['CODEBUDDY_API_KEY'] == 'explicit-search-key', 'Web search must receive the configured credential'
    models = json.loads((config_dir / 'models.json').read_text())
    assert models['models'][0]['url'] == 'http://127.0.0.1:18001/v1/chat/completions'
    assert models['models'][0]['apiKey'] == session_runtime.DEFAULT_PROXY_API_KEY
    assert connection.initialize_versions == [2]
    assert session.protocol_version == 2
    assert session.acp_cli == 'workbuddy'
    assert connection.config_calls == [('remote-1', 'permission_mode', 'default')], (
        'A deferred session must ask for each permission; craft mode would bypass them'
    )
    await session.close()
    assert record['stopped'] == [resources['processes'][0]], 'wb is stopped by closing stdin, not terminate'
    assert not resources['processes'][0].terminated


@pytest.mark.parametrize(
    ('skip_permissions', 'options', 'modes', 'expected'),
    [
        (True, {}, _PERMISSION_MODES, [('remote-1', 'permission_mode', 'fullAccess')]),
        (
            False,
            {'permission_mode': 'acceptEdits'},
            _PERMISSION_MODES,
            [('remote-1', 'permission_mode', 'acceptEdits')],
        ),
        (False, {}, None, []),
    ],
    ids=['skip-is-full-access', 'explicit-override', 'plan-mode-offers-none'],
)
async def test_workbuddy_permission_mode_follows_skip_permissions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    skip_permissions: bool,
    options: dict[str, Any],
    modes: list[str] | None,
    expected: list[tuple[str, str, str]],
) -> None:
    class Peer(FakeV2Connection):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.permission_modes = modes

    _patch_startup_dependencies(monkeypatch, tmp_path, connection_type=Peer)
    _patch_workbuddy(monkeypatch, tmp_path)
    kwargs = _workbuddy_kwargs(tmp_path, skip_permissions=skip_permissions, permission_policy='allow')
    kwargs['workbuddy_options'] = {'version': '5.11.0', **options}
    session = await session_runtime._start_session(**kwargs)
    assert cast(FakeV2Connection, session.connection).config_calls == expected
    await session.close()


async def test_workbuddy_rejects_a_permission_mode_the_session_does_not_offer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    resources = _patch_startup_dependencies(monkeypatch, tmp_path, connection_type=FakeV2Connection)
    record = _patch_workbuddy(monkeypatch, tmp_path)
    kwargs = _workbuddy_kwargs(tmp_path)
    kwargs['workbuddy_options'] = {'version': '5.11.0', 'permission_mode': 'dontAsk'}
    with pytest.raises(ValueError, match='dontAsk.*not offered'):
        await session_runtime._start_session(**kwargs)
    assert record['stopped'] == resources['processes'], 'Failed startup stops wb the WorkBuddy way'
    assert resources['proxies'][0].stopped == 1


async def test_workbuddy_refuses_mcp_servers_before_initialize(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    resources = _patch_startup_dependencies(monkeypatch, tmp_path, connection_type=FakeV2Connection)
    _patch_workbuddy(monkeypatch, tmp_path)
    server = {'type': 'http', 'name': 'tools', 'url': 'http://127.0.0.1:1/mcp', 'headers': []}
    with pytest.raises(ValueError, match='mcpServers'):
        await session_runtime._start_session(**_workbuddy_kwargs(tmp_path, extension_config={'mcp_servers': [server]}))
    assert resources['connections'][0].initialize_versions == [], 'host_tools must fail before the handshake'
    assert _CHANNEL_LOG['exited'] == ['s1'], 'The extension is rolled back'


@pytest.mark.parametrize(
    'overrides',
    [{'workbuddy_options': {'version': '5.11.0'}}, {'fork_source_session_id': 's1'}, {'acp_cli': 'gemini'}],
    ids=['options', 'fork', 'unknown-cli'],
)
async def test_codebuddy_rejects_workbuddy_only_options_before_spawning(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, overrides: dict[str, Any]
) -> None:
    resources = _patch_startup_dependencies(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match='acp_cli'):
        await session_runtime._start_session(**_startup_kwargs(tmp_path, **overrides))
    assert resources['processes'] == []
    assert resources['installs'] == 0


async def _workbuddy_source(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[_CodeBuddyAcpSession, dict]:
    resources = _patch_startup_dependencies(monkeypatch, tmp_path, connection_type=FakeV2Connection)
    resources['wb'] = _patch_workbuddy(monkeypatch, tmp_path)
    resources['config'] = _CHANNEL_LOG
    source = await session_runtime._get_or_create_session(**_workbuddy_kwargs(tmp_path))
    return source, resources


def _fork_kwargs(tmp_path: Path, resources: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    kwargs = _workbuddy_kwargs(tmp_path, fork_source_session_id='s1')
    kwargs.update(session_id='fork-1', **overrides)
    return kwargs


async def test_fork_shares_the_process_and_owns_its_session_and_channels(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source, resources = await _workbuddy_source(monkeypatch, tmp_path)
    connection = cast(FakeV2Connection, source.connection)
    fork = await session_runtime._get_or_create_session(**_fork_kwargs(tmp_path, resources))
    assert connection.forked == [('remote-1', source.cwd)]
    assert fork.proc is source.proc
    assert fork.connection is connection
    assert fork.proxy is source.proxy
    assert not fork.owns_process
    assert fork.fork_source == 's1'
    assert source.forks == {'fork-1'}
    assert len(resources['processes']) == 1, 'A fork starts no process'
    assert resources['config']['entered'] == ['s1', 'fork-1'], 'The fork gets its own Host channels'
    permission_handler, _ = connection.bound['remote-fork-1']
    assert permission_handler.channel == 'fork-1', "The fork's requests reach the fork's channel"
    assert connection.config_calls[1:] == [
        ('remote-fork-1', 'mode', 'ask'),
        ('remote-fork-1', 'permission_mode', 'default'),
    ], 'The idle fork switches mode first, then the mode change is undone for permissions'
    relay = resources['config']['relays']['fork-1']
    await run_codebuddy_session_turn(**_fork_kwargs(tmp_path, resources, prompt='how far?', relay_turn_id='t-q'))
    assert relay.session_ids == ['remote-fork-1']
    assert connection.prompts == ['how far?']
    assert (await close_codebuddy_session(session_id='fork-1'))['success']
    assert connection.closed_sessions == ['remote-fork-1'], 'Closing a fork closes only its remote session'
    assert 'remote-fork-1' not in connection.bound
    assert resources['config']['exited'] == ['fork-1'], "Only the fork's extension exits"
    assert resources['wb']['stopped'] == [], 'The source process is not stopped'
    assert source.is_running, 'The source process keeps running'
    assert source.forks == set()
    assert 'fork-1' not in session_runtime._SESSIONS
    await source.close()


async def test_closing_the_source_closes_its_forks_first(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source, resources = await _workbuddy_source(monkeypatch, tmp_path)
    session_runtime._SESSIONS['s1'] = source
    fork = await session_runtime._get_or_create_session(**_fork_kwargs(tmp_path, resources))
    assert session_runtime._SESSIONS['fork-1'] is fork
    assert (await close_codebuddy_session(session_id='s1'))['success']
    assert fork.closing
    assert 'fork-1' not in session_runtime._SESSIONS
    assert cast(FakeV2Connection, source.connection).closed_sessions == ['remote-fork-1']
    assert resources['config']['exited'] == ['fork-1', 's1']
    assert resources['wb']['stopped'] == [source.proc]


async def test_closing_a_fork_after_its_source_died_succeeds(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source, resources = await _workbuddy_source(monkeypatch, tmp_path)
    fork = await session_runtime._get_or_create_session(**_fork_kwargs(tmp_path, resources))
    cast(FakeProc, source.proc).returncode = 1
    assert (await fork.close())['success'], 'A fork that died with its source has nothing left to close'
    assert cast(FakeV2Connection, source.connection).closed_sessions == []
    await source.close()


@pytest.mark.parametrize('problem', ['codebuddy-source', 'other-cwd', 'missing-source', 'policy'])
async def test_fork_refuses_a_source_it_cannot_branch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, problem: str
) -> None:
    source, resources = await _workbuddy_source(monkeypatch, tmp_path)
    kwargs = _fork_kwargs(tmp_path, resources)
    if problem == 'codebuddy-source':
        source.protocol_version = 1
    elif problem == 'other-cwd':
        kwargs['working_dir'] = str(tmp_path / 'elsewhere')
    elif problem == 'missing-source':
        kwargs['fork_source_session_id'] = 'nobody'
    else:
        kwargs['permission_policy'] = 'allow'
    with pytest.raises((RuntimeError, ValueError)):
        await session_runtime._start_session(**kwargs)
    connection = cast(FakeV2Connection, source.connection)
    assert connection.forked == []
    assert connection.bound == {}
    assert resources['config']['entered'] == ['s1'], 'A refused fork never enters an extension'
    await source.close()


async def test_fork_waits_out_a_source_start_that_is_still_in_flight(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(session_runtime, '_FORK_SOURCE_POLL_S', 0.01)
    session_runtime._SESSION_STARTS['s1'] = session_runtime._CodeBuddySessionStart()

    async def finish_the_start() -> None:
        await asyncio.sleep(0.05)
        _make_session()
        session_runtime._SESSION_STARTS.pop('s1', None)

    finishing = asyncio.create_task(finish_the_start())
    source = await session_runtime._await_fork_source('s1')
    assert source is session_runtime._SESSIONS['s1'], 'A question dispatched during startup waits for it'
    await finishing


async def test_fork_source_start_that_fails_without_a_session_reports_not_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(session_runtime, '_FORK_SOURCE_POLL_S', 0.01)
    session_runtime._SESSION_STARTS['s1'] = session_runtime._CodeBuddySessionStart()

    async def fail_the_start() -> None:
        await asyncio.sleep(0.05)
        session_runtime._SESSION_STARTS.pop('s1', None)

    failing = asyncio.create_task(fail_the_start())
    assert await session_runtime._await_fork_source('s1') is None
    await failing


async def test_fork_setup_failure_closes_the_remote_fork(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source, resources = await _workbuddy_source(monkeypatch, tmp_path)
    connection = cast(FakeV2Connection, source.connection)

    async def refuse_mode(**kwargs: Any) -> list[dict[str, Any]]:
        raise AcpRemoteError({'code': -32602, 'message': 'unknown mode'})

    monkeypatch.setattr(connection, 'set_config_option', refuse_mode)
    with pytest.raises(AcpRemoteError):
        await session_runtime._start_session(**_fork_kwargs(tmp_path, resources))
    assert connection.closed_sessions == ['remote-fork-1']
    assert connection.bound == {}
    assert resources['config']['exited'] == ['fork-1']
    assert source.forks == set()
    await source.close()


async def test_stop_workbuddy_process_closes_stdin_then_stops_the_group(monkeypatch: pytest.MonkeyPatch) -> None:
    stopped: list[int] = []

    class Stdin:
        def __init__(self, proc: FakeProc) -> None:
            self.proc = proc

        def close(self) -> None:
            self.proc.returncode = 0

    async def stop_process_group(pgid: int) -> None:
        stopped.append(pgid)

    monkeypatch.setattr('nexus.extensions.agents.workbuddy.runtime_setup._stop_process_group', stop_process_group)
    proc = FakeProc()
    proc.pid = 4321  # type: ignore[attr-defined]
    proc.stdin = Stdin(proc)  # type: ignore[assignment]
    await session_runtime._stop_workbuddy_process(cast(Any, proc))
    assert proc.returncode == 0, 'wb exits on stdin EOF'
    assert not proc.terminated, 'No signal is sent to the wb process itself'
    assert stopped == [4321], 'Agent processes wb left behind are stopped with its group'


@pytest.mark.asyncio
async def test_side_question_sends_remembered_answers_and_waits_for_the_first_prompt() -> None:
    class Connection:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, list[dict[str, str]]]] = []

        async def side_question(
            self, *, session_id: str, question: str, turns: list[dict[str, str]] | None, timeout: float
        ) -> str:
            del timeout
            self.calls.append((session_id, question, list(turns or [])))
            return f'答:{question}'

        async def close(self) -> None:
            return None

    connection = Connection()
    session = _make_session(
        connection=connection,
        protocol_version=2,
        prompt_landed=False,
        initialize_result={
            'capabilities': {'session': {'_meta': {'codebuddy.ai/sideQuestion': {'version': 1, 'maxTurns': 20}}}},
        },
    )
    pending = asyncio.create_task(session.ask_side_question('为什么？'))
    await asyncio.sleep(0)
    assert connection.calls == [], 'A draft session is not asked until its first prompt is accepted'
    session.prompt_landed = True
    first = await pending
    second = await session.ask_side_question('然后呢？')

    assert first == {
        'success': True,
        'answer': '答:为什么？',
        'session_id': 's1',
        'stderr': '',
        'error_code': None,
    }
    assert connection.calls == [
        ('remote-1', '为什么？', []),
        ('remote-1', '然后呢？', [{'question': '为什么？', 'answer': '答:为什么？'}]),
    ]
    assert second['answer'] == '答:然后呢？'
