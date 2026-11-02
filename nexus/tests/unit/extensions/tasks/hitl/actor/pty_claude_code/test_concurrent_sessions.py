"""Concurrent sessions on the PTY backend.

Several sessions run at once without their traces getting mixed, kept apart by the
label their own sandbox proxy stamps on each LLM request. What is pinned here is that
each session gets a distinct label, that each recovers its own share of the run-level
export, and that no manager state is involved in either.
"""

from types import SimpleNamespace
from typing import Any

import pytest

from nexus.extensions.tasks.hitl.actor.base import BatchTurnResult, SessionMessage, TurnResult
from nexus.extensions.tasks.hitl.actor.labelling.handler import LabelledLLMRequestHandler
from nexus.extensions.tasks.hitl.actor.pty_claude_code import (
    PtyClaudeCodeSession,
    PtyClaudeCodeSessionManager,
)
from tests.unit.extensions.tasks.hitl.actor.invoke_double import read_invoke


class StubClient:
    """The client seat on the handler; nothing here calls through it."""


class FakeRuntime:
    """Records invocations; every turn reports completed."""

    default_venv_id = 'worker-venv'
    """Non-None so a request that drops it is visible. A turn outside the venv finds
    none of the sessions the backend registered."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.listeners: list[str] = []

    async def ws_connect(self) -> None:
        return

    def on(self, event: str, _handler: Any) -> None:
        self.listeners.append(event)

    def off(self, event: str, _handler: Any = None) -> None:
        if event in self.listeners:
            self.listeners.remove(event)

    async def emit(self, event: str, payload: Any = None) -> None:
        return None

    async def invoke(self, target: Any, /, *_args: Any, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(read_invoke(target, kwargs))
        return {
            'success': True,
            'status': 'completed',
            'completed': True,
            'session_id': 'runtime-side',
            'output': 'done',
            'stderr': '',
            'exit_code': 0,
            'events': [],
            'turn_id': 1,
        }

    def turn_calls(self) -> list[dict[str, Any]]:
        return [kw for name, kw in self.calls if name == 'run_claude_code_interactive_session_turn']


@pytest.fixture
def manager(monkeypatch):
    """A set-up manager plus its fake runtime."""

    def build(**kwargs: Any):
        runtime = FakeRuntime()

        async def get_runtime(name='main'):
            assert name == kwargs.get('runtime_name', 'main')
            return runtime

        ctx = SimpleNamespace(get_runtime=get_runtime, get_openai_client=lambda _name='main': StubClient())
        monkeypatch.setattr(
            'nexus.extensions.tasks.hitl.actor.pty_claude_code.manager.require_context',
            lambda: ctx,
        )
        mgr = PtyClaudeCodeSessionManager(default_working_dir='/testbed', **kwargs)
        return mgr, runtime

    return build


# --- labels keep concurrent sessions apart ---------------------------------


@pytest.mark.asyncio
async def test_each_session_gets_its_own_label(manager):
    mgr, _ = manager()
    await mgr.setup()

    opened = [await mgr.start(name=name) for name in ('a', 'b', 'c')]

    assert [session.label for session in opened] == ['worker/a', 'worker/b', 'worker/c']


@pytest.mark.asyncio
async def test_sessions_keep_distinct_ids(manager):
    mgr, _ = manager()
    await mgr.setup()

    first = await mgr.start(name='a')
    second = await mgr.start(name='b')

    assert first.session_id != second.session_id
    assert first.label != second.label


@pytest.mark.asyncio
async def test_a_repeated_session_name_is_refused(manager):
    """Two sessions under one name would share a label, so neither trace is
    recoverable."""
    mgr, _ = manager()
    await mgr.setup()
    await mgr.start(name='a')

    with pytest.raises(ValueError, match='already exists'):
        await mgr.start(name='a')


# --- turns ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sessions_can_take_turns_with_no_shared_state(manager):
    mgr, runtime = manager()
    await mgr.setup()
    first = await mgr.start(name='a')
    second = await mgr.start(name='b')

    one = await mgr.send(first, SessionMessage(content='task a'))
    two = await mgr.send(second, SessionMessage(content='task b'))

    assert (one.kind, two.kind) == ('completed', 'completed')
    assert len(runtime.turn_calls()) == 2
    # The handler holds no per-turn trace state at all.
    assert not hasattr(mgr._handler, 'scope')


@pytest.mark.asyncio
async def test_the_manager_drives_several_sessions_through_one_call(manager):
    mgr, runtime = manager()
    await mgr.setup()
    first = await mgr.start(name='a')
    second = await mgr.start(name='b')

    turned = [
        turn
        async for turn in mgr.run_turns(
            [(first, SessionMessage(content='task a')), (second, SessionMessage(content='task b'))]
        )
    ]

    assert {turn.session.name for turn in turned} == {'a', 'b'}
    assert all(turn.kind == 'completed' for turn in turned)
    assert len(runtime.turn_calls()) == 2
    # The last turn to finish has nobody left running.
    assert turned[-1].inflight == frozenset()


@pytest.mark.asyncio
async def test_a_session_is_reachable_by_its_harness_id(manager):
    """A ``TurnResult`` carries ``session_id`` and no name, so reading one back
    means asking the manager which session it belongs to."""
    mgr, _ = manager()
    await mgr.setup()
    session = await mgr.start(name='a')

    assert mgr.session_of(session.session_id) is session
    assert mgr.session_of('not-a-session') is None


# --- traces -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_each_session_recovers_its_own_share_of_the_export(manager):
    """The assertion the label migration inverted.

    Under a shared tracker every session returned the same run-level tree. Labels
    make each session's share recoverable, so the two must differ.
    """
    mgr, _ = manager()
    await mgr.setup()
    first = await mgr.start(name='a')
    second = await mgr.start(name='b')

    run_level = {
        'trajectories': [
            {'role': 'worker/a', 'messages': [{'role': 'assistant', 'content': 'A'}]},
            {'role': 'worker/b', 'messages': [{'role': 'assistant', 'content': 'B'}]},
            {'role': 'user/orchestrator', 'messages': [{'role': 'assistant', 'content': 'U'}]},
        ]
    }
    from nexus.trace import processor_context
    from nexus.trace.processors.chat_tracker import ChatTrackerProcessor

    tracker = ChatTrackerProcessor(tools_mode='branch_last')
    tracker.snapshot_result = lambda: run_level  # type: ignore[method-assign]

    with processor_context([tracker], finalize_on_exit=False):
        only_a = mgr.traces(first)
        only_b = mgr.traces(second)

    assert [t['messages'][-1]['content'] for t in only_a['trajectories']] == ['A']
    assert [t['messages'][-1]['content'] for t in only_b['trajectories']] == ['B']
    assert only_a != only_b, 'sessions must not share one run-level tree'


@pytest.mark.asyncio
async def test_a_session_whose_name_prefixes_another_is_not_absorbed(manager):
    """``a`` must not pick up ``a2``'s trajectories: the match is on the whole label,
    not a prefix."""
    mgr, _ = manager()
    await mgr.setup()
    short = await mgr.start(name='a')
    long = await mgr.start(name='a2')

    run_level = {
        'trajectories': [
            {'role': 'worker/a', 'messages': [{'role': 'assistant', 'content': 'A'}]},
            {'role': 'worker/a2', 'messages': [{'role': 'assistant', 'content': 'A2'}]},
        ]
    }
    from nexus.trace import processor_context
    from nexus.trace.processors.chat_tracker import ChatTrackerProcessor

    tracker = ChatTrackerProcessor(tools_mode='branch_last')
    tracker.snapshot_result = lambda: run_level  # type: ignore[method-assign]

    with processor_context([tracker], finalize_on_exit=False):
        assert [t['messages'][-1]['content'] for t in mgr.traces(short)['trajectories']] == ['A']
        assert [t['messages'][-1]['content'] for t in mgr.traces(long)['trajectories']] == ['A2']


# --- handler ownership ------------------------------------------------------


@pytest.mark.asyncio
async def test_setup_takes_the_runtime_handler_instead_of_building_one(manager):
    mgr, runtime = manager()
    await mgr.setup()

    assert isinstance(mgr._handler, LabelledLLMRequestHandler)
    # Exactly one listener: a second handler here would double-serve requests.
    assert runtime.listeners == ['llm_request']
    await mgr.shutdown()
    assert runtime.listeners == []


@pytest.mark.asyncio
async def test_an_injected_handler_is_used_and_not_released(manager):
    mgr, runtime = manager()
    injected = LabelledLLMRequestHandler(runtime=runtime, openai_client=StubClient())
    injected.register()

    mgr, runtime2 = manager(llm_handler=injected)
    await mgr.setup()

    assert mgr._handler is injected
    await mgr.shutdown()
    # Still registered: it belongs to whoever injected it, possibly another
    # manager on the same runtime.
    assert runtime.listeners == ['llm_request']


@pytest.mark.asyncio
async def test_shutdown_forgets_every_session(manager):
    mgr, runtime = manager()
    await mgr.setup()
    for name in ('a', 'b', 'c'):
        await mgr.start(name=name)

    await mgr.shutdown()

    assert mgr._sessions_by_name == {}
    assert runtime.listeners == []


# --- batch turn results -----------------------------------------------------


def test_a_batch_turn_reports_who_is_still_running():
    """``inflight`` is the field that makes concurrency worth having.

    Without it a driver reacting to a fast session cannot tell whether to act now
    or wait, so it would have to wait for everyone -- which is what running
    sessions concurrently was meant to avoid.
    """
    session = PtyClaudeCodeSession(name='a', session_id='s-a', backend='pty_claude_code')
    turn = BatchTurnResult(
        session=session,
        result=TurnResult(kind='completed', output='done'),
        inflight=frozenset({'b', 'c'}),
    )

    assert turn.kind == 'completed'
    assert turn.inflight == frozenset({'b', 'c'})
    # Frozen: a record of what happened, not a mutable cursor.
    with pytest.raises(AttributeError):
        turn.inflight = frozenset()  # type: ignore[misc]
