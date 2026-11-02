"""Driving several sessions through one turn each, concurrently.

``SessionManager.run_turns`` is what makes the concurrency reachable: sessions could
already run at once -- each carries its own trace label -- but without a driving
primitive every caller assembled the tasks itself, and one that looped sequentially
got none of the benefit while looking correct.

It yields as each turn lands, so a caller can act on the first result -- read its
report, send that session its next message -- while the others are still running.
"""

import asyncio

from typing import Any

import pytest

from nexus.extensions.tasks.hitl.actor.base import (
    BatchTurnResult,
    Session,
    SessionManager,
    SessionMessage,
    TurnResult,
)


class Driver(SessionManager):
    """A manager that only implements ``send``, which is all ``run_turns`` uses."""

    backend = 'driver'
    product = 'Driver'

    def __init__(self, *, delays: dict[str, float] | None = None, failing: set[str] | None = None) -> None:
        self.delays = delays or {}
        self.failing = failing or set()
        self.sent: list[str] = []

    async def setup(self) -> None: ...

    async def start(self, *, name: str, metadata: Any = None, **kwargs: object) -> Session:
        return Session(name=name, session_id=f's-{name}', backend=self.backend)

    async def send(self, session: Session, message: SessionMessage) -> TurnResult:
        await asyncio.sleep(self.delays.get(session.name, 0))
        if session.name in self.failing:
            raise RuntimeError(f'{session.name} is gone')
        self.sent.append(session.name)
        return TurnResult(kind='completed', output=message.content, session_id=session.session_id)

    def session_of(self, session_id: str) -> Session | None:
        return None

    def report(self, result: TurnResult, **_kwargs: object) -> str:
        return result.output

    def traces(self, session: Session) -> dict[str, object]:
        return {}

    async def close(self, session: Session) -> None: ...

    async def shutdown(self) -> None: ...


def turns(*names: str) -> list[tuple[Session, SessionMessage]]:
    return [
        (Session(name=name, session_id=f's-{name}', backend='driver'), SessionMessage(content=f'do {name}'))
        for name in names
    ]


async def collect(manager: SessionManager, batch: list[tuple[Session, SessionMessage]]) -> list[BatchTurnResult]:
    return [turn async for turn in manager.run_turns(batch)]


@pytest.mark.asyncio
async def test_sessions_run_at_once_rather_than_one_after_another() -> None:
    """Each send holds for a tick; sequential execution would order the finishes by
    argument position instead of by duration."""
    delays = {'slow': 0.05, 'quick': 0.01, 'mid': 0.03}

    turned = await collect(Driver(delays=delays), turns('slow', 'quick', 'mid'))

    assert [turn.session.name for turn in turned] == ['quick', 'mid', 'slow']


@pytest.mark.asyncio
async def test_a_turn_arrives_while_the_others_are_still_running() -> None:
    """Yielding rather than returning a batch is what makes acting on the first
    result possible: the slow sessions have not finished when the quick one lands."""
    manager = Driver(delays={'quick': 0.01, 'slow': 0.2})
    seen: list[tuple[str, frozenset[str]]] = []

    async for turn in manager.run_turns(turns('quick', 'slow')):
        seen.append((turn.session.name, turn.inflight))
        if turn.session.name == 'quick':
            assert manager.sent == ['quick'], 'slow is still mid-turn'
            break

    assert seen == [('quick', frozenset({'slow'}))]


@pytest.mark.asyncio
async def test_each_turn_names_who_is_still_working() -> None:
    """``inflight`` is what lets a driver decide between acting now and waiting."""
    turned = await collect(Driver(delays={'a': 0.01, 'b': 0.03, 'c': 0.05}), turns('a', 'b', 'c'))

    assert [sorted(turn.inflight) for turn in turned] == [['b', 'c'], ['c'], []]


@pytest.mark.asyncio
async def test_one_session_failing_does_not_discard_the_others() -> None:
    """A raise would lose the turns that did finish, which cost a model call each."""
    manager = Driver(delays={'a': 0.01, 'broken': 0.02, 'c': 0.03}, failing={'broken'})

    turned = await collect(manager, turns('a', 'broken', 'c'))
    by_name = {turn.session.name: turn for turn in turned}

    assert len(turned) == 3
    assert by_name['broken'].kind == 'terminal'
    assert 'is gone' in str(by_name['broken'].result.metadata['error'])
    assert by_name['a'].kind == 'completed'
    assert by_name['c'].kind == 'completed'


@pytest.mark.asyncio
async def test_a_session_addressed_twice_is_refused_before_anything_is_sent() -> None:
    """Two turns on one conversation race: both resume the same session, and the
    second reaches a worker mid-turn."""
    manager = Driver()
    doubled = turns('a', 'b') + turns('a')

    with pytest.raises(ValueError, match='addressed twice'):
        await collect(manager, doubled)

    assert manager.sent == [], 'a bad batch must not leave half the sessions having taken a turn'


@pytest.mark.asyncio
async def test_addressing_nobody_sends_nothing() -> None:
    manager = Driver()

    assert await collect(manager, []) == []
    assert manager.sent == []


@pytest.mark.asyncio
async def test_one_session_is_a_batch_of_one() -> None:
    """No separate path for the ordinary case, which is what a recipe drives today."""
    turned = await collect(Driver(), turns('worker'))

    assert [turn.session.name for turn in turned] == ['worker']
    assert turned[0].inflight == frozenset()
