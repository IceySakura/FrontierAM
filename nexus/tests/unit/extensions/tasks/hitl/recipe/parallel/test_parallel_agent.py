"""Opening several sessions and running a script across them.

``run_turns`` itself is covered by ``actor/test_run_turns.py``; what is pinned here is
the agent's use of it -- one manager, a workspace per session, and the record it leaves
of what ran together.
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
from nexus.extensions.tasks.hitl.recipe.parallel.agent import (
    ParallelStep,
    check_script,
    check_sessions,
    workspace_of,
)


def step(*sends: tuple[str, str]) -> ParallelStep:
    return ParallelStep.model_validate({'parallel': [{'to': to, 'message': text} for to, text in sends]})


# -- the session list -------------------------------------------------------


def test_the_names_given_are_the_sessions_opened() -> None:
    assert check_sessions(['alpha', 'beta']) == ['alpha', 'beta']


def test_no_sessions_is_refused() -> None:
    with pytest.raises(ValueError, match='at least one session'):
        check_sessions([])


def test_a_repeated_name_is_refused() -> None:
    """Two sessions of one name would share a label, and nothing downstream could tell
    their trajectories apart."""
    with pytest.raises(ValueError, match='repeated'):
        check_sessions(['alpha', 'alpha'])


def test_a_name_holding_a_separator_is_refused() -> None:
    """It would shift every later label segment and still look well-formed."""
    with pytest.raises(ValueError, match='must not contain'):
        check_sessions(['alpha/beta'])


# -- the script -------------------------------------------------------------


def test_a_script_addressing_the_open_sessions_passes() -> None:
    check_script([step(('alpha', 'a'), ('beta', 'b'))], sessions=['alpha', 'beta'])


def test_an_unknown_recipient_is_refused_and_the_open_ones_named() -> None:
    with pytest.raises(ValueError, match="addresses \\['gamma'\\]"):
        check_script([step(('gamma', 'x'))], sessions=['alpha', 'beta'])


def test_addressing_one_session_twice_in_a_step_is_refused() -> None:
    """``run_turns`` refuses this as well; refusing here names the step."""
    with pytest.raises(ValueError, match='step 1 addresses'):
        check_script([step(('alpha', 'one'), ('alpha', 'two'))], sessions=['alpha'])


def test_a_step_addressing_nobody_is_refused() -> None:
    with pytest.raises(ValueError, match='step 1 addresses nobody'):
        check_script([ParallelStep()], sessions=['alpha'])


def test_an_empty_script_is_refused() -> None:
    with pytest.raises(ValueError, match='at least one step'):
        check_script([], sessions=['alpha'])


def test_the_failing_step_is_named_by_its_position() -> None:
    steps = [step(('alpha', 'fine')), step(('beta', 'also fine')), step(('gamma', 'no'))]

    with pytest.raises(ValueError, match='step 3'):
        check_script(steps, sessions=['alpha', 'beta'])


def test_one_session_addressed_across_steps_is_fine() -> None:
    """The limit is one turn at a time, not one turn per run."""
    check_script([step(('alpha', 'first')), step(('alpha', 'second'))], sessions=['alpha'])


# -- workspaces -------------------------------------------------------------


def test_each_session_works_in_its_own_directory() -> None:
    """Two Claude Code processes in one directory make a crossed file look like a
    crossed conversation."""
    assert workspace_of('alpha', working_dir='/workspace', shared=False) == '/workspace/alpha'


def test_a_trailing_separator_does_not_double() -> None:
    assert workspace_of('alpha', working_dir='/workspace/', shared=False) == '/workspace/alpha'


def test_sharing_is_available_for_a_run_that_wants_it() -> None:
    assert workspace_of('alpha', working_dir='/workspace', shared=True) == '/workspace'


# -- what the agent records, against a manager that does nothing real -------


class Recorder(SessionManager):
    """A manager that answers immediately and counts what it was asked to do."""

    backend = 'recorder'

    def __init__(self) -> None:
        self.setups = 0
        self.started: list[tuple[str, str]] = []
        self.sent: list[tuple[str, str]] = []

    async def setup(self) -> None:
        self.setups += 1

    async def start(self, *, name: str, metadata: Any = None, **kwargs: Any) -> Session:
        workspace = str(kwargs.get('workspace') or '')
        self.started.append((name, workspace))
        return Session(name=name, session_id=f'id-{name}', backend=self.backend, metadata={})

    async def send(self, session: Session, message: SessionMessage) -> TurnResult:
        self.sent.append((session.name, message.content))
        return TurnResult(kind='completed', output=f'{session.name} did it', session_id=session.session_id)

    def session_of(self, session_id: str) -> Session | None:
        return None

    def report(self, result: TurnResult, **kwargs: Any) -> str:
        return result.output

    def traces(self, session: Session) -> dict[str, object]:
        return {}

    async def close(self, session: Session) -> None:
        return None

    async def shutdown(self) -> None:
        return None


@pytest.mark.asyncio
async def test_every_addressed_session_is_sent_to() -> None:
    manager = Recorder()
    sessions = {name: await manager.start(name=name) for name in ('alpha', 'beta')}
    turns = [
        (sessions['alpha'], SessionMessage(content='write ALPHA')),
        (sessions['beta'], SessionMessage(content='write BETA')),
    ]

    events = [event async for event in manager.run_turns(turns)]

    assert len(events) == 2
    assert sorted(manager.sent) == [('alpha', 'write ALPHA'), ('beta', 'write BETA')]


@pytest.mark.asyncio
async def test_the_record_of_a_batch_names_who_was_still_running() -> None:
    """What the overlap check reads. The last to finish has nobody left beside it."""
    manager = Recorder()
    sessions = {name: await manager.start(name=name) for name in ('alpha', 'beta')}
    turns = [(sessions[name], SessionMessage(content='go')) for name in ('alpha', 'beta')]

    events: list[BatchTurnResult] = [event async for event in manager.run_turns(turns)]

    assert [len(event.inflight) for event in events] == [1, 0]


class Paced(Recorder):
    """A manager whose sessions finish in a stated order, slowest last."""

    def __init__(self, delays: dict[str, float]) -> None:
        super().__init__()
        self.delays = delays

    async def send(self, session: Session, message: SessionMessage) -> TurnResult:
        await asyncio.sleep(self.delays.get(session.name, 0.0))
        return await super().send(session, message)


@pytest.mark.asyncio
async def test_a_follow_up_goes_out_while_the_slower_session_is_still_working() -> None:
    """The pattern the run exists to exercise: read the first result, answer it, and
    let the rest go on. A follow-up sent after the loop would find nobody running."""
    manager = Paced({'alpha': 0.0, 'beta': 0.05})
    sessions = {name: await manager.start(name=name) for name in ('alpha', 'beta')}
    turns = [(sessions[name], SessionMessage(content='first')) for name in ('alpha', 'beta')]

    followed: list[tuple[str, list[str]]] = []
    async for event in manager.run_turns(turns):
        if event.kind == 'completed' and event.session.name == 'alpha':
            await manager.send(event.session, SessionMessage(content='second'))
            followed.append((event.session.name, sorted(event.inflight)))

    assert followed == [('alpha', ['beta'])]
    assert ('alpha', 'second') in manager.sent


@pytest.mark.asyncio
async def test_a_session_that_did_not_complete_gets_no_follow_up() -> None:
    """There is nothing to continue, and sending would report a turn that never was."""

    class Refuses(Recorder):
        async def send(self, session: Session, message: SessionMessage) -> TurnResult:
            await super().send(session, message)
            return TurnResult(kind='refused', session_id=session.session_id)

    manager = Refuses()
    session = await manager.start(name='alpha')

    sent_after = []
    async for event in manager.run_turns([(session, SessionMessage(content='first'))]):
        if event.kind == 'completed':
            sent_after.append(event.session.name)

    assert sent_after == []
    assert [name for name, _ in manager.sent] == ['alpha']
