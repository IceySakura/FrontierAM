"""The turn protocol: four kinds, and what a manager must implement.

``kind`` replaces the earlier ``success`` / ``session_lost`` pair, which could not
tell a message that was never delivered from one whose worker died mid-turn -- the
two produced identical results while calling for opposite responses from a driver.
"""

import pytest

from nexus.extensions.tasks.hitl.actor.base import (
    BatchTurnResult,
    Session,
    SessionManager,
    SessionMessage,
    TurnResult,
)
from nexus.extensions.tasks.hitl.trace.labels import worker_label


# --- the four kinds are distinguishable -------------------------------------


def test_each_kind_answers_the_questions_a_driver_asks() -> None:
    """The properties a driver branches on, per kind.

    ``refused`` and ``terminal`` differing on ``delivered`` is the point: retrying
    is right for one and pointless for the other.
    """
    reads = {
        kind: (result.awaiting_answer, result.session_lost, result.delivered)
        for kind in ('completed', 'needs_input', 'terminal', 'refused')
        if (result := TurnResult(kind=kind))
    }

    assert reads == {
        'completed': (False, False, True),
        'needs_input': (True, False, True),
        'terminal': (False, True, True),
        'refused': (False, False, False),
    }
    assert len(set(reads.values())) == 4, 'two kinds a driver cannot tell apart'


def test_a_refused_send_is_not_reported_as_a_lost_session() -> None:
    """A guard-blocked message leaves the session usable."""
    refused = TurnResult(kind='refused', raw={'status': 'blocked_by_guard'})

    assert not refused.session_lost
    assert not refused.delivered
    assert not refused.awaiting_answer


def test_kind_is_required() -> None:
    """No default: a backend that omits the outcome fails loudly."""
    with pytest.raises(TypeError):
        TurnResult()  # type: ignore[call-arg]


def test_a_result_is_frozen() -> None:
    result = TurnResult(kind='completed')
    with pytest.raises(AttributeError):
        result.kind = 'terminal'  # type: ignore[misc]


# --- SessionManager is abstract ---------------------------------------------


def test_an_incomplete_manager_cannot_be_constructed() -> None:
    """Abstract, not a Protocol used as a base.

    A Protocol subclass inherits the ``...`` bodies, so a manager missing every
    method would construct and then return None from each call.
    """

    class Empty(SessionManager):
        backend = 'empty'

    with pytest.raises(TypeError, match='abstract'):
        Empty()  # type: ignore[abstract]


def test_a_manager_missing_one_method_is_refused() -> None:
    class NoReport(SessionManager):
        backend = 'no_report'

        async def setup(self) -> None: ...
        async def start(self, *, name: str, **kwargs: object) -> Session: ...  # type: ignore[empty-body]
        async def send(self, session: Session, message: SessionMessage) -> TurnResult: ...  # type: ignore[empty-body]
        def session_of(self, session_id: str) -> Session | None: ...  # type: ignore[empty-body]
        def traces(self, session: Session) -> dict[str, object]: ...  # type: ignore[empty-body]
        async def close(self, session: Session) -> None: ...
        async def shutdown(self) -> None: ...

    with pytest.raises(TypeError, match='report'):
        NoReport()  # type: ignore[abstract]


# --- BatchTurnResult derives its kind ---------------------------------------


def test_a_batch_turn_takes_its_kind_from_the_result() -> None:
    """One source: a batch turn cannot disagree with the turn it reports."""
    session = Session(name='a', session_id='s-a', backend='b')
    turn = BatchTurnResult(
        session=session,
        result=TurnResult(kind='needs_input'),
        inflight=frozenset({'b', 'c'}),
    )

    assert turn.kind == 'needs_input'
    assert turn.inflight == frozenset({'b', 'c'})


def test_a_session_carries_its_own_label() -> None:
    """Attribution is a property of the session, not of a tracker it holds."""
    first = Session(name='worker', session_id='s', backend='b')
    second = Session(name='reviewer', session_id='s2', backend='b')

    assert first.label == worker_label('worker')
    assert first.label != second.label
