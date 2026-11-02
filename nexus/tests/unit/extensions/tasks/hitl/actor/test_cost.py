"""Per-turn worker cost: the context length read off the tracker, and how it is worded."""

from collections.abc import Iterator
from contextlib import contextmanager

from nexus.extensions.tasks.hitl.actor import cost
from nexus.extensions.tasks.hitl.actor.base import Session, TurnCost
from nexus.extensions.tasks.hitl.actor.cost import account_worker_turn, append_cost, render_cost
from nexus.extensions.tasks.hitl.trace.usage import HITLUsageProcessor
from nexus.extensions.tasks.hitl.trace.usage.parse import UsageTrace
from nexus.extensions.tasks.hitl.trace.usage.tracker import UsageTracker
from nexus.trace.context import context_processors


def a_session() -> Session:
    return Session(name='worker', session_id='s1', backend='print')


@contextmanager
def usage_with_call_inputs(label: str, *inputs: int) -> Iterator[None]:
    """A usage processor in context whose label recorded calls of the given input lengths."""
    tracker = UsageTracker()
    for input_tokens in inputs:
        tracker._add_point(UsageTrace(profile='default', valid=True, input_tokens=input_tokens))
    processor = HITLUsageProcessor()
    processor._by_label[label] = tracker
    token = context_processors.set([processor])
    try:
        yield
    finally:
        context_processors.reset(token)


def test_a_turn_reports_the_latest_calls_length_not_the_peak() -> None:
    """A round ends on the worker's own message, so the last call is its current context;
    a big call followed by a shorter one reports the shorter."""
    session = a_session()
    with usage_with_call_inputs(session.label, 200_000, 150_000):
        turn = account_worker_turn(session, seconds=12.0)

    assert turn.context_tokens == 150_000
    assert turn.duration_s == 12.0


def test_off_the_trace_path_the_read_is_zero_not_a_crash() -> None:
    """With no usage processor in context the read returns 0."""
    assert cost._worker_context_tokens('worker/worker') == 0
    assert account_worker_turn(a_session(), seconds=1.0).context_tokens == 0


def test_a_rendered_round_shows_time_and_context_size() -> None:
    line = render_cost(TurnCost(duration_s=200.0, context_tokens=180_000))

    assert '3m20s' in line
    assert '180k' in line


def test_a_rendered_round_never_speaks_of_scoring_or_a_budget() -> None:
    """The user watches what a round cost; it is not told a scored budget."""
    line = render_cost(TurnCost(duration_s=5.0, context_tokens=1_200)).lower()

    for leak in ('budget', 'score', 'scoring', 'hidden', 'cap'):
        assert leak not in line


def test_append_cost_adds_the_line_only_when_a_turn_has_one() -> None:
    turn_cost = TurnCost(duration_s=5.0, context_tokens=1_200)

    assert append_cost('the report', None) == 'the report'
    appended = append_cost('the report', turn_cost)
    assert appended.startswith('the report')
    assert appended.endswith(render_cost(turn_cost))
