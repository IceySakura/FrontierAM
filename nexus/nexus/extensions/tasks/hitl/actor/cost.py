"""Per-turn worker cost: how long a turn took and how much context the worker carries.

The context size is the most recent call's input length under the session's label, not
the sum of every call's tokens: a round is an agent loop of many calls that each resend
the whole history, so a sum counts it many times over.
"""

import logging

from nexus.extensions.tasks.hitl.actor.base import Session, TurnCost
from nexus.extensions.tasks.hitl.trace.usage import HITLUsageProcessor
from nexus.trace.context import get_processor


logger = logging.getLogger(__name__)


def account_worker_turn(session: Session, *, seconds: float) -> TurnCost:
    """This turn's cost: ``seconds`` elapsed, and the context the worker is carrying."""
    return TurnCost(duration_s=seconds, context_tokens=_worker_context_tokens(session.label))


def _worker_context_tokens(label: str) -> int:
    """The history length of the label's most recent call, 0 with no processor.

    The last call, not the peak: a round ends on the worker's own message, so it follows
    a compaction downward too.
    """
    processor = get_processor(HITLUsageProcessor)
    if not isinstance(processor, HITLUsageProcessor):
        return 0
    entry = processor.get_result().get('by_label', {}).get(label) or {}
    series = entry.get('series') if isinstance(entry, dict) else None
    inputs = str(series.get('input', '')).split(',') if isinstance(series, dict) else []
    tail = [value for value in inputs if value.strip()]
    return int(tail[-1]) if tail else 0


def render_cost(cost: TurnCost) -> str:
    """The line a worker report carries: this turn's time, and the worker's context size."""
    return (
        f'This round took {_human_duration(cost.duration_s)}; '
        f'the worker is carrying about {_human_tokens(cost.context_tokens)} tokens of context.'
    )


def append_cost(report: str, cost: TurnCost | None) -> str:
    """A turn report with its cost line after it, or unchanged when there is no cost."""
    if cost is None:
        return report
    line = render_cost(cost)
    return f'{report}\n\n{line}' if report.strip() else line


def _human_duration(seconds: float) -> str:
    whole = int(round(seconds))
    if whole < 60:
        return f'{whole}s'
    minutes, rem = divmod(whole, 60)
    return f'{minutes}m{rem:02d}s'


def _human_tokens(count: int) -> str:
    if count < 1000:
        return str(count)
    if count < 1_000_000:
        return f'{count / 1000:.0f}k'
    return f'{count / 1_000_000:.1f}M'


__all__ = ['account_worker_turn', 'append_cost', 'render_cost']
