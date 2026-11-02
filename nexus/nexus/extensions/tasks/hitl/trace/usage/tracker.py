"""Accumulating parsed usage into per-source totals."""

import logging

from collections.abc import Iterable, Mapping

from nexus.extensions.tasks.hitl.trace.usage.parse import UsageTrace, model_of, parse_usage
from nexus.extensions.tasks.hitl.trace.usage.profiles import DEFAULT_PROFILE, UsageProfile


logger = logging.getLogger(__name__)


# Counters that sum across calls. `calls` and `invalid_calls` are counts of calls
# rather than of tokens, and sum the same way.
SUMMED_FIELDS = (
    'calls',
    'invalid_calls',
    'input_tokens',
    'output_tokens',
    'total_tokens',
    'cache_read_tokens',
    'cache_creation_tokens',
    'reasoning_tokens',
)

# One counter per turn, in the order the series lists them. A turn is one completion:
# the user agent's loop makes exactly one per turn.
SERIES_FIELDS = ('input', 'output', 'cache_read', 'cache_creation', 'reasoning', 'messages')

# How many turns one label keeps. Past it the middle is dropped and both ends are
# kept. Set beyond reach: a turn costs 32 bytes, and the longest single label across
# the recorded batches made 411 calls. Split evenly between the ends, so this stays
# even.
SERIES_LIMIT = 2_147_483_646

SERIES_SEPARATOR = ','


class UsageTracker:
    """Totals for one source, kept per model as well as overall.

    Also a per-turn series and two high-water marks. A turn's ``input_tokens`` *is*
    the length of the history it was sent, so the series is the growth curve and its
    maximum is the longest the conversation ever got -- which the totals cannot say,
    and which is the number a turn cap or a compaction threshold is chosen against.
    ``messages`` is that same length counted in messages rather than tokens.

    Both peaks are maxima rather than last values: a history does not only grow.
    Compaction shortens it, and a worker's own sub-calls carry shorter lists, so the
    final turn's length is not the run's high-water mark.

    The series is one comma-separated string per counter rather than one array per
    turn, because the record is read by people: 264 turns as arrays render as 2116
    lines of a pretty-printed record against 10 lines this way, and fewer bytes. The
    response itself is still not kept -- the trajectories already hold what was sent.
    """

    def __init__(self, profile: UsageProfile = DEFAULT_PROFILE) -> None:
        self.profile = profile
        self._totals = dict.fromkeys(SUMMED_FIELDS, 0)
        self._by_model: dict[str, dict[str, int]] = {}
        # Held in two halves so truncation drops the middle rather than the tail:
        # the tail is where the run ended up, which is the half a reader wants most.
        self._head: list[tuple[int, ...]] = []
        self._tail: list[tuple[int, ...]] = []
        self._dropped = 0
        self._peak_input = 0
        self._peak_messages = 0

    def record(
        self,
        response_dump: Mapping[str, object],
        *,
        model: str = '',
        profile: UsageProfile | None = None,
        messages: int = 0,
    ) -> UsageTrace:
        """Add one turn's usage. Returns what was parsed, valid or not.

        ``messages`` is how many messages the request carried, which the caller reads
        off the span; 0 when it is not known.
        """
        trace = parse_usage(profile or self.profile, response_dump)
        if not trace.valid:
            self._totals['invalid_calls'] += 1
            return trace

        self._totals['calls'] += 1
        model_totals = self._by_model.setdefault(
            model or model_of(response_dump) or 'unknown',
            dict.fromkeys(SUMMED_FIELDS, 0),
        )
        model_totals['calls'] += 1
        for field, value in _token_fields(trace):
            self._totals[field] += value
            model_totals[field] += value
        self._add_point(trace, messages=messages)
        return trace

    def _add_point(self, trace: UsageTrace, *, messages: int = 0) -> None:
        """Keep this turn's point, dropping from the middle once past the limit."""
        self._peak_input = max(self._peak_input, trace.input_tokens)
        self._peak_messages = max(self._peak_messages, messages)
        point = (
            trace.input_tokens,
            trace.output_tokens,
            trace.cache_read_tokens,
            trace.cache_creation_tokens,
            trace.reasoning_tokens,
            messages,
        )
        half = SERIES_LIMIT // 2
        if len(self._head) < half:
            self._head.append(point)
            return
        self._tail.append(point)
        if len(self._tail) > half:
            self._tail.pop(0)
            self._dropped += 1

    def series(self) -> dict[str, str]:
        """One comma-separated series per counter, in turn order."""
        points = (*self._head, *self._tail)
        return {
            field: SERIES_SEPARATOR.join(str(point[index]) for point in points)
            for index, field in enumerate(SERIES_FIELDS)
        }

    def summary(self) -> dict[str, object]:
        return {
            **self._totals,
            'by_model': {model: dict(totals) for model, totals in self._by_model.items()},
            'peak_input_tokens': self._peak_input,
            'peak_messages': self._peak_messages,
            'series': self.series(),
            'series_dropped': self._dropped,
        }


def _token_fields(trace: UsageTrace) -> tuple[tuple[str, int], ...]:
    return (
        ('input_tokens', trace.input_tokens),
        ('output_tokens', trace.output_tokens),
        ('total_tokens', trace.total_tokens),
        ('cache_read_tokens', trace.cache_read_tokens),
        ('cache_creation_tokens', trace.cache_creation_tokens),
        ('reasoning_tokens', trace.reasoning_tokens),
    )


def combine(summaries: Iterable[Mapping[str, object]]) -> dict[str, int]:
    """Sum the counters across summaries, leaving per-model detail to each one.

    A free function so a reader of an exported record totals the same way the run
    did, instead of reimplementing it.
    """
    totals = dict.fromkeys(SUMMED_FIELDS, 0)
    for summary in summaries:
        for key in SUMMED_FIELDS:
            value = summary.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                totals[key] += value
    return totals


__all__ = ['SERIES_FIELDS', 'SERIES_LIMIT', 'SERIES_SEPARATOR', 'SUMMED_FIELDS', 'UsageTracker', 'combine']
