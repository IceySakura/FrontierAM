"""Detect dense ``let me`` / ``让me`` / ``让我`` repetition in assistant output.

A stalled model restates the same intent over and over without ever acting.
Three markers inside a 200-character window leaves little room for anything
else in that span, so the threshold flags restatement rather than a marker that
merely introduces the next step.

The threshold was measured against a step of real rollouts rather than picked
for tidiness. Six markers matched nothing at all: every trajectory used the
phrase, often more than a hundred times, but spread out. Four flagged 1.6% of
solved and 29.5% of failed trajectories; three flags 44% and 82%. Three is
therefore a deliberately wide net that fires on ordinary verbosity, not only on
a hard stall -- callers that deduct reward should size the coefficient for that.

Shared by the SWE / terminal / code_mt monitors. Flag-only: callers mirror the
result into ``extra_reward_metrics`` so the trajectory hit rate shows up as
``critic/extra_reward_metrics/let_me_loop_hit/mean``.
"""

from __future__ import annotations

import re

from typing import Any


WINDOW_CHARS = 200
MIN_OCCURRENCES = 3

_HSPACE = r'[\t \u00a0\u1680\u2000-\u200b\u202f\u205f\u3000]'
_HSPACE_RE = re.compile(_HSPACE)
_MARKER_RE = re.compile(
    rf'(?<![A-Za-z])let{_HSPACE}+me(?![A-Za-z])|让{_HSPACE}*me|让我',
    re.IGNORECASE,
)
_INLINE_CODE_RE = re.compile(r'`[^`\n]*`')
_EN_NEGATION_RE = re.compile(
    r'(?:do\s+not|don[\'’]?t|won[\'’]?t|can(?:not|[\'’]?t)|never|not)\s*$',
    re.IGNORECASE,
)
_ZH_NEGATION_RE = re.compile(r'(?:不|没|别|勿|不能|不会|不再|没有)\s*$')

_NEGATION_LOOKBACK = 24
_SNIPPET_CONTEXT = 80
_MAX_HITS = 5


def _variant(surface: str) -> str:
    compact = _HSPACE_RE.sub('', surface).lower()
    if compact == 'letme':
        return 'let me'
    if compact == '让me':
        return '让me'
    return '让我'


def _excluded_ranges(text: str) -> list[tuple[int, int]]:
    """Character ranges holding code, which must not count as prose repetition.

    Returned merged and non-overlapping so a fenced block that also contains
    inline backticks stays a single range.
    """
    ranges: list[tuple[int, int]] = []
    offset = 0
    fence: str | None = None
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip()
        marker = stripped[:3]
        if fence:
            ranges.append((offset, offset + len(line)))
            if marker == fence:
                fence = None
        elif marker in {'```', '~~~'}:
            fence = marker
            ranges.append((offset, offset + len(line)))
        elif stripped.startswith('>'):
            ranges.append((offset, offset + len(line)))
        offset += len(line)
    ranges.extend((match.start(), match.end()) for match in _INLINE_CODE_RE.finditer(text))

    merged: list[tuple[int, int]] = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _is_negated(text: str, start: int, variant: str) -> bool:
    prefix = text[max(0, start - _NEGATION_LOOKBACK) : start]
    if variant == 'let me':
        return bool(_EN_NEGATION_RE.search(prefix))
    return bool(_ZH_NEGATION_RE.search(prefix))


def _eligible_matches(text: str) -> list[re.Match[str]]:
    ranges = _excluded_ranges(text)
    range_index = 0
    eligible: list[re.Match[str]] = []
    for match in _MARKER_RE.finditer(text):
        while range_index < len(ranges) and ranges[range_index][1] <= match.start():
            range_index += 1
        if range_index < len(ranges):
            start, end = ranges[range_index]
            if start < match.end() and match.start() < end:
                continue
        if not _is_negated(text, match.start(), _variant(match.group(0))):
            eligible.append(match)
    return eligible


def _densest_window(
    matches: list[re.Match[str]],
    window_chars: int,
    min_occurrences: int,
) -> tuple[int, int, int] | None:
    """Return ``(count, start, end)`` for the densest qualifying window.

    Ties break towards the tightest span, then the earliest position, so the
    reported snippet is the most damning one rather than the first found.
    """
    if len(matches) < min_occurrences:
        return None
    best_key: tuple[int, int, int] | None = None
    best: tuple[int, int, int] | None = None
    left = 0
    for right, match in enumerate(matches):
        while left <= right and match.end() - matches[left].start() > window_chars:
            left += 1
        count = right - left + 1
        if count < min_occurrences:
            continue
        start = matches[left].start()
        end = match.end()
        key = (-count, end - start, start)
        if best_key is None or key < best_key:
            best_key = key
            best = (count, start, end)
    return best


def _snippet(text: str, start: int, end: int) -> str:
    left = max(0, start - _SNIPPET_CONTEXT)
    right = min(len(text), end + _SNIPPET_CONTEXT)
    return text[left:right].replace('\n', '\\n')


def _assistant_fields(messages: list[dict] | None) -> list[tuple[int, str, str]]:
    """Yield ``(turn_index, field, text)`` for assistant reasoning and content.

    Both are scanned because a model that emits no ``<think>`` block puts its
    whole reasoning in ``content``; the two are reported separately so the
    split stays visible.
    """
    fields: list[tuple[int, str, str]] = []
    for turn_index, message in enumerate(messages or []):
        if not isinstance(message, dict) or message.get('role') != 'assistant':
            continue
        for field in ('reasoning_content', 'content'):
            text = message.get(field)
            if isinstance(text, str) and text:
                fields.append((turn_index, field, text))
    return fields


def detect_let_me_loop(
    messages: list[dict] | None,
    *,
    window_chars: int = WINDOW_CHARS,
    min_occurrences: int = MIN_OCCURRENCES,
) -> dict[str, Any]:
    """Flag assistant turns that repeat ``let me`` markers at stall density."""
    hits: list[dict[str, Any]] = []
    marker_counts = {'reasoning_content': 0, 'content': 0}
    hit_field_names: set[str] = set()
    max_window_count = 0

    for turn_index, field, text in _assistant_fields(messages):
        matches = _eligible_matches(text)
        marker_counts[field] += len(matches)
        window = _densest_window(matches, window_chars, min_occurrences)
        if window is None:
            continue
        count, start, end = window
        hit_field_names.add(field)
        max_window_count = max(max_window_count, count)
        hits.append({
            'turn_index': turn_index,
            'field': field,
            'count': count,
            'span_chars': end - start,
            'snippet': _snippet(text, start, end),
        })

    reasoning_count = marker_counts['reasoning_content']
    content_count = marker_counts['content']
    return {
        'is_let_me_loop': bool(hits),
        'reasoning_hit': 'reasoning_content' in hit_field_names,
        'content_hit': 'content' in hit_field_names,
        # One entry per (turn, field) pair that crossed the density threshold.
        'hit_fields': len(hits),
        'max_window_count': max_window_count,
        'marker_count': reasoning_count + content_count,
        'reasoning_marker_count': reasoning_count,
        'content_marker_count': content_count,
        'window_chars': window_chars,
        'min_occurrences': min_occurrences,
        'hits': hits[:_MAX_HITS],
    }


def apply_let_me_loop_penalty(
    reward_score: float | None,
    extra_fields: dict[str, Any],
    penalty: float = 0.0,
    threshold: float = 1.0,
) -> float | None:
    """Mirror the detection into metrics and optionally deduct from the reward.

    Bails when the monitor did not run, so a task with the family switched off
    reports nothing rather than a misleading zero. With the default ``penalty``
    of 0 this only writes metadata, which is the flag-only mode the metrics are
    collected in; a configured penalty deducts from a trajectory that scored at
    or above ``threshold``, matching the other quality penalties.
    """
    info = extra_fields.get('let_me_loop')
    if not isinstance(info, dict):
        return reward_score

    hit = bool(info.get('is_let_me_loop'))
    try:
        original_score = float(reward_score) if reward_score is not None else None
    except (TypeError, ValueError):
        original_score = None

    is_eligible = hit and original_score is not None and original_score >= threshold
    should_penalize = is_eligible and penalty > 0
    applied_penalty = 0.0

    if should_penalize:
        assert original_score is not None  # guaranteed by is_eligible check
        applied_penalty = min(float(penalty), original_score)
        reward_score = original_score - applied_penalty
        info['penalized_score'] = reward_score

    info['eligible'] = is_eligible
    info['penalized'] = should_penalize
    info['penalty'] = applied_penalty
    if original_score is not None:
        info['original_score'] = original_score

    reward_extra_info = extra_fields.get('reward_extra_info')
    if not isinstance(reward_extra_info, dict):
        reward_extra_info = {}
        extra_fields['reward_extra_info'] = reward_extra_info
    metrics = reward_extra_info.get('extra_reward_metrics')
    if not isinstance(metrics, dict):
        metrics = {}
        reward_extra_info['extra_reward_metrics'] = metrics

    reward_extra_info['is_let_me_loop'] = 1.0 if hit else 0.0
    reward_extra_info['let_me_loop_eligible'] = 1.0 if is_eligible else 0.0
    reward_extra_info['let_me_loop_penalized'] = 1.0 if should_penalize else 0.0
    reward_extra_info['let_me_loop_penalty'] = applied_penalty
    if original_score is not None and hit:
        reward_extra_info['let_me_loop_original_score'] = original_score

    metrics['let_me_loop_hit'] = 1.0 if hit else 0.0
    metrics['let_me_loop_reasoning_hit'] = 1.0 if info.get('reasoning_hit') else 0.0
    metrics['let_me_loop_content_hit'] = 1.0 if info.get('content_hit') else 0.0
    metrics['let_me_loop_hit_fields'] = float(info.get('hit_fields') or 0)
    metrics['let_me_loop_max_window_count'] = float(info.get('max_window_count') or 0)
    metrics['let_me_loop_marker_count'] = float(info.get('marker_count') or 0)
    metrics['let_me_loop_reasoning_marker_count'] = float(info.get('reasoning_marker_count') or 0)
    metrics['let_me_loop_content_marker_count'] = float(info.get('content_marker_count') or 0)
    metrics['let_me_loop_eligible'] = 1.0 if is_eligible else 0.0
    metrics['let_me_loop_penalized'] = 1.0 if should_penalize else 0.0
    metrics['let_me_loop_penalty'] = applied_penalty

    return reward_score


__all__ = [
    'MIN_OCCURRENCES',
    'WINDOW_CHARS',
    'apply_let_me_loop_penalty',
    'detect_let_me_loop',
]
