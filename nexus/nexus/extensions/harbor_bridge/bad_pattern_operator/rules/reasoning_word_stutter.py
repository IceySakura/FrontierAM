"""Detect sparse adjacent word stutters across assistant thinking turns."""

from __future__ import annotations

from collections import Counter

from ._reasoning_stutter import DETECTOR_VERSION, MIN_OCCURRENCES, occurrences
from .base import Rule
from .single_turn_thinking_too_long import thinking_span
from ..context import DetectionContext
from ..models import Finding


def _excerpt(text: str, start: int, end: int) -> str:
    return text[max(0, start - 100) : min(len(text), end + 100, start + 220)]


def scan(ctx: DetectionContext) -> list[Finding]:
    stutters = []
    word_counts = Counter()
    for index, message in ctx.assistant_messages():
        thinking, path, offset = thinking_span(message, ctx.content(index))
        matches = occurrences(thinking)
        if matches:
            stutters.append((index, thinking, path, offset, matches))
            word_counts.update(match['word'] for match in matches)
    total = sum(word_counts.values())
    if total < MIN_OCCURRENCES:
        return []

    findings = []
    for index, thinking, path, offset, matches in stutters:
        source_matches = [
            {**match, 'start': match['start'] + offset, 'end': match['end'] + offset} for match in matches
        ]
        evidence = {
            'detector_version': DETECTOR_VERSION,
            'trajectory_occurrences': total,
            'min_trajectory_occurrences': MIN_OCCURRENCES,
            'trajectory_word_counts': dict(word_counts),
            'message_occurrences': len(matches),
            'occurrences': source_matches[:20],
            'occurrences_truncated': len(matches) > 20,
        }
        quotes = [
            {
                'kind': 'adjacent_word_stutter',
                'turn_index': index,
                'role': 'assistant',
                'word': match['word'],
                'start': match['start'] + offset,
                'end': match['end'] + offset,
                'repeats': match['repeats'],
                'snippet': _excerpt(thinking, match['start'], match['end']),
            }
            for match in matches[:3]
        ]
        findings.append(
            Finding(
                message_index=index,
                message=f'Assistant thinking contains adjacent word stutters at {total} locations across the trajectory.',
                path=path,
                excerpt=quotes[0]['snippet'],
                severity='medium',
                detail={
                    'channel': 'reasoning',
                    'checks': ['adjacent_word_stutter'],
                    'adjacent_word_stutter': evidence,
                },
                quotes=quotes,
            )
        )
    return findings


RULE = Rule('reasoning_word_stutter', 'Adjacent word stutters in assistant thinking', scan)
