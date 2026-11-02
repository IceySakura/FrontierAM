"""Public detection engine.

The engine builds one normalized :class:`DetectionContext`, invokes independent
rule modules, and serializes their findings. Rule logic lives under ``rules/``.
"""

from __future__ import annotations

import math

from collections import Counter
from typing import Any

from .context import DEFAULT_MAX_THINKING_CHARS, DetectionContext
from .models import Finding
from .rules import RULES, RULE_TITLES
from .utils import excerpt


def _serialize(rule_id: str, title: str, finding: Finding, context: DetectionContext) -> dict[str, Any]:
    index = finding.message_index
    item: dict[str, Any] = {
        'problem_type': rule_id,
        'title': title,
        'message_index': index,
        'turn_index': index,
        'role': context.messages[index].get('role')
        if 0 <= index < len(context.messages) and isinstance(context.messages[index], dict)
        else None,
        'assistant_turn': context.assistant_numbers.get(index),
        'message': finding.message,
        'confidence': finding.confidence,
    }
    if finding.path is not None:
        item['path'] = finding.path
    if finding.excerpt not in (None, ''):
        item['excerpt'] = excerpt(finding.excerpt)
    if finding.tool_call_index is not None:
        item['tool_call_index'] = finding.tool_call_index
    if finding.tool_call_id:
        item['tool_call_id'] = finding.tool_call_id
    if finding.origin_message_index is not None:
        item['origin_message_index'] = finding.origin_message_index
        item['origin_assistant_turn'] = context.assistant_numbers.get(finding.origin_message_index)
    item['is_issue'] = finding.is_issue
    if finding.severity is not None:
        item['severity'] = finding.severity
    if finding.detail:
        item['detail'] = finding.detail
    if finding.quotes:
        item['quotes'] = finding.quotes
    return item


def detect_tool_call_bad_patterns(
    messages: list[dict[str, Any]] | None,
    tools: list[dict[str, Any]] | None = None,
    *,
    reasoning_effort: str | None = None,
    max_hits: int = 100,
    enabled_rules: set[str] | None = None,
    ignore_incomplete_final_assistant: bool = True,
    termination_reason: str | None = None,
    max_thinking_chars: int = DEFAULT_MAX_THINKING_CHARS,
) -> dict[str, Any]:
    """Detect all enabled bad patterns and return turn-level structured labels."""
    unknown = (enabled_rules or set()) - set(RULE_TITLES)
    if unknown:
        raise ValueError(f'unknown rules: {", ".join(sorted(unknown))}')
    context = DetectionContext(
        messages,
        tools,
        reasoning_effort=reasoning_effort,
        ignore_incomplete_final_assistant=ignore_incomplete_final_assistant,
        termination_reason=termination_reason,
        max_thinking_chars=max_thinking_chars,
    )
    hits: list[dict[str, Any]] = []
    for rule in RULES:
        if enabled_rules is not None and rule.id not in enabled_rules:
            continue
        for finding in rule.scan(context):
            hits.append(_serialize(rule.id, rule.title, finding, context))

    deduped: list[dict[str, Any]] = []
    seen = set()
    for item in hits:
        # Distinct schema violations on the same call (for example a type
        # mismatch plus an undeclared argument) share the same container path.
        # Include the diagnostic text so count metrics retain both findings
        # while exact duplicate emissions are still collapsed.
        key = (
            item['problem_type'],
            item['message_index'],
            item.get('tool_call_index'),
            tuple(item.get('path', [])),
            item.get('message'),
        )
        if key not in seen:
            seen.add(key)
            deduped.append(item)
    issue_hits = [item for item in deduped if item.get('is_issue', True)]
    nonissue_evidence = [item for item in deduped if not item.get('is_issue', True)]
    counts = Counter(item['problem_type'] for item in issue_hits)
    suggested_penalties: dict[str, float] = {}
    for item in issue_hits:
        detail = item.get('detail')
        if not isinstance(detail, dict) or 'suggested_penalty' not in detail:
            continue
        rule_id = str(item['problem_type'])
        suggested_penalty = float(detail['suggested_penalty'])
        if not math.isfinite(suggested_penalty) or suggested_penalty < 0:
            raise ValueError(
                f'Rule {rule_id!r} emitted invalid suggested_penalty {suggested_penalty!r}; '
                'expected a finite non-negative number'
            )
        suggested_penalties[rule_id] = max(suggested_penalties.get(rule_id, 0.0), suggested_penalty)
    bad_turns = sorted({item.get('origin_message_index', item['message_index']) for item in issue_hits})
    named_calls = [event for event in context.calls if event.name]
    schema_matched_calls = [event for event in named_calls if event.name in context.registry]
    return {
        'is_bad_pattern': bool(issue_hits),
        'issue_count': len(issue_hits),
        'issue_counts': dict(sorted(counts.items())),
        'suggested_penalties': dict(sorted(suggested_penalties.items())),
        'bad_turns': bad_turns,
        'assistant_turn_count': len(context.assistant_numbers),
        'tool_call_count': len(context.calls),
        'named_tool_call_count': len(named_calls),
        'schema_registry_tool_count': len(context.registry),
        'schema_matched_call_count': len(schema_matched_calls),
        # Zero tool calls contribute zero matched/total counts. Consumers that
        # need call-weighted coverage should divide the aggregated matched-call
        # count by the aggregated named-call count; the per-trajectory ratio is
        # still useful for spotting trajectories whose advertised schemas are
        # partially missing.
        'schema_coverage': len(schema_matched_calls) / len(named_calls) if named_calls else 0.0,
        'hits': issue_hits[: max(0, max_hits)],
        'hits_truncated': len(issue_hits) > max_hits,
        'nonissue_evidence': nonissue_evidence[: max(0, max_hits)],
        'nonissue_evidence_count': len(nonissue_evidence),
    }


def detect_record(record: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    return detect_tool_call_bad_patterns(
        record.get('messages'),
        record.get('tools'),
        reasoning_effort=record.get('reasoning_effort'),
        **kwargs,
    )
