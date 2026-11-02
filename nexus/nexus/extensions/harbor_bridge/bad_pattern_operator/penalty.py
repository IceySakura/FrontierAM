"""Reward settlement for Harbor tool-call bad-pattern detections."""

from __future__ import annotations

from typing import Any

from nexus.extensions.harbor_bridge.bad_pattern_operator.policy import ToolCallBadPatternPolicy


def apply_tool_call_bad_pattern_penalty(
    reward_score: float,
    detection: dict[str, Any],
    policy: ToolCallBadPatternPolicy,
) -> tuple[float, dict[str, float]]:
    """Deduct the largest eligible configured penalty without stacking rules.

    The Harbor Verifier reward remains authoritative and unchanged.  The returned
    score is the training-facing score, capped at zero, matching the existing
    Toolathlon bad-pattern settlement semantics. Rule-local penalty/threshold
    overrides inherit from the policy-level defaults when omitted.
    """
    original_score = float(reward_score)
    raw_counts = detection.get('issue_counts')
    raw_counts = raw_counts if isinstance(raw_counts, dict) else {}
    counts = {str(rule_id): int(count) for rule_id, count in raw_counts.items() if count}
    is_bad_pattern = bool(detection.get('is_bad_pattern')) or bool(counts)
    penalized_rules = policy.penalized_rules()
    penalty_counts = {rule_id: count for rule_id, count in counts.items() if rule_id in penalized_rules}
    is_penalty_pattern = bool(penalty_counts)
    eligible_counts = {
        rule_id: count
        for rule_id, count in penalty_counts.items()
        if original_score >= policy.reward_threshold_for(rule_id)
    }
    eligible = bool(eligible_counts)
    suggested_penalties = detection.get('suggested_penalties')
    suggested_penalties = suggested_penalties if isinstance(suggested_penalties, dict) else {}
    configured_penalties = {
        rule_id: min(
            policy.penalty_for(rule_id),
            float(suggested_penalties.get(rule_id, policy.penalty_for(rule_id))),
        )
        for rule_id in eligible_counts
    }
    max_configured_penalty = max(configured_penalties.values(), default=0.0)
    applied_penalty_rules = sorted(
        rule_id
        for rule_id, configured_penalty in configured_penalties.items()
        if configured_penalty == max_configured_penalty
    )
    applied_penalty = min(max_configured_penalty, original_score) if eligible and max_configured_penalty > 0 else 0.0
    penalized_score = original_score - applied_penalty

    metrics: dict[str, float] = {
        'tool_call_bad_pattern_hit': 1.0 if is_bad_pattern else 0.0,
        'tool_call_bad_pattern_count': float(detection.get('issue_count', 0) or 0),
        'tool_call_bad_pattern_penalty_hit': 1.0 if is_penalty_pattern else 0.0,
        'tool_call_bad_pattern_penalty_count': float(sum(penalty_counts.values())),
        'tool_call_bad_pattern_eligible': 1.0 if eligible else 0.0,
        'tool_call_bad_pattern_penalized': 1.0 if applied_penalty > 0 else 0.0,
        'tool_call_bad_pattern_penalty': applied_penalty,
        'tool_call_bad_pattern_tool_call_count': float(detection.get('tool_call_count', 0) or 0),
        'tool_call_bad_pattern_schema_registry_tool_count': float(detection.get('schema_registry_tool_count', 0) or 0),
        'tool_call_bad_pattern_schema_matched_call_count': float(detection.get('schema_matched_call_count', 0) or 0),
        'tool_call_bad_pattern_schema_coverage': float(detection.get('schema_coverage', 0.0) or 0.0),
    }
    # Emit a stable pair of metrics for every enabled rule, including zeroes.
    # Otherwise batch aggregation only sees samples where a rule fired and the
    # resulting mean is conditional on hits rather than a true batch hit rate.
    for rule_id in sorted(policy.enabled_rules()):
        count = float(counts.get(rule_id, 0))
        metrics[f'tool_call_bad_pattern/{rule_id}/count'] = count
        metrics[f'tool_call_bad_pattern/{rule_id}/hit'] = 1.0 if count > 0 else 0.0

    detection.update({
        'penalty_pattern': is_penalty_pattern,
        'penalty_issue_count': sum(penalty_counts.values()),
        'penalty_issue_counts': dict(sorted(penalty_counts.items())),
        'penalty_eligible_issue_count': sum(eligible_counts.values()),
        'penalty_eligible_issue_counts': dict(sorted(eligible_counts.items())),
        'penalty_applied_rule_ids': applied_penalty_rules if applied_penalty > 0 else [],
        'eligible': eligible,
        'penalized': applied_penalty > 0,
        'penalty': applied_penalty,
        'original_score': original_score,
        'penalized_score': penalized_score,
    })
    return penalized_score, metrics


__all__ = ['apply_tool_call_bad_pattern_penalty']
