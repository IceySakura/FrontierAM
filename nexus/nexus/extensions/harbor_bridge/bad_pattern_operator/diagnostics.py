"""Structured diagnostics for selected bad-pattern rules."""

from __future__ import annotations

from typing import Any

from nexus.extensions.harbor_bridge.bad_pattern_operator.policy import ToolCallBadPatternPolicy


FINAL_TURN_LANGUAGE_CH_RULE = 'final_turn_language_ch'


def final_turn_language_ch_diagnostic(
    detection: dict[str, Any],
    policy: ToolCallBadPatternPolicy,
    *,
    agent_name: str | None,
    instance_id: str,
    original_score: float,
    final_score: float,
) -> dict[str, Any] | None:
    """Build one content-free, grep-friendly diagnostic for the Chinese rule."""
    if FINAL_TURN_LANGUAGE_CH_RULE not in policy.enabled_rules_for_agent(agent_name):
        return None

    evidence = next(
        (
            item
            for item in [
                *(detection.get('hits') or []),
                *(detection.get('nonissue_evidence') or []),
            ]
            if isinstance(item, dict) and item.get('problem_type') == FINAL_TURN_LANGUAGE_CH_RULE
        ),
        None,
    )
    detail = evidence.get('detail') if isinstance(evidence, dict) else None
    detail = detail if isinstance(detail, dict) else {}
    issue_counts = detection.get('issue_counts')
    issue_counts = issue_counts if isinstance(issue_counts, dict) else {}
    hit = int(issue_counts.get(FINAL_TURN_LANGUAGE_CH_RULE, 0) or 0) > 0
    penalty_enabled = FINAL_TURN_LANGUAGE_CH_RULE in policy.penalized_rules()
    reward_threshold = policy.reward_threshold_for(FINAL_TURN_LANGUAGE_CH_RULE)
    threshold_met = float(original_score) >= reward_threshold
    applied_rule_ids = detection.get('penalty_applied_rule_ids')
    applied_rule_ids = applied_rule_ids if isinstance(applied_rule_ids, list) else []
    penalized = FINAL_TURN_LANGUAGE_CH_RULE in applied_rule_ids

    return {
        'event': FINAL_TURN_LANGUAGE_CH_RULE,
        'status': 'completed',
        'entered': True,
        'instance_id': instance_id,
        'agent_name': agent_name,
        'policy_id': policy.id,
        'rule_enabled': True,
        'evidence_found': evidence is not None,
        'hit': hit,
        'message_index': evidence.get('message_index') if isinstance(evidence, dict) else None,
        'chinese_ratio': detail.get('chinese_ratio'),
        'chinese_units': detail.get('chinese_units'),
        'total_language_units': detail.get('total_language_units'),
        'script_units': detail.get('script_units'),
        'suggested_penalty': detail.get('suggested_penalty'),
        'suppressed': bool(detail.get('suppressed', False)),
        'suppressed_reason': detail.get('suppressed_reason'),
        'termination_reason': detail.get('termination_reason'),
        'penalty_enabled': penalty_enabled,
        'configured_penalty_cap': policy.penalty_for(FINAL_TURN_LANGUAGE_CH_RULE),
        'reward_threshold': reward_threshold,
        'threshold_met': threshold_met,
        'eligible': hit and penalty_enabled and threshold_met,
        'penalized': penalized,
        'applied_penalty': float(detection.get('penalty', 0.0) or 0.0) if penalized else 0.0,
        'original_score': float(original_score),
        'final_score': float(final_score),
    }


__all__ = ['FINAL_TURN_LANGUAGE_CH_RULE', 'final_turn_language_ch_diagnostic']
