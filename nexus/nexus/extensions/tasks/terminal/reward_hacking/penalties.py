"""Reward penalties for terminal / terminus_bench tasks.

Self-contained: detection happens upstream in ``monitors.py`` (writes
``codeswitch`` / ``output_format_bad_pattern`` / ``terminus_json_parse_error``
into ``extra_fields``); this module consumes those keys and mutates the
score + extra_fields in-place. Each sub-penalty bails when its specific
detection key is absent.
"""

from __future__ import annotations

import logging

from typing import Any

from nexus.utils.reasoning_repetition import apply_let_me_loop_penalty


logger = logging.getLogger(__name__)


def _ensure_reward_extra_info(extra_fields: dict[str, Any]) -> dict[str, Any]:
    reward_extra_info = extra_fields.get('reward_extra_info')
    if not isinstance(reward_extra_info, dict):
        reward_extra_info = {}
        extra_fields['reward_extra_info'] = reward_extra_info
    return reward_extra_info


def _ensure_extra_reward_metrics(reward_extra_info: dict[str, Any]) -> dict[str, Any]:
    extra_reward_metrics = reward_extra_info.get('extra_reward_metrics')
    if not isinstance(extra_reward_metrics, dict):
        extra_reward_metrics = {}
        reward_extra_info['extra_reward_metrics'] = extra_reward_metrics
    return extra_reward_metrics


def _apply_codeswitch_penalty(
    reward_score: float | None,
    extra_fields: dict[str, Any],
    penalty: float,
    threshold: float,
) -> float | None:
    codeswitch_info = extra_fields.get('codeswitch')
    if not isinstance(codeswitch_info, dict):
        return reward_score

    is_codeswitch = bool(codeswitch_info.get('is_codeswitch'))

    try:
        original_score = float(reward_score) if reward_score is not None else None
    except (TypeError, ValueError):
        original_score = None

    is_eligible = is_codeswitch and original_score is not None and original_score >= threshold
    should_penalize = is_eligible and penalty > 0
    applied_penalty = 0.0

    if should_penalize:
        assert original_score is not None  # guaranteed by is_eligible check
        applied_penalty = min(float(penalty), original_score)
        reward_score = original_score - applied_penalty
        codeswitch_info['penalized_score'] = reward_score

    reward_extra_info = _ensure_reward_extra_info(extra_fields)
    codeswitch_info['penalized'] = should_penalize
    codeswitch_info['penalty'] = applied_penalty
    if original_score is not None:
        codeswitch_info['original_score'] = original_score

    reward_extra_info['is_codeswitch'] = 1.0 if is_codeswitch else 0.0
    reward_extra_info['codeswitch_eligible'] = 1.0 if is_eligible else 0.0
    reward_extra_info['codeswitch_penalized'] = 1.0 if should_penalize else 0.0
    reward_extra_info['codeswitch_penalty'] = applied_penalty
    if original_score is not None and is_codeswitch:
        reward_extra_info['original_score'] = original_score

    extra_reward_metrics = _ensure_extra_reward_metrics(reward_extra_info)
    extra_reward_metrics['codeswitch_detected'] = 1.0 if is_codeswitch else 0.0
    extra_reward_metrics['codeswitch_hit'] = 1.0 if is_codeswitch else 0.0
    extra_reward_metrics['codeswitch_eligible'] = 1.0 if is_eligible else 0.0
    extra_reward_metrics['codeswitch_penalized'] = 1.0 if should_penalize else 0.0
    extra_reward_metrics['codeswitch_penalty'] = applied_penalty

    return reward_score


def _apply_output_format_bad_pattern_penalty(
    reward_score: float | None,
    extra_fields: dict[str, Any],
    penalty: float,
    threshold: float,
) -> float | None:
    bad_pattern_info = extra_fields.get('output_format_bad_pattern')
    if not isinstance(bad_pattern_info, dict):
        return reward_score

    f6_7_count = int(float(bad_pattern_info.get('F6_7_count', 0) or 0))
    f7_2_count = int(float(bad_pattern_info.get('F7_2_count', 0) or 0))
    f7_3_count = int(float(bad_pattern_info.get('F7_3_count', 0) or 0))
    f6_7_hit = bool(bad_pattern_info.get('F6_7_think_tag_in_content')) or f6_7_count > 0
    f7_2_hit = bool(bad_pattern_info.get('F7_2_reasoning_collapsed_tool_calls')) or f7_2_count > 0
    f7_3_hit = bool(bad_pattern_info.get('F7_3_tool_call_leak_in_content')) or f7_3_count > 0
    is_bad_pattern = bool(bad_pattern_info.get('is_bad_pattern')) or f6_7_hit or f7_2_hit or f7_3_hit

    try:
        original_score = float(reward_score) if reward_score is not None else None
    except (TypeError, ValueError):
        original_score = None

    is_eligible = is_bad_pattern and original_score is not None and original_score >= threshold
    should_penalize = is_eligible and penalty > 0
    applied_penalty = 0.0

    if should_penalize:
        assert original_score is not None  # guaranteed by is_eligible check
        applied_penalty = min(float(penalty), original_score)
        reward_score = original_score - applied_penalty

    reward_extra_info = _ensure_reward_extra_info(extra_fields)
    reward_extra_info['output_format_bad_pattern_hit'] = 1.0 if is_bad_pattern else 0.0
    reward_extra_info['output_format_bad_pattern_eligible'] = 1.0 if is_eligible else 0.0
    reward_extra_info['output_format_bad_pattern_penalized'] = 1.0 if should_penalize else 0.0
    reward_extra_info['output_format_bad_pattern_penalty'] = applied_penalty
    reward_extra_info['F6_7_think_tag_in_content_hit'] = 1.0 if f6_7_hit else 0.0
    reward_extra_info['F7_2_reasoning_collapsed_tool_calls_hit'] = 1.0 if f7_2_hit else 0.0
    reward_extra_info['F7_3_tool_call_leak_in_content_hit'] = 1.0 if f7_3_hit else 0.0
    reward_extra_info['F6_7_think_tag_in_content_count'] = float(f6_7_count)
    reward_extra_info['F7_2_reasoning_collapsed_tool_calls_count'] = float(f7_2_count)
    reward_extra_info['F7_3_tool_call_leak_in_content_count'] = float(f7_3_count)
    if original_score is not None and is_bad_pattern:
        reward_extra_info['output_format_bad_pattern_original_score'] = original_score

    bad_pattern_info.update({
        'is_bad_pattern': is_bad_pattern,
        'F6_7_think_tag_in_content': f6_7_hit,
        'F7_2_reasoning_collapsed_tool_calls': f7_2_hit,
        'F7_3_tool_call_leak_in_content': f7_3_hit,
        'F6_7_count': f6_7_count,
        'F7_2_count': f7_2_count,
        'F7_3_count': f7_3_count,
        'eligible': is_eligible,
        'penalized': should_penalize,
        'penalty': applied_penalty,
    })
    if original_score is not None:
        bad_pattern_info['original_score'] = original_score
    if should_penalize:
        bad_pattern_info['penalized_score'] = reward_score

    extra_reward_metrics = _ensure_extra_reward_metrics(reward_extra_info)
    extra_reward_metrics['output_format_bad_pattern_hit'] = 1.0 if is_bad_pattern else 0.0
    extra_reward_metrics['output_format_bad_pattern_eligible'] = 1.0 if is_eligible else 0.0
    extra_reward_metrics['output_format_bad_pattern_penalized'] = 1.0 if should_penalize else 0.0
    extra_reward_metrics['output_format_bad_pattern_penalty'] = applied_penalty
    extra_reward_metrics['F6_7_think_tag_in_content_hit'] = 1.0 if f6_7_hit else 0.0
    extra_reward_metrics['F7_2_reasoning_collapsed_tool_calls_hit'] = 1.0 if f7_2_hit else 0.0
    extra_reward_metrics['F7_3_tool_call_leak_in_content_hit'] = 1.0 if f7_3_hit else 0.0
    extra_reward_metrics['F6_7_think_tag_in_content_count'] = float(f6_7_count)
    extra_reward_metrics['F7_2_reasoning_collapsed_tool_calls_count'] = float(f7_2_count)
    extra_reward_metrics['F7_3_tool_call_leak_in_content_count'] = float(f7_3_count)

    return reward_score


def _apply_tbench_json_fix(
    reward_score: float | None,
    extra_fields: dict[str, Any],
    enabled: bool,
) -> float | None:
    info = extra_fields.get('terminus_json_parse_error')
    if not isinstance(info, dict):
        return reward_score

    is_hit = bool(info.get('is_terminus_json_parse_error'))

    try:
        original_score = float(reward_score) if reward_score is not None else None
    except (TypeError, ValueError):
        original_score = None

    if enabled and is_hit:
        reward_score = 0.0

    reward_extra_info = _ensure_reward_extra_info(extra_fields)
    if original_score is not None:
        info['original_score'] = original_score
    reward_extra_info['is_terminus_json_parse_error'] = 1.0 if is_hit else 0.0
    if original_score is not None and is_hit:
        reward_extra_info['tbench_json_fix_original_score'] = original_score

    extra_reward_metrics = _ensure_extra_reward_metrics(reward_extra_info)
    extra_reward_metrics['is_terminus_json_parse_error'] = 1.0 if is_hit else 0.0

    return reward_score


def apply_terminal_penalties(
    reward_score: float | None,
    extra_fields: dict[str, Any],
    *,
    codeswitch_penalty: float = 0.0,
    codeswitch_penalty_reward_threshold: float = 1.0,
    output_format_bad_pattern_penalty: float = 0.0,
    output_format_bad_pattern_reward_threshold: float = 1.0,
    let_me_loop_penalty: float = 0.0,
    let_me_loop_reward_threshold: float = 1.0,
    tbench_json_fix: bool = False,
) -> float | None:
    """Apply all configured terminal-task penalties.

    Each sub-penalty bails when its detection key is missing from
    ``extra_fields``. When nothing is configured (defaults: all 0/False),
    each call is effectively a no-op even if detection keys are present
    (penalty=0 → no score change; only metadata mirrors are written).
    """
    if not isinstance(extra_fields, dict):
        return reward_score

    try:
        reward_score = _apply_codeswitch_penalty(
            reward_score,
            extra_fields,
            penalty=float(codeswitch_penalty or 0),
            threshold=float(codeswitch_penalty_reward_threshold or 0),
        )
    except Exception:
        logger.exception('Error in _apply_codeswitch_penalty')

    try:
        reward_score = _apply_output_format_bad_pattern_penalty(
            reward_score,
            extra_fields,
            penalty=float(output_format_bad_pattern_penalty or 0),
            threshold=float(output_format_bad_pattern_reward_threshold or 0),
        )
    except Exception:
        logger.exception('Error in _apply_output_format_bad_pattern_penalty')

    try:
        reward_score = _apply_tbench_json_fix(
            reward_score,
            extra_fields,
            enabled=bool(tbench_json_fix),
        )
    except Exception:
        logger.exception('Error in _apply_tbench_json_fix')

    try:
        reward_score = apply_let_me_loop_penalty(
            reward_score,
            extra_fields,
            penalty=float(let_me_loop_penalty or 0),
            threshold=1.0 if let_me_loop_reward_threshold is None else float(let_me_loop_reward_threshold),
        )
    except Exception:
        logger.exception('Error in apply_let_me_loop_penalty')

    return reward_score


__all__ = ['apply_terminal_penalties']
