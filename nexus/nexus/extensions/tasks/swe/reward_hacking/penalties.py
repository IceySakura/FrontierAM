"""Reward penalties for SWE tasks.

Self-contained: detection happens upstream (``reward_hacking/monitors.py``
writes ``codeswitch`` / ``output_format_bad_pattern`` /
``str_replace_old_eq_new`` into ``extra_fields``; ``reward_hacking/detector.py``
writes ``reward_hack_mask`` / ``anti_hack_violations``). This module consumes
those keys, mutates score + extra_fields in-place, and settles what a detected
hack costs: a ``CODE_REWARD_HACKING`` termination reason either way, plus an
overriding reward when the dataset asked to penalise rather than mask.
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


def _apply_str_replace_old_eq_new_penalty(
    reward_score: float | None,
    extra_fields: dict[str, Any],
    penalty: float,
    threshold: float,
) -> float | None:
    detection = extra_fields.get('str_replace_old_eq_new')
    if not isinstance(detection, dict):
        return reward_score

    str_replace_count = int(detection.get('str_replace_count', 0))
    old_eq_new_count = int(detection.get('old_eq_new_count', 0))

    try:
        original_score = float(reward_score) if reward_score is not None else None
    except (TypeError, ValueError):
        original_score = None

    is_eligible = old_eq_new_count > 0 and original_score is not None and original_score >= threshold
    should_penalize = is_eligible and penalty > 0
    applied_penalty = 0.0

    if should_penalize:
        assert original_score is not None  # guaranteed by is_eligible check
        applied_penalty = min(float(penalty), original_score)
        reward_score = original_score - applied_penalty

    reward_extra_info = _ensure_reward_extra_info(extra_fields)
    reward_extra_info['str_replace_count'] = float(str_replace_count)
    reward_extra_info['str_replace_old_eq_new_count'] = float(old_eq_new_count)
    reward_extra_info['str_replace_old_eq_new_eligible'] = 1.0 if is_eligible else 0.0
    reward_extra_info['str_replace_old_eq_new_penalized'] = 1.0 if should_penalize else 0.0
    reward_extra_info['str_replace_old_eq_new_penalty'] = applied_penalty
    if original_score is not None and old_eq_new_count > 0:
        reward_extra_info['str_replace_old_eq_new_original_score'] = original_score

    detection.update({
        'eligible': is_eligible,
        'penalized': should_penalize,
        'penalty': applied_penalty,
    })
    if original_score is not None:
        detection['original_score'] = original_score
    if should_penalize:
        detection['penalized_score'] = reward_score

    extra_reward_metrics = _ensure_extra_reward_metrics(reward_extra_info)
    extra_reward_metrics['str_replace_count'] = float(str_replace_count)
    extra_reward_metrics['str_replace_old_eq_new_count'] = float(old_eq_new_count)
    extra_reward_metrics['str_replace_old_eq_new_penalized'] = 1.0 if should_penalize else 0.0
    extra_reward_metrics['str_replace_old_eq_new_penalty'] = applied_penalty

    return reward_score


def _normalize_hack_metadata(
    extra_fields: dict[str, Any],
    termination_reason: str,
    enabled: bool,
    reward_override: float | None = None,
) -> tuple[str, float | None]:
    """Settle what a detected hack costs, and record it.

    Only fires when ``enabled`` is True (off by default). Reads
    ``anti_hack_violations`` / ``reward_hack_mask`` written by the SWE
    reward-hacking detector and writes ``is_hack`` / ``mask_reason`` mirrors
    into ``extra_fields`` so downstream masking has consistent inputs.

    A detected hack is always relabelled ``CODE_REWARD_HACKING``, so the termination metrics
    report it whichever disposition is configured. The label does not decide the disposition:
    masking follows from listing it in hytuner's ``trainer.additional_terminate_reason``, which
    a run that penalises instead has to leave out, or the penalised sample is masked as well and
    its reward never reaches a gradient.

    Returns:
        ``(termination_reason, reward_to_apply)``, the second None when the score stays as earned.
    """
    if not enabled:
        return termination_reason, None

    is_hack = (
        bool(extra_fields.get('anti_hack_violations'))
        or bool(extra_fields.get('reward_hack_mask'))
        or termination_reason in {'ANTI_HACK_SOURCE_DOWNLOAD', 'CODE_REWARD_HACKING'}
    )
    if not is_hack:
        return termination_reason, None

    extra_fields['is_hack'] = True
    extra_fields['mask_reason'] = 'hack'
    reward_extra_info = _ensure_reward_extra_info(extra_fields)
    reward_extra_info['is_hack'] = 1.0
    reward_extra_info['mask_reason'] = 'hack'
    if reward_override is None:
        return 'CODE_REWARD_HACKING', None

    # acc answers whether the instance was honestly solved, which a hack never is; the reward is
    # what carries the penalty. Masking forced this to 0 and an override has to read the same way.
    reward_extra_info['score'] = 0.0
    reward_extra_info['acc'] = 0.0
    extra_reward_metrics = _ensure_extra_reward_metrics(reward_extra_info)
    extra_reward_metrics['hack_penalized'] = 1.0
    extra_reward_metrics['hack_reward_override'] = float(reward_override)
    return 'CODE_REWARD_HACKING', float(reward_override)


def apply_swe_penalties(
    reward_score: float | None,
    extra_fields: dict[str, Any],
    termination_reason: str | None,
    *,
    codeswitch_penalty: float = 0.0,
    codeswitch_penalty_reward_threshold: float = 1.0,
    output_format_bad_pattern_penalty: float = 0.0,
    output_format_bad_pattern_reward_threshold: float = 1.0,
    str_replace_old_eq_new_penalty: float = 0.0,
    str_replace_old_eq_new_reward_threshold: float = 1.0,
    let_me_loop_penalty: float = 0.0,
    let_me_loop_reward_threshold: float = 1.0,
    enable_anti_hack: bool = False,
    hack_reward_override: float | None = None,
) -> tuple[float | None, str | None]:
    """Apply all configured SWE-task penalties.

    Returns ``(new_score, new_termination_reason)``. Each sub-penalty bails
    when its detection key is missing; with all defaults (0 / False) the
    function is a no-op aside from writing mirror metadata when detection
    keys already exist.
    """
    if not isinstance(extra_fields, dict):
        return reward_score, termination_reason

    new_term = termination_reason or ''
    hack_reward: float | None = None
    try:
        new_term, hack_reward = _normalize_hack_metadata(
            extra_fields,
            new_term,
            enabled=bool(enable_anti_hack),
            reward_override=hack_reward_override,
        )
    except Exception:
        logger.exception('Error in _normalize_hack_metadata')

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
        reward_score = _apply_str_replace_old_eq_new_penalty(
            reward_score,
            extra_fields,
            penalty=float(str_replace_old_eq_new_penalty or 0),
            threshold=float(str_replace_old_eq_new_reward_threshold or 0),
        )
    except Exception:
        logger.exception('Error in _apply_str_replace_old_eq_new_penalty')

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
        reward_score = apply_let_me_loop_penalty(
            reward_score,
            extra_fields,
            penalty=float(let_me_loop_penalty or 0),
            threshold=1.0 if let_me_loop_reward_threshold is None else float(let_me_loop_reward_threshold),
        )
    except Exception:
        logger.exception('Error in apply_let_me_loop_penalty')

    # Last: a hack replaces the reward, so the quality penalties above are void.
    if hack_reward is not None:
        reward_score = hack_reward

    return reward_score, (new_term or None)


__all__ = ['apply_swe_penalties']
