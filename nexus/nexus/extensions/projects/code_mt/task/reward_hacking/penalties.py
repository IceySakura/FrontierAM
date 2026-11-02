"""Penalty application for the code_mt reward-hacking / baseline plug-in.

Ordering (each step operates on the running score):
  1. SWE penalties (codeswitch -> str_replace no-op -> output_format). The SWE
     anti-hack score-zeroing normalization is deliberately DISABLED here (we pass
     enable_anti_hack=False into apply_swe_penalties) so the task never zeros the
     reward itself — see step 1a.
  1a. OBSERVE-ALWAYS hack normalization (code_mt-local, decoupled from the
     enable_anti_hack switch): always writes the flat ``is_hack`` 0/1 metric and,
     when a hack is detected, returns termination_reason='CODE_REWARD_HACKING'.
     It does NOT touch the score. Whether a flagged trajectory is actually masked
     out (score zeroed + dropped from loss) is decided ENTIRELY by the trainer's
     ``additional_terminate_reason`` on the hytuner side: list CODE_REWARD_HACKING
     there to mask, omit it to let the trajectory train normally alongside
     ENV_DONE. This makes the ``enable_anti_hack`` switch irrelevant to metrics.
  1b. Mirror baseline (flag-only) detections into extra_reward_metrics so every
     baseline family is visible as critic/extra_reward_metrics/baseline_*/mean.
  1c. Mirror the let_me repetition monitor as
     critic/extra_reward_metrics/let_me_loop_*/mean, and deduct its penalty when one
     is configured (0.0 by default, which leaves the step flag-only).
  2. Baseline penalties (ordered) — active when config.baseline_penalties is set.
  3. Clip the post-penalty score to config.reward_lower_bound, but ONLY for
     samples whose reward was actually reduced by a penalty — an unpenalized
     (e.g. failed, score=0) sample is never lifted up to the floor.

Warns (does not error) when any penalty is enabled while score_kind == 'binary',
since a binary reward makes fractional penalties invisible (the dataset should
be aligned to `partial` for penalties to matter — binary capability is kept).
"""

from __future__ import annotations

import logging

from typing import Any

from nexus.extensions.projects.code_mt.task.reward_hacking._vendored_swe import apply_swe_penalties
from nexus.utils.reasoning_repetition import apply_let_me_loop_penalty

from .config import RewardHackingConfig


logger = logging.getLogger(__name__)

# Canonical hack termination reason (kept as a literal so this module has no
# hard dependency on the hytuner TerminationReason enum — the string is the
# contract both sides agree on).
CODE_REWARD_HACKING = 'CODE_REWARD_HACKING'


def _ensure_reward_extra_info(extra_fields: dict[str, Any]) -> dict[str, Any]:
    rei = extra_fields.get('reward_extra_info')
    if not isinstance(rei, dict):
        rei = {}
        extra_fields['reward_extra_info'] = rei
    return rei


def _ensure_extra_reward_metrics(reward_extra_info: dict[str, Any]) -> dict[str, Any]:
    erm = reward_extra_info.get('extra_reward_metrics')
    if not isinstance(erm, dict):
        erm = {}
        reward_extra_info['extra_reward_metrics'] = erm
    return erm


def _normalize_hack_observe_always(
    extra_fields: dict[str, Any],
    termination_reason: str | None,
    *,
    suppress_flip: bool = False,
) -> str | None:
    """Always write the flat ``is_hack`` metric; flip termination on a hit.

    Decoupled from ``enable_anti_hack``: detection already ran in monitors.py
    (gated by enable_reward_hacking_detection, default ON) and wrote
    ``reward_hack_mask`` / ``anti_hack_violations`` into extra_fields. Here we
    only MIRROR that into the flat columns hytuner aggregates, and set the
    CODE_REWARD_HACKING termination so ``traj/*/hack/ratio`` reflects the true
    detection rate. The SCORE is never touched — masking is the trainer's job
    (additional_terminate_reason).

    ``suppress_flip`` (validation runs) keeps detection and every metric intact
    but leaves ``termination_reason`` alone, mirroring SWE's rule that a
    validation split records hack rate without paying the consequence
    (``tasks/swe/task.py``: ``hack_reward_override=None if is_validation``).
    Without it a single detection on an eval sample flips termination to
    CODE_REWARD_HACKING and — for any run listing that reason in
    ``additional_terminate_reason`` — silently drops the sample, corrupting the
    eval denominator.
    """
    term = termination_reason
    is_hack = (
        bool(extra_fields.get('anti_hack_violations'))
        or bool(extra_fields.get('reward_hack_mask'))
        or str(term) in {'ANTI_HACK_SOURCE_DOWNLOAD', CODE_REWARD_HACKING}
    )

    # Flat metric — written for EVERY sample (0.0 or 1.0) so the ratio has the
    # correct denominator; read by compute_trajectory_metrics -> traj/*/hack.
    reward_extra_info = _ensure_reward_extra_info(extra_fields)
    reward_extra_info['is_hack'] = 1.0 if is_hack else 0.0
    erm = _ensure_extra_reward_metrics(reward_extra_info)
    erm['hack_detected'] = 1.0 if is_hack else 0.0

    if not is_hack:
        return term

    # Validation: observed and counted above, but never penalised.
    if suppress_flip:
        return term

    # Mark the sample as a hack so the trainer's maskout (if it lists
    # CODE_REWARD_HACKING) can act on it; top-level mirrors for metric_utils.
    extra_fields['is_hack'] = True
    extra_fields['mask_reason'] = 'hack'
    reward_extra_info['mask_reason'] = 'hack'
    return CODE_REWARD_HACKING


def _mirror_baseline_metrics(extra_fields: dict[str, Any], config: RewardHackingConfig) -> None:
    """Expose baseline (flag-only) detections as extra_reward_metrics so each is
    visible as critic/extra_reward_metrics/baseline_<name>/mean. No score impact."""
    if not config.monitor_on('baseline'):
        return
    baseline = extra_fields.get('baseline')
    if not isinstance(baseline, dict):
        return
    reward_extra_info = _ensure_reward_extra_info(extra_fields)
    erm = _ensure_extra_reward_metrics(reward_extra_info)
    for name in ('empty_final_answer', 'noop_edit', 'trailing_tool_failure', 'submit_loop'):
        info = baseline.get(name)
        hit = bool(info.get(f'is_{name}')) if isinstance(info, dict) else False
        erm[f'baseline_{name}'] = 1.0 if hit else 0.0


def _apply_baseline_penalties(
    reward_score: float | None,
    extra_fields: dict[str, Any],
    config: RewardHackingConfig,
) -> float | None:
    """Ordered baseline penalties. No-op while baseline_penalties is empty."""
    if reward_score is None or not config.baseline_penalties:
        return reward_score
    if not config.monitor_on('baseline'):
        return reward_score
    baseline = extra_fields.get('baseline')
    if not isinstance(baseline, dict):
        return reward_score

    threshold = config.baseline_penalty_reward_threshold
    erm = _ensure_extra_reward_metrics(_ensure_reward_extra_info(extra_fields))
    for name in sorted(config.baseline_penalties):
        penalty = float(config.baseline_penalties[name] or 0)
        if penalty <= 0:
            continue
        info = baseline.get(name)
        hit = bool(info.get(f'is_{name}')) if isinstance(info, dict) else False
        eligible = hit and reward_score >= threshold
        # Visibility: whether this baseline family actually deducted reward.
        erm[f'baseline_{name}_penalized'] = 1.0 if eligible else 0.0
        if not eligible:
            continue
        applied = min(penalty, reward_score)
        reward_score = reward_score - applied
        if isinstance(info, dict):
            info['penalized'] = True
            info['penalty'] = applied
    return reward_score


def apply_code_mt_penalties(
    reward_score: float | None,
    extra_fields: dict[str, Any],
    termination_reason: str | None,
    *,
    config: RewardHackingConfig,
) -> tuple[float | None, str | None]:
    """Apply all configured penalties + anti-hack termination. Returns (score, term)."""
    if not isinstance(extra_fields, dict):
        return reward_score, termination_reason

    score = reward_score
    term = termination_reason
    score_before = None
    try:
        score_before = float(score) if score is not None else None
    except (TypeError, ValueError):
        score_before = None

    # A family that is not selected by the parquet whitelist must not penalize:
    # zero its penalty here so only mirror metadata (no deduction) is written for it.
    codeswitch_penalty = config.codeswitch_penalty if config.monitor_on('codeswitch') else 0.0
    output_format_bad_pattern_penalty = (
        config.output_format_bad_pattern_penalty if config.monitor_on('output_format_bad_pattern') else 0.0
    )
    str_replace_old_eq_new_penalty = (
        config.str_replace_old_eq_new_penalty if config.monitor_on('str_replace_noop') else 0.0
    )

    # 1. SWE penalties ONLY (codeswitch / str_replace / output_format). We force
    #    enable_anti_hack=False so apply_swe_penalties never zeros the score for a
    #    hack — that decision is decoupled and handled in step 1a + the trainer.
    try:
        score, term = apply_swe_penalties(
            score,
            extra_fields,
            term,
            codeswitch_penalty=codeswitch_penalty,
            codeswitch_penalty_reward_threshold=config.codeswitch_penalty_reward_threshold,
            output_format_bad_pattern_penalty=output_format_bad_pattern_penalty,
            output_format_bad_pattern_reward_threshold=config.output_format_bad_pattern_reward_threshold,
            str_replace_old_eq_new_penalty=str_replace_old_eq_new_penalty,
            str_replace_old_eq_new_reward_threshold=config.str_replace_old_eq_new_reward_threshold,
            enable_anti_hack=False,
        )
    except Exception:
        logger.exception('apply_swe_penalties failed; score/term left as-is')

    # 1a. Observe-always hack normalization (independent of enable_anti_hack):
    #     always writes is_hack 0/1, flips termination to CODE_REWARD_HACKING on a
    #     hit, never changes the score.
    if config.monitor_on('anti_hack'):
        try:
            term = _normalize_hack_observe_always(extra_fields, term, suppress_flip=config.is_validation)
        except Exception:
            logger.exception('_normalize_hack_observe_always failed')

    # 1b. Mirror baseline flags into extra_reward_metrics (visibility only).
    try:
        _mirror_baseline_metrics(extra_fields, config)
    except Exception:
        logger.exception('_mirror_baseline_metrics failed')

    # 1c. let_me repetition: mirror its flags and deduct when the family is selected
    #     and given a coefficient. A family outside the whitelist must not penalize.
    try:
        score = apply_let_me_loop_penalty(
            score,
            extra_fields,
            penalty=config.let_me_loop_penalty if config.monitor_on('let_me_loop') else 0.0,
            threshold=config.let_me_loop_reward_threshold,
        )
    except Exception:
        logger.exception('apply_let_me_loop_penalty failed')

    # 2. Baseline penalties (active when baseline_penalties is set).
    try:
        score = _apply_baseline_penalties(score, extra_fields, config)
    except Exception:
        logger.exception('_apply_baseline_penalties failed')

    # 3. Hard floor: penalties may subtract the reward down to — but not below —
    #    the floor. Apply ONLY when a penalty actually reduced the score, so a
    #    genuinely failed (unpenalized) sample is never lifted up to the floor.
    if score is not None and score_before is not None and score < score_before:
        floor = max(0.0, float(config.reward_lower_bound))
        if score < floor:
            score = floor
        # BUGFIX: the floor must NEVER lift a penalized sample above its original raw
        # score. If raw < floor (e.g. RepoQA continuous 0.39 with floor 0.8), flooring
        # the penalized 0.37 up to 0.8 would INFLATE the reward. Cap at raw so:
        #   raw >= floor  -> penalty applies, clipped down to floor (e.g. 1.0->0.98);
        #   raw <  floor  -> penalty effectively nullified (score == raw, no inflation).
        # anti_hack's hard-zero is exempt (it zeroes in task.py BEFORE penalties; a 0
        # score can't be reduced, so this reduced-branch is never entered for it).
        if score > score_before:
            score = score_before

    # NOTE (2026-08-01): score_kind='binary' + penalties is now the INTENDED
    # design. The 0/1 verdict is applied PRE-PENALTY (scoring.compute_score), then
    # these penalties deduct from that 0/1 base and task.py passes the result
    # through unchanged (no post-penalty re-snap). So a penalized 1.0 correctly
    # becomes a continuous 0.98 — penalties are NOT invisible. The old warning
    # (assuming a post-penalty snap erased them) is therefore removed.

    return score, term


__all__ = ['apply_code_mt_penalties']
