"""Log-only ``monitor`` report builder for the code_mt reward-hacking plug-in.

This is PURE MIRROR: it re-reads the detection + penalty annotations that
monitors.py / penalties.py already wrote into ``extra_fields`` and lays them out
as one ``monitor`` dict on CodeMTTaskOutput. It performs NO scoring logic of its
own — the deduction was already computed in task.py; this only surfaces, per
selected family, whether it triggered and how much it deducted, so the whole
picture is visible in the DB from task_config alone.

Per-family shape:
  * ``on``        — whether the family is selected by the parquet whitelist.
  * ``triggered`` — whether the detector flagged it.
  * penalty fields (``penalty_config`` / ``penalty_applied``) are included ONLY
    when a penalty is configured (>0) for that family. A monitor with no penalty
    configured is flag-only: it reports ``triggered`` and nothing about score.
"""

from __future__ import annotations

from typing import Any

from .config import RewardHackingConfig


def _num(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _dict(value: Any) -> dict[str, Any]:
    """Extract a dict from a possibly-None/non-dict value."""
    return value if isinstance(value, dict) else {}


def _codeswitch_item(config: RewardHackingConfig, extra: dict[str, Any]) -> dict[str, Any]:
    info = _dict(extra.get('codeswitch'))
    item: dict[str, Any] = {
        'on': config.monitor_on('codeswitch'),
        'triggered': bool(info.get('is_codeswitch')),
    }
    if config.monitor_on('codeswitch') and config.codeswitch_penalty > 0:
        item['penalty_config'] = config.codeswitch_penalty
        item['penalty_applied'] = _num(info.get('penalty')) or 0.0
    return item


def _bad_pattern_item(config: RewardHackingConfig, extra: dict[str, Any]) -> dict[str, Any]:
    info = _dict(extra.get('output_format_bad_pattern'))
    item: dict[str, Any] = {
        'on': config.monitor_on('output_format_bad_pattern'),
        'triggered': bool(info.get('is_bad_pattern')),
        'F6_7_count': int(_num(info.get('F6_7_count')) or 0),
        'F7_2_count': int(_num(info.get('F7_2_count')) or 0),
        'F7_3_count': int(_num(info.get('F7_3_count')) or 0),
    }
    if config.monitor_on('output_format_bad_pattern') and config.output_format_bad_pattern_penalty > 0:
        item['penalty_config'] = config.output_format_bad_pattern_penalty
        item['penalty_applied'] = _num(info.get('penalty')) or 0.0
    return item


def _str_replace_item(config: RewardHackingConfig, extra: dict[str, Any]) -> dict[str, Any]:
    info = _dict(extra.get('str_replace_old_eq_new'))
    tool_call = _dict(info.get('tool_call'))
    function_call = _dict(info.get('function_call'))
    old_eq_new = int(_num(info.get('old_eq_new_count')) or 0)
    item: dict[str, Any] = {
        'on': config.monitor_on('str_replace_noop'),
        'triggered': old_eq_new > 0,
        'str_replace_count': int(_num(info.get('str_replace_count')) or 0),
        'old_eq_new_count': old_eq_new,
        'tool_call': {
            'count': int(_num(tool_call.get('count')) or 0),
            'old_eq_new_count': int(_num(tool_call.get('old_eq_new_count')) or 0),
        },
        'function_call': {
            'count': int(_num(function_call.get('count')) or 0),
            'old_eq_new_count': int(_num(function_call.get('old_eq_new_count')) or 0),
        },
    }
    if config.monitor_on('str_replace_noop') and config.str_replace_old_eq_new_penalty > 0:
        item['penalty_config'] = config.str_replace_old_eq_new_penalty
        item['penalty_applied'] = _num(info.get('penalty')) or 0.0
    return item


def _let_me_loop_item(config: RewardHackingConfig, extra: dict[str, Any]) -> dict[str, Any]:
    info = _dict(extra.get('let_me_loop'))
    item: dict[str, Any] = {
        'on': config.monitor_on('let_me_loop'),
        'triggered': bool(info.get('is_let_me_loop')),
        'hit_fields': int(_num(info.get('hit_fields')) or 0),
        'max_window_count': int(_num(info.get('max_window_count')) or 0),
        'marker_count': int(_num(info.get('marker_count')) or 0),
    }
    if config.monitor_on('let_me_loop') and config.let_me_loop_penalty > 0:
        item['penalty_config'] = config.let_me_loop_penalty
        item['penalty_applied'] = _num(info.get('penalty')) or 0.0
    return item


def _anti_hack_item(
    config: RewardHackingConfig,
    extra: dict[str, Any],
    termination_reason: str | None,
) -> dict[str, Any]:
    triggered = (
        bool(extra.get('is_hack'))
        or bool(extra.get('anti_hack_violations'))
        or bool(extra.get('reward_hack_mask'))
        or termination_reason == 'CODE_REWARD_HACKING'
    )
    return {
        'on': config.monitor_on('anti_hack'),
        'triggered': triggered,
        # anti_hack does not subtract a fixed amount; it normalizes termination
        # (score already zeroed upstream). Report the resulting termination_reason
        # only when the anti-hack normalization is actually enabled.
        'enabled': config.enable_anti_hack and config.monitor_on('anti_hack'),
        'termination_reason': termination_reason,
    }


def _baseline_item(config: RewardHackingConfig, extra: dict[str, Any]) -> dict[str, Any]:
    baseline = _dict(extra.get('baseline'))
    subitems: dict[str, Any] = {}
    for name in ('empty_final_answer', 'noop_edit', 'trailing_tool_failure', 'submit_loop'):
        info = _dict(baseline.get(name))
        sub: dict[str, Any] = {'triggered': bool(info.get(f'is_{name}'))}
        penalty = float(config.baseline_penalties.get(name, 0.0) or 0.0)
        if config.monitor_on('baseline') and penalty > 0:
            sub['penalty_config'] = penalty
            sub['penalty_applied'] = _num(info.get('penalty')) or 0.0
        subitems[name] = sub
    return {
        'on': config.monitor_on('baseline'),
        'items': subitems,
    }


def build_monitor_report(
    config: RewardHackingConfig,
    extra_fields: dict[str, Any] | None,
    *,
    base_score: float | None,
    final_score: float | None,
    termination_reason: str | None = None,
) -> dict[str, Any]:
    """Assemble the log-only ``monitor`` dict from post-penalty ``extra_fields``.

    ``base_score`` is the reward BEFORE penalties, ``final_score`` AFTER (and
    after score_kind resolution). Both are carried verbatim so the total
    deduction is inspectable without re-deriving it.
    """
    extra = extra_fields if isinstance(extra_fields, dict) else {}

    base = _num(base_score)
    final = _num(final_score)
    total_penalty = None
    if base is not None and final is not None:
        total_penalty = round(base - final, 6)

    return {
        'monitors_selected': list(config.monitors) if config.monitors else 'ALL',
        'score_kind': config.score_kind,
        'base_score': base,
        'final_score': final,
        'total_penalty': total_penalty,
        'reward_lower_bound': config.reward_lower_bound,
        'termination_reason': termination_reason,
        'items': {
            'codeswitch': _codeswitch_item(config, extra),
            'output_format_bad_pattern': _bad_pattern_item(config, extra),
            'str_replace_noop': _str_replace_item(config, extra),
            'let_me_loop': _let_me_loop_item(config, extra),
            'anti_hack': _anti_hack_item(config, extra, termination_reason),
            'baseline': _baseline_item(config, extra),
        },
    }


__all__ = ['build_monitor_report']
