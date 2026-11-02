"""Config resolution for the code_mt reward-hacking / baseline plug-in.

The plug-in is scaffold-agnostic (works for BOTH the tool_call and text_parse
loops) and is driven entirely by fields on CodeMTTaskInput, which come from the
`task_config` block of the parquet `nexus` column — task.toml is never touched.

Defaults are deliberately no-op:
  * monitors FLAG by default (enable_output_monitoring / enable_reward_hacking_detection True),
  * every penalty is 0.0 (subtracts nothing),
  * anti-hack termination normalization is OFF (enable_anti_hack False),
  * reward floor is 0.0.

So with an unmodified dataset the plug-in only writes mirror metadata into
extra_fields and leaves score + termination_reason exactly as they were.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# Canonical monitor-family names — the allowed values of the parquet
# task_config.monitors whitelist. Each family maps to one detector plus an
# optional penalty. 'str_replace_noop' additionally reports the tool_call vs
# function_call channel breakdown. Keep in sync with monitors.py / report.py.
MONITOR_NAMES = (
    'codeswitch',
    'output_format_bad_pattern',
    'str_replace_noop',
    'let_me_loop',
    'anti_hack',
    'baseline',
)


@dataclass
class RewardHackingConfig:
    """Resolved, plain-data view of the reward-hacking config for one run."""

    # --- multi-select: which monitor families to run + penalize (empty = ALL) ---
    # Parquet task_config.monitors is a whitelist of MONITOR_NAMES. A family not
    # selected is neither flagged nor penalized. Empty keeps every family ON
    # (backward compatible with the previous enable_output_monitoring behaviour).
    monitors: list[str] = field(default_factory=list)

    # --- monitoring switches (flag-only; safe to leave ON) ---
    enable_output_monitoring: bool = True
    enable_reward_hacking_detection: bool = True
    # --- anti-hack (controls CODE_REWARD_HACKING termination normalization) ---
    enable_anti_hack: bool = False
    # Validation split: keep detecting and keep every hack metric, but do not
    # apply the consequence (no CODE_REWARD_HACKING flip, no CM-1 score zeroing).
    # Mirrors SWE, which gates only the consequence and never the detection:
    # `tasks/swe/task.py` passes `hack_reward_override=None if is_validation`.
    is_validation: bool = False

    # --- penalties (ordered): each subtracts from the reward, never below floor ---
    codeswitch_penalty: float = 0.0
    codeswitch_penalty_reward_threshold: float = 1.0
    output_format_bad_pattern_penalty: float = 0.0
    output_format_bad_pattern_reward_threshold: float = 1.0
    str_replace_old_eq_new_penalty: float = 0.0
    str_replace_old_eq_new_reward_threshold: float = 1.0
    let_me_loop_penalty: float = 0.0
    let_me_loop_reward_threshold: float = 1.0
    # Baseline penalties are DESIGN-ONLY for now (flag-only monitors); kept here
    # so the wiring exists but stays empty => no-op.
    baseline_penalties: dict[str, float] = field(default_factory=dict)
    baseline_penalty_reward_threshold: float = 1.0

    # Hard floor the post-penalty reward is clipped to (task.toml lower bound).
    reward_lower_bound: float = 0.0

    # score_kind of the run — used only to WARN when penalties meet a binary
    # score (datasets should align to partial so penalties are observable).
    score_kind: str = 'binary'

    def monitor_on(self, name: str) -> bool:
        """Whether monitor family ``name`` is selected (empty whitelist = ALL)."""
        return (not self.monitors) or (name in self.monitors)

    def any_penalty_enabled(self) -> bool:
        if self.monitor_on('codeswitch') and self.codeswitch_penalty > 0:
            return True
        if self.monitor_on('output_format_bad_pattern') and self.output_format_bad_pattern_penalty > 0:
            return True
        if self.monitor_on('str_replace_noop') and self.str_replace_old_eq_new_penalty > 0:
            return True
        if self.monitor_on('let_me_loop') and self.let_me_loop_penalty > 0:
            return True
        if self.monitor_on('baseline'):
            return any(v > 0 for v in self.baseline_penalties.values())
        return False


def _get(obj: Any, name: str, default: Any) -> Any:
    try:
        val = getattr(obj, name, default)
    except Exception:
        return default
    return default if val is None else val


def resolve_config(task_input: Any) -> RewardHackingConfig:
    """Build a RewardHackingConfig from a CodeMTTaskInput (fail-open to defaults).

    Reads via getattr so an older task_input that lacks these fields simply gets
    the no-op defaults.
    """
    baseline = _get(task_input, 'baseline_penalties', {})
    if not isinstance(baseline, dict):
        baseline = {}
    monitors = _get(task_input, 'monitors', [])
    if not isinstance(monitors, (list, tuple)):
        monitors = []
    return RewardHackingConfig(
        monitors=[str(m) for m in monitors],
        enable_output_monitoring=bool(_get(task_input, 'enable_output_monitoring', True)),
        enable_reward_hacking_detection=bool(_get(task_input, 'enable_reward_hacking_detection', True)),
        enable_anti_hack=bool(_get(task_input, 'enable_anti_hack', False)),
        is_validation=bool(_get(task_input, 'is_validation', False)),
        codeswitch_penalty=float(_get(task_input, 'codeswitch_penalty', 0.0)),
        codeswitch_penalty_reward_threshold=float(_get(task_input, 'codeswitch_penalty_reward_threshold', 1.0)),
        output_format_bad_pattern_penalty=float(_get(task_input, 'output_format_bad_pattern_penalty', 0.0)),
        output_format_bad_pattern_reward_threshold=float(
            _get(task_input, 'output_format_bad_pattern_reward_threshold', 1.0)
        ),
        str_replace_old_eq_new_penalty=float(_get(task_input, 'str_replace_old_eq_new_penalty', 0.0)),
        str_replace_old_eq_new_reward_threshold=float(_get(task_input, 'str_replace_old_eq_new_reward_threshold', 1.0)),
        let_me_loop_penalty=float(_get(task_input, 'let_me_loop_penalty', 0.0)),
        let_me_loop_reward_threshold=float(_get(task_input, 'let_me_loop_reward_threshold', 1.0)),
        baseline_penalties={str(k): float(v) for k, v in baseline.items()},
        baseline_penalty_reward_threshold=float(_get(task_input, 'baseline_penalty_reward_threshold', 1.0)),
        reward_lower_bound=float(_get(task_input, 'reward_lower_bound', 0.0)),
        score_kind=str(_get(task_input, 'score_kind', 'binary')),
    )


__all__ = ['MONITOR_NAMES', 'RewardHackingConfig', 'resolve_config']
