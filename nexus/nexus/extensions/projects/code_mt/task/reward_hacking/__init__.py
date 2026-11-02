"""Scoring plug-in for code_mt: reward-hacking + baseline monitoring.

Scaffold-agnostic (works for BOTH the tool_call and text_parse loops). Reuses
the SWE reward-hacking detectors/penalties (they take OpenAI-chat messages) and
adds code_mt baseline (flag-only) monitors.

Contract (see task.py integration, after the score/termination are computed):
    cfg   = resolve_reward_hacking_config(task_input)
    extra = run_code_mt_monitors(messages, config=cfg, ...)
    score, termination_reason = apply_code_mt_penalties(score, extra, term, config=cfg)

Defaults are no-op: monitors FLAG only, all penalties 0, anti-hack OFF, floor
0.0 — score and termination are untouched until enabled via the parquet overlay.
"""

from .config import RewardHackingConfig
from .config import resolve_config as resolve_reward_hacking_config
from .monitors import run_code_mt_monitors
from .penalties import apply_code_mt_penalties
from .report import build_monitor_report


__all__ = [
    'RewardHackingConfig',
    'resolve_reward_hacking_config',
    'run_code_mt_monitors',
    'apply_code_mt_penalties',
    'build_monitor_report',
]
