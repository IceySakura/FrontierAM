"""Reward-hacking / output-quality monitors and penalties for terminal tasks.

Mirrors the swe/reward_hacking layout. Two layers:

- monitors: pure-function detectors that inspect agent messages and write
  flag-only results into ``extra_fields`` (codeswitch / output_format_bad_pattern
  / terminus_json_parse_error). They never change the score.
- penalties: consumes those detection keys to mutate the score and mirror
  metrics into ``reward_extra_info`` / ``extra_reward_metrics``.

Both are gated by ``enable_output_monitoring`` in task_config (default False).
"""

from .monitors import run_terminal_monitors
from .penalties import apply_terminal_penalties


__all__ = ['run_terminal_monitors', 'apply_terminal_penalties']
