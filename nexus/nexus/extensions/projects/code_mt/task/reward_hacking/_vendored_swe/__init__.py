"""Reward-hacking detection for the SWE task (detector v1.2).

Trimmed from the TraX detector: this package delivers a single actionable vector,
online_answer_fetch — a rollout that fetches the package-under-fix's own
source/commit/test from the network instead of solving the task. Validated on real
rollout data (swe ~58%, 0 false positives; consumption-gated; per-fetch downstream
inference). The SWE task flags only — it never changes the reward.

Layers (a future general package can lift the first):
- signals: deterministic network-fetch signals. The repo-independent part (above the
  classify_net seam in signals.py) is repo-agnostic and liftable.
- adapters / rules: the SWE-binding layer (instance_id / repo / patch semantics) plus
  the actionable tier rule; these depend on SWE task semantics and stay here.
- detector: the entrypoint the SWE task calls (detect_reward_hacking).
"""

from .detector import detect_reward_hacking
from .monitors import run_swe_monitors
from .penalties import apply_swe_penalties


__all__ = ['detect_reward_hacking', 'run_swe_monitors', 'apply_swe_penalties']
