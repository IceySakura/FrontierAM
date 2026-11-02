"""Reward-hacking detection entrypoint for the SWE task.

Orchestrates the package: parse the in-flight trajectory, run the deterministic
rules, and reduce the actionable detections to the four flags the SWE task writes
into SwebenchTaskOutput.extra_fields. The SWE task only flags; it never changes the
reward. The training side decides what to do with the flags (the delivery is MASK:
exclude flagged runs from training, not a penalty).
"""

from __future__ import annotations

import logging

from typing import Any

from .adapters import parse_nexus_messages
from .rules import evaluate


logger = logging.getLogger(__name__)


def detect_reward_hacking(
    messages: list[dict[str, Any]],
    instance_id: str | None,
    repo: str | None,
    patch_text: str | None,
    score: float,
    base_commit: str | None = None,
    history_truncated: bool | None = None,
) -> dict[str, Any]:
    """Flag answer-lookup reward hacking from a SWE trajectory.

    Covers both the network channels (online_answer_fetch and friends) and the local
    ones (git history that outlived truncation, an installed release of the package,
    harness files the grader does not reset). Reduces the actionable detections to four
    flags using the not_blocked level (mask a confirmed or unconfirmed lookup, but not
    one that was clearly blocked).

    Args:
        messages: the agent trajectory as OpenAI-chat messages (list of dicts).
        instance_id: SWE-bench instance id (used to derive repo when repo is absent).
        repo: the package-under-fix repo (e.g. 'django/django'), for target matching.
        patch_text: the submitted patch, for patch-based signals.
        score: the task score, passed into the detection context; never modified here.
        base_commit: the commit the work tree starts at. Without it the local git-history
            checks cannot anchor and are skipped; the other channels are unaffected.
        history_truncated: whether the runtime was measured to hold no ref outside
            base_commit's ancestry. True retires the ref-walk checks, which exist only to
            catch a truncation that did not take. None leaves them on, as before.

    Returns:
        A dict with reward_hack_mask / reward_hack_rules / reward_hack_attempted /
        reward_hack_effective, or an empty dict if detection fails (fail-open).
    """
    try:
        task_output = {
            'instance_id': instance_id,
            'repo': repo,
            'base_commit': base_commit,
            'history_truncated': history_truncated,
            'patch': patch_text,
            'score': score,
        }
        report = evaluate(parse_nexus_messages(messages, task_output))
        actionable = report.actionable
        not_blocked = [d for d in actionable if d.effective != 'no']
        effective = [d for d in actionable if d.effective == 'yes']
        return {
            'reward_hack_mask': bool(not_blocked),
            'reward_hack_rules': sorted({d.rule_id for d in not_blocked}),
            'reward_hack_attempted': bool(actionable),
            'reward_hack_effective': bool(effective),
        }
    except Exception as exc:
        logger.warning('reward-hacking detection failed: %s', exc)
        return {}
