"""Reward-hacking detection over the HITL worker trajectory.

Flag-only: it runs the SWE task's ``detect_reward_hacking`` over the worker's chat
history and returns the flag dict, never touching the score. Both channels are
covered -- the network one (fetching the package-under-fix's own source, commit or
test) and the local one (git history that outlived truncation, an installed release,
harness files the grader does not reset) -- but the local checks need
``base_commit``, and are skipped without it.

The flags (reward_hack_mask / _rules / _attempted / _effective) are written into
``eval_details['reward_hack']`` by the caller. Keeping detection here, separate from
score handling, leaves room for an RL run to later branch on the same flags to mask
or penalize.
"""

from typing import Any

from nexus.extensions.tasks.swe.reward_hacking import detect_reward_hacking


def worker_chat_messages(trajectories: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten the worker's chat-tracker traces into one OpenAI-chat message list.

    ``trajectories`` is the agent output's trajectories dict; its ``worker.traces``
    entry is a chat-tracker result ``{'trajectories': [...]}`` where each element is
    either a list of messages (global_first mode) or a ``{'messages': [...]}`` dict
    (branch_last mode). The worker is one agent whose history may be split across
    several trajectories by context-management compaction, so concatenating them
    yields the full command sequence the detector needs. Returns ``[]`` when no
    worker traces are present (e.g. the worker never ran).
    """
    worker = trajectories.get('worker') or {}
    traces = worker.get('traces') or {}
    messages: list[dict[str, Any]] = []
    for trajectory in traces.get('trajectories') or []:
        if isinstance(trajectory, dict):
            messages.extend(trajectory.get('messages') or [])
        elif isinstance(trajectory, list):
            messages.extend(trajectory)
    return messages


def detect_worker_reward_hacking(
    trajectories: dict[str, Any],
    instance_id: str,
    repo: str | None,
    patch_text: str,
    score: float,
    *,
    base_commit: str | None = None,
    history_truncated: bool | None = None,
) -> dict[str, Any]:
    """Run answer-lookup detection over the worker trajectory (flag-only).

    Returns the detector's flag dict, or an empty dict when there is nothing to flag
    or no worker trajectory is available. The underlying detector is itself fail-open;
    a missing/oddly-shaped trajectory simply yields ``{}`` so detection never affects
    the task's score or return.

    ``base_commit`` is what the local git-history checks anchor on -- without it they
    are skipped and only the network channels are covered. ``history_truncated``
    retires the ref-walk checks when the runtime was measured to hold no ref outside
    that commit's ancestry; ``None`` leaves them on, which is the conservative reading
    for a run that did not measure.
    """
    messages = worker_chat_messages(trajectories)
    if not messages:
        return {}
    return (
        detect_reward_hacking(
            messages,
            instance_id,
            repo,
            patch_text,
            score,
            base_commit=base_commit,
            history_truncated=history_truncated,
        )
        or {}
    )
