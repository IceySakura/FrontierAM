"""Summarize the health of one hy-live episode for the reward layer."""

from __future__ import annotations

from typing import Any

from .coordination.models import AgentTask, TaskFailureKind


FORMAT_ERROR_TERMINATION = 'FORMAT_ERROR'
"""Episode termination for a policy contract violation, scored zero rather than masked."""
TRUNCATION_TERMINATION = 'TRUNCATION'
"""Episode termination for a cerebellum request past the context length, as the other agents report it."""
USER_LEFT_EARLY_TERMINATION = 'USER_LEFT_EARLY'
"""The simulated user had finished (said goodbye or used up its turns) while a permission or a question still
waited for it, so the harness answered no on its behalf. The rest of the episode is not the user's doing."""


def aggregate_episode_health(
    *,
    tasks: list[AgentTask],
    episode_termination: str | None = None,
) -> dict[str, Any]:
    """Summarize episode health from background task results and optional episode termination."""
    failure_kinds = _collect_failure_kinds(tasks)
    termination_reason = episode_termination or _termination_reason_from_tasks(tasks)
    health_status = _health_status_for(termination_reason)
    return {
        'health_status': health_status,
        'failure_kinds': failure_kinds,
        'termination_reason': termination_reason,
    }


def _collect_failure_kinds(tasks: list[AgentTask]) -> list[TaskFailureKind]:
    kinds: list[TaskFailureKind] = []
    for task in tasks:
        if task.result is None or task.result.failure_kind is None:
            continue
        if task.result.failure_kind not in kinds:
            kinds.append(task.result.failure_kind)
    return kinds


def _termination_reason_from_tasks(tasks: list[AgentTask]) -> str:
    """Map finished background results to a termination reason.

    A task past its working budget is a task failure the cerebellum was told about, so reporting it is
    scored like any other failure. Only the episode deadline, passed in as ``episode_termination``,
    ends an episode as ``TASK_TIMEOUT``.
    """
    has_infrastructure = any(task.result is not None and task.result.failure_kind == 'infrastructure' for task in tasks)
    return 'ENV_ERROR' if has_infrastructure else 'ENV_DONE'


def _health_status_for(termination_reason: str) -> str:
    if termination_reason == 'ENV_DONE':
        return 'healthy'
    if termination_reason == 'TASK_TIMEOUT':
        return 'timeout'
    if termination_reason == 'ENV_ERROR':
        return 'error'
    if termination_reason == FORMAT_ERROR_TERMINATION:
        # NOTE(jaxlliu): distinct from 'degraded' so the reward side can tell a policy
        # contract violation, which scores zero, from an environment problem, which the
        # trainer masks out.
        return 'format_error'
    if termination_reason == USER_LEFT_EARLY_TERMINATION:
        return 'user_left'
    return 'degraded'


__all__ = [
    'FORMAT_ERROR_TERMINATION',
    'TRUNCATION_TERMINATION',
    'USER_LEFT_EARLY_TERMINATION',
    'aggregate_episode_health',
]
