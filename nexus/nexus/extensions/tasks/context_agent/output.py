"""Helpers for consuming Context Agent outputs from task wrappers."""

from __future__ import annotations

from typing import Any

from nexus.core import WithMainTrajectory
from nexus.extensions.agents.base.schema import TerminationReason


_TOKEN_LIMIT_EXIT_STATUSES = frozenset({
    TerminationReason.TRUNCATION.value,
    TerminationReason.OUTPUT_LENGTH_EXCEEDED.value,
})
_MAX_STEP_EXIT_STATUSES = frozenset({
    TerminationReason.MAX_ITERATIONS.value,
    'MAX_ITERATIONS',
})
_MISSING = object()


def extract_final_answer(agent_output: Any) -> str | None:
    """Prefer the explicit output field, with trajectory parsing for compatibility."""
    final_answer = getattr(agent_output, 'final_answer', None)
    if final_answer is not None:
        return str(final_answer)

    if isinstance(agent_output, WithMainTrajectory):
        for message in reversed(agent_output.main_trajectory):
            tool_calls = message.get('tool_calls')
            has_tool_calls = isinstance(tool_calls, list) and bool(tool_calls)
            if message.get('role') == 'assistant' and message.get('content') and not has_tool_calls:
                return str(message['content'])
    return None


def agent_tool_error_extra_fields(agent_output: Any) -> dict[str, Any]:
    """Report observed tool errors without applying a reward policy."""
    count = (
        agent_output.get('tool_name_resolution_error_count', 0)
        if isinstance(agent_output, dict)
        else getattr(agent_output, 'tool_name_resolution_error_count', 0)
    )
    count = max(0, int(count or 0))
    return {
        'reward_extra_info': {
            'tool_name_resolution_error_count': count,
            'tool_name_resolution_zeroed': 0.0,
            'tool_name_resolution_original_score': 0.0,
        },
    }


def apply_tool_name_resolution_score(
    score: float,
    agent_output: Any,
    *,
    enabled: bool,
    eligible: bool = True,
) -> tuple[float, dict[str, Any]]:
    """Apply a task-owned score rule while retaining unmodified Judge evidence."""
    extra = agent_tool_error_extra_fields(agent_output)
    zeroed = enabled and eligible and score >= 0 and extra['reward_extra_info']['tool_name_resolution_error_count'] > 0
    extra['reward_extra_info'].update(
        tool_name_resolution_zeroed=float(zeroed),
        tool_name_resolution_original_score=float(score),
    )
    return (0.0 if zeroed else score), extra


def agent_exit_status_value(agent_output: Any) -> str:
    """Return an Agent exit status as its serialized string value."""
    status = getattr(agent_output, 'exit_status', None)
    if status is None and isinstance(agent_output, dict):
        status = agent_output.get('exit_status')
    value = getattr(status, 'value', status)
    return str(value) if value else ''


def agent_termination_reason(agent_output: Any) -> str:
    """Normalize Agent exits consistently across general and native task outputs."""
    exit_status = agent_exit_status_value(agent_output)
    return {
        '': 'ENV_DONE',
        'SUCCESS': 'ENV_DONE',
        'ENV_DONE': 'ENV_DONE',
        'MAX_ITERATIONS': 'MAX_STEPS',
        'MAX_STEPS': 'MAX_STEPS',
        'TRUNCATION': 'TRUNCATION',
        'OUTPUT_LENGTH_EXCEEDED': 'TRUNCATION',
        'FORMAT_ERROR': 'FORMAT_ERROR',
        'INITIALIZATION_ERROR': 'INITIALIZATION_ERROR',
        'ENV_ERROR': 'ENV_ERROR',
        'ENV_TIMEOUT': 'ENV_TIMEOUT',
        'REWARD_ERROR': 'REWARD_ERROR',
        'TIMEOUT': 'TASK_TIMEOUT',
        'TASK_TIMEOUT': 'TASK_TIMEOUT',
        'PROVIDER_ERROR': 'UNHANDLED_ERROR',
        'UNHANDLED_ERROR': 'UNHANDLED_ERROR',
    }.get(exit_status, 'UNHANDLED_ERROR')


def judge_skip_reason(agent_output: Any, final_answer: Any) -> str | None:
    """Explain why evaluation should skip an incomplete token-limited rollout."""
    exit_status = agent_exit_status_value(agent_output)
    if exit_status in _TOKEN_LIMIT_EXIT_STATUSES:
        return f'Judge skipped because agent exited with {exit_status}.'
    declared_final_answer = (
        agent_output.get('final_answer', _MISSING)
        if isinstance(agent_output, dict)
        else getattr(agent_output, 'final_answer', _MISSING)
    )
    if declared_final_answer is _MISSING:
        declared_final_answer = final_answer
    if exit_status in _MAX_STEP_EXIT_STATUSES and not str(declared_final_answer or '').strip():
        return f'Judge skipped because agent exited with {exit_status} without a final answer.'
    return None


__all__ = [
    'agent_exit_status_value',
    'agent_termination_reason',
    'agent_tool_error_extra_fields',
    'apply_tool_name_resolution_score',
    'extract_final_answer',
    'judge_skip_reason',
]
