"""Model-visible runtime state messages for context compaction."""

from __future__ import annotations

from typing import Any


TRAJECTORY_BUDGET_STATE_MESSAGE_MARKER = '_is_trajectory_budget_state'
COMPACTION_RESUME_MESSAGE_MARKER = '_is_compaction_resume'

_COMPACTION_RESUME_CONTENT = (
    '<compaction_resume>\n'
    'Continue the original task from the handoff summary above. Proceed with the next action. '
    'Do not repeat the handoff summary.\n'
    '</compaction_resume>'
)


def build_trajectory_budget_state_message(
    *,
    token_budget: int | None,
    token_consumed: int,
    completed_compactions: int,
) -> dict[str, Any]:
    """Render model-visible trajectory budget progress for initial or compacted context."""
    total = 'infinity' if token_budget is None else str(token_budget)
    remaining = 'infinity' if token_budget is None else str(max(token_budget - token_consumed, 0))
    return {
        'role': 'user',
        'content': (
            '<trajectory_budget_state>\n'
            f'Total response-token budget: {total}\n'
            f'Consumed response tokens: {token_consumed}\n'
            f'Remaining response tokens: {remaining}\n'
            f'Completed compactions: {completed_compactions}\n'
            'When a finite response-token budget is close to exhaustion, stop further exploration and provide '
            'the best-supported final answer.\n'
            '</trajectory_budget_state>'
        ),
        TRAJECTORY_BUDGET_STATE_MESSAGE_MARKER: True,
    }


def build_compaction_resume_message() -> dict[str, Any]:
    """Request a fresh assistant turn after an assistant-role handoff summary."""
    return {
        'role': 'user',
        'content': _COMPACTION_RESUME_CONTENT,
        COMPACTION_RESUME_MESSAGE_MARKER: True,
    }


def is_trajectory_budget_state_message(message: dict[str, Any]) -> bool:
    return bool(message.get(TRAJECTORY_BUDGET_STATE_MESSAGE_MARKER))


def is_compaction_runtime_state_message(message: dict[str, Any]) -> bool:
    """Return whether a message is transient context regenerated after compaction."""
    return bool(message.get(TRAJECTORY_BUDGET_STATE_MESSAGE_MARKER) or message.get(COMPACTION_RESUME_MESSAGE_MARKER))
