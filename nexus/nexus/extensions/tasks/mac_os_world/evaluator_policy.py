"""Fail-closed policies for macOSWorld evaluator commands."""

from __future__ import annotations

from typing import Any


_UNSAFE_SCRIPT_EDITOR_FRONT_DOCUMENT_GRADER = (
    'osascript -e \'tell application "Script Editor" to get contents of front document\' '
    '2>/dev/null | osascript - 2>/dev/null | grep -q "Script Editor" && echo "True" || echo "False" '
)


def unsafe_agent_authored_evaluator_reason(grading_command: Any) -> str | None:
    """Identify the exact upstream grader that executes agent-authored AppleScript."""

    if (
        isinstance(grading_command, list)
        and len(grading_command) == 1
        and isinstance(grading_command[0], list)
        and grading_command[0] == [_UNSAFE_SCRIPT_EDITOR_FRONT_DOCUMENT_GRADER, 100]
    ):
        return 'executes_script_editor_front_document'
    return None


__all__ = ['unsafe_agent_authored_evaluator_reason']
