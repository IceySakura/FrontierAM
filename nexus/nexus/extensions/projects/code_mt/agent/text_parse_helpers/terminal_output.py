"""Terminal output handling for Terminus2 agent.

This module provides utilities for capturing and processing terminal output,
including incremental output detection and output length limiting.
"""

from dataclasses import dataclass

from nexus.core import require_context


@dataclass
class Terminus2State:
    """Track terminal state for incremental output detection."""

    previous_buffer: str | None = None


def _find_new_content(previous_buffer: str, current_buffer: str) -> str | None:
    """Find new content between two terminal buffers.

    Retains the last line of previous_buffer as overlapping context.
    """
    pb = '' if previous_buffer is None else previous_buffer.strip()

    # if pb in current_buffer:
    #     idx = current_buffer.index(pb)
    #     if "\n" in pb:
    #         idx = pb.rfind("\n")
    #     return current_buffer[idx:]
    # return None

    if pb in current_buffer:
        start_idx = current_buffer.index(pb)
        if '\n' in pb:
            return current_buffer[start_idx + pb.rfind('\n') :]
        return current_buffer[start_idx:]
    return None


async def get_incremental_output(session_id: str, state: Terminus2State) -> str:
    """Get incremental terminal output since last call.

    This function tracks the terminal buffer state and returns only new content
    that has been added since the last call. This provides better context for
    the LLM by avoiding repeated output.

    The output format includes a header indicating whether this is new output
    or the current terminal screen (when new content can't be determined).

    Args:
        session_id: Terminal session ID
        state: State object tracking previous buffer

    Returns:
        str: Formatted output with "New Terminal Output:" or "Current Terminal Screen:" header
    """
    runtime = await require_context().get_runtime()

    # Capture full buffer history
    full_pane_response = await runtime.capture_terminal_pane(session_id=session_id, capture_entire=True)
    current_buffer = full_pane_response.content

    # First capture - no previous state
    if state.previous_buffer is None:
        state.previous_buffer = current_buffer
        # Return only visible screen
        visible_pane_response = await runtime.capture_terminal_pane(session_id=session_id, capture_entire=False)
        return f'Current Terminal Screen:\n{visible_pane_response.content}'

    # Try to find new content
    new_content = _find_new_content(state.previous_buffer, current_buffer)

    # Update state
    state.previous_buffer = current_buffer

    if new_content is not None:
        if new_content.strip():
            # Found new content
            return f'New Terminal Output:\n{new_content}'
        # No new content, show current screen
        visible_pane_response = await runtime.capture_terminal_pane(session_id=session_id, capture_entire=False)
        return f'Current Terminal Screen:\n{visible_pane_response.content}'
    # Couldn't reliably determine new content, fall back to current screen
    visible_pane_response = await runtime.capture_terminal_pane(session_id=session_id, capture_entire=False)
    return f'Current Terminal Screen:\n{visible_pane_response.content}'


def limit_output_length(output: str, max_bytes: int = 10000) -> str:
    """Limit output to specified byte length, keeping first and last portions.

    Args:
        output: The terminal output to potentially truncate
        max_bytes: Maximum allowed bytes (default 10000)

    Returns:
        str: Original output if under limit, or truncated with middle omitted
    """
    if len(output.encode('utf-8')) <= max_bytes:
        return output

    # Calculate portions (half each for first and last)
    portion_size = max_bytes // 2

    # Convert to bytes for accurate splitting
    output_bytes = output.encode('utf-8')

    # Get first portion
    first_portion = output_bytes[:portion_size].decode('utf-8', errors='ignore')

    # Get last portion
    last_portion = output_bytes[-portion_size:].decode('utf-8', errors='ignore')

    # Calculate omitted bytes
    omitted_bytes = len(output_bytes) - len(first_portion.encode('utf-8')) - len(last_portion.encode('utf-8'))

    return f'{first_portion}\n[... output limited to {max_bytes} bytes; {omitted_bytes} interior bytes omitted ...]\n{last_portion}'


def get_completion_confirmation_message(terminal_output: str, parser_name: str = 'json') -> str:
    """Return the task completion confirmation message.

    Args:
        terminal_output: Current terminal output
        parser_name: Parser format ('json' or 'xml')
    """
    base = (
        f'Current terminal state:\n{terminal_output}\n\n'
        'Are you sure you want to mark the task as complete? '
        "This will trigger your solution to be graded and you won't be able to "
        'make any further corrections. If so, '
    )
    if parser_name == 'xml':
        return base + 'include <task_complete>true</task_complete> in your response again.'
    return base + 'include "task_complete": true in your JSON response again.'
