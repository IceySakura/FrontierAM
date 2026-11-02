"""CodeAct-agent tools for nexus framework.

This module provides the core tools used by CodeAct-agent:
- terminal: Execute bash commands in the runtime environment
- file_editor: View, create, and edit files with str_replace operations
- task_tracker: Structured task management for development workflows
- think: Think through a problem step by step
- finish: Signal completion of the current task

All tools use context for state management:
- context.tool_state['working_dir']: Current working directory
- context.tool_state['terminal_session_id']: Terminal session ID for bash
- context.tool_state['task_list']: Current task list for task_tracker
"""

import asyncio
import contextlib
import time

from pathlib import PurePosixPath
from typing import Literal

from openai.types.chat import ChatCompletionContentPartParam
from pydantic import BaseModel, ConfigDict, Field

from nexus.core import ToolContext, ToolResponse, ToolType, tool_registry
from nexus.extensions.tools.file_editor import FileEditor, ToolError
from nexus.runtime.executors import (
    CapturePaneRequest,
    CreateTerminalSessionRequest,
    DestroyTerminalSessionRequest,
    QueryCommandStatusRequest,
    SendKeysToSessionRequest,
    StartCommandInSessionRequest,
)


# ============== Security Risk Type ==============

# Security risk levels for actions - allows the LLM to self-report the safety risk
SecurityRiskType = Literal['UNKNOWN', 'LOW', 'MEDIUM', 'HIGH']
SECURITY_RISK_DESCRIPTION = (
    'Security risk levels for actions.\n\n'
    'Based on OpenHands security risk levels but adapted for agent-sdk.\n'
    'Integer values allow for easy comparison and ordering.'
)


# ============== Constants ==============

# Maximum output size for terminal commands
MAX_CMD_OUTPUT_SIZE = 30000

# Default truncation notice (without file saving)
DEFAULT_TRUNCATE_NOTICE = (
    '<response clipped><NOTE>Due to the max output limit, only part of the full response has been shown to you.</NOTE>'
)

# Truncation notice for file editor (different wording)
FILE_TRUNCATE_NOTICE = (
    '<response clipped><NOTE>To save on context only part of this file has been shown to you. '
    'You should retry this tool after you have searched inside the file with `grep -n` in order to find '
    'the line numbers of what you are looking for.</NOTE>'
)


# ============== Utility Functions ==============


def maybe_truncate(
    content: str,
    truncate_after: int | None = None,
    truncate_notice: str = DEFAULT_TRUNCATE_NOTICE,
) -> str:
    """Truncate the middle of content if it exceeds the specified length.

    Keeps the head and tail of the content to preserve context at both ends.

    Args:
        content: The text content to potentially truncate
        truncate_after: Maximum length before truncation. If None, no truncation occurs
        truncate_notice: Notice to insert in the middle when content is truncated

    Returns:
        Original content if under limit, or truncated content with head and tail
        preserved
    """
    # Early exits: no truncation requested, or content already within limit
    if not truncate_after or len(content) <= truncate_after or truncate_after < 0:
        return content

    # If even the base notice doesn't fit, return a slice of it
    if len(truncate_notice) >= truncate_after:
        return truncate_notice[:truncate_after]

    # Calculate available space for content (after notice)
    available_chars = truncate_after - len(truncate_notice)
    # Prefer giving the "extra" char to head (ceil split)
    head_chars = available_chars // 2 + (available_chars % 2)
    tail_chars = available_chars - head_chars

    return content[:head_chars] + truncate_notice + (content[-tail_chars:] if tail_chars > 0 else '')


def get_working_dir(context: ToolContext) -> str:
    """Get current working directory from context."""
    return context.tool_state.get('working_dir', '/testbed')


def _make_output(text: str) -> list[ChatCompletionContentPartParam | dict]:
    """Helper to create standard output format."""
    return [{'type': 'text', 'text': text}]


# ============== Terminal Tool ==============

# Default timeout for no output change (soft timeout)
NO_CHANGE_TIMEOUT_SECONDS = 30
# Polling interval for command status
POLL_INTERVAL = 0.5
# Timeout message template
TIMEOUT_MESSAGE_TEMPLATE = (
    'You can: (1) Set `is_input` to `true` and `command` to an empty string to retrieve more logs, '
    'or (2) Send `C-c` (Ctrl+C) to interrupt the running process.'
)


def _is_special_key(command: str) -> bool:
    """Check if the command is a special key (e.g., C-c, C-d, C-z).

    Special keys are of the form C-<key> where key is a single character.
    """
    _command = command.strip()
    return _command.startswith('C-') and len(_command) == 3


class TerminalInput(BaseModel):
    """Input for terminal command execution."""

    command: str = Field(
        description='The bash command to execute. Can be empty string to view additional logs when previous exit code is `-1`. Can be `C-c` (Ctrl+C) to interrupt the currently running process. Note: You can only execute one bash command at a time. If you need to run multiple commands sequentially, you can use `&&` or `;` to chain them together.',
    )
    is_input: bool = Field(
        default=False,
        description='If True, the command is an input to the running process. If False, the command is a bash command to be executed in the terminal. Default is False.',
    )
    timeout: float | None = Field(
        default=None,
        ge=0,
        description=f"Optional. Sets a maximum time limit (in seconds) for running the command. If the command takes longer than this limit, you'll be asked whether to continue or stop it. If you don't set a value, the command will instead pause and ask for confirmation when it produces no new output for {NO_CHANGE_TIMEOUT_SECONDS} seconds. Use a higher value if the command is expected to take a long time (like installation or testing), or if it has a known fixed duration (like sleep).",
    )
    reset: bool = Field(
        default=False,
        description='If True, reset the terminal by creating a new session. Use this only when the terminal becomes unresponsive. Note that all previously set environment variables and session state will be lost after reset. Cannot be used with is_input=True.',
    )
    # security_risk:  SecurityRiskType = Field(description=SECURITY_RISK_DESCRIPTION)


async def _create_terminal_session(context: ToolContext) -> tuple[str, dict]:
    """Create a new terminal session and initialize it.

    Returns:
        Tuple of (session_id, context_updates)
    """
    from nexus.runtime.executors.terminal import create_terminal_session, send_keys_to_session

    create_response = await create_terminal_session(CreateTerminalSessionRequest())
    session_id = create_response.session_id

    # Change to working directory
    working_dir = get_working_dir(context)
    await send_keys_to_session(
        SendKeysToSessionRequest(
            session_id=session_id,
            keys=[f'cd {working_dir}', 'Enter'],
        )
    )

    # Wait a bit for the cd command to complete
    await asyncio.sleep(0.5)

    return session_id, {'terminal_session_id': session_id}


def _format_terminal_output(
    output: str,
    exit_code: int | None,
    is_timeout: bool = False,
) -> str:
    """Format terminal output with metadata suffix."""
    result = output

    if exit_code is not None:
        if exit_code == -1 or is_timeout:
            result += '\n[Command still running - soft timeout reached]'
        else:
            result += f'\n[Command finished with exit code {exit_code}]'

    return result


@tool_registry.register(
    'codeact/terminal',
    meta={
        'name': 'terminal',
        'description': (
            'Execute a bash command in the terminal within a persistent shell session.\n\n\n### Command Execution\n* One command at a time: You can only execute one bash command at a time. If you need to run multiple commands sequentially, use `&&` or `;` to chain them together.\n* Persistent session: Commands execute in a persistent shell session where environment variables, virtual environments, and working directory persist between commands.\n* Soft timeout: Commands have a soft timeout of 10 seconds, once that\'s reached, you have the option to continue or interrupt the command (see section below for details)\n* Shell options: Do NOT use `set -e`, `set -eu`, or `set -euo pipefail` in shell scripts or commands in this environment. The runtime may not support them and can cause unusable shell sessions. If you want to run multi-line bash commands, write the commands to a file and then run it, instead.\n\n### Long-running Commands\n* For commands that may run indefinitely, run them in the background and redirect output to a file, e.g. `python3 app.py > server.log 2>&1 &`.\n* For commands that may run for a long time (e.g. installation or testing commands), or commands that run for a fixed amount of time (e.g. sleep), you should set the "timeout" parameter of your function call to an appropriate value.\n* If a bash command returns exit code `-1`, this means the process hit the soft timeout and is not yet finished. By setting `is_input` to `true`, you can:\n  - Send empty `command` to retrieve additional logs\n  - Send text (set `command` to the text) to STDIN of the running process\n  - Send control commands like `C-c` (Ctrl+C), `C-d` (Ctrl+D), or `C-z` (Ctrl+Z) to interrupt the process\n  - If you do C-c, you can re-start the process with a longer "timeout" parameter to let it run to completion\n\n### Best Practices\n* Directory verification: Before creating new directories or files, first verify the parent directory exists and is the correct location.\n* Directory management: Try to maintain working directory by using absolute paths and avoiding excessive use of `cd`.\n\n### Output Handling\n* Output truncation: If the output exceeds a maximum length, it will be truncated before being returned.\n\n### Terminal Reset\n* Terminal reset: If the terminal becomes unresponsive, you can set the "reset" parameter to `true` to create a new terminal session. This will terminate the current session and start fresh.\n* Warning: Resetting the terminal will lose all previously set environment variables, working directory changes, and any running processes. Use this only when the terminal stops responding to commands.\n'
        ),
        'tool_type': ToolType.RUNTIME,
    },
)
async def terminal_tool(params: TerminalInput, context: ToolContext) -> ToolResponse:
    """Execute a terminal command in the runtime environment.

    Supports:
    - Normal command execution
    - Input to running process (is_input=True)
    - Terminal reset (reset=True)
    - Soft timeout with exit_code=-1

    State updates:
    - terminal_session_id: Session ID for persistent terminal
    - terminal_command_running: Whether a command is currently running
    """
    from nexus.runtime.executors.terminal import (
        capture_pane,
        destroy_terminal_session,
        query_command_status,
        send_keys_to_session,
        start_command_in_session,
    )

    session_id = context.tool_state.get('terminal_session_id')
    command_running = context.tool_state.get('terminal_command_running', False)
    context_updates: dict = {}

    # Handle reset request
    if params.reset:
        if params.is_input:
            return ToolResponse(
                output=_make_output('Error: Cannot use reset=True with is_input=True'),
                meta={'error': True},
            )

        # Destroy existing session if any
        if session_id:
            with contextlib.suppress(Exception):
                await destroy_terminal_session(DestroyTerminalSessionRequest(session_id=session_id))

        # Create new session
        session_id, context_updates = await _create_terminal_session(context)
        context_updates['terminal_command_running'] = False

        return ToolResponse(
            output=_make_output('Terminal session has been reset. All previous state has been cleared.'),
            context_updates=context_updates,
            meta={'command': None, 'exit_code': 0, 'reset': True},
        )

    # Create session if needed
    if session_id is None:
        session_id, context_updates = await _create_terminal_session(context)
        command_running = False

    command = params.command.strip()

    # Check if no command is running - handle empty command and is_input cases
    if not command_running:
        if command == '':
            return ToolResponse(
                output=_make_output('No previous running command to retrieve logs from.'),
                context_updates=context_updates,
                meta={'is_error': True},
            )
        if params.is_input:
            return ToolResponse(
                output=_make_output('No previous running command to interact with.'),
                context_updates=context_updates,
                meta={'is_error': True},
            )

    # Get or create command_id for polling
    command_id = context.tool_state.get('terminal_command_id')

    # Check if previous command is still running and trying to send a new command (not is_input)
    if command_running and command and not params.is_input:
        # Capture current output to show what's happening
        pane_response = await capture_pane(CapturePaneRequest(session_id=session_id))
        error_msg = (
            f'[Below is the output of the previous command.]\n'
            f'{pane_response.content}\n'
            f'[Your command "{command}" is NOT executed. The previous command is still running - '
            f'You CANNOT send new commands until the previous command is completed. '
            f'By setting `is_input` to `true`, you can interact with the current process: '
            f'send empty command to retrieve more logs, or send `C-c` (Ctrl+C) to interrupt it.]'
        )
        return ToolResponse(
            output=_make_output(error_msg),
            context_updates=context_updates,
            meta={'command': command, 'is_error': True},
        )

    # Send command or input to terminal
    if command:
        is_special_key = _is_special_key(command)
        if params.is_input:
            # is_input mode: send input to running process
            if is_special_key:
                # Special keys (C-c, C-d, C-z) - send without Enter
                await send_keys_to_session(SendKeysToSessionRequest(session_id=session_id, keys=command))
            else:
                # Regular input - send with Enter
                await send_keys_to_session(SendKeysToSessionRequest(session_id=session_id, keys=[command, 'Enter']))
        else:
            # Normal command execution - start new command
            try:
                cmd_info = await start_command_in_session(
                    StartCommandInSessionRequest(
                        session_id=session_id,
                        command=command,
                    )
                )
                command_id = cmd_info.command_id
                context_updates['terminal_command_id'] = command_id

            except Exception as e:
                # Handle command start errors
                error_message = f'Error: {e}'

                # Match case-insensitively: the runtime raises
                # 'Command failed to start' (lowercase 'failed').
                if 'failed to start' in str(e).lower():
                    try:
                        pane_response = await capture_pane(CapturePaneRequest(session_id=session_id))
                        error_message = (
                            'Your bash command failed to start. '
                            'Please fix the syntax errors and try again. This can be the result '
                            'of not adhering to the syntax for multi-line commands. Here is the captured shell content:\n'
                            + pane_response.content
                        )
                    except Exception:
                        pass

                return ToolResponse(
                    output=_make_output(error_message),
                    context_updates=context_updates,
                    meta={'command': command, 'is_error': True},
                )

    # Sanity check: command_id should always be available at this point
    # - For new commands: set in L306
    # - For is_input mode: should exist in context (since command_running=True)
    if command_id is None:
        raise RuntimeError(
            'Unexpected state: command_id is None. This indicates a bug in terminal_tool state management.'
        )

    # Polling loop for command completion
    start_time = time.time()
    last_change_time = start_time
    is_blocking = params.timeout is not None  # Hard timeout mode

    # Get initial output - for is_input mode, query the existing command first
    if params.is_input:
        try:
            cmd_info = await query_command_status(
                QueryCommandStatusRequest(
                    session_id=session_id,
                    command_id=command_id,
                )
            )
            last_output = cmd_info.output or ''
        except Exception:
            last_output = ''
    else:
        last_output = cmd_info.output or ''

    while True:
        await asyncio.sleep(POLL_INTERVAL)

        # Query command status
        try:
            cmd_info = await query_command_status(
                QueryCommandStatusRequest(
                    session_id=session_id,
                    command_id=command_id,
                )
            )
        except Exception as e:
            return ToolResponse(
                output=_make_output(f'Error querying command status: {e}'),
                context_updates=context_updates,
                meta={'command': command, 'is_error': True},
            )

        current_output = cmd_info.output or ''

        # Check if output changed
        if current_output != last_output:
            last_output = current_output
            last_change_time = time.time()

        # 1) Command completed - check if end_time is set
        if cmd_info.end_time is not None:
            exit_code = cmd_info.exit_code
            formatted_output = _format_terminal_output(current_output, exit_code)

            context_updates['terminal_command_running'] = False

            return ToolResponse(
                output=_make_output(maybe_truncate(formatted_output, truncate_after=MAX_CMD_OUTPUT_SIZE)),
                context_updates=context_updates,
                meta={
                    'command': command,
                    'exit_code': exit_code,
                    'stdout': cmd_info.stdout,
                    'stderr': cmd_info.stderr,
                },
            )

        # 2) No-change timeout - output hasn't changed for NO_CHANGE_TIMEOUT_SECONDS
        # Only applies when not in blocking (hard timeout) mode
        time_since_last_change = time.time() - last_change_time
        if not is_blocking and time_since_last_change >= NO_CHANGE_TIMEOUT_SECONDS:
            context_updates['terminal_command_running'] = True

            formatted_output = (
                f'{current_output}\n'
                f'[The command has no new output after {NO_CHANGE_TIMEOUT_SECONDS} seconds. '
                f'{TIMEOUT_MESSAGE_TEMPLATE}]'
            )

            return ToolResponse(
                output=_make_output(maybe_truncate(formatted_output, truncate_after=MAX_CMD_OUTPUT_SIZE)),
                context_updates=context_updates,
                meta={
                    'command': command,
                    'exit_code': -1,
                    'timeout_type': 'no_change',
                },
            )

        # 3) Hard timeout - total time exceeded params.timeout
        if is_blocking and params.timeout is not None:
            elapsed_time = time.time() - start_time
            if elapsed_time >= params.timeout:
                context_updates['terminal_command_running'] = True

                formatted_output = (
                    f'{current_output}\n'
                    f'[The command timed out after {params.timeout} seconds. '
                    f'{TIMEOUT_MESSAGE_TEMPLATE}]'
                )

                return ToolResponse(
                    output=_make_output(maybe_truncate(formatted_output, truncate_after=MAX_CMD_OUTPUT_SIZE)),
                    context_updates=context_updates,
                    meta={
                        'command': command,
                        'exit_code': -1,
                        'timeout_type': 'hard',
                    },
                )


# ============== String Replace Editor Tool ==============

# Global FileEditor instance (lazily initialized per working directory)
_file_editors: dict[str, FileEditor] = {}

# Base description for file editor tool (without working directory info)
FILE_EDITOR_DESCRIPTION = """Custom editing tool for viewing, creating and editing files in plain-text format
* State is persistent across command calls and discussions with the user
* If `path` is an image file (.png, .jpg, .jpeg, .gif, .webp, .bmp), `view` displays the image content
* If `path` is a text file, `view` displays the result of applying `cat -n`. If `path` is a directory, `view` lists non-hidden files and directories up to 2 levels deep
* The `create` command cannot be used if the specified `path` already exists as a file
* If a `command` generates a long output, it will be truncated and marked with `<response clipped>`
* The `undo_edit` command will revert the last edit made to the file at `path`
* This tool can be used for creating and editing files in plain-text format.


Before using this tool:
1. Use the view tool to understand the file's contents and context
2. Verify the directory path is correct (only applicable when creating new files):
   - Use the view tool to verify the parent directory exists and is the correct location

When making edits:
   - Ensure the edit results in idiomatic, correct code
   - Do not leave the code in a broken state
   - Always use absolute file paths (starting with /)

CRITICAL REQUIREMENTS FOR USING THIS TOOL:

1. EXACT MATCHING: The `old_str` parameter must match EXACTLY one or more consecutive lines from the file, including all whitespace and indentation. The tool will fail if `old_str` matches multiple locations or doesn't match exactly with the file content.

2. UNIQUENESS: The `old_str` must uniquely identify a single instance in the file:
   - Include sufficient context before and after the change point (3-5 lines recommended)
   - If not unique, the replacement will not be performed

3. REPLACEMENT: The `new_str` parameter should contain the edited lines that replace the `old_str`. Both strings must be different.

Remember: when making multiple file edits in a row to the same file, you should prefer to send all edits in a single message with multiple calls to this tool, rather than multiple messages with a single call each.
"""


def get_file_editor_description(working_dir: str) -> str:
    """Construct the file editor description with working directory info."""
    return (
        f'{FILE_EDITOR_DESCRIPTION}\n'
        f'Your current working directory is: {working_dir}\n'
        f'When exploring project structure, start with this directory instead of the root filesystem.'
    )


def _get_file_editor(working_dir: str) -> FileEditor:
    """Get or create a FileEditor instance for the given working directory."""
    if working_dir not in _file_editors:
        _file_editors[working_dir] = FileEditor(workspace_root=working_dir)
    return _file_editors[working_dir]


class FileEditorInput(BaseModel):
    """Input for the string replacement editor."""

    command: Literal['view', 'create', 'str_replace', 'insert', 'undo_edit'] = Field(
        description='The command to run. Allowed options are: `view`, `create`, `str_replace`, `insert`, `undo_edit`.'
    )
    path: str = Field(description='Absolute path to file or directory.')
    file_text: str | None = Field(
        default=None,
        description='Required parameter of `create` command, with the content of the file to be created.',
    )
    old_str: str | None = Field(
        default=None,
        description='Required parameter of `str_replace` command containing the string in `path` to replace.',
    )
    new_str: str | None = Field(
        default=None,
        description=(
            'Optional parameter of `str_replace` command containing the new string '
            '(if not given, no string will be added). Required parameter of `insert` command '
            'containing the string to insert.'
        ),
    )
    insert_line: int | None = Field(
        default=None,
        description=(
            'Required parameter of `insert` command. '
            'The `new_str` will be inserted AFTER the line `insert_line` of `path`.'
        ),
    )
    view_range: list[int] | None = Field(
        default=None,
        description=(
            'Optional parameter of `view` command when `path` points to a file. '
            'If none is given, the full file is shown. If provided, the file will be shown '
            'in the indicated line number range, e.g. [11, 12] will show lines 11 and 12. '
            'Indexing at 1 to start. Setting `[start_line, -1]` shows all lines from `start_line` to the end of the file.'
        ),
    )
    # security_risk: SecurityRiskType = Field(description=SECURITY_RISK_DESCRIPTION)


@tool_registry.register(
    'codeact/file_editor',
    meta={
        'name': 'file_editor',
        'description': FILE_EDITOR_DESCRIPTION,
        'tool_type': ToolType.RUNTIME,
    },
)
async def file_editor_tool(params: FileEditorInput, context: ToolContext) -> ToolResponse:
    """File editor for file operations.

    Uses FileEditor from nexus.extensions.tools.file_editor for all file operations.
    The FileEditor handles:
    - File viewing with line numbers
    - Directory listing
    - File creation
    - String replacement with uniqueness checking
    - Line insertion
    - Undo functionality with history management
    """
    working_dir = get_working_dir(context)

    # Resolve path - ensure absolute path
    path_str = params.path
    if not path_str.startswith('/'):
        path_str = str(PurePosixPath(working_dir) / path_str)

    # Get the FileEditor instance for this working directory
    editor = _get_file_editor(working_dir)

    try:
        # Call the FileEditor with the appropriate parameters
        observation = editor(
            command=params.command,
            path=path_str,
            file_text=params.file_text,
            view_range=params.view_range,
            old_str=params.old_str,
            new_str=params.new_str,
            insert_line=params.insert_line,
        )

        # Build output based on observation
        output_text = observation.text

        # Handle image content if present
        if observation.image_base64 and observation.image_mime_type:
            return ToolResponse(
                output=[
                    {'type': 'text', 'text': output_text},
                    {
                        'type': 'image_url',
                        'image_url': {'url': f'data:{observation.image_mime_type};base64,{observation.image_base64}'},
                    },
                ],
                meta={
                    'command': observation.command,
                    'path': observation.path,
                },
            )

        return ToolResponse(
            output=_make_output(output_text),
            meta={
                'command': observation.command,
                'path': observation.path,
            },
        )

    except ToolError as e:
        return ToolResponse(
            output=_make_output(f'Error: {e.message}'),
            meta={'error': str(e)},
        )
    except Exception as e:
        return ToolResponse(
            output=_make_output(f'Error: {e}'),
            meta={'error': str(e)},
        )


# ============== Task Tracker Tool ==============

TASK_TRACKER_DESCRIPTION = "This tool provides structured task management capabilities for development workflows.\nIt enables systematic tracking of work items, progress monitoring, and efficient\norganization of complex development activities.\n\nThe tool maintains visibility into project status and helps communicate\nprogress effectively to users.\n\n## Application Guidelines\n\nUtilize this tool in the following situations:\n\n1. Multi-phase development work - When projects involve multiple sequential or\n   parallel activities\n2. Complex implementation tasks - Work requiring systematic planning and\n   coordination across multiple components\n3. Explicit user request for task organization - When users specifically ask\n   for structured task management\n4. Multiple concurrent requirements - When users present several work items\n   that need coordination\n5. Project initiation - Capture and organize user requirements at project start\n6. Work commencement - Update task status to in_progress before beginning\n   implementation. Maintain focus by limiting active work to one task\n7. Task completion - Update status to done and identify any additional work\n   that emerged during implementation\n\n## Situations Where Tool Usage Is Unnecessary\n\nAvoid using this tool when:\n\n1. Single atomic tasks that require no decomposition\n2. Trivial operations where tracking adds no organizational value\n3. Simple activities completable in minimal steps\n4. Pure information exchange or discussion\n\nNote: For single straightforward tasks, proceed with direct implementation\nrather than creating tracking overhead.\n\n## Usage Scenarios\n\n**Scenario A: Feature Development with Validation**\nUser request: \"Build a user authentication system with login/logout functionality.\nDon't forget to include input validation and error handling!\"\n\nResponse approach: I'll implement a user authentication system with comprehensive\nvalidation. Let me organize this work systematically.\n\n*Task breakdown includes:*\n1. Design authentication API endpoints and data models\n2. Implement user login/logout controllers with session management\n3. Add client-side form validation and error display\n4. Create backend input sanitization and security checks\n5. Implement comprehensive error handling for edge cases\n6. Perform integration testing and fix any issues\n\n**Scenario B: Codebase Refactoring**\nUser request: \"I need to update all instances of the method 'fetchData' to\n'retrieveInformation' throughout the entire project\"\n\nResponse approach: Let me first analyze your codebase to identify all locations\nwhere 'fetchData' appears.\n\n*After code analysis*\n\nI've located 12 occurrences of 'fetchData' across 6 files in your project.\nI'll create a systematic plan to update these references.\n\n*Organized task list includes specific file-by-file updates*\n\n**Scenario C: Multi-feature Development**\nUser request: \"Set up a blog platform with these components: admin dashboard,\narticle management, comment system, and user profiles.\"\n\nResponse approach: I'll help build your blog platform. Let me break down these\ncomponents into manageable implementation phases.\n\n*Creates structured plan with each feature decomposed into specific development tasks*\n\n## Counter-examples - Direct Implementation Preferred\n\n**Counter-example A: Simple inquiry**\nUser request: \"What's the syntax for a for loop in JavaScript?\"\n\nResponse approach: Here's how to write a for loop in JavaScript:\n\n```javascript\nfor (let i = 0; i < array.length; i++) {\n    console.log(array[i]);\n}\n```\n\n*No task tracking needed - single informational response*\n\n**Counter-example B: Minor modification**\nUser request: \"Please add a docstring to the processPayment function explaining\nits parameters.\"\n\nResponse approach: I'll add a docstring to explain the processPayment function.\n\n*Direct edit without task tracking for simple modification*\n\n## Status Management and Workflow\n\n1. **Status Values**: Track work using these states:\n   - todo: Not yet initiated\n   - in_progress: Currently active (maintain single focus)\n   - done: Successfully completed\n\n2. **Workflow Practices**:\n   - Update status dynamically as work progresses\n   - Mark completion immediately upon task finish\n   - Limit active work to ONE task at any given time\n   - Complete current activities before initiating new ones\n   - Remove obsolete tasks from tracking entirely\n\n3. **Completion Criteria**:\n   - Mark tasks as done only when fully achieved\n   - Keep status as in_progress if errors, blocks, or partial completion exist\n   - Create new tasks for discovered issues or dependencies\n   - Never mark done when:\n       - Test suites are failing\n       - Implementation remains incomplete\n       - Unresolved errors persist\n       - Required resources are unavailable\n\n4. **Task Organization**:\n   - Write precise, actionable descriptions\n   - Decompose complex work into manageable units\n   - Use descriptive, clear naming conventions\n\nWhen uncertain, favor using this tool. Proactive task management demonstrates\nsystematic approach and ensures comprehensive requirement fulfillment."


# Type alias for task tracker status
TaskTrackerStatusType = Literal['todo', 'in_progress', 'done']


class TaskItem(BaseModel):
    """A single task item."""

    title: str = Field(..., description='A brief title for the task.')
    notes: str = Field(default='', description='Additional details or notes about the task.')
    status: TaskTrackerStatusType = Field(
        default='todo',
        description="The current status of the task. One of 'todo', 'in_progress', or 'done'.",
    )


class TaskTrackerInput(BaseModel):
    """Input for task tracker tool."""

    command: Literal['view', 'plan'] = Field(
        default='view',
        description='The command to execute. `view` shows the current task list. `plan` creates or updates the task list based on provided requirements and progress. Always `view` the current list before making changes.',
    )
    task_list: list[TaskItem] = Field(
        default_factory=list,
        description='The full task list. Required parameter of `plan` command.',
    )
    # security_risk: SecurityRiskType = Field(description=SECURITY_RISK_DESCRIPTION)


def _format_task_list(task_list: list[TaskItem]) -> str:
    """Format the task list for display.

    Uses emoji icons for status:
    - ⏳ todo
    - 🔄 in_progress
    - ✅ done
    """
    if not task_list:
        return 'No tasks in the list.'

    content = '# Task List\n\n'
    for i, task in enumerate(task_list, 1):
        status_icon = {'todo': '⏳', 'in_progress': '🔄', 'done': '✅'}.get(task.status, '⏳')

        content += f'{i}. {status_icon} {task.title}\n'
        if task.notes:
            content += f'   {task.notes}\n'
        content += '\n'

    return content.strip()


@tool_registry.register(
    'codeact/task_tracker',
    meta={
        'name': 'task_tracker',
        'description': TASK_TRACKER_DESCRIPTION,
        'tool_type': ToolType.LOCAL,
    },
)
async def task_tracker_tool(params: TaskTrackerInput, context: ToolContext) -> ToolResponse:
    """Task tracker for structured task management.

    State updates:
    - task_list: The current list of tasks (stored as list of dicts for serialization)
    """
    # Read task list from context and convert dicts back to TaskItem models
    task_list_raw = context.tool_state.get('task_list', [])
    task_list: list[TaskItem] = [TaskItem.model_validate(t) for t in task_list_raw]

    if params.command == 'view':
        # Return the current task list
        if not task_list:
            return ToolResponse(
                output=_make_output('No task list found. Use the "plan" command to create one.'),
                meta={'command': params.command, 'task_list': []},
            )

        content = _format_task_list(task_list)
        return ToolResponse(
            output=_make_output(content),
            meta={'command': params.command, 'task_list': task_list_raw},
        )

    if params.command == 'plan':
        # Update the task list (store as dicts for serialization)
        new_task_list = [task.model_dump() for task in params.task_list]

        return ToolResponse(
            output=_make_output(f'Task list has been updated with {len(new_task_list)} item(s).'),
            context_updates={'task_list': new_task_list},
            meta={'command': params.command, 'task_list': new_task_list},
        )

    # Unknown command (should not happen due to Literal type, but handle gracefully)
    return ToolResponse(
        output=_make_output(f'Unknown command: {params.command}. Supported commands are "view" and "plan".'),
        meta={'command': params.command, 'error': True},
    )


# ============== Think Tool ==============


class ThinkInput(BaseModel):
    """Input for think tool."""

    model_config = ConfigDict(
        json_schema_extra={'description': 'Action for logging a thought without making any changes.'},
    )

    thought: str = Field(description='The thought to log.')


@tool_registry.register(
    'codeact/think',
    meta={
        'name': 'think',
        'description': (
            'Use the tool to think about something. It will not obtain new information or make any changes to the repository, but just log the thought. Use it when complex reasoning or brainstorming is needed.\n\nCommon use cases:\n1. When exploring a repository and discovering the source of a bug, call this tool to brainstorm several unique ways of fixing the bug, and assess which change(s) are likely to be simplest and most effective.\n2. After receiving test results, use this tool to brainstorm ways to fix failing tests.\n3. When planning a complex refactoring, use this tool to outline different approaches and their tradeoffs.\n4. When designing a new feature, use this tool to think through architecture decisions and implementation details.\n5. When debugging a complex issue, use this tool to organize your thoughts and hypotheses.\n\nThe tool simply logs your thought process for better transparency and does not execute any code or make changes.'
        ),
        'tool_type': ToolType.LOCAL,
    },
)
async def think_tool(params: ThinkInput, context: ToolContext) -> ToolResponse:
    """Log a thought for transparency.

    This tool does not make any changes, it just acknowledges the thought.
    The thought is stored in meta for debugging/logging purposes.
    """
    return ToolResponse(
        output=_make_output('Your thought has been logged.'),
        meta={'thought': params.thought},
    )


# ============== Finish Tool ==============


class FinishInput(BaseModel):
    """Input for finish tool."""

    message: str = Field(description='Final message to send to the user.')


@tool_registry.register(
    'codeact/finish',
    meta={
        'name': 'finish',
        'description': (
            "Signals the completion of the current task or conversation.\n\nUse this tool when:\n- You have successfully completed the user's requested task\n- You cannot proceed further due to technical limitations or missing information\n\nThe message should include:\n- A clear summary of actions taken and their results\n- Any next steps for the user\n- Explanation if you're unable to complete the task\n- Any follow-up questions if more information is needed\n"
        ),
        'tool_type': ToolType.LOCAL,
    },
)
async def finish_tool(params: FinishInput, context: ToolContext) -> ToolResponse:
    """Signal completion of the current task.

    Sets the exit_status to 'finished' to indicate the agent should stop.
    """
    return ToolResponse(
        output=_make_output(params.message),
        context_updates={'exit_status': 'finished'},
    )
