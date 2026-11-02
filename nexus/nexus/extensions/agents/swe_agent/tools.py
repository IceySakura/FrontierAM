"""SWE-agent tools adapted for nexus framework.

This module provides the core tools used by SWE-agent:
- bash: Execute bash commands in the runtime environment
- str_replace_editor: View, create, and edit files with str_replace operations
- submit: Submit the solution as a git patch (with review stage)

Each tool is registered twice, once under `swe_agent/` and once under
`ch_swe_agent/`, sharing one implementation across both languages.

All tools use context for state management:
- context.tool_state['working_dir']: Current working directory
- context.tool_state['terminal_session_id']: Terminal session ID for bash
- context.tool_state['should_exit']: Flag for two-stage submit
"""

import asyncio
import logging
import os
import subprocess

from pathlib import PurePosixPath
from typing import Literal

from openai.types.chat import ChatCompletionContentPartParam
from pydantic import BaseModel, Field

from nexus.core import ToolContext, ToolResponse, ToolType, tool_registry
from nexus.extensions.tools.file_editor import FileEditor, ToolError
from nexus.runtime.executors import (
    CapturePaneRequest,
    CreateTerminalSessionRequest,
    RunCommandInSessionRequest,
    SendKeysToSessionRequest,
)
from nexus.trace import get_current_span_id, update_span


logger = logging.getLogger(__name__)


def _resolve_infra_error() -> type[Exception]:
    """Load the terminal error type or provide a compatible fatal exception.

    Runtime images may lack the error module or its dependencies. The fallback
    keeps the class name and fatal flag used by the agent's error handling.
    """
    try:
        from nexus.runtime.executors.terminal import TerminalInfraError as _Err  # noqa: N814

        return _Err
    except Exception:  # Handle missing error classes and runtime dependencies.

        class TerminalInfraError(RuntimeError):  # noqa: N818
            """Fatal terminal error for runtimes without the shared exception type."""

            fatal = True

        return TerminalInfraError


# ============== Constants ==============

MAX_RESPONSE_LEN = 16000
TRUNCATED_MESSAGE = (
    '<response clipped><NOTE>To save on context only part of this file has been shown to you. '
    'You should retry this tool after you have searched inside the file with `grep -n` in order to find '
    'the line numbers of what you are looking for.</NOTE>'
)

# Select stateless commands or a persistent terminal from the configured backend.
BASH_BACKEND = os.getenv('SWE_AGENT_BASH_BACKEND', 'terminal').strip().lower()

# Stateless commands need an explicit working directory and non-interactive environment.
_STATELESS_ENV = 'export PAGER=cat MANPAGER=cat LESS=-R PIP_PROGRESS_BAR=off TQDM_DISABLE=1'


# ============== Utility Functions ==============


def maybe_truncate(content: str, max_length: int = MAX_RESPONSE_LEN) -> str:
    """Truncate content and append a notice if content exceeds the specified length."""
    if len(content) <= max_length:
        return content
    return content[:max_length] + TRUNCATED_MESSAGE


def get_working_dir(context: ToolContext) -> str:
    """Get current working directory from context."""
    return context.tool_state.get('working_dir', '/testbed')


def _make_output(text: str) -> list[ChatCompletionContentPartParam | dict]:
    """Helper to create standard output format."""
    return [{'type': 'text', 'text': text}]


class ToolInfrastructureError(RuntimeError):
    """The command executor failed before returning a normal command result."""

    fatal = True


def resolve_bash_backend(requested: str | None) -> Literal['terminal', 'stateless']:
    """Prefer per-run configuration and retain the legacy environment default."""
    backend = requested or BASH_BACKEND
    if backend == 'stateless':
        return 'stateless'
    if backend in ('terminal', 'tmux'):
        return 'terminal'
    raise ValueError(f'Unsupported SWE bash backend: {backend}')


# ============== Bash Tool ==============


class BashInput(BaseModel):
    """Input for bash command execution."""

    command: str = Field(description='The bash command to execute.')


BASH_DESCRIPTION = (
    'Execute a bash command in the runtime environment. '
    'Use this to run shell commands, navigate directories, search files, '
    'run tests, and execute Python scripts.'
)


@tool_registry.register(
    'swe_agent/bash',
    meta={
        'name': 'bash',
        'description': BASH_DESCRIPTION,
        'tool_type': ToolType.RUNTIME,
    },
)
async def bash_tool(params: BashInput, context: ToolContext) -> ToolResponse:
    """Execute a command through the configured shell backend.

    The persistent backend stores its session ID in tool state. Working-directory
    updates are returned through the tool context.
    """
    return await _run_bash(params.command, context)


async def _run_bash(command: str, context: ToolContext) -> ToolResponse:
    """Dispatch both localized tools through the same configured backend."""
    if resolve_bash_backend(context.tool_state.get('bash_backend')) == 'stateless':
        return await _run_bash_stateless(command, context)
    return await _bash_tmux(BashInput(command=command), context)


async def _bash_stateless(params: BashInput, context: ToolContext) -> ToolResponse:
    """Retain the legacy callable used by existing runtime invocations."""
    return await _run_bash_stateless(params.command, context)


async def _run_bash_stateless(command: str, context: ToolContext) -> ToolResponse:
    """Run one command without persistent terminal state or capture files."""
    from nexus.runtime.executors.command import run_command
    from nexus.runtime.executors.models.command import CommandRunStatus, RunCommandRequest

    response = await run_command(
        RunCommandRequest(
            command=command,
            cwd=get_working_dir(context),
            timeout=300,
            extra_env={
                'PAGER': 'cat',
                'MANPAGER': 'cat',
                'LESS': '-R',
                'PIP_PROGRESS_BAR': 'off',
                'TQDM_DISABLE': '1',
            },
        )
    )
    body = (response.stdout or '') + (response.stderr or '')
    if response.status == CommandRunStatus.Error:
        raise ToolInfrastructureError(f'Command executor failed: {body[:500]}')
    if response.status == CommandRunStatus.TimeLimitExceeded:
        body = maybe_truncate(body) + '\n\nYour command timed out after 300 seconds and was terminated.'
    return ToolResponse(
        output=_make_output(body),
        meta={'command': command, 'exit_code': response.return_code, 'status': response.status.value},
    )


async def _bash_tmux(params: BashInput, context: ToolContext) -> ToolResponse:
    """Execute a command in a persistent terminal session and retain its tool state."""
    from nexus.runtime.executors.terminal import (
        capture_pane,
        create_terminal_session,
        run_command_in_session,
        send_keys_to_session,
    )

    session_id = context.tool_state.get('terminal_session_id')
    context_updates = {}

    # Create session if needed
    if session_id is None:
        create_response = await create_terminal_session(CreateTerminalSessionRequest())
        session_id = create_response.session_id

        # Set up environment variables for non-interactive use
        await run_command_in_session(
            RunCommandInSessionRequest(
                session_id=session_id,
                command='export PAGER=cat MANPAGER=cat LESS=-R PIP_PROGRESS_BAR=off TQDM_DISABLE=1',
                timeout=300,
            )
        )

        # Change to working directory
        working_dir = get_working_dir(context)
        await run_command_in_session(
            RunCommandInSessionRequest(
                session_id=session_id,
                command=f'cd {working_dir}',
                timeout=300,
            )
        )

        context_updates['terminal_session_id'] = session_id

    # Execute the command
    try:
        run_response = await run_command_in_session(
            RunCommandInSessionRequest(
                session_id=session_id,
                command=params.command,
                timeout=300,
            )
        )
        return ToolResponse(
            output=_make_output(run_response.output),
            context_updates=context_updates,
            meta={
                'command': params.command,
                'exit_code': getattr(run_response, 'exit_code', None),
            },
        )

    except Exception as e:
        error_message = 'Error: ' + str(e)
        error_str = str(e)
        # Match terminal startup errors without depending on capitalization.
        error_lower = error_str.lower()

        # Propagate fatal terminal failures so the agent stops using an unusable executor.
        if type(e).__name__ == 'TerminalInfraError' or getattr(e, 'fatal', False) is True:
            logger.error(f'[A-4] terminal infra failure, aborting tool call: {error_str}')
            raise

        # If command failed to start or a command is already running,
        # send Ctrl+C to recover the shell for subsequent commands
        if 'failed to start' in error_lower or 'already running' in error_lower:
            try:
                # Send Ctrl+C to interrupt any stuck process and recover shell
                await send_keys_to_session(SendKeysToSessionRequest(session_id=session_id, keys='C-c'))
                await asyncio.sleep(0.5)
            except Exception:
                pass

            try:
                pane_response = await capture_pane(CapturePaneRequest(session_id=session_id))
                error_message = (
                    'Your bash command failed to start. '
                    'A Ctrl+C has been sent to recover the shell. '
                    'Please fix the syntax errors and try again. This can be the result '
                    'of not adhering to the syntax for multi-line commands. Here is the captured shell content:\n'
                    + pane_response.content
                )
            except Exception:
                pass

        return ToolResponse(
            output=_make_output(error_message),
        )


# ============== String Replace Editor Tool ==============

# Global FileEditor instance (lazily initialized per working directory)
_file_editors: dict[str, FileEditor] = {}


def _get_file_editor(working_dir: str) -> FileEditor:
    """Get or create a FileEditor instance for the given working directory."""
    if working_dir not in _file_editors:
        _file_editors[working_dir] = FileEditor(workspace_root=working_dir)
    return _file_editors[working_dir]


class StrReplaceEditorInput(BaseModel):
    """Input for the string replacement editor."""

    command: Literal['view', 'create', 'str_replace', 'insert', 'undo_edit'] = Field(
        description='The command to run. Allowed options are: `view`, `create`, `str_replace`, `insert`, `undo_edit`.'
    )
    path: str = Field(description='Absolute path to file or directory, e.g. `/testbed/file.py` or `/testbed`.')
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
            'Indexing at 1 to start. Setting `[start_line, -1]` shows all lines from `start_line` to the end.'
        ),
    )


STR_REPLACE_EDITOR_DESCRIPTION = (
    'Custom editing tool for viewing, creating and editing files. '
    'State is persistent across command calls. '
    'If `path` is a file, `view` displays the result of applying `cat -n`. '
    'If `path` is a directory, `view` lists non-hidden files and directories up to 2 levels deep. '
    'The `create` command cannot be used if the specified `path` already exists as a file. '
    'If a `command` generates a long output, it will be truncated and marked with `<response clipped>`. '
    'The `undo_edit` command will revert the last edit made to the file at `path`.\n\n'
    'Notes for using the `str_replace` command:\n'
    '* The `old_str` parameter should match EXACTLY one or more consecutive lines from the original file. '
    'Be mindful of whitespaces!\n'
    '* If the `old_str` parameter is not unique in the file, the replacement will not be performed. '
    'Make sure to include enough context in `old_str` to make it unique.\n'
    '* The `new_str` parameter should contain the edited lines that should replace the `old_str`.'
)


@tool_registry.register(
    'swe_agent/str_replace_editor',
    meta={
        'name': 'str_replace_editor',
        'description': STR_REPLACE_EDITOR_DESCRIPTION,
        'tool_type': ToolType.RUNTIME,
    },
)
async def str_replace_editor_tool(params: StrReplaceEditorInput, context: ToolContext) -> ToolResponse:
    """String replacement editor for file operations.

    Uses FileEditor from nexus.extensions.tools.file_editor for all file operations.
    The FileEditor handles:
    - File viewing with line numbers
    - Directory listing
    - File creation
    - String replacement with uniqueness checking
    - Line insertion
    - Undo functionality with history management
    """
    return await _run_str_replace_editor(params, context)


async def _run_str_replace_editor(params: StrReplaceEditorInput, context: ToolContext) -> ToolResponse:
    """Shared body behind every localized editor tool."""
    working_dir = get_working_dir(context)

    # Resolve path - ensure absolute path
    path_str = params.path
    if not path_str.startswith('/'):
        path_str = str(PurePosixPath(working_dir) / path_str)

    # Get the FileEditor instance for this working directory
    editor = _get_file_editor(working_dir)

    try:
        observation = await asyncio.to_thread(
            editor,
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


# ============== Submit Tool ==============

SUBMIT_REVIEW_MESSAGE = """Thank you for your work on this issue. Please carefully follow the steps below to help review your changes.

1. If you made any changes to your code after running the reproduction script, please run the reproduction script again.
   If the reproduction script is failing, please revisit your changes and make sure they are correct.
   If you have already removed your reproduction script, please ignore this step.
2. Remove your reproduction script (if you haven't done so already).
3. If you have modified any TEST files, please revert them to the state they had before you started fixing the issue.
   You can do this with `git checkout -- /path/to/test/file.py`. Use below <diff> to find the files you need to revert.
4. Run the submit command again to confirm.

Here is a list of all of your changes:

<diff>
{diff}
</diff>"""


class SubmitInput(BaseModel):
    """Input for submit command."""

    pass  # No arguments needed


REVIEW_SUBMIT_DESCRIPTION = (
    'Submit your changes as a git patch. '
    'The first call will show you a review of your changes. '
    'Call submit again to confirm and finalize your submission.'
)


@tool_registry.register(
    'swe_agent/review_submit',
    meta={
        'name': 'submit',
        'description': REVIEW_SUBMIT_DESCRIPTION,
        'tool_type': ToolType.RUNTIME,
    },
)
async def review_submit_tool(_params: SubmitInput, context: ToolContext) -> ToolResponse:
    """Submit the solution as a git patch with review stage.

    First call: Returns review message with diff, sets should_exit=False
    Second call: Actually submits, sets should_exit=True and done=True

    Args:
        _params: Submit input (no parameters needed)
        context: Tool context for state management
    """
    return await _run_review_submit(context, SUBMIT_REVIEW_MESSAGE)


async def _run_review_submit(context: ToolContext, review_message: str) -> ToolResponse:
    """Shared body behind every localized submit tool."""
    working_dir = get_working_dir(context)
    should_exit = context.tool_state.get('should_exit', False)

    try:
        # Stage all changes and create patch
        subprocess.run(
            'git add -A',
            shell=True,
            capture_output=True,
            cwd=working_dir,
        )

        subprocess.run(
            'git diff --cached > /root/model.patch',
            shell=True,
            cwd=working_dir,
        )

        with open('/root/model.patch') as f:
            patch = f.read()

        # First call: return review message, don't actually submit yet
        if not should_exit:
            review_output = review_message.format(diff=patch)
            return ToolResponse(
                output=_make_output(review_output),
                meta={
                    'stage': 'review',
                },
                context_updates={
                    'should_exit': True,  # Next call will actually submit
                },
            )

        # Report the submitted patch through the existing Nexus trace.
        span_id = get_current_span_id()
        if span_id:
            update_span(span_id, attributes={'hytuner.patch_text': patch or ''})

        # Second call: actually submit
        return ToolResponse(
            output=_make_output(''),
            context_updates={
                'exit_status': 'submitted',
            },
        )

    except Exception as e:
        return ToolResponse(
            output=_make_output(f'Error creating submission: {e}'),
            meta={'error': str(e)},
        )


# ============== Chinese Variant (ch_swe_agent) ==============
#
# `ch_swe_agent` runs the exact same tool bodies as `swe_agent`; only the text the
# model reads is translated. Each tool below therefore re-declares just its schema
# and description, then forwards to the shared body above. Fields left out of the
# Chinese input models inherit their English description.


class ChBashInput(BashInput):
    """Input for bash command execution."""

    command: str = Field(description='要执行的 bash 命令。')


@tool_registry.register(
    'ch_swe_agent/bash',
    meta={
        'name': 'bash',
        'description': BASH_DESCRIPTION,
        'tool_type': ToolType.RUNTIME,
    },
)
async def ch_bash_tool(params: ChBashInput, context: ToolContext) -> ToolResponse:
    """Execute a bash command in the runtime environment."""
    # Route the localized tool through the shared backend and state handling.
    return await bash_tool(params, context)


class ChStrReplaceEditorInput(StrReplaceEditorInput):
    """Input for the string replacement editor."""

    command: Literal['view', 'create', 'str_replace', 'insert', 'undo_edit'] = Field(
        description='要执行的编辑命令。允许的选项：`view`、`create`、`str_replace`、`insert`、`undo_edit`。'
    )
    path: str = Field(description='文件或目录的绝对路径，例如 `/testbed/file.py` 或 `/testbed`。')
    file_text: str | None = Field(
        default=None,
        description='`create` 命令必需参数，表示要创建文件的完整内容。',
    )
    old_str: str | None = Field(
        default=None,
        description='`str_replace` 命令必需参数，表示 `path` 中需要被替换的原始字符串。',
    )


@tool_registry.register(
    'ch_swe_agent/str_replace_editor',
    meta={
        'name': 'str_replace_editor',
        'description': STR_REPLACE_EDITOR_DESCRIPTION,
        'tool_type': ToolType.RUNTIME,
    },
)
async def ch_str_replace_editor_tool(params: ChStrReplaceEditorInput, context: ToolContext) -> ToolResponse:
    """String replacement editor for file operations."""
    return await _run_str_replace_editor(params, context)


CH_SUBMIT_REVIEW_MESSAGE = """感谢你处理这个问题。请仔细按下面步骤审查你的修改。

1. 如果你在运行复现脚本后又改动了代码，请重新运行复现脚本。
   如果复现脚本失败，请回看你的修改并确认它们是正确的。
   如果你已经删除了复现脚本，可以忽略这一步。
2. 删除复现脚本（如果还没有删除）。
3. 如果你修改了任何测试文件，请把它们恢复到开始修复问题之前的状态。
   你可以使用 `git checkout -- /path/to/test/file.py`。请根据下面的 <diff> 找到需要恢复的文件。
4. 再次运行 submit 命令进行确认。

下面是你全部修改的列表：

<diff>
{diff}
</diff>"""


@tool_registry.register(
    'ch_swe_agent/review_submit',
    meta={
        'name': 'submit',
        'description': REVIEW_SUBMIT_DESCRIPTION,
        'tool_type': ToolType.RUNTIME,
    },
)
async def ch_review_submit_tool(_params: SubmitInput, context: ToolContext) -> ToolResponse:
    """Submit the solution as a git patch with review stage."""
    return await _run_review_submit(context, CH_SUBMIT_REVIEW_MESSAGE)
