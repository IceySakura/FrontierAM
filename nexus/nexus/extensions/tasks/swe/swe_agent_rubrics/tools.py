"""SWE-agent Rubrics tools for nexus framework.

Tools:
- bash: Execute bash commands in the runtime environment
- str_replace_editor: View, create, and edit files
- submit: Submit the rubrics.yaml file (with review stage)

State managed via context.tool_state:
- working_dir: Current working directory
- terminal_session_id: Terminal session ID for bash
- should_exit: Flag for two-stage submit
- exit_status: Set to 'submitted' on final submit
"""

import asyncio

from pathlib import PurePosixPath
from typing import Literal

from openai.types.chat import ChatCompletionContentPartParam
from pydantic import BaseModel, Field

from nexus.core import ToolContext, ToolResponse, ToolType, tool_registry
from nexus.extensions.tools.file_editor import FileEditor, ToolError
from nexus.runtime.executors.terminal import (
    CapturePaneRequest,
    CreateTerminalSessionRequest,
    RunCommandInSessionRequest,
    SendKeysToSessionRequest,
    capture_pane,
    create_terminal_session,
    run_command_in_session,
    send_keys_to_session,
)


# ============== Constants ==============

MAX_RESPONSE_LEN = 16000
TRUNCATED_MESSAGE = (
    '<response clipped><NOTE>To save on context only part of this file has been shown to you. '
    'You should retry this tool after you have searched inside the file with `grep -n` in order to find '
    'the line numbers of what you are looking for.</NOTE>'
)


def maybe_truncate(content: str, max_length: int = MAX_RESPONSE_LEN) -> str:
    if len(content) <= max_length:
        return content
    return content[:max_length] + TRUNCATED_MESSAGE


def get_working_dir(context: ToolContext) -> str:
    return context.tool_state.get('working_dir', '/testbed')


def _make_output(text: str) -> list[ChatCompletionContentPartParam | dict]:
    return [{'type': 'text', 'text': text}]


# ============== Bash Tool ==============


class BashInput(BaseModel):
    command: str = Field(description='The bash command to execute.')


@tool_registry.register(
    'swe_agent_rubrics/bash',
    meta={
        'name': 'bash',
        'description': (
            'Execute a bash command in the runtime environment. '
            'Use this to run shell commands, navigate directories, search files, '
            'and explore the codebase.'
        ),
        'tool_type': ToolType.RUNTIME,
    },
)
async def bash_tool(params: BashInput, context: ToolContext) -> ToolResponse:
    session_id = context.tool_state.get('terminal_session_id')
    context_updates = {}

    if session_id is None:
        create_response = await create_terminal_session(CreateTerminalSessionRequest())
        session_id = create_response.session_id

        await run_command_in_session(
            RunCommandInSessionRequest(
                session_id=session_id,
                command='export PAGER=cat MANPAGER=cat LESS=-R PIP_PROGRESS_BAR=off TQDM_DISABLE=1 GIT_PAGER=cat',
                timeout=300,
            )
        )

        working_dir = get_working_dir(context)
        await run_command_in_session(
            RunCommandInSessionRequest(
                session_id=session_id,
                command=f'cd {working_dir}',
                timeout=300,
            )
        )

        context_updates['terminal_session_id'] = session_id

    try:
        run_response = await run_command_in_session(
            RunCommandInSessionRequest(
                session_id=session_id,
                command=params.command,
                timeout=300,
            )
        )
        return ToolResponse(
            output=_make_output(run_response.output or ''),
            context_updates=context_updates,
            meta={
                'command': params.command,
                'exit_code': getattr(run_response, 'exit_code', None),
            },
        )

    except Exception as e:
        error_message = 'Error: ' + str(e)
        error_str = str(e).lower()

        # If command failed to start or a command is already running,
        # send Ctrl+C to recover the shell for subsequent commands.
        if 'failed to start' in error_str or 'already running' in error_str:
            try:
                await send_keys_to_session(SendKeysToSessionRequest(session_id=session_id, keys='C-c'))
                await asyncio.sleep(0.5)
            except Exception:
                pass

            try:
                pane_response = await capture_pane(CapturePaneRequest(session_id=session_id))
                error_message = (
                    'Your bash command failed to start. '
                    'A Ctrl+C has been sent to recover the shell. '
                    'Please fix the syntax errors and try again. Here is the captured shell content:\n'
                    + pane_response.content
                )
            except Exception:
                pass

        return ToolResponse(output=_make_output(error_message))


# ============== String Replace Editor Tool ==============

_file_editors: dict[str, FileEditor] = {}


def _get_file_editor(working_dir: str) -> FileEditor:
    if working_dir not in _file_editors:
        _file_editors[working_dir] = FileEditor(workspace_root=working_dir)
    return _file_editors[working_dir]


class StrReplaceEditorInput(BaseModel):
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


@tool_registry.register(
    'swe_agent_rubrics/str_replace_editor',
    meta={
        'name': 'str_replace_editor',
        'description': (
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
        ),
        'tool_type': ToolType.RUNTIME,
    },
)
async def str_replace_editor_tool(params: StrReplaceEditorInput, context: ToolContext) -> ToolResponse:
    working_dir = get_working_dir(context)

    path_str = params.path
    if not path_str.startswith('/'):
        path_str = str(PurePosixPath(working_dir) / path_str)

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

        output_text = observation.text

        if observation.image_base64 and observation.image_mime_type:
            return ToolResponse(
                output=[
                    {'type': 'text', 'text': output_text},
                    {
                        'type': 'image_url',
                        'image_url': {'url': f'data:{observation.image_mime_type};base64,{observation.image_base64}'},
                    },
                ],
                meta={'command': observation.command, 'path': observation.path},
            )

        return ToolResponse(
            output=_make_output(output_text),
            meta={'command': observation.command, 'path': observation.path},
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


class SubmitInput(BaseModel):
    pass


@tool_registry.register(
    'swe_agent_rubrics/review_submit',
    meta={
        'name': 'submit',
        'description': (
            'Submit your changes as a git patch. '
            'The first call will show you a review of your changes. '
            'Call submit again to confirm and finalize your submission.'
        ),
        'tool_type': ToolType.RUNTIME,
    },
)
async def review_submit_tool(_params: SubmitInput, context: ToolContext) -> ToolResponse:
    """Two-stage submit: first call shows review, second call confirms."""
    working_dir = get_working_dir(context)
    should_exit = context.tool_state.get('should_exit', False)
    rubrics_file = 'rubrics.yaml'
    rubrics_path = f'{working_dir}/{rubrics_file}'

    async def run_in_runtime(command: str) -> str:
        session_id = context.tool_state.get('terminal_session_id')
        if session_id is None:
            create_response = await create_terminal_session(CreateTerminalSessionRequest())
            session_id = create_response.session_id
            context.tool_state['terminal_session_id'] = session_id
            await run_command_in_session(
                RunCommandInSessionRequest(
                    session_id=session_id,
                    command='export PAGER=cat MANPAGER=cat',
                    timeout=60,
                )
            )
        result = await run_command_in_session(
            RunCommandInSessionRequest(session_id=session_id, command=command, timeout=300)
        )
        return result.output or ''

    try:
        check_output = await run_in_runtime(f'test -f {rubrics_path} && echo "exists" || echo "not_found"')
        if 'not_found' in check_output or 'exists' not in check_output:
            return ToolResponse(
                output=_make_output(
                    f'Error: {rubrics_file} not found at {rubrics_path}. '
                    'Please create the rubrics.yaml file first using str_replace_editor with command="create".'
                ),
                meta={'error': 'rubrics.yaml not found'},
            )

        if not should_exit:
            rubrics_content = await run_in_runtime(f'cat {rubrics_path}')
            review_output = (
                f'Thank you for writing the rubrics. Here is a review of your rubrics.yaml:\n\n'
                f'<rubrics_yaml>\n'
                f'{rubrics_content[:5000]}{"... (truncated)" if len(rubrics_content) > 5000 else ""}\n'
                f'</rubrics_yaml>\n\n'
                f'Please verify the rubrics are correct, then call submit again to confirm.'
            )
            return ToolResponse(
                output=_make_output(review_output),
                meta={'stage': 'review', 'rubrics_length': len(rubrics_content)},
                context_updates={'should_exit': True},
            )

        return ToolResponse(
            output=_make_output('Rubrics submitted successfully.'),
            context_updates={'exit_status': 'submitted'},
        )

    except Exception as e:
        return ToolResponse(
            output=_make_output(f'Error creating submission: {e}'),
            meta={'error': str(e)},
        )
