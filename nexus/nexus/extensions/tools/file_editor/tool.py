"""str_replace_editor tool — standalone registration of the file editor."""

import asyncio

from pathlib import PurePosixPath
from typing import Literal

from openai.types.chat import ChatCompletionContentPartParam
from pydantic import BaseModel, Field

from nexus.core import ToolContext, ToolResponse, ToolType, tool_registry
from nexus.extensions.tools.file_editor import FileEditor, ToolError


MAX_RESPONSE_LEN = 16000
TRUNCATED_MESSAGE = (
    '<response clipped><NOTE>To save on context only part of this file has been shown to you. '
    'You should retry this tool after you have searched inside the file with `grep -n` in order to find '
    'the line numbers of what you are looking for.</NOTE>'
)

_file_editors: dict[str, FileEditor] = {}


def _get_file_editor(working_dir: str) -> FileEditor:
    if working_dir not in _file_editors:
        _file_editors[working_dir] = FileEditor(workspace_root=working_dir)
    return _file_editors[working_dir]


def _maybe_truncate(content: str, max_length: int = MAX_RESPONSE_LEN) -> str:
    if len(content) <= max_length:
        return content
    return content[:max_length] + TRUNCATED_MESSAGE


def _make_output(text: str) -> list[ChatCompletionContentPartParam | dict]:
    return [{'type': 'text', 'text': text}]


def _get_working_dir(context: ToolContext) -> str:
    return context.tool_state.get('working_dir', '/testbed')


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


@tool_registry.register(
    'str_replace_editor',
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
    """String replacement editor for file operations."""
    working_dir = _get_working_dir(context)

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
        return ToolResponse(output=_make_output(f'Error: {e.message}'), meta={'error': str(e)})
    except Exception as e:
        return ToolResponse(output=_make_output(f'Error: {e}'), meta={'error': str(e)})
