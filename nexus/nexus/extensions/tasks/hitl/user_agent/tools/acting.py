"""Tools whose body does the whole job, acting where the worker is working.

``read`` and ``bash`` so far. A call here has already happened by the time it
returns. A handler over one of these observes rather than executes. What may live
here is not limited to checking: a tool from elsewhere in the tree belongs here too,
and ``verification`` is the usage these two declare rather than the boundary of the
module.

Both declare one usage and share one usage doc.

``bash`` runs in a persistent terminal session inside the runtime, created on first
use and remembered in ``tool_state``.

``read`` is ``str_replace_editor``'s ``view`` without the four commands that write.
Its extras are declared host-side in this module's ``pyproject.toml``. ``FileEditor``
raises on construction without ``cachetools``, before any path is read.
"""

import asyncio
import logging
import shlex

from collections.abc import Sequence
from pathlib import Path, PurePosixPath

from pydantic import BaseModel, Field

from nexus.core import ToolContext, ToolResponse, ToolType, tool_registry
from nexus.extensions.tasks.hitl.user_agent.tools.usage_docs import USAGE, USAGE_DOC
from nexus.extensions.tasks.hitl.utils.text import text_parts, truncate
from nexus.extensions.tools.file_editor.editor import FileEditor
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


logger = logging.getLogger(__name__)


READ_TOOL = 'hitl/user/read'
BASH_TOOL = 'hitl/user/bash'

WORKING_DIR_STATE = 'working_dir'
SESSION_STATE = 'terminal_session_id'
OUTPUT_LIMIT = 20_000

# The longest a single command may be waited on.
TIMEOUT_CEILING = 1800

VERIFICATION_DOC = """\
## Checking for yourself

You have full read, write and command access to the working directory, the same place
the worker is working.

You are the person who asked for the work, not the one building it. Do not survey the
project, do not re-check what you have already checked, and do not start a heavy job
such as a full build or a whole test suite unless nothing narrower answers you.\
"""

# Schema names of the tools a user checks something with. What each one does reaches
# the model as that tool's own ``description``.
VERIFYING_TOOLS: frozenset[str] = frozenset({'bash', 'read', 'str_replace_editor'})


def can_verify(names: Sequence[str]) -> bool:
    """Whether any of ``names`` is a tool a user checks something with."""
    return any(name in VERIFYING_TOOLS for name in names)


class ReadInput(BaseModel):
    path: str = Field(
        description='Path to file or directory, e.g. `/testbed/file.py` or `/testbed`. '
        'Relative paths resolve against the working directory.'
    )
    view_range: list[int] | None = Field(
        default=None,
        description=(
            'Optional parameter when `path` points to a file. If none is given, the full file is '
            'shown. If provided, the file will be shown in the indicated line number range, e.g. '
            '[11, 12] will show lines 11 and 12. Indexing at 1 to start. Setting `[start_line, -1]` '
            'shows all lines from `start_line` to the end.'
        ),
    )


class BashInput(BaseModel):
    command: str = Field(description='The bash command to execute.')
    timeout: float = Field(
        default=300.0,
        gt=0,
        le=TIMEOUT_CEILING,
        description='Seconds to wait before giving up.',
    )


def working_dir(context: ToolContext) -> str:
    """Where a command runs, from ``tool_state``. Refused rather than defaulted."""
    value = context.tool_state.get(WORKING_DIR_STATE)
    if not isinstance(value, str) or not value:
        raise ValueError(
            f'{WORKING_DIR_STATE} is missing from tool_state. There is nowhere to run this. '
            'The agent sets it with toolset.tool_state(tools, working_dir=...).'
        )
    return value


def resolve_path(path: str, context: ToolContext) -> str:
    if path.startswith('/'):
        return path
    return str(PurePosixPath(working_dir(context)) / path)


# One editor per working directory, matching how the editor tool itself caches them
# (``tools/file_editor/tool.py``). Viewing keeps no state. This only avoids rebuilding
# the encoding cache on every call.
_editors: dict[str, FileEditor] = {}


def file_editor(directory: str) -> FileEditor:
    """The viewer for ``directory``, built on first use."""
    editor = _editors.get(directory)
    if editor is None:
        editor = FileEditor()
        _editors[directory] = editor
    return editor


async def ensure_session(context: ToolContext) -> tuple[str, dict[str, str]]:
    """The terminal session for this run, created on first use.

    Returns the id and the ``context_updates`` that remember it. Every later call
    shares the session rather than opening one shell per command.
    """
    session_id = context.tool_state.get(SESSION_STATE)
    if isinstance(session_id, str) and session_id:
        return session_id, {}

    created = await create_terminal_session(CreateTerminalSessionRequest())
    session_id = created.session_id
    await run_command_in_session(
        RunCommandInSessionRequest(
            session_id=session_id,
            command='export PAGER=cat MANPAGER=cat LESS=-R PIP_PROGRESS_BAR=off TQDM_DISABLE=1',
            timeout=300,
        )
    )
    await run_command_in_session(
        RunCommandInSessionRequest(
            session_id=session_id,
            command=f'cd {shlex.quote(working_dir(context))}',
            timeout=300,
        )
    )
    return session_id, {SESSION_STATE: session_id}


@tool_registry.register(
    READ_TOOL,
    meta={
        'name': 'read',
        'description': (
            'Custom tool for viewing files and directories. '
            'If `path` is a file, `view` displays the result of applying `cat -n`. '
            'If `path` is a directory, `view` lists non-hidden files and directories up to 2 levels deep. '
            'An image file is shown as the image itself. '
            'If the output is long, it will be truncated and marked with `<response clipped>`.'
        ),
        'tool_type': ToolType.RUNTIME,
        USAGE: 'verification',
        USAGE_DOC: VERIFICATION_DOC,
    },
)
async def read_tool(params: ReadInput, context: ToolContext) -> ToolResponse:
    """View a file, a directory, or an image, through the shared file editor.

    The same ``view`` the editor tool offers, without its four writing commands.
    """
    path = resolve_path(params.path, context)

    try:
        observation = await asyncio.to_thread(
            file_editor(working_dir(context)).view,
            Path(path),
            params.view_range,
        )
    except Exception as exc:  # noqa: BLE001 - the simulator must see the failure
        # ``view`` raises for a missing path and for a range past the end of the file.
        # Both are answered rather than raised.
        logger.info('user view of %s failed: %s', path, exc)
        return ToolResponse(output=text_parts(f'Cannot view {path}: {exc}'))

    text = truncate(observation.text or '', OUTPUT_LIMIT)
    if observation.image_base64 and observation.image_mime_type:
        # Not truncated. A cut base64 string is not an image. It reaches the model as
        # an image part.
        return ToolResponse(
            output=[
                {'type': 'text', 'text': text},
                {
                    'type': 'image_url',
                    'image_url': {'url': f'data:{observation.image_mime_type};base64,{observation.image_base64}'},
                },
            ],
        )
    return ToolResponse(output=text_parts(text or f'Nothing to show for {path}'))


@tool_registry.register(
    BASH_TOOL,
    meta={
        'name': 'bash',
        'description': (
            'Execute a bash command in the runtime environment. '
            'Use this to run shell commands, navigate directories, search files, '
            'run tests, and execute Python scripts. '
            'Commands start in the working directory. '
            'One shell is kept across calls. A directory you change into stays changed.'
        ),
        'tool_type': ToolType.RUNTIME,
        USAGE: 'verification',
        USAGE_DOC: VERIFICATION_DOC,
    },
)
async def bash_tool(params: BashInput, context: ToolContext) -> ToolResponse:
    session_id, updates = await ensure_session(context)

    try:
        result = await run_command_in_session(
            RunCommandInSessionRequest(session_id=session_id, command=params.command, timeout=params.timeout)
        )
    except Exception as exc:  # noqa: BLE001 - the simulator must see the failure
        message = await _recover_shell(session_id, exc)
        return ToolResponse(output=text_parts(message), context_updates=updates)

    return ToolResponse(
        output=text_parts(truncate(result.output or '', OUTPUT_LIMIT)),
        context_updates=updates,
        meta={'command': params.command},
    )


async def _recover_shell(session_id: str, exc: Exception) -> str:
    """Free a shell left mid-command, and report what it was showing.

    A command whose syntax leaves the shell waiting for more input blocks every
    later call. The failure has to be cleared rather than only reported.
    """
    detail = str(exc)
    if 'Failed to start' not in detail and 'already running' not in detail:
        return f'Error: {detail}'

    try:
        await send_keys_to_session(SendKeysToSessionRequest(session_id=session_id, keys='C-c'))
    except Exception as interrupt_failure:  # noqa: BLE001 - report the original problem regardless
        logger.warning('could not interrupt session %s: %s', session_id, interrupt_failure)
        return f'Error: {detail}'

    try:
        pane = await capture_pane(CapturePaneRequest(session_id=session_id))
    except Exception as capture_failure:  # noqa: BLE001 - same
        logger.warning('could not capture session %s: %s', session_id, capture_failure)
        return f'Error: {detail} (the shell was interrupted and should work again)'

    return (
        'Your command did not start. A Ctrl+C was sent and the shell should work again. '
        f'Fix the command and retry. The shell was showing:\n{truncate(pane.content or "", OUTPUT_LIMIT)}'
    )


__all__ = [
    'BASH_TOOL',
    'OUTPUT_LIMIT',
    'READ_TOOL',
    'SESSION_STATE',
    'VERIFICATION_DOC',
    'VERIFYING_TOOLS',
    'WORKING_DIR_STATE',
    'BashInput',
    'ReadInput',
    'can_verify',
    'ensure_session',
    'resolve_path',
    'working_dir',
]
