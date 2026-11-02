"""Self-contained tools for code_mt_task.

Provides bash, str_replace_editor, submit, switch_runtime_image.

code_mt must not import anything from sibling task/agent packages (swe_agent,
swe_setup_agent, terminal, swe). These tools intentionally duplicate the small
amount of logic needed so code_mt is fully self-contained.
"""

import asyncio
import logging
import re

from pathlib import PurePosixPath
from typing import Literal

from openai.types.chat import ChatCompletionContentPartParam
from pydantic import BaseModel, Field

from nexus.core import ToolContext, ToolResponse, ToolType, require_context, tool_registry
from nexus.extensions.projects.code_mt.agent.images import (
    KNOWN_IMAGES,
    image_alias,
    resolve_mirror_image,
    validate_image,
)
from nexus.extensions.tools.file_editor import FileEditor, ToolError
from nexus.runtime.executors import (
    CapturePaneRequest,
    CreateTerminalSessionRequest,
    RunCommandInSessionRequest,
    SendKeysToSessionRequest,
)


logger = logging.getLogger(__name__)


MAX_RESPONSE_LEN = 16000
TRUNCATED_MESSAGE = (
    '<response clipped><NOTE>To save on context only part of this file has been shown to you. '
    'You should retry this tool after you have searched inside the file with `grep -n` in order to find '
    'the line numbers of what you are looking for.</NOTE>'
)


def _maybe_truncate(content: str, max_length: int = MAX_RESPONSE_LEN) -> str:
    if len(content) <= max_length:
        return content
    return content[:max_length] + TRUNCATED_MESSAGE


def _get_working_dir(context: ToolContext) -> str:
    return context.tool_state.get('working_dir', '/testbed')


def _get_bash_timeout(context: ToolContext) -> int:
    """Per-command bash hard-kill timeout (seconds).

    Seeded by tool_call_loop from CodeMTAgentInput.bash_timeout and shipped to
    the sandbox inside context.tool_state via runtime.call_tool. Defaults to 300
    for backward compatibility with images/configs that don't set it.
    """
    return context.tool_state.get('bash_timeout', 300)


def _make_output(text: str) -> list[ChatCompletionContentPartParam | dict]:
    return [{'type': 'text', 'text': text}]


# ============== Bash Tool ==============


class BashInput(BaseModel):
    command: str = Field(description='The bash command to execute.')


@tool_registry.register(
    'code_mt/bash',
    meta={
        'name': 'bash',
        'description': (
            'Execute a bash command in the runtime environment. '
            'Use this to run shell commands, navigate directories, search files, '
            'run tests, and execute Python scripts.'
        ),
        'tool_type': ToolType.RUNTIME,
    },
)
async def bash_tool(params: BashInput, context: ToolContext) -> ToolResponse:
    """Execute a bash command in a persistent tmux session inside the runtime."""
    from nexus.runtime.executors.terminal import (
        capture_pane,
        create_terminal_session,
        run_command_in_session,
        send_keys_to_session,
    )

    session_id = context.tool_state.get('terminal_session_id')
    context_updates: dict = {}

    if session_id is None:
        create_response = await create_terminal_session(CreateTerminalSessionRequest())
        session_id = create_response.session_id

        await run_command_in_session(
            RunCommandInSessionRequest(
                session_id=session_id,
                command='export PAGER=cat MANPAGER=cat LESS=-R PIP_PROGRESS_BAR=off TQDM_DISABLE=1',
                timeout=300,
            )
        )
        working_dir = _get_working_dir(context)
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
                timeout=_get_bash_timeout(context),
            )
        )
        # Return the RAW bash output, mirroring swe_agent/tools.py:137
        # (`_make_output(run_response.output)` — NO in-tool truncation). swe
        # defers all clipping to the agent layer's head+tail maybe_truncate at
        # DEFAULT_TEXT_CONTENT_LIMIT=50000, which code_mt's tool_call_loop
        # `_truncate_observation` already mirrors byte-for-byte. The old
        # in-tool `_maybe_truncate` clipped bash HEAD-ONLY at 16000 with a
        # file-oriented notice ("only part of this file…"), which both diverged
        # from swe and was wrong for command output. Aligned to swe.
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

        if 'Failed to start' in error_str or 'already running' in error_str:
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
                    'Please fix the syntax errors and try again. This can be the result '
                    'of not adhering to the syntax for multi-line commands. Here is the captured shell content:\n'
                    + pane_response.content
                )
            except Exception:
                pass

        return ToolResponse(output=_make_output(error_message))


# ============== str_replace_editor Tool ==============

# Process-wide cache keyed by (run_id, working_dir). Keyed by run_id so concurrent
# tasks within the same worker process don't share undo-stack / state. (Old code
# keyed only by working_dir and could leak FileEditor state across tasks.)
_file_editors: dict[str, FileEditor] = {}


def _get_file_editor(context: ToolContext) -> FileEditor:
    working_dir = _get_working_dir(context)
    run_id = context.run_id or 'default'
    key = f'{run_id}:{working_dir}'
    if key not in _file_editors:
        _file_editors[key] = FileEditor(workspace_root=working_dir)
    return _file_editors[key]


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
    'code_mt/str_replace_editor',
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
    working_dir = _get_working_dir(context)

    path_str = params.path
    if not path_str.startswith('/'):
        path_str = str(PurePosixPath(working_dir) / path_str)

    editor = _get_file_editor(context)

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

        if observation.image_base64 and observation.image_mime_type:
            return ToolResponse(
                output=[
                    {'type': 'text', 'text': observation.text},
                    {
                        'type': 'image_url',
                        'image_url': {'url': f'data:{observation.image_mime_type};base64,{observation.image_base64}'},
                    },
                ],
                meta={'command': observation.command, 'path': observation.path},
            )
        return ToolResponse(
            output=_make_output(observation.text),
            meta={'command': observation.command, 'path': observation.path},
        )
    except ToolError as e:
        return ToolResponse(output=_make_output(f'Error: {e.message}'), meta={'error': str(e)})
    except Exception as e:
        return ToolResponse(output=_make_output(f'Error: {e}'), meta={'error': str(e)})


# ============== Submit Tool (single-call; used by multi-turn scenarios) ==============


class SubmitInput(BaseModel):
    explanation: str = Field(default='', description='Brief summary of what was done.')


@tool_registry.register(
    'code_mt/submit',
    meta={
        'name': 'submit',
        'description': (
            'Submit your work for evaluation. Call this when you have completed the task. '
            'The system will run automated verification and return results. '
            'If the submission fails verification, you may receive feedback and get another attempt.'
        ),
        'tool_type': ToolType.LOCAL,
    },
)
async def submit(params: SubmitInput, context: ToolContext) -> ToolResponse:
    return ToolResponse(
        output=_make_output('Submission received. Your work will be evaluated.'),
        context_updates={'exit_status': 'submitted'},
    )


# ============== Submit Answer Tool (QA-style; carries the final answer) ==============
#
# For tasks whose deliverable is a natural-language ANSWER rather than a code
# change (e.g. RepoQA), the answer never touches the container filesystem, so no
# grader/tests can surface it. This tool lets the model submit that answer
# explicitly: the `answer` string is stashed on ctx.tool_state['answer'] via
# context_updates, and code_mt scoring.extract_agent_trajectory reads it with
# priority (falling back to the last assistant message when absent). It also sets
# exit_status='submitted' so the loop terminates like submit. Kept separate from
# `submit` so the plain submit tool is never polluted with an answer field.


class SubmitAnswerInput(BaseModel):
    answer: str = Field(
        description=(
            'Your final answer to the question. Provide the complete answer text '
            'here. Call this tool once you have finished investigating and are '
            'ready to submit; the answer you pass is what will be evaluated.'
        ),
    )


@tool_registry.register(
    'code_mt/submit_answer',
    meta={
        'name': 'submit_answer',
        'description': (
            'Submit your final answer for evaluation. Use this when the task asks a '
            'question and expects a written answer (rather than a code change): pass '
            'your complete answer in the `answer` field. Call this when you are done; '
            'the submitted answer is what gets graded.'
        ),
        'tool_type': ToolType.LOCAL,
    },
)
async def submit_answer(params: SubmitAnswerInput, context: ToolContext) -> ToolResponse:
    return ToolResponse(
        output=_make_output('Answer received. Your submission will be evaluated.'),
        context_updates={'exit_status': 'submitted', 'answer': params.answer},
    )


# ============== Submit Patch Tool (two-stage git-diff review; SWE-style) ==============
#
# Mirrors swe_agent/review_submit word-for-word (message + mechanism), but is
# fully self-contained inside code_mt so we never import from swe_agent.
# Exposed to the model as `submit` (same schema name) — the tool_call_loop
# forwards any tool named `submit` to whichever submit tool the dataset
# registered in `tools`. Two-stage semantics are enforced by this tool alone
# via `context.tool_state['should_exit']`, independent of the agent-input
# `two_stage_submit` flag (that flag governs the plain code_mt/submit tool).

SUBMIT_PATCH_REVIEW_MESSAGE = """Thank you for your work on this issue. Please carefully follow the steps below to help review your changes.

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


class SubmitPatchInput(BaseModel):
    pass


@tool_registry.register(
    'code_mt/submit_patch',
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
async def submit_patch(_params: SubmitPatchInput, context: ToolContext) -> ToolResponse:
    """Submit the solution as a git patch with review stage.

    First call: `git add -A && git diff --cached > /root/model.patch`, return
    the review message with the diff inline, and set should_exit=True so the
    next call finalizes.
    Second call: return empty output with exit_status='submitted'.
    """
    import subprocess

    working_dir = _get_working_dir(context)
    should_exit = context.tool_state.get('should_exit', False)

    try:
        subprocess.run('git add -A', shell=True, capture_output=True, cwd=working_dir)
        subprocess.run('git diff --cached > /root/model.patch', shell=True, cwd=working_dir)
        with open('/root/model.patch') as f:
            patch = f.read()

        if not should_exit:
            review_output = SUBMIT_PATCH_REVIEW_MESSAGE.format(diff=patch)
            return ToolResponse(
                output=_make_output(review_output),
                meta={'stage': 'review'},
                context_updates={'should_exit': True},
            )

        return ToolResponse(
            output=_make_output(''),
            context_updates={'exit_status': 'submitted'},
        )
    except Exception as e:
        return ToolResponse(
            output=_make_output(f'Error creating submission: {e}'),
            meta={'error': str(e)},
        )


# ============== switch_runtime_image Tool ==============


def _canonicalize_switch_alias(raw: str) -> str:
    """Map a conventional toolchain image name onto a code_mt KNOWN_IMAGES alias.

    The model naturally emits standard docker names (``python:3.9-slim``,
    ``golang:1.20``, ``openjdk:21-slim``, ``dotnet-sdk:9.0``, ``ubuntu-c:22.04``…)
    but the mirror registry only knows the idiosyncratic aliases in KNOWN_IMAGES
    (``python-git:3.9-slim``, ``golang-local:1.20``, ``java-git:21`` …). Without
    this, validate_image() rejected ~100% of real switch requests.

    Idempotent on names that are already valid aliases. Internal mirror URLs and
    names we can't confidently map are returned unchanged (resolve_mirror_image
    passes full URLs through; unknown short names are rejected with a hint list).
    Scoped to switch_runtime_image only — does not touch the shared helpers.
    """
    name = (raw or '').strip()
    low = name.lower()
    if not low:
        return name
    # Already a fully-qualified internal mirror URL -> leave for resolve_mirror_image.
    if 'swe_builder_eval_base_' in low or low.startswith('mirrors.tencent.com'):
        return name

    seg = low.split('/')[-1]  # drop external host (mcr.microsoft.com/, docker.io/)
    repo = seg.split(':', 1)[0]
    tag = seg.split(':', 1)[1] if ':' in seg else ''
    vm = re.search(r'(\d+(?:\.\d+)?)', tag) or re.search(r'(\d+(?:\.\d+)?)', repo)
    ver = vm.group(1) if vm else ''
    major = ver.split('.')[0] if ver else ''
    mm = re.match(r'(\d+\.\d+)', ver)
    mv = mm.group(1) if mm else ''

    def has(*keys: str) -> bool:
        return any(k in repo or k in tag for k in keys)

    # Java: JDK version can be buried among other numbers (e.g.
    # ``maven:3.9-eclipse-temurin-17`` -> 17). Prefer a valid java major.
    if (
        has('openjdk', 'temurin', 'corretto', 'zulu', 'jdk', 'java')
        or 'temurin' in low
        or 'openjdk' in low
        or repo in ('maven', 'gradle')
    ):
        nums = re.findall(r'\d+', seg)
        for jv in ('21', '17', '11', '8'):
            if jv in nums:
                return f'java-git:{jv}'
        return f'java-git:{major}' if major else name
    if has('dotnet', 'aspnet') or 'dotnet' in low or 'aspnet' in low:
        return f'dotnet-git:{ver}' if ver else name
    if repo in ('go', 'golang') or has('golang'):
        return f'golang-local:{mv}' if mv else name
    if has('rust', 'cargo'):
        return f'rust-git:{mv}-slim' if mv else 'rust-git:latest'
    if has('node'):
        if major.isdigit():
            return f'node-git:{min(int(major), 22)}-bullseye'
        return name
    if has('python', 'cpython'):
        return f'python-git:{mv}-slim' if mv else name
    if has('php'):
        return f'php-git:{mv}-cli' if mv else name
    if has('ruby'):
        return f'ruby-git:{mv}-slim' if mv else name
    if repo in ('ubuntu-c', 'ubuntu-cpp') or has('gcc', 'g++', 'clang', 'cpp', 'c++'):
        return f'ubuntu-cpp:{ver}' if ver in ('20.04', '22.04') else 'ubuntu-cpp:22.04'
    if repo in ('ubuntu', 'ubuntu-git'):
        return f'ubuntu-git:{ver}' if ver in ('20.04', '22.04') else name
    return name


def _switch_available_images_msg() -> str:
    """Compact, grouped listing of valid switch aliases for the rejection hint."""
    groups: dict[str, list[str]] = {}
    for img in KNOWN_IMAGES:
        fam, _, tag = img.partition(':')
        groups.setdefault(fam, []).append(tag)
    lines = []
    for fam in sorted(groups):
        lines.append(f'  {fam}: ' + ', '.join(sorted(groups[fam])))
    return '\n'.join(lines)


class SwitchRuntimeImageInput(BaseModel):
    new_image: str = Field(description='New base image name (e.g., python-git:3.11-slim, node-git:18-bullseye)')
    reason: str = Field(default='', description='Reason for switching')


@tool_registry.register(
    'code_mt/switch_runtime_image',
    meta={
        'name': 'switch_runtime_image',
        'description': (
            'Switch runtime to a new base image when current image cannot meet requirements.\n\n'
            'WARNING:\n'
            '- Destroys current runtime container\n'
            '- Conversation history is preserved (in host process)\n'
            '- Testbed will be reinitialized via setup_commands\n\n'
            'Use this when:\n'
            '- Current base image lacks essential system dependencies\n'
            '- Need a language-specific image with pre-installed environment\n\n'
            'Max switches: 3'
        ),
        'tool_type': ToolType.LOCAL,
    },
)
async def switch_runtime_image(params: SwitchRuntimeImageInput, context: ToolContext) -> ToolResponse:
    ctx = require_context()

    max_switches = ctx.task_config.get('max_base_image_switches', 3)
    switch_count = ctx.tool_state.get('base_image_switch_count', 0)

    if switch_count >= max_switches:
        return ToolResponse(
            output=_make_output(
                f'Reached maximum base image switch limit ({max_switches}). '
                f'Consider using ubuntu-git:22.04 and manually install required tools.'
            ),
            meta={'success': False, 'error': 'max_switches_reached'},
        )

    old_image = ctx.task_config.get('base_image', 'unknown')

    # Map conventional docker names (python:3.9-slim, openjdk:21, dotnet-sdk:9.0,
    # golang:1.20, ubuntu-c:22.04 …) onto the registry's KNOWN_IMAGES aliases
    # before the existence check. A full internal mirror URL passes through.
    canonical_image = _canonicalize_switch_alias(params.new_image)
    is_full_url = (
        canonical_image.lower().startswith('mirrors.tencent.com') or 'swe_builder_eval_base_' in canonical_image.lower()
    )

    if not is_full_url and not validate_image(canonical_image):
        logger.warning(
            '[SWITCH_RUNTIME] Image %r (canonical %r) not in known registry, rejecting switch',
            params.new_image,
            canonical_image,
        )
        return ToolResponse(
            output=_make_output(
                f'Image "{params.new_image}" is not available in the registry. '
                f'Current image kept: {image_alias(old_image)}\n\n'
                f'Pick one of the available images (use the exact alias) and call '
                f'switch_runtime_image again:\n{_switch_available_images_msg()}'
            ),
            meta={'success': False, 'error': 'invalid_image', 'requested': params.new_image},
        )

    resolved_image = resolve_mirror_image(canonical_image, old_image)

    logger.debug('=' * 80)
    logger.debug('[SWITCH_RUNTIME] Switching base image')
    logger.debug('[SWITCH_RUNTIME]   Requested: %s  ->  canonical: %s', params.new_image, canonical_image)
    logger.debug('[SWITCH_RUNTIME]   Old: %s', old_image)
    logger.debug('[SWITCH_RUNTIME]   New (resolved):  %s', resolved_image)
    logger.debug('[SWITCH_RUNTIME]   Reason: %s', params.reason)
    logger.debug('[SWITCH_RUNTIME]   Remaining: %d', max_switches - switch_count - 1)
    logger.debug('=' * 80)

    new_runtime = await ctx.replace_runtime('main', image=resolved_image)

    ctx.tool_state['base_image_switch_count'] = switch_count + 1
    ctx.task_config['base_image'] = resolved_image
    ctx.tool_state.pop('terminal_session_id', None)

    # Re-run setup_commands on the new runtime so /testbed is reinitialized.
    # Shared with the turn-0 auto-setup hook (autosetup._check_and_switch_image)
    # so both switch paths repopulate /testbed identically and never drift.
    from nexus.extensions.projects.code_mt.agent.autosetup import (
        TestbedInitError,
        reinit_testbed_via_setup_commands,
    )

    try:
        await reinit_testbed_via_setup_commands(new_runtime, ctx.task_config, log_tag='SWITCH_RUNTIME')
    except TestbedInitError as e:
        # The switch happened but /testbed could not be repopulated. Unlike the
        # turn-0 path (which fails the episode), this is an explicit mid-episode
        # agent action: surface the failure as a tool error so the agent knows
        # the repo is missing and can recover manually (it has git available).
        # Do NOT report success — that would hide the empty /testbed.
        logger.error('[SWITCH_RUNTIME] reinit failed after switch: %s', e)
        return ToolResponse(
            output=_make_output(
                f'Runtime switched to {image_alias(resolved_image)}, but /testbed could NOT be '
                f'reinitialized (repo checkout failed): {e}\n\n'
                f'The working directory is likely empty. Re-clone the repo manually before '
                f'continuing (e.g. git init && git fetch --depth=1 origin <base_commit> && '
                f'git checkout --force FETCH_HEAD).'
            ),
            meta={'success': False, 'error': 'testbed_reinit_failed', 'new_image': resolved_image},
        )

    message = (
        f'Runtime switched to {image_alias(resolved_image)}. '
        f'Testbed reinitialized via setup_commands. '
        f'Remaining switches: {max_switches - switch_count - 1}'
    )
    return ToolResponse(
        output=_make_output(message),
        meta={
            'success': True,
            'old_image': old_image,
            'new_image': resolved_image,
            'remaining_switches': max_switches - switch_count - 1,
        },
    )
