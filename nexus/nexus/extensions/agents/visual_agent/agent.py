"""Resolve visual inputs and delegate inference and tool execution to main_agent."""

import copy
import logging
import time

from contextlib import nullcontext
from typing import Annotated, Any, Literal

from openai.types.chat import ChatCompletionContentPartParam
from pydantic import Field, field_validator

from nexus.core import (
    agent_registry,
    require_context,
)
from nexus.core.agent import validate_history
from nexus.extensions.agents.main.agent import (
    MainAgentInput,
    MainAgentOutput,
    run_main_agent,
)
from nexus.extensions.agents.visual_agent.resources.images import resolve_user_prompt
from nexus.extensions.agents.visual_agent.resources.tool_images import HOOK_PATH, image_hook_context
from nexus.extensions.agents.visual_agent.resources.workspace import (
    WithWorkspace,
    upload_workspace,
)
from nexus.extensions.agents.visual_agent.tools import (
    DEFAULT_VISUAL_TOOLS,
)
from nexus.extensions.compaction.runtime_state import is_compaction_runtime_state_message


logger = logging.getLogger(__name__)


class VisualAgentInput(MainAgentInput, WithWorkspace):
    """Main-agent configuration with visual input handling and tool defaults."""

    model_config = {'extra': 'forbid'}

    # Keep input-only file/video fields until media resolution.
    user_prompt: (
        str | list[Annotated[dict[str, Any] | ChatCompletionContentPartParam, Field(union_mode='left_to_right')]]
    )
    max_iterations: int = Field(default=30, gt=0, description='Maximum model/tool iterations.')
    tools: list[str] = Field(
        default_factory=lambda: list(DEFAULT_VISUAL_TOOLS),
        description='Registered tools; an empty list or null uses the default code interpreter.',
    )
    parallel_tool_calls: bool = Field(default=True, description='Execute independent tool calls concurrently.')
    input_image_paths: bool = Field(default=True, description='Include uploaded image paths in the prompt.')
    input_image_path_position: Literal['before', 'after'] = Field(
        default='before', description='Place the uploaded image path note before or after its input image.'
    )
    inline_images: bool = Field(default=True, description='Inline image bytes; otherwise use uploaded paths.')
    upload_images_to_runtime: bool = Field(
        default=True, description='Upload supported media when a runtime is configured.'
    )
    show_image_dimensions: bool = Field(
        default=False, description='Annotate input and tool images with raw pixel sizes.'
    )
    tool_error_continue: bool = Field(
        default=True,
        description='Return tool errors to the model as observations and keep going, rather '
        'than aborting the run. Visual exploration involves trial-and-error image code.',
    )

    @field_validator('tools', mode='before')
    @classmethod
    def default_tools(cls, value: Any) -> Any:
        return list(DEFAULT_VISUAL_TOOLS) if value is None else value

    @field_validator('history')
    @classmethod
    def validate_conversation(cls, history: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        validate_history(history)
        return history


class VisualAgentOutput(MainAgentOutput):
    """Main-agent result plus the prepared workspace directory."""

    workspace: str = '/tmp'
    generated_message_start: int | None = Field(
        default=None, ge=0, description='Start of this visual run in main_trajectory; unavailable after compaction.'
    )
    trajectory_compacted: bool = Field(
        default=False, description='Whether context compaction occurred during this visual run.'
    )


async def _resolve_user_prompt(
    agent_input: VisualAgentInput,
    user_prompt: str | list[Any],
) -> str | list[Any]:
    """Resolve media parts before MainAgentInput validates the backend prompt."""
    return await resolve_user_prompt(
        user_prompt,
        input_image_paths=agent_input.input_image_paths,
        input_image_path_position=agent_input.input_image_path_position,
        inline_images=agent_input.inline_images,
        upload_to_runtime=agent_input.upload_images_to_runtime,
        show_image_dimensions=agent_input.show_image_dimensions,
    )


def _build_main_input(
    agent_input: VisualAgentInput,
    user_prompt: str | list[ChatCompletionContentPartParam],
    history: list[dict[str, Any]] | None,
) -> MainAgentInput:
    """Build a :class:`MainAgentInput` from the visual-shaped input."""
    # getattr also preserves excluded runtime-only fields such as openai_client_key.
    values = {name: getattr(agent_input, name) for name in MainAgentInput.model_fields}
    values.update(
        user_prompt=user_prompt,
        history=history,
        tools=list(agent_input.tools or DEFAULT_VISUAL_TOOLS),
        post_iteration_hook=HOOK_PATH if agent_input.show_image_dimensions else agent_input.post_iteration_hook,
    )
    return MainAgentInput(**values)


def _count_prompt_images(user_prompt: str | list[ChatCompletionContentPartParam]) -> int:
    """Count image content parts in a user prompt, for logging only."""
    if isinstance(user_prompt, str):
        return 0
    return sum(1 for part in user_prompt if isinstance(part, dict) and part.get('type') in {'image_url', 'image'})


def _count_prompt_videos(user_prompt: str | list[Any]) -> int:
    """Count video content parts in a user prompt, for logging only."""
    if isinstance(user_prompt, str):
        return 0
    return sum(1 for part in user_prompt if isinstance(part, dict) and part.get('type') in {'video_url', 'video'})


@agent_registry.register('visual_agent', meta={'gateway': {'scaffold_policy': 'nexus_main_agent'}})
async def run_visual_agent(agent_input: VisualAgentInput) -> VisualAgentOutput:
    """Visual agent entry point.

    Ingests ``images`` into a multimodal prompt (uploading local files to the
    sandbox), builds a :class:`MainAgentInput` with visual-reasoning defaults
    (prompt, tool set, image budget), and delegates to :func:`run_main_agent`.
    The trajectory, exit status and final answer are forwarded back unmodified.
    """
    t0 = time.monotonic()
    prompt = copy.deepcopy(agent_input.user_prompt)
    history = copy.deepcopy(agent_input.history)
    if agent_input.upload_workspace is not None:
        runtime = await require_context().get_runtime('main')
        await upload_workspace(runtime, agent_input.upload_workspace, agent_input.workspace)
    # Counted before resolution: it rewrites videos into text notes, so the
    # original input is the only place the video count still exists.
    n_videos = _count_prompt_videos(prompt)
    if n_videos:
        logger.warning(
            '[visual_agent] Video support is WIP (work in progress): '
            'video ingestion and sandbox-based frame reading are experimental and may change.'
        )
    user_prompt = await _resolve_user_prompt(agent_input, prompt)
    if agent_input.upload_workspace is not None:
        note = f'[Workspace directory available at {agent_input.workspace}]'
        if isinstance(user_prompt, str):
            user_prompt = f'{user_prompt}\n\n{note}'
        else:
            user_prompt = list(user_prompt)
            user_prompt.append({'type': 'text', 'text': note})
    main_input = _build_main_input(agent_input, user_prompt, history)
    logger.info(
        '[visual_agent] start | max_iter=%d tools=%s images=%d videos=%d image_history_limit=%s',
        main_input.max_iterations,
        main_input.tools,
        _count_prompt_images(user_prompt),
        n_videos,
        main_input.image_history_limit,
    )

    hook_scope = (
        image_hook_context(agent_input.post_iteration_hook) if agent_input.show_image_dimensions else nullcontext()
    )
    ctx = require_context()
    initial_manager = ctx.compaction_manager
    initial_segments = initial_manager.segment_count if initial_manager is not None else 0
    with hook_scope:
        main_output = await run_main_agent(main_input)

    manager = ctx.compaction_manager
    previous_segments = initial_segments if manager is initial_manager else 0
    trajectory_compacted = (manager is not None and manager.segment_count > previous_segments) or any(
        message.get('_is_compacted_summary') for message in main_output.main_trajectory
    )
    generated_message_start = None
    if not trajectory_compacted:
        # main_agent prepends a system message and appends the current user prompt.
        # Runtime messages may be refreshed during initial context preparation.
        prompt_index = 1 + sum(not is_compaction_runtime_state_message(message) for message in history or [])
        original_positions = [
            index
            for index, message in enumerate(main_output.main_trajectory)
            if not is_compaction_runtime_state_message(message)
        ]
        if prompt_index < len(original_positions):
            position = original_positions[prompt_index]
            if main_output.main_trajectory[position].get('role') == 'user':
                generated_message_start = position + 1

    logger.info(
        '[visual_agent] done | exit=%s iter=%d elapsed=%.1fs answer_len=%d',
        main_output.exit_status,
        main_output.iterations,
        time.monotonic() - t0,
        len(main_output.final_answer) if main_output.final_answer else 0,
    )
    return VisualAgentOutput(
        **main_output.model_dump(),
        workspace=agent_input.workspace,
        generated_message_start=generated_message_start,
        trajectory_compacted=trajectory_compacted,
    )


__all__ = [
    'DEFAULT_VISUAL_TOOLS',
    'VisualAgentInput',
    'VisualAgentOutput',
    'run_visual_agent',
]
