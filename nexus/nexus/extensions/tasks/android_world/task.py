"""Nexus task adapter for Android World benchmark tasks.

Mirrors the OSWorld task adapter's shape: the task orchestrates the run and
delegates GUI control to the generic ``main_agent``. The Android World task
lifecycle (initialize / score / tear_down) lives on the runtime side
(:class:`~nexus.runtime.providers.android_world.client.AndroidWorldRuntime`),
so this adapter only sequences those calls around the agent loop.
"""

import logging

from typing import Any, cast

from pydantic import Field

from nexus.core import TaskRunInput, TaskRunOutput, WithAgentRunner, WithScore, require_context, task_registry
from nexus.extensions.tasks.android_world.prompts import mobile_computer_tool_prompt
from nexus.extensions.tools.mobile.observation import (
    MOBILE_OBSERVATION_RUNTIME_NAME_STATE,
    MOBILE_OBSERVATION_SCREEN_HEIGHT_STATE,
    MOBILE_OBSERVATION_SCREEN_WIDTH_STATE,
    MOBILE_OBSERVATION_STEP_PAUSE_STATE,
    mobile_screenshot_content_part,
)
from nexus.extensions.tools.mobile.tools import MOBILE_ANSWER_STATE, MOBILE_LAST_ACTION_STATE, MOBILE_TOOL_NAMES
from nexus.runtime.providers.android_world.client import AndroidWorldRuntime


logger = logging.getLogger(__name__)

MOBILE_OBSERVATION_HOOK = 'nexus.extensions.tools.mobile.observation::mobile_observation'


class AndroidWorldTaskInput(TaskRunInput, WithAgentRunner):
    """Input for a single Android World task run."""

    task_type: str = 'ContactsAddContact'
    seed: int = 0
    task_idx: int = 0
    instruction: str | None = None
    runtime_name: str = 'android'
    run_evaluator: bool = True
    step_pause: float = 2.0


class AndroidWorldTaskOutput(TaskRunOutput, WithScore):
    completed: bool
    task_type: str
    task_goal: str
    agent_answer: str | None = None
    raw_score: dict[str, Any] = Field(default_factory=dict)
    agent_output: Any | None = None
    error: str | None = None


def build_android_world_agent_tools() -> list[str]:
    """Build the agent tool list for the Android World action space (HyData-aligned mobile tools)."""
    return list(MOBILE_TOOL_NAMES)


def build_android_world_system_prompt(*, screen_width: int = 1080, screen_height: int = 1920) -> str:
    """Build the Android World system prompt for the current screen size.

    Returns the native function-calling prompt for the HyData-aligned ``mobile/*`` tools
    (normalized ``[0, 1]`` coordinates; the tools scale to real pixels internally).
    """
    return mobile_computer_tool_prompt(screen_width=screen_width, screen_height=screen_height)


def build_android_world_user_prompt(
    instruction: str,
    screenshot_response: Any | None = None,
) -> str | list[dict[str, Any]]:
    """Build the per-task user prompt with instruction and optional initial screenshot."""
    prompt = f'You are asked to complete the following task: {instruction}'
    if screenshot_response is None:
        return prompt
    screenshot_part = mobile_screenshot_content_part(screenshot_response)
    if screenshot_part is None:
        return prompt
    return [{'type': 'text', 'text': prompt}, screenshot_part]


@task_registry.register('android_world')
async def run_android_world(task_input: AndroidWorldTaskInput) -> AndroidWorldTaskOutput:
    """Run one Android World task through a Nexus agent and the native scorer."""
    ctx = require_context()
    runtime = cast(AndroidWorldRuntime, await ctx.get_runtime(task_input.runtime_name))

    # Step 1: Initialize the task (runtime-side lifecycle) and get its goal.
    try:
        task_goal = await runtime.task_initialize(task_input.task_type, task_input.seed, task_input.task_idx)
    except Exception as e:
        logger.error('Android World task initialization failed: %s', e)
        return AndroidWorldTaskOutput(
            score=0.0,
            completed=False,
            task_type=task_input.task_type,
            task_goal=f'Initialization failed: {e}',
            error=str(e),
        )

    instruction = task_input.instruction or task_goal

    # Step 2: Resolve screen size for the prompt and configure the observation hook.
    size = await runtime.gui_get_screen_size()
    screen_width = int(getattr(size, 'width', None) or 1080)
    screen_height = int(getattr(size, 'height', None) or 1920)

    ctx.tool_state[MOBILE_OBSERVATION_RUNTIME_NAME_STATE] = task_input.runtime_name
    ctx.tool_state[MOBILE_OBSERVATION_STEP_PAUSE_STATE] = task_input.step_pause
    # Publish the real screen size so a text→tool_call bridge client can
    # scale normalized 1000x1000 coordinates to real pixels.
    ctx.tool_state[MOBILE_OBSERVATION_SCREEN_WIDTH_STATE] = screen_width
    ctx.tool_state[MOBILE_OBSERVATION_SCREEN_HEIGHT_STATE] = screen_height
    ctx.tool_state.pop(MOBILE_LAST_ACTION_STATE, None)
    ctx.tool_state.pop(MOBILE_ANSWER_STATE, None)

    system_prompt = build_android_world_system_prompt(screen_width=screen_width, screen_height=screen_height)
    initial_screenshot = await runtime.gui_screenshot()
    user_prompt = build_android_world_user_prompt(instruction, initial_screenshot)

    # Step 3: Run the agent.
    agent_output = await task_input.run_agent(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        post_iteration_hook=MOBILE_OBSERVATION_HOOK,
        tools=build_android_world_agent_tools(),
    )

    # Step 4: Score using the runtime-native scorer.
    raw_score: dict[str, Any] = {}
    score = 0.0
    if task_input.run_evaluator:
        try:
            raw_score = await runtime.task_score()
            score = float(raw_score.get('score', 0.0))
            logger.info('Android World task score: %s', score)
        except Exception as e:
            logger.error('Android World task score failed: %s', e)

    # Step 5: Tear down the task environment.
    try:
        await runtime.task_tear_down()
    except Exception as e:
        logger.warning('Android World task tear_down failed: %s', e)

    return AndroidWorldTaskOutput(
        score=score,
        completed=True,
        task_type=task_input.task_type,
        task_goal=task_goal,
        agent_answer=ctx.tool_state.get(MOBILE_ANSWER_STATE),
        raw_score=raw_score,
        agent_output=agent_output,
    )


__all__ = [
    'AndroidWorldTaskInput',
    'AndroidWorldTaskOutput',
    'build_android_world_agent_tools',
    'build_android_world_system_prompt',
    'build_android_world_user_prompt',
    'run_android_world',
]
