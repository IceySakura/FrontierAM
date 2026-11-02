"""Shared mobile screenshot observation hook for Android World agent loops."""

from __future__ import annotations

import asyncio

from typing import Any

from nexus.extensions.agents.main.agent import PostIterationHookResult


MOBILE_OBSERVATION_RUNTIME_NAME_STATE = 'mobile_observation_runtime_name'
MOBILE_OBSERVATION_STEP_PAUSE_STATE = 'mobile_observation_step_pause'
# Real device screen size, published by the task so a text→tool_call bridge client
# can scale normalized 1000x1000 coordinates to real pixels.
MOBILE_OBSERVATION_SCREEN_WIDTH_STATE = 'mobile_observation_screen_width'
MOBILE_OBSERVATION_SCREEN_HEIGHT_STATE = 'mobile_observation_screen_height'


def mobile_screenshot_content_part(response: Any) -> dict[str, Any] | None:
    """Convert a GUI screenshot response into an OpenAI-style image content part."""
    if not getattr(response, 'success', False) or not getattr(response, 'image', None):
        return None
    image_format = getattr(response, 'format', None) or 'jpeg'
    return {'type': 'image_url', 'image_url': {'url': f'data:image/{image_format};base64,{response.image}'}}


async def mobile_observation(ctx: Any, hook_context: Any) -> PostIterationHookResult | None:
    """Post-iteration hook that appends the current phone screenshot."""
    runtime_name = str(ctx.tool_state.get(MOBILE_OBSERVATION_RUNTIME_NAME_STATE, 'android'))
    step_pause = float(ctx.tool_state.get(MOBILE_OBSERVATION_STEP_PAUSE_STATE, 2.0) or 0)
    if step_pause > 0:
        await asyncio.sleep(step_pause)
    runtime = await ctx.get_runtime(runtime_name)
    screenshot = await runtime.gui_screenshot()
    screenshot_part = mobile_screenshot_content_part(screenshot)
    if screenshot_part is None:
        return None
    return PostIterationHookResult(
        extra_messages=[
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': 'Current phone screenshot.'},
                    screenshot_part,
                ],
            }
        ]
    )


__all__ = [
    'MOBILE_OBSERVATION_RUNTIME_NAME_STATE',
    'MOBILE_OBSERVATION_SCREEN_HEIGHT_STATE',
    'MOBILE_OBSERVATION_SCREEN_WIDTH_STATE',
    'MOBILE_OBSERVATION_STEP_PAUSE_STATE',
    'mobile_observation',
    'mobile_screenshot_content_part',
]
