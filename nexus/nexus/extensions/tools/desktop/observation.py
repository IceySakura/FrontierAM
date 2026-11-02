"""Shared desktop screenshot observation hook for GUI task loops."""

from __future__ import annotations

import asyncio

from collections.abc import Sequence
from typing import Any

from nexus.extensions.agents.main.agent import PostIterationHookResult


DESKTOP_OBSERVATION_RUNTIME_NAME_STATE = 'desktop_observation_runtime_name'
DESKTOP_OBSERVATION_STEP_PAUSE_STATE = 'desktop_observation_step_pause'
LEGACY_OS_WORLD_RUNTIME_NAME_STATE = 'os_world_runtime_name'
LEGACY_OS_WORLD_STEP_PAUSE_STATE = 'os_world_step_pause'
WAIT_MARKER = 'WAIT'


def desktop_screenshot_content_part(response: Any) -> dict[str, Any] | None:
    """Convert a GUI screenshot response into an OpenAI-style image content part."""
    if not getattr(response, 'success', False) or not getattr(response, 'image', None):
        return None
    image_format = getattr(response, 'format', None) or 'png'
    return {'type': 'image_url', 'image_url': {'url': f'data:image/{image_format};base64,{response.image}'}}


def _tool_messages_include_marker(tool_messages: Sequence[dict[str, Any]], marker: str) -> bool:
    def _contains(value: Any) -> bool:
        if value == marker:
            return True
        if isinstance(value, dict):
            return any(_contains(item) for item in value.values())
        if isinstance(value, list):
            return any(_contains(item) for item in value)
        return False

    return any(_contains(message.get('content')) for message in tool_messages)


def _tool_state_value(tool_state: dict[str, Any], key: str, legacy_key: str, default: Any) -> Any:
    if key in tool_state:
        return tool_state[key]
    return tool_state.get(legacy_key, default)


async def desktop_observation(ctx: Any, hook_context: Any) -> PostIterationHookResult | None:
    """Post-iteration hook that appends the current desktop screenshot."""
    runtime_name = str(
        _tool_state_value(
            ctx.tool_state,
            DESKTOP_OBSERVATION_RUNTIME_NAME_STATE,
            LEGACY_OS_WORLD_RUNTIME_NAME_STATE,
            'main',
        )
    )
    step_pause = float(
        _tool_state_value(
            ctx.tool_state,
            DESKTOP_OBSERVATION_STEP_PAUSE_STATE,
            LEGACY_OS_WORLD_STEP_PAUSE_STATE,
            2.0,
        )
        or 0
    )
    if step_pause > 0:
        if _tool_messages_include_marker(hook_context.tool_messages, WAIT_MARKER):
            await asyncio.sleep(step_pause)
        await asyncio.sleep(step_pause)
    runtime = await ctx.get_runtime(runtime_name)
    screenshot = await runtime.gui_screenshot()
    screenshot_part = desktop_screenshot_content_part(screenshot)
    if screenshot_part is None:
        return None
    return PostIterationHookResult(
        extra_messages=[
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': 'Current desktop screenshot.'},
                    screenshot_part,
                ],
            }
        ]
    )


__all__ = [
    'DESKTOP_OBSERVATION_RUNTIME_NAME_STATE',
    'DESKTOP_OBSERVATION_STEP_PAUSE_STATE',
    'LEGACY_OS_WORLD_RUNTIME_NAME_STATE',
    'LEGACY_OS_WORLD_STEP_PAUSE_STATE',
    'desktop_observation',
    'desktop_screenshot_content_part',
]
