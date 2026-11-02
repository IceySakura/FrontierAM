"""Mobile GUI tools for the Android World runtime, aligned with HyData's mobile action set.

Each HyData mobile action is a separate tool (``mobile/<name>``) rather than one dispatcher tool.
Coordinates are HyData-canonical **normalized ``[0, 1]``** (``point`` / ``from`` / ``to`` as
``[x, y]`` fractions of screen width/height); each tool scales them to real device pixels
internally (reading the screen size the task published to ``ctx.tool_state``) before calling the
runtime, since the Android World backend expects real pixels.

Only the subset of HyData mobile actions the Android World backend can actually execute is exposed
(see ``docs``/plan): ``take_screenshot`` (done by the per-step observation hook), ``call_user``,
``mobile.drag`` (no native drag), ``mobile.close_app`` (backend only navigates home), and the
``menu`` button are intentionally omitted.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from nexus.core import ToolContext, ToolResponse, ToolType, require_context, tool_registry
from nexus.extensions.tools.mobile.observation import (
    MOBILE_OBSERVATION_SCREEN_HEIGHT_STATE,
    MOBILE_OBSERVATION_SCREEN_WIDTH_STATE,
)


MOBILE_RUNTIME_NAME = 'android'
MOBILE_LAST_ACTION_STATE = 'android_world_last_action'
MOBILE_ANSWER_STATE = 'android_world_answer'
MOBILE_LAST_TOOL_STATE = 'mobile_last_tool'

# Fallback screen size if the task did not publish one to tool_state.
_DEFAULT_SCREEN_WIDTH = 1080
_DEFAULT_SCREEN_HEIGHT = 2400

# HyData mobile.button_press buttons the AndroidWorld backend can execute (no independent `menu`).
_BUTTON_KEY_MAP = {'back': 'Back', 'home': 'Home', 'enter': 'Enter'}


def _screen_size() -> tuple[int, int]:
    """Real device screen size published by the task, for [0,1] -> pixel scaling."""
    ctx = require_context()
    w = ctx.tool_state.get(MOBILE_OBSERVATION_SCREEN_WIDTH_STATE)
    h = ctx.tool_state.get(MOBILE_OBSERVATION_SCREEN_HEIGHT_STATE)
    try:
        return (int(w) if w else _DEFAULT_SCREEN_WIDTH, int(h) if h else _DEFAULT_SCREEN_HEIGHT)
    except (TypeError, ValueError):
        return _DEFAULT_SCREEN_WIDTH, _DEFAULT_SCREEN_HEIGHT


def _scale_point(point: list[float]) -> tuple[int, int]:
    """Scale a normalized ``[x, y]`` in ``[0, 1]`` to real device pixels, clamped to screen."""
    to_w, to_h = _screen_size()
    x = int(float(point[0]) * to_w)
    y = int(float(point[1]) * to_h)
    return max(0, min(x, to_w - 1)), max(0, min(y, to_h - 1))


async def _runtime() -> Any:
    """Resolve the shared Android World runtime from the active tool context."""
    ctx = require_context()
    return await ctx.get_runtime(MOBILE_RUNTIME_NAME)


def _ok(result: Any) -> tuple[bool, str | None]:
    """Normalize an executor result (model object or dict) to (success, error)."""
    if isinstance(result, dict):
        return bool(result.get('success')), result.get('error')
    return bool(getattr(result, 'success', False)), getattr(result, 'error', None)


def _response(
    name: str, success: bool, error: str | None, *, terminates: bool = False, answer: str | None = None
) -> ToolResponse:
    """Build a ToolResponse with the shared success/termination convention."""
    text = f'{name}: {"success" if success else f"failed ({error})"}'
    context_updates: dict[str, Any] = {MOBILE_LAST_TOOL_STATE: f'mobile/{name}'}
    if terminates:
        context_updates[MOBILE_LAST_ACTION_STATE] = name
        context_updates[MOBILE_ANSWER_STATE] = answer or ''
    return ToolResponse(
        output=[{'type': 'text', 'text': text}],
        meta={'success': success, 'error': error, 'termination_requested': terminates},
        context_updates=context_updates,
    )


_POINT_FIELD = Field(..., min_length=2, max_length=2, description='Normalized [x, y] in [0, 1] (fraction of screen).')


# ---------------------------------------------------------------------------
# gui_core (shared) actions
# ---------------------------------------------------------------------------


class ClickInput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    point: list[float] = _POINT_FIELD
    button: Literal['left', 'right', 'middle'] = Field('left', description='Mouse button (mobile: left = tap).')
    repeat: int = Field(1, ge=1, description='Consecutive identical clicks merged into one action.')


@tool_registry.register(
    'mobile/click',
    meta={'name': 'click', 'description': 'Tap at a normalized point [x, y] in [0, 1].', 'tool_type': ToolType.LOCAL},
)
async def mobile_click_tool(params: ClickInput, context: ToolContext) -> ToolResponse:
    """Tap the screen at a normalized point."""
    runtime = await _runtime()
    x, y = _scale_point(params.point)
    success, error = _ok(await runtime.gui_tap(x, y))
    return _response('click', success, error)


class TypeInput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    text: str = Field(..., description='Text to type into the focused field.')
    point: list[float] | None = Field(
        None, min_length=2, max_length=2, description='Optional point to tap/focus first.'
    )
    interval: float | None = Field(None, ge=0, description='Seconds between keystrokes (advisory).')


@tool_registry.register(
    'mobile/type',
    meta={
        'name': 'type',
        'description': 'Type text; optionally tap a point to focus a field first.',
        'tool_type': ToolType.LOCAL,
    },
)
async def mobile_type_tool(params: TypeInput, context: ToolContext) -> ToolResponse:
    """Type text into the currently focused input, optionally focusing a field first."""
    runtime = await _runtime()
    if params.point is not None:
        x, y = _scale_point(params.point)
        await runtime.gui_tap(x, y)
    success, error = _ok(await runtime.gui_type(params.text))
    return _response('type', success, error)


class PressInput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    key: str = Field(..., description='Key to press and release (e.g. enter, tab).')
    repeat: int = Field(1, ge=1, description='Consecutive identical presses merged into one action.')


@tool_registry.register(
    'mobile/press',
    meta={'name': 'press', 'description': 'Press and release a single key (e.g. enter).', 'tool_type': ToolType.LOCAL},
)
async def mobile_press_tool(params: PressInput, context: ToolContext) -> ToolResponse:
    """Press a keyboard key (shared gui_core action)."""
    runtime = await _runtime()
    success = True
    error = None
    for _ in range(params.repeat):
        success, error = _ok(await runtime.gui_press_key(params.key))
        if not success:
            break
    return _response('press', success, error)


class WaitInput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    duration: float = Field(..., ge=0, description='Seconds to wait for the UI to settle.')


@tool_registry.register(
    'mobile/wait',
    meta={'name': 'wait', 'description': 'Wait for the UI to settle.', 'tool_type': ToolType.LOCAL},
)
async def mobile_wait_tool(params: WaitInput, context: ToolContext) -> ToolResponse:
    """Wait for the environment to settle."""
    import asyncio

    await asyncio.sleep(min(params.duration, 5))
    return _response('wait', True, None)


class AnswerInput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    text: str = Field(..., description="Answer to the user's query.")


@tool_registry.register(
    'mobile/answer',
    meta={
        'name': 'answer',
        'description': "Submit an answer to the user's query and finish the task.",
        'tool_type': ToolType.LOCAL,
    },
)
async def mobile_answer_tool(params: AnswerInput, context: ToolContext) -> ToolResponse:
    """Submit a textual answer and terminate the task."""
    runtime = await _runtime()
    success, error = _ok(await runtime.call_executor('aw_answer', {'answer': params.text}))
    return _response('answer', success, error, terminates=True, answer=params.text)


class TerminateInput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    status: Literal['success', 'failure', 'infeasible'] = Field(..., description='Final task status.')
    reason: str | None = Field(None, description='Optional explanation.')


@tool_registry.register(
    'mobile/terminate',
    meta={
        'name': 'terminate',
        'description': 'End the task with a final status (success/failure/infeasible).',
        'tool_type': ToolType.LOCAL,
    },
)
async def mobile_terminate_tool(params: TerminateInput, context: ToolContext) -> ToolResponse:
    """End the task, recording a final status."""
    runtime = await _runtime()
    # Signal completion to the Android World backend (no answer text).
    success, error = _ok(await runtime.call_executor('aw_answer', {'answer': ''}))
    return _response('terminate', success, error, terminates=True, answer='')


# ---------------------------------------------------------------------------
# mobile.* actions
# ---------------------------------------------------------------------------


class SwipeInput(BaseModel):
    model_config = ConfigDict(extra='forbid', populate_by_name=True)
    from_: list[float] = Field(
        ..., alias='from', min_length=2, max_length=2, description='Start point, normalized [x, y] in [0, 1].'
    )
    to: list[float] = Field(..., min_length=2, max_length=2, description='End point, normalized [x, y] in [0, 1].')
    duration: float | None = Field(None, ge=0, description='Seconds for the gesture (advisory).')


@tool_registry.register(
    'mobile/swipe',
    meta={
        'name': 'mobile.swipe',
        'description': 'Swipe/fling from one normalized point to another.',
        'tool_type': ToolType.LOCAL,
    },
)
async def mobile_swipe_tool(params: SwipeInput, context: ToolContext) -> ToolResponse:
    """Coordinate swipe gesture (also used for directional scrolling)."""
    runtime = await _runtime()
    x1, y1 = _scale_point(params.from_)
    x2, y2 = _scale_point(params.to)
    coords = f'{x1},{y1},{x2},{y2}'
    success, error = _ok(await runtime.call_executor('aw_swipe', {'direction': coords}))
    return _response('swipe', success, error)


class LongPressInput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    point: list[float] = _POINT_FIELD
    duration: float = Field(0.5, ge=0, description='Hold duration in seconds.')


@tool_registry.register(
    'mobile/long_press',
    meta={
        'name': 'mobile.long_press',
        'description': 'Touch and hold at a normalized point.',
        'tool_type': ToolType.LOCAL,
    },
)
async def mobile_long_press_tool(params: LongPressInput, context: ToolContext) -> ToolResponse:
    """Long-press at a normalized point."""
    runtime = await _runtime()
    x, y = _scale_point(params.point)
    success, error = _ok(await runtime.gui_long_press(x, y))
    return _response('long_press', success, error)


class DoubleClickInput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    point: list[float] = _POINT_FIELD
    interval: float | None = Field(None, ge=0, description='Seconds between the two taps (advisory).')


@tool_registry.register(
    'mobile/double_click',
    meta={
        'name': 'mobile.double_click',
        'description': 'Double-tap at a normalized point.',
        'tool_type': ToolType.LOCAL,
    },
)
async def mobile_double_click_tool(params: DoubleClickInput, context: ToolContext) -> ToolResponse:
    """Double-tap at a normalized point."""
    runtime = await _runtime()
    x, y = _scale_point(params.point)
    success, error = _ok(await runtime.gui_double_tap(x, y))
    return _response('double_click', success, error)


class ButtonPressInput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    button: Literal['back', 'home', 'enter'] = Field(..., description='System navigation button.')


@tool_registry.register(
    'mobile/button_press',
    meta={
        'name': 'mobile.button_press',
        'description': 'Press a system navigation button (back/home/enter).',
        'tool_type': ToolType.LOCAL,
    },
)
async def mobile_button_press_tool(params: ButtonPressInput, context: ToolContext) -> ToolResponse:
    """Press a system navigation key."""
    runtime = await _runtime()
    key = _BUTTON_KEY_MAP[params.button]
    success, error = _ok(await runtime.gui_press_key(key))
    return _response('button_press', success, error)


class OpenAppInput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    package: str = Field(..., description='App package/bundle id or name to open.')


@tool_registry.register(
    'mobile/open_app',
    meta={'name': 'mobile.open_app', 'description': 'Open a mobile app by package/name.', 'tool_type': ToolType.LOCAL},
)
async def mobile_open_app_tool(params: OpenAppInput, context: ToolContext) -> ToolResponse:
    """Open a mobile app."""
    runtime = await _runtime()
    success, error = _ok(await runtime.gui_launch_app(params.package))
    return _response('open_app', success, error)


# Model-visible tool names, in HyData mobile order (gui_core subset + mobile.*).
MOBILE_TOOL_NAMES = [
    'mobile/click',
    'mobile/type',
    'mobile/press',
    'mobile/wait',
    'mobile/answer',
    'mobile/terminate',
    'mobile/swipe',
    'mobile/long_press',
    'mobile/double_click',
    'mobile/button_press',
    'mobile/open_app',
]


__all__ = [
    'MOBILE_ANSWER_STATE',
    'MOBILE_LAST_ACTION_STATE',
    'MOBILE_RUNTIME_NAME',
    'MOBILE_TOOL_NAMES',
    '_scale_point',
    'mobile_answer_tool',
    'mobile_button_press_tool',
    'mobile_click_tool',
    'mobile_double_click_tool',
    'mobile_long_press_tool',
    'mobile_open_app_tool',
    'mobile_press_tool',
    'mobile_swipe_tool',
    'mobile_terminate_tool',
    'mobile_type_tool',
    'mobile_wait_tool',
]
