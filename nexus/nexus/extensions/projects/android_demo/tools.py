"""Android GUI tools for the Android Demo project.

These tools provide LOCAL-type wrappers around the runtime's GUI methods,
allowing the agent to interact with Android devices through the ExecutionContext.
"""

from pydantic import BaseModel, Field

from nexus.core import ToolContext, ToolResponse, ToolType, require_context, tool_registry


# ==================== Input Models ====================


class ScreenshotInput(BaseModel):
    """Input for screenshot tool - no parameters needed."""

    pass


class TapInput(BaseModel):
    """Input for tap tool."""

    x: int = Field(..., description='X coordinate to tap')
    y: int = Field(..., description='Y coordinate to tap')


class SwipeInput(BaseModel):
    """Input for swipe tool."""

    x1: int = Field(..., description='Start X coordinate')
    y1: int = Field(..., description='Start Y coordinate')
    x2: int = Field(..., description='End X coordinate')
    y2: int = Field(..., description='End Y coordinate')
    duration: int = Field(300, description='Swipe duration in milliseconds')


class LongPressInput(BaseModel):
    """Input for long press tool."""

    x: int = Field(..., description='X coordinate to long press')
    y: int = Field(..., description='Y coordinate to long press')
    duration: int = Field(1000, description='Press duration in milliseconds')


class TypeTextInput(BaseModel):
    """Input for type text tool."""

    text: str = Field(..., description='Text to type')


class PressKeyInput(BaseModel):
    """Input for press key tool."""

    key: str = Field(..., description='Key name: Back, Home, Menu, Enter, Delete, Tab, Escape, etc.')


class LaunchAppInput(BaseModel):
    """Input for launch app tool."""

    app: str = Field(..., description='App name or package name to launch')


class ResetAppsInput(BaseModel):
    """Input for reset apps tool."""

    pass


class DeviceInfoInput(BaseModel):
    """Input for device info tool."""

    pass


# ==================== Tool Implementations ====================


@tool_registry.register(
    'android/screenshot',
    meta={
        'name': 'screenshot',
        'description': 'Take a screenshot of the current Android screen. '
        'Returns a base64-encoded PNG image. Always use this first to observe the current state before taking actions.',
        'tool_type': ToolType.LOCAL,
    },
)
async def screenshot_tool(params: ScreenshotInput, context: ToolContext) -> ToolResponse:
    """Take a screenshot of the Android device."""
    ctx = require_context()
    runtime = await ctx.get_runtime('android')
    response = await runtime.gui_screenshot()

    if not response.success:
        return ToolResponse(
            output=[{'type': 'text', 'text': f'Screenshot failed: {response.error or "Unknown error"}'}],
            meta={'success': False, 'error': response.error},
        )

    # Return image as base64 for vision models
    output = []
    if response.image:
        output.append({
            'type': 'image_url',
            'image_url': {'url': f'data:image/{response.format};base64,{response.image}'},
        })
    output.append({
        'type': 'text',
        'text': f'Screenshot captured. Screen size: {response.width}x{response.height}',
    })

    return ToolResponse(
        output=output,
        meta={
            'success': True,
            'width': response.width,
            'height': response.height,
        },
    )


@tool_registry.register(
    'android/tap',
    meta={
        'name': 'tap',
        'description': 'Tap at the specified coordinates on the Android screen. '
        'Coordinates are in pixels based on the screen resolution.',
        'tool_type': ToolType.LOCAL,
    },
)
async def tap_tool(params: TapInput, context: ToolContext) -> ToolResponse:
    """Tap at specified coordinates."""
    ctx = require_context()
    runtime = await ctx.get_runtime('android')
    response = await runtime.gui_tap(params.x, params.y)

    if not response.success:
        return ToolResponse(
            output=[{'type': 'text', 'text': f'Tap failed at ({params.x}, {params.y}): {response.error}'}],
            meta={'success': False, 'error': response.error},
        )

    return ToolResponse(
        output=[{'type': 'text', 'text': f'Tapped at ({params.x}, {params.y})'}],
        meta={'success': True},
    )


@tool_registry.register(
    'android/swipe',
    meta={
        'name': 'swipe',
        'description': 'Swipe from one point to another on the Android screen. '
        'Use this for scrolling or drag gestures. Duration is in milliseconds.',
        'tool_type': ToolType.LOCAL,
    },
)
async def swipe_tool(params: SwipeInput, context: ToolContext) -> ToolResponse:
    """Swipe from one point to another."""
    ctx = require_context()
    runtime = await ctx.get_runtime('android')
    response = await runtime.gui_swipe(params.x1, params.y1, params.x2, params.y2, duration=params.duration)

    if not response.success:
        return ToolResponse(
            output=[
                {
                    'type': 'text',
                    'text': f'Swipe failed from ({params.x1}, {params.y1}) to ({params.x2}, {params.y2}): {response.error}',
                }
            ],
            meta={'success': False, 'error': response.error},
        )

    return ToolResponse(
        output=[
            {
                'type': 'text',
                'text': f'Swiped from ({params.x1}, {params.y1}) to ({params.x2}, {params.y2})',
            }
        ],
        meta={'success': True},
    )


@tool_registry.register(
    'android/long_press',
    meta={
        'name': 'long_press',
        'description': 'Long press at the specified coordinates. '
        'Duration is in milliseconds (default 1000ms = 1 second).',
        'tool_type': ToolType.LOCAL,
    },
)
async def long_press_tool(params: LongPressInput, context: ToolContext) -> ToolResponse:
    """Long press at specified coordinates."""
    ctx = require_context()
    runtime = await ctx.get_runtime('android')
    response = await runtime.gui_long_press(params.x, params.y, duration=params.duration)

    if not response.success:
        return ToolResponse(
            output=[
                {
                    'type': 'text',
                    'text': f'Long press failed at ({params.x}, {params.y}): {response.error}',
                }
            ],
            meta={'success': False, 'error': response.error},
        )

    return ToolResponse(
        output=[
            {
                'type': 'text',
                'text': f'Long pressed at ({params.x}, {params.y}) for {params.duration}ms',
            }
        ],
        meta={'success': True},
    )


@tool_registry.register(
    'android/type_text',
    meta={
        'name': 'type_text',
        'description': 'Type text into the currently focused input field. '
        'Make sure an input field is focused (by tapping it) before using this.',
        'tool_type': ToolType.LOCAL,
    },
)
async def type_text_tool(params: TypeTextInput, context: ToolContext) -> ToolResponse:
    """Type text into the focused input field."""
    ctx = require_context()
    runtime = await ctx.get_runtime('android')
    response = await runtime.gui_type(params.text)

    if not response.success:
        return ToolResponse(
            output=[{'type': 'text', 'text': f'Type text failed: {response.error}'}],
            meta={'success': False, 'error': response.error},
        )

    return ToolResponse(
        output=[{'type': 'text', 'text': f'Typed: "{params.text}"'}],
        meta={'success': True},
    )


@tool_registry.register(
    'android/press_key',
    meta={
        'name': 'press_key',
        'description': 'Press a special key on Android. '
        'Supported keys: Back, Home, Menu, Enter, Delete, Tab, Escape, Space, VolumeUp, VolumeDown, Power.',
        'tool_type': ToolType.LOCAL,
    },
)
async def press_key_tool(params: PressKeyInput, context: ToolContext) -> ToolResponse:
    """Press a special key."""
    ctx = require_context()
    runtime = await ctx.get_runtime('android')
    response = await runtime.gui_press_key(params.key)

    if not response.success:
        return ToolResponse(
            output=[{'type': 'text', 'text': f'Press key failed: {response.error}'}],
            meta={'success': False, 'error': response.error},
        )

    return ToolResponse(
        output=[{'type': 'text', 'text': f'Pressed key: {params.key}'}],
        meta={'success': True},
    )


@tool_registry.register(
    'android/launch_app',
    meta={
        'name': 'launch_app',
        'description': 'Launch an application by name or package name. '
        'Examples: "Settings", "Chrome", "com.android.settings".',
        'tool_type': ToolType.LOCAL,
    },
)
async def launch_app_tool(params: LaunchAppInput, context: ToolContext) -> ToolResponse:
    """Launch an application."""
    ctx = require_context()
    runtime = await ctx.get_runtime('android')
    response = await runtime.gui_launch_app(params.app)

    if not response.success:
        return ToolResponse(
            output=[{'type': 'text', 'text': f'Launch app failed: {response.error}'}],
            meta={'success': False, 'error': response.error},
        )

    return ToolResponse(
        output=[{'type': 'text', 'text': f'Launched app: {params.app}'}],
        meta={'success': True},
    )


@tool_registry.register(
    'android/reset_apps',
    meta={
        'name': 'reset_apps',
        'description': 'Close all currently running Android applications except essential system apps.',
        'tool_type': ToolType.LOCAL,
    },
)
async def reset_apps_tool(params: ResetAppsInput, context: ToolContext) -> ToolResponse:
    """Reset Android app state by force-stopping running apps."""
    ctx = require_context()
    runtime = await ctx.get_runtime('android')
    response = await runtime.gui_reset_apps()

    if not response.success:
        return ToolResponse(
            output=[{'type': 'text', 'text': f'Reset apps failed: {response.error}'}],
            meta={'success': False, 'error': response.error},
        )

    packages = response.stopped_packages or []
    return ToolResponse(
        output=[{'type': 'text', 'text': f'Reset complete. Stopped packages: {packages}'}],
        meta={'success': True, 'stopped_packages': packages},
    )


@tool_registry.register(
    'android/device_info',
    meta={
        'name': 'device_info',
        'description': 'Get Android device details including model, platform version, resolution and current app.',
        'tool_type': ToolType.LOCAL,
    },
)
async def device_info_tool(params: DeviceInfoInput, context: ToolContext) -> ToolResponse:
    """Get Android device information."""
    ctx = require_context()
    runtime = await ctx.get_runtime('android')
    response = await runtime.gui_get_device_info()

    if not response.success:
        return ToolResponse(
            output=[{'type': 'text', 'text': f'Device info failed: {response.error}'}],
            meta={'success': False, 'error': response.error},
        )

    payload = {
        'device_name': response.device_name,
        'platform_version': response.platform_version,
        'automation_name': response.automation_name,
        'window_width': response.window_width,
        'window_height': response.window_height,
        'wm_size': response.wm_size,
        'wm_density': response.wm_density,
        'current_package': response.current_package,
        'current_activity': response.current_activity,
    }
    return ToolResponse(
        output=[{'type': 'text', 'text': str(payload)}],
        meta={'success': True, **payload},
    )
