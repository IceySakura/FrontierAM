"""Android GUI Agent for automating Android device interactions.

This agent uses vision-capable LLMs to observe the screen and execute
GUI actions to complete user tasks on Android devices.
"""

import base64
import json
import logging
import os

from datetime import datetime

from nexus.core import (
    AgentRunInput,
    AgentRunOutput,
    Registry,
    ToolExecutor,
    WithMainTrajectory,
    WithUserPrompt,
    agent_registry,
    require_context,
    tool_registry,
)


logger = logging.getLogger(__name__)


# Import tools to register them
Registry.import_module('nexus.extensions.projects.android_demo.tools')


SYSTEM_PROMPT = """You are an Android GUI automation agent. Your task is to interact with an Android device to complete user requests.

## Available Tools

You have access to the following tools:
- screenshot: Take a screenshot to observe the current screen state
- tap: Tap at specific coordinates (x, y)
- swipe: Swipe from one point to another for scrolling or dragging
- long_press: Long press at coordinates for context menus
- type_text: Type text into a focused input field
- press_key: Press special keys (Back, Home, Enter, etc.)

## Important Guidelines

1. **Always observe first**: Before taking any action, use the screenshot tool to see the current screen state.

2. **Coordinate system**: Coordinates are in pixels. The screen resolution varies by device. Use the screenshot dimensions to understand the coordinate range.

3. **Common UI patterns**:
   - Time pickers: Usually have scrollable hour/minute columns. Swipe up/down to change values.
   - Toggle switches: Tap to toggle on/off
   - Lists: Swipe up to scroll down, swipe down to scroll up
   - Buttons: Tap the center of the button
   - App icons on home screen: Tap to open the app

4. **After each action**: Take another screenshot to verify the action's effect before proceeding.

5. **Error recovery**: If an action doesn't work as expected, try alternative approaches:
   - If tap doesn't work, try a different position
   - Use the Back key to return to previous screens
   - Swipe to find elements that may be off-screen

6. **Task completion**: When you believe the task is complete, take a final screenshot to verify the result. The system will automatically evaluate whether the task was successful.

## Example Workflow

1. Take screenshot to see current state
2. Analyze the screen and plan the next action
3. Execute the action (tap, swipe, type, etc.)
4. Take screenshot to verify the result
5. Repeat until task appears complete
6. Take a final screenshot for verification

Remember: Be methodical and verify each step before proceeding to the next."""


class AndroidGuiAgentInput(AgentRunInput, WithUserPrompt):
    """Input for the Android GUI Agent."""

    max_iterations: int = 30
    """Maximum number of agent iterations before stopping."""

    system_prompt: str | None = None
    """Optional custom system prompt. If not provided, uses the default."""

    screenshot_dir: str | None = None
    """Directory to save screenshots. If None, screenshots are not saved to disk."""


class AndroidGuiAgentOutput(AgentRunOutput, WithMainTrajectory):
    """Output from the Android GUI Agent."""

    screenshot_paths: list[str] = []
    """List of paths to saved screenshots."""


def _save_screenshot(screenshot_dir: str, image_base64: str, step: int, action: str) -> str | None:
    """Save a screenshot to the specified directory.

    Args:
        screenshot_dir: Directory to save the screenshot
        image_base64: Base64-encoded image data
        step: Step number
        action: Action name for the filename

    Returns:
        Path to the saved screenshot, or None if save failed
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(screenshot_dir, exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%H%M%S')
        safe_action = action.replace('/', '_').replace(' ', '_')[:20]
        filename = f'step_{step:03d}_{timestamp}_{safe_action}.png'
        filepath = os.path.join(screenshot_dir, filename)

        # Fix base64 padding if needed
        padding = len(image_base64) % 4
        if padding:
            image_base64 += '=' * (4 - padding)

        # Decode and save
        image_data = base64.b64decode(image_base64)
        with open(filepath, 'wb') as f:
            f.write(image_data)

        logger.debug(f'Screenshot saved: {filepath}')
        return filepath
    except Exception as e:
        logger.warning(f'Failed to save screenshot: {e}')
        return None


@agent_registry.register('android_gui_agent')
async def run_android_gui_agent(agent_input: AndroidGuiAgentInput) -> AndroidGuiAgentOutput:
    """Run the Android GUI Agent.

    The agent observes the screen via screenshots and executes GUI actions
    to complete the user's task. It uses a vision-capable LLM to understand
    the screen content and decide on actions.
    """
    ctx = require_context()
    client = ctx.openai_client

    # Create tool executor with Android GUI tools
    executor = ToolExecutor([
        tool_registry.get('android/screenshot'),
        tool_registry.get('android/tap'),
        tool_registry.get('android/swipe'),
        tool_registry.get('android/long_press'),
        tool_registry.get('android/type_text'),
        tool_registry.get('android/press_key'),
    ])

    system_prompt = agent_input.system_prompt or SYSTEM_PROMPT
    history: list[dict] = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': agent_input.user_prompt},
    ]

    screenshot_paths: list[str] = []
    screenshot_step = 0

    for iteration in range(agent_input.max_iterations):
        logger.debug(f'Android GUI Agent iteration {iteration + 1}/{agent_input.max_iterations}')

        # Call LLM with tools
        response = await client.create_completion(
            messages=history,
            tools=executor.get_tool_schemas() or None,
        )

        if not response.choices:
            logger.error('No choices returned in the response')
            break

        message = response.choices[0].message
        history.append(message.model_dump(exclude_none=True))

        if message.content:
            logger.debug(f'Assistant: {message.content}')

        # No tool calls - conversation complete
        if not message.tool_calls:
            logger.info('No tool calls, agent finished')
            break

        # Process tool calls
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            try:
                tool_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                error_msg = f'Failed to parse tool arguments: {tool_call.function.arguments}'
                logger.warning(error_msg)
                history.append({
                    'role': 'tool',
                    'tool_call_id': tool_call.id,
                    'content': f'Error: {error_msg}',
                })
                continue

            logger.debug(f'Tool call: {tool_name}({json.dumps(tool_args, ensure_ascii=False)})')

            tool_response = None
            try:
                tool_response = await executor.call(tool_name, tool_args)

                # Format output for history
                output_parts = []
                for item in tool_response.output:
                    if item.get('type') == 'text':
                        output_parts.append(item.get('text', ''))
                    elif item.get('type') == 'image_url':
                        output_parts.append('[Screenshot captured]')
                output = '\n'.join(output_parts) if output_parts else str(tool_response.output)

            except Exception as e:
                error_msg = f'Error executing tool {tool_name}: {e}'
                logger.warning(error_msg)
                output = f'Error: {error_msg}'

            logger.debug(f'Tool output: {output[:200]}...' if len(output) > 200 else f'Tool output: {output}')

            # For screenshot tool, include the image in the history for vision models
            if tool_name == 'screenshot' and tool_response is not None and tool_response.output:
                content = []
                for item in tool_response.output:
                    if item.get('type') == 'text':
                        content.append({'type': 'text', 'text': item.get('text', '')})
                    elif item.get('type') == 'image_url':
                        content.append(item)
                        # Save screenshot to disk if screenshot_dir is configured
                        if agent_input.screenshot_dir:
                            image_url = item.get('image_url', {}).get('url', '')
                            if image_url.startswith('data:image/'):
                                # Extract base64 data after the comma
                                base64_data = image_url.split(',', 1)[1] if ',' in image_url else ''
                                if base64_data:
                                    screenshot_step += 1
                                    path = _save_screenshot(
                                        agent_input.screenshot_dir, base64_data, screenshot_step, 'screenshot'
                                    )
                                    if path:
                                        screenshot_paths.append(path)
                history.append({
                    'role': 'tool',
                    'tool_call_id': tool_call.id,
                    'content': content,
                })
            else:
                history.append({
                    'role': 'tool',
                    'tool_call_id': tool_call.id,
                    'content': output,
                })

    return AndroidGuiAgentOutput(
        main_trajectory=history,
        screenshot_paths=screenshot_paths,
    )
