"""Set Alarm Task for the Android Demo project.

This task orchestrates the Android GUI Agent to set an alarm,
then uses a judge model to verify the result from the final screenshot.
"""

import base64
import json
import logging
import os

from datetime import datetime

from nexus.core import (
    TaskRunInput,
    TaskRunOutput,
    WithAgentRunner,
    WithScore,
    require_context,
    task_registry,
)


logger = logging.getLogger(__name__)


USER_PROMPT_TEMPLATE = """Please set an alarm on this Android device.

## Task Details
- **Alarm Time**: {alarm_time} AM

## Steps to Complete
1. Find and open the alarm/clock application from the home screen
2. Create a new alarm or edit an existing one
3. Set the time to {alarm_time} AM
4. Make sure the alarm is enabled (toggled ON)
5. Verify the alarm appears in the alarm list

## Important Notes
- First take a screenshot to observe the current screen state
- Navigate to the alarm app by tapping on its icon on the home screen
- After setting the alarm, verify that it shows {alarm_time} and is enabled
- The alarm toggle/switch should be in the ON position
- Take a final screenshot showing the alarm list with the new alarm"""


JUDGE_PROMPT_TEMPLATE = """You are evaluating whether an Android GUI automation task was completed successfully.

## Task Description
The agent was asked to set an alarm for {alarm_time} AM on an Android device.

## Evaluation Criteria
Please analyze the screenshot and determine if:
1. An alarm for {alarm_time} (or {alarm_time} AM) is visible in the alarm list
2. The alarm is enabled/active (toggle switch is ON, or the alarm is not grayed out)

## Response Format
Respond with a JSON object:
{{
    "success": true/false,
    "reasoning": "Your detailed explanation of why you made this judgment"
}}

Be strict in your evaluation. The alarm time must match exactly ({alarm_time} AM), and the alarm must be clearly enabled."""


class SetAlarmTaskInput(TaskRunInput, WithAgentRunner):
    """Input for the Set Alarm Task."""

    alarm_time: str = '9:30'
    """The alarm time to set (format: H:MM or HH:MM)."""

    screenshot_dir: str | None = None
    """Directory to save screenshots. If None, screenshots are not saved to disk."""

    user_prompt: str | None = None
    """Optional custom user prompt. If not provided, uses the default template."""


class SetAlarmTaskOutput(TaskRunOutput, WithScore):
    """Output from the Set Alarm Task."""

    alarm_set: bool = False
    """Whether the alarm was set according to the agent."""

    final_screen_description: str | None = None
    """Description of the final screen state."""

    judge_reasoning: str | None = None
    """The judge model's reasoning for its decision."""

    final_screenshot_path: str | None = None
    """Path to the saved final screenshot."""

    agent_screenshot_paths: list[str] = []
    """List of screenshot paths from the agent."""


@task_registry.register('android_set_alarm_task')
async def run_set_alarm_task(task_input: SetAlarmTaskInput) -> SetAlarmTaskOutput:
    """Run the Set Alarm Task.

    This task:
    1. Constructs a detailed prompt for the agent
    2. Runs the Android GUI Agent to set the alarm
    3. Takes a final screenshot
    4. Uses a judge model to verify if the alarm was set correctly
    5. Returns the score based on the judge's evaluation
    """
    ctx = require_context()

    # Build the user prompt from template
    user_prompt = USER_PROMPT_TEMPLATE.format(
        alarm_time=task_input.alarm_time,
    )

    # If task_input has a user_prompt override, use it
    if task_input.user_prompt:
        user_prompt = task_input.user_prompt

    logger.debug(f'Running Set Alarm Task: alarm_time={task_input.alarm_time}')

    # Run the agent with screenshot_dir if configured
    agent_kwargs: dict = {'user_prompt': user_prompt}
    if task_input.screenshot_dir:
        agent_kwargs['screenshot_dir'] = task_input.screenshot_dir

    agent_output = await task_input.run_agent(**agent_kwargs)

    # Extract screenshot paths from agent output
    agent_screenshot_paths: list[str] = []
    if hasattr(agent_output, 'screenshot_paths'):
        paths = getattr(agent_output, 'screenshot_paths', [])
        agent_screenshot_paths = list(paths) if paths else []

    logger.info('Agent completed, proceeding to judge evaluation')

    # Get final screenshot for judge evaluation
    final_screenshot: str | None = None
    final_screenshot_path: str | None = None
    try:
        runtime = await ctx.get_runtime('android')
        screenshot_response = await runtime.gui_screenshot()
        final_screenshot = screenshot_response.image if screenshot_response.success else None

        # Save final screenshot to disk if screenshot_dir is configured
        if final_screenshot and task_input.screenshot_dir:
            try:
                os.makedirs(task_input.screenshot_dir, exist_ok=True)
                timestamp = datetime.now().strftime('%H%M%S')
                filename = f'final_{timestamp}_judge.png'
                final_screenshot_path = os.path.join(task_input.screenshot_dir, filename)

                # Fix base64 padding if needed
                screenshot_data = final_screenshot
                padding = len(screenshot_data) % 4
                if padding:
                    screenshot_data += '=' * (4 - padding)

                image_data = base64.b64decode(screenshot_data)
                with open(final_screenshot_path, 'wb') as f:
                    f.write(image_data)
                logger.info(f'Final screenshot saved: {final_screenshot_path}')
            except Exception as e:
                logger.warning(f'Failed to save final screenshot: {e}')
                final_screenshot_path = None
    except Exception as e:
        logger.warning(f'Failed to get final screenshot: {e}')

    # Use judge client to evaluate the result
    judge_success = False
    judge_reasoning = None

    if final_screenshot:
        try:
            # Get the judge client (fallback to main if not configured)
            try:
                judge_client = ctx.get_openai_client('judge')
            except Exception:
                logger.debug('Judge client not configured, using main client')
                judge_client = ctx.openai_client

            judge_prompt = JUDGE_PROMPT_TEMPLATE.format(alarm_time=task_input.alarm_time)

            judge_response = await judge_client.create_completion(
                messages=[
                    {
                        'role': 'user',
                        'content': [
                            {'type': 'text', 'text': judge_prompt},
                            {
                                'type': 'image_url',
                                'image_url': {'url': f'data:image/png;base64,{final_screenshot}'},
                            },
                        ],
                    }
                ],
            )

            if judge_response.choices and judge_response.choices[0].message.content:
                judge_content = judge_response.choices[0].message.content
                logger.debug(f'Judge response: {judge_content}')

                # Parse JSON from judge response
                try:
                    # Try to extract JSON from the response
                    json_start = judge_content.find('{')
                    json_end = judge_content.rfind('}') + 1
                    if json_start >= 0 and json_end > json_start:
                        json_str = judge_content[json_start:json_end]
                        judge_result = json.loads(json_str)
                        judge_success = judge_result.get('success', False)
                        judge_reasoning = judge_result.get('reasoning', '')
                except json.JSONDecodeError as e:
                    logger.warning(f'Failed to parse judge JSON: {e}')
                    judge_reasoning = judge_content

        except Exception as e:
            logger.warning(f'Judge evaluation failed: {e}')
            judge_reasoning = f'Judge evaluation failed: {e}'

    # Calculate final score based purely on judge evaluation
    score = 1.0 if judge_success else 0.0

    # If we couldn't get a screenshot or run the judge, score is 0
    if final_screenshot is None:
        score = 0.0
        judge_reasoning = 'Could not capture final screenshot for verification'

    logger.info(f'Task result: score={score}, judge_success={judge_success}')

    return SetAlarmTaskOutput(
        score=score,
        alarm_set=judge_success,
        final_screen_description=judge_reasoning,
        judge_reasoning=judge_reasoning,
        final_screenshot_path=final_screenshot_path,
        agent_screenshot_paths=agent_screenshot_paths,
    )
