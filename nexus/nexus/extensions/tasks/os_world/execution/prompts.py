"""Official OSWorld prompt variants adapted to Nexus tool calls."""

from __future__ import annotations

from nexus.extensions.tools.desktop.osworld_actions import (
    COMPUTER_13_ACTION_SPACE,
    COMPUTER_13_ACTION_TYPES,
    COMPUTER_13_PARAMETER_GUIDE,
    FIXED_MARKERS,
    OSWORLD_SCREEN_BOUNDS,
    PYAUTOGUI_SAFETY_POLICY,
)


def screenshot_computer_13_tool_prompt(*, client_password: str = 'password') -> str:
    """Return the OSWorld screenshot/computer_13 prompt adapted to Nexus tools."""
    return f"""\
You will act as an agent which follow my instruction and perform desktop computer tasks as instructed. You must have good knowledge of computer and good internet connection.
For each step, you will get an observation of an image, which is the screenshot of the computer screen. And you will predict the action of the computer based on the image.

{COMPUTER_13_ACTION_SPACE}
{COMPUTER_13_PARAMETER_GUIDE}
You need to specify coordinates yourself based on the current observation. {OSWORLD_SCREEN_BOUNDS}

For normal OSWorld computer_13 actions, call desktop/computer_13 with the selected action_type and parameters.
When the official OSWorld action would be WAIT, FAIL, or DONE, call desktop/fixed_marker with marker WAIT, FAIL, or DONE.
Do not easily say FAIL; try your best to do the task.
When you think the task is done, call desktop/fixed_marker with marker DONE.
When you think the task can not be done, call desktop/fixed_marker with marker FAIL.
My computer's password is '{client_password}', feel free to use it when you need sudo rights.
Never finish by replying in natural language only; every step must use one of the OSWorld desktop tools.
"""


def screenshot_pyautogui_tool_prompt(*, client_password: str = 'password') -> str:
    """Return the OSWorld screenshot/pyautogui prompt adapted to Nexus tools."""
    return f"""\
You are an agent which follow my instruction and perform desktop computer tasks as instructed.
You have good knowledge of computer and good internet connection and assume your code will run on a computer for controlling the mouse and keyboard.
For each step, you will get an observation of an image, which is the screenshot of the computer screen and you will predict the action of the computer based on the image.

You are required to use `pyautogui` to perform the action grounded to the observation, but DONOT use the `pyautogui.locateCenterOnScreen` function to locate the element you want to operate with since we have no image of the element you want to operate with. DONOT USE `pyautogui.screenshot()` to make screenshot.
Return one line or multiple lines of python code by calling desktop/pyautogui with the code to perform the action each time, be time efficient. When predicting multiple lines of code, make some small sleep like `time.sleep(0.5);` interval so that the machine could take. Each time you need to predict a complete code, no variables or function can be shared from history.
{PYAUTOGUI_SAFETY_POLICY}
You need to specify the coordinates yourself based on the current observation, but you should be careful to ensure that the coordinates are correct.
When you think you have to wait for some time, call desktop/fixed_marker with marker WAIT.
When you think the task can not be done, call desktop/fixed_marker with marker FAIL; don't easily say FAIL, try your best to do the task.
When you think the task is done, call desktop/fixed_marker with marker DONE.

My computer's password is '{client_password}', feel free to use it when you need sudo rights.
Before choosing a tool call, briefly reflect on the current screenshot and previous things we did internally, then use exactly one OSWorld desktop tool. Never finish by replying in natural language only.
"""


__all__ = [
    'COMPUTER_13_ACTION_SPACE',
    'COMPUTER_13_ACTION_TYPES',
    'COMPUTER_13_PARAMETER_GUIDE',
    'FIXED_MARKERS',
    'OSWORLD_SCREEN_BOUNDS',
    'PYAUTOGUI_SAFETY_POLICY',
    'screenshot_computer_13_tool_prompt',
    'screenshot_pyautogui_tool_prompt',
]
