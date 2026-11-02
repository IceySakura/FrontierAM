"""V2 task-side prompt and desktop tool binding; agents retain their own guides."""

from nexus.extensions.tools.desktop.observation import desktop_screenshot_content_part
from nexus.extensions.tools.desktop.osworld_actions import (
    COMPUTER_13_ACTION_SPACE,
    COMPUTER_13_PARAMETER_GUIDE,
    OSWORLD_SCREEN_BOUNDS,
    PYAUTOGUI_SAFETY_POLICY,
)


RESOURCE_POLICY = (
    'System resources: total RAM is 64 GB; approximately 37 GB of disk space is initially free. '
    'This budget is shared with desktop applications and background services, not reserved for your process. '
    'Available memory and disk space change during execution: check free -h and df -h / before large operations. '
    'Avoid loading entire videos or datasets into memory, unnecessary CUDA/model downloads, '
    'unbounded parallel jobs, and large temporary files. Stream/chunk data and remove only your own disposable files.'
)


def build_os_world_agent_tools(action_space='pyautogui'):
    return [f'desktop/{action_space}', 'desktop/fixed_marker']


def build_os_world_system_prompt(*, action_space='pyautogui', client_password):
    actions = (
        f'{COMPUTER_13_ACTION_SPACE}\n{COMPUTER_13_PARAMETER_GUIDE}'
        if action_space == 'computer_13'
        else f'Use desktop/pyautogui to execute screenshot-grounded desktop actions.\n{PYAUTOGUI_SAFETY_POLICY}'
    )
    return (
        f'Complete the requested desktop task using the current screenshot.\n{actions}\n'
        f'{OSWORLD_SCREEN_BOUNDS}\n'
        f'{RESOURCE_POLICY}\n'
        'Use desktop/fixed_marker with DONE when finished, FAIL when infeasible, or WAIT to wait.\n'
        f"The computer's sudo password is '{client_password}'."
    )


def build_os_world_user_prompt(instruction, screenshot=None):
    text = f'You are asked to complete the following task: {instruction}'
    part = desktop_screenshot_content_part(screenshot) if screenshot is not None else None
    return [{'type': 'text', 'text': text}, part] if part else text
