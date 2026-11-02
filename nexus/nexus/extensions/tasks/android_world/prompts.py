"""Android World prompt adapted to Nexus tool calls."""

from __future__ import annotations

from nexus.extensions.tools.mobile.mobile_actions import (
    MOBILE_ACTION_SPACE,
    MOBILE_ACTION_TYPES,
    MOBILE_PARAMETER_GUIDE,
    MOBILE_TERMINATION_GUIDANCE,
)


def mobile_computer_tool_prompt(*, screen_width: int = 1080, screen_height: int = 1920) -> str:
    """Return the Android World native (function-calling) mobile prompt.

    Actions are the HyData-aligned ``mobile/*`` tools; coordinates are normalized ``[0, 1]``
    (the tools scale to real pixels internally, so the prompt is resolution-agnostic —
    ``screen_width`` / ``screen_height`` are accepted for signature compatibility but unused).
    """
    return f"""\
You will act as an agent which follows my instruction and performs mobile phone tasks as instructed.
For each step, you will get an observation of an image, which is a screenshot of the phone screen, and you will predict the next action based on the image.

{MOBILE_ACTION_SPACE}
{MOBILE_PARAMETER_GUIDE}
All coordinates are normalized to [0, 1] as fractions of the screen: the top-left corner is [0, 0] and the bottom-right corner is [1, 1].

For every step, call one of the mobile tools with the appropriate parameters.
{MOBILE_TERMINATION_GUIDANCE}
Do not easily give up; try your best to complete the task before finishing.
"""


__all__ = [
    'MOBILE_ACTION_SPACE',
    'MOBILE_ACTION_TYPES',
    'MOBILE_PARAMETER_GUIDE',
    'MOBILE_TERMINATION_GUIDANCE',
    'mobile_computer_tool_prompt',
]
