"""Apple-specific GUI integration configuration helpers."""

from __future__ import annotations

import os

from dataclasses import dataclass
from typing import Any

from tests.integration.gui_agent.config_common import (
    DEFAULT_BASE_IMAGE,
    DEFAULT_MODEL,
    OpenAISettings,
    build_identity_gui_run_config,
    env_flag,
    env_int,
    resolve_openai_settings,
)


DEFAULT_CHROME_COMMAND = (
    'rm -rf /tmp/google-chrome-profile; exec google-chrome --no-sandbox --disable-gpu '
    '--no-first-run --no-default-browser-check '
    '--user-data-dir=/tmp/google-chrome-profile --window-position=0,0 --window-size=1280,800 '
    '--start-maximized --new-window https://www.apple.com'
)
DEFAULT_OBSERVE_BEFORE_CLICK_SECONDS = 10
DEFAULT_OBSERVE_AFTER_CLICK_SECONDS = 10


@dataclass(frozen=True)
class ObservationSettings:
    """Settings that slow down the Apple demo for live observation."""

    observe: bool
    before_click_seconds: int
    after_click_seconds: int


def resolve_observation_settings(env: dict[str, str] | None = None) -> ObservationSettings:
    """Resolve optional observation delays from the environment."""
    environment = env or dict(os.environ)

    return ObservationSettings(
        observe=env_flag(environment, 'NEXUS_GUI_OBSERVE'),
        before_click_seconds=env_int(
            environment, 'NEXUS_GUI_OBSERVE_BEFORE_CLICK_SECONDS', DEFAULT_OBSERVE_BEFORE_CLICK_SECONDS
        ),
        after_click_seconds=env_int(
            environment, 'NEXUS_GUI_OBSERVE_AFTER_CLICK_SECONDS', DEFAULT_OBSERVE_AFTER_CLICK_SECONDS
        ),
    )


def build_user_prompt(
    chrome_command: str = DEFAULT_CHROME_COMMAND,
    *,
    observe: bool = False,
    before_click_seconds: int = DEFAULT_OBSERVE_BEFORE_CLICK_SECONDS,
    after_click_seconds: int = DEFAULT_OBSERVE_AFTER_CLICK_SECONDS,
) -> str:
    """Build the Apple homepage prompt used by the GUI integration demo."""
    prompt = (
        'Use the desktop GUI tools to complete this task in order:\n'
        '1. Call the desktop/start tool with '
        '{"desktop_session":"gnome-session","screen_width":1280,"screen_height":800,'
        '"session_options":{"gnome_session_name":"ubuntu"}} '
        'to start the Ubuntu desktop session.\n'
        '2. Call the desktop/screen_size tool to confirm the desktop resolution.\n'
        f'3. Launch the app command: {chrome_command}\n'
    )
    if observe:
        prompt += f'4. Call desktop/bash with command: sleep {before_click_seconds}\n'
        prompt += (
            '5. Call desktop/screenshot.\n'
            '6. Click the center of the blue "Learn more" button by calling desktop/computer_13 '
            'with action_type CLICK and parameters {"x":555,"y":400}. '
            'This coordinate is for the fixed 1280x800 Google Chrome window; do not click the phone image.\n'
        )
        prompt += f'7. Call desktop/bash with command: sleep {after_click_seconds}\n'
        prompt += '8. Call desktop/screenshot again to confirm the page after the click.\n'
        prompt += '9. Reply with a short summary and stop calling tools.\n'
        prompt += 'Use desktop/bash only for the requested sleep commands. Google Chrome is already installed in the runtime environment.'
        return prompt

    prompt += (
        f'4. Call desktop/bash with command: sleep {before_click_seconds}\n'
        '5. Call desktop/screenshot.\n'
        '6. Click the center of the blue "Learn more" button by calling desktop/computer_13 '
        'with action_type CLICK and parameters {"x":555,"y":400}. '
        'This coordinate is for the fixed 1280x800 Google Chrome window; do not click the phone image.\n'
        f'7. Call desktop/bash with command: sleep {after_click_seconds}\n'
        '8. Call desktop/screenshot again to confirm the page after the click.\n'
        '9. Reply with a short summary and stop calling tools.\n'
        'Use desktop/bash only for the requested sleep commands. Google Chrome is already installed in the runtime environment.'
    )
    return prompt


def build_run_config(
    *,
    api_key: str,
    base_url: str = '',
    model: str = DEFAULT_MODEL,
    observe: bool = False,
    before_click_seconds: int = DEFAULT_OBSERVE_BEFORE_CLICK_SECONDS,
    after_click_seconds: int = DEFAULT_OBSERVE_AFTER_CLICK_SECONDS,
) -> dict[str, Any]:
    """Build the `RunInput` payload for the Apple GUI integration flow."""
    return build_identity_gui_run_config(
        api_key=api_key,
        base_url=base_url,
        model=model,
        user_prompt=build_user_prompt(
            observe=observe,
            before_click_seconds=before_click_seconds,
            after_click_seconds=after_click_seconds,
        ),
        max_iterations=12,
        tools=[
            'desktop/start',
            'desktop/screen_size',
            'desktop/screenshot',
            'desktop/launch_app',
            'desktop/computer_13',
            'desktop/bash',
        ],
    )


__all__ = [
    'DEFAULT_BASE_IMAGE',
    'DEFAULT_CHROME_COMMAND',
    'DEFAULT_MODEL',
    'DEFAULT_OBSERVE_AFTER_CLICK_SECONDS',
    'DEFAULT_OBSERVE_BEFORE_CLICK_SECONDS',
    'ObservationSettings',
    'OpenAISettings',
    'build_run_config',
    'build_user_prompt',
    'resolve_observation_settings',
    'resolve_openai_settings',
]
