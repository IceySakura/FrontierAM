"""Reusable configuration helpers for the local GUI agent integration flow."""

from __future__ import annotations

import os

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from tests.integration.gui_agent.config_common import (
    DEFAULT_BASE_IMAGE,
    DEFAULT_MODEL,
    DEFAULT_RUNTIME_EXTRA_ARGS,
    DEFAULT_RUNTIME_NAME,
    OpenAISettings,
    build_identity_gui_run_config,
    env_flag,
    env_int,
    resolve_openai_settings,
)


DEFAULT_DEMO_FILE = '/tmp/nexus_gui_agent_demo.txt'
DEFAULT_DEMO_TEXT = 'hello from nexus desktop gui agent'
DEFAULT_SECONDARY_DEMO_FILE = '/tmp/nexus_gui_agent_secondary_demo.txt'
DEFAULT_SECONDARY_DEMO_TEXT = 'hello from nexus secondary desktop window'
DEFAULT_OBSERVE_AFTER_LAUNCH_SECONDS = 10
DEFAULT_OBSERVE_BEFORE_CLOSE_SECONDS = 10


@dataclass(frozen=True)
class ObservationSettings:
    """Settings that slow down the local GUI demo for live observation."""

    observe: bool
    after_launch_seconds: int
    before_close_seconds: int


def resolve_observation_settings(env: Mapping[str, str] | None = None) -> ObservationSettings:
    """Resolve optional observation delays from the environment."""
    environment = env or os.environ

    return ObservationSettings(
        observe=env_flag(environment, 'NEXUS_GUI_OBSERVE'),
        after_launch_seconds=env_int(
            environment, 'NEXUS_GUI_OBSERVE_AFTER_LAUNCH_SECONDS', DEFAULT_OBSERVE_AFTER_LAUNCH_SECONDS
        ),
        before_close_seconds=env_int(
            environment, 'NEXUS_GUI_OBSERVE_BEFORE_CLOSE_SECONDS', DEFAULT_OBSERVE_BEFORE_CLOSE_SECONDS
        ),
    )


def build_user_prompt(
    file_path: str = DEFAULT_DEMO_FILE,
    file_text: str = DEFAULT_DEMO_TEXT,
    secondary_file_path: str = DEFAULT_SECONDARY_DEMO_FILE,
    secondary_file_text: str = DEFAULT_SECONDARY_DEMO_TEXT,
    *,
    observe: bool = False,
    after_launch_seconds: int = DEFAULT_OBSERVE_AFTER_LAUNCH_SECONDS,
    before_close_seconds: int = DEFAULT_OBSERVE_BEFORE_CLOSE_SECONDS,
) -> str:
    """Build the deterministic GUI demo prompt used by local integration runs."""
    prompt = (
        'Use the desktop GUI tools to complete this task in order:\n'
        '1. Call the desktop/start tool with '
        '{"desktop_session":"gnome-session","session_options":{"gnome_session_name":"ubuntu"}} '
        'to start the Ubuntu desktop session.\n'
        '2. Call the desktop/screen_size tool to confirm the desktop resolution.\n'
        '3. Launch the app command: xterm -title NexusPrimary -geometry 86x24+80+70 -fa Monospace -fs 12\n'
    )
    if observe:
        prompt += (
            f'4. Call desktop/bash with command: sleep {after_launch_seconds}\n'
            '5. Call desktop/screenshot.\n'
            '6. Call desktop/computer_13 with action_type CLICK and parameters {"x":220,"y":170}.\n'
            f'7. Call desktop/computer_13 with action_type TYPING and parameters {{"text":"printf %s {file_text!r} > {file_path}"}}.\n'
            '8. Call desktop/computer_13 with action_type PRESS and parameters {"key":"enter"}.\n'
            f'9. Call desktop/bash with command: cat {file_path} and confirm the content matches exactly.\n'
            '10. Launch the app command: xterm -title NexusSecondary -geometry 78x20+560+110 -fa Monospace -fs 12\n'
            '11. Call desktop/computer_13 with action_type CLICK and parameters {"x":690,"y":190}.\n'
            f'12. Call desktop/computer_13 with action_type TYPING and parameters {{"text":"printf %s {secondary_file_text!r} > {secondary_file_path}"}}.\n'
            '13. Call desktop/computer_13 with action_type PRESS and parameters {"key":"enter"}.\n'
            f'14. Call desktop/bash with command: cat {secondary_file_path} and confirm the content matches exactly.\n'
            "15. Launch the app command: xmessage -buttons Close:0 -geometry 300x110+420+300 'Nexus GUI dialog A'\n"
            "16. Launch the app command: xmessage -buttons Close:0 -geometry 300x110+760+340 'Nexus GUI dialog B'\n"
            f'17. Call desktop/bash with command: sleep {before_close_seconds}\n'
            '18. Call desktop/screenshot again and confirm multiple desktop windows are visible.\n'
            '19. Close the first dialog with desktop/close_app using app=Nexus GUI dialog A.\n'
            '20. Close the second dialog with desktop/close_app using app=Nexus GUI dialog B.\n'
            '21. Close the secondary xterm with desktop/close_app using app=NexusSecondary.\n'
            '22. Close the primary xterm with desktop/close_app using app=NexusPrimary.\n'
            '23. Call desktop/screenshot one final time.\n'
            '24. Reply with a short summary and stop calling tools.\n'
            'Use desktop/bash only for the requested sleep commands and the required file verification command.'
        )
        return prompt

    prompt += (
        '4. Call desktop/screenshot.\n'
        '5. Call desktop/computer_13 with action_type CLICK and parameters {"x":220,"y":170}.\n'
        f'6. Call desktop/computer_13 with action_type TYPING and parameters {{"text":"printf %s {file_text!r} > {file_path}"}}.\n'
        '7. Call desktop/computer_13 with action_type PRESS and parameters {"key":"enter"}.\n'
        f'8. Call desktop/bash with command: cat {file_path} and confirm the content matches exactly.\n'
        '9. Launch the app command: xterm -title NexusSecondary -geometry 78x20+560+110 -fa Monospace -fs 12\n'
        '10. Call desktop/computer_13 with action_type CLICK and parameters {"x":690,"y":190}.\n'
        f'11. Call desktop/computer_13 with action_type TYPING and parameters {{"text":"printf %s {secondary_file_text!r} > {secondary_file_path}"}}.\n'
        '12. Call desktop/computer_13 with action_type PRESS and parameters {"key":"enter"}.\n'
        f'13. Call desktop/bash with command: cat {secondary_file_path} and confirm the content matches exactly.\n'
        "14. Launch the app command: xmessage -buttons Close:0 -geometry 300x110+420+300 'Nexus GUI dialog A'\n"
        "15. Launch the app command: xmessage -buttons Close:0 -geometry 300x110+760+340 'Nexus GUI dialog B'\n"
        '16. Call desktop/screenshot again and confirm multiple desktop windows are visible.\n'
        '17. Close the first dialog with desktop/close_app using app=Nexus GUI dialog A.\n'
        '18. Close the second dialog with desktop/close_app using app=Nexus GUI dialog B.\n'
        '19. Close the secondary xterm with desktop/close_app using app=NexusSecondary.\n'
        '20. Close the primary xterm with desktop/close_app using app=NexusPrimary.\n'
        '21. Call desktop/screenshot one final time.\n'
        '22. Reply with a short summary and stop calling tools.'
    )
    return prompt


def build_run_config(
    *,
    api_key: str,
    base_url: str = '',
    model: str = DEFAULT_MODEL,
    observe: bool = False,
    after_launch_seconds: int = DEFAULT_OBSERVE_AFTER_LAUNCH_SECONDS,
    before_close_seconds: int = DEFAULT_OBSERVE_BEFORE_CLOSE_SECONDS,
) -> dict[str, Any]:
    """Build the `RunInput` payload for the local GUI integration flow."""
    return build_identity_gui_run_config(
        api_key=api_key,
        base_url=base_url,
        model=model,
        user_prompt=build_user_prompt(
            observe=observe,
            after_launch_seconds=after_launch_seconds,
            before_close_seconds=before_close_seconds,
        ),
        max_iterations=30,
        tools=[
            'desktop/start',
            'desktop/screen_size',
            'desktop/screenshot',
            'desktop/launch_app',
            'desktop/close_app',
            'desktop/computer_13',
            'desktop/bash',
        ],
    )


__all__ = [
    'DEFAULT_BASE_IMAGE',
    'DEFAULT_DEMO_FILE',
    'DEFAULT_DEMO_TEXT',
    'DEFAULT_MODEL',
    'DEFAULT_OBSERVE_AFTER_LAUNCH_SECONDS',
    'DEFAULT_OBSERVE_BEFORE_CLOSE_SECONDS',
    'DEFAULT_RUNTIME_EXTRA_ARGS',
    'DEFAULT_RUNTIME_NAME',
    'DEFAULT_SECONDARY_DEMO_FILE',
    'DEFAULT_SECONDARY_DEMO_TEXT',
    'ObservationSettings',
    'OpenAISettings',
    'build_run_config',
    'build_user_prompt',
    'resolve_observation_settings',
    'resolve_openai_settings',
]
