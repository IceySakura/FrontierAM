"""Windows Clock getters for WAA evaluation.

All functions have the unified async signature:
    async def get_xxx(runtime: AGSRawWindowsAgentArenaRuntime, config: dict) -> Any
"""

from __future__ import annotations

import logging

from typing import Any

from nexus.extensions.tasks.waa.registry import getter_registry
from nexus.runtime.providers.ags.waa import AGSRawWindowsAgentArenaRuntime


logger = logging.getLogger('nexus.waa.getters.windows_clock')


@getter_registry.register()
async def get_check_if_timer_started(runtime: AGSRawWindowsAgentArenaRuntime, config: dict) -> Any:
    """Check if a timer with the specified duration has been started.

    Returns 'True' if timer exists (HTTP 200), 'False' otherwise.
    """
    try:
        result = await runtime.check_if_timer_started(config['hours'], config['minutes'], config['seconds'])
        if isinstance(result, dict) and 'error' in result:
            return 'False'
        return 'True'
    except Exception:
        return 'False'


@getter_registry.register()
async def get_check_if_world_clock_exists(runtime: AGSRawWindowsAgentArenaRuntime, config: dict) -> Any:
    """Check if a world clock entry exists for the specified city/country.

    Returns 'True' if world clock exists (HTTP 200), 'False' otherwise.
    """
    try:
        result = await runtime.check_if_world_clock_exists(config['city'], config['country'])
        if isinstance(result, dict) and 'error' in result:
            return 'False'
        return 'True'
    except Exception:
        return 'False'
