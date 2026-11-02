"""General getters for WAA evaluation.

All functions have the unified async signature:
    async def get_xxx(runtime: AGSRawWindowsAgentArenaRuntime, config: dict) -> Any
"""

from __future__ import annotations

import logging

from typing import Any

from nexus.extensions.tasks.waa.registry import getter_registry
from nexus.runtime.providers.ags.waa import AGSRawWindowsAgentArenaRuntime


logger = logging.getLogger('nexus.waa.getters.general')


@getter_registry.register()
async def get_vm_command_line(runtime: AGSRawWindowsAgentArenaRuntime, config: dict) -> Any:
    """Execute a command on the VM and return stdout."""
    command = config['command']
    shell = config.get('shell', False)
    result = await runtime.execute(command, shell=shell)
    return result.get('output')
