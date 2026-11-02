"""VLC getters for WAA evaluation.

All functions have the unified async signature:
    async def get_xxx(runtime: AGSRawWindowsAgentArenaRuntime, config: dict) -> Any
"""

from __future__ import annotations

import asyncio
import logging

from typing import Any

from nexus.extensions.tasks.waa.registry import getter_registry
from nexus.runtime.providers.ags.waa import AGSRawWindowsAgentArenaRuntime


logger = logging.getLogger('nexus.waa.getters.vlc')


@getter_registry.register()
async def get_vlc_config(runtime: AGSRawWindowsAgentArenaRuntime, config: dict) -> Any:
    """Download the VLC configuration file (vlcrc) from the VM."""
    from nexus.extensions.tasks.waa.setup import cache_dir

    result = await runtime.execute(
        ['python', '-c', r"import os; print(os.path.expanduser('~\\AppData\\Roaming\\vlc\\vlcrc'))"], shell=False
    )
    config_path = result['output'].strip()
    local_path = cache_dir / config['dest']
    content = await runtime.get_file(config_path)
    await asyncio.to_thread(local_path.write_bytes, content)
    return str(local_path)
