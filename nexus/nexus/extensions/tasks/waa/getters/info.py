"""VM info getters for WAA evaluation.

All functions have the unified async signature:
    async def get_xxx(runtime: AGSRawWindowsAgentArenaRuntime, config: dict) -> Any
"""

from __future__ import annotations

import asyncio
import logging

from typing import Any

from nexus.extensions.tasks.waa.registry import getter_registry
from nexus.runtime.providers.ags.waa import AGSRawWindowsAgentArenaRuntime


logger = logging.getLogger('nexus.waa.getters.info')


@getter_registry.register()
async def get_vm_wallpaper(runtime: AGSRawWindowsAgentArenaRuntime, config: dict) -> Any:
    """Download the current VM wallpaper to local cache."""
    from nexus.extensions.tasks.waa.setup import cache_dir

    local_path = cache_dir / config['dest']
    content = await runtime.wallpaper()
    await asyncio.to_thread(local_path.write_bytes, content)
    return str(local_path)
