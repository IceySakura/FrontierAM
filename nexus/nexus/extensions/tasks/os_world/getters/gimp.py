"""GIMP getters."""

from __future__ import annotations

from typing import Any

from nexus.extensions.tasks.os_world.evaluation.registry import getter_registry
from nexus.extensions.tasks.os_world.getters.file import get_vm_file


@getter_registry.register()
async def get_gimp_config_file(runtime: Any, config: dict[str, Any]) -> str | None:
    file_name = str(config.get('file_name', 'gimprc'))
    dest = str(config.get('dest', file_name))
    result = await get_vm_file(
        runtime,
        {
            **config,
            'path': f'/home/user/.config/GIMP/2.10/{file_name}',
            'dest': dest,
        },
    )
    return str(result) if result is not None else None
