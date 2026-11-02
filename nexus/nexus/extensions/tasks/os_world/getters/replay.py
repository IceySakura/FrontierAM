"""Replay getter placeholder."""

from __future__ import annotations

from typing import Any

from nexus.extensions.tasks.os_world.evaluation.registry import getter_registry


@getter_registry.register()
async def get_replay(runtime: Any, config: dict[str, Any]) -> None:
    return None
