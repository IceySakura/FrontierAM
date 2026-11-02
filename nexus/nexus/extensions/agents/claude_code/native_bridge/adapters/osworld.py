"""AGS OSWorld Claude Code native bridge adapter."""

from __future__ import annotations

from nexus.runtime.providers.ags.osworld import (
    AGSOSWorldExecutorRuntime,
    AGSOSWorldRuntime,
    require_ags_osworld_runtime,
)

from .posix import BasePosixClaudeNativeBridgeAdapter


class OSWorldClaudeNativeBridgeAdapter(BasePosixClaudeNativeBridgeAdapter):
    """Install and run Claude Code on AGS OSWorld."""

    def __init__(self, runtime: AGSOSWorldExecutorRuntime) -> None:
        super().__init__(require_ags_osworld_runtime(runtime))
        self.runtime: AGSOSWorldRuntime


__all__ = ['OSWorldClaudeNativeBridgeAdapter']
