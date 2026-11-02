"""DevCloud macOS Claude Code native bridge adapter."""

from __future__ import annotations

from nexus.runtime.providers.devcloud.client import DevCloudRuntime

from .posix import BasePosixClaudeNativeBridgeAdapter


class DevCloudMacOSClaudeNativeBridgeAdapter(BasePosixClaudeNativeBridgeAdapter):
    """Install and run Claude Code on DevCloud macOS."""

    def __init__(self, runtime: DevCloudRuntime) -> None:
        super().__init__(runtime)


__all__ = ['DevCloudMacOSClaudeNativeBridgeAdapter']
