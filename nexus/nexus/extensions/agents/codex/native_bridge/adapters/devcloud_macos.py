"""DevCloud macOS Codex native bridge adapter."""

from nexus.runtime.providers.devcloud.client import DevCloudRuntime

from .posix import BasePosixCodexNativeBridgeAdapter


class DevCloudMacOSCodexNativeBridgeAdapter(BasePosixCodexNativeBridgeAdapter):
    def __init__(self, runtime: DevCloudRuntime) -> None:
        super().__init__(runtime)


__all__ = ['DevCloudMacOSCodexNativeBridgeAdapter']
