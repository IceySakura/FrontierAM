"""SmartRun macOS Codex native bridge adapter."""

from nexus.runtime.bridge.adapters.smartrun_macos import detached_shell_command
from nexus.runtime.providers.smartrun.client import SmartRunRuntime

from .posix import BasePosixCodexNativeBridgeAdapter


class SmartRunMacOSCodexNativeBridgeAdapter(BasePosixCodexNativeBridgeAdapter):
    def __init__(self, runtime: SmartRunRuntime) -> None:
        super().__init__(runtime)

    def _launch_detached_shell(self, command: str, pid_file: str, *, strict_cleanup: bool) -> str:
        return detached_shell_command(command, pid_file)
