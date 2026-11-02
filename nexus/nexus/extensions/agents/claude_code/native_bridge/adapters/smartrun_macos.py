"""SmartRun macOS Claude native bridge adapter."""

from nexus.runtime.bridge.adapters.smartrun_macos import detached_shell_command
from nexus.runtime.providers.smartrun.client import SmartRunRuntime

from .posix import BasePosixClaudeNativeBridgeAdapter


class SmartRunMacOSClaudeNativeBridgeAdapter(BasePosixClaudeNativeBridgeAdapter):
    def __init__(self, runtime: SmartRunRuntime) -> None:
        super().__init__(runtime)

    def _launch_detached_shell(self, command: str, pid_file: str) -> str:
        return detached_shell_command(command, pid_file)
