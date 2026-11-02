"""VS Code getters."""

from __future__ import annotations

from typing import Any

from nexus.extensions.tasks.os_world.evaluation.errors import EvaluatorInfrastructureError
from nexus.extensions.tasks.os_world.evaluation.registry import getter_registry
from nexus.extensions.tasks.os_world.getters.file import get_vm_file
from nexus.extensions.tasks.os_world.getters.runtime import response_returncode, run_python


@getter_registry.register()
async def get_vscode_config(runtime: Any, config: dict[str, Any]) -> str:
    """Query the benchmark extension in the current VS Code window.

    This file is produced by the extension, not the agent. Historical workspace
    caches must not be used as evidence of the current window's configuration.
    """
    command = config.get('vscode_extension_command')
    if not isinstance(command, str) or not command.strip():
        raise EvaluatorInfrastructureError('VS Code getter requires vscode_extension_command')
    script = f"""
import time
from pathlib import Path
import pyautogui

output = Path({str(config['path'])!r})
def fingerprint():
    try:
        stat = output.stat()
        return stat.st_ino, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size
    except FileNotFoundError:
        return None

before = fingerprint()
pyautogui.hotkey('ctrl', 'shift', 'p')
time.sleep(0.5)
pyautogui.write({command!r}, interval=0.05)
pyautogui.press('enter')
deadline = time.monotonic() + 5
while time.monotonic() < deadline:
    time.sleep(0.25)
    current = fingerprint()
    if current is not None and current != before:
        break
else:
    raise RuntimeError('VS Code extension did not produce a fresh configuration file')
"""
    result = await run_python(runtime, script, timeout=30)
    if response_returncode(result) != 0:
        raise EvaluatorInfrastructureError('VS Code extension query failed or produced no fresh configuration file')
    try:
        return await get_vm_file(runtime, config)
    except FileNotFoundError as exc:
        raise EvaluatorInfrastructureError('VS Code extension configuration file is missing after query') from exc
