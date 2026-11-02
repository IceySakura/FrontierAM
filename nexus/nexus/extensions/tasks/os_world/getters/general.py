"""General VM state getters for OSWorld evaluation."""

from __future__ import annotations

from typing import Any

from nexus.extensions.tasks.os_world.evaluation.registry import getter_registry
from nexus.extensions.tasks.os_world.getters.runtime import execute_command, response_stderr, response_stdout


@getter_registry.register()
async def get_vm_command_line(runtime: Any, config: dict[str, Any]) -> str | None:
    result = await execute_command(
        runtime,
        config['command'],
        shell=bool(config.get('shell', False)),
        timeout=int(config.get('timeout', 120)),
    )
    return response_stdout(result)


@getter_registry.register()
async def get_vm_command_error(runtime: Any, config: dict[str, Any]) -> str | None:
    result = await execute_command(
        runtime,
        config['command'],
        shell=bool(config.get('shell', False)),
        timeout=int(config.get('timeout', 120)),
    )
    return response_stderr(result)


@getter_registry.register()
async def get_vm_terminal_output(runtime: Any, config: dict[str, Any]) -> Any:
    """Return the runtime terminal response unchanged and propagate transport errors."""
    if hasattr(runtime, 'terminal_output'):
        return await runtime.terminal_output()
    if hasattr(runtime, 'capture_pane'):
        response = await runtime.capture_pane()
        return getattr(response, 'content', None) or getattr(response, 'text', None) or ''
    return ''
