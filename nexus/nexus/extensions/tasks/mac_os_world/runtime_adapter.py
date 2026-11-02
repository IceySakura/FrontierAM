"""Runtime adapter for macOSWorld-specific executor compatibility."""

from __future__ import annotations

import asyncio

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from nexus.runtime.executors.models.gui import GuiScreenshotResponse


class MacOSWorldRuntimeAdapter:
    """Adapt macOSWorld agent execution without taking ownership of the runtime."""

    def __init__(self, wrapped_runtime: Any, *, screenshot_timeout: float) -> None:
        self.wrapped_runtime = wrapped_runtime
        self.screenshot_timeout = float(screenshot_timeout)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.wrapped_runtime, name)

    async def run_command(self, command: str, **kwargs: Any) -> Any:
        # DevCloud's login shell selects pyenv Python 3.7 without PyAutoGUI.
        # Prefer the Python installation used by the macOS GUI server.
        command = 'export PATH="/usr/local/bin:$PATH"\n' + command
        return await self.wrapped_runtime.run_command(command=command, **kwargs)

    async def call_executor(self, executor_name: str, arg: Any) -> Any:
        if executor_name != 'gui_screenshot':
            return await self.wrapped_runtime.call_executor(executor_name, arg)
        return (await self.gui_screenshot()).model_dump()

    async def gui_screenshot(self) -> Any:
        try:
            return await asyncio.wait_for(
                self.wrapped_runtime.gui_screenshot(),
                timeout=self.screenshot_timeout,
            )
        except asyncio.TimeoutError:
            return GuiScreenshotResponse(
                success=False,
                error=f'macOSWorld screenshot exceeded {self.screenshot_timeout:g}s',
            )

    async def drain_trace(self, timeout: float | None = None) -> Any:
        drain = getattr(self.wrapped_runtime, 'drain_trace', None)
        if callable(drain):
            return await drain(timeout=timeout)
        return None


@contextmanager
def mac_os_world_runtime_adapter(
    ctx: Any,
    name: str,
    runtime: Any,
    *,
    screenshot_timeout: float,
) -> Iterator[MacOSWorldRuntimeAdapter]:
    """Temporarily install one non-owning adapter in the context runtime cache."""

    previous = ctx._runtimes.get(name)
    existing = ctx._runtimes.get(name)
    if isinstance(existing, MacOSWorldRuntimeAdapter):
        yield existing
        return
    adapter = (
        runtime
        if isinstance(runtime, MacOSWorldRuntimeAdapter)
        else MacOSWorldRuntimeAdapter(runtime, screenshot_timeout=screenshot_timeout)
    )
    ctx._runtimes[name] = adapter
    try:
        yield adapter
    finally:
        if ctx._runtimes.get(name) is adapter:
            if previous is None:
                ctx._runtimes.pop(name, None)
            else:
                ctx._runtimes[name] = previous


__all__ = ['MacOSWorldRuntimeAdapter', 'mac_os_world_runtime_adapter']
