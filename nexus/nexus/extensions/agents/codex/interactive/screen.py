"""Virtual terminal for the Codex TUI viewport.

Thin wrapper over the shared ``VirtualScreen`` (pyte on a dedicated worker
thread). The shared one is used deliberately: it keeps pyte parsing off the event
loop, which matters because the Codex TUI repaints the whole screen continuously
and a single chunk can cost hundreds of milliseconds to parse inline.
"""

from __future__ import annotations

from nexus.runtime.executors.terminal.screen import VirtualScreen


class CodexScreen:
    """Feed raw PTY bytes; read the current viewport as plain text."""

    def __init__(self, rows: int = 50, cols: int = 200) -> None:
        self._screen = VirtualScreen(rows=rows, cols=cols)

    def feed(self, data: bytes) -> None:
        self._screen.feed(data)

    async def latest_screen(self) -> str:
        """Current viewport, ANSI removed. Not the full scrollback."""
        return await self._screen.display()

    def close(self) -> None:
        self._screen.close()


__all__ = ['CodexScreen']
