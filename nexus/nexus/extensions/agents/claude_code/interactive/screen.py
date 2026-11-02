"""Virtual terminal screen for accurate viewport snapshots.

Wraps pyte to maintain a rows×cols screen buffer that correctly emulates
cursor movement, clears, and TUI redraws. Feed raw PTY bytes; read the
current viewport as plain text at any time.
"""

from __future__ import annotations

import pyte


class TerminalScreen:
    """Virtual terminal driven by pyte — provides ``latest_screen``."""

    def __init__(self, rows: int, cols: int) -> None:
        self._screen = pyte.Screen(cols, rows)
        self._stream = pyte.ByteStream(self._screen)

    def feed(self, data: bytes) -> None:
        """Feed raw PTY output into the emulator."""
        self._stream.feed(data)

    def render(self) -> str:
        """Current viewport as plain text (rows × cols, no ANSI)."""
        return '\n'.join(line.rstrip() for line in self._screen.display).rstrip()


__all__ = ['TerminalScreen']
