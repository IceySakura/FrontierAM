"""Terminal infrastructure error types."""

from __future__ import annotations


class TerminalInfraError(RuntimeError):
    """Failure that leaves the terminal channel unusable.

    Callers use the fatal flag to stop execution when the terminal cannot recover.
    """

    def __init__(self, message: str, *, fatal: bool = True) -> None:
        super().__init__(message)
        self.fatal = fatal
