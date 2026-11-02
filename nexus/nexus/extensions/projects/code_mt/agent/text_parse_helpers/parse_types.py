"""Shared data types for terminus2 response parsers (JSON and XML)."""

from dataclasses import dataclass


@dataclass
class ParsedCommand:
    """A parsed command from the agent response."""

    keystrokes: str
    duration: float


@dataclass
class ParseResult:
    """Result of parsing an agent response."""

    commands: list[ParsedCommand]
    is_task_complete: bool
    error: str
    warning: str
    analysis: str = ''
    plan: str = ''
