"""Internal data models shared by the detector engine and individual rules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Finding:
    message_index: int
    message: str
    path: list[Any] | None = None
    excerpt: Any = ''
    tool_call_index: int | None = None
    tool_call_id: str | None = None
    origin_message_index: int | None = None
    confidence: str = 'established'
    is_issue: bool = True
    severity: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)
    quotes: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class CallEvent:
    message_index: int
    tool_call_index: int
    call: dict[str, Any]
    function: dict[str, Any] | None
    call_id: str | None
    name: str
    arguments: dict[str, Any] | None
    arguments_error: str | None
    result_index: int | None = None
    result: dict[str, Any] | None = None
