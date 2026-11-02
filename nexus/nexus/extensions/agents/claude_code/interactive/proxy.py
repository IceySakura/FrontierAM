"""Interactive PTY Anthropic proxy with recent_events tracking.

Subclass of the base ``AnthropicProxy`` that overrides request/response
observer hooks to build a structured event log. Event extraction, truncation,
and buffer state live in ``events.py``.
"""

from __future__ import annotations

from typing import Any

from nexus.runtime.server.services.anthropic_proxy.server import AnthropicProxy

from .events import EventBuffer


class PtyAnthropicProxy(AnthropicProxy):
    """AnthropicProxy variant with structured tool-call event tracking.

    Delegates event state to ``EventBuffer``; hooks into the base class's
    request/response observer methods — no route duplication.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.event_buffer = EventBuffer()

    def mark_turn_start(self) -> None:
        self.event_buffer.mark_turn_start()

    def get_event_delta(self) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        return self.event_buffer.get_delta()

    def commit_event_cursor(self, seq: int) -> None:
        self.event_buffer.commit(seq)

    def _on_request_body(self, body: dict[str, Any]) -> None:
        self.event_buffer.append_from_request(body)

    def _on_response(self, response: Any) -> None:
        self.event_buffer.append_from_response(response)


__all__ = ['PtyAnthropicProxy']
