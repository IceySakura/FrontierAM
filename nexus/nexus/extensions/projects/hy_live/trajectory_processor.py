"""Capture the cerebellum's latest actual request and unmodified response."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from nexus.trace import Span, SpanEndEvent, SpanProcessor, SpanStartEvent


class MainTrajectoryProcessor(SpanProcessor):
    """Expose a main-client message view without reconstructing agent history.

    The latest request replaces the view, so rewritten histories are not merged
    into a fictitious linear trajectory. A failed request has no appended answer.
    Use per-call tracing when all earlier branches or rejected outputs are needed.
    """

    def __init__(self, *, client_key: str = 'main') -> None:
        self.client_key = client_key
        self._span_id: str | None = None
        self._messages: list[dict[str, Any]] = []

    def on_start(self, span: Span, event: SpanStartEvent) -> None:
        if span.name != 'openai_completion' or span.attributes.get('_client_key') != self.client_key:
            return
        self._span_id = span.span_id
        self._messages = deepcopy(span.attributes['inputs']['messages'])

    def on_end(self, span: Span, event: SpanEndEvent) -> None:
        if span.span_id != self._span_id:
            return
        choices = span.attributes.get('outputs', {}).get('choices', [])
        if choices:
            self._messages.append(deepcopy(choices[0]['message']))

    def snapshot_result(self) -> list[dict[str, Any]]:
        """Return an isolated snapshot, including an unanswered in-flight request."""
        return deepcopy(self._messages)

    def get_result(self) -> list[dict[str, Any]]:
        """Return the final isolated message view."""
        return self.snapshot_result()
