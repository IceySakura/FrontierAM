from __future__ import annotations

from copy import deepcopy
from typing import Any

from nexus.trace.events import SpanEndEvent
from nexus.trace.processors.base import SpanProcessor
from nexus.trace.types import Span


class ProviderCallCaptureProcessor(SpanProcessor):
    """Capture provider-native call diagnostics for tests only."""

    def __init__(self) -> None:
        self._calls: list[dict[str, Any]] = []

    def on_end(self, span: Span, event: SpanEndEvent) -> None:
        del event
        if span.name != 'openai_completion':
            return
        span_provider_call = span.attributes.get('provider_call')
        if not isinstance(span_provider_call, dict):
            return

        provider_io = deepcopy(span_provider_call)
        provider_io.setdefault('attempts', [])
        error = deepcopy(provider_io.get('error'))
        if error is not None and not isinstance(error, dict):
            error = {'message': str(error)}
        if span.status.code == 'ERROR':
            if error is None:
                error = {}
            if span.status.message is not None:
                error['message'] = span.status.message

        duration_ms = None
        if span.end_time_unix_nano is not None:
            duration_ms = (span.end_time_unix_nano - span.start_time_unix_nano) / 1_000_000

        self._calls.append({
            'span_id': span.span_id,
            'client_key': span.attributes.get('_client_key'),
            'phase': span.attributes.get('_chat_phase'),
            'model': provider_io.get('model'),
            'provider': provider_io.get('provider'),
            'protocol': provider_io.get('protocol'),
            'provider_io': provider_io,
            'start_time_unix_nano': span.start_time_unix_nano,
            'end_time_unix_nano': span.end_time_unix_nano,
            'duration_ms': duration_ms,
            'status': span.status.model_dump(),
            'error': error,
        })

    def get_result(self) -> list[dict[str, Any]]:
        return deepcopy(self._calls)
