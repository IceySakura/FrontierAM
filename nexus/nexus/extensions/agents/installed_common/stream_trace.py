from __future__ import annotations

import copy
import logging
import time

from dataclasses import dataclass
from typing import Any


logger = logging.getLogger(__name__)

MAX_MERGED_TEXT_BYTES = 64 * 1024


def _partial_text(event: dict[str, Any]) -> tuple[tuple[Any, ...], str] | None:
    if event.get('type') != 'stream_event':
        return None
    stream_event = event.get('event')
    if not isinstance(stream_event, dict) or stream_event.get('type') != 'content_block_delta':
        return None
    delta = stream_event.get('delta')
    if not isinstance(delta, dict) or delta.get('type') != 'text_delta':
        return None
    text = delta.get('text')
    if not isinstance(text, str):
        return None
    return (
        (
            event.get('session_id'),
            event.get('parent_tool_use_id'),
            stream_event.get('index'),
        ),
        text,
    )


@dataclass
class StreamTraceBuffer:
    pending_span_id: str | None = None
    pending_event: dict[str, Any] | None = None
    pending_key: tuple[Any, ...] | None = None
    pending_text_bytes: int = 0

    def push(self, span_id: str, event: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
        partial = _partial_text(event)
        if partial is None:
            ready = self.flush()
            ready.append((span_id, event))
            return ready

        key, text = partial
        text_bytes = len(text.encode('utf-8'))
        if text_bytes >= MAX_MERGED_TEXT_BYTES:
            ready = self.flush()
            ready.append((span_id, event))
            return ready

        if (
            self.pending_event is not None
            and self.pending_span_id == span_id
            and self.pending_key == key
            and self.pending_text_bytes + text_bytes <= MAX_MERGED_TEXT_BYTES
        ):
            self.pending_event['event']['delta']['text'] += text
            self.pending_text_bytes += text_bytes
            return []

        ready = self.flush()
        self.pending_span_id = span_id
        self.pending_event = copy.deepcopy(event)
        self.pending_key = key
        self.pending_text_bytes = text_bytes
        return ready

    def flush(self) -> list[tuple[str, dict[str, Any]]]:
        if self.pending_span_id is None or self.pending_event is None:
            return []
        ready = [(self.pending_span_id, self.pending_event)]
        self.pending_span_id = None
        self.pending_event = None
        self.pending_key = None
        self.pending_text_bytes = 0
        return ready


def _buffer_for(session: Any) -> StreamTraceBuffer:
    buffer = getattr(session, '_stream_trace_buffer', None)
    if buffer is None:
        buffer = StreamTraceBuffer()
        session._stream_trace_buffer = buffer
    return buffer


def _emit_trace_event(session: Any, agent_key: str, span_id: str, event: dict[str, Any]) -> None:
    from nexus.trace import Event, update_span

    update_span(
        span_id,
        events=[
            Event(
                time_unix_nano=time.time_ns(),
                name='agent.stream',
                attributes={
                    'agent': agent_key,
                    'session_id': session.session_id,
                    'cwd': session.cwd,
                    'event_type': event.get('type'),
                    'event': event,
                },
            )
        ],
    )


def record_stream_trace_event(
    session: Any,
    event: dict[str, Any],
    *,
    agent_key: str,
    enabled: bool = True,
) -> None:
    if not enabled:
        return
    try:
        from nexus.trace import get_current_span_id

        span_id = get_current_span_id()
        if span_id is None:
            return
        for ready_span_id, ready_event in _buffer_for(session).push(span_id, event):
            _emit_trace_event(session, agent_key, ready_span_id, ready_event)
    except Exception:
        logger.debug('Failed to record stream-json trace event', exc_info=True)


def flush_stream_trace_events(session: Any, *, agent_key: str) -> None:
    try:
        for span_id, event in _buffer_for(session).flush():
            _emit_trace_event(session, agent_key, span_id, event)
    except Exception:
        logger.debug('Failed to flush stream-json trace events', exc_info=True)


__all__ = [
    'MAX_MERGED_TEXT_BYTES',
    'StreamTraceBuffer',
    'flush_stream_trace_events',
    'record_stream_trace_event',
]
