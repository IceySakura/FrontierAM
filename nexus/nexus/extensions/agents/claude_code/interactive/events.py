"""Structured event extraction and truncation for recent_events tracking.

Provides helpers that convert raw Anthropic Messages request/response bodies
into compact, truncated event dicts suitable for the User Model's context
window. Used by ``PtyAnthropicProxy`` to build the ``recent_events`` stream.
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Truncation helpers.
# ---------------------------------------------------------------------------

_TRUNCATE_MARKER = '\n...[truncated]...\n'


def _truncate_str(value: str, *, head: int, tail: int) -> tuple[str, bool]:
    limit = head + tail
    if len(value) <= limit:
        return value, False
    return value[:head] + _TRUNCATE_MARKER + value[-tail:], True


def _truncate_tool_result_content(content: Any, *, is_error: bool) -> tuple[str, bool]:
    """Truncate tool_result content to a summary string."""
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get('type') == 'text':
                    parts.append(str(block.get('text', '')))
                elif block.get('type') == 'image':
                    parts.append('[image]')
                else:
                    parts.append(f'[{block.get("type", "unknown")}]')
            else:
                parts.append(str(block))
        text = '\n'.join(parts)
    elif isinstance(content, str):
        text = content
    else:
        text = str(content) if content else ''
    head, tail = (1400, 600) if is_error else (700, 300)
    return _truncate_str(text, head=head, tail=tail)


def _truncate_tool_input(input_dict: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Truncate string values in tool_use input (recursive for nested structures)."""
    truncated_flag = False
    max_list_items = 20
    max_depth = 6

    def _walk(obj: Any, depth: int = 0) -> Any:
        nonlocal truncated_flag
        if isinstance(obj, str):
            val, was_cut = _truncate_str(obj, head=150, tail=50)
            truncated_flag = truncated_flag or was_cut
            return val
        if isinstance(obj, dict):
            if depth >= max_depth:
                truncated_flag = True
                return '[nested object truncated]'
            return {k: _walk(v, depth + 1) for k, v in obj.items()}
        if isinstance(obj, list):
            if depth >= max_depth:
                truncated_flag = True
                return '[nested list truncated]'
            items = [_walk(item, depth + 1) for item in obj[:max_list_items]]
            if len(obj) > max_list_items:
                truncated_flag = True
                items.append(f'[...{len(obj) - max_list_items} more items truncated]')
            return items
        return obj

    result = _walk(input_dict)
    return result, truncated_flag


# ---------------------------------------------------------------------------
# Event extraction from Anthropic Messages request/response bodies.
# ---------------------------------------------------------------------------


def extract_request_events(body: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract tool_result events from the LAST user message in a request.

    Anthropic Messages API carries full conversation history, so scanning all
    messages would re-extract tool_results from earlier turns. Only the last
    user message contains new results from the current exchange.
    """
    events: list[dict[str, Any]] = []
    messages = body.get('messages')
    if not isinstance(messages, list) or not messages:
        return events
    last_user_msg = None
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get('role') == 'user':
            last_user_msg = msg
            break
    if last_user_msg is None:
        return events
    content = last_user_msg.get('content')
    if not isinstance(content, list):
        return events
    for block in content:
        if not isinstance(block, dict) or block.get('type') != 'tool_result':
            continue
        tool_use_id = block.get('tool_use_id', '')
        is_error = bool(block.get('is_error', False))
        raw_content = block.get('content', '')
        text, was_truncated = _truncate_tool_result_content(raw_content, is_error=is_error)
        event: dict[str, Any] = {
            'type': 'tool_result',
            'tool_use_id': tool_use_id,
            'content': text,
            'is_error': is_error,
        }
        if was_truncated:
            event['truncated'] = True
        events.append(event)
    return events


def extract_response_events(response: Any) -> list[dict[str, Any]]:
    """Extract tool_use and text events from an Anthropic response."""
    events: list[dict[str, Any]] = []
    if isinstance(response, dict):
        content = response.get('content')
    elif hasattr(response, 'content'):
        content = response.content
    else:
        return events
    if not isinstance(content, list):
        return events
    for block in content:
        if isinstance(block, dict):
            block_type = block.get('type')
        elif hasattr(block, 'type'):
            block_type = block.type
            block = block.model_dump() if hasattr(block, 'model_dump') else vars(block)
        else:
            continue
        if block_type == 'tool_use':
            raw_input = block.get('input', {})
            input_dict = raw_input if isinstance(raw_input, dict) else {}
            truncated_input, was_truncated = _truncate_tool_input(input_dict)
            event: dict[str, Any] = {
                'type': 'tool_use',
                'id': block.get('id', ''),
                'name': block.get('name', ''),
                'input': truncated_input,
            }
            if was_truncated:
                event['truncated'] = True
            events.append(event)
        elif block_type == 'text':
            raw_text = block.get('text', '')
            text, was_truncated = _truncate_str(str(raw_text), head=350, tail=150)
            event = {'type': 'text', 'text': text}
            if was_truncated:
                event['truncated'] = True
            events.append(event)
    return events


class EventBuffer:
    """Incremental event buffer with seq numbering, turn labelling, and cursor commit.

    Owns all event state so ``proxy.py`` only needs to call
    ``append_from_request``/``append_from_response``/``get_delta``/``commit``.
    """

    _MAX_DELTA_EVENTS = 1000

    def __init__(self) -> None:
        self._events: list[dict[str, Any]] = []
        self._next_seq = 1
        self._last_emitted_seq = 0
        self._current_turn_id = 0

    def mark_turn_start(self) -> None:
        self._current_turn_id += 1

    def append_from_request(self, body: dict[str, Any]) -> None:
        self.append(extract_request_events(body))

    def append_from_response(self, response: Any) -> None:
        self.append(extract_response_events(response))

    def append(self, raw_events: list[dict[str, Any]]) -> None:
        """Append already-extracted events and assign session ordering metadata."""
        for raw in raw_events:
            event = dict(raw)
            event['seq'] = self._next_seq
            event['turn_id'] = self._current_turn_id
            self._next_seq += 1
            self._events.append(event)

    def get_delta(self, max_events: int | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Return events since last commit, plus a cursor dict."""
        cap = max_events if max_events is not None else self._MAX_DELTA_EVENTS
        pending = [e for e in self._events if e['seq'] > self._last_emitted_seq]
        latest_seq = pending[-1]['seq'] if pending else self._last_emitted_seq
        dropped = max(0, len(pending) - cap)
        delivered = pending[-cap:] if dropped else pending
        cursor = {
            'after_seq': self._last_emitted_seq,
            'latest_seq': latest_seq,
            'count': len(delivered),
            'truncated': dropped > 0,
            'dropped_count': dropped,
        }
        return delivered, cursor

    def commit(self, seq: int) -> None:
        """Advance cursor and release old events."""
        latest = self._next_seq - 1
        self._last_emitted_seq = max(self._last_emitted_seq, min(seq, latest))
        self._events = [e for e in self._events if e['seq'] > self._last_emitted_seq]


__all__ = [
    'EventBuffer',
    '_truncate_str',
    '_truncate_tool_input',
    '_truncate_tool_result_content',
    'extract_request_events',
    'extract_response_events',
]
