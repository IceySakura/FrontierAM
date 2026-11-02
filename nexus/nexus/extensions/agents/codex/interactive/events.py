"""Bounded event buffer for the caller's ``recent_events`` stream.

Events come from the app-server notifications the relay taps (``item/*`` mostly):
tool calls, file changes, command output, agent messages. Payloads are truncated
so one noisy turn cannot blow up the caller's context, and delivery is cursor
based so each boundary returns only what is new.

This is deliberately independent of the Claude Code event extractor: the source
is Codex's own thread-item protocol, not Anthropic Messages bodies.
"""

from __future__ import annotations

from itertools import islice
from typing import Any


_TRUNCATE_MARKER = '\n...[truncated]...\n'
# Per-container element cap, applied to both lists and dicts.
_MAX_ITEMS = 20


def _truncate(value: str, *, head: int, tail: int) -> tuple[str, bool]:
    if len(value) <= head + tail:
        return value, False
    return value[:head] + _TRUNCATE_MARKER + value[-tail:], True


def _shrink(obj: Any, depth: int = 0) -> tuple[Any, bool]:
    """Recursively bound strings, list length, and nesting depth."""
    cut = False
    if isinstance(obj, str):
        text, was_cut = _truncate(obj, head=400, tail=150)
        return text, was_cut
    if isinstance(obj, dict):
        if depth >= 5:
            return '[nested object truncated]', True
        out: dict[str, Any] = {}
        # islice, not a list slice: materializing every pair first would make the
        # cost scale with the dict's width even though only _MAX_ITEMS are kept.
        for key, value in islice(obj.items(), _MAX_ITEMS):
            out[key], hit = _shrink(value, depth + 1)
            cut = cut or hit
        remaining = len(obj) - _MAX_ITEMS
        if remaining > 0:
            out['...'] = f'{remaining} more keys truncated'
            cut = True
        return out, cut
    if isinstance(obj, list):
        if depth >= 5:
            return '[nested list truncated]', True
        items = []
        for value in obj[:_MAX_ITEMS]:
            shrunk, hit = _shrink(value, depth + 1)
            cut = cut or hit
            items.append(shrunk)
        if len(obj) > _MAX_ITEMS:
            items.append(f'[...{len(obj) - _MAX_ITEMS} more items truncated]')
            cut = True
        return items, cut
    return obj, False


def extract_event(message: dict[str, Any]) -> dict[str, Any] | None:
    """Turn one app-server notification into a compact event, or None to skip."""
    method = message.get('method')
    if not isinstance(method, str) or not method.startswith('item/'):
        return None
    # Deltas are per-token streaming noise; the completed item carries the content.
    if method.endswith('/delta') or 'Delta' in method:
        return None
    params = message.get('params') or {}
    item = params.get('item') if isinstance(params, dict) else None
    event: dict[str, Any] = {
        'type': method,
        'thread_id': params.get('threadId') if isinstance(params, dict) else None,
        'turn_id': params.get('turnId') if isinstance(params, dict) else None,
    }
    if isinstance(item, dict):
        payload, truncated = _shrink({k: v for k, v in item.items() if k != 'id'})
        event['item_type'] = item.get('itemType') or item.get('type')
        event['item_id'] = item.get('id')
        event['item'] = payload
        if truncated:
            event['truncated'] = True
    return event


class EventBuffer:
    """Sequenced, cursor-committed, capacity-bounded event log for one session.

    The cap is enforced on *append*, not only when a delta is read: a long turn
    that is never drained would otherwise grow without limit. Evicted events are
    counted so the caller learns it did not see everything.
    """

    MAX_DELTA_EVENTS = 500
    MAX_BUFFERED_EVENTS = 2000

    def __init__(self) -> None:
        self._events: list[dict[str, Any]] = []
        self._next_seq = 1
        self._last_emitted_seq = 0
        self._turn_id = 0
        self._evicted = 0

    def mark_turn_start(self) -> None:
        self._turn_id += 1

    @property
    def evicted_count(self) -> int:
        """Events dropped by capacity eviction and not yet reported."""
        return self._evicted

    def append_message(
        self,
        message: dict[str, Any],
        *,
        thread_id: str | None = None,
        turn_id: str | None = None,
    ) -> None:
        event = extract_event(message)
        if event is None:
            return
        if thread_id is not None and event['thread_id'] != thread_id:
            return
        if turn_id is not None and event['turn_id'] != turn_id:
            return
        event['seq'] = self._next_seq
        event['nexus_turn'] = self._turn_id
        self._next_seq += 1
        self._events.append(event)
        overflow = len(self._events) - self.MAX_BUFFERED_EVENTS
        if overflow > 0:
            # Evict the oldest: the newest events describe what Codex is doing now.
            del self._events[:overflow]
            self._evicted += overflow

    def get_delta(self, max_events: int | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        cap = self.MAX_DELTA_EVENTS if max_events is None else max_events
        pending = [e for e in self._events if e['seq'] > self._last_emitted_seq]
        latest = pending[-1]['seq'] if pending else self._last_emitted_seq
        over_cap = max(0, len(pending) - cap)
        delivered = pending[-cap:] if over_cap else pending
        # Total gap the caller cannot see: evicted on append + trimmed for this read.
        dropped = self._evicted + over_cap
        cursor = {
            'after_seq': self._last_emitted_seq,
            'latest_seq': latest,
            'count': len(delivered),
            'truncated': dropped > 0,
            'dropped_count': dropped,
            'evicted_count': self._evicted,
        }
        return delivered, cursor

    def commit(self, seq: int) -> None:
        latest = self._next_seq - 1
        self._last_emitted_seq = max(self._last_emitted_seq, min(seq, latest))
        self._events = [e for e in self._events if e['seq'] > self._last_emitted_seq]
        # Eviction has now been reported to the caller; stop re-counting it.
        self._evicted = 0


def empty_event_cursor() -> dict[str, Any]:
    return {'after_seq': 0, 'latest_seq': 0, 'count': 0, 'truncated': False, 'dropped_count': 0}


__all__ = ['EventBuffer', 'empty_event_cursor', 'extract_event']
