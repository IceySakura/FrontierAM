"""Read authoritative per-turn results from DSH JSONL persistence."""

from __future__ import annotations

import asyncio
import json
import time

from dataclasses import dataclass
from pathlib import Path
from typing import Any


class DshSessionLogError(RuntimeError):
    """The DSH session log is missing, malformed, or discontinuous."""


def _read_complete_records(path: Path) -> list[dict[str, Any]]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise DshSessionLogError(f'Failed to read DSH session log {path}: {exc}') from exc
    lines = payload.split(b'\n')
    if lines and lines[-1]:
        lines.pop()
    records: list[dict[str, Any]] = []
    for index, line in enumerate(lines, start=1):
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DshSessionLogError(f'Invalid JSON in {path} line {index}: {exc}') from exc
        if not isinstance(record, dict):
            raise DshSessionLogError(f'Invalid record in {path} line {index}: expected an object')
        records.append(record)
    return records


def _find_session_log(root: Path, session_id: str, cwd: str) -> Path | None:
    matches: list[Path] = []
    for candidate in root.rglob('session.jsonl') if root.exists() else ():
        records = _read_complete_records(candidate)
        if not records:
            continue
        header = records[0]
        if header.get('type') == 'session' and header.get('id') == session_id and header.get('cwd') == cwd:
            matches.append(candidate)
    if len(matches) > 1:
        raise DshSessionLogError(f'Multiple DSH logs found for session {session_id}: {matches!r}')
    return matches[0] if matches else None


@dataclass
class DshSessionLog:
    path: Path
    session_id: str
    cwd: str
    next_seq: int = 0

    @classmethod
    async def wait_for_session(
        cls,
        *,
        root: Path,
        session_id: str,
        cwd: str,
        timeout: float,
    ) -> DshSessionLog:
        deadline = time.monotonic() + timeout
        while True:
            path = await asyncio.to_thread(_find_session_log, root, session_id, cwd)
            if path is not None:
                return cls(path=path, session_id=session_id, cwd=cwd)
            if time.monotonic() >= deadline:
                raise TimeoutError(f'Timed out waiting for DSH session log for {session_id}')
            await asyncio.sleep(0.05)

    def _new_events(self) -> list[dict[str, Any]]:
        records = _read_complete_records(self.path)
        if not records or records[0].get('type') != 'session':
            raise DshSessionLogError(f'DSH session log has no valid header: {self.path}')
        events = records[1:]
        expected = 0
        for event in events:
            seq = event.get('seq')
            if not isinstance(seq, int) or seq != expected:
                raise DshSessionLogError(
                    f'DSH session log sequence is discontinuous at {self.path}: expected {expected}, got {seq!r}'
                )
            expected += 1
        if self.next_seq > expected:
            raise DshSessionLogError(
                f'DSH session log was truncated at {self.path}: cursor {self.next_seq}, next sequence {expected}'
            )
        return events[self.next_seq :]

    async def wait_for_turn_end(self, *, timeout: float) -> list[dict[str, Any]]:
        deadline = time.monotonic() + timeout
        while True:
            events = await asyncio.to_thread(self._new_events)
            if any(event.get('type') == 'turn/end' for event in events):
                await asyncio.sleep(0.05)
                events = await asyncio.to_thread(self._new_events)
                self.next_seq += len(events)
                return events
            if time.monotonic() >= deadline:
                raise TimeoutError(f'Timed out waiting for turn/end in DSH session {self.session_id}')
            await asyncio.sleep(0.05)


def _message_text(event: dict[str, Any]) -> str:
    data = event.get('data')
    message = data.get('message') if isinstance(data, dict) else None
    content = message.get('content') if isinstance(message, dict) else None
    if not isinstance(content, list):
        return ''
    return ''.join(
        str(block['text'])
        for block in content
        if isinstance(block, dict) and block.get('type') == 'text' and isinstance(block.get('text'), str)
    )


def summarize_turn(events: list[dict[str, Any]], *, acp_text: str = '') -> dict[str, Any]:
    """Summarize one DSH JSONL delta, treating turn/end as authoritative."""
    turn_end = next((event for event in reversed(events) if event.get('type') == 'turn/end'), None)
    if turn_end is None:
        raise DshSessionLogError('DSH turn delta contains no turn/end event')
    data = turn_end.get('data')
    reason = data.get('reason') if isinstance(data, dict) else None
    kind = reason.get('kind') if isinstance(reason, dict) else None
    if not isinstance(kind, str):
        raise DshSessionLogError(f'DSH turn/end contains no valid reason: {turn_end!r}')
    fallback = ''.join(_message_text(event) for event in events if event.get('type') == 'assistant/message')
    return {
        'success': kind == 'completed',
        'status': kind,
        'output': acp_text or fallback,
        'result_event': turn_end,
    }


__all__ = ['DshSessionLog', 'DshSessionLogError', 'summarize_turn']
