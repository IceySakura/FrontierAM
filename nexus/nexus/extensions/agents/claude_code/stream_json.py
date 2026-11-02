"""Small helpers for Claude Code's stdio stream-json protocol."""

from __future__ import annotations

import json

from typing import Any


def encode_user_message(prompt: str, *, session_id: str, parent_tool_use_id: str | None = None) -> bytes:
    """Return one JSONL user message for Claude Code stdin."""
    content: str | list[dict[str, Any]]
    if parent_tool_use_id:
        content = [
            {
                'type': 'tool_result',
                'tool_use_id': parent_tool_use_id,
                'content': prompt,
                'is_error': False,
            }
        ]
    else:
        content = prompt
    payload = {
        'type': 'user',
        'message': {
            'role': 'user',
            'content': content,
        },
        'parent_tool_use_id': parent_tool_use_id,
        'session_id': session_id,
    }
    return (json.dumps(payload, ensure_ascii=False, separators=(',', ':')) + '\n').encode('utf-8')


def parse_stream_line(line: bytes) -> dict[str, Any]:
    """Decode one stdout JSONL line from Claude Code."""
    return json.loads(line.decode('utf-8', errors='replace'))


def extract_text_from_message(message: dict[str, Any]) -> str:
    """Extract human-readable text from a Claude message event."""
    raw_message = message.get('message')
    if not isinstance(raw_message, dict):
        return ''

    content = raw_message.get('content')
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ''

    chunks: list[str] = []
    for item in content:
        if isinstance(item, dict) and item.get('type') == 'text' and isinstance(item.get('text'), str):
            chunks.append(item['text'])
    return ''.join(chunks)


def summarize_turn_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a compact turn summary from raw stream-json events."""
    assistant_parts: list[str] = []
    result_event: dict[str, Any] | None = None
    session_id: str | None = None

    for event in events:
        event_type = event.get('type')
        if isinstance(event.get('session_id'), str):
            session_id = event['session_id']
        if event_type == 'assistant':
            text = extract_text_from_message(event)
            if text:
                assistant_parts.append(text)
        elif event_type == 'result':
            result_event = event
            if isinstance(event.get('session_id'), str):
                session_id = event['session_id']

    result_text = ''
    is_error = False
    if result_event:
        result_text = str(result_event.get('result') or '')
        is_error = bool(result_event.get('is_error'))

    assistant_text = ''.join(assistant_parts) or result_text
    return {
        'session_id': session_id,
        'assistant_text': assistant_text,
        'result_text': result_text,
        'is_error': is_error,
        'result_event': result_event or {},
    }


def native_prompt_too_long(raw: str) -> bool:
    """Trust only the last CC result envelope, not model/tool text about limits."""
    for line in reversed(raw.splitlines()):
        try:
            event = json.loads(line)
        except (ValueError, TypeError):
            continue
        if isinstance(event, dict) and event.get('type') == 'result':
            return (
                event.get('is_error') is True
                and event.get('api_error_status') == 400
                and event.get('terminal_reason') == 'prompt_too_long'
            )
    return False


def compaction_summary(raw: str) -> dict[str, Any]:
    """Small audit alongside the unmodified CC stream, never a history rewrite."""
    boundaries: list[dict[str, Any]] = []
    for line in raw.splitlines():
        try:
            event = json.loads(line)
        except (ValueError, TypeError):
            continue
        if not isinstance(event, dict):
            continue
        if event.get('type') == 'system' and event.get('subtype') == 'compact_boundary':
            metadata = event.get('compact_metadata') or {}
            if not isinstance(metadata, dict):
                metadata = {}
            boundaries.append({
                'trigger': metadata.get('trigger'),
                'pre_tokens': metadata.get('pre_tokens'),
                'next_input_tokens': None,
                'token_drop_verified': None,
            })
        elif event.get('type') == 'assistant' and boundaries and boundaries[-1]['next_input_tokens'] is None:
            message = event.get('message')
            usage = message.get('usage') if isinstance(message, dict) else None
            if not isinstance(usage, dict) or 'input_tokens' not in usage:
                continue
            parts = [
                usage.get(key, 0) or 0
                for key in ('input_tokens', 'cache_read_input_tokens', 'cache_creation_input_tokens')
            ]
            if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in parts):
                continue
            tokens = sum(parts)
            if tokens <= 0:
                continue
            boundary = boundaries[-1]
            boundary['next_input_tokens'] = tokens
            if isinstance(boundary['pre_tokens'], int) and not isinstance(boundary['pre_tokens'], bool):
                boundary['token_drop_verified'] = tokens < boundary['pre_tokens']
    return {'count': len(boundaries), 'boundaries': boundaries}


class ParentSessionState:
    """Track root output independently of interleaved child session events."""

    def __init__(self, session_id: str | None = None) -> None:
        self.session_id = session_id
        self.final_text: str | None = None
        self.result_subtype: str | None = None
        self.result_seen = False
        self.failed = False
        self.pending_tools: set[str] = set()
        self.terminal_marker: str | None = None

    def feed(self, event: dict[str, Any]) -> bool:
        """Return whether this event belongs to the parent session."""
        if event.get('parent_tool_use_id'):
            return False
        session = event.get('session_id')
        if self.session_id is None and isinstance(session, str) and session:
            self.session_id = session
        if self.session_id is not None and session is not None and session != self.session_id:
            return False
        kind = event.get('type')
        if kind == 'result':
            self.result_seen = True
            self.result_subtype = event.get('subtype')
            self.failed = self.failed or bool(event.get('is_error') or event.get('api_error_status'))
            if event.get('result'):
                self.final_text = str(event['result'])
            self.terminal_marker = self._marker(self.final_text) if not self.failed else None
        elif kind in ('assistant', 'user'):
            message = event.get('message') or {}
            blocks = message.get('content') if isinstance(message, dict) else None
            if not isinstance(blocks, list):
                return True
            if kind == 'user':
                for block in blocks:
                    if isinstance(block, dict) and block.get('type') == 'tool_result':
                        self.pending_tools.discard(str(block.get('tool_use_id', '')))
                return True
            texts = []
            for block in blocks:
                if not isinstance(block, dict):
                    continue
                if block.get('type') == 'tool_use':
                    self.pending_tools.add(str(block.get('id', '')))
                elif block.get('type') == 'text' and isinstance(block.get('text'), str):
                    texts.append(block['text'])
            text = '\n'.join(texts)
            # A CLI result is authoritative even if a late assistant event arrives.
            if text and not self.result_seen:
                self.final_text = text
            self.terminal_marker = self._marker(text) if not self.failed and not self.pending_tools else None
        if self.pending_tools or self.failed:
            self.terminal_marker = None
        return True

    @staticmethod
    def _marker(text: str | None) -> str | None:
        if not text:
            return None
        lines = text.strip().splitlines()
        marker = lines[-1].strip()
        # A dedicated terminal line, not an inline mention or code example.
        if marker not in {'TASK_DONE', 'TASK_INFEASIBLE'} or '```' in text or '~~~' in text:
            return None
        return marker


def parse_transcript(raw: str) -> tuple[list[dict], str | None, str | None]:
    """Parse claude -p --output-format stream-json output.

    Returns (trajectory, final_text, result_subtype). Each line is one JSON event.
    """
    trajectory: list[dict] = []
    events = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(ev, dict):
            continue
        events.append(ev)
    root_session = next(
        (
            e.get('session_id')
            for e in events
            if e.get('type') == 'system'
            and e.get('subtype') == 'init'
            and not e.get('parent_tool_use_id')
            and e.get('session_id')
        ),
        None,
    )
    parent = ParentSessionState(root_session)
    for ev in events:
        parent.feed(ev)
        etype = ev.get('type')
        if etype == 'assistant':
            message = ev.get('message')
            if not isinstance(message, dict) or not isinstance(message.get('content'), list):
                continue
            for block in message['content']:
                if not isinstance(block, dict):
                    continue
                bt = block.get('type')
                if bt == 'text':
                    trajectory.append({'kind': 'text', 'text': block.get('text', '')})
                elif bt == 'tool_use':
                    trajectory.append({'kind': 'tool_use', 'name': block.get('name'), 'input': block.get('input')})
    return trajectory, parent.final_text, parent.result_subtype


def extract_session_id(raw: str) -> str | None:
    """Extract the latest non-empty top-level session id from CC stream JSON."""

    session_id: str | None = None
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        value = event.get('session_id')
        if isinstance(value, str) and value.strip():
            session_id = value.strip()
    return session_id
