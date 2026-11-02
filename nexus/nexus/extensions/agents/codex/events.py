"""Public Codex event stream models and pure event projections."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field


class CodexEvent(BaseModel):
    """One ordered event produced by a Codex execution driver."""

    seq: int
    timestamp: float | None = None
    direction: Literal['inbound', 'outbound'] | None = None
    """Message direction for app-server traffic; None for synthetic events."""
    type: str
    data: dict[str, Any] = Field(default_factory=dict)


class CodexEventStream(BaseModel):
    """Self-describing event stream exposed in :class:`CodexOutput`."""

    version: int = 1
    source: Literal['app_server']
    events: list[CodexEvent] = Field(default_factory=list)


@dataclass
class AppServerEventReduction:
    """State derived from a single app-server thread/turn event stream."""

    final_message: str = ''
    turn_status: str | None = None
    error: Any = None
    timeline_segments: list[dict[str, Any]] = field(default_factory=list)


_OBSERVATION_NAMES = {
    'commandExecution': 'shell',
    'fileChange': 'apply_patch',
    'mcpToolCall': 'mcp_tool_call',
    'dynamicToolCall': 'dynamic_tool_call',
    'webSearch': 'web_search',
    'imageView': 'view_image',
    'imageGeneration': 'image_generation',
    'collabToolCall': 'collab_agent_tool_call',
}


def _canonical_method(method: str) -> str:
    return method.replace('/', '.')


def app_server_message_to_event(
    message: dict[str, Any],
    *,
    seq: int,
    timestamp: float,
    direction: Literal['inbound', 'outbound'],
    request_methods: dict[Any, str],
) -> CodexEvent:
    """Convert one JSON-RPC message into the public event envelope."""
    method = message.get('method')
    request_id = message.get('id')
    if isinstance(method, str):
        params = message.get('params')
        data = dict(params) if isinstance(params, dict) else {'params': params}
        if request_id is not None:
            data = {'requestId': request_id, **data}
        return CodexEvent(
            seq=seq,
            timestamp=timestamp,
            direction=direction,
            type=_canonical_method(method),
            data=data,
        )

    correlated_method = request_methods.get(request_id) if isinstance(request_id, (int, str)) else None
    event_type = (
        f'{_canonical_method(correlated_method)}.response' if correlated_method else 'protocol.unknown_response'
    )
    data: dict[str, Any] = {'requestId': request_id}
    if 'result' in message:
        data['result'] = message.get('result')
    if 'error' in message:
        data['error'] = message.get('error')
    return CodexEvent(seq=seq, timestamp=timestamp, direction=direction, type=event_type, data=data)


def _tool_args(item: dict[str, Any]) -> dict[str, Any] | None:
    item_type = item.get('type')
    if item_type == 'commandExecution':
        return {
            'command': item.get('command'),
            'cwd': item.get('cwd'),
            'exit_code': item.get('exitCode'),
            'status': item.get('status'),
        }
    if item_type == 'fileChange':
        return {'changes': item.get('changes'), 'status': item.get('status')}
    if item_type in {'mcpToolCall', 'dynamicToolCall'}:
        return {'arguments': item.get('arguments'), 'status': item.get('status')}
    if item_type == 'webSearch':
        return {'query': item.get('query')}
    if item_type == 'imageView':
        return {'path': item.get('path')}
    return None


def _timeline_segment(
    item: dict[str, Any],
    *,
    started_at: float | None,
    completed_at: float,
    include_tool_args: bool,
) -> dict[str, Any] | None:
    item_type = item.get('type')
    name = _OBSERVATION_NAMES.get(item_type)
    if name is None:
        return None
    if item_type == 'mcpToolCall':
        server = item.get('server')
        tool = item.get('tool')
        name = '.'.join(str(part) for part in (server, tool) if part) or name
    elif item_type in {'dynamicToolCall', 'collabToolCall'}:
        name = str(item.get('tool') or name)

    duration_ms = item.get('durationMs')
    if started_at is None and isinstance(duration_ms, (int, float)):
        started_at = completed_at - max(0, duration_ms) / 1000
    start_time = min(started_at or completed_at - 1e-6, completed_at - 1e-9)
    segment: dict[str, Any] = {
        'name': name,
        'start_time': start_time,
        'end_time': completed_at,
    }
    if include_tool_args:
        args = _tool_args(item)
        if args is not None:
            segment['tool_args'] = args
    return segment


def reduce_app_server_events(
    events: list[CodexEvent],
    *,
    thread_id: str | None = None,
    turn_id: str | None = None,
    include_tool_args: bool = False,
) -> AppServerEventReduction:
    """Derive final message, turn status, error, and tool timeline from the journal."""
    reduction = AppServerEventReduction()
    item_started_at: dict[str, float] = {}

    for event in events:
        data = event.data
        event_thread_id = data.get('threadId')
        event_turn_id = data.get('turnId')
        if thread_id is not None and event_thread_id not in {None, thread_id}:
            continue
        if turn_id is not None and event_turn_id not in {None, turn_id}:
            continue

        if event.type == 'item.started':
            item = data.get('item') or {}
            item_id = item.get('id')
            if isinstance(item_id, str) and event.timestamp is not None:
                item_started_at[item_id] = event.timestamp
        elif event.type == 'item.completed':
            item = data.get('item') or {}
            item_id = item.get('id')
            if item.get('type') == 'agentMessage':
                reduction.final_message = str(item.get('text') or '')
            completed_at = event.timestamp
            if completed_at is None:
                continue
            segment = _timeline_segment(
                item,
                started_at=item_started_at.pop(item_id, None) if isinstance(item_id, str) else None,
                completed_at=completed_at,
                include_tool_args=include_tool_args,
            )
            if segment is not None:
                reduction.timeline_segments.append(segment)
        elif event.type == 'error':
            reduction.error = data.get('error') or data
        elif event.type == 'turn.completed':
            turn = data.get('turn') or {}
            if turn_id is not None and turn.get('id') != turn_id:
                continue
            reduction.turn_status = turn.get('status')
            reduction.error = turn.get('error') or reduction.error

    return reduction


def classify_app_server_termination(events: list[CodexEvent], exit_code: int, timed_out: bool) -> str:
    """Map execution facts to a Nexus termination reason on the host."""
    if timed_out:
        return 'TASK_TIMEOUT'
    if exit_code == 0:
        return 'ENV_DONE'
    if exit_code < 0:
        return 'CODEX_ERROR'
    thread_id = next(
        (
            (event.data.get('result') or {}).get('thread', {}).get('id')
            for event in events
            if event.type == 'thread.start.response'
        ),
        None,
    )
    reduction = reduce_app_server_events(events, thread_id=thread_id)
    goal_status = None
    for event in events:
        if thread_id is not None and event.data.get('threadId') not in {None, thread_id}:
            continue
        if event.type == 'thread.goal.updated':
            goal_status = (event.data.get('goal') or {}).get('status')
        elif event.type == 'thread.goal.cleared':
            goal_status = 'cleared'
        elif event.type == 'thread.goal.set.response' and goal_status is None:
            goal_status = ((event.data.get('result') or {}).get('goal') or {}).get('status')
    goal_reasons = {
        'paused': 'CODEX_GOAL_PAUSED',
        'blocked': 'CODEX_GOAL_BLOCKED',
        'usageLimited': 'CODEX_GOAL_USAGE_LIMITED',
        'budgetLimited': 'CODEX_GOAL_BUDGET_LIMITED',
        'cleared': 'CODEX_GOAL_CLEARED',
    }
    # Codex also blocks goals on terminal turn errors; preserve the specific truncation cause.
    if any(
        marker in str(reduction.error).lower()
        for marker in ('contextwindowexceeded', 'context_length_exceeded', 'context length')
    ):
        return 'TRUNCATION'
    if goal_status in goal_reasons:
        return goal_reasons[goal_status]
    return 'CODEX_TURN_FAILED'


__all__ = [
    'AppServerEventReduction',
    'CodexEvent',
    'CodexEventStream',
    'app_server_message_to_event',
    'reduce_app_server_events',
    'classify_app_server_termination',
]
