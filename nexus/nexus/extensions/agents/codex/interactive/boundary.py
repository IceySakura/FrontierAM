"""Turn boundary adapter: decides when a turn ended, and how.

Correlation is strict, because the TUI's stream carries more than our turn. A
single PTY submission can make Codex open auxiliary threads, and stale events from
the previous turn can still be in flight when the next prompt is typed. So:

  * the session locks onto one thread id (the first thread that starts a turn
    after our first submission) and ignores every other thread;
  * within that thread, only the id of the turn currently in flight is accepted.

A ``turn/completed`` whose ids do not match both locks is dropped, never treated as
this turn finishing. No screen text and no silence window participates in the
decision.
"""

from __future__ import annotations

import logging

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


logger = logging.getLogger(__name__)

TURN_STARTED = 'turn/started'
TURN_COMPLETED = 'turn/completed'
ITEM_COMPLETED = 'item/completed'
# Sent when a server request stops needing an answer: it was answered, or the turn
# completed/was interrupted and Codex cleaned it up. Without consuming it a stale
# pending interaction would be replayed to the caller on this or the next turn.
SERVER_REQUEST_RESOLVED = 'serverRequest/resolved'

# Server→client JSON-RPC *requests* (they carry an id and expect a reply). Codex
# Codex 通过这些请求发起命令、文件和权限审批，或直接提出问题。
# 问题会交给调用方；TUI 使用 approval_policy=never，因此审批请求属于协议异常。
APPROVAL_REQUEST_METHODS = frozenset({
    'item/commandExecution/requestApproval',
    'item/fileChange/requestApproval',
    'item/permissions/requestApproval',
})
USER_INPUT_REQUEST_METHODS = frozenset({'item/tool/requestUserInput'})
MCP_ELICITATION_REQUEST_METHODS = frozenset({'mcpServer/elicitation/request'})
INTERACTION_REQUEST_METHODS = APPROVAL_REQUEST_METHODS | USER_INPUT_REQUEST_METHODS | MCP_ELICITATION_REQUEST_METHODS

_METHOD_LABELS = {
    'item/tool/requestUserInput': 'request_user_input',
    'mcpServer/elicitation/request': 'mcp_elicitation',
}


class TurnOutcome(str, Enum):
    """Terminal classification for one turn."""

    COMPLETED = 'completed'
    FAILED = 'failed'
    INTERRUPTED = 'interrupted'


@dataclass
class TurnResult:
    """A matched turn boundary."""

    outcome: TurnOutcome
    thread_id: str
    turn_id: str
    error_message: str | None = None
    final_message: str = ''
    """Last assistant message of this turn, taken from its own item/completed."""


@dataclass
class PendingInteraction:
    """A server→client request the TUI is currently blocked on."""

    interaction_id: str
    request_id: str | int
    method: str
    kind: str  # 'approval' | 'user_input'
    params: dict[str, Any] = field(default_factory=dict)
    thread_id: str | None = None
    turn_id: str | None = None
    claimed: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Render for the Host, including fields the User Model reads directly.

        ``hint`` / ``tool_name`` / ``tool_input`` are what surfaces in a
        ``UserInputRequest``.
        """
        return {
            'interaction_id': self.interaction_id,
            'method': self.method,
            'kind': self.kind,
            'params': self.params,
            'thread_id': self.thread_id,
            'turn_id': self.turn_id,
            'hint': self.hint(),
            'tool_name': self.tool_name(),
            'tool_input': self.tool_input(),
        }

    def tool_name(self) -> str:
        """Short, stable label for the thing being asked about."""
        return _METHOD_LABELS.get(self.method, self.method.rsplit('/', 1)[-1] or self.method)

    def tool_input(self) -> dict[str, Any]:
        """The decision-relevant payload, flattened for display."""
        params = self.params if isinstance(self.params, dict) else {}
        out: dict[str, Any] = {}
        questions = params.get('questions')
        if isinstance(questions, list) and questions:
            out['questions'] = [
                {
                    'id': q.get('id'),
                    'question': q.get('question'),
                    'header': q.get('header'),
                    'options': [o.get('label') for o in (q.get('options') or []) if isinstance(o, dict)],
                }
                for q in questions
                if isinstance(q, dict)
            ]
        if self.method == 'mcpServer/elicitation/request':
            for key in ('serverName', 'mode', 'message', 'url', 'requestedSchema'):
                value = params.get(key)
                if value not in (None, '', [], {}):
                    out[key] = value
        return out

    def hint(self) -> str:
        """One human-readable line describing what is being asked."""
        payload = self.tool_input()
        questions = payload.get('questions') or []
        if len(questions) > 1:
            rendered = '; '.join(
                f'{question.get("id")}: {question.get("question") or question.get("header") or ""}'
                for question in questions
            )
            return f'Codex asks multiple questions. Reply with a JSON object keyed by question id: {rendered}'
        if questions:
            first = questions[0]
            text = first.get('question') or first.get('header') or ''
            options = [o for o in (first.get('options') or []) if o]
            suffix = f' Options: {", ".join(options)}.' if options else ''
            return f'Codex asks: {text}{suffix}'
        if self.method == 'mcpServer/elicitation/request':
            message = payload.get('message') or 'An MCP server requests input.'
            return f'Codex MCP elicitation: {message}'
        return 'Codex is asking a question and is waiting for an answer.'


_AGENT_MESSAGE_ITEM_TYPES = frozenset({'agentMessage', 'assistantMessage', 'message'})


def extract_message_text(item: dict[str, Any]) -> str:
    """Pull plain text out of one thread item, tolerating shape variations."""
    for key in ('text', 'message'):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value
    content = item.get('content')
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                text = block.get('text')
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(block, str):
                parts.append(block)
        return '\n'.join(part for part in parts if part)
    return ''


class CodexBoundaryAdapter:
    """Correlates app-server notifications with the session's own turn."""

    def __init__(self) -> None:
        self.thread_id: str | None = None
        self.active_turn_id: str | None = None
        self.last_result: TurnResult | None = None
        self.pending_interaction: PendingInteraction | None = None
        self._interaction_queue: deque[PendingInteraction] = deque()
        # (request_id, thread_id, turn_id) of the last acknowledged request. The
        # ids mirror PendingInteraction, where both may be None.
        self._last_resolved_interaction: tuple[str | int, str | None, str | None] | None = None
        self._awaiting_turn = False
        self._turn_closed = False
        self._turn_messages: dict[str, list[str]] = {}

    # ----- turn lifecycle -------------------------------------------------- #

    def expect_turn(self) -> None:
        """Arm the adapter just before the TUI receives a prompt."""
        self._awaiting_turn = True
        self._turn_closed = False
        self.active_turn_id = None
        self.last_result = None
        # A turn must never inherit a request raised during the previous one.
        self.clear_interactions()
        self._last_resolved_interaction = None

    def release(self) -> None:
        self._awaiting_turn = False
        if self.active_turn_id is not None:
            self._turn_messages.pop(self.active_turn_id, None)
        self.active_turn_id = None
        self.clear_interactions()

    def clear_interactions(self) -> None:
        """清除当前 Turn 关联的全部请求。"""
        self.pending_interaction = None
        self._interaction_queue.clear()

    def live_turn_scope(self) -> tuple[str, str] | None:
        """返回仍在处理中的 Thread 和 Turn。"""
        if self.thread_id is None or self.active_turn_id is None or self._turn_closed:
            return None
        return self.thread_id, self.active_turn_id

    # ----- message intake --------------------------------------------------- #

    def handle(self, message: dict[str, Any]) -> None:
        """Consume one decoded app-server message."""
        method = message.get('method')
        if not method:
            return
        params = message.get('params') or {}
        if not isinstance(params, dict):
            return
        if message.get('id') is not None and method in INTERACTION_REQUEST_METHODS:
            self._handle_interaction_request(message, params)
            return
        if method == TURN_STARTED:
            self._handle_turn_started(params)
        elif method == TURN_COMPLETED:
            self._handle_turn_completed(params)
        elif method == ITEM_COMPLETED:
            self._handle_item_completed(params)
        elif method == SERVER_REQUEST_RESOLVED:
            self._handle_server_request_resolved(params)

    def _handle_server_request_resolved(self, params: dict[str, Any]) -> None:
        """Clear only the exact request attached to the still-active turn."""
        request_id = params.get('requestId')
        pending = self.pending_interaction
        candidates = [pending, *self._interaction_queue] if pending is not None else list(self._interaction_queue)
        resolved = next(
            (
                interaction
                for interaction in candidates
                if interaction.request_id == request_id
                and params.get('threadId') == interaction.thread_id
                and interaction.thread_id == self.thread_id
                and not self._turn_closed
                and interaction.turn_id == self.active_turn_id
            ),
            None,
        )
        if resolved is None:
            return
        # Resolved notifications across the supported range do not contain turnId. If a
        # future version adds it, reject a mismatch rather than weakening scope.
        if 'turnId' in params and params['turnId'] != resolved.turn_id:
            return
        if resolved is pending:
            self.pending_interaction = self._interaction_queue.popleft() if self._interaction_queue else None
        else:
            self._interaction_queue.remove(resolved)
        self._last_resolved_interaction = (resolved.request_id, resolved.thread_id, resolved.turn_id)
        logger.info('[codex-pty] interaction %s resolved by the server', resolved.interaction_id)

    def interaction_was_resolved(self, pending: PendingInteraction) -> bool:
        """Whether Codex acknowledged this exact request/thread/turn tuple."""
        return self._last_resolved_interaction == (pending.request_id, pending.thread_id, pending.turn_id)

    def _handle_item_completed(self, params: dict[str, Any]) -> None:
        """Record assistant messages belonging to this thread's current turn."""
        if params.get('threadId') != self.thread_id:
            return
        turn_id = params.get('turnId') or self.active_turn_id
        if turn_id is None or turn_id != self.active_turn_id:
            # Auxiliary threads and earlier turns must not pollute this answer.
            return
        item = params.get('item')
        if not isinstance(item, dict):
            return
        item_type = str(item.get('itemType') or item.get('type') or '')
        if item_type not in _AGENT_MESSAGE_ITEM_TYPES:
            return
        text = extract_message_text(item)
        if text:
            self._turn_messages.setdefault(str(turn_id), []).append(text)

    def _handle_interaction_request(self, message: dict[str, Any], params: dict[str, Any]) -> None:
        """Accept a request only if it belongs to this thread AND the live turn.

        A request from an auxiliary thread, from an already-finished turn, or a
        duplicate of one still pending must never become the current interaction:
        the caller would be asked to approve something other than what it thinks.
        """
        method = str(message.get('method'))
        kind = 'user_input' if method in USER_INPUT_REQUEST_METHODS | MCP_ELICITATION_REQUEST_METHODS else 'approval'
        request_id = message.get('id')
        if isinstance(request_id, bool) or not isinstance(request_id, (str, int)):
            logger.debug('[codex-pty] dropping %s with invalid request id', method)
            return
        interaction_id = str(request_id)
        thread_id = params.get('threadId')
        turn_id = params.get('turnId')

        if self.thread_id is None or self.active_turn_id is None or self._turn_closed:
            # No live turn to attach it to (e.g. it arrived after completion).
            logger.debug('[codex-pty] dropping %s with no live turn', method)
            return
        if thread_id != self.thread_id:
            logger.debug('[codex-pty] dropping %s from foreign thread %s', method, thread_id)
            return
        if method in MCP_ELICITATION_REQUEST_METHODS and turn_id is None:
            # MCP elicitations are identified by their server request id. The
            # official schema permits a null turnId when app-server cannot add
            # turn context, so associate the request with this thread's only
            # active turn.
            turn_id = self.active_turn_id
        elif turn_id != self.active_turn_id:
            logger.debug('[codex-pty] dropping %s from turn %s (active=%s)', method, turn_id, self.active_turn_id)
            return
        if self.pending_interaction is not None:
            if self.pending_interaction.request_id == request_id or any(
                queued.request_id == request_id for queued in self._interaction_queue
            ):
                logger.debug('[codex-pty] ignoring duplicate request %s', interaction_id)
                return
            self._interaction_queue.append(
                PendingInteraction(
                    interaction_id=interaction_id,
                    request_id=request_id,
                    method=method,
                    kind=kind,
                    params=params,
                    thread_id=self.thread_id,
                    turn_id=turn_id,
                )
            )
            logger.info(
                '[codex-pty] interaction %s queued behind %s',
                interaction_id,
                self.pending_interaction.interaction_id,
            )
            return

        self.pending_interaction = PendingInteraction(
            interaction_id=interaction_id,
            request_id=request_id,
            method=method,
            kind=kind,
            params=params if isinstance(params, dict) else {},
            thread_id=self.thread_id,
            turn_id=turn_id,
        )
        logger.info('[codex-pty] interaction requested: %s (%s)', method, kind)

    def _handle_turn_started(self, params: dict[str, Any]) -> None:
        thread_id = params.get('threadId')
        turn = params.get('turn') or {}
        turn_id = turn.get('id')
        if not thread_id or not turn_id:
            return
        if self.thread_id is None and self._awaiting_turn:
            # Lock onto the first thread that starts a turn for our submission.
            self.thread_id = str(thread_id)
            logger.info('[codex-pty] locked session thread %s', self.thread_id)
        if thread_id != self.thread_id:
            logger.debug('[codex-pty] dropping turn/started from auxiliary thread %s', thread_id)
            return
        if self.active_turn_id is None and self._awaiting_turn:
            self.active_turn_id = str(turn_id)
            logger.info('[codex-pty] turn started thread=%s turn=%s', self.thread_id, self.active_turn_id)

    def _handle_turn_completed(self, params: dict[str, Any]) -> None:
        thread_id = params.get('threadId')
        turn = params.get('turn') or {}
        turn_id = turn.get('id')
        if thread_id != self.thread_id:
            return
        if self.active_turn_id is None or turn_id != self.active_turn_id:
            # Residue from an earlier turn must never satisfy the current one.
            logger.debug('[codex-pty] dropping stale turn/completed turn=%s (active=%s)', turn_id, self.active_turn_id)
            return
        status = str(turn.get('status') or '')
        error = turn.get('error') or {}
        message = error.get('message') if isinstance(error, dict) else None
        if status == 'completed':
            outcome = TurnOutcome.COMPLETED
        elif status == 'interrupted':
            outcome = TurnOutcome.INTERRUPTED
        else:
            outcome = TurnOutcome.FAILED
        # Prefer the messages streamed as item/completed for this turn; fall back
        # to the items embedded in the terminal payload itself.
        messages = list(self._turn_messages.get(str(turn_id), []))
        if not messages:
            for item in turn.get('items') or []:
                if not isinstance(item, dict):
                    continue
                if str(item.get('itemType') or item.get('type') or '') in _AGENT_MESSAGE_ITEM_TYPES:
                    text = extract_message_text(item)
                    if text:
                        messages.append(text)
        self.last_result = TurnResult(
            outcome=outcome,
            thread_id=str(thread_id),
            turn_id=str(turn_id),
            error_message=message,
            final_message=messages[-1] if messages else '',
        )
        # The turn is over, so any request still pending for it is moot, and no
        # request arriving later may attach to it: mark the turn closed right here
        # rather than waiting for release() in the session layer.
        self.clear_interactions()
        self._turn_messages.pop(str(turn_id), None)
        self._turn_closed = True
        logger.info('[codex-pty] turn completed turn=%s status=%s', turn_id, status)

    # ----- queries ---------------------------------------------------------- #

    def take_interaction(self) -> PendingInteraction | None:
        pending = self.pending_interaction
        self.pending_interaction = self._interaction_queue.popleft() if self._interaction_queue else None
        return pending

    def claim_interaction(self, interaction_id: str) -> PendingInteraction | None:
        """Atomically reserve one pending request for exactly one response."""
        pending = self.pending_interaction
        if pending is None or pending.interaction_id != interaction_id or pending.claimed:
            return None
        pending.claimed = True
        return pending


__all__ = [
    'APPROVAL_REQUEST_METHODS',
    'INTERACTION_REQUEST_METHODS',
    'MCP_ELICITATION_REQUEST_METHODS',
    'USER_INPUT_REQUEST_METHODS',
    'CodexBoundaryAdapter',
    'PendingInteraction',
    'TurnOutcome',
    'TurnResult',
]
