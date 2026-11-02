"""One-pass normalization and tool-call/result association for all rules."""

from __future__ import annotations

from typing import Any

from .models import CallEvent, Finding
from .utils import function, parse_arguments, text, tool_index


DEFAULT_MAX_THINKING_CHARS = 100_000


class DetectionContext:
    def __init__(
        self,
        messages: list[dict[str, Any]] | None,
        tools: list[dict[str, Any]] | None,
        *,
        reasoning_effort: str | None,
        ignore_incomplete_final_assistant: bool,
        termination_reason: str | None = None,
        max_thinking_chars: int = DEFAULT_MAX_THINKING_CHARS,
    ):
        if type(max_thinking_chars) is not int or max_thinking_chars <= 0:
            raise ValueError('max_thinking_chars must be a positive integer')
        self.max_thinking_chars = max_thinking_chars
        self.messages = messages if isinstance(messages, list) else []
        self.tools = tools if isinstance(tools, list) else []
        self.reasoning_effort = reasoning_effort
        self.ignore_incomplete_final_assistant = ignore_incomplete_final_assistant
        self.termination_reason = str(termination_reason or '').strip().upper()
        self.registry = tool_index(self.tools)
        self.assistant_numbers: dict[int, int] = {}
        self.calls: list[CallEvent] = []
        self.malformed: list[Finding] = []
        self.duplicate_ids: list[Finding] = []
        self.orphan_responses: list[Finding] = []
        self.missing_responses: list[Finding] = []
        self._build()

    def assistant_messages(self):
        for index, message in enumerate(self.messages):
            if isinstance(message, dict) and message.get('role') == 'assistant':
                yield index, message

    def content(self, index: int) -> str:
        return text(self.messages[index])

    def reasoning(self, index: int) -> str:
        message = self.messages[index]
        return text(message, 'reasoning_content') or text(message, 'thinking')

    def _build(self) -> None:
        assistant_turn = 0
        pending: dict[str, CallEvent] = {}
        seen_ids: set[str] = set()
        for index, raw in enumerate(self.messages):
            if not isinstance(raw, dict):
                continue
            role = raw.get('role')
            if role == 'assistant':
                self.assistant_numbers[index] = assistant_turn
                assistant_turn += 1
                raw_calls = raw.get('tool_calls')
                if raw_calls is not None and not isinstance(raw_calls, list):
                    self.malformed.append(Finding(index, 'tool_calls 不是 list', ['tool_calls'], raw_calls))
                    continue
                for call_index, call in enumerate(raw_calls or []):
                    if not isinstance(call, dict):
                        self.malformed.append(
                            Finding(
                                index,
                                'tool_call 不是 object',
                                ['tool_calls', call_index],
                                call,
                                tool_call_index=call_index,
                            )
                        )
                        continue
                    fn = function(call)
                    raw_id = call.get('id')
                    call_id = raw_id if isinstance(raw_id, str) and raw_id else None
                    name = str(fn.get('name') or '') if fn else ''
                    arguments, error = parse_arguments(fn.get('arguments') if fn else None)
                    event = CallEvent(index, call_index, call, fn, call_id, name, arguments, error)
                    self.calls.append(event)
                    if call_id is None:
                        self.malformed.append(
                            Finding(
                                index,
                                'tool_call 缺少非空 id',
                                ['tool_calls', call_index, 'id'],
                                tool_call_index=call_index,
                            )
                        )
                    else:
                        if call_id in seen_ids:
                            self.duplicate_ids.append(
                                Finding(
                                    index,
                                    f'重复 tool_call id {call_id!r}',
                                    ['tool_calls', call_index, 'id'],
                                    tool_call_index=call_index,
                                    tool_call_id=call_id,
                                )
                            )
                        seen_ids.add(call_id)
                        pending[call_id] = event
                    if fn is None:
                        self.malformed.append(
                            Finding(
                                index,
                                'tool_call.function 缺失或不是 object',
                                ['tool_calls', call_index, 'function'],
                                tool_call_index=call_index,
                                tool_call_id=call_id,
                            )
                        )
                    elif not name:
                        self.malformed.append(
                            Finding(
                                index,
                                'tool_call 缺少 function.name',
                                ['tool_calls', call_index, 'function', 'name'],
                                tool_call_index=call_index,
                                tool_call_id=call_id,
                            )
                        )
            elif role == 'tool':
                raw_id = raw.get('tool_call_id')
                call_id = raw_id if isinstance(raw_id, str) and raw_id else None
                if call_id is None:
                    self.orphan_responses.append(Finding(index, 'tool 回执缺少 tool_call_id', ['tool_call_id']))
                elif call_id not in pending:
                    self.orphan_responses.append(
                        Finding(index, f'tool 回执引用未知 call id {call_id!r}', ['tool_call_id'], tool_call_id=call_id)
                    )
                else:
                    event = pending.pop(call_id)
                    event.result_index = index
                    event.result = raw
        final_index = len(self.messages) - 1
        for call_id, event in pending.items():
            if self.ignore_incomplete_final_assistant and event.message_index == final_index:
                continue
            self.missing_responses.append(
                Finding(
                    event.message_index,
                    f'tool call {call_id!r} 没有匹配的 tool 回执',
                    ['tool_calls', event.tool_call_index],
                    tool_call_index=event.tool_call_index,
                    tool_call_id=call_id,
                )
            )
