"""Direct event-fact policy for the official Claude Code CLI wire protocol.

Pinned to Claude Code 2.1.286 (npm ``stable`` on 2026-10-09), captured through the
Anthropic proxy in print, stream-json and interactive sessions. It shares the Claude wire
grammar of the HyCode policy, minus what the official CLI no longer emits:

- Compaction uses the same trigger and resume markers.
- Subagent requests, and only those, carry ``x-claude-code-agent-id``. Spawns go through
  ``Agent``, whose result ends with the ``agentId`` trailer; background agents join
  through ``<task-notification>``, sent as a user message when the main thread is idle or
  appended to the running tool's result otherwise.
- The ``Task`` alias and the ``TaskOutput`` tool no longer exist.
- Session-title requests are utility traffic. The interactive TUI also sends a
  prompt-suggestion request after each turn that forks from the main prefix;
  ``CLAUDE_CODE_ENABLE_PROMPT_SUGGESTION=false`` stops it at the source. HyCode's
  session-memory and magic-docs prompts are gone from the binary.
- Resent tool calls are rewritten unless the CLI echoes wire tool inputs:
  ``normalizeToolInput`` strips ``cd <cwd> && `` from Bash and fills ``replace_all`` into
  Edit. The echo follows the GrowthBook flag ``tengu_humble_hammock``, which is off when the
  flag service is unreachable (as in sandboxes); ``CLAUDE_CODE_HUMBLE_HAMMOCK=1`` forces it
  on. Strict identity keeps each rewrite as a fork. JSON number spelling (``5.0`` -> ``5``)
  is equal under protocol identity.
"""

from __future__ import annotations

import json
import re

from dataclasses import replace
from typing import Any

from nexus.gateway.policies.hycode import ASYNC_LAUNCH_ACK, COMPACT_SUMMARY_MARKER, COMPACT_TRIGGER_MARKER
from nexus.gateway.scaffold_policy import (
    AnchoredIdentity,
    BoundaryFact,
    ChildPromptCandidate,
    DeliveryReceipt,
    GenericScaffoldPolicy,
    InputMessage,
    InvocationKind,
    NotificationDelivery,
    OutputMessage,
    RequestEvent,
    ResponseEvent,
    ResponsePolicy,
    SidechainMembership,
    SpawnBinding,
    SpawnIntent,
    protocol_message_identity,
)


CLAUDE_CODE_SIGNALS_VERSION = '2.1.286'

SPAWN_TOOL = 'Agent'

_AGENT_ID_LINE = re.compile(r'(?m)^agentId:\s*([^\s(<]+)\s+\((?:internal ID\b|use SendMessage\b)')
_TASK_NOTIFICATION = re.compile(r'<task-notification>(.*?)</task-notification>', re.DOTALL)
_SYSTEM_NOTIFICATION = re.compile(
    r'<system-reminder>\s*\[SYSTEM NOTIFICATION - NOT USER INPUT\](.*?)</system-reminder>', re.DOTALL
)
_TERMINAL_TASK_STATUSES = frozenset({'completed', 'failed', 'killed'})

# A small wire grammar, not a second policy DSL. All required fragments must
# occur in the selected message; ``line`` allows a flattened system prompt.
_UTILITY_MARKERS = (
    (
        'session_title',
        'system',
        'line',
        ('You are naming a coding session so the user can pick it out of a long list of sessions.',),
    ),
    (
        'prompt_suggestion',
        'user',
        'last',
        ('[SUGGESTION MODE: Suggest what the user might naturally type next into Claude Code.]',),
    ),
)


class ClaudeCodeScaffoldPolicy(GenericScaffoldPolicy):
    """Translate pinned Claude Code wire signals into scaffold-neutral facts."""

    response_policy = ResponsePolicy.PREFER_EXTENDED_ELSE_LAST

    def __init__(self) -> None:
        super().__init__(policy_id='claude_code')

    @staticmethod
    def _text_blocks(content: Any) -> list[str]:
        if isinstance(content, str):
            return [content]
        if not isinstance(content, list):
            return []
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.extend(ClaudeCodeScaffoldPolicy._text_blocks(item.get('text')))
                parts.extend(ClaudeCodeScaffoldPolicy._text_blocks(item.get('content')))
        return parts

    @classmethod
    def _text(cls, message: dict[str, Any]) -> str:
        return '\n'.join(cls._text_blocks(message.get('content')))

    @classmethod
    def _starts_a_block(cls, message: dict[str, Any], marker: str) -> bool:
        return any(block.lstrip().startswith(marker) for block in cls._text_blocks(message.get('content')))

    @staticmethod
    def _message_identity(message: dict[str, Any]):
        if message.get('role') != 'assistant' or not message.get('tool_calls'):
            return protocol_message_identity(message)
        normalized = dict(message)
        content = normalized.get('content')
        is_blank = isinstance(content, str) and content.strip() == ''
        if content is None or is_blank:
            normalized['content'] = None
        return protocol_message_identity(normalized)

    @staticmethod
    def _argument(function: dict[str, Any], key: str) -> str:
        try:
            arguments = json.loads(function.get('arguments') or '{}')
        except (TypeError, ValueError):
            return ''
        value = arguments.get(key) if isinstance(arguments, dict) else None
        return value if isinstance(value, str) else ''

    @staticmethod
    def _tag(text: str, name: str) -> str | None:
        match = re.search(rf'<{re.escape(name)}>(.*?)</{re.escape(name)}>', text, re.DOTALL)
        return match.group(1).strip() if match is not None else None

    @classmethod
    def _utility_membership(cls, messages: tuple[dict[str, Any], ...]) -> SidechainMembership | None:
        for kind, role, position, required in _UTILITY_MARKERS:
            candidates = messages[-1:] if position == 'last' else messages
            for message in candidates:
                if message.get('role') != role:
                    continue
                text = cls._text(message)
                if position == 'line':
                    matched = any(all(fragment in line for fragment in required) for line in text.splitlines())
                else:
                    matched = all(fragment in text for fragment in required)
                if matched:
                    return SidechainMembership(
                        chain_id='',
                        kind=kind,
                        detector=f'marker:claude_code-{CLAUDE_CODE_SIGNALS_VERSION}',
                    )
        return None

    @classmethod
    def _message_facts(cls, messages, anchors):
        spawns = []
        bindings = []
        receipts = []
        notifications = []
        tool_names: dict[str, str] = {}

        for message, anchor in zip(messages, anchors, strict=True):
            for call in message.get('tool_calls') or []:
                function = call.get('function') or {}
                call_id = call.get('id')
                name = function.get('name')
                if not isinstance(call_id, str) or not isinstance(name, str):
                    continue
                tool_names[call_id] = name
                if name == SPAWN_TOOL:
                    prompt = ' '.join(cls._argument(function, 'prompt').split())
                    if prompt:
                        spawns.append(SpawnIntent(anchor, call_id, prompt))

        for message, anchor in zip(messages, anchors, strict=True):
            text = cls._text(message)
            call_id = message.get('tool_call_id')
            if message.get('role') == 'tool' and isinstance(call_id, str) and tool_names.get(call_id) == SPAWN_TOOL:
                # A synchronous result is the child's output; a background launch only acknowledges it.
                receipts.append(DeliveryReceipt(anchor, call_id, not text.lstrip().startswith(ASYNC_LAUNCH_ACK)))
                match = _AGENT_ID_LINE.search(text)
                if match is not None:
                    bindings.append(SpawnBinding(anchor, call_id, match.group(1)))

            if message.get('role') == 'user':
                bodies = _TASK_NOTIFICATION.findall(text)
            elif message.get('role') == 'tool':
                # A notification that lands while a tool runs is appended to that tool's result, inside
                # the producer's system-notification reminder; tool output itself is never a notification.
                bodies = [
                    body
                    for reminder in _SYSTEM_NOTIFICATION.findall(text)
                    for body in _TASK_NOTIFICATION.findall(reminder)
                ]
            else:
                continue
            for body in bodies:
                if cls._tag(body, 'task-type') is not None:
                    continue
                agent_id = cls._tag(body, 'task-id')
                status = cls._tag(body, 'status')
                if agent_id is None or status not in _TERMINAL_TASK_STATUSES:
                    continue
                tool_use_id = cls._tag(body, 'tool-use-id')
                spawning_tool = tool_names.get(tool_use_id) if tool_use_id is not None else None
                if spawning_tool not in (None, SPAWN_TOOL):
                    continue
                if message.get('role') == 'tool' and spawning_tool != SPAWN_TOOL:
                    continue
                notifications.append(NotificationDelivery(anchor, agent_id))
                if tool_use_id is not None:
                    bindings.append(SpawnBinding(anchor, tool_use_id, agent_id))

        return tuple(spawns), tuple(bindings), tuple(receipts), tuple(notifications)

    @classmethod
    def _child_prompts(cls, messages: tuple[dict[str, Any], ...]):
        for index, message in enumerate(messages):
            if message.get('role') != 'user':
                continue
            candidates = []
            for block in cls._text_blocks(message.get('content')):
                if block.lstrip().startswith('<system-reminder>'):
                    continue
                prompt = ' '.join(block.split())
                if prompt:
                    candidates.append(ChildPromptCandidate(InputMessage(index), prompt, len(candidates)))
            if candidates:
                return tuple(candidates)
        return ()

    @classmethod
    def _boundary_facts(cls, messages: tuple[dict[str, Any], ...]):
        user_indexes = [index for index, message in enumerate(messages) if message.get('role') == 'user']
        if not user_indexes:
            return ()
        facts = []
        first, last = user_indexes[0], user_indexes[-1]
        if cls._starts_a_block(messages[first], COMPACT_SUMMARY_MARKER):
            facts.append(
                BoundaryFact('claude_code_compaction_resume', InputMessage(first), 'compaction_resume', 'compaction')
            )
        # A past summary trigger describes a historical invocation, not the next
        # output. Keep it out of boundary assembly as well as budget classification.
        if last == len(messages) - 1 and cls._starts_a_block(messages[last], COMPACT_TRIGGER_MARKER):
            facts.append(
                BoundaryFact(
                    'claude_code_summary_inference',
                    InputMessage(last),
                    'summary_inference',
                    'compaction',
                    at_request_edge=True,
                )
            )
        return tuple(facts)

    def inspect_event(self, event):
        facts = super().inspect_event(event)
        request = event.request if isinstance(event, ResponseEvent) else event
        if not isinstance(request, RequestEvent):
            return facts

        meta = request.request_meta or {}
        agent_id = self._optional_str(meta.get('agent_id'))
        membership = self._utility_membership(request.messages)
        if membership is None and agent_id is not None:
            membership = SidechainMembership(chain_id=agent_id, kind='subagent', detector='header:agent_id')
        chain_key = 'main' if membership is None else (membership.chain_id or f'utility:{membership.kind}')
        context_key = (self.policy_id, chain_key, membership is not None)

        request_boundaries = self._boundary_facts(request.messages)
        summary_boundary = next(
            (
                boundary
                for boundary in request_boundaries
                if boundary.segment_kind == 'summary_inference' and boundary.at_request_edge
            ),
            None,
        )
        if summary_boundary is not None:
            if facts.invocation_source == 'request_meta':
                if facts.invocation_kind != InvocationKind.SUMMARY_INFERENCE:
                    raise ValueError('request metadata invocation_kind conflicts with the Claude Code summary trigger')
            else:
                facts = replace(
                    facts,
                    invocation_kind=InvocationKind.SUMMARY_INFERENCE,
                    invocation_source=summary_boundary.rule_name,
                )

        if isinstance(event, ResponseEvent):
            messages = (event.message,)
            anchors = (OutputMessage(),)
            boundary_facts = ()
            child_prompts = ()
        else:
            messages = event.messages
            anchors = tuple(InputMessage(index) for index in range(len(messages)))
            boundary_facts = request_boundaries
            child_prompts = (
                self._child_prompts(messages) if membership is not None and membership.kind == 'subagent' else ()
            )
        spawns, bindings, receipts, notifications = self._message_facts(messages, anchors)
        timeline = dict(facts.timeline)
        if timeline:
            timeline['is_sidechain'] = membership is not None
        return replace(
            facts,
            message_identities=tuple(
                AnchoredIdentity(anchor, self._message_identity(message))
                for message, anchor in zip(messages, anchors, strict=True)
            ),
            context_key=context_key,
            membership=membership,
            spawn_intents=spawns,
            spawn_bindings=bindings,
            delivery_receipts=receipts,
            notification_deliveries=notifications,
            boundary_facts=boundary_facts,
            child_prompts=child_prompts,
            timeline=tuple(timeline.items()),
        )


CLAUDE_CODE_POLICY = ClaudeCodeScaffoldPolicy()

__all__ = ['CLAUDE_CODE_POLICY', 'CLAUDE_CODE_SIGNALS_VERSION', 'ClaudeCodeScaffoldPolicy']
