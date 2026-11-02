"""Direct event-fact policy for WorkBuddy request identity headers.

The WorkBuddy proxy copies ``x-agent-*`` and ``x-conversation-*`` headers into
``request_meta.workbuddy``. Chain membership, summary invocations, and compaction resumes
come from those headers. Message content binds an ``Agent`` call to the child that it started,
because no header names the spawning tool call, and marks the ``<task-notification>`` that wb
later drops from history.

This module also owns the header and purpose vocabulary, so the runtime proxy projects
exactly the fields that this policy interprets.
"""

from __future__ import annotations

import json
import re

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from nexus.gateway.scaffold_policy import (
    AnchoredIdentity,
    BoundaryFact,
    ChildPromptCandidate,
    DeliveryReceipt,
    GenericScaffoldPolicy,
    InputMessage,
    InvocationKind,
    OutputMessage,
    ParentClaim,
    RequestEvent,
    ResponseEvent,
    ResponsePolicy,
    SidechainMembership,
    SpawnIntent,
    protocol_message_identity,
)


WORKBUDDY_META_KEY = 'workbuddy'
WORKBUDDY_HEADER_FIELDS: dict[str, str] = {
    'x-agent-type': 'agent_type',
    'x-agent-purpose': 'agent_purpose',
    'x-conversation-id': 'conversation_id',
    'x-parent-conversation-id': 'parent_conversation_id',
    'x-root-request-id': 'root_request_id',
    'x-conversation-request-id': 'conversation_request_id',
    'x-request-id': 'request_id',
}

# Purposes observed with WorkBuddy 5.7.0. Unknown values must be added here only after a
# capture shows which conversation they belong to and whether they summarize history.
CONVERSATION_PURPOSE = 'conversation'
TITLE_PURPOSE = 'conversation_topic'
SUBAGENT_PURPOSE_PREFIX = 'subagent:'
SUMMARY_PURPOSES = frozenset({'context_summary_max_token'})
SUMMARY_DETECTOR = 'workbuddy_agent_purpose'

WORKBUDDY_SIGNALS_VERSION = '5.7.0'
SPAWN_TOOL_NAME = 'Agent'
# Only an async spawn returns this acknowledgement. wb spawns async only when its team system is on and
# the run is not single-shot; otherwise it runs the child to completion and returns the reply even when
# the call asked for run_in_background.
BACKGROUND_SPAWN_ACK = 'Spawned successfully. agent_id:'

# wb shows a background task's <task-notification> to one model call and drops it from later history,
# so the next request no longer extends the prefix that call was generated from. Requests cannot show
# this alone; the proxy compares consecutive requests and sets this request_meta key.
TASK_NOTIFICATION_PREFIX = '<task-notification>'
RETRACTED_TASK_NOTIFICATION_KEY = 'workbuddy_retracted_task_notification'

# No header names the run's main conversation, so the proxy records it here. A child whose parent is that
# conversation must claim the 'main' chain; a raw conversation id names no chain, and the parent would then
# fall back to prompt matches, which other chains' Agent calls with the same prompt make ambiguous.
MAIN_CONVERSATION_ID_KEY = 'workbuddy_main_conversation_id'

_USER_QUERY_BLOCK = re.compile(r'\s*<user_query>(.*)</user_query>\s*', re.DOTALL)


def build_workbuddy_request_meta(headers: Mapping[str, str]) -> dict[str, Any] | None:
    """Copy WorkBuddy identity headers into Gateway request metadata without interpreting them.

    Returns:
        ``{'request_id': ..., 'workbuddy': {...}}`` with the non-empty headers, or None when the
        request carries none of them.
    """
    lowered = {str(key).lower(): str(value) for key, value in headers.items()}
    fields = {field: lowered[header] for header, field in WORKBUDDY_HEADER_FIELDS.items() if lowered.get(header)}
    if not fields:
        return None
    request_meta: dict[str, Any] = {WORKBUDDY_META_KEY: fields}
    if 'request_id' in fields:
        request_meta['request_id'] = fields['request_id']
    return request_meta


def trailing_task_notification(messages: Sequence[Mapping[str, Any]]) -> str | None:
    """Return the text of a ``<task-notification>`` sent as the request's last message, if any."""
    if not messages or messages[-1].get('role') != 'user':
        return None
    text = message_text(messages[-1])
    return text if text is not None and text.lstrip().startswith(TASK_NOTIFICATION_PREFIX) else None


def message_text(message: Mapping[str, Any]) -> str | None:
    """Join a message's text content, or None when it is not all text."""
    texts = WorkBuddyScaffoldPolicy._text_blocks(message.get('content'))
    return ''.join(texts) if texts is not None else None


def is_agent_turn_purpose(purpose: str | None) -> bool:
    """Return whether a purpose is an ordinary main or child agent turn."""
    if purpose == CONVERSATION_PURPOSE:
        return True
    return (
        isinstance(purpose, str) and purpose.startswith(SUBAGENT_PURPOSE_PREFIX) and purpose != SUBAGENT_PURPOSE_PREFIX
    )


@dataclass(frozen=True)
class _RequestIdentity:
    purpose: str
    agent_type: str | None
    conversation_id: str
    parent_conversation_id: str | None
    main_conversation_id: str | None = None

    @property
    def is_title(self) -> bool:
        return self.purpose == TITLE_PURPOSE

    @property
    def is_summary(self) -> bool:
        return self.purpose in SUMMARY_PURPOSES

    @property
    def is_child(self) -> bool:
        return self.parent_conversation_id is not None


class WorkBuddyScaffoldPolicy(GenericScaffoldPolicy):
    """Translate WorkBuddy request identity headers into scaffold-neutral facts."""

    response_policy = ResponsePolicy.PREFER_EXTENDED_ELSE_LAST

    def __init__(self) -> None:
        super().__init__(policy_id='workbuddy')

    def inspect_event(self, event):
        facts = super().inspect_event(event)
        request = event.request if isinstance(event, ResponseEvent) else event
        if not isinstance(request, RequestEvent):
            return facts

        meta = request.request_meta or {}
        identity = self._request_identity(meta)
        membership, context_key, parent_claim = self._membership(identity)

        boundary_facts = ()
        if isinstance(event, RequestEvent):
            boundary_facts = self._boundary_facts(identity, meta, len(request.messages))
        if identity.is_summary:
            if facts.invocation_source == 'request_meta' and facts.invocation_kind != InvocationKind.SUMMARY_INFERENCE:
                raise ValueError(
                    f'WorkBuddy x-agent-purpose={identity.purpose!r} is a summary request, but '
                    f'request_meta.invocation_kind={meta.get("invocation_kind")!r}'
                )
            if facts.invocation_source != 'request_meta':
                facts = replace(
                    facts,
                    invocation_kind=InvocationKind.SUMMARY_INFERENCE,
                    invocation_source='workbuddy_summary_inference',
                )
        elif facts.invocation_kind == InvocationKind.SUMMARY_INFERENCE:
            raise ValueError(
                f'request_meta.invocation_kind=summary_inference conflicts with WorkBuddy '
                f'x-agent-purpose={identity.purpose!r}; summary purposes are {sorted(SUMMARY_PURPOSES)}'
            )

        if isinstance(event, ResponseEvent):
            messages = (event.message,)
            anchors = (OutputMessage(),)
            child_prompts = ()
        else:
            messages = event.messages
            anchors = tuple(InputMessage(index) for index in range(len(messages)))
            child_prompts = self._child_prompts(messages) if identity.is_child and not identity.is_summary else ()
        # A summary request re-renders past turns under its own system prompt; those Agent calls
        # and results are history, and claiming them again would move joins onto the copies.
        spawn_intents, delivery_receipts = (
            ((), ()) if identity.is_summary else self._agent_call_facts(messages, anchors)
        )

        timeline = dict(facts.timeline)
        if timeline:
            timeline.update({
                'agent_id': identity.conversation_id,
                'parent_agent_id': identity.parent_conversation_id,
                'is_sidechain': membership is not None,
                'scaffold': 'workbuddy',
                'workbuddy_purpose': identity.purpose,
            })
        return replace(
            facts,
            message_identities=tuple(
                AnchoredIdentity(anchor, self._message_identity(message))
                for message, anchor in zip(messages, anchors, strict=True)
            ),
            context_key=context_key,
            agent_id=identity.conversation_id,
            membership=membership,
            parent_claim=parent_claim,
            spawn_intents=spawn_intents,
            delivery_receipts=delivery_receipts,
            boundary_facts=boundary_facts,
            child_prompts=child_prompts,
            timeline=tuple((key, value) for key, value in timeline.items() if value is not None),
        )

    @staticmethod
    def _request_identity(meta: dict[str, Any]) -> _RequestIdentity:
        raw = meta.get(WORKBUDDY_META_KEY)
        if not isinstance(raw, dict):
            raise ValueError(
                'WorkBuddy request has no request_meta.workbuddy; run WorkBuddy behind WorkBuddyOpenAIProxy '
                'so its x-agent-* and x-conversation-* headers reach the Gateway'
            )
        purpose = raw.get('agent_purpose')
        conversation_id = raw.get('conversation_id')
        if not isinstance(purpose, str) or not purpose:
            raise ValueError('WorkBuddy request is missing the x-agent-purpose header')
        if not isinstance(conversation_id, str) or not conversation_id:
            raise ValueError(f'WorkBuddy request with x-agent-purpose={purpose!r} is missing x-conversation-id')
        known = purpose in (TITLE_PURPOSE, CONVERSATION_PURPOSE) or purpose in SUMMARY_PURPOSES
        if not known and not is_agent_turn_purpose(purpose):
            expected = sorted({
                TITLE_PURPOSE,
                CONVERSATION_PURPOSE,
                f'{SUBAGENT_PURPOSE_PREFIX}<type>',
                *SUMMARY_PURPOSES,
            })
            raise ValueError(
                f'Unknown WorkBuddy x-agent-purpose={purpose!r}; expected one of {expected}. Capture the request '
                'headers and register the purpose in nexus/gateway/policies/workbuddy.py before trusting its segments.'
            )
        parent_conversation_id = raw.get('parent_conversation_id') or None
        agent_type = raw.get('agent_type')
        main_conversation_id = meta.get(MAIN_CONVERSATION_ID_KEY)
        identity = _RequestIdentity(
            purpose,
            agent_type,
            conversation_id,
            parent_conversation_id,
            main_conversation_id if isinstance(main_conversation_id, str) and main_conversation_id else None,
        )
        if identity.is_title:
            return identity
        expected_agent_type = 'subagent' if identity.is_child else 'main'
        if agent_type != expected_agent_type:
            raise ValueError(
                f'WorkBuddy x-agent-type={agent_type!r} conflicts with '
                f'x-parent-conversation-id={parent_conversation_id!r}; expected x-agent-type={expected_agent_type!r}'
            )
        if not identity.is_summary and purpose.startswith(SUBAGENT_PURPOSE_PREFIX) != identity.is_child:
            raise ValueError(
                f'WorkBuddy x-agent-purpose={purpose!r} conflicts with '
                f'x-parent-conversation-id={parent_conversation_id!r}'
            )
        return identity

    def _membership(self, identity: _RequestIdentity):
        if identity.is_title:
            membership = SidechainMembership(chain_id='', kind='session_title', detector='header:x-agent-purpose')
            return membership, (self.policy_id, 'utility:session_title', True), None
        parent_conversation_id = identity.parent_conversation_id
        if parent_conversation_id is None:
            return None, (self.policy_id, 'main', False), None
        attributes = [('parent_conversation_id', parent_conversation_id)]
        if identity.purpose.startswith(SUBAGENT_PURPOSE_PREFIX):
            attributes.append(('subagent_type', identity.purpose[len(SUBAGENT_PURPOSE_PREFIX) :]))
        membership = SidechainMembership(
            chain_id=identity.conversation_id,
            kind='subagent',
            detector='header:x-parent-conversation-id',
            attributes=tuple(attributes),
        )
        parent_chain_id = 'main' if parent_conversation_id == identity.main_conversation_id else parent_conversation_id
        return (
            membership,
            (self.policy_id, identity.conversation_id, True),
            ParentClaim(parent_chain_id),
        )

    @staticmethod
    def _boundary_facts(identity: _RequestIdentity, meta: dict[str, Any], message_count: int):
        if message_count == 0:
            return ()
        anchor = InputMessage(message_count - 1)
        if identity.is_summary:
            return (
                BoundaryFact(
                    'workbuddy_summary_inference',
                    anchor,
                    'summary_inference',
                    'compaction',
                    from_request_context=True,
                    at_request_edge=True,
                ),
            )
        if meta.get('invocation_kind') == 'compaction_resume':
            if identity.is_title:
                raise ValueError('WorkBuddy title requests cannot resume a compacted conversation')
            return (
                BoundaryFact(
                    'workbuddy_compaction_resume',
                    anchor,
                    'compaction_resume',
                    'compaction',
                    from_request_context=True,
                    at_request_edge=True,
                ),
            )
        if meta.get(RETRACTED_TASK_NOTIFICATION_KEY) is True:
            return (
                BoundaryFact(
                    'workbuddy_task_notification_retracted',
                    anchor,
                    'render_restart',
                    'history_rewrite',
                    from_request_context=True,
                    at_request_edge=True,
                ),
            )
        return ()

    @staticmethod
    def _text_blocks(content: Any) -> list[str] | None:
        """Return the texts of a string or all-text block list, or None for other content."""
        if isinstance(content, str):
            return [content]
        if not isinstance(content, list) or not content:
            return None
        texts = []
        for block in content:
            if not isinstance(block, dict) or block.get('type') != 'text' or not isinstance(block.get('text'), str):
                return None
            texts.append(block['text'])
        return texts

    @classmethod
    def _message_identity(cls, message: dict[str, Any]):
        # WorkBuddy echoes assistant text as [{'type': 'text', 'text': ..., 'annotations': []}],
        # while the Gateway records the generated string.
        if message.get('role') != 'assistant':
            return protocol_message_identity(message)
        normalized = dict(message)
        texts = cls._text_blocks(normalized.get('content'))
        if texts is not None:
            normalized['content'] = ''.join(texts)
        content = normalized.get('content')
        if normalized.get('tool_calls') and (content is None or (isinstance(content, str) and not content.strip())):
            normalized['content'] = None
        return protocol_message_identity(normalized)

    @staticmethod
    def _normalized_prompt(text: str) -> str:
        return ' '.join(text.split())

    @classmethod
    def _child_prompts(cls, messages: tuple[dict[str, Any], ...]):
        for index, message in enumerate(messages):
            if message.get('role') != 'user':
                continue
            candidates = []
            for block in cls._text_blocks(message.get('content')) or ():
                match = _USER_QUERY_BLOCK.fullmatch(block)
                prompt = cls._normalized_prompt(match.group(1) if match is not None else block)
                if prompt:
                    candidates.append(ChildPromptCandidate(InputMessage(index), prompt, len(candidates)))
            if candidates:
                return tuple(candidates)
        return ()

    @classmethod
    def _agent_call_facts(cls, messages, anchors):
        spawn_intents = []
        spawn_call_ids: set[str] = set()
        for message, anchor in zip(messages, anchors, strict=True):
            for call in message.get('tool_calls') or ():
                function = call.get('function') if isinstance(call, dict) else None
                call_id = call.get('id') if isinstance(call, dict) else None
                if not isinstance(function, dict) or function.get('name') != SPAWN_TOOL_NAME:
                    continue
                if not isinstance(call_id, str) or not call_id:
                    continue
                spawn_call_ids.add(call_id)
                prompt = _json_object(function.get('arguments')).get('prompt')
                if isinstance(prompt, str) and (normalized := cls._normalized_prompt(prompt)):
                    spawn_intents.append(SpawnIntent(anchor, call_id, normalized))

        # An async spawn's later delivery protocol has not been captured, so its child stays unjoined.
        delivery_receipts = tuple(
            DeliveryReceipt(anchor, message['tool_call_id'], not cls._is_background_spawn_ack(message))
            for message, anchor in zip(messages, anchors, strict=True)
            if message.get('role') == 'tool' and message.get('tool_call_id') in spawn_call_ids
        )
        return tuple(spawn_intents), delivery_receipts

    @classmethod
    def _is_background_spawn_ack(cls, message: dict[str, Any]) -> bool:
        return ''.join(cls._text_blocks(message.get('content')) or ()).lstrip().startswith(BACKGROUND_SPAWN_ACK)


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return {}
    return value if isinstance(value, dict) else {}


WORKBUDDY_POLICY = WorkBuddyScaffoldPolicy()

__all__ = [
    'CONVERSATION_PURPOSE',
    'SUBAGENT_PURPOSE_PREFIX',
    'SUMMARY_DETECTOR',
    'SUMMARY_PURPOSES',
    'TITLE_PURPOSE',
    'WORKBUDDY_HEADER_FIELDS',
    'WORKBUDDY_META_KEY',
    'WORKBUDDY_POLICY',
    'WORKBUDDY_SIGNALS_VERSION',
    'WorkBuddyScaffoldPolicy',
    'build_workbuddy_request_meta',
    'is_agent_turn_purpose',
]
