"""Version-tolerant direct trajectory policy for Codex Responses requests."""

from __future__ import annotations

import json
import re

from collections.abc import Iterator
from dataclasses import replace

from nexus.gateway.scaffold_policy import (
    BoundaryFact,
    ChildPromptCandidate,
    DeclaredChainClaim,
    GenericScaffoldPolicy,
    InputMessage,
    NotificationDelivery,
    ObservationEvent,
    OutputMessage,
    ParentClaim,
    RequestEvent,
    ResponseEvent,
    ResponsePolicy,
    SidechainMembership,
    SpawnBinding,
    SpawnIntent,
)
from nexus.runtime.server.services.codex_compaction import (
    DEFAULT_CODEX_COMPACTION_PROMPT_PREFIX,
    final_user_message_text,
)


_SPAWN_TOOL_NAMESPACES = frozenset({None, 'collaboration', 'multi_agent_v1'})
_SUBAGENT_V1_NOTIFICATION = re.compile(r'<subagent_notification>\s*(.*?)\s*</subagent_notification>', re.DOTALL)
_SUBAGENT_V1_TERMINAL_STATUSES = frozenset({'completed', 'failed', 'errored', 'cancelled'})


class CodexScaffoldPolicy(GenericScaffoldPolicy):
    """Prefer explicit Codex metadata and conservatively fall back to prompt evidence."""

    response_policy = ResponsePolicy.PREFER_EXTENDED_ELSE_LAST

    def __init__(self) -> None:
        super().__init__(policy_id='codex')

    def inspect_event(self, event):
        facts = super().inspect_event(event)
        request = event.request if isinstance(event, ResponseEvent) else event
        if isinstance(request, ObservationEvent):
            messages = request.parent_messages or ()
            request_meta = request.request_meta
        elif isinstance(request, RequestEvent):
            messages = request.messages
            request_meta = request.request_meta
        else:
            return facts

        meta = request_meta if isinstance(request_meta, dict) else {}
        raw_codex = meta.get('codex')
        codex = raw_codex if isinstance(raw_codex, dict) else {}
        request_id = self._optional_str(meta.get('request_id'))
        request_kind = self._optional_str(codex.get('request_kind'))
        session_id = self._optional_str(codex.get('session_id'))
        thread_id = self._optional_str(codex.get('thread_id'))
        window_id = self._optional_str(codex.get('window_id'))
        parent_thread_id = self._optional_str(codex.get('parent_thread_id'))
        thread_source = self._optional_str(codex.get('thread_source'))
        subagent_kind = self._optional_str(codex.get('subagent_kind'))

        if request_kind == 'memory':
            context_key = (self.policy_id, 'memory', request_id)
            membership = None
            declared_claim = None
            parent_claim = None
        elif session_id is not None and thread_id is not None and window_id is not None:
            context_key = (self.policy_id, thread_id, window_id)
            is_root = thread_id == session_id
            if is_root or parent_thread_id is None:
                membership = None
                declared_claim = None
                parent_claim = None
            else:
                parent_chain_id = 'main' if parent_thread_id == session_id else parent_thread_id
                kind = (
                    'subagent'
                    if thread_source == 'subagent' and subagent_kind == 'thread_spawn'
                    else subagent_kind or 'sidechain'
                )
                attributes = tuple(
                    (key, value)
                    for key, value in (
                        ('thread_source', thread_source),
                        ('codex_subagent_kind', subagent_kind),
                        ('parent_chain_id', parent_chain_id),
                        ('subagent_header', self._optional_str(codex.get('subagent_header'))),
                        ('parent_thread_id', parent_thread_id),
                    )
                    if value is not None
                )
                membership = SidechainMembership(
                    chain_id=thread_id,
                    kind=kind,
                    parent_chain_id=parent_chain_id,
                    detector='codex_turn_metadata',
                    attributes=attributes,
                )
                declared_claim = DeclaredChainClaim(thread_id, kind, parent_chain_id)
                parent_claim = ParentClaim(parent_chain_id)
        else:
            context_key = (self.policy_id, 'main', False)
            membership = None
            declared_claim = None
            parent_claim = None

        child_prompts = (
            self._child_prompt_candidates(messages)
            if membership is not None and isinstance(event, RequestEvent)
            else ()
        )
        spawn_intents = self._spawn_intents(event) if isinstance(event, ResponseEvent) else ()
        spawn_bindings = self._spawn_bindings(messages, meta) if isinstance(event, RequestEvent) else ()
        notification_deliveries = self._notification_deliveries(messages) if isinstance(event, RequestEvent) else ()
        boundary_facts = self._boundary_facts(event, messages, meta, request_kind)
        timeline = dict(facts.timeline)
        if timeline:
            timeline.update({
                'request_id': request_id,
                'agent_id': thread_id,
                'parent_agent_id': parent_thread_id,
                'is_sidechain': membership is not None,
                'scaffold': 'codex',
                'scaffold_version': self._optional_str(meta.get('scaffold_version')),
                'request_kind': request_kind,
                'window_id': window_id,
                'codex_metadata_source': self._optional_str(meta.get('codex_metadata_source')),
            })
        return replace(
            facts,
            context_key=context_key,
            request_id=request_id,
            agent_id=thread_id,
            membership=membership,
            declared_claim=declared_claim,
            parent_claim=parent_claim,
            spawn_intents=spawn_intents,
            spawn_bindings=spawn_bindings,
            notification_deliveries=notification_deliveries,
            boundary_facts=boundary_facts,
            child_prompts=child_prompts,
            timeline=tuple((key, value) for key, value in timeline.items() if value is not None),
        )

    @staticmethod
    def _child_prompt_candidates(messages) -> tuple[ChildPromptCandidate[InputMessage], ...]:
        candidates = []
        order = 0
        for index, message in enumerate(messages):
            if message.get('role') not in ('user', 'developer'):
                continue
            content = message.get('content')
            if not isinstance(content, str) or not content:
                continue
            candidates.append(ChildPromptCandidate(InputMessage(index), content, order))
            order += 1
        return tuple(candidates)

    @staticmethod
    def _spawn_intents(event: ResponseEvent) -> tuple[SpawnIntent[OutputMessage], ...]:
        return tuple(
            SpawnIntent(OutputMessage(), call_id, child_prompt)
            for call_id, arguments in _spawn_calls(event.message, event.request.request_meta)
            if (child_prompt := _child_prompt(arguments))
        )

    @staticmethod
    def _spawn_bindings(messages, request_meta) -> tuple[SpawnBinding[InputMessage], ...]:
        spawn_call_ids: set[str] = set()
        bindings = []
        for index, message in enumerate(messages):
            spawn_call_ids.update(call_id for call_id, _ in _spawn_calls(message, request_meta))
            call_id = message.get('tool_call_id')
            if message.get('role') != 'tool' or not isinstance(call_id, str) or call_id not in spawn_call_ids:
                continue
            agent_id = _json_object(message.get('content')).get('agent_id')
            if isinstance(agent_id, str) and agent_id:
                bindings.append(SpawnBinding(InputMessage(index), call_id, agent_id))
        return tuple(bindings)

    @staticmethod
    def _notification_deliveries(messages) -> tuple[NotificationDelivery[InputMessage], ...]:
        deliveries = []
        for index, message in enumerate(messages):
            if message.get('role') != 'user':
                continue
            content = message.get('content')
            if not isinstance(content, str):
                continue
            for raw_notification in _SUBAGENT_V1_NOTIFICATION.findall(content):
                try:
                    notification = json.loads(raw_notification)
                except (TypeError, ValueError):
                    continue
                if not isinstance(notification, dict):
                    continue
                agent_id = notification.get('agent_path')
                status = notification.get('status')
                if not isinstance(agent_id, str) or not agent_id or not isinstance(status, dict):
                    continue
                if not _SUBAGENT_V1_TERMINAL_STATUSES.intersection(status):
                    continue
                deliveries.append(NotificationDelivery(InputMessage(index), agent_id))
        return tuple(deliveries)

    @staticmethod
    def _boundary_facts(event, messages, meta, request_kind):
        if not isinstance(event, RequestEvent):
            return ()
        anchor = InputMessage(len(messages) - 1) if messages else OutputMessage()
        invocation_kind = meta.get('invocation_kind')
        if request_kind == 'compaction':
            return (
                BoundaryFact(
                    'codex_compaction_metadata',
                    anchor,
                    'summary_inference',
                    'compaction',
                    from_request_context=True,
                    at_request_edge=True,
                ),
            )
        if invocation_kind in ('compaction_resume', 'context_restart_unknown'):
            is_compaction = invocation_kind == 'compaction_resume'
            return (
                BoundaryFact(
                    'codex_compaction_resume' if is_compaction else 'codex_context_restart_unknown',
                    anchor,
                    invocation_kind,
                    'compaction' if is_compaction else 'context_restart',
                    from_request_context=True,
                    at_request_edge=True,
                ),
            )
        if request_kind is not None:
            return ()
        detector = meta.get('codex_compaction_detector')
        if not isinstance(detector, str):
            text = final_user_message_text(messages, require_responses_message_type=False)
            detector = (
                'default_prompt_prefix'
                if text is not None and text.strip().startswith(DEFAULT_CODEX_COMPACTION_PROMPT_PREFIX)
                else None
            )
        if detector is None:
            return ()
        return (
            BoundaryFact(
                'codex_compaction_prompt_fallback',
                anchor,
                'summary_inference',
                'compaction',
                from_request_context=isinstance(meta.get('codex_compaction_detector'), str),
                at_request_edge=True,
            ),
        )


def _json_object(value: object) -> dict:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return {}
    return value if isinstance(value, dict) else {}


def _is_spawn_tool(provider_tool_name: object, request_meta: dict | None) -> bool:
    tool_identities = request_meta.get('codex_tool_identities') if isinstance(request_meta, dict) else None
    if isinstance(tool_identities, dict) and isinstance(provider_tool_name, str):
        tool_identity = tool_identities.get(provider_tool_name)
        if isinstance(tool_identity, dict):
            return (
                tool_identity.get('tool_type') == 'function'
                and tool_identity.get('name') == 'spawn_agent'
                and tool_identity.get('namespace') in _SPAWN_TOOL_NAMESPACES
            )
    return provider_tool_name == 'spawn_agent'


def _spawn_calls(message: dict, request_meta: dict | None) -> Iterator[tuple[str, dict]]:
    """Yield (tool_call_id, arguments) for each spawn_agent call in one assistant message."""
    for tool_call in message.get('tool_calls') or ():
        if not isinstance(tool_call, dict) or not isinstance(tool_call.get('id'), str):
            continue
        function = tool_call.get('function')
        if isinstance(function, dict) and _is_spawn_tool(function.get('name'), request_meta):
            yield tool_call['id'], _json_object(function.get('arguments'))


def _child_prompt(arguments: dict) -> str | None:
    prompt = arguments.get('message') or arguments.get('prompt') or arguments.get('task')
    items = arguments.get('items')
    # Codex V1 accepts text items instead of message; the separator must match the
    # Responses converter so the child's first prompt still correlates.
    if prompt is None and isinstance(items, list) and items:
        texts = [
            item['text']
            for item in items
            if isinstance(item, dict) and item.get('type') == 'text' and isinstance(item.get('text'), str)
        ]
        if len(texts) == len(items):
            prompt = '\n\n'.join(texts)
    return prompt if isinstance(prompt, str) and prompt else None


CODEX_POLICY = CodexScaffoldPolicy()

__all__ = ['CODEX_POLICY', 'CodexScaffoldPolicy']
