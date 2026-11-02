"""Forward WorkBuddy request identity and compaction phases to the Gateway."""

from __future__ import annotations

import logging

from collections.abc import Mapping
from typing import Any

from nexus.gateway.policies.workbuddy import (
    CONVERSATION_PURPOSE,
    MAIN_CONVERSATION_ID_KEY,
    RETRACTED_TASK_NOTIFICATION_KEY,
    SUMMARY_DETECTOR,
    SUMMARY_PURPOSES,
    WORKBUDDY_META_KEY,
    build_workbuddy_request_meta,
    is_agent_turn_purpose,
    message_text,
    trailing_task_notification,
)
from nexus.runtime.server.services.openai_proxy import OpenAIProxy
from nexus.runtime.server.services.rsio_llm_bridge import NEXUS_AGENT_COMPACTION_FIELD, NEXUS_COMPACTION_DETECTOR_FIELD


logger = logging.getLogger(__name__)


def backfill_reasoning_content(message: dict[str, Any]) -> bool:
    """Copy a plaintext ``reasoning`` string onto ``reasoning_content``.

    wb's ``ReasoningContentBackfillRule`` does this, then skips every compatibility rule when
    the request host is on the product-gateway list. Nexus publishes the local proxy as
    ``CODEBUDDY_BASE_URL``, so that host matches and the rule never runs. The proxy keeps the
    copy: a later wb release that runs the rule again still leaves an existing string in place.
    """
    if message.get('role') != 'assistant':
        return False
    reasoning = message.get('reasoning')
    if not isinstance(reasoning, str) or not reasoning:
        return False
    if isinstance(message.get('reasoning_content'), str):
        return False
    message['reasoning_content'] = reasoning
    return True


def _backfill_messages(messages: Any) -> tuple[int, int]:
    """Return ``(copied, already_present)`` for assistant messages."""
    copied = 0
    already_present = 0
    if not isinstance(messages, list):
        return copied, already_present
    for message in messages:
        if not isinstance(message, dict) or message.get('role') != 'assistant':
            continue
        reasoning = message.get('reasoning')
        if isinstance(reasoning, str) and reasoning and isinstance(message.get('reasoning_content'), str):
            already_present += 1
        if backfill_reasoning_content(message):
            copied += 1
    return copied, already_present


class WorkBuddyOpenAIProxy(OpenAIProxy):
    """Project WorkBuddy identity headers into request metadata and mark compaction resumes.

    WorkBuddy declares summary requests in ``x-agent-purpose`` but sends nothing that marks the
    first turn after a summary. The proxy therefore owns that phase: a successful summary on one
    conversation makes the next agent turn on the same conversation a ``compaction_resume``.

    The same holds for a background task's ``<task-notification>``: wb sends it as the last
    message of one request and leaves it out of later history. The proxy remembers it per
    conversation and marks the next agent turn that no longer carries it.

    No header names the main conversation either, so the proxy records it from the main agent's
    turns and stamps it onto later requests; a child of it then claims the ``main`` chain.

    ``chat_template_kwargs`` are pinned onto every request, so the backend renders WorkBuddy's
    turns with the same template settings as training, e.g. ``{'reasoning_effort': 'no_think'}``.
    """

    def __init__(self, *args: Any, chat_template_kwargs: Mapping[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._chat_template_kwargs = dict(chat_template_kwargs or {})
        self._pending_compaction_resumes: set[str] = set()
        self._task_notifications: dict[str, str] = {}
        self._main_conversation_id: str | None = None

    def _prepare_request_body(self, body: dict[str, Any]) -> dict[str, Any]:
        copied, already_present = _backfill_messages(body.get('messages'))
        if copied or already_present:
            logger.info(
                'reasoning_content backfill request copied=%d already_present=%d',
                copied,
                already_present,
            )
        if not self._chat_template_kwargs:
            return body
        # The host OpenAI SDK only forwards non-standard fields such as chat_template_kwargs via extra_body.
        extra_body = dict(body.get('extra_body') or {})
        extra_body['chat_template_kwargs'] = {
            **(extra_body.get('chat_template_kwargs') or {}),
            **self._chat_template_kwargs,
        }
        prepared = {**body, 'extra_body': extra_body}
        if 'reasoning_effort' in self._chat_template_kwargs:
            # HY backends map a top-level reasoning_effort onto the same template field.
            prepared.pop('reasoning_effort', None)
        return prepared

    def _inspect_response_payload(self, payload: dict[str, Any]) -> None:
        # The parent dumps the host completion into this dict before SSE replay and the JSON
        # response, and only forwards an existing reasoning_content. Copy here so a `reasoning`
        # field still reaches wb.
        choices = payload.get('choices')
        if not isinstance(choices, list):
            return
        messages = [
            choice.get('message')
            for choice in choices
            if isinstance(choice, dict) and isinstance(choice.get('message'), dict)
        ]
        copied, already_present = _backfill_messages(messages)
        if copied or already_present:
            logger.info(
                'reasoning_content backfill response copied=%d already_present=%d',
                copied,
                already_present,
            )

    def _apply_request_headers(self, body: dict[str, Any], headers: Mapping[str, str]) -> dict[str, Any]:
        request_meta = build_workbuddy_request_meta(headers)
        if request_meta is None:
            return body
        identity = request_meta[WORKBUDDY_META_KEY]
        purpose = identity.get('agent_purpose')
        if purpose == CONVERSATION_PURPOSE and not identity.get('parent_conversation_id'):
            self._main_conversation_id = identity.get('conversation_id')
        if self._main_conversation_id is not None:
            request_meta[MAIN_CONVERSATION_ID_KEY] = self._main_conversation_id
        annotated = {**body, 'request_meta': request_meta}
        if purpose in SUMMARY_PURPOSES:
            annotated[NEXUS_AGENT_COMPACTION_FIELD] = True
            annotated[NEXUS_COMPACTION_DETECTOR_FIELD] = SUMMARY_DETECTOR
        elif is_agent_turn_purpose(purpose):
            conversation_id = identity.get('conversation_id')
            if conversation_id in self._pending_compaction_resumes:
                request_meta['invocation_kind'] = 'compaction_resume'
            elif self._drops_task_notification(conversation_id, body.get('messages')):
                request_meta[RETRACTED_TASK_NOTIFICATION_KEY] = True
        return annotated

    def _drops_task_notification(self, conversation_id: str | None, messages: Any) -> bool:
        notification = self._task_notifications.get(conversation_id) if conversation_id is not None else None
        if notification is None or not isinstance(messages, list):
            return False
        return all(
            message_text(message) != notification
            for message in messages
            if isinstance(message, dict) and message.get('role') == 'user'
        )

    def _record_completed_request(self, body: dict[str, Any]) -> None:
        request_meta = body.get('request_meta')
        if not isinstance(request_meta, dict):
            return
        identity = request_meta.get(WORKBUDDY_META_KEY)
        conversation_id = identity.get('conversation_id') if isinstance(identity, dict) else None
        if conversation_id is None:
            return
        # A failed resume keeps the marker so WorkBuddy's retry is still the resume turn.
        if body.get(NEXUS_AGENT_COMPACTION_FIELD) is True:
            self._pending_compaction_resumes.add(conversation_id)
        elif request_meta.get('invocation_kind') == 'compaction_resume':
            self._pending_compaction_resumes.discard(conversation_id)
        if is_agent_turn_purpose(identity.get('agent_purpose')):
            notification = trailing_task_notification(body.get('messages') or [])
            if notification is None:
                self._task_notifications.pop(conversation_id, None)
            else:
                self._task_notifications[conversation_id] = notification


__all__ = ['WorkBuddyOpenAIProxy', 'backfill_reasoning_content']
