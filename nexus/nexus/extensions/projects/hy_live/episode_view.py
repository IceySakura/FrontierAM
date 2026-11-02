"""Project one hy-live episode into the view an episode-level reward scores.

The cerebellum conversation (``main_trajectory``) carries every ``role=user`` message the
cerebellum saw, but real user turns and the injected messages (``[通知]``, ``[失败]``) all share
that role. The reward needs to tell them apart to
know what triggered each cerebellum turn, so this module labels each message by its template
and attaches what each delivered ``[通知]`` told: which facts and, for a midbrain receipt, its
status.
"""

from __future__ import annotations

from typing import Any, Literal

from hylive.prompts import FAILURE_TEMPLATE, NOTICE_TEMPLATE

from .user_simulator import CerebellumResponse


EPISODE_VIEW_KEY = 'hylive_episode'

MessageKind = Literal[
    'system',
    'user_speech',
    'user_text',
    'notice',
    'failure',
    'assistant',
    'tool',
]

_INJECTED_KINDS: tuple[tuple[str, MessageKind], ...] = tuple(
    (template.split('{', 1)[0], kind)
    for template, kind in (
        (NOTICE_TEMPLATE, 'notice'),
        (FAILURE_TEMPLATE, 'failure'),
    )
)
# Every delivered notice is told through this template, in delivery order.
_DELIVERED_KINDS = frozenset({'notice'})


def classify_user_message(content: Any) -> MessageKind:
    """Label one ``role=user`` message by the template that produced it."""
    if isinstance(content, list):
        return 'user_speech'
    if not isinstance(content, str):
        raise TypeError(f'hy-live user message content must be str or list, got {type(content).__name__}')
    for head, kind in _INJECTED_KINDS:
        if content.startswith(head):
            return kind
    return 'user_text'


def _compact_speech_content(content: list[Any]) -> list[dict[str, Any]]:
    # NOTE(jaxlliu): the reward pairs the k-th speech turn with the k-th transcript, so only the
    # part types matter; dropping the audio refs keeps the view small.
    return [{'type': part.get('type')} if isinstance(part, dict) else {'type': 'text'} for part in content]


def build_episode_view(
    *,
    messages: list[dict[str, Any]],
    delivery_responses: list[CerebellumResponse],
    termination_reason: str,
    health_status: str,
    tools: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the labelled cerebellum conversation consumed by the episode reward.

    Args:
        messages: The cerebellum conversation, system message first.
        delivery_responses: One entry per notice the cerebellum answered, in delivery order.
        termination_reason: Episode termination reason reported to the trainer.
        health_status: Episode health derived from the termination reason.
        tools: The tool specs the cerebellum saw for the whole episode.

    Returns:
        A JSON-serializable dict with the termination reason, health status, tools and
        messages, each message carrying a ``kind`` label. A ``notice`` message the cerebellum
        answered also carries ``notice_kinds`` (the kind of every fact it told)
        and ``receipt_status`` (the midbrain's status for a receipt, None for a backend fact).

    Raises:
        ValueError: A message has an unexpected role, or more notices were answered than the
            conversation delivered.
    """
    notice_metadata = [
        response.metadata for response in delivery_responses if response.metadata.get('source') == 'notification'
    ]
    view_messages: list[dict[str, Any]] = []
    delivered = 0
    for message in messages:
        role = message.get('role')
        content = message.get('content')
        if role == 'user':
            kind = classify_user_message(content)
        elif role in ('system', 'assistant', 'tool'):
            kind = role
        else:
            raise ValueError(
                f'hy-live cerebellum message has unexpected role {role!r}; expected system/user/assistant/tool'
            )

        view_message: dict[str, Any] = {'role': role, 'kind': kind, 'content': content}
        # classify_user_message labels only list content as speech; the isinstance spells that out for ty.
        if kind == 'user_speech' and isinstance(content, list):
            view_message['content'] = _compact_speech_content(content)
        if role == 'assistant' and message.get('tool_calls'):
            view_message['tool_calls'] = message['tool_calls']
        if role == 'tool':
            view_message['name'] = message.get('name')
        if kind in _DELIVERED_KINDS:
            # NOTE(jaxlliu): a notice whose cerebellum turn raised a format error has a message but no
            # response, so it is the one notice allowed to lack metadata. It is always the last one,
            # and a FORMAT_ERROR episode is scored without the judge.
            if delivered < len(notice_metadata):
                view_message['notice_kinds'] = list(notice_metadata[delivered].get('notice_kinds') or [])
                view_message['receipt_status'] = notice_metadata[delivered].get('receipt_status')
            delivered += 1
        view_messages.append(view_message)

    if delivered < len(notice_metadata):
        raise ValueError(
            f'hy-live recorded {len(notice_metadata)} notice responses but only {delivered} notice messages '
            'in the cerebellum conversation'
        )
    return {
        'termination_reason': termination_reason,
        'health_status': health_status,
        'messages': view_messages,
        'tools': list(tools),
    }


__all__ = ['EPISODE_VIEW_KEY', 'MessageKind', 'build_episode_view', 'classify_user_message']
