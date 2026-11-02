"""WorkBuddy proxy projects identity headers and owns the compaction resume phase."""

from __future__ import annotations

import json

from typing import Any

import httpx
import pytest


pytest.importorskip('fastapi', reason='WorkBuddy proxy tests require Nexus runtime server dependencies')

from nexus.extensions.agents.workbuddy.gateway_proxy import WorkBuddyOpenAIProxy
from nexus.gateway.policies.workbuddy import MAIN_CONVERSATION_ID_KEY, RETRACTED_TASK_NOTIFICATION_KEY
from nexus.runtime.server.services.rsio_llm_bridge import (
    NEXUS_AGENT_COMPACTION_FIELD,
    NEXUS_COMPACTION_DETECTOR_FIELD,
)


MAIN_ID = 'main-conversation'
CHILD_ID = 'child-conversation'
NOTIFICATION = (
    '<task-notification>\n<task-id>HEevJO</task-id>\n<status>completed</status>\n</task-notification>\n\n'
    'Use the TaskOutput tool with task_id="HEevJO" to retrieve the full output if you need to act on it.'
)
HISTORY = [{'role': 'user', 'content': 'task'}, {'role': 'tool', 'tool_call_id': 'fg-1', 'content': 'fg-done'}]
NOTIFIED = [*HISTORY, {'role': 'user', 'content': [{'type': 'text', 'text': NOTIFICATION}]}]
READ = [
    *HISTORY,
    {'role': 'assistant', 'content': 'read it', 'tool_calls': [{'id': 'read-1', 'type': 'function'}]},
    {'role': 'tool', 'tool_call_id': 'read-1', 'content': 'bg-done'},
]


class _Bridge:
    def __init__(self) -> None:
        self.bodies: list[dict[str, Any]] = []
        self.fail_next = False
        self.response: dict[str, Any] = {
            'id': 'response',
            'choices': [{'message': {'role': 'assistant', 'content': 'done'}}],
        }

    async def request(self, body: dict[str, Any], timeout: float) -> dict[str, Any]:
        self.bodies.append(body)
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError('host completion failed')
        return self.response


def _headers(
    purpose: str,
    conversation_id: str = MAIN_ID,
    *,
    parent_conversation_id: str | None = None,
    request_id: str = 'request-1',
) -> dict[str, str]:
    headers = {
        'X-Agent-Type': 'subagent' if parent_conversation_id is not None else 'main',
        'X-Agent-Purpose': purpose,
        'X-Conversation-Id': conversation_id,
        'X-Request-Id': request_id,
        'Authorization': 'Bearer secret',
    }
    if parent_conversation_id is not None:
        headers['X-Parent-Conversation-Id'] = parent_conversation_id
    return headers


async def _post(
    proxy: WorkBuddyOpenAIProxy,
    headers: dict[str, str],
    *,
    stream: bool = False,
    extra: dict[str, Any] | None = None,
) -> httpx.Response:
    body = {
        'model': 'general-model',
        'messages': [{'role': 'user', 'content': 'task'}],
        'stream': stream,
        **(extra or {}),
    }
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=proxy._create_app()), base_url='http://proxy'
    ) as client:
        response = await client.post('/v1/chat/completions', json=body, headers=headers)
        await response.aread()
    return response


def _proxy(**kwargs: Any) -> tuple[WorkBuddyOpenAIProxy, _Bridge]:
    proxy = WorkBuddyOpenAIProxy(sio=None, **kwargs)
    bridge = _Bridge()
    proxy.bridge = bridge
    return proxy, bridge


@pytest.mark.asyncio
async def test_identity_headers_become_request_meta() -> None:
    proxy, bridge = _proxy()

    await _post(proxy, _headers('subagent:general-purpose', CHILD_ID, parent_conversation_id=MAIN_ID))

    assert bridge.bodies[0]['request_meta'] == {
        'request_id': 'request-1',
        'workbuddy': {
            'agent_type': 'subagent',
            'agent_purpose': 'subagent:general-purpose',
            'conversation_id': CHILD_ID,
            'parent_conversation_id': MAIN_ID,
            'request_id': 'request-1',
        },
    }
    assert NEXUS_AGENT_COMPACTION_FIELD not in bridge.bodies[0]


@pytest.mark.asyncio
async def test_main_conversation_is_stamped_on_requests_after_the_main_turn() -> None:
    proxy, bridge = _proxy()

    await _post(proxy, _headers('conversation_topic', 'title-conversation'))
    await _post(proxy, _headers('conversation'))
    await _post(proxy, _headers('subagent:general-purpose', CHILD_ID, parent_conversation_id=MAIN_ID))
    await _post(proxy, _headers('subagent:general-purpose', 'grandchild', parent_conversation_id=CHILD_ID))

    title, main, child, grandchild = bridge.bodies
    assert MAIN_CONVERSATION_ID_KEY not in title['request_meta']
    assert [body['request_meta'][MAIN_CONVERSATION_ID_KEY] for body in (main, child, grandchild)] == [MAIN_ID] * 3


@pytest.mark.asyncio
async def test_request_without_workbuddy_headers_is_forwarded_unchanged() -> None:
    proxy, bridge = _proxy()

    await _post(proxy, {'Authorization': 'Bearer secret'})

    assert 'request_meta' not in bridge.bodies[0]


@pytest.mark.asyncio
@pytest.mark.parametrize('stream', [False, True])
async def test_successful_summary_marks_only_the_next_agent_turn_as_resume(stream: bool) -> None:
    proxy, bridge = _proxy()

    await _post(proxy, _headers('context_summary_max_token'), stream=stream)
    await _post(proxy, _headers('conversation_topic', 'title-conversation'))
    await _post(proxy, _headers('conversation'))
    await _post(proxy, _headers('conversation'))

    summary, title, resume, ordinary = bridge.bodies
    assert summary[NEXUS_AGENT_COMPACTION_FIELD] is True
    assert summary[NEXUS_COMPACTION_DETECTOR_FIELD] == 'workbuddy_agent_purpose'
    assert 'invocation_kind' not in title['request_meta']
    assert resume['request_meta']['invocation_kind'] == 'compaction_resume'
    assert 'invocation_kind' not in ordinary['request_meta']


@pytest.mark.asyncio
async def test_failed_summary_does_not_arm_a_resume() -> None:
    proxy, bridge = _proxy()
    bridge.fail_next = True

    assert (await _post(proxy, _headers('context_summary_max_token'))).status_code == 500
    await _post(proxy, _headers('conversation'))

    assert 'invocation_kind' not in bridge.bodies[1]['request_meta']


@pytest.mark.asyncio
async def test_failed_resume_keeps_the_marker_for_the_retry() -> None:
    proxy, bridge = _proxy()
    await _post(proxy, _headers('context_summary_max_token'))
    bridge.fail_next = True

    assert (await _post(proxy, _headers('conversation'))).status_code == 500
    await _post(proxy, _headers('conversation'))

    assert [body['request_meta'].get('invocation_kind') for body in bridge.bodies[1:]] == [
        'compaction_resume',
        'compaction_resume',
    ]


@pytest.mark.asyncio
async def test_resume_marker_is_scoped_to_the_summarized_conversation() -> None:
    proxy, bridge = _proxy()

    await _post(proxy, _headers('context_summary_max_token', CHILD_ID))
    await _post(proxy, _headers('conversation'))
    await _post(proxy, _headers('subagent:general-purpose', CHILD_ID, parent_conversation_id=MAIN_ID))

    assert 'invocation_kind' not in bridge.bodies[1]['request_meta']
    assert bridge.bodies[2]['request_meta']['invocation_kind'] == 'compaction_resume'


@pytest.mark.asyncio
async def test_dropped_task_notification_marks_only_the_next_agent_turn() -> None:
    proxy, bridge = _proxy()

    await _post(proxy, _headers('conversation'), extra={'messages': NOTIFIED})
    await _post(proxy, _headers('conversation'), extra={'messages': READ})
    await _post(proxy, _headers('conversation'), extra={'messages': [*READ, {'role': 'user', 'content': 'next'}]})

    notified, retracted, later = bridge.bodies
    assert RETRACTED_TASK_NOTIFICATION_KEY not in notified['request_meta']
    assert retracted['request_meta'][RETRACTED_TASK_NOTIFICATION_KEY] is True
    assert RETRACTED_TASK_NOTIFICATION_KEY not in later['request_meta']


@pytest.mark.asyncio
async def test_task_notification_kept_in_history_or_on_another_conversation_is_not_a_retraction() -> None:
    proxy, bridge = _proxy()

    await _post(proxy, _headers('conversation'), extra={'messages': NOTIFIED})
    child_headers = _headers('subagent:general-purpose', CHILD_ID, parent_conversation_id=MAIN_ID)
    await _post(proxy, child_headers, extra={'messages': READ})
    await _post(proxy, _headers('conversation_topic', 'title-conversation'), extra={'messages': READ})
    await _post(proxy, _headers('conversation'), extra={'messages': [*NOTIFIED, *READ[2:]]})

    assert all(RETRACTED_TASK_NOTIFICATION_KEY not in body['request_meta'] for body in bridge.bodies)


@pytest.mark.asyncio
async def test_chat_template_kwargs_replace_the_top_level_reasoning_effort() -> None:
    proxy, bridge = _proxy(chat_template_kwargs={'reasoning_effort': 'no_think'})

    await _post(
        proxy,
        _headers('conversation'),
        extra={
            'reasoning_effort': 'high',
            'extra_body': {'top_k': 20, 'chat_template_kwargs': {'reasoning_effort': 'high', 'keep': True}},
        },
    )

    body = bridge.bodies[0]
    assert 'reasoning_effort' not in body
    assert body['extra_body'] == {
        'top_k': 20,
        'chat_template_kwargs': {'reasoning_effort': 'no_think', 'keep': True},
    }


@pytest.mark.asyncio
async def test_without_chat_template_kwargs_the_body_keeps_its_reasoning_effort() -> None:
    proxy, bridge = _proxy()

    await _post(proxy, _headers('conversation'), extra={'reasoning_effort': 'high'})

    assert bridge.bodies[0]['reasoning_effort'] == 'high'
    assert 'extra_body' not in bridge.bodies[0]


@pytest.mark.asyncio
async def test_request_copies_assistant_reasoning_onto_reasoning_content() -> None:
    proxy, bridge = _proxy()
    messages = [
        {'role': 'user', 'content': 'task', 'reasoning': 'ignore me'},
        {'role': 'assistant', 'content': 'first', 'reasoning': 'plan the edit'},
        {'role': 'assistant', 'content': 'second', 'reasoning': 'other', 'reasoning_content': 'kept'},
        {'role': 'assistant', 'content': 'third', 'reasoning': ''},
        {'role': 'tool', 'tool_call_id': 'read-1', 'content': 'done'},
    ]

    await _post(proxy, _headers('conversation'), extra={'messages': messages})

    forwarded = bridge.bodies[0]['messages']
    assert 'reasoning_content' not in forwarded[0]
    assert forwarded[1]['reasoning_content'] == 'plan the edit'
    assert forwarded[1]['reasoning'] == 'plan the edit'
    assert forwarded[2]['reasoning_content'] == 'kept'
    assert 'reasoning_content' not in forwarded[3]


@pytest.mark.asyncio
@pytest.mark.parametrize('stream', [False, True])
async def test_response_copies_reasoning_onto_reasoning_content(stream: bool) -> None:
    proxy, bridge = _proxy()
    bridge.response = {
        'id': 'response',
        'choices': [{'message': {'role': 'assistant', 'content': 'done', 'reasoning': 'plan the edit'}}],
    }

    response = await _post(proxy, _headers('conversation'), stream=stream)

    assert response.status_code == 200
    if stream:
        deltas = [
            choice.get('delta') or {}
            for line in response.text.splitlines()
            if line.startswith('data: ') and not line.endswith('[DONE]')
            for choice in json.loads(line.removeprefix('data: ')).get('choices') or []
        ]
        assert any(delta.get('reasoning_content') == 'plan the edit' for delta in deltas)
    else:
        assert response.json()['choices'][0]['message']['reasoning_content'] == 'plan the edit'
