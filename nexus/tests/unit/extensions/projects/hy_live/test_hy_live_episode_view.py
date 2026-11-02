from __future__ import annotations

import asyncio
import json

import pytest

from hylive.cerebellum_contract import CEREBELLUM_TOOLS
from hylive.cerebellum_rules import take_notices
from hylive.notices import receipt_status
from hylive.prompts import FAILURE_TEMPLATE, NOTICE_TEMPLATE

from nexus.extensions.projects.hy_live import agent as agent_module
from nexus.extensions.projects.hy_live import task as task_module
from nexus.extensions.projects.hy_live.episode_view import (
    EPISODE_VIEW_KEY,
    build_episode_view,
    classify_user_message,
)
from nexus.extensions.projects.hy_live.health import FORMAT_ERROR_TERMINATION
from nexus.extensions.projects.hy_live.user_simulator import CerebellumResponse, SimulatedConversation


_AUDIO_PART = {'type': 'audio', 'audio': {'patch': '/data/a.wav', 'audio_length': 16000, 'init_frequency': 16000}}


def _notice_response(content: str, kinds: list[str], receipt_status: str | None) -> CerebellumResponse:
    return CerebellumResponse(
        content=content,
        metadata={'source': 'notification', 'notice_kinds': kinds, 'receipt_status': receipt_status},
    )


def _delegate_call() -> list[dict]:
    return [
        {
            'id': 'call-1',
            'type': 'function',
            'function': {'name': 'delegate', 'arguments': json.dumps({'text': '写一个贪吃蛇'})},
        }
    ]


def test_classify_user_message_tells_every_template_apart() -> None:
    assert classify_user_message([_AUDIO_PART]) == 'user_speech'
    assert classify_user_message(NOTICE_TEMPLATE.format(fact='有一项工作做完了。')) == 'notice'
    assert classify_user_message(FAILURE_TEMPLATE.format(detail='timeout')) == 'failure'
    assert classify_user_message('[通知] 这是用户自己说的话') == 'user_text', (
        'Only the full template header marks an injected message'
    )
    assert classify_user_message('再帮我改一下') == 'user_text'


def test_episode_view_labels_messages_and_attaches_notice_metadata() -> None:
    permission = NOTICE_TEMPLATE.format(fact='有一条权限请求需要用户决定。\n工作：贪吃蛇')
    receipt = NOTICE_TEMPLATE.format(fact='status=ok\n贪吃蛇还在写')
    messages = [
        {'role': 'system', 'content': 'cerebellum prompt'},
        {'role': 'user', 'content': '写一个贪吃蛇'},
        {'role': 'assistant', 'content': '', 'tool_calls': _delegate_call()},
        {'role': 'tool', 'tool_call_id': 'call-1', 'name': 'delegate', 'content': '{"status": "submitted"}'},
        {'role': 'assistant', 'content': '我去写个贪吃蛇'},
        {'role': 'user', 'content': permission},
        {'role': 'assistant', 'content': '要允许吗'},
        {'role': 'user', 'content': receipt},
        {'role': 'assistant', 'content': '还在写'},
    ]

    view = build_episode_view(
        messages=messages,
        delivery_responses=[
            CerebellumResponse(content='我去写个贪吃蛇', metadata={'source': 'user'}),
            _notice_response('要允许吗', ['permission'], None),
            _notice_response('还在写', ['question'], 'ok'),
        ],
        termination_reason='ENV_DONE',
        health_status='healthy',
        tools=CEREBELLUM_TOOLS,
    )

    assert [message['kind'] for message in view['messages']] == [
        'system',
        'user_text',
        'assistant',
        'tool',
        'assistant',
        'notice',
        'assistant',
        'notice',
        'assistant',
    ]
    assert view['messages'][2]['tool_calls'] == _delegate_call()
    assert view['messages'][3]['name'] == 'delegate'
    assert view['messages'][5]['notice_kinds'] == ['permission']
    assert view['messages'][5]['receipt_status'] is None
    assert view['messages'][7]['notice_kinds'] == ['question']
    assert view['messages'][7]['receipt_status'] == 'ok'
    assert [tool['function']['name'] for tool in view['tools']] == [
        tool['function']['name'] for tool in CEREBELLUM_TOOLS
    ]
    json.dumps(view, ensure_ascii=False)


def test_episode_view_compacts_speech_turns() -> None:
    view = build_episode_view(
        messages=[{'role': 'system', 'content': 'p'}, {'role': 'user', 'content': [_AUDIO_PART]}],
        delivery_responses=[],
        termination_reason='ENV_DONE',
        health_status='healthy',
        tools=CEREBELLUM_TOOLS,
    )

    assert view['messages'][1] == {'role': 'user', 'kind': 'user_speech', 'content': [{'type': 'audio'}]}


def test_episode_view_leaves_metadata_off_a_notice_that_ended_in_format_error() -> None:
    view = build_episode_view(
        messages=[
            {'role': 'system', 'content': 'p'},
            {'role': 'user', 'content': NOTICE_TEMPLATE.format(fact='有一项工作做完了。')},
        ],
        delivery_responses=[],
        termination_reason='FORMAT_ERROR',
        health_status='format_error',
        tools=CEREBELLUM_TOOLS,
    )

    assert 'notice_kinds' not in view['messages'][1]


def test_episode_view_rejects_more_notice_responses_than_notice_messages() -> None:
    with pytest.raises(ValueError, match='2 notice responses but only 1'):
        build_episode_view(
            messages=[{'role': 'user', 'content': NOTICE_TEMPLATE.format(fact='有一项工作做完了。')}],
            delivery_responses=[
                _notice_response('a', ['completed'], None),
                _notice_response('b', ['completed'], None),
            ],
            termination_reason='ENV_DONE',
            health_status='healthy',
            tools=CEREBELLUM_TOOLS,
        )


def test_receipt_status_is_read_only_from_midbrain_receipts() -> None:
    receipt = {'kind': 'cancel', 'fact': 'status=ask\n有两项都叫贪吃蛇', 'priority': ''}
    backend_fact = {'kind': 'completed', 'fact': 'status=ok', 'priority': 'normal'}

    assert receipt_status(receipt) == 'ask'
    assert receipt_status(backend_fact) is None


def test_merged_completions_keep_every_part_kind() -> None:
    def completed(work: str) -> dict:
        return {
            'kind': 'completed',
            'content': '',
            'path': '',
            'fact': f'{work}做好了',
            'priority': 'normal',
            'fields': [('work', work), ('result', '做好了')],
            'delivery_ids': [],
            'kinds': ['completed'],
        }

    notice, held = take_notices([completed('贪吃蛇'), completed('俄罗斯方块')])

    assert held == []
    assert notice is not None
    assert notice['kinds'] == ['completed', 'completed']


def test_live_agent_task_exposes_episode_view_in_extra_fields() -> None:
    async def run_agent(**_kwargs):
        return agent_module.LiveAgentAgentOutput(
            main_trajectory=[
                {'role': 'system', 'content': 'cerebellum prompt'},
                {'role': 'user', 'content': '你好'},
                {'role': 'assistant', 'content': '你好呀'},
            ],
            owner_id='owner-1',
            summary='你好呀',
            conversation=SimulatedConversation(turns=[], ended_by_user=True),
            tasks=[],
            episode_termination=FORMAT_ERROR_TERMINATION,
        )

    result = asyncio.run(task_module.live_agent_task(task_module.LiveAgentTaskInput(prompt='hi', run_agent=run_agent)))

    view = result.extra_fields[EPISODE_VIEW_KEY]
    assert view['termination_reason'] == 'FORMAT_ERROR'
    assert view['health_status'] == 'format_error'
    assert [message['kind'] for message in view['messages']] == ['system', 'user_text', 'assistant']
    assert view['tools'] == CEREBELLUM_TOOLS, 'The episode reward judges tool decisions against the cerebellum tools'
