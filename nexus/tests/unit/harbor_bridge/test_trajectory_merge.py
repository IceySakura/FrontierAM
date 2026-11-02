from __future__ import annotations

import json

from nexus.extensions.harbor_bridge.anti_hack.strategies.anthropic_blocks import (
    normalize_anthropic_block_messages,
)
from nexus.extensions.harbor_bridge.anti_hack.strategies.restricted_source_access_v1 import (
    detect_reward_hacking,
)
from nexus.extensions.harbor_bridge.anti_hack.strategies.trajectory_merge import merge_trajectory_messages
from nexus.extensions.harbor_bridge.anti_hack.text_parse import normalize_text_parse_messages


COMMAND = 'curl -s https://raw.githubusercontent.com/django/django/main/django/db/models.py'


def test_anthropic_tool_normalization_preserves_thinking_as_reasoning() -> None:
    messages = [
        {
            'role': 'assistant',
            'content': [
                {'type': 'thinking', 'thinking': 'Let me inspect the repository first.'},
                {'type': 'text', 'text': 'I will inspect the repository.'},
                {'type': 'tool_use', 'id': 'call-1', 'name': 'Read', 'input': {'file_path': '/tmp/a.py'}},
            ],
        }
    ]

    normalized = normalize_anthropic_block_messages(messages)

    assert normalized is not None
    assert normalized[0]['content'] == 'I will inspect the repository.'
    assert normalized[0]['reasoning_content'] == 'Let me inspect the repository first.'
    assert normalized[0]['tool_calls'][0]['function']['name'] == 'Read'


def test_anthropic_tool_result_normalization_preserves_error_metadata() -> None:
    messages = [
        {
            'role': 'assistant',
            'content': [{
                'type': 'tool_use',
                'id': 'call-1',
                'name': 'Edit',
                'input': {'file_path': '/tmp/a.py', 'old_string': 'a', 'new_string': 'b'},
            }],
        },
        {
            'role': 'user',
            'content': [{
                'type': 'tool_result',
                'tool_use_id': 'call-1',
                'content': '<tool_use_error>String to replace not found</tool_use_error>',
                'is_error': True,
                'status': 'error',
                'inferred_outcome': 'failure',
                'exit_code': 1,
            }],
        },
    ]

    normalized = normalize_anthropic_block_messages(messages)

    assert normalized is not None
    assert normalized[1] == {
        'role': 'tool',
        'tool_call_id': 'call-1',
        'content': '<tool_use_error>String to replace not found</tool_use_error>',
        'is_error': True,
        'status': 'error',
        'inferred_outcome': 'failure',
        'exit_code': 1,
    }


def _call(call_id: str, observation: str | None) -> list[dict]:
    call = {
        'role': 'assistant',
        'content': '',
        'tool_calls': [
            {
                'id': call_id,
                'type': 'function',
                'function': {'name': 'Bash', 'arguments': json.dumps({'command': COMMAND})},
            }
        ],
    }
    if observation is None:
        return [call]
    return [call, {'role': 'tool', 'tool_call_id': call_id, 'content': observation}]


def _call_ids(messages: list[dict]) -> list[str]:
    return [str(call.get('id') or '') for message in messages for call in message.get('tool_calls') or []]


def _results(messages: list[dict]) -> dict[str, str]:
    return {
        str(message.get('tool_call_id') or ''): str(message.get('content') or '')
        for message in messages
        if message.get('role') == 'tool'
    }


def _detected(messages: list[dict]) -> bool:
    messages = normalize_text_parse_messages(messages) or messages
    result = detect_reward_hacking(
        messages,
        'django__django-1',
        'django/django',
        None,
        0.0,
        task_type='SWE',
        base_commit='a' * 40,
        history_truncated=False,
    )
    return bool(result['reward_hack_mask'])


def test_same_command_with_different_ids_keeps_both_branch_calls() -> None:
    base = _call('main-call', 'def fixed_behavior():\n    return True')
    snapshot = _call('subagent-call', 'curl: (6) Could not resolve host')

    merged = merge_trajectory_messages(base, snapshot)

    assert _call_ids(merged) == ['subagent-call', 'main-call']
    assert set(_results(merged)) == {'subagent-call', 'main-call'}
    assert _detected(base) is True
    assert _detected(snapshot) is False
    assert _detected(merged) is True


def test_different_ids_do_not_backfill_each_others_observations() -> None:
    base = _call('main-call', 'downloaded source')
    snapshot = _call('subagent-call', None)

    merged = merge_trajectory_messages(base, snapshot)

    assert _call_ids(merged) == ['subagent-call', 'main-call']
    assert _results(merged) == {'main-call': 'downloaded source'}


def test_ordinal_fallback_remains_available_when_one_side_has_no_id() -> None:
    base = _call('', 'downloaded source')
    snapshot = _call('snapshot-call', None)

    merged = merge_trajectory_messages(base, snapshot)

    assert _call_ids(merged) == ['snapshot-call']
    assert _results(merged) == {'snapshot-call': 'downloaded source'}


def test_text_parse_merge_keeps_the_final_execution_result() -> None:
    base = [
        {'role': 'assistant', 'content': f'<keystrokes>{COMMAND}</keystrokes>'},
        {'role': 'user', 'content': 'curl: (6) Could not resolve host: raw.githubusercontent.com'},
    ]
    snapshot = [{'role': 'assistant', 'content': f'<keystrokes>{COMMAND}</keystrokes>'}]

    merged = merge_trajectory_messages(base, snapshot)

    assert _call_ids(merged) == ['']
    assert _results(merged) == {'': 'curl: (6) Could not resolve host: raw.githubusercontent.com'}
    assert _detected(base) is False
    assert _detected(merged) is False


def test_text_parse_merge_does_not_backfill_a_different_command() -> None:
    other_command = 'curl -s https://raw.githubusercontent.com/django/django/main/README.rst'
    base = [
        {'role': 'assistant', 'content': f'<keystrokes>{COMMAND}</keystrokes>'},
        {'role': 'user', 'content': 'curl: (6) Could not resolve host: raw.githubusercontent.com'},
    ]
    snapshot = [{'role': 'assistant', 'content': f'<keystrokes>{other_command}</keystrokes>'}]

    merged = merge_trajectory_messages(base, snapshot)

    assert len(_call_ids(merged)) == 2
    assert [
        message.get('content') for message in merged if message.get('role') == 'tool' and message.get('content')
    ] == ['curl: (6) Could not resolve host: raw.githubusercontent.com']
