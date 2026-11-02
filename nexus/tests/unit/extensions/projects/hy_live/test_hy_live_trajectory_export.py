"""Cerebellum and midbrain logs export as ``cerebellum.jsonl`` / ``midbrain.jsonl`` records."""

from __future__ import annotations

import json
import stat

from datetime import datetime
from pathlib import Path

from nexus.extensions.projects.hy_live.example import cli
from nexus.extensions.projects.hy_live.trajectory_export import (
    BEIJING,
    dialogue_record,
    normalize_messages,
    stamp,
    trajectory_record,
)


_AT = datetime(2026, 10, 9, 11, 31, 13, 459740, tzinfo=BEIJING).timestamp()
_TOOL = {
    'type': 'function',
    'function': {
        'name': 'delegate',
        'description': '交给后台',
        'parameters': {'type': 'object', 'properties': {'text': {'type': 'string'}}, 'required': ['text']},
    },
}


def test_dialogue_merges_both_layers_by_time_and_opens_turns_on_real_user_lines() -> None:
    cerebellum_log = [
        {'role': 'system', 'content': '小脑', '_at': _AT},
        {'role': 'user', 'content': '写一个贪吃蛇', '_at': _AT + 1},
        {
            'role': 'assistant',
            'content': '',
            'tool_calls': [{'id': 'c1', 'function': {'name': 'delegate', 'arguments': '{"text": "写一个贪吃蛇"}'}}],
            '_at': _AT + 2,
        },
        {
            'role': 'tool',
            'tool_call_id': 'c1',
            'name': 'delegate',
            'content': '{"status": "submitted"}',
            '_at': _AT + 3,
        },
        {'role': 'user', 'content': '[通知]\n权限', '_at': _AT + 6, '_injected': True},
        {'role': 'user', 'content': '都允许', '_at': _AT + 8},
    ]
    midbrain_log = [
        {'role': 'system', 'content': '中脑', '_at': _AT},
        {'role': 'user', 'content': '小脑调用了 delegate', '_at': _AT + 4},
        {'role': 'assistant', 'content': '', '_at': _AT + 5},
    ]

    record = dialogue_record([_TOOL], cerebellum_log, [], midbrain_log)

    assert record['tools'] == {'cerebellum': [_TOOL], 'midbrain': []}
    assert record['system'] == {'cerebellum': '小脑', 'midbrain': '中脑'}
    assert [(turn['turn'], turn['user']) for turn in record['turns']] == [(1, '写一个贪吃蛇'), (2, '都允许')]
    first = record['turns'][0]
    assert first['at'] == stamp(_AT + 1)
    assert [(message['layer'], message['role']) for message in first['messages']] == [
        ('cerebellum', 'assistant'),
        ('cerebellum', 'tool'),
        ('midbrain', 'user'),
        ('midbrain', 'assistant'),
        ('cerebellum', 'user'),
    ]
    assert first['messages'][0]['tool_calls'][0]['function']['arguments'] == {'text': '写一个贪吃蛇'}
    assert record['turns'][1]['messages'] == []


def test_cerebellum_log_exports_in_reference_shape() -> None:
    log = [
        {'role': 'system', 'content': '你是谁', '_at': _AT},
        {'role': 'user', 'content': '做一个俄罗斯方块网页', '_at': _AT + 1},
        {
            'role': 'assistant',
            'content': None,
            'reasoning_content': '想一想',
            'provider_specific_fields': {'x': 1},
            'tool_calls': [
                {
                    'id': 'call_1',
                    'type': 'function',
                    'function': {'name': 'delegate', 'arguments': '{"text": "做一个俄罗斯方块网页"}'},
                }
            ],
            '_at': _AT + 2,
        },
        {
            'role': 'tool',
            'tool_call_id': 'call_1',
            'name': 'delegate',
            'content': '{"status": "submitted"}',
            '_at': _AT + 3,
        },
        {'role': 'assistant', 'content': '我这就去做。', '_at': _AT + 4},
        {'role': 'user', 'content': '[通知]\nstatus=ok\n已停掉', '_injected': True, '_at': _AT + 5},
    ]

    assert normalize_messages(log) == [
        {'role': 'system', 'content': '你是谁', 'at': '2026-10-09T11:31:13.459740+08:00'},
        {'role': 'user', 'content': '做一个俄罗斯方块网页', 'at': '2026-10-09T11:31:14.459740+08:00'},
        {
            'role': 'assistant',
            'content': '',
            'at': '2026-10-09T11:31:15.459740+08:00',
            'tool_calls': [
                {
                    'id': 'call_1',
                    'type': 'function',
                    'function': {'name': 'delegate', 'arguments': {'text': '做一个俄罗斯方块网页'}},
                }
            ],
        },
        {
            'role': 'tool',
            'content': '{"status": "submitted"}',
            'at': '2026-10-09T11:31:16.459740+08:00',
            'tool_call_id': 'call_1',
            'name': 'delegate',
        },
        {'role': 'assistant', 'content': '我这就去做。', 'at': '2026-10-09T11:31:17.459740+08:00'},
        {'role': 'user', 'content': '[通知]\nstatus=ok\n已停掉', 'at': '2026-10-09T11:31:18.459740+08:00'},
    ], 'Keep role/content/at/tool_calls/tool_call_id/name only, in that order, with arguments parsed'


def test_tool_message_without_name_takes_it_from_the_earlier_call() -> None:
    log = [
        {
            'role': 'assistant',
            'content': '',
            'tool_calls': [{'id': 'r1', 'function': {'name': 'reply', 'arguments': {'status': 'ok', 'say': '好了'}}}],
        },
        {'role': 'tool', 'tool_call_id': 'r1', 'content': '{"ok": true}'},
        {'role': 'tool', 'tool_call_id': 'unknown', 'content': '{}'},
    ]

    normalized = normalize_messages(log)

    assert normalized[1] == {'role': 'tool', 'content': '{"ok": true}', 'tool_call_id': 'r1', 'name': 'reply'}
    assert normalized[2]['name'] == '', 'An unmatched call id leaves the name empty rather than guessing'
    assert 'at' not in normalized[0], 'A message without a timestamp gets no at field'


def test_unparseable_arguments_stay_as_the_raw_string() -> None:
    log = [
        {
            'role': 'assistant',
            'content': '',
            'tool_calls': [
                {'id': 'c1', 'function': {'name': 'web_search', 'arguments': '{"query": 2026年10月9日 今日新闻}'}},
                {'id': 'c2', 'function': {'name': 'delegate', 'arguments': None}},
            ],
        }
    ]

    calls = normalize_messages(log)[0]['tool_calls']

    assert calls[0]['function']['arguments'] == '{"query": 2026年10月9日 今日新闻}'
    assert calls[1]['function']['arguments'] == {}


def test_trajectory_record_is_written_as_one_json_line() -> None:
    record = trajectory_record([_TOOL], [{'role': 'user', 'content': '你好', '_at': _AT, '_snapshot_at': 2}])

    assert record == {'tools': [_TOOL], 'messages': [{'role': 'user', 'content': '你好', 'at': stamp(_AT)}]}
    line = cli.trajectory_line(record)
    assert line.endswith('\n')
    assert line.count('\n') == 1
    assert json.loads(line) == record


def test_example_cli_writes_one_redacted_line_per_layer(tmp_path: Path) -> None:
    cerebellum_record = trajectory_record([_TOOL], [{'role': 'user', 'content': 'key=secret-value', '_at': _AT}])
    midbrain_record = trajectory_record([], [{'role': 'system', 'content': '中脑', '_at': _AT}])
    task_output = {'cerebellum_trajectory': cerebellum_record, 'midbrain_trajectory': midbrain_record}

    written = cli._write_trajectories(
        tmp_path, task_output, sensitive_source={'LIVE_AGENT_POLICY_API_KEY': 'secret-value'}
    )

    assert set(written) == {'cerebellum', 'midbrain'}
    cerebellum_lines = (tmp_path / 'cerebellum.jsonl').read_text(encoding='utf-8').splitlines()
    assert len(cerebellum_lines) == 1
    assert 'secret-value' not in cerebellum_lines[0], 'Credentials echoed in a message are redacted'
    assert json.loads((tmp_path / 'midbrain.jsonl').read_text(encoding='utf-8')) == midbrain_record
    assert stat.S_IMODE((tmp_path / 'midbrain.jsonl').stat().st_mode) == 0o600, 'Use owner-only mode'


def test_example_cli_skips_layers_the_task_output_lacks(tmp_path: Path) -> None:
    assert cli._write_trajectories(tmp_path, {'policy_messages': []}, sensitive_source={}) == {}
    assert cli._write_trajectories(tmp_path, None, sensitive_source={}) == {}
    assert not list(tmp_path.iterdir())


def test_stamp_rejects_missing_and_non_numeric_times() -> None:
    assert stamp(None) == ''
    assert stamp(0) == ''
    assert stamp(True) == ''
    assert stamp(_AT) == '2026-10-09T11:31:13.459740+08:00'
