from __future__ import annotations

import pytest

from nexus.extensions.harbor_bridge.bad_pattern_operator import detect_tool_call_bad_patterns


TOOLS = [
    {
        'type': 'function',
        'function': {
            'name': 'weather',
            'parameters': {
                'type': 'object',
                'properties': {
                    'city': {'type': 'string'},
                    'days': {'anyOf': [{'type': 'integer'}, {'type': 'null'}]},
                    'nested': {'$ref': '#/$defs/nested'},
                },
                'required': ['city'],
                'additionalProperties': False,
                '$defs': {
                    'nested': {
                        'type': 'object',
                        'properties': {'enabled': {'type': 'boolean'}},
                        'required': ['enabled'],
                    }
                },
            },
        },
    }
]


def _call(arguments: object) -> dict:
    return {
        'id': 'call-1',
        'type': 'function',
        'function': {'name': 'weather', 'arguments': arguments},
    }


def test_replaced_tool_call_args_invalid_uses_full_schema_and_one_finding_per_call() -> None:
    result = detect_tool_call_bad_patterns(
        [{
            'role': 'assistant',
            'content': '',
            'tool_calls': [_call({'days': 'bad', 'nested': {'enabled': 'yes'}, 'extra': 1})],
        }],
        TOOLS,
        enabled_rules={'tool_call_args_invalid'},
    )

    assert result['issue_counts'] == {'tool_call_args_invalid': 1}
    hit = result['hits'][0]
    assert hit['detail']['source_detector'] == 'tool_call_args_invalid'
    assert set(hit['detail']['violation_kinds']) >= {
        'missing_required',
        'type_mismatch',
        'unknown_parameter',
    }


def test_tool_call_args_invalid_preserves_unjudgeable_evidence_without_counting_a_hit() -> None:
    tools = [{
        'type': 'function',
        'function': {
            'name': 'weather',
            'parameters': {'$ref': 'https://example.com/weather.schema.json'},
        },
    }]
    result = detect_tool_call_bad_patterns(
        [{'role': 'assistant', 'content': '', 'tool_calls': [_call({'city': 3})]}],
        tools,
        enabled_rules={'tool_call_args_invalid'},
    )

    assert result['is_bad_pattern'] is False
    assert result['issue_counts'] == {}
    assert result['nonissue_evidence_count'] == 1
    assert result['nonissue_evidence'][0]['detail']['judgement'] == 'unjudgeable'


def test_replaced_final_turn_language_is_per_user_window_and_harness_aware() -> None:
    result = detect_tool_call_bad_patterns(
        [
            {
                'role': 'user',
                'content': '<system-reminder>ignore</system-reminder>'
                '<user_query>请详细分析系统失败原因并给出完整修复方案</user_query>',
            },
            {
                'role': 'assistant',
                'content': (
                    'The system failed because the runtime configuration was inconsistent. '
                    'We should restore the original image and resume from the valid checkpoint.'
                ),
            },
        ],
        enabled_rules={'final_turn_language'},
    )

    assert result['issue_counts'] == {'final_turn_language': 1}
    assert result['hits'][0]['detail']['source_detector'] == 'zh_ask_foreign_reply'


@pytest.mark.parametrize(
    ('termination_reason', 'finish_reason'),
    [
        ('TRUNCATION', None),
        ('ENV_DONE', 'length'),
    ],
)
def test_truncated_final_turn_language_is_nonissue_evidence(
    termination_reason: str,
    finish_reason: str | None,
) -> None:
    assistant = {
        'role': 'assistant',
        'content': (
            'The implementation is incomplete because the generated tool call was cut off. '
            'We should not treat this partial turn as the final answer to the user.'
        ),
    }
    if finish_reason is not None:
        assistant['finish_reason'] = finish_reason

    result = detect_tool_call_bad_patterns(
        [
            {'role': 'user', 'content': '请详细分析系统失败原因并给出完整修复方案'},
            assistant,
        ],
        enabled_rules={'final_turn_language'},
        termination_reason=termination_reason,
    )

    assert result['issue_counts'] == {}
    assert result['nonissue_evidence_count'] == 1
    evidence = result['nonissue_evidence'][0]
    assert evidence['problem_type'] == 'final_turn_language'
    assert evidence['detail']['suppressed_reason'] == 'truncated_final_assistant_not_user_facing'


def test_final_turn_language_skips_hy_tool_call_only_final_assistant() -> None:
    result = detect_tool_call_bad_patterns(
        [
            {'role': 'user', 'content': '请详细分析系统失败原因并给出完整修复方案'},
            {
                'role': 'assistant',
                'content': (
                    '<tool_calls:6124c78e><tool_call:6124c78e>Write'
                    '<arg_key:6124c78e>file_path</arg_key:6124c78e>'
                    '<arg_value:6124c78e>/tmp/example.py</arg_value:6124c78e>'
                    '<arg_key:6124c78e>content</arg_key:6124c78e>'
                    '<arg_value:6124c78e>'
                    'The implementation is complete and all tests should pass successfully.'
                    '</arg_value:6124c78e></tool_call:6124c78e></tool_calls:6124c78e>'
                ),
            },
        ],
        enabled_rules={'final_turn_language'},
        termination_reason='ENV_DONE',
    )

    assert result['issue_counts'] == {}
    assert result['nonissue_evidence_count'] == 1
    evidence = result['nonissue_evidence'][0]
    assert evidence['problem_type'] == 'final_turn_language'
    assert evidence['detail']['suppressed_reason'] == 'tool_call_only_final_assistant'


def test_truncation_only_suppresses_final_language_on_global_last_assistant() -> None:
    first_reply = (
        'The first answer is complete and remains a real language mismatch. '
        'It must still be reported even if a later user turn is truncated.'
    )
    final_reply = (
        'The second answer is incomplete because generation stopped at the response limit. '
        'It is not a completed user-facing final answer.'
    )
    result = detect_tool_call_bad_patterns(
        [
            {'role': 'user', 'content': '请先分析第一个完整问题并说明最终结论'},
            {'role': 'assistant', 'content': first_reply},
            {'role': 'user', 'content': '请继续分析第二个问题并给出完整最终答案'},
            {'role': 'assistant', 'content': final_reply},
        ],
        enabled_rules={'final_turn_language'},
        termination_reason='OUTPUT_LENGTH_EXCEEDED',
    )

    assert result['issue_counts'] == {'final_turn_language': 1}
    assert result['hits'][0]['message_index'] == 1
    assert result['nonissue_evidence_count'] == 1
    evidence = result['nonissue_evidence'][0]
    assert evidence['message_index'] == 3
    assert evidence['detail']['suppressed_reason'] == 'truncated_final_assistant_not_user_facing'


@pytest.mark.parametrize(
    ('chinese_units', 'latin_units', 'expected_penalty'),
    [
        (90, 10, 0.0),
        (89, 11, 0.1),
        (80, 20, 0.1),
        (79, 21, 0.2),
        (70, 30, 0.2),
        (69, 31, 0.3),
        (60, 40, 0.3),
        (59, 41, 0.4),
        (50, 50, 0.4),
        (49, 51, 0.5),
    ],
)
def test_final_turn_language_ch_uses_ten_percent_penalty_steps(
    chinese_units: int,
    latin_units: int,
    expected_penalty: float,
) -> None:
    content = '中' * chinese_units + ' ' + 'word ' * latin_units

    result = detect_tool_call_bad_patterns(
        [{'role': 'assistant', 'content': content}],
        enabled_rules={'final_turn_language_ch'},
    )

    evidence = (result['hits'] or result['nonissue_evidence'])[0]
    assert evidence['detail']['chinese_ratio'] == pytest.approx(chinese_units / 100)
    assert evidence['detail']['suggested_penalty'] == expected_penalty
    assert result['issue_counts'] == (
        {'final_turn_language_ch': 1} if expected_penalty > 0 else {}
    )


def test_final_turn_language_ch_excludes_code_and_uses_only_final_assistant_content() -> None:
    result = detect_tool_call_bad_patterns(
        [
            {'role': 'assistant', 'content': 'word ' * 100},
            {'role': 'tool', 'content': 'word ' * 100},
            {
                'role': 'assistant',
                'content': (
                    '中' * 9
                    + ' word `inline english words` ```text\nforeign words here\n```'
                    + '<tool_calls:token><tool_call:token>word word word</tool_call:token></tool_calls:token>'
                ),
                'tool_calls': [_call({'payload': 'word ' * 100})],
            },
        ],
        enabled_rules={'final_turn_language_ch'},
    )

    assert result['issue_counts'] == {}
    evidence = result['nonissue_evidence'][0]
    assert evidence['message_index'] == 2
    assert evidence['detail']['chinese_ratio'] == pytest.approx(0.9)
    assert evidence['detail']['script_units'] == {'cjk': 9, 'latin': 1}


def test_final_turn_language_ch_does_not_penalize_truncated_final_content() -> None:
    result = detect_tool_call_bad_patterns(
        [{'role': 'assistant', 'content': 'word ' * 100}],
        enabled_rules={'final_turn_language_ch'},
        termination_reason='TRUNCATION',
    )

    assert result['issue_counts'] == {}
    evidence = result['nonissue_evidence'][0]
    assert evidence['detail']['suggested_penalty'] == 0.5
    assert evidence['detail']['suppressed_reason'] == 'truncated_final_assistant_not_user_facing'
    assert result['suggested_penalties'] == {}


@pytest.mark.parametrize('termination_reason', ['MAX_STEPS', 'TASK_TIMEOUT', 'FORMAT_ERROR'])
def test_final_turn_language_ch_does_not_penalize_abnormal_termination(
    termination_reason: str,
) -> None:
    result = detect_tool_call_bad_patterns(
        [{'role': 'assistant', 'content': 'word ' * 100}],
        enabled_rules={'final_turn_language_ch'},
        termination_reason=termination_reason,
    )

    assert result['issue_counts'] == {}
    evidence = result['nonissue_evidence'][0]
    assert evidence['detail']['suggested_penalty'] == 0.5
    assert evidence['detail']['suppressed'] is True
    assert evidence['detail']['suppressed_reason'] == 'abnormal_termination_no_user_facing_final'
    assert evidence['detail']['termination_reason'] == termination_reason
    assert result['suggested_penalties'] == {}


def test_reasoning_selftalk_flood_uses_density_rule() -> None:
    result = detect_tool_call_bad_patterns(
        [{
            'role': 'assistant',
            'reasoning_content': 'Let me\nHere\nNow\nOK\n' * 9,
            'content': '',
        }],
        enabled_rules={'reasoning_selftalk_flood'},
    )

    assert result['issue_counts'] == {'reasoning_selftalk_flood': 1}


def _selftalk(reasoning: str) -> dict:
    return detect_tool_call_bad_patterns(
        [{'role': 'assistant', 'reasoning_content': reasoning, 'content': ''}],
        enabled_rules={'reasoning_selftalk_flood'},
    )['issue_counts']


@pytest.mark.parametrize(
    'reasoning',
    [
        # Two "let me" beside ASCII glyph rows and a numbered list: plenty of
        # newlines, but the window does not read as self-talk.
        'Let me look at IDX10 pixel-exact.\nActually, let me check:\n' + '..\n' * 9 + '##\n##\n#.\n#.\n' + '..\n' * 8,
        'Let me compute the statistics precisely.\n' + ''.join(f'{3 + i / 10:.3f}\n' for i in range(30)) + 'Let me',
        'Let me index query:\n' + ''.join(f'{i} {c}\n' for i, c in enumerate('QICKAKAQTDREDLRTLLRYYNQSE', 1)),
    ],
)
def test_reasoning_selftalk_flood_ignores_keywords_beside_lists(reasoning: str) -> None:
    assert _selftalk(reasoning) == {}


def test_reasoning_selftalk_flood_catches_one_newline_per_line_self_talk() -> None:
    assert _selftalk('Let me search.\n' * 14) == {'reasoning_selftalk_flood': 1}


def test_reasoning_selftalk_flood_catches_repeated_chinese_announcements() -> None:
    reasoning = (
        '现在建任务列表并开始写。\n让我创建任务并写文件。\n我先建几个任务。\n实际上我直接写文件。让我建任务。\n'
        '由于分析已完成，直接写。\n让我先建任务。\n然后写 JSON。\nlet我一次性写好。\n先建任务。\n好。\n'
        '让我开始。\n执行。\n让我先建任务。\n实际上可以直接写。\n让我写。\n'
    )

    assert _selftalk(reasoning) == {'reasoning_selftalk_flood': 1}


def test_reasoning_selftalk_flood_ignores_plan_list_below_line_share() -> None:
    # Count and keyword floors are met (4 让我 + 16 newlines), but only 4 of 16
    # lines are self-talk.
    reasoning = (
        '让我先并行做几件事：\n1. 获取当前时间\n2. 搜索邮件\n3. 查看云盘\n让我查航班。\n4. 搜索会议室\n'
        '5. 获取乘机人\n6. 查询航班\n让我查酒店。\n7. 查询酒店\n8. 预订\n9. 通知用户\n让我开始。\n'
        '10. 收尾\n11. 复核\n12. 汇总\n'
    )

    assert _selftalk(reasoning) == {}


def test_hy_and_all_family_protocol_rules_are_disjoint() -> None:
    result = detect_tool_call_bad_patterns(
        [{
            'role': 'assistant',
            'content': (
                '<tool_call:6124c78e>x</tool_call:6124c78e>'
                '<think:6124c78e>x</think:6124c78e>'
                '<|tool_call_begin|>x<|tool_call_end|>'
                '<seed:think>x</seed:think>'
            ),
        }],
        enabled_rules={
            'tool_call_protocol_leak_hy',
            'tool_call_protocol_leak_all_family',
            'think_protocol_leak_hy',
            'think_protocol_leak_all_family',
        },
    )

    assert result['issue_counts'] == {
        'think_protocol_leak_all_family': 1,
        'think_protocol_leak_hy': 1,
        'tool_call_protocol_leak_all_family': 1,
        'tool_call_protocol_leak_hy': 1,
    }


def test_all_family_tool_protocol_regex_does_not_cross_lines() -> None:
    result = detect_tool_call_bad_patterns(
        [{
            'role': 'assistant',
            'content': 'the <parameters of this call are\nlong -> and ordinary prose',
        }],
        enabled_rules={'tool_call_protocol_leak_all_family'},
    )

    assert result['issue_counts'] == {}
    assert result['nonissue_evidence_count'] == 0


def test_all_family_tool_protocol_keeps_valid_single_line_forms() -> None:
    result = detect_tool_call_bad_patterns(
        [{
            'role': 'assistant',
            'content': '<tool_call><invoke name="Bash"><|tool_call_begin|>',
        }],
        enabled_rules={'tool_call_protocol_leak_all_family'},
    )

    assert result['issue_counts'] == {'tool_call_protocol_leak_all_family': 1}


def test_all_family_tool_protocol_parameters_tag_is_case_sensitive() -> None:
    cxx_result = detect_tool_call_bad_patterns(
        [{
            'role': 'assistant',
            'content': 'std::vector<Parameters> value; py::class_<Parameters>(module, "P");',
        }],
        enabled_rules={'tool_call_protocol_leak_all_family'},
    )
    protocol_result = detect_tool_call_bad_patterns(
        [{'role': 'assistant', 'content': '<parameters>'}],
        enabled_rules={'tool_call_protocol_leak_all_family'},
    )

    assert cxx_result['issue_counts'] == {}
    assert protocol_result['issue_counts'] == {'tool_call_protocol_leak_all_family': 1}


def test_all_family_tool_protocol_treats_code_literals_as_nonissue_evidence() -> None:
    result = detect_tool_call_bad_patterns(
        [{
            'role': 'assistant',
            'reasoning_content': 'The prior call used `<arg_key>page_size</arg_key>`.',
            'content': '',
        }],
        enabled_rules={'tool_call_protocol_leak_all_family'},
    )

    assert result['issue_counts'] == {}
    assert result['nonissue_evidence_count'] == 1
    ignored = result['nonissue_evidence'][0]['detail']['ignored_matches']
    assert {item['reason'] for item in ignored} == {'inline_code'}


def test_all_family_tool_protocol_ignores_nonterminal_file_content_tags() -> None:
    result = detect_tool_call_bad_patterns(
        [{
            'role': 'assistant',
            'content': '',
            'tool_calls': [{
                'id': 'call-1',
                'type': 'function',
                'function': {
                    'name': 'Write',
                    'arguments': {
                        'file_path': '/tmp/example.cpp',
                        'content': "std::vector<parameters> value; grep '</parameter>' file",
                    },
                },
            }],
        }],
        enabled_rules={'tool_call_protocol_leak_all_family'},
    )

    assert result['issue_counts'] == {}
    assert result['nonissue_evidence_count'] == 1
    ignored = result['nonissue_evidence'][0]['detail']['ignored_matches']
    assert {item['reason'] for item in ignored} == {'file_content_non_terminal_protocol_text'}


def test_all_family_tool_protocol_ignores_nonterminal_nested_edit_content_tags() -> None:
    result = detect_tool_call_bad_patterns(
        [{
            'role': 'assistant',
            'content': '',
            'tool_calls': [{
                'id': 'call-1',
                'type': 'function',
                'function': {
                    'name': 'filesystem_edit_file',
                    'arguments': {
                        'path': '/tmp/example.py',
                        'edits': [{
                            'oldText': 'old value',
                            'newText': "marker = '</tool_call>'\nprint(marker)",
                        }],
                    },
                },
            }],
        }],
        enabled_rules={'tool_call_protocol_leak_all_family'},
    )

    assert result['issue_counts'] == {}
    assert result['nonissue_evidence_count'] == 1
    ignored = result['nonissue_evidence'][0]['detail']['ignored_matches']
    assert ignored[0]['reason'] == 'file_content_non_terminal_protocol_text'
    assert ignored[0]['argument_name'] == 'edits[0].newText'


def test_all_family_tool_protocol_ignores_shell_quoted_tag_literal() -> None:
    result = detect_tool_call_bad_patterns(
        [{
            'role': 'assistant',
            'content': '',
            'tool_calls': [{
                'id': 'call-1',
                'type': 'function',
                'function': {
                    'name': 'Bash',
                    'arguments': {'command': "grep '</parameter>' generated.log"},
                },
            }],
        }],
        enabled_rules={'tool_call_protocol_leak_all_family'},
    )

    assert result['issue_counts'] == {}
    assert result['nonissue_evidence_count'] == 1
    ignored = result['nonissue_evidence'][0]['detail']['ignored_matches']
    assert {item['reason'] for item in ignored} == {'shell_quoted_literal'}


def test_all_family_tool_protocol_keeps_trailing_file_content_leak() -> None:
    result = detect_tool_call_bad_patterns(
        [{
            'role': 'assistant',
            'content': '',
            'tool_calls': [{
                'id': 'call-1',
                'type': 'function',
                'function': {
                    'name': 'Write',
                    'arguments': {
                        'file_path': '/tmp/index.html',
                        'content': '<html></html>\n</content>\n</invoke>',
                    },
                },
            }],
        }],
        enabled_rules={'tool_call_protocol_leak_all_family'},
    )

    assert result['issue_counts'] == {'tool_call_protocol_leak_all_family': 1}
    assert result['hits'][0]['detail']['fields_hit'] == ['tool_call']


def test_all_family_tool_protocol_keeps_trailing_nested_edit_content_leak() -> None:
    result = detect_tool_call_bad_patterns(
        [{
            'role': 'assistant',
            'content': '',
            'tool_calls': [{
                'id': 'call-1',
                'type': 'function',
                'function': {
                    'name': 'filesystem_edit_file',
                    'arguments': {
                        'path': '/tmp/example.py',
                        'edits': [{
                            'oldText': 'old value',
                            'newText': 'replacement\n</tool_call>',
                        }],
                    },
                },
            }],
        }],
        enabled_rules={'tool_call_protocol_leak_all_family'},
    )

    assert result['issue_counts'] == {'tool_call_protocol_leak_all_family': 1}
    assert result['hits'][0]['detail']['fields_hit'] == ['tool_call']


def test_all_family_tool_protocol_keeps_plain_leak_alongside_code_reference() -> None:
    result = detect_tool_call_bad_patterns(
        [{
            'role': 'assistant',
            'content': 'Example: `<arg_key>x</arg_key>`. Actual leaked close: </invoke>',
        }],
        enabled_rules={'tool_call_protocol_leak_all_family'},
    )

    assert result['issue_counts'] == {'tool_call_protocol_leak_all_family': 1}
    assert result['hits'][0]['detail']['ignored_matches']


def test_truncated_incomplete_final_protocol_fragments_are_nonissue_evidence() -> None:
    cases = {
        'tool_call_protocol_leak_hy': (
            '<tool_calls:6124c78e><tool_call:6124c78e>Bash'
            '<arg_key:6124c78e>command</arg_key:6124c78e>'
        ),
        'tool_call_protocol_leak_all_family': '<tool_calls><tool_call>Bash<arg_key>command</arg_key>',
        'think_protocol_leak_hy': '<think:6124c78e>unfinished reasoning',
        'think_protocol_leak_all_family': '<think>unfinished reasoning',
    }

    for rule_id, content in cases.items():
        result = detect_tool_call_bad_patterns(
            [{'role': 'assistant', 'content': content}],
            enabled_rules={rule_id},
            termination_reason='OUTPUT_LENGTH_EXCEEDED',
        )

        assert result['issue_counts'] == {}
        assert result['nonissue_evidence_count'] == 1
        evidence = result['nonissue_evidence'][0]
        assert evidence['problem_type'] == rule_id
        assert evidence['detail']['suppressed_reason'] == 'truncated_final_assistant_protocol_monitor_only'


def test_incomplete_final_protocol_fragment_still_hits_without_truncation() -> None:
    result = detect_tool_call_bad_patterns(
        [{
            'role': 'assistant',
            'content': '<tool_calls:6124c78e><tool_call:6124c78e>Bash',
        }],
        enabled_rules={'tool_call_protocol_leak_hy'},
        termination_reason='ENV_DONE',
    )

    assert result['issue_counts'] == {'tool_call_protocol_leak_hy': 1}


@pytest.mark.parametrize(
    ('rule_id', 'content'),
    [
        (
            'tool_call_protocol_leak_hy',
            (
                '<tool_calls:6124c78e><tool_call:6124c78e>Bash'
                '</tool_call:6124c78e></tool_calls:6124c78e>'
            ),
        ),
        ('tool_call_protocol_leak_all_family', '<tool_calls><tool_call>Bash</tool_call></tool_calls>'),
        ('think_protocol_leak_hy', '<think:6124c78e>done</think:6124c78e>'),
        ('think_protocol_leak_all_family', '<think>done</think>'),
    ],
)
def test_complete_final_protocol_leak_is_monitor_only_on_truncation(
    rule_id: str,
    content: str,
) -> None:
    result = detect_tool_call_bad_patterns(
        [{'role': 'assistant', 'content': content}],
        enabled_rules={rule_id},
        termination_reason='TRUNCATION',
    )

    assert result['issue_counts'] == {}
    evidence = result['nonissue_evidence'][0]
    assert evidence['problem_type'] == rule_id
    assert evidence['detail']['suppressed_reason'] == 'truncated_final_assistant_protocol_monitor_only'


def test_truncated_final_protocol_in_tool_call_argument_is_monitor_only() -> None:
    result = detect_tool_call_bad_patterns(
        [{
            'role': 'assistant',
            'content': '',
            'tool_calls': [{
                'id': 'call-1',
                'type': 'function',
                'function': {
                    'name': 'Bash',
                    'arguments': {
                        'command': '<tool_calls:6124c78e><tool_call:6124c78e>Bash',
                    },
                },
            }],
        }],
        enabled_rules={'tool_call_protocol_leak_hy'},
        termination_reason='TRUNCATION',
    )

    assert result['issue_counts'] == {}
    evidence = result['nonissue_evidence'][0]
    assert evidence['detail']['fields_hit'] == ['tool_call']
    assert evidence['detail']['suppressed_reason'] == 'truncated_final_assistant_protocol_monitor_only'


def test_truncation_only_suppresses_the_final_assistant_turn() -> None:
    result = detect_tool_call_bad_patterns(
        [
            {
                'role': 'assistant',
                'content': '<tool_calls:6124c78e><tool_call:6124c78e>Bash',
            },
            {'role': 'assistant', 'content': 'final truncated text'},
        ],
        enabled_rules={'tool_call_protocol_leak_hy'},
        termination_reason='TRUNCATION',
    )

    assert result['issue_counts'] == {'tool_call_protocol_leak_hy': 1}


def test_policy_can_keep_incomplete_final_protocol_fragment_as_issue() -> None:
    result = detect_tool_call_bad_patterns(
        [{
            'role': 'assistant',
            'content': '<tool_calls:6124c78e><tool_call:6124c78e>Bash',
        }],
        enabled_rules={'tool_call_protocol_leak_hy'},
        ignore_incomplete_final_assistant=False,
        termination_reason='TRUNCATION',
    )

    assert result['issue_counts'] == {'tool_call_protocol_leak_hy': 1}
