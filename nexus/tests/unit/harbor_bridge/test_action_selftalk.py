from __future__ import annotations

import pytest

from nexus.extensions.harbor_bridge.bad_pattern_operator import detect_tool_call_bad_patterns
from nexus.extensions.harbor_bridge.bad_pattern_operator.rules._action_selftalk import detect_action_selftalk


LOOP = '好。\n我执行。\n输出。\n写。\n好。\n执行。\n' * 4
CORRUPT_LOOP = (
    r'IInl\n执行。\nl\n好。\Inl\n完成。\nl\n写。\Inl\n输出。lInlln够。'
    r'\nl\n跑。\\nl\n我执行。Inlln好。\nl\n完成。I\nl\n写。I\nl\n输出。'
) * 3


@pytest.mark.parametrize(
    'thinking',
    [
        LOOP,
        CORRUPT_LOOP,
        CORRUPT_LOOP.replace(r'\n', '\n'),
        'OK.\nLet me write.\nI output.\nNow.\nI execute.\n' * 5,
        '好。\n执行。\n（我意识到这是一个无限循环。）\n写。\n输出。\n' * 6,
        '（输出）\n好。\n（执行）\n我写。\n（工具调用）\n' * 6,
    ],
)
def test_action_promises_loop(thinking: str) -> None:
    result = detect_action_selftalk(thinking)
    assert result is not None
    assert result['short_beats'] >= 10
    assert result['action_beats'] >= 5
    assert result['excerpt'] in thinking


@pytest.mark.parametrize(
    'thinking',
    [
        '我先核对报表。完成。\n把三处结果写入文件。好。\n' * 4,
        '好。\n执行。\n写。\n输出。\n',
        '让我核对每一场比赛。\n' + ''.join(f'{i} H\n{i} W\n' for i in range(40)),
        '让我执行。\n' + '[7]\n[]\n' * 40,
        '| 状态 | 结果 |\n' + '| 好 | 完成 |\n' * 40,
        '1. 好。\n2. 执行。\n3. 输出。\n4. 写。\n' * 20,
        '- 好。\n- 执行。\n- 输出。\n- 写。\n' * 20,
        '> 好。\n> 执行。\n> 输出。\n> 写。\n' * 20,
        '```text\n' + LOOP + '```\n',
        '~~~text\n' + LOOP + '~~~\n',
        'Inline null online campaignIds\n' * 20,
        '好。\n完成。\n现在。\n' * 20,
        ''.join(f'任务 {i} 已完成。\n步骤 {i} 执行。\n' for i in range(30)),
        ('先检查输入对象和权限。\n确认返回数据包含所有待处理项。\n比较两个来源的状态和时间戳。\n好。\n执行。\n') * 8,
        '',
    ],
)
def test_normal_reasoning_and_structured_data(thinking: str) -> None:
    assert detect_action_selftalk(thinking) is None


@pytest.mark.parametrize('field', ['thinking', 'reasoning_content', 'reasoning'])
def test_registered_check_is_thinking_only(field: str) -> None:
    result = detect_tool_call_bad_patterns(
        [
            {'role': 'user', field: LOOP},
            {'role': 'assistant', 'content': LOOP},
            {'role': 'tool', field: LOOP},
            {'role': 'assistant', field: LOOP},
        ],
        enabled_rules={'reasoning_selftalk_flood'},
    )
    action_hits = [h for h in result['hits'] if 'action_selftalk_loop' in h['detail']['checks']]
    assert len(action_hits) == 1
    assert action_hits[0]['message_index'] == 3
    assert action_hits[0]['detail']['channel'] == 'reasoning'


def test_turns_are_not_combined_and_thinking_has_precedence() -> None:
    result = detect_tool_call_bad_patterns(
        [
            {'role': 'assistant', 'thinking': 'Normal analysis.', 'reasoning_content': LOOP},
            *[{'role': 'assistant', 'reasoning_content': '好。\n执行。\n写。\n输出。\n'} for _ in range(8)],
        ],
        enabled_rules={'reasoning_selftalk_flood'},
    )
    assert not any('action_selftalk_loop' in h['detail']['checks'] for h in result['hits'])


def test_same_channel_is_coalesced() -> None:
    thinking = 'Let me search.\n' * 14 + LOOP
    result = detect_tool_call_bad_patterns(
        [{'role': 'assistant', 'reasoning_content': thinking}],
        enabled_rules={'reasoning_selftalk_flood'},
    )
    assert result['issue_counts'] == {'reasoning_selftalk_flood': 1}
    assert 'selftalk_density' in result['hits'][0]['detail']['checks']
    assert 'action_selftalk_loop' in result['hits'][0]['detail']['checks']
    assert result['suggested_penalties'] == {}


def test_new_only_check_settles_at_configured_penalty() -> None:
    # Substantive rows spread the beats beyond BPO's 200-character keyword bar.
    thinking = ('好。\n输出。\n写。\n我执行。\n（我已经决定，停止这段独白并输出工具调用。）\n') * 5
    result = detect_tool_call_bad_patterns(
        [{'role': 'assistant', 'reasoning_content': thinking}],
        enabled_rules={'reasoning_selftalk_flood'},
    )
    assert any('action_selftalk_loop' in h['detail']['checks'] for h in result['hits'])
    assert 'selftalk_density' not in result['hits'][0]['detail']['checks']
    assert result['suggested_penalties'] == {}
