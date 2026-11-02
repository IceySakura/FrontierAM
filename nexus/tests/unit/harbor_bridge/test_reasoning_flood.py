from __future__ import annotations

import pytest

from nexus.extensions.harbor_bridge.bad_pattern_operator import (
    RULE_TITLES,
    ToolCallBadPatternPolicy,
    apply_tool_call_bad_pattern_penalty,
    detect_tool_call_bad_patterns,
)
from nexus.extensions.harbor_bridge.bad_pattern_operator.context import DetectionContext
from nexus.extensions.harbor_bridge.bad_pattern_operator.rules._checkpoint_reasoning_flood import (
    scan as scan_checkpoint,
)
from nexus.extensions.harbor_bridge.bad_pattern_operator.rules._reasoning_flood import detect_thinking


FLOOD = ''.join(
    f'Hmm.\n\nWhat about "The {name}..."? no.\n\nOK.\n\nLet me search.\n\n'
    for name in ['King', 'Queen', 'Prince', 'Duke', 'Earl', 'Lord', 'Knight', 'Squire']
)
CHECKLIST = 'Final checks.\n\n' + ''.join(f'### Another subtle: `path{i}` normalized.\nyes.\n\n' for i in range(60))
MENTIONS = 'Drafting answer.\n' + ''.join(
    f'Need maybe maybe mention "item {i} is config only". Good.\n' for i in range(60)
)
NORMAL = ('Let me look at the failing test first.\nThe assertion compares two dicts.\nI will fix the key order.\n') * 3


@pytest.mark.parametrize(
    ('thinking', 'kind'),
    [
        (FLOOD, 'filler_flood'),
        (CHECKLIST, 'template_loop'),
        (MENTIONS, 'template_loop'),
        (''.join(f'No.{i} item variant A{i}\n' for i in range(1, 61)), None),
        (''.join(f'{i}. FT: "topic {i}" NO: "outcome {i}"\n' for i in range(1, 61)), None),
        (''.join(f'Line {i}: "x = {i}" added\n' for i in range(39, 150)), None),
        ('Here is the script:\n```ruby\n' + 'end\n' * 100 + '```\nDone.\n', None),
        ('def f\n' + 'end\n' * 100, None),
        (NORMAL, None),
        ('```text\n' + FLOOD + MENTIONS + '```\n', None),
        ('Name: Alice Status: ready\n' * 60, None),
        ('x_value = something;\n' * 60, None),
        ('at module.run(file.py:39)\n' * 60, None),
        ('', None),
        (None, None),
    ],
)
def test_source_judgments(thinking, kind) -> None:
    result = detect_thinking(thinking)
    assert (result['kind'] if result else None) == kind
    if result:
        assert result['excerpt'] in thinking


@pytest.mark.parametrize('field', ['thinking', 'reasoning_content', 'reasoning'])
def test_registered_rule_preserves_evidence_and_turn_indices(field: str) -> None:
    result = detect_tool_call_bad_patterns(
        [
            {'role': 'user', field: FLOOD},
            {'role': 'assistant', field: FLOOD},
            {'role': 'tool', field: FLOOD},
            {'role': 'assistant', field: MENTIONS},
        ],
        enabled_rules={'reasoning_selftalk_flood'},
    )
    assert 'reasoning_selftalk_flood' in RULE_TITLES
    assert 'reasoning_flood' not in RULE_TITLES
    assert result['issue_counts'] == {'reasoning_selftalk_flood': 2}
    assert result['bad_turns'] == [1, 3]
    for hit, index, kind in zip(result['hits'], [1, 3], ['filler_flood', 'template_loop']):
        assert hit['message_index'] == index
        assert hit['assistant_turn'] == (index - 1) // 2
        checkpoint = hit['detail']['checkpoint_reasoning_flood']
        assert checkpoint['kind'] == kind
        assert checkpoint['source_rule_version'] == 1
        assert hit['excerpt'] in (FLOOD if index == 1 else MENTIONS)
        assert any(
            quote.get('snippet') in (FLOOD if index == 1 else MENTIONS)
            for quote in hit['quotes']
            if quote.get('snippet')
        )


@pytest.mark.parametrize(
    ('fields', 'expected_field'),
    [
        ({'thinking': NORMAL, 'reasoning_content': FLOOD}, None),
        ({'thinking': FLOOD, 'reasoning_content': NORMAL}, 'thinking'),
        ({'thinking': ' ', 'reasoning_content': FLOOD, 'reasoning': NORMAL}, 'reasoning_content'),
        ({'thinking': None, 'reasoning_content': 123, 'reasoning': FLOOD}, 'reasoning'),
    ],
)
def test_source_field_precedence(fields: dict, expected_field: str | None) -> None:
    context = DetectionContext(
        [{'role': 'assistant', **fields}], [], reasoning_effort=None, ignore_incomplete_final_assistant=True
    )
    findings = scan_checkpoint(context)
    assert len(findings) == int(expected_field is not None)
    if expected_field:
        assert findings[0].path == [expected_field]


def test_does_not_scan_content_or_arguments_or_combine_turns() -> None:
    result = detect_tool_call_bad_patterns(
        [
            {'role': 'assistant', 'reasoning_content': 'Repeated check.\n' * 25, 'content': MENTIONS},
            {
                'role': 'assistant',
                'reasoning_content': 'Repeated check.\n' * 25,
                'tool_calls': [
                    {
                        'id': 'call-1',
                        'type': 'function',
                        'function': {'name': 'Write', 'arguments': {'content': MENTIONS}},
                    }
                ],
            },
        ],
        enabled_rules={'reasoning_selftalk_flood'},
    )
    assert result['issue_counts'] == {}


@pytest.mark.parametrize(('count', 'total', 'hit'), [(49, 49, False), (50, 625, True), (50, 626, False)])
def test_template_thresholds(count: int, total: int, hit: bool) -> None:
    thinking = 'Repeated check.\n' * count + ''.join(f'Unique observation {"x" * i}.\n' for i in range(total - count))
    assert (detect_thinking(thinking) is not None) is hit


@pytest.mark.parametrize(
    ('thinking', 'hit'),
    [
        ('Hmm.\n\n' * 7 + '\n' * 30, False),
        ('Hmm.\n\n' * 8 + '\n' * 3, False),
        ('Hmm.\n\n' * 8 + '\n' * 4, True),
        (('Hmm.\n' + 'x' * 200 + '\n') * 30, False),
    ],
)
def test_filler_thresholds(thinking: str, hit: bool) -> None:
    assert (detect_thinking(thinking) is not None) is hit


def test_one_finding_per_turn_with_both_kinds_and_truncated_output() -> None:
    result = detect_tool_call_bad_patterns(
        [{'role': 'assistant', 'reasoning_content': FLOOD + MENTIONS}] * 2,
        enabled_rules={'reasoning_selftalk_flood'},
        max_hits=1,
        termination_reason='TRUNCATION',
    )
    assert result['issue_counts'] == {'reasoning_selftalk_flood': 2}
    assert result['hits_truncated'] is True
    hit = result['hits'][0]
    checkpoint = hit['detail']['checkpoint_reasoning_flood']
    assert checkpoint['kind'] == 'filler_flood'
    assert checkpoint['filler_window']
    assert checkpoint['template']['count'] == 60
    assert set(hit['detail']['checks']) == {'selftalk_density', 'filler_flood', 'template_loop'}


def test_policy_penalty_is_not_stacked_with_overlapping_rules() -> None:
    policy = ToolCallBadPatternPolicy.model_validate(
        {
            'id': 'test-flood',
            'penalty': 0.5,
            'reward_threshold': 0.5,
            'patterns': {
                'reasoning_selftalk_flood': {},
            },
        }
    )
    result = detect_tool_call_bad_patterns(
        [{'role': 'assistant', 'reasoning_content': FLOOD}],
        **policy.detection_kwargs(),
    )
    assert result['issue_counts'] == {'reasoning_selftalk_flood': 1}
    score, metrics = apply_tool_call_bad_pattern_penalty(1.0, result, policy)
    assert score == 0.5
    assert result['penalty'] == 0.5
    assert metrics['tool_call_bad_pattern/reasoning_selftalk_flood/hit'] == 1.0
    score, _ = apply_tool_call_bad_pattern_penalty(0.4, result, policy)
    assert score == 0.4


def test_monitor_only_and_disabled_rule() -> None:
    policy = ToolCallBadPatternPolicy.model_validate(
        {
            'id': 'monitor-flood',
            'penalty': 0.3,
            'patterns': {'reasoning_selftalk_flood': {'penalize': False}},
        }
    )
    result = detect_tool_call_bad_patterns(
        [{'role': 'assistant', 'reasoning_content': FLOOD}],
        **policy.detection_kwargs(),
    )
    assert apply_tool_call_bad_pattern_penalty(1.0, result, policy)[0] == 1.0
    disabled = detect_tool_call_bad_patterns([{'role': 'assistant', 'reasoning_content': FLOOD}], enabled_rules=set())
    assert disabled['issue_counts'] == {}


@pytest.mark.parametrize(
    ('thinking', 'penalty'),
    [(MENTIONS, 0.5), (MENTIONS, 0.2), (MENTIONS, 1.0), ('Let me search.\n' * 14, 0.8)],
)
def test_merged_hits_use_configured_penalty(thinking: str, penalty: float) -> None:
    policy = ToolCallBadPatternPolicy.model_validate(
        {'id': 'merged', 'penalty': penalty, 'patterns': {'reasoning_selftalk_flood': {}}}
    )
    result = detect_tool_call_bad_patterns(
        [{'role': 'assistant', 'reasoning_content': thinking}], **policy.detection_kwargs()
    )
    assert result['suggested_penalties'] == {}
    score, _ = apply_tool_call_bad_pattern_penalty(1.0, result, policy)
    assert score == pytest.approx(1.0 - penalty)


def test_checkpoint_only_hit_uses_rule_override() -> None:
    policy = ToolCallBadPatternPolicy.model_validate(
        {
            'id': 'heavy-flood',
            'penalty': 0.1,
            'reward_threshold': 0.7,
            'patterns': {'reasoning_selftalk_flood': {'penalty': 1.0, 'reward_threshold': 0.1}},
        }
    )
    result = detect_tool_call_bad_patterns(
        [{'role': 'assistant', 'reasoning_content': MENTIONS}], **policy.detection_kwargs()
    )
    assert result['hits'][0]['detail']['checks'] == ['template_loop']
    assert apply_tool_call_bad_pattern_penalty(0.5, result, policy)[0] == 0.0
    assert apply_tool_call_bad_pattern_penalty(0.05, result, policy)[0] == 0.05


@pytest.mark.parametrize('max_hits', [0, 1, 100])
def test_old_hit_in_another_channel_or_turn_keeps_configured_penalty(max_hits: int) -> None:
    policy = ToolCallBadPatternPolicy.model_validate(
        {'id': 'merged', 'penalty': 0.8, 'patterns': {'reasoning_selftalk_flood': {}}}
    )
    result = detect_tool_call_bad_patterns(
        [
            {'role': 'assistant', 'reasoning_content': MENTIONS},
            {'role': 'assistant', 'content': 'Let me search.\n' * 14},
        ],
        enabled_rules={'reasoning_selftalk_flood'},
        max_hits=max_hits,
    )
    assert result['issue_counts'] == {'reasoning_selftalk_flood': 2}
    assert result['suggested_penalties'] == {}
    assert apply_tool_call_bad_pattern_penalty(1.0, result, policy)[0] == pytest.approx(0.2)


def test_old_channels_are_retained_and_same_channel_is_coalesced() -> None:
    result = detect_tool_call_bad_patterns(
        [
            {
                'role': 'assistant',
                'reasoning_content': FLOOD,
                'content': 'Let me search.\n' * 14,
                'tool_calls': [
                    {
                        'id': 'call-1',
                        'type': 'function',
                        'function': {'name': 'Write', 'arguments': {'content': 'Let me search.\n' * 14}},
                    }
                ],
            }
        ],
        enabled_rules={'reasoning_selftalk_flood'},
    )
    assert result['issue_counts'] == {'reasoning_selftalk_flood': 3}
    assert {hit['detail']['channel'] for hit in result['hits']} == {'reasoning', 'content', 'tool_call'}
    reasoning = next(hit for hit in result['hits'] if hit['detail']['channel'] == 'reasoning')
    assert reasoning['detail']['checks'] == ['selftalk_density', 'filler_flood']
    assert reasoning['detail']['n_keyword'] >= 4
    assert len(reasoning['quotes']) == 2
