from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from pydantic import ValidationError

from nexus.extensions.harbor_bridge.bad_pattern_operator import (
    ToolCallBadPatternPolicy,
    apply_tool_call_bad_pattern_penalty,
    detect_tool_call_bad_patterns,
)
from nexus.extensions.harbor_bridge.bad_pattern_operator.rules.single_turn_thinking_too_long import thinking_text


RULE_ID = 'single_turn_thinking_too_long'


def detect(messages, **kwargs):
    return detect_tool_call_bad_patterns(messages, enabled_rules={RULE_ID}, **kwargs)


@pytest.mark.parametrize(('chars', 'hit'), [(99_999, False), (100_000, False), (100_001, True)])
def test_default_strict_character_boundary(chars, hit):
    result = detect([{'role': 'assistant', 'reasoning_content': 'a' * chars}])
    assert bool(result['issue_counts']) == hit
    assert result['suggested_penalties'] == {}
    if hit:
        finding = result['hits'][0]
        assert finding['detail']['thinking_chars'] == chars
        assert finding['detail']['max_thinking_chars'] == 100_000
        assert 'suggested_penalty' not in finding['detail']
        assert finding['quotes'][0]['head_excerpt'] == 'a' * 300
        assert finding['quotes'][0]['tail_excerpt'] == 'a' * 300


@pytest.mark.parametrize('limit', [50_000, 100_000, 200_000])
def test_policy_override_reaches_detector(limit):
    policy = ToolCallBadPatternPolicy.model_validate(
        {
            'id': 'length',
            'patterns': {RULE_ID: {'max_thinking_chars': limit}},
        }
    )
    result = detect_tool_call_bad_patterns(
        [
            {'role': 'assistant', 'reasoning_content': 'a' * limit},
            {'role': 'assistant', 'reasoning_content': 'a' * (limit + 1)},
        ],
        **policy.detection_kwargs(),
    )
    assert result['issue_counts'] == {RULE_ID: 1}
    assert result['bad_turns'] == [1]
    assert result['hits'][0]['detail']['max_thinking_chars'] == limit


def test_omitted_policy_threshold_uses_default():
    policy = ToolCallBadPatternPolicy.model_validate({'id': 'default', 'patterns': {RULE_ID: {}}})
    result = detect_tool_call_bad_patterns(
        [{'role': 'assistant', 'reasoning_content': 'a' * 100_001}], **policy.detection_kwargs()
    )
    assert result['hits'][0]['detail']['max_thinking_chars'] == 100_000


def test_yaml_policy_threshold_and_monitor_mode():
    path = Path(__file__).resolve().parents[3] / 'nexus/extensions/harbor_bridge/configs/policies/bad-pattern-v2.yaml'
    raw = yaml.safe_load(path.read_text())['body']['task_config']['tool_call_bad_pattern_policy']
    policy = ToolCallBadPatternPolicy.model_validate(raw)
    assert policy.patterns[RULE_ID].max_thinking_chars == 100_000
    assert policy.patterns[RULE_ID].penalize is False
    assert policy.detection_kwargs()['max_thinking_chars'] == 100_000
    raw['patterns'][RULE_ID]['max_thinking_chars'] = 200_000
    policy = ToolCallBadPatternPolicy.model_validate(raw)
    assert policy.detection_kwargs()['max_thinking_chars'] == 200_000


@pytest.mark.parametrize('invalid', [0, -1, True, 1.5, '100000'])
def test_invalid_policy_and_direct_thresholds(invalid):
    with pytest.raises(ValidationError):
        ToolCallBadPatternPolicy.model_validate(
            {
                'id': 'invalid',
                'patterns': {RULE_ID: {'max_thinking_chars': invalid}},
            }
        )
    with pytest.raises(ValueError, match='positive integer'):
        detect([], max_thinking_chars=invalid)


def test_threshold_rejected_for_unrelated_rule():
    with pytest.raises(ValidationError, match='only supported'):
        ToolCallBadPatternPolicy.model_validate(
            {
                'id': 'invalid',
                'patterns': {'repetition_tool_call': {'max_thinking_chars': 100_000}},
            }
        )


def test_disabled_rule_does_not_forward_threshold():
    policy = ToolCallBadPatternPolicy.model_validate(
        {
            'id': 'disabled',
            'patterns': {
                RULE_ID: {'enabled': False, 'max_thinking_chars': 5},
                'repetition_tool_call': {},
            },
        }
    )
    assert 'max_thinking_chars' not in policy.detection_kwargs()
    result = detect_tool_call_bad_patterns(
        [{'role': 'assistant', 'reasoning_content': 'a' * 100_001}], **policy.detection_kwargs()
    )
    assert RULE_ID not in result['issue_counts']


@pytest.mark.parametrize(
    'fields',
    [
        {'reasoning_content': 'abc'},
        {'_reasoning_content': 'abc'},
        {'provider_specific_fields': {'reasoning': 'abc'}},
        {'provider_specific_fields': {'reasoning_content': 'abc'}},
        {'thinking': 'abc'},
        {'reasoning': 'abc'},
    ],
)
def test_reasoning_sources(fields):
    assert thinking_text(fields)[0] == 'abc'


@pytest.mark.parametrize('tag', ['think', 'think:6124c78e', 'think_abcdef'])
def test_content_fallback_and_unclosed_tags(tag):
    assert thinking_text({}, f'  <{tag}>abc</{tag}>answer')[0] == 'abc'
    assert thinking_text({}, f'<{tag}>abc')[0] == 'abc'
    assert thinking_text({}, f'answer <{tag}>abc')[0] == ''
    assert thinking_text({'reasoning_content': 'own'}, f'<{tag}>abc')[0] == 'own'
    assert thinking_text({'reasoning_content': ' '}, f'<{tag}>abc')[0] == 'abc'
    result = detect([{'role': 'assistant', 'content': f'<{tag}>abc</{tag}>answer'}], max_thinking_chars=2)
    assert result['hits'][0]['detail']['thinking_chars'] == 3
    assert result['hits'][0]['path'] == ['content']


def test_no_other_roles_answers_arguments_or_cross_turn_counting():
    result = detect(
        [
            {'role': 'user', 'reasoning_content': 'a' * 100_001},
            {'role': 'assistant', 'reasoning_content': 'a' * 60_000, 'content': 'b' * 100_001},
            {'role': 'assistant', 'reasoning_content': 'a' * 60_000},
            {
                'role': 'assistant',
                'tool_calls': [
                    {
                        'id': 'call-1',
                        'type': 'function',
                        'function': {'name': 'Write', 'arguments': {'content': 'a' * 100_001}},
                    }
                ],
            },
            {'role': 'tool', 'reasoning_content': 'a' * 100_001},
        ]
    )
    assert result['issue_counts'] == {}


def test_unicode_characters_not_utf8_bytes():
    assert detect([{'role': 'assistant', 'reasoning_content': '\u4e2d' * 100_000}])['issue_counts'] == {}
    result = detect([{'role': 'assistant', 'reasoning_content': '\U0001f600' * 100_001}])
    assert result['hits'][0]['detail']['thinking_chars'] == 100_001


@pytest.mark.parametrize('penalize', [True, False])
def test_only_configured_penalty_is_used(penalize):
    policy = ToolCallBadPatternPolicy.model_validate(
        {
            'id': 'penalty',
            'penalty': 0.6,
            'reward_threshold': 0.5,
            'patterns': {RULE_ID: {'max_thinking_chars': 5, 'penalize': penalize, 'penalty': 0.4}},
        }
    )
    result = detect_tool_call_bad_patterns(
        [{'role': 'assistant', 'reasoning_content': 'abcdef', 'finish_reason': 'length'}],
        **policy.detection_kwargs(),
        termination_reason='TRUNCATION',
    )
    assert result['hits'][0]['detail']['finish_reason'] == 'length'
    assert result['suggested_penalties'] == {}
    assert 'suggested_penalty' not in result['hits'][0]['detail']
    score, _ = apply_tool_call_bad_pattern_penalty(1.0, result, policy)
    assert score == pytest.approx(0.6 if penalize else 1.0)


def test_max_hits_does_not_change_issue_counts():
    result = detect([{'role': 'assistant', 'reasoning_content': 'abcdef'}] * 2, max_thinking_chars=5, max_hits=0)
    assert result['issue_counts'] == {RULE_ID: 2}
    assert result['hits'] == []
    assert result['hits_truncated']
