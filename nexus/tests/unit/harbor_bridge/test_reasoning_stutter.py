from pathlib import Path

import pytest
import yaml

from nexus.extensions.harbor_bridge.bad_pattern_operator import (
    ToolCallBadPatternPolicy,
    apply_tool_call_bad_pattern_penalty,
    detect_tool_call_bad_patterns,
)
from nexus.extensions.harbor_bridge.bad_pattern_operator.rules._reasoning_stutter import occurrences


RULE = 'reasoning_word_stutter'
LOOP_RULE = 'reasoning_repetition_loop'


def detect(messages, **kwargs):
    return detect_tool_call_bad_patterns(messages, enabled_rules={RULE}, **kwargs)


@pytest.mark.parametrize('word', ['maybe', 'perhaps', 'need', 'likely'])
def test_each_allowed_word_and_case(word):
    text = f'{word.title()} {word.upper()} check this. {word} {word} try that. {word} {word} finish.'
    result = detect([{'role': 'assistant', 'reasoning_content': text}])
    assert result['issue_counts'] == {RULE: 1}
    evidence = result['hits'][0]['detail']['adjacent_word_stutter']
    assert evidence['trajectory_occurrences'] == 3
    assert evidence['trajectory_word_counts'] == {word: 3}
    assert all(item['repeats'] == 2 for item in evidence['occurrences'])
    assert result['suggested_penalties'] == {}
    assert 'suggested_penalty' not in result['hits'][0]['detail']


@pytest.mark.parametrize('count', [0, 1, 2, 3, 4])
def test_trajectory_location_boundary(count):
    messages = [{'role': 'assistant', 'reasoning_content': 'maybe maybe check states.'} for _ in range(count)]
    result = detect(messages)
    assert bool(result['issue_counts']) == (count >= 3)
    if count >= 3:
        assert result['bad_turns'] == list(range(count))
        assert result['issue_counts'] == {RULE: count}
        assert all(hit['detail']['adjacent_word_stutter']['trajectory_occurrences'] == count for hit in result['hits'])


@pytest.mark.parametrize(
    'text',
    [
        'yes yes yes\nYES yes\nYes YES',
        'OK ok\nok OK ok\nOK OK',
        '420->520 100 yes yes yes\n520->624 104 yes yes yes\n1200->1360 160 yes yes yes',
        'Expected friend lines ok ok; checks ok ok access not_found. yes yes.',
    ],
)
def test_yes_ok_data_and_confirmation_do_not_count(text):
    assert occurrences(text) == []
    assert detect([{'role': 'assistant', 'reasoning_content': text}])['issue_counts'] == {}


def test_yes_ok_cannot_complete_trajectory_threshold():
    text = 'maybe maybe inspect. perhaps perhaps check. yes yes. ok ok.'
    assert len(occurrences(text)) == 2
    assert detect([{'role': 'assistant', 'reasoning_content': text}])['issue_counts'] == {}


def test_yes_ok_does_not_remove_real_stutter_evidence():
    text = 'maybe maybe inspect. yes yes. perhaps perhaps check. ok ok. need need verify.'
    result = detect([{'role': 'assistant', 'reasoning_content': text}])
    evidence = result['hits'][0]['detail']['adjacent_word_stutter']
    assert evidence['trajectory_occurrences'] == 3
    assert evidence['trajectory_word_counts'] == {'maybe': 1, 'perhaps': 1, 'need': 1}


def test_user_examples_three_locations_across_messages():
    result = detect(
        [
            {
                'role': 'assistant',
                'reasoning_content': 'Also maybe maybe run a final check of rule states? '
                'Maybe maybe check the three files content maybe via read?',
            },
            {'role': 'tool', 'content': 'result'},
            {'role': 'assistant', 'reasoning_content': 'Perhaps perhaps one or more existing records need revocation?'},
        ]
    )
    assert result['bad_turns'] == [0, 2]
    evidence = result['hits'][0]['detail']['adjacent_word_stutter']
    assert evidence['trajectory_word_counts'] == {'maybe': 2, 'perhaps': 1}
    assert evidence['trajectory_occurrences'] == 3


def test_maximal_run_is_one_location_not_overlapping_pairs():
    text = 'maybe maybe maybe maybe maybe'
    assert len(occurrences(text)) == 1
    assert occurrences(text)[0]['repeats'] == 5
    assert detect([{'role': 'assistant', 'reasoning_content': text}])['issue_counts'] == {}


@pytest.mark.parametrize(
    'text',
    [
        'maybe check maybe again. perhaps think perhaps next. need inspect need afterwards.',
        'compose compose inspect inspect finish finish.',
        'maybes maybes perhapsX perhapsX needs needs good_good yesyes ok_ok',
        'maybe, maybe. perhaps; perhaps. need. Need.',
        'maybe\nmaybe\nperhaps\nperhaps\nneed\nneed',
        'maybe maybex perhaps perhapsx need needx',
        'maybe-maybe perhaps-perhaps need-need',
        'maybe maybe perhaps perhaps',  # Two locations are not three.
    ],
)
def test_not_adjacent_stutter_locations(text):
    assert detect([{'role': 'assistant', 'reasoning_content': text}])['issue_counts'] == {}


@pytest.mark.parametrize(
    'wrapper',
    [
        '```text\n%s\n```',
        '~~~text\n%s\n~~~',
        '`%s`',
        '``%s``',
        '"%s"',
        '\u201c%s\u201d',
        '> %s',
    ],
)
def test_quoted_examples_and_code_are_excluded(wrapper):
    text = wrapper % 'maybe maybe perhaps perhaps need need'
    assert occurrences(text) == []
    assert detect([{'role': 'assistant', 'reasoning_content': text}])['issue_counts'] == {}


def test_offsets_survive_quoted_and_code_exclusions():
    text = '`maybe maybe` ignored. maybe maybe inspect.\nPerhaps perhaps proceed. Need need verify.'
    result = detect([{'role': 'assistant', 'reasoning_content': text}])
    evidence = result['hits'][0]['detail']['adjacent_word_stutter']
    assert evidence['trajectory_occurrences'] == 3
    for quote in result['hits'][0]['quotes']:
        assert text[quote['start'] : quote['end']].lower() == f'{quote["word"]} {quote["word"]}'
        assert quote['snippet'] in text


def test_excluded_spans_cannot_connect_separate_words():
    text = 'maybe "example" maybe. perhaps `code` perhaps. need ``value`` need.'
    assert occurrences(text) == []
    assert detect([{'role': 'assistant', 'reasoning_content': text}])['issue_counts'] == {}


def test_other_roles_answer_and_arguments_do_not_contribute():
    repeated = 'maybe maybe perhaps perhaps need need'
    result = detect(
        [
            {'role': 'user', 'reasoning_content': repeated},
            {'role': 'assistant', 'content': repeated},
            {'role': 'assistant', 'reasoning_content': 'maybe maybe check.'},
            {
                'role': 'assistant',
                'reasoning_content': 'perhaps perhaps check.',
                'tool_calls': [
                    {
                        'id': 'call-1',
                        'type': 'function',
                        'function': {'name': 'Write', 'arguments': {'content': repeated}},
                    }
                ],
            },
            {'role': 'tool', 'reasoning_content': repeated},
        ]
    )
    assert result['issue_counts'] == {}


def test_message_boundaries_do_not_form_runs():
    messages = [
        {'role': 'assistant', 'reasoning_content': word}
        for word in ['maybe', 'maybe', 'perhaps', 'perhaps', 'need', 'need']
    ]
    assert detect(messages)['issue_counts'] == {}


@pytest.mark.parametrize(
    'fields',
    [
        {'thinking': 'maybe maybe perhaps perhaps need need'},
        {'reasoning': 'maybe maybe perhaps perhaps need need'},
        {'provider_specific_fields': {'reasoning': 'maybe maybe perhaps perhaps need need'}},
        {'content': '<think:6124c78e>maybe maybe perhaps perhaps need need</think:6124c78e>answer'},
    ],
)
def test_provider_reasoning_and_content_thinking_fallback(fields):
    assert detect([{'role': 'assistant', **fields}])['issue_counts'] == {RULE: 1}


def test_stutter_rule_does_not_include_dense_repetition():
    result = detect([{'role': 'assistant', 'reasoning_content': 'A' * 1100}])
    assert result['issue_counts'] == {}


def test_existing_repetition_rule_does_not_include_sparse_stutters():
    result = detect_tool_call_bad_patterns(
        [{'role': 'assistant', 'reasoning_content': 'maybe maybe inspect. perhaps perhaps check. need need verify.'}],
        enabled_rules={LOOP_RULE},
    )
    assert result['issue_counts'] == {}


def test_stutter_and_repetition_are_independent_findings():
    text = 'maybe maybe inspect. perhaps perhaps compare. need need verify.\n' + 'A' * 1100
    result = detect_tool_call_bad_patterns(
        [{'role': 'assistant', 'reasoning_content': text}],
        enabled_rules={RULE, LOOP_RULE},
    )
    assert result['issue_counts'] == {RULE: 1, LOOP_RULE: 1}
    by_rule = {hit['problem_type']: hit for hit in result['hits']}
    assert 'unit_run' in by_rule[LOOP_RULE]['detail']['checks']
    assert 'adjacent_word_stutter' not in by_rule[LOOP_RULE]['detail']['checks']
    assert by_rule[RULE]['detail']['checks'] == ['adjacent_word_stutter']


def test_hit_cap_does_not_change_global_count():
    result = detect([{'role': 'assistant', 'reasoning_content': 'maybe maybe inspect.'}] * 4, max_hits=1)
    assert result['issue_counts'] == {RULE: 4}
    assert result['hits_truncated']
    assert result['hits'][0]['detail']['adjacent_word_stutter']['trajectory_occurrences'] == 4


def test_evidence_is_bounded():
    text = 'maybe maybe check. ' * 100
    result = detect([{'role': 'assistant', 'reasoning_content': text}])
    hit = result['hits'][0]
    evidence = hit['detail']['adjacent_word_stutter']
    assert evidence['trajectory_occurrences'] == 100
    assert len(evidence['occurrences']) == 20
    assert evidence['occurrences_truncated']
    assert len(hit['quotes']) == 3


@pytest.mark.parametrize('penalize', [True, False])
def test_configured_policy_penalty_only(penalize):
    policy = ToolCallBadPatternPolicy.model_validate(
        {
            'id': 'stutter',
            'penalty': 0.5,
            'patterns': {RULE: {'penalize': penalize, 'penalty': 0.3}},
        }
    )
    result = detect_tool_call_bad_patterns(
        [{'role': 'assistant', 'reasoning_content': 'maybe maybe perhaps perhaps need need'}],
        **policy.detection_kwargs(),
    )
    assert result['suggested_penalties'] == {}
    score, _ = apply_tool_call_bad_pattern_penalty(1.0, result, policy)
    assert score == pytest.approx(0.7 if penalize else 1.0)


def test_shipped_yaml_enables_independent_stutter_rule():
    path = Path(__file__).resolve().parents[3] / 'nexus/extensions/harbor_bridge/configs/policies/bad-pattern-v2.yaml'
    raw = yaml.safe_load(path.read_text())['body']['task_config']['tool_call_bad_pattern_policy']
    policy = ToolCallBadPatternPolicy.model_validate(raw)
    assert RULE in policy.enabled_rules()
    assert policy.patterns[RULE].penalize is True
    assert policy.penalty_for(RULE) == 0.3
    assert policy.reward_threshold_for(RULE) == 0.5
    result = detect_tool_call_bad_patterns(
        [{'role': 'assistant', 'reasoning_content': 'maybe maybe inspect. perhaps perhaps check. need need verify.'}],
        **policy.detection_kwargs(),
    )
    assert result['issue_counts'] == {RULE: 1}
    score, metrics = apply_tool_call_bad_pattern_penalty(1.0, result, policy)
    assert score == pytest.approx(0.7)
    assert result['penalty_applied_rule_ids'] == [RULE]
    assert metrics[f'tool_call_bad_pattern/{RULE}/hit'] == 1.0
    assert metrics[f'tool_call_bad_pattern/{LOOP_RULE}/hit'] == 0.0


@pytest.mark.parametrize(('reward', 'expected'), [(1.0, 0.7), (0.49, 0.49)])
def test_overlapping_rules_use_maximum_configured_penalty_without_stacking(reward, expected):
    policy = ToolCallBadPatternPolicy.model_validate(
        {
            'id': 'independent',
            'penalty': 0.5,
            'reward_threshold': 0.5,
            'patterns': {
                RULE: {'penalty': 0.3},
                LOOP_RULE: {'penalty': 0.2},
            },
        }
    )
    text = 'maybe maybe inspect. perhaps perhaps compare. need need verify.\n' + 'A' * 1100
    result = detect_tool_call_bad_patterns(
        [{'role': 'assistant', 'reasoning_content': text}], **policy.detection_kwargs()
    )
    assert result['issue_counts'] == {RULE: 1, LOOP_RULE: 1}
    assert result['suggested_penalties'] == {}
    score, _ = apply_tool_call_bad_pattern_penalty(reward, result, policy)
    assert score == pytest.approx(expected)


def test_disabling_stutter_does_not_disable_repetition():
    policy = ToolCallBadPatternPolicy.model_validate(
        {'id': 'disabled', 'patterns': {RULE: {'enabled': False}, LOOP_RULE: {}}}
    )
    text = 'maybe maybe inspect. perhaps perhaps compare. need need verify.\n' + 'A' * 1100
    result = detect_tool_call_bad_patterns(
        [{'role': 'assistant', 'reasoning_content': text}], **policy.detection_kwargs()
    )
    assert result['issue_counts'] == {LOOP_RULE: 1}
