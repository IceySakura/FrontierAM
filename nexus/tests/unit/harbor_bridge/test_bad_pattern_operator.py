from __future__ import annotations

import pytest

from nexus.extensions.harbor_bridge.bad_pattern_operator import (
    RULE_TITLES,
    ToolCallBadPatternPolicy,
    apply_tool_call_bad_pattern_penalty,
    detect_tool_call_bad_patterns,
)
from nexus.extensions.harbor_bridge.bad_pattern_operator.diagnostics import (
    final_turn_language_ch_diagnostic,
)


def _call(arguments: object, *, call_id: str = 'call-1', name: str = 'weather') -> dict:
    return {
        'id': call_id,
        'type': 'function',
        'function': {'name': name, 'arguments': arguments},
    }


def _repeated_tool_messages(
    count: int,
    *,
    name: str = 'query_status',
    content: str = 'running',
    call_id_prefix: str = 'poll',
) -> list[dict]:
    messages = []
    for index in range(count):
        call_id = f'{call_id_prefix}-{index}'
        messages.extend([
            {
                'role': 'assistant',
                'content': '',
                'tool_calls': [_call({'job_id': 'job-1'}, call_id=call_id, name=name)],
            },
            {
                'role': 'tool',
                'tool_call_id': call_id,
                'content': content,
            },
        ])
    return messages


TOOLS = [
    {
        'type': 'function',
        'function': {
            'name': 'weather',
            'parameters': {
                'type': 'object',
                'properties': {'city': {'type': 'string'}},
                'required': ['city'],
                'additionalProperties': False,
            },
        },
    }
]


def test_policy_selects_only_enabled_rules() -> None:
    policy = ToolCallBadPatternPolicy.model_validate(
        {
            'id': 'selected-rules-v1',
            'penalty': 0.25,
            'patterns': {
                'tool_call_args_invalid': {'enabled': True, 'penalize': False},
                'tool_call_hallucination': {'enabled': False},
            },
        }
    )

    assert policy.enabled_rules() == {'tool_call_args_invalid'}
    assert policy.penalized_rules() == set()
    assert policy.penalty == 0.25


def test_policy_replaces_default_final_language_rule_only_for_ch_code_mt() -> None:
    policy = ToolCallBadPatternPolicy.model_validate(
        {
            'id': 'language-policy-v1',
            'patterns': {
                'final_turn_language': {'enabled': True},
                'final_turn_language_ch': {'enabled': True},
            },
        }
    )

    assert policy.enabled_rules_for_agent('code_mt') == {'final_turn_language'}
    assert policy.enabled_rules_for_agent('ch_code_mt') == {'final_turn_language_ch'}
    assert policy.enabled_rules_for_agent(None) == {'final_turn_language'}


def test_policy_rejects_unknown_or_empty_rule_selection() -> None:
    with pytest.raises(ValueError, match='unknown bad patterns'):
        ToolCallBadPatternPolicy.model_validate(
            {
                'id': 'bad-policy',
                'patterns': {'not_a_rule': {'enabled': True}},
            }
        )

    with pytest.raises(ValueError, match='at least one'):
        ToolCallBadPatternPolicy.model_validate(
            {
                'id': 'disabled-policy',
                'patterns': {'tool_call_args_invalid': {'enabled': False}},
            }
        )


def test_detector_runs_only_configured_bad_patterns() -> None:
    messages = [
        {
            'role': 'assistant',
            'content': '',
            'tool_calls': [
                _call({'city': 3}),
                _call({}, call_id='call-2', name='ghost'),
            ],
        }
    ]
    result = detect_tool_call_bad_patterns(
        messages,
        TOOLS,
        enabled_rules={'tool_call_args_invalid'},
    )

    assert result['issue_counts'] == {'tool_call_args_invalid': 1}
    assert {hit['problem_type'] for hit in result['hits']} == {'tool_call_args_invalid'}
    assert result['schema_registry_tool_count'] == 1
    assert result['schema_matched_call_count'] == 1
    assert result['schema_coverage'] == 0.5


def test_any_number_of_patterns_applies_one_fixed_penalty() -> None:
    policy = ToolCallBadPatternPolicy.model_validate(
        {
            'id': 'penalty-v1',
            'penalty': 0.3,
            'reward_threshold': 0.5,
            'patterns': {
                'tool_call_args_invalid': {'enabled': True},
                'tool_call_hallucination': {'enabled': True},
            },
        }
    )
    detection = {
        'is_bad_pattern': True,
        'issue_count': 3,
        'issue_counts': {
            'tool_call_args_invalid': 2,
            'tool_call_hallucination': 1,
        },
        'hits': [],
    }

    score, metrics = apply_tool_call_bad_pattern_penalty(1.0, detection, policy)

    assert score == 0.7
    assert detection['penalty'] == 0.3
    assert metrics['tool_call_bad_pattern_count'] == 3.0
    assert metrics['tool_call_bad_pattern/tool_call_args_invalid/count'] == 2.0
    assert metrics['tool_call_bad_pattern/tool_call_args_invalid/hit'] == 1.0
    assert metrics['tool_call_bad_pattern/tool_call_hallucination/count'] == 1.0
    assert metrics['tool_call_bad_pattern/tool_call_hallucination/hit'] == 1.0
    assert 'tool_call_bad_pattern/tool_call_args_invalid/penalty' not in metrics
    assert metrics['tool_call_bad_pattern_tool_call_count'] == 0.0
    assert metrics['tool_call_bad_pattern_schema_registry_tool_count'] == 0.0
    assert metrics['tool_call_bad_pattern_schema_matched_call_count'] == 0.0
    assert metrics['tool_call_bad_pattern_schema_coverage'] == 0.0


def test_rule_suggested_penalty_scales_configured_penalty_cap() -> None:
    policy = ToolCallBadPatternPolicy.model_validate(
        {
            'id': 'language-penalty-v1',
            'penalty': 0.5,
            'reward_threshold': 0.5,
            'patterns': {
                'final_turn_language_ch': {
                    'enabled': True,
                    'penalize': True,
                    'penalty': 0.5,
                },
            },
        }
    )
    detection = detect_tool_call_bad_patterns(
        [{'role': 'assistant', 'content': '中' * 70 + ' ' + 'word ' * 30}],
        **policy.detection_kwargs(agent_name='ch_code_mt'),
    )

    score, metrics = apply_tool_call_bad_pattern_penalty(1.0, detection, policy)

    assert detection['suggested_penalties'] == {'final_turn_language_ch': 0.2}
    assert score == pytest.approx(0.8)
    assert detection['penalty'] == pytest.approx(0.2)
    assert metrics['tool_call_bad_pattern_penalty'] == pytest.approx(0.2)


def test_final_turn_language_ch_diagnostic_records_applied_penalty() -> None:
    policy = ToolCallBadPatternPolicy.model_validate(
        {
            'id': 'language-diagnostic-v1',
            'patterns': {
                'final_turn_language_ch': {
                    'enabled': True,
                    'penalize': True,
                    'penalty': 0.5,
                    'reward_threshold': 0.5,
                },
            },
        }
    )
    detection = detect_tool_call_bad_patterns(
        [{'role': 'assistant', 'content': '中' * 70 + ' ' + 'word ' * 30}],
        **policy.detection_kwargs(agent_name='ch_code_mt'),
    )
    score, _ = apply_tool_call_bad_pattern_penalty(1.0, detection, policy)

    event = final_turn_language_ch_diagnostic(
        detection,
        policy,
        agent_name='ch_code_mt',
        instance_id='example-instance',
        original_score=1.0,
        final_score=score,
    )

    assert event is not None
    assert event['status'] == 'completed'
    assert event['entered'] is True
    assert event['evidence_found'] is True
    assert event['hit'] is True
    assert event['chinese_ratio'] == pytest.approx(0.7)
    assert event['script_units'] == {'cjk': 70, 'latin': 30}
    assert event['suggested_penalty'] == pytest.approx(0.2)
    assert event['threshold_met'] is True
    assert event['eligible'] is True
    assert event['penalized'] is True
    assert event['applied_penalty'] == pytest.approx(0.2)
    assert event['original_score'] == pytest.approx(1.0)
    assert event['final_score'] == pytest.approx(0.8)


def test_final_turn_language_ch_diagnostic_records_clean_and_inactive_cases() -> None:
    policy = ToolCallBadPatternPolicy.model_validate(
        {
            'id': 'language-diagnostic-v1',
            'patterns': {
                'final_turn_language_ch': {
                    'enabled': True,
                    'penalize': True,
                    'penalty': 0.5,
                    'reward_threshold': 0.5,
                },
            },
        }
    )
    detection = detect_tool_call_bad_patterns(
        [{'role': 'assistant', 'content': '中' * 90 + ' ' + 'word ' * 10}],
        **policy.detection_kwargs(agent_name='ch_code_mt'),
    )
    score, _ = apply_tool_call_bad_pattern_penalty(1.0, detection, policy)

    event = final_turn_language_ch_diagnostic(
        detection,
        policy,
        agent_name='ch_code_mt',
        instance_id='example-instance',
        original_score=1.0,
        final_score=score,
    )

    assert event is not None
    assert event['entered'] is True
    assert event['evidence_found'] is True
    assert event['hit'] is False
    assert event['chinese_ratio'] == pytest.approx(0.9)
    assert event['suggested_penalty'] == 0.0
    assert event['eligible'] is False
    assert event['penalized'] is False
    assert event['applied_penalty'] == 0.0
    assert event['final_score'] == pytest.approx(1.0)
    assert (
        final_turn_language_ch_diagnostic(
            detection,
            policy,
            agent_name='code_mt',
            instance_id='example-instance',
            original_score=1.0,
            final_score=score,
        )
        is None
    )


def test_detected_monitor_only_pattern_does_not_apply_penalty() -> None:
    policy = ToolCallBadPatternPolicy.model_validate(
        {
            'id': 'monitor-only-v1',
            'penalty': 0.3,
            'patterns': {
                'tool_call_args_invalid': {'enabled': True, 'penalize': False},
                'tool_call_hallucination': {'enabled': True, 'penalize': True},
            },
        }
    )
    detection = {
        'is_bad_pattern': True,
        'issue_count': 2,
        'issue_counts': {'tool_call_args_invalid': 2},
        'hits': [],
    }

    score, metrics = apply_tool_call_bad_pattern_penalty(1.0, detection, policy)

    assert score == 1.0
    assert detection['penalty_pattern'] is False
    assert detection['penalty_issue_count'] == 0
    assert detection['penalty_issue_counts'] == {}
    assert detection['penalized'] is False
    assert metrics['tool_call_bad_pattern_hit'] == 1.0
    assert metrics['tool_call_bad_pattern_penalty_hit'] == 0.0
    assert metrics['tool_call_bad_pattern/tool_call_args_invalid/count'] == 2.0
    assert metrics['tool_call_bad_pattern/tool_call_args_invalid/hit'] == 1.0
    assert metrics['tool_call_bad_pattern/tool_call_hallucination/count'] == 0.0
    assert metrics['tool_call_bad_pattern/tool_call_hallucination/hit'] == 0.0


@pytest.mark.parametrize('termination_reason', ['TRUNCATION', 'OUTPUT_LENGTH_EXCEEDED'])
def test_trajectory_truncation_applies_rule_local_penalty(termination_reason: str) -> None:
    policy = ToolCallBadPatternPolicy.model_validate(
        {
            'id': 'truncation-penalty-v1',
            'penalty': 0.5,
            'reward_threshold': 0.5,
            'patterns': {
                'trajectory_truncation': {
                    'enabled': True,
                    'penalize': True,
                    'penalty': 0.1,
                    'reward_threshold': 0.0,
                },
            },
        }
    )
    detection = detect_tool_call_bad_patterns(
        [{'role': 'assistant', 'content': 'partial output'}],
        termination_reason=termination_reason,
        **policy.detection_kwargs(),
    )

    score, metrics = apply_tool_call_bad_pattern_penalty(0.822, detection, policy)

    assert score == pytest.approx(0.722)
    assert detection['issue_counts'] == {'trajectory_truncation': 1}
    assert detection['penalty'] == 0.1
    assert detection['penalty_applied_rule_ids'] == ['trajectory_truncation']
    assert detection['hits'][0]['detail'] == {
        'trajectory_level': True,
        'termination_reason': termination_reason,
    }
    assert metrics['tool_call_bad_pattern/trajectory_truncation/hit'] == 1.0


def test_trajectory_truncation_does_not_hit_completed_trajectory() -> None:
    detection = detect_tool_call_bad_patterns(
        [{'role': 'assistant', 'content': 'done'}],
        termination_reason='ENV_DONE',
        enabled_rules={'trajectory_truncation'},
    )

    assert detection['issue_counts'] == {}
    assert detection['hits'] == []


def test_trajectory_truncation_penalty_does_not_stack_with_larger_rule_penalty() -> None:
    policy = ToolCallBadPatternPolicy.model_validate(
        {
            'id': 'truncation-no-stack-v1',
            'patterns': {
                'trajectory_truncation': {
                    'enabled': True,
                    'penalty': 0.1,
                    'reward_threshold': 0.0,
                },
                'tool_call_args_invalid': {
                    'enabled': True,
                    'penalty': 0.5,
                    'reward_threshold': 0.0,
                },
            },
        }
    )
    detection = detect_tool_call_bad_patterns(
        [{
            'role': 'assistant',
            'content': '',
            'tool_calls': [_call({'city': 3})],
        }],
        TOOLS,
        termination_reason='TRUNCATION',
        **policy.detection_kwargs(),
    )

    score, _ = apply_tool_call_bad_pattern_penalty(0.8, detection, policy)

    assert score == pytest.approx(0.3)
    assert detection['penalty_applied_rule_ids'] == ['tool_call_args_invalid']


def test_truncated_final_protocol_evidence_does_not_reduce_reward() -> None:
    policy = ToolCallBadPatternPolicy.model_validate(
        {
            'id': 'truncated-protocol-v1',
            'penalty': 0.3,
            'reward_threshold': 0.5,
            'patterns': {
                'tool_call_protocol_leak_hy': {
                    'enabled': True,
                    'penalize': True,
                    'penalty': 1.0,
                    'reward_threshold': 0.1,
                },
            },
        }
    )
    detection = detect_tool_call_bad_patterns(
        [{
            'role': 'assistant',
            'content': (
                '<tool_calls:6124c78e><tool_call:6124c78e>Bash'
                '</tool_call:6124c78e></tool_calls:6124c78e>'
            ),
        }],
        termination_reason='TRUNCATION',
        **policy.detection_kwargs(),
    )

    score, metrics = apply_tool_call_bad_pattern_penalty(0.822, detection, policy)

    assert score == 0.822
    assert detection['issue_counts'] == {}
    assert detection['nonissue_evidence_count'] == 1
    assert detection['penalty'] == 0.0
    assert metrics['tool_call_bad_pattern/tool_call_protocol_leak_hy/hit'] == 0.0


def test_penalizable_pattern_applies_penalty_alongside_monitor_only_hits() -> None:
    policy = ToolCallBadPatternPolicy.model_validate(
        {
            'id': 'mixed-v1',
            'penalty': 0.3,
            'patterns': {
                'tool_call_args_invalid': {'enabled': True, 'penalize': False},
                'tool_call_hallucination': {'enabled': True, 'penalize': True},
            },
        }
    )
    detection = {
        'is_bad_pattern': True,
        'issue_count': 3,
        'issue_counts': {
            'tool_call_args_invalid': 2,
            'tool_call_hallucination': 1,
        },
        'hits': [],
    }

    score, metrics = apply_tool_call_bad_pattern_penalty(1.0, detection, policy)

    assert score == 0.7
    assert detection['penalty_pattern'] is True
    assert detection['penalty_issue_count'] == 1
    assert detection['penalty_issue_counts'] == {'tool_call_hallucination': 1}
    assert metrics['tool_call_bad_pattern_penalty_count'] == 1.0


def test_fixed_penalty_is_capped_at_zero_and_respects_threshold() -> None:
    policy = ToolCallBadPatternPolicy.model_validate(
        {
            'id': 'penalty-v1',
            'penalty': 1.0,
            'reward_threshold': 0.5,
            'patterns': {'tool_call_args_invalid': {'enabled': True}},
        }
    )
    detection = {
        'is_bad_pattern': True,
        'issue_count': 1,
        'issue_counts': {'tool_call_args_invalid': 1},
        'hits': [],
    }

    score, _ = apply_tool_call_bad_pattern_penalty(0.8, detection.copy(), policy)
    below_threshold_score, _ = apply_tool_call_bad_pattern_penalty(0.4, detection.copy(), policy)

    assert score == 0.0
    assert below_threshold_score == 0.4


def test_rule_local_penalty_and_threshold_override_policy_defaults() -> None:
    policy = ToolCallBadPatternPolicy.model_validate(
        {
            'id': 'per-rule-penalty-v1',
            'penalty': 0.3,
            'reward_threshold': 0.5,
            'patterns': {
                'repetition_tool_call': {'enabled': True, 'penalize': True},
                'tool_call_args_invalid': {
                    'enabled': True,
                    'penalize': True,
                    'penalty': 1.0,
                    'reward_threshold': 0.1,
                },
            },
        }
    )
    detection = {
        'is_bad_pattern': True,
        'issue_count': 2,
        'issue_counts': {
            'repetition_tool_call': 1,
            'tool_call_args_invalid': 1,
        },
        'hits': [],
    }

    score, metrics = apply_tool_call_bad_pattern_penalty(0.4, detection, policy)

    assert score == 0.0
    assert detection['penalty'] == 0.4
    assert detection['penalty_issue_counts'] == {
        'repetition_tool_call': 1,
        'tool_call_args_invalid': 1,
    }
    assert detection['penalty_eligible_issue_counts'] == {'tool_call_args_invalid': 1}
    assert detection['penalty_applied_rule_ids'] == ['tool_call_args_invalid']
    assert metrics['tool_call_bad_pattern_eligible'] == 1.0


def test_multiple_rule_local_penalties_take_the_maximum_without_stacking() -> None:
    policy = ToolCallBadPatternPolicy.model_validate(
        {
            'id': 'per-rule-no-stack-v1',
            'penalty': 0.3,
            'reward_threshold': 0.5,
            'patterns': {
                'tool_call_args_invalid': {
                    'enabled': True,
                    'penalize': True,
                    'penalty': 0.6,
                    'reward_threshold': 0.1,
                },
                'tool_call_hallucination': {
                    'enabled': True,
                    'penalize': True,
                    'penalty': 1.0,
                    'reward_threshold': 0.1,
                },
            },
        }
    )
    detection = {
        'is_bad_pattern': True,
        'issue_count': 2,
        'issue_counts': {
            'tool_call_args_invalid': 1,
            'tool_call_hallucination': 1,
        },
        'hits': [],
    }

    score, _ = apply_tool_call_bad_pattern_penalty(1.0, detection, policy)

    assert score == 0.0
    assert detection['penalty'] == 1.0
    assert detection['penalty_applied_rule_ids'] == ['tool_call_hallucination']


def test_all_rule_ids_are_available_to_policy_profiles() -> None:
    assert len(RULE_TITLES) == 39
    assert 'single_turn_thinking_too_long' in RULE_TITLES
    assert 'reasoning_word_stutter' in RULE_TITLES


@pytest.mark.parametrize(
    ('field', 'text'),
    [
        ('reasoning_content', 'final answer is near. ' + ' — hmm' * 80 + ' ok then'),
        ('content', 'Pick the schema option:' + ' choice' * 70 + ' done.'),
        ('reasoning_content', '\n'.join(['Let me check the config file again.'] * 10)),
        ('reasoning_content', 'The fix is in the parser.' + ' ' * 260 + 'Then run tests.'),
    ],
)
def test_reasoning_repetition_loop_detects_within_turn_generation_collapse(field: str, text: str) -> None:
    result = detect_tool_call_bad_patterns(
        [{'role': 'assistant', field: text, 'content': '' if field != 'content' else text}],
        enabled_rules={'reasoning_repetition_loop'},
    )

    assert result['issue_counts'] == {'reasoning_repetition_loop': 1}
    assert result['hits'][0]['detail']['channel'] == ('content' if field == 'content' else 'reasoning')


def test_reasoning_repetition_loop_coalesces_overlapping_checks_per_channel() -> None:
    result = detect_tool_call_bad_patterns(
        [{
            'role': 'assistant',
            'reasoning_content': 'The clue is clear. ' + ' — hmm' * 80,
            'content': '',
        }],
        enabled_rules={'reasoning_repetition_loop'},
    )

    assert result['issue_count'] == 1
    detail = result['hits'][0]['detail']
    assert detail['raw_finding_count'] > 1
    assert {'unit_run', 'phrase_run'} <= set(detail['checks'])


def test_reasoning_repetition_loop_requires_same_dominant_word_across_windows() -> None:
    from nexus.extensions.harbor_bridge.bad_pattern_operator.rules.reasoning_repetition_loop import (
        _back_to_back_windows,
        _dominant_word,
    )

    assert _back_to_back_windows('alpha ' * 60 + 'beta ' * 60, 'dominant_word', _dominant_word) == []


def test_reasoning_repetition_loop_ignores_tool_call_arguments() -> None:
    result = detect_tool_call_bad_patterns(
        [{
            'role': 'assistant',
            'reasoning_content': 'Write the generated fixture.',
            'content': '',
            'tool_calls': [{
                'id': 'call-1',
                'type': 'function',
                'function': {
                    'name': 'Write',
                    'arguments': {'content': 'Q' * 2000},
                },
            }],
        }],
        enabled_rules={'reasoning_repetition_loop'},
    )

    assert result['issue_counts'] == {}


@pytest.mark.parametrize(
    'text',
    [
        ' '.join(['0'] * 120),
        '\n'.join(['| a | b | c |'] * 3 + ['|---|---|---|'] * 2),
        'render(\n  state,\n' + '  undefined,\n' * 8 + ')',
        'times: ' + '7:00 PM, ' * 40,
        '\n'.join(f'#include "ngraph/op/{name}.hpp"' for name in ('acos', 'add', 'and', 'asin', 'atan') * 4),
        'visualized fixture whitespace: ' + '⏎␣' * 200,
    ],
)
def test_reasoning_repetition_loop_ignores_structured_data_and_code(text: str) -> None:
    result = detect_tool_call_bad_patterns(
        [{'role': 'assistant', 'reasoning_content': text, 'content': ''}],
        enabled_rules={'reasoning_repetition_loop'},
    )

    assert result['issue_counts'] == {}


def test_reasoning_repetition_loop_uses_configured_fixed_penalty() -> None:
    policy = ToolCallBadPatternPolicy.model_validate(
        {
            'id': 'repetition-loop-v1',
            'penalty': 1.0,
            'reward_threshold': 0.1,
            'patterns': {
                'reasoning_repetition_loop': {
                    'enabled': True,
                    'penalize': True,
                    'penalty': 0.3,
                    'reward_threshold': 0.5,
                },
            },
        }
    )
    detection = detect_tool_call_bad_patterns(
        [{
            'role': 'assistant',
            'reasoning_content': 'The clue is clear. ' + ' — hmm' * 80,
            'content': '',
        }],
        **policy.detection_kwargs(),
    )

    score, _ = apply_tool_call_bad_pattern_penalty(0.8, detection, policy)

    assert score == pytest.approx(0.5)
    assert detection['penalty_applied_rule_ids'] == ['reasoning_repetition_loop']


def test_tool_call_hallucination_accepts_successfully_dispatched_dynamic_tool() -> None:
    messages = [
        {
            'role': 'assistant',
            'content': '',
            'tool_calls': [_call({}, name='dynamically_registered_tool')],
        },
        {
            'role': 'tool',
            'tool_call_id': 'call-1',
            'content': '{"ok": true}',
        },
    ]

    result = detect_tool_call_bad_patterns(
        messages,
        [],
        enabled_rules={'tool_call_hallucination'},
    )

    assert result['issue_counts'] == {}


def test_tool_call_hallucination_ignores_runtime_unavailable_evidence() -> None:
    messages = [
        {
            'role': 'assistant',
            'content': '',
            'tool_calls': [_call({'city': 'Shenzhen'})],
        },
        {
            'role': 'tool',
            'tool_call_id': 'call-1',
            'content': 'Tool weather is not available in the current environment',
            'is_error': True,
        },
    ]

    result = detect_tool_call_bad_patterns(
        messages,
        TOOLS,
        enabled_rules={'tool_call_hallucination'},
    )

    assert result['issue_counts'] == {}


def test_tool_call_hallucination_detects_undefined_tool_call() -> None:
    messages = [
        {
            'role': 'assistant',
            'content': '',
            'tool_calls': [_call({}, name='missing_tool')],
        },
    ]

    result = detect_tool_call_bad_patterns(
        messages,
        TOOLS,
        enabled_rules={'tool_call_hallucination'},
    )

    assert result['issue_counts'] == {'tool_call_hallucination': 1}
    assert result['hits'][0]['detail']['availability'] == 'undefined'
    assert result['hits'][0]['detail']['availability_reason'] == 'tool_absent_from_complete_registry'


def test_tool_call_hallucination_keeps_registry_proof_over_runtime_unavailability() -> None:
    messages = [
        {
            'role': 'assistant',
            'content': '',
            'tool_calls': [_call({}, name='missing_tool')],
        },
        {
            'role': 'tool',
            'tool_call_id': 'call-1',
            'content': 'No such tool available: missing_tool',
            'is_error': True,
        },
    ]

    result = detect_tool_call_bad_patterns(
        messages,
        TOOLS,
        enabled_rules={'tool_call_hallucination'},
    )

    assert result['issue_counts'] == {'tool_call_hallucination': 1}
    detail = result['hits'][0]['detail']
    assert detail['availability'] == 'undefined'
    assert detail['availability_reason'] == 'tool_absent_from_complete_registry'
    assert detail['runtime_availability'] == 'unavailable'


def test_tool_call_hallucination_detects_explicitly_inactive_tool_call() -> None:
    remove_tool = {
        'type': 'function',
        'function': {
            'name': 'remove_tool',
            'description': 'Remove and disable a tool from the active tool registry.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'tool_name': {
                        'type': 'string',
                        'description': 'The tool name to remove.',
                    },
                },
                'required': ['tool_name'],
            },
        },
    }
    messages = [
        {
            'role': 'assistant',
            'content': '',
            'tool_calls': [_call({'tool_name': 'weather'}, name='remove_tool')],
        },
        {
            'role': 'tool',
            'tool_call_id': 'call-1',
            'content': '{"success": true}',
        },
        {
            'role': 'assistant',
            'content': '',
            'tool_calls': [_call({'city': 'Shenzhen'}, call_id='call-2')],
        },
    ]

    result = detect_tool_call_bad_patterns(
        messages,
        [*TOOLS, remove_tool],
        enabled_rules={'tool_call_hallucination'},
    )

    assert result['issue_counts'] == {'tool_call_hallucination': 1}
    assert result['hits'][0]['detail']['availability'] == 'unavailable'
    assert result['hits'][0]['detail']['availability_reason'] == 'tool_inactive_at_call'


ACTIVATE_TOOL = {
    'type': 'function',
    'function': {
        'name': 'mcp__erp__activate_tools',
        'description': 'Activate a tool so that it can be called.',
        'parameters': {
            'type': 'object',
            'properties': {'toolName': {'type': 'string', 'description': 'The tool name to activate.'}},
            'required': ['toolName'],
        },
    },
}


def _rejected_call(name: str, *, before: tuple[dict, ...] = ()) -> list[dict]:
    return [
        *before,
        {'role': 'assistant', 'content': '', 'tool_calls': [_call({}, call_id='call-x', name=name)]},
        {
            'role': 'tool',
            'tool_call_id': 'call-x',
            'content': f'<tool_use_error>Error: No such tool available: {name}</tool_use_error>',
        },
    ]


def test_tool_call_hallucination_detects_unseen_tool_rejected_in_dynamic_registry() -> None:
    result = detect_tool_call_bad_patterns(
        _rejected_call('mcp__erp__get_vendor_stock'),
        [*TOOLS, ACTIVATE_TOOL],
        enabled_rules={'tool_call_hallucination'},
    )

    assert result['issue_counts'] == {'tool_call_hallucination': 1}
    detail = result['hits'][0]['detail']
    assert detail['availability'] == 'unjudgeable'
    assert detail['availability_reason'] == 'dynamic_registry_without_complete_snapshot'
    assert detail['runtime_availability'] == 'unavailable'
    assert detail['unseen_before_call'] is True


def test_tool_call_hallucination_accepts_tool_returned_by_dynamic_registry() -> None:
    activation = (
        {
            'role': 'assistant',
            'content': '',
            'tool_calls': [_call({'toolName': 'getSupplierList'}, call_id='call-a', name='mcp__erp__activate_tools')],
        },
        {
            'role': 'tool',
            'tool_call_id': 'call-a',
            'content': '{"activated": true, "requiresToolsListRefresh": true, "toolName": "getSupplierList"}',
        },
    )

    result = detect_tool_call_bad_patterns(
        _rejected_call('mcp__erp__getSupplierList', before=activation),
        [*TOOLS, ACTIVATE_TOOL],
        enabled_rules={'tool_call_hallucination'},
    )

    assert result['issue_counts'] == {}


def test_tool_call_hallucination_accepts_dynamic_tool_named_by_user() -> None:
    result = detect_tool_call_bad_patterns(
        _rejected_call(
            'mcp__erp__get_vendor_stock',
            before=({'role': 'user', 'content': '用 mcp__erp__get_vendor_stock 查一下库存'},),
        ),
        [*TOOLS, ACTIVATE_TOOL],
        enabled_rules={'tool_call_hallucination'},
    )

    assert result['issue_counts'] == {}


def test_tool_call_hallucination_accepts_dynamic_tool_short_name_from_user() -> None:
    result = detect_tool_call_bad_patterns(
        _rejected_call(
            'mcp__erp__get_vendor_stock',
            before=({'role': 'user', 'content': '请调用 get_vendor_stock 查询库存'},),
        ),
        [*TOOLS, ACTIVATE_TOOL],
        enabled_rules={'tool_call_hallucination'},
    )

    assert result['issue_counts'] == {}


def test_tool_call_hallucination_needs_runtime_rejection_in_dynamic_registry() -> None:
    messages = [
        {'role': 'assistant', 'content': '', 'tool_calls': [_call({}, name='mcp__erp__get_vendor_stock')]},
        {'role': 'tool', 'tool_call_id': 'call-1', 'content': '{"ok": true}'},
    ]

    result = detect_tool_call_bad_patterns(
        messages,
        [*TOOLS, ACTIVATE_TOOL],
        enabled_rules={'tool_call_hallucination'},
    )

    assert result['issue_counts'] == {}


def test_repetition_tool_call_still_flags_immediate_non_polling_duplicate() -> None:
    result = detect_tool_call_bad_patterns(
        _repeated_tool_messages(2, content='completed'),
        enabled_rules={'repetition_tool_call'},
    )

    assert result['issue_counts'] == {'repetition_tool_call': 1}


@pytest.mark.parametrize('with_results', [False, True])
def test_repetition_tool_call_flags_same_turn_equivalent_requests(with_results: bool) -> None:
    calls = [
        _call({'city': 'Paris', 'units': 'metric'}, call_id='first'),
        _call({'city': 'London'}, call_id='other'),
        _call('{"units": "metric", "city": "Paris"}', call_id='second'),
    ]
    # Additional transport metadata does not change the request identity.
    calls[2]['transport_metadata'] = {'attempt': 2}
    messages = [{'role': 'assistant', 'content': '', 'tool_calls': calls}]
    if with_results:
        messages.extend(
            [
                {'role': 'tool', 'tool_call_id': 'first', 'content': 'sunny'},
                {'role': 'tool', 'tool_call_id': 'other', 'content': 'cloudy'},
                {'role': 'tool', 'tool_call_id': 'second', 'content': 'rainy'},
            ]
        )
    result = detect_tool_call_bad_patterns(messages, enabled_rules={'repetition_tool_call'})

    assert result['issue_counts'] == {'repetition_tool_call': 1}
    hit = result['hits'][0]
    assert hit['message_index'] == 0
    assert hit['detail'] == {
        'check': 'same_turn_duplicate_request',
        'duplicate_count': 2,
        'tool_call_indices': [0, 2],
    }


def test_repetition_tool_call_flags_same_turn_polling_without_cross_turn_allowance() -> None:
    calls = [_call({'job_id': 'job-1'}, call_id=f'poll-{i}', name='query_status') for i in range(20)]
    messages = [{'role': 'assistant', 'content': '', 'tool_calls': calls}]
    messages.extend({'role': 'tool', 'tool_call_id': call['id'], 'content': 'queued'} for call in calls)

    result = detect_tool_call_bad_patterns(messages, enabled_rules={'repetition_tool_call'})

    assert result['issue_counts'] == {'repetition_tool_call': 1}
    assert result['hits'][0]['detail']['duplicate_count'] == 20


def test_repetition_tool_call_keeps_distinct_requests_and_malformed_arguments_separate() -> None:
    calls = [
        _call({'city': 'Paris'}, call_id='paris'),
        _call({'city': 'London'}, call_id='london'),
        _call({'city': 'Paris'}, call_id='different-tool', name='forecast'),
        _call('{invalid', call_id='invalid-first'),
        _call('{invalid', call_id='invalid-second'),
    ]
    result = detect_tool_call_bad_patterns(
        [{'role': 'assistant', 'content': '', 'tool_calls': calls}],
        enabled_rules={'repetition_tool_call'},
    )

    assert result['issue_counts'] == {}


def test_repetition_tool_call_reports_each_same_turn_duplicate_group_once() -> None:
    calls = [
        _call({'city': city}, call_id=f'call-{index}')
        for index, city in enumerate(['Paris', 'London', 'Paris', 'London', 'Paris'])
    ]
    result = detect_tool_call_bad_patterns(
        [{'role': 'assistant', 'content': '', 'tool_calls': calls}],
        enabled_rules={'repetition_tool_call'},
    )

    assert result['issue_counts'] == {'repetition_tool_call': 2}
    assert [hit['detail']['duplicate_count'] for hit in result['hits']] == [3, 2]


def test_repetition_tool_call_same_turn_image_reads_remain_monitor_only() -> None:
    calls = [_call({'file_path': '/tmp/image.png'}, name='Read', call_id=f'image-{i}') for i in range(2)]
    result = detect_tool_call_bad_patterns(
        [{'role': 'assistant', 'content': '', 'tool_calls': calls}],
        enabled_rules={'repetition_tool_call'},
    )

    assert result['issue_counts'] == {}
    assert result['nonissue_evidence_count'] == 1
    assert result['nonissue_evidence'][0]['detail']['monitor_reason'] == 'repeated_image_read'


def test_repetition_tool_call_same_turn_blocking_timeouts_remain_exempt() -> None:
    calls = [_call({'job_id': 'job-1'}, name='TaskOutput', call_id=f'wait-{i}') for i in range(2)]
    messages = [{'role': 'assistant', 'content': '', 'tool_calls': calls}]
    messages.extend(
        {'role': 'tool', 'tool_call_id': call['id'], 'content': '<retrieval_status>timeout</retrieval_status>'}
        for call in calls
    )
    result = detect_tool_call_bad_patterns(messages, enabled_rules={'repetition_tool_call'})

    assert result['issue_counts'] == {}


@pytest.mark.parametrize('penalize', [False, True])
def test_repetition_tool_call_same_turn_uses_existing_penalty_settings(penalize: bool) -> None:
    policy = ToolCallBadPatternPolicy.model_validate(
        {
            'id': 'same-turn-repetition',
            'penalty': 1.0,
            'reward_threshold': 0.0,
            'patterns': {
                'repetition_tool_call': {
                    'enabled': True,
                    'penalize': penalize,
                    'penalty': 0.3,
                    'reward_threshold': 0.7,
                },
            },
        }
    )
    calls = [_call({'city': 'Paris'}, call_id=f'call-{i}') for i in range(2)]
    detection = detect_tool_call_bad_patterns(
        [{'role': 'assistant', 'content': '', 'tool_calls': calls}],
        enabled_rules=policy.enabled_rules(),
    )
    score, metrics = apply_tool_call_bad_pattern_penalty(1.0, detection.copy(), policy)
    below_threshold_score, _ = apply_tool_call_bad_pattern_penalty(0.6, detection.copy(), policy)

    assert score == (0.7 if penalize else 1.0)
    assert below_threshold_score == 0.6
    assert metrics['tool_call_bad_pattern/repetition_tool_call/hit'] == 1.0
    assert metrics['tool_call_bad_pattern_penalized'] == float(penalize)


@pytest.mark.parametrize(
    'status',
    [
        'queued',
        'running',
        'pending',
        'in_progress',
        'processing',
        'not_ready',
        'rate_limited',
        'throttled',
        'temporarily_unavailable',
        'preparation_active',
    ],
)
def test_repetition_tool_call_allows_three_transient_poll_repeats(status: str) -> None:
    result = detect_tool_call_bad_patterns(
        _repeated_tool_messages(4, content=status),
        enabled_rules={'repetition_tool_call'},
    )

    assert result['issue_counts'] == {}


@pytest.mark.parametrize(
    'status',
    [
        '【生成中】语料生成任务尚未完成，请间隔约 30 秒后再次调用 get_generation_status 查询',
        '任务排队中，请稍后查询',
        '任务处理中',
        '等待生成',
        '任务运行中',
        '服务暂不可用，请稍后重试',
    ],
)
def test_repetition_tool_call_allows_chinese_transient_poll_repeats(status: str) -> None:
    result = detect_tool_call_bad_patterns(
        _repeated_tool_messages(4, content=status),
        enabled_rules={'repetition_tool_call'},
    )

    assert result['issue_counts'] == {}


def test_repetition_tool_call_flags_chinese_polling_after_repeat_allowance() -> None:
    result = detect_tool_call_bad_patterns(
        _repeated_tool_messages(5, content='【生成中】任务尚未完成，请稍后再次查询'),
        enabled_rules={'repetition_tool_call'},
    )

    assert result['issue_counts'] == {'repetition_tool_call': 1}
    assert '连续重复超过 3 次' in result['hits'][0]['message']


def test_repetition_tool_call_monitors_repeated_image_read_without_penalty() -> None:
    messages = _repeated_tool_messages(
        2,
        name='Read',
        content='image/png;base64,same-image',
    )
    # Override the helper's status-query arguments with an actual image path.
    for message in messages:
        for call in message.get('tool_calls') or []:
            call['function']['arguments'] = {'file_path': '/tmp/screenshot.png'}
    result = detect_tool_call_bad_patterns(
        messages,
        enabled_rules={'repetition_tool_call'},
    )

    assert result['issue_counts'] == {}
    assert result['nonissue_evidence_count'] == 1
    assert result['nonissue_evidence'][0]['detail']['monitor_reason'] == 'repeated_image_read'


def test_repetition_tool_call_flags_polling_after_repeat_allowance() -> None:
    result = detect_tool_call_bad_patterns(
        _repeated_tool_messages(5, content='<retrieval_status>pending</retrieval_status>'),
        enabled_rules={'repetition_tool_call'},
    )

    assert result['issue_counts'] == {'repetition_tool_call': 1}
    assert '连续重复超过 3 次' in result['hits'][0]['message']


def test_repetition_tool_call_treats_task_output_timeout_as_waiting() -> None:
    result = detect_tool_call_bad_patterns(
        _repeated_tool_messages(
            6,
            name='TaskOutput',
            content='<retrieval_status>timeout</retrieval_status>',
        ),
        enabled_rules={'repetition_tool_call'},
    )

    assert result['issue_counts'] == {}


def test_repetition_tool_call_resets_poll_streak_after_intervening_wait() -> None:
    messages = _repeated_tool_messages(4, content='pending', call_id_prefix='before')
    messages.extend(_repeated_tool_messages(1, name='Wait', content='completed', call_id_prefix='wait'))
    messages.extend(_repeated_tool_messages(4, content='pending', call_id_prefix='after'))

    result = detect_tool_call_bad_patterns(
        messages,
        enabled_rules={'repetition_tool_call'},
    )

    assert result['issue_counts'] == {}


def test_edit_failure_requires_consecutive_failures_after_recovery() -> None:
    def edit(call_id: str) -> dict:
        return {
            'role': 'assistant',
            'content': '',
            'tool_calls': [{
                'id': call_id,
                'type': 'function',
                'function': {
                    'name': 'edit',
                    'arguments': {'file_path': '/tmp/a.py', 'old_string': 'a', 'new_string': 'b'},
                },
            }],
        }

    messages = [
        edit('call-1'),
        {'role': 'tool', 'tool_call_id': 'call-1', 'content': 'old_string not found', 'is_error': True},
        edit('call-2'),
        {'role': 'tool', 'tool_call_id': 'call-2', 'content': 'Done'},
        edit('call-3'),
        {'role': 'tool', 'tool_call_id': 'call-3', 'content': 'old_string not found', 'is_error': True},
    ]

    result = detect_tool_call_bad_patterns(
        messages,
        enabled_rules={'edit_call_failure', 'edit_call_never_recovered'},
    )

    assert result['issue_counts'] == {}


def test_edit_failure_recognizes_nonzero_exit_code_and_trailing_streak() -> None:
    def edit(call_id: str) -> dict:
        return {
            'role': 'assistant',
            'content': '',
            'tool_calls': [{
                'id': call_id,
                'type': 'function',
                'function': {
                    'name': 'edit',
                    'arguments': {'file_path': '/tmp/a.py', 'old_string': 'a', 'new_string': 'b'},
                },
            }],
        }

    messages = [
        edit('call-1'),
        {'role': 'tool', 'tool_call_id': 'call-1', 'content': 'old_string not found', 'exit_code': 1},
        edit('call-2'),
        {'role': 'tool', 'tool_call_id': 'call-2', 'content': 'old_string not found', 'inferred_outcome': 'failure'},
    ]

    result = detect_tool_call_bad_patterns(
        messages,
        enabled_rules={'edit_call_failure', 'edit_call_never_recovered'},
    )

    assert result['issue_counts'] == {
        'edit_call_failure': 1,
        'edit_call_never_recovered': 1,
    }


def test_edit_failure_recognizes_tool_use_error_protocol_without_metadata() -> None:
    def edit(call_id: str) -> dict:
        return {
            'role': 'assistant',
            'content': '',
            'tool_calls': [{
                'id': call_id,
                'type': 'function',
                'function': {
                    'name': 'Edit',
                    'arguments': {'file_path': '/tmp/a.py', 'old_string': 'a', 'new_string': 'b'},
                },
            }],
        }

    messages = [
        edit('call-1'),
        {
            'role': 'tool',
            'tool_call_id': 'call-1',
            'content': (
                '<tool_use_error source="harness">'
                'String to replace not found in file.'
                '</tool_use_error>'
            ),
        },
        edit('call-2'),
        {
            'role': 'tool',
            'tool_call_id': 'call-2',
            'content': '  <TOOL_USE_ERROR>String to replace not found in file.</TOOL_USE_ERROR>',
        },
    ]

    result = detect_tool_call_bad_patterns(
        messages,
        enabled_rules={'edit_call_failure', 'edit_call_never_recovered'},
    )

    assert result['issue_counts'] == {
        'edit_call_failure': 1,
        'edit_call_never_recovered': 1,
    }


def test_tool_use_error_protocol_marker_is_anchored_to_result_start() -> None:
    def edit(call_id: str) -> dict:
        return {
            'role': 'assistant',
            'content': '',
            'tool_calls': [{
                'id': call_id,
                'type': 'function',
                'function': {
                    'name': 'Edit',
                    'arguments': {'file_path': '/tmp/a.py', 'old_string': 'a', 'new_string': 'b'},
                },
            }],
        }

    content = 'Documentation example: <tool_use_error>String to replace not found in file.</tool_use_error>'
    messages = [
        edit('call-1'),
        {'role': 'tool', 'tool_call_id': 'call-1', 'content': content},
        edit('call-2'),
        {'role': 'tool', 'tool_call_id': 'call-2', 'content': content},
    ]

    result = detect_tool_call_bad_patterns(
        messages,
        enabled_rules={'edit_call_failure', 'edit_call_never_recovered'},
    )

    assert result['issue_counts'] == {}


def test_edit_never_recovered_is_cleared_by_successful_write() -> None:
    def call(call_id: str, name: str, arguments: dict) -> dict:
        return {
            'role': 'assistant',
            'content': '',
            'tool_calls': [{
                'id': call_id,
                'type': 'function',
                'function': {'name': name, 'arguments': arguments},
            }],
        }

    path = '/tmp/a.py'
    edit_arguments = {'file_path': path, 'old_string': 'a', 'new_string': 'b'}
    messages = [
        call('call-1', 'Edit', edit_arguments),
        {'role': 'tool', 'tool_call_id': 'call-1', 'content': 'String to replace not found', 'is_error': True},
        call('call-2', 'Edit', edit_arguments),
        {'role': 'tool', 'tool_call_id': 'call-2', 'content': 'String to replace not found', 'is_error': True},
        call('call-3', 'Write', {'file_path': path, 'content': 'replacement'}),
        {'role': 'tool', 'tool_call_id': 'call-3', 'content': 'The file has been updated successfully.'},
    ]

    result = detect_tool_call_bad_patterns(
        messages,
        enabled_rules={'edit_call_failure', 'edit_call_never_recovered'},
    )

    assert result['issue_counts'] == {'edit_call_failure': 1}


def test_edit_never_recovered_is_not_cleared_by_failed_write() -> None:
    def call(call_id: str, name: str, arguments: dict) -> dict:
        return {
            'role': 'assistant',
            'content': '',
            'tool_calls': [{
                'id': call_id,
                'type': 'function',
                'function': {'name': name, 'arguments': arguments},
            }],
        }

    path = '/tmp/a.py'
    edit_arguments = {'file_path': path, 'old_string': 'a', 'new_string': 'b'}
    messages = [
        call('call-1', 'Edit', edit_arguments),
        {'role': 'tool', 'tool_call_id': 'call-1', 'content': 'String to replace not found', 'is_error': True},
        call('call-2', 'Edit', edit_arguments),
        {'role': 'tool', 'tool_call_id': 'call-2', 'content': 'String to replace not found', 'is_error': True},
        call('call-3', 'Write', {'file_path': path, 'content': 'replacement'}),
        {'role': 'tool', 'tool_call_id': 'call-3', 'content': 'Error: permission denied', 'is_error': True},
    ]

    result = detect_tool_call_bad_patterns(
        messages,
        enabled_rules={'edit_call_failure', 'edit_call_never_recovered'},
    )

    assert result['issue_counts'] == {
        'edit_call_failure': 1,
        'edit_call_never_recovered': 1,
    }


def test_policy_rejects_retired_legacy_deferred_config() -> None:
    with pytest.raises(ValueError, match='Extra inputs are not permitted'):
        ToolCallBadPatternPolicy.model_validate(
            {
                'id': 'bad-pattern-v1',
                'patterns': {'edit_call_failure': {'enabled': True, 'penalize': True}},
                'legacy_deferred': {
                    'str_replace_old_eq_new': {
                        'enabled': True,
                        'penalty': 0.1,
                        'reward_threshold': 1.0,
                    }
                },
            }
        )
