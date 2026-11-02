import pytest

from nexus.runner import single
from nexus.runner.hytuner import (
    HyTunerRunInput,
    _apply_hytuner_reward_policy,
    _configure_codex_training_compaction,
    _enable_tool_error_wrapping_for_segment_shaping,
    _resolve_termination_reason,
    _reward_policy_from_input,
)


def _run_input(agent_name: str = 'main_agent') -> HyTunerRunInput:
    return HyTunerRunInput(
        task_name='task',
        task_config={},
        agent_name=agent_name,
        agent_config={'max_iterations': 3},
    )


def test_reward_policy_is_explicit_per_input_and_defaults_to_continuous() -> None:
    run_input = _run_input()

    assert _reward_policy_from_input(run_input) is None
    assert _apply_hytuner_reward_policy(0.75, None) == 0.75
    assert _apply_hytuner_reward_policy(None, None) is None


def test_binary_reward_policy_uses_configured_threshold() -> None:
    run_input = _run_input()
    run_input.task_config['hytuner_reward_policy'] = {
        'mode': 'binary',
        'threshold': 0.8,
    }

    policy = _reward_policy_from_input(run_input)

    assert policy is not None
    assert policy.model_dump() == {'mode': 'binary', 'threshold': 0.8}
    assert _apply_hytuner_reward_policy(0.79, policy) == 0.0
    assert _apply_hytuner_reward_policy(0.8, policy) == 1.0
    assert _apply_hytuner_reward_policy(1.0, policy) == 1.0


def test_binary_reward_is_applied_before_bad_pattern_penalty() -> None:
    policy = _reward_policy_from_input(
        HyTunerRunInput(
            task_name='task',
            task_config={'hytuner_reward_policy': {'mode': 'binary', 'threshold': 1.0}},
        )
    )

    assert (
        _apply_hytuner_reward_policy(
            0.9,
            policy,
            {
                'tool_call_bad_pattern': {
                    'penalized': True,
                    'original_score': 1.0,
                    'penalty': 0.1,
                }
            },
        )
        == pytest.approx(0.9)
    )


@pytest.mark.parametrize(
    ('policy', 'message'),
    [
        ({'mode': 'continuous', 'threshold': 1.0}, "Input should be 'binary'"),
        ({'mode': 'binary'}, 'Field required'),
        ({'mode': 'binary', 'threshold': 1.0, 'topic': 'ToolUse'}, 'Extra inputs are not permitted'),
        ({'mode': 'binary', 'threshold': float('nan')}, 'Input should be a finite number'),
    ],
)
def test_reward_policy_rejects_unsupported_or_incomplete_shapes(policy: dict, message: str) -> None:
    run_input = _run_input()
    run_input.task_config['hytuner_reward_policy'] = policy

    with pytest.raises(ValueError, match=message):
        _reward_policy_from_input(run_input)


@pytest.mark.parametrize(
    ('reported_reason', 'success', 'error_type', 'expected'),
    [
        ('REWARD_ERROR', False, 'RuntimeError', 'REWARD_ERROR'),
        ('ENV_ERROR', False, 'RuntimeError', 'ENV_ERROR'),
        (None, False, 'TimeoutError', 'TASK_TIMEOUT'),
        (None, False, 'RuntimeError', 'UNHANDLED_ERROR'),
        ('ENV_DONE', False, 'RuntimeError', 'UNHANDLED_ERROR'),
        (None, True, None, 'ENV_DONE'),
    ],
)
def test_termination_reason_prefers_explicit_task_failure(
    reported_reason: str | None,
    success: bool,
    error_type: str | None,
    expected: str,
) -> None:
    error = None if error_type is None else single.ErrorInfo(type=error_type, message='task failed')

    assert (
        _resolve_termination_reason(
            reported_reason,
            single.RunOutput(success=success, error=error),
        )
        == expected
    )


@pytest.mark.parametrize(
    ('reported_reason', 'success', 'expected'),
    [
        (None, True, 'TRUNCATION'),
        (None, False, 'TRUNCATION'),
        ('ENV_DONE', True, 'ENV_DONE'),
        ('OUTPUT_LENGTH_EXCEEDED', True, 'OUTPUT_LENGTH_EXCEEDED'),
    ],
)
def test_gateway_truncation_applies_only_when_agent_reports_nothing(
    reported_reason: str | None, success: bool, expected: str
) -> None:
    error = None if success else single.ErrorInfo(type='ContextLengthExceededError', message='too long')
    result = single.RunOutput(success=success, error=error)

    assert _resolve_termination_reason(reported_reason, result, gateway_truncated=True) == expected


def test_tool_error_turn_penalty_enables_failed_tool_response_wrapping_for_main_agent() -> None:
    run_input = _run_input()

    _enable_tool_error_wrapping_for_segment_shaping(
        run_input,
        {'policy': 'tool_error_turn_penalty_only'},
    )

    assert run_input.agent_config == {
        'max_iterations': 3,
        'wrap_failed_tool_response_as_tool_error': True,
    }


def test_tool_error_turn_penalty_legacy_key_enables_failed_tool_response_wrapping_for_main_agent() -> None:
    run_input = _run_input()

    _enable_tool_error_wrapping_for_segment_shaping(
        run_input,
        {'shaping_policy': 'tool_error_turn_penalty_only'},
    )

    assert run_input.agent_config == {
        'max_iterations': 3,
        'wrap_failed_tool_response_as_tool_error': True,
    }


def test_non_tool_error_policy_does_not_enable_failed_tool_response_wrapping() -> None:
    run_input = _run_input()

    _enable_tool_error_wrapping_for_segment_shaping(
        run_input,
        {'policy': 'legacy'},
    )

    assert run_input.agent_config == {'max_iterations': 3}


def test_explicit_failed_tool_response_wrapping_false_is_preserved() -> None:
    run_input = _run_input()
    run_input.agent_config['wrap_failed_tool_response_as_tool_error'] = False

    _enable_tool_error_wrapping_for_segment_shaping(
        run_input,
        {'policy': 'tool_error_turn_penalty_only'},
    )

    assert run_input.agent_config == {
        'max_iterations': 3,
        'wrap_failed_tool_response_as_tool_error': False,
    }


def test_non_main_agent_does_not_enable_failed_tool_response_wrapping() -> None:
    run_input = _run_input(agent_name='custom_agent')

    _enable_tool_error_wrapping_for_segment_shaping(
        run_input,
        {'policy': 'tool_error_turn_penalty_only'},
    )

    assert run_input.agent_config == {'max_iterations': 3}


def test_codex_training_defaults_to_local_summary_compaction() -> None:
    run_input = _run_input(agent_name='codex')

    _configure_codex_training_compaction(run_input)

    assert run_input.agent_config == {'max_iterations': 3, 'use_azure_provider': False}
