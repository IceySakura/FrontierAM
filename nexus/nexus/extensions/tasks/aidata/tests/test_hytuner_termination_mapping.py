import json
import logging

from nexus.extensions.tasks.aidata.aidata_orchestrator import classify
from nexus.extensions.tasks.aidata.engine.models import AidataTaskOutput
from nexus.extensions.tasks.aidata.engine.task import (
    _TERMINATION_INITIALIZATION_ERROR,
    _TERMINATION_REWARD_ERROR,
    _TERMINATION_UNHANDLED_ERROR,
    _compact_rubric_status,
    _hytuner_extra_fields,
    _invalid_task_output,
    _log_mcp_tools_summary,
    _log_rubric_summary,
    _termination_from_rubric_results,
)


def test_unavailable_rubric_verdict_maps_to_reward_error() -> None:
    rows = [
        {'id': 'judge_a', 'verdict': 'UNAVAILABLE', 'passed': False},
        {'id': 'normal_miss', 'verdict': 'N', 'passed': False},
    ]

    assert _termination_from_rubric_results(rows) == (
        _TERMINATION_REWARD_ERROR,
        'judge_unavailable',
        ['judge_a'],
    )


def test_evaluator_error_maps_to_unhandled_error() -> None:
    rows = [
        {'id': 'shell_probe', 'verdict': 'ERROR', 'detail': 'run_command_failed: disconnected'},
    ]

    assert _termination_from_rubric_results(rows) == (
        _TERMINATION_UNHANDLED_ERROR,
        'rubric_evaluator_error',
        ['shell_probe'],
    )


def test_rubric_authoring_error_maps_to_initialization_error() -> None:
    rows = [
        {'id': 'bad_authoring', 'verdict': 'INVALID', 'passed': False},
        {'id': 'unknown_assert', 'verdict': 'UNKNOWN_ASSERT_TYPE', 'passed': False},
    ]

    assert _termination_from_rubric_results(rows) == (
        _TERMINATION_INITIALIZATION_ERROR,
        'rubric_authoring_error',
        ['bad_authoring', 'unknown_assert'],
    )


def test_normal_failed_rubric_verdicts_do_not_create_termination_reason() -> None:
    rows = [
        {'id': 'wrong_answer', 'verdict': 'N', 'passed': False},
        {'id': 'missing_file', 'verdict': 'MISSING', 'passed': False},
    ]

    assert _termination_from_rubric_results(rows) is None


def test_aidata_output_preserves_hytuner_extra_fields() -> None:
    extra_fields = _hytuner_extra_fields(
        _TERMINATION_REWARD_ERROR,
        failure_phase='rubric_evaluation',
        failure_kind='judge_unavailable',
        invalid_check_ids=['judge_a'],
    )

    output = AidataTaskOutput(task_id='aidata', score=0.0, extra_fields=extra_fields)

    dumped = output.model_dump()
    assert dumped['extra_fields']['hytuner.termination_reason'] == _TERMINATION_REWARD_ERROR
    assert dumped['extra_fields']['aidata.score_valid'] is False
    assert dumped['extra_fields']['aidata.invalid_check_ids'] == ['judge_a']


def test_invalid_task_output_logs_termination_mapping(caplog) -> None:
    caplog.set_level(logging.WARNING, logger='nexus.extensions.tasks.aidata.engine.task')

    output = _invalid_task_output(
        task_id='aidata_case',
        start=0.0,
        error_message='runtime acquire failed: readiness timeout',
        termination_reason='ENV_ERROR',
        failure_phase='runtime_acquire',
        failure_kind='runtime_acquire_failed',
    )

    assert output.extra_fields['hytuner.termination_reason'] == 'ENV_ERROR'
    assert 'invalid sample task=aidata_case reason=ENV_ERROR phase=runtime_acquire' in caplog.text
    assert 'kind=runtime_acquire_failed' in caplog.text


def test_compact_rubric_status_encodes_ordered_results() -> None:
    rows = [
        {'passed': True, 'verdict': 'Y'},
        {'passed': False, 'verdict': 'N'},
        {'passed': False, 'verdict': 'MISSING'},
        {'passed': False, 'verdict': 'UNAVAILABLE'},
        {'passed': False, 'verdict': 'ERROR'},
        {'passed': False, 'verdict': 'UNKNOWN_ASSERT_TYPE'},
    ]

    assert _compact_rubric_status(rows) == 'YNMUEI'


def test_mcp_tools_summary_logs_one_line(caplog) -> None:
    caplog.set_level(logging.WARNING, logger='nexus.extensions.tasks.aidata.engine.task')

    _log_mcp_tools_summary(
        'task_1',
        'case_1',
        {
            'wecom/im_search': object(),
            'base/finish': object(),
        },
    )

    assert '[aidata] mcp task=task_1 case=case_1 count=2 tools=base/finish,wecom/im_search' in caplog.text
    assert caplog.text.count('\n') == 1


def test_rubric_summary_logs_compact_status_without_check_ids(caplog) -> None:
    caplog.set_level(logging.WARNING, logger='nexus.extensions.tasks.aidata.engine.task')

    _log_rubric_summary(
        'task_1',
        'case_1',
        [
            {'id': 'hidden_id_a', 'passed': True, 'verdict': 'Y'},
            {'id': 'hidden_id_b', 'passed': False, 'verdict': 'UNAVAILABLE'},
        ],
        termination_reason='REWARD_ERROR',
        failure_kind='judge_unavailable',
    )

    assert '[aidata] rubric task=task_1 case=case_1 total=2 status=YU' in caplog.text
    assert 'invalid=REWARD_ERROR/judge_unavailable' in caplog.text
    assert 'hidden_id_a' not in caplog.text
    assert 'hidden_id_b' not in caplog.text
    assert caplog.text.count('\n') == 1


def test_reward_script_error_is_quarantined_as_a_case_bug(tmp_path) -> None:
    result_path = tmp_path / 'result.json'
    result_path.write_text(
        json.dumps({
            'success': True,
            'task_output': {
                'score': 0.0,
                'extra_fields': {
                    'aidata.score_valid': False,
                    'aidata.failure_phase': 'reward_calculation',
                    'aidata.failure_kind': 'reward_script_error',
                },
            },
        }),
        encoding='utf-8',
    )

    outcome, kind, score, _ = classify(result_path, rc=0, timed_out=False)

    assert (outcome, kind, score) == ('quarantine', 'reward_script_error', 0.0)
