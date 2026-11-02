"""
Grading logic for SWE-Bench-Multilingual dataset.

This module provides functionality to evaluate model predictions against test results.
"""

from typing import Any

from nexus.extensions.tasks.swe.swe_bench_multilingual.constants import (
    APPLY_PATCH_FAIL,
    END_TEST_OUTPUT,
    FAIL_TO_PASS,
    KEY_INSTANCE_ID,
    KEY_PREDICTION,
    PASS_TO_PASS,
    RESET_FAILED,
    START_TEST_OUTPUT,
    TESTS_ERROR,
    TESTS_TIMEOUT,
    ResolvedStatus,
    TestStatus,
)
from nexus.extensions.tasks.swe.swe_bench_multilingual.test_spec import TestSpec


def resolve_test_name(expected_name: str, status_map: dict[str, str]) -> str | None:
    """
    Resolve a test case name to the actual key in the status map.

    Handles empty parametrized test names (e.g., "test_name[]" should match "test_name[unit0]")
    where pytest displays empty-string params differently than expected.

    Args:
        expected_name: The expected test case name from gold results
        status_map: The status map from parsed test logs

    Returns:
        The resolved test name if found, otherwise None
    """
    # Direct match
    if expected_name in status_map:
        return expected_name

    # Handle empty parametrized test names
    if expected_name.endswith('[]'):
        base_name = expected_name[:-2]  # Remove trailing "[]"
        for log_name in status_map:
            if log_name.startswith(base_name + '[') and log_name.endswith(']'):
                return log_name

    return None


def test_passed(case: str, sm: dict[str, str]) -> bool:
    resolved_case = resolve_test_name(case, sm)
    if resolved_case is None:
        return False
    return sm[resolved_case] in [TestStatus.PASSED.value, TestStatus.XFAIL.value]


def test_failed(case: str, sm: dict[str, str]) -> bool:
    resolved_case = resolve_test_name(case, sm)
    if resolved_case is None:
        return True  # Not found means failed
    return sm[resolved_case] in [
        TestStatus.FAILED.value,
        TestStatus.ERROR.value,
    ]


def get_logs_eval(test_spec: TestSpec, log_fp: str) -> tuple[dict[str, str], bool]:
    """
    Retrieve evaluation results for a task instance from its corresponding log file

    Args:
        test_spec: Test specification
        log_fp: path to log file
    Returns:
        tuple: (status_map, patch_applied_successfully)
    """
    with open(log_fp, encoding='utf-8') as f:
        content = f.read()
        bad_codes = list(
            filter(
                lambda x: x in content,
                [
                    APPLY_PATCH_FAIL,
                    RESET_FAILED,
                    TESTS_ERROR,
                    TESTS_TIMEOUT,
                ],
            )
        )
        if bad_codes:
            return {}, False
        if not (START_TEST_OUTPUT in content and END_TEST_OUTPUT in content):
            # Test patch did not apply
            return {}, False

        # Get status map of evaluation results
        test_content = content.split(START_TEST_OUTPUT)[1].split(END_TEST_OUTPUT)[0]

        # For multilingual, we need to parse test results
        # This is a simplified parser - in practice, you'd need language-specific parsers
        status_map = parse_test_output(test_content, test_spec)

        return status_map, True


def parse_test_output(test_content: str, test_spec: TestSpec) -> dict[str, str]:
    """
    Parse test output to extract test status.
    Uses language-specific parsers based on the repository.

    Args:
        test_content: Test output content
        test_spec: Test specification

    Returns:
        dict: Mapping of test name to status
    """
    from nexus.extensions.tasks.swe.swe_bench_multilingual.log_parsers import get_parser_for_repo

    # Get the appropriate parser for this repository
    parser = get_parser_for_repo(test_spec.repo)

    # Parse the test output
    return parser(test_content, test_spec)


def get_eval_tests_report(
    eval_status_map: dict[str, str],
    gold_results: dict[str, str | list[str]],
) -> dict[str, dict[str, list[str]]]:
    """
    Create a report based on failure/pass change from gold results to eval results.

    Args:
        eval_status_map: evaluation status map
        gold_results: gold results
    Returns:
        report: report of metrics
    """
    # Calculate resolution metrics
    f2p_success = []
    f2p_failure = []
    for test_case in gold_results[FAIL_TO_PASS]:
        if test_passed(test_case, eval_status_map):
            f2p_success.append(test_case)
        elif test_failed(test_case, eval_status_map):
            f2p_failure.append(test_case)

    # Calculate maintenance metrics
    p2p_success = []
    p2p_failure = []
    for test_case in gold_results[PASS_TO_PASS]:
        if test_passed(test_case, eval_status_map):
            p2p_success.append(test_case)
        elif test_failed(test_case, eval_status_map):
            p2p_failure.append(test_case)

    return {
        FAIL_TO_PASS: {
            'success': f2p_success,
            'failure': f2p_failure,
        },
        PASS_TO_PASS: {
            'success': p2p_success,
            'failure': p2p_failure,
        },
    }


def compute_fail_to_pass(report: dict[str, dict[str, Any]]) -> float:
    """
    Compute fail-to-pass metric.
    """
    total = len(report[FAIL_TO_PASS]['success']) + len(report[FAIL_TO_PASS]['failure'])
    if total == 0:
        return 1
    return len(report[FAIL_TO_PASS]['success']) / total


def compute_pass_to_pass(report: dict[str, dict[str, Any]]) -> float:
    """
    Compute pass-to-pass metric.
    """
    total = len(report[PASS_TO_PASS]['success']) + len(report[PASS_TO_PASS]['failure'])
    if total == 0:
        return 1
    return len(report[PASS_TO_PASS]['success']) / total


def get_resolution_status(report: dict[str, dict[str, Any]]) -> str:
    """
    Determine resolved status of an evaluation instance

    Criteria:
        - If fail-to-pass = 1 and pass-to-pass = 1 -> FULL
        - If (fail-to-pass < 1 and > 0) and pass-to-pass = 1 -> PARTIAL
        - Otherwise -> NO
    """
    f2p = compute_fail_to_pass(report)
    p2p = compute_pass_to_pass(report)

    if f2p == 1 and p2p == 1:
        return ResolvedStatus.FULL.value
    if f2p < 1 and f2p > 0 and p2p == 1:
        return ResolvedStatus.PARTIAL.value
    return ResolvedStatus.NO.value


def get_eval_report(
    test_spec: TestSpec,
    prediction: dict[str, str],
    test_log_path: str,
    include_tests_status: bool,
) -> dict[str, Any]:
    """
    Generate a report of model evaluation results from a prediction, task instance,
    and evaluation log.

    Args:
        test_spec: test spec containing instance_id, FAIL_TO_PASS, and PASS_TO_PASS
        prediction: prediction containing instance_id, model_name_or_path, and model_patch
        test_log_path: path to evaluation log
        include_tests_status: whether to include the status of each test in the returned report
    Returns:
        report: report of metrics
    """
    report_map = {}

    instance_id = prediction[KEY_INSTANCE_ID]
    report_map[instance_id] = {
        'patch_is_None': False,
        'patch_exists': False,
        'patch_successfully_applied': False,
        'resolved': False,
    }

    # Check if the model patch exists
    if prediction[KEY_PREDICTION] is None:
        report_map[instance_id]['patch_is_None'] = True
        return report_map
    report_map[instance_id]['patch_exists'] = True

    # Get evaluation logs
    eval_status_map, found = get_logs_eval(test_spec, test_log_path)

    if not found:
        return report_map
    report_map[instance_id]['patch_successfully_applied'] = True

    eval_ref = {
        KEY_INSTANCE_ID: test_spec.instance_id,
        FAIL_TO_PASS: test_spec.FAIL_TO_PASS,
        PASS_TO_PASS: test_spec.PASS_TO_PASS,
    }

    report = get_eval_tests_report(eval_status_map, eval_ref)
    if get_resolution_status(report) == ResolvedStatus.FULL.value:
        report_map[instance_id]['resolved'] = True

    if include_tests_status:
        report_map[instance_id]['tests_status'] = report

    return report_map
