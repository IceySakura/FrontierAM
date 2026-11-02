"""
Grading logic for SWE-Bench-plus-plus dataset.
Copied from SWE-Bench-plus-plus/swebench/harness/grading.py
"""

from typing import Any

from nexus.extensions.tasks.swe.swe_bench_plus_plus.constants import (
    APPLY_PATCH_FAIL,
    END_TEST_OUTPUT,
    FAIL_ONLY_REPOS,
    FAIL_TO_FAIL,
    FAIL_TO_PASS,
    KEY_INSTANCE_ID,
    KEY_PREDICTION,
    PASS_TO_FAIL,
    PASS_TO_PASS,
    RESET_FAILED,
    START_TEST_OUTPUT,
    TESTS_ERROR,
    TESTS_TIMEOUT,
    EvalType,
    ResolvedStatus,
    TestStatus,
)
from nexus.extensions.tasks.swe.swe_bench_plus_plus.log_parsers import MAP_REPO_TO_PARSER
from nexus.extensions.tasks.swe.swe_bench_plus_plus.test_spec import TestSpec


def test_passed(case: str, sm: dict[str, str]) -> bool:
    return case in sm and sm[case] in [TestStatus.PASSED.value, TestStatus.XFAIL.value]


def test_failed(case: str, sm: dict[str, str]) -> bool:
    return case not in sm or sm[case] in [TestStatus.FAILED.value, TestStatus.ERROR.value, TestStatus.SKIPPED.value]


def get_logs_eval(test_spec: TestSpec, log_fp: str) -> tuple[dict[str, str], bool]:
    """
    Retrieve evaluation results for a task instance from its corresponding log file

    Args:
        log_fp (str): path to log file
    Returns:
        bool: whether the patch applied successfully
        dict: status map
    """
    repo = test_spec.repo

    # Get parser from test_spec if available, otherwise fall back to global map
    if hasattr(test_spec, 'log_parser') and test_spec.log_parser is not None:
        parser_or_code = test_spec.log_parser
    else:
        parser_or_code = MAP_REPO_TO_PARSER.get(repo, MAP_REPO_TO_PARSER.get('default'))

    if isinstance(parser_or_code, str):
        namespace = {}
        exec(parser_or_code, {}, namespace)
        log_parser = namespace['parse_log_to_json']
    else:
        log_parser = parser_or_code

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
            return {}, False

        # Get status map of evaluation results
        test_content = content.split(START_TEST_OUTPUT)[1].split(END_TEST_OUTPUT)[0]

        # Try parsing the content between markers first
        status_map = log_parser(test_content)

        # If no test results found between markers, try parsing the entire log content
        if not status_map:
            status_map = log_parser(content)

        return status_map, True


def get_eval_tests_report(
    eval_status_map: dict[str, str],
    gold_results: dict[str, str | list[str]],
    calculate_to_fail: bool = False,
    eval_type: EvalType = EvalType.PASS_AND_FAIL,
) -> dict[str, dict[str, list[str]]]:
    """
    Create a report based on failure/pass change from gold results to eval results.
    """

    def check_pass_and_fail(test_case, eval_status_map, success, failed):
        if test_passed(test_case, eval_status_map):
            success.append(test_case)
        elif test_failed(test_case, eval_status_map):
            failed.append(test_case)

    def check_fail_only(test_case, eval_status_map, success, failed):
        if test_case in eval_status_map and eval_status_map[test_case] == TestStatus.FAILED.value:
            failed.append(test_case)
        else:
            success.append(test_case)

    check_test_case = check_pass_and_fail if eval_type == EvalType.PASS_AND_FAIL else check_fail_only

    # Calculate resolution metrics
    f2p_success = []
    f2p_failure = []
    for test_case in gold_results[FAIL_TO_PASS]:
        check_test_case(test_case, eval_status_map, f2p_success, f2p_failure)

    # Calculate maintenance metrics
    p2p_success = []
    p2p_failure = []
    for test_case in gold_results[PASS_TO_PASS]:
        check_test_case(test_case, eval_status_map, p2p_success, p2p_failure)

    results = {
        FAIL_TO_PASS: {
            'success': f2p_success,
            'failure': f2p_failure,
        },
        PASS_TO_PASS: {
            'success': p2p_success,
            'failure': p2p_failure,
        },
    }

    f2f_success = []
    f2f_failure = []
    p2f_success = []
    p2f_failure = []
    if calculate_to_fail:
        for test_case in gold_results.get(FAIL_TO_FAIL, []):
            check_test_case(test_case, eval_status_map, f2f_success, f2f_failure)
        for test_case in gold_results.get(PASS_TO_FAIL, []):
            check_test_case(test_case, eval_status_map, p2f_success, p2f_failure)

    results.update({
        FAIL_TO_FAIL: {
            'success': f2f_success,
            'failure': f2f_failure,
        },
        PASS_TO_FAIL: {
            'success': p2f_success,
            'failure': p2f_failure,
        },
    })
    return results


def compute_fail_to_pass(report: dict[str, dict[str, Any]]) -> float:
    """
    Compute fail-to-pass metric. Accepts single report as argument.
    """
    total = len(report[FAIL_TO_PASS]['success']) + len(report[FAIL_TO_PASS]['failure'])
    if total == 0:
        return 1
    return len(report[FAIL_TO_PASS]['success']) / total


def compute_pass_to_pass(report: dict[str, dict[str, Any]]) -> float:
    """
    Compute pass-to-pass metric. Accepts single report as argument.
    """
    total = len(report[PASS_TO_PASS]['success']) + len(report[PASS_TO_PASS]['failure'])
    if total == 0:
        return 1
    return len(report[PASS_TO_PASS]['success']) / total


def get_resolution_status(report: dict[str, dict[str, Any]]) -> str:
    """
    Determine resolved status of an evaluation instance
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

    eval_type = EvalType.FAIL_ONLY if test_spec.repo in FAIL_ONLY_REPOS else EvalType.PASS_AND_FAIL

    report = get_eval_tests_report(eval_status_map, eval_ref, eval_type=eval_type)
    if get_resolution_status(report) == ResolvedStatus.FULL.value:
        report_map[instance_id]['resolved'] = True

    if include_tests_status:
        report_map[instance_id]['tests_status'] = report

    return report_map
