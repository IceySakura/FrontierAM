from pathlib import Path
from typing import Any

from nexus.extensions.tasks.swe.swe_smith.constants import (
    APPLY_PATCH_FAIL,
    FAIL_TO_FAIL,
    FAIL_TO_PASS,
    KEY_INSTANCE_ID,
    KEY_PREDICTION,
    PASS_TO_FAIL,
    PASS_TO_PASS,
    TESTS_TIMEOUT,
    TEST_OUTPUT_END,
    TEST_OUTPUT_START,
    ResolvedStatus,
    TestStatus,
)
from nexus.extensions.tasks.swe.swe_smith.profiles import registry
from nexus.extensions.tasks.swe.swe_smith.test_spec import TestSpec


def compute_fail_to_pass(report: dict[str, dict[str, list[str]]]) -> float:
    """Compute the ratio of fail-to-pass tests that succeeded."""
    f2p = report.get(FAIL_TO_PASS, {})
    success = len(f2p.get('success', []))
    failure = len(f2p.get('failure', []))
    total = success + failure
    return success / total if total > 0 else 0


def compute_pass_to_pass(report: dict[str, dict[str, list[str]]]) -> float:
    """Compute the ratio of pass-to-pass tests that succeeded."""
    p2p = report.get(PASS_TO_PASS, {})
    success = len(p2p.get('success', []))
    failure = len(p2p.get('failure', []))
    total = success + failure
    return success / total if total > 0 else 1  # Default to 1 if no P2P tests


def get_resolution_status(report: dict[str, dict[str, list[str]]]) -> str:
    """
    Determine resolved status of an evaluation instance.

    Criteria:
    - If fail-to-pass (Resolution) = 1 and pass-to-pass (Maintenance) = 1 -> FULL
    - If (fail-to-pass (Resolution) < 1 and > 0) and pass-to-pass (Maintenance) = 1 -> PARTIAL
    - Otherwise -> NO
    """
    f2p = compute_fail_to_pass(report)
    p2p = compute_pass_to_pass(report)
    if f2p == 1 and p2p == 1:
        return ResolvedStatus.FULL.value
    if 0 < f2p < 1 and p2p == 1:
        return ResolvedStatus.PARTIAL.value
    return ResolvedStatus.NO.value


def read_test_output(filename: str):
    content = Path(filename).read_text()
    if APPLY_PATCH_FAIL in content:
        return None, False
    if TESTS_TIMEOUT in content:
        return None, False
    if TEST_OUTPUT_START not in content or TEST_OUTPUT_END not in content:
        return content, False
    start_sep = f"+ : '{TEST_OUTPUT_START}'"
    end_sep = f"+ : '{TEST_OUTPUT_END}'"
    start_idx = content.find(start_sep)
    end_idx = content.find(end_sep)
    if start_idx > end_idx:
        raise ValueError('Invalid test output - Start and end markers are not in correct order')
    return content[start_idx:end_idx][len(start_sep) :], True


def get_valid_report(
    val_pregold_path: str,
    val_postgold_path: str,
    instance: dict,
) -> dict[str, list[str]]:
    """
    Get a report of changes in test pass/fail status between pre-gold and post-gold validation logs

    Args:
        val_pregold_path (str): path to pre-gold validation log
        val_postgold_path (str): path to post-gold validation log
    Returns:
        report (dict): map of type of status change to list of test cases
    """
    rp = registry.get(instance['repo'])

    val_pregold_output, found_pregold = read_test_output(val_pregold_path)
    val_postgold_output, found_postgold = read_test_output(val_postgold_path)
    pregold_sm = rp.log_parser(val_pregold_output) if found_pregold and val_pregold_output else {}
    postgold_sm = rp.log_parser(val_postgold_output) if found_postgold and val_postgold_output else {}

    report = {
        FAIL_TO_PASS: [],
        PASS_TO_PASS: [],
        FAIL_TO_FAIL: [],
        PASS_TO_FAIL: [],
    }

    for test_case in postgold_sm:
        if test_case not in pregold_sm:
            continue
        if pregold_sm[test_case] == TestStatus.PASSED.value and postgold_sm[test_case] == TestStatus.PASSED.value:
            report[PASS_TO_PASS].append(test_case)
        elif pregold_sm[test_case] == TestStatus.FAILED.value and postgold_sm[test_case] == TestStatus.PASSED.value:
            report[FAIL_TO_PASS].append(test_case)
        elif pregold_sm[test_case] == TestStatus.FAILED.value and postgold_sm[test_case] == TestStatus.FAILED.value:
            report[FAIL_TO_FAIL].append(test_case)
        elif pregold_sm[test_case] == TestStatus.PASSED.value and postgold_sm[test_case] == TestStatus.FAILED.value:
            report[PASS_TO_FAIL].append(test_case)

    return report


def test_passed(case: str, sm: dict[str, str]) -> bool:
    return case in sm and sm[case] in [
        TestStatus.PASSED.value,
        TestStatus.XFAIL.value,
    ]


def test_failed(case: str, sm: dict[str, str]) -> bool:
    return case not in sm or sm[case] in [
        TestStatus.FAILED.value,
        TestStatus.ERROR.value,
    ]


def get_eval_tests_report(
    eval_status_map: dict[str, str],
    gold_results: dict[str, str],
    calculate_to_fail: bool = False,
) -> dict[str, dict[str, list[str]]]:
    """
    Create a report based on failure/pass change from gold results to eval results.

    Args:
        eval_sm (dict): evaluation status map
        gold_results (dict): gold results
        calculate_to_fail (bool): whether to calculate metrics for "x to fail" tests
    Returns:
        report (dict): report of metrics

    Metric Definitions (Gold Result Pair + Eval Result):
    - Fail-Pass (F2P) + P: Success (Resolution)
    - Pass-Pass (P2P) + P: Success (Maintenance)
    - Fail-Pass (F2P) + F: Failure
    - Pass-Pass (P2P) + F: Failure

    Miscellaneous Definitions
    - Fail-Fail (F2F) + F: Failure Maintenance
    - Pass-Fail (P2F) + F: Not considered
    - Fail-Fail (F2F) + P: Success (Extra Credit)
    - Pass-Fail (P2F) + P: Not considered
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
        # Calculate "extra credit" metrics
        for test_case in gold_results[FAIL_TO_FAIL]:
            if test_passed(test_case, eval_status_map):
                f2f_success.append(test_case)
            elif test_failed(test_case, eval_status_map):
                f2f_failure.append(test_case)
        # Calculate not considered metrics
        for test_case in gold_results[PASS_TO_FAIL]:
            if test_passed(test_case, eval_status_map):
                p2f_success.append(test_case)
            elif test_failed(test_case, eval_status_map):
                p2f_failure.append(test_case)

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
        test_spec: TestSpec containing instance_id, FAIL_TO_PASS, and PASS_TO_PASS
        prediction: prediction containing instance_id and model_patch
        test_log_path: path to evaluation log
        include_tests_status: whether to include the status of each test in the returned report
    Returns:
        report (dict): report of metrics
    """
    instance_id = prediction[KEY_INSTANCE_ID]
    report_map: dict[str, Any] = {
        instance_id: {
            'patch_is_None': False,
            'patch_exists': False,
            'patch_successfully_applied': False,
            'resolved': False,
        }
    }

    # Check if model patch exists
    if prediction.get(KEY_PREDICTION) is None:
        report_map[instance_id]['patch_is_None'] = True
        return report_map
    report_map[instance_id]['patch_exists'] = True

    # Get evaluation logs
    rp = registry.get(test_spec.repo)
    test_output, found = read_test_output(test_log_path)
    if not found or test_output is None:
        return report_map
    report_map[instance_id]['patch_successfully_applied'] = True

    test_status_map = rp.log_parser(test_output)

    # Build gold results from test_spec
    eval_ref: dict[str, Any] = {
        KEY_INSTANCE_ID: test_spec.instance_id,
        FAIL_TO_PASS: test_spec.FAIL_TO_PASS,
        PASS_TO_PASS: test_spec.PASS_TO_PASS,
    }

    # Get evaluation test report
    report = get_eval_tests_report(test_status_map, eval_ref)
    if get_resolution_status(report) == ResolvedStatus.FULL.value:
        report_map[instance_id]['resolved'] = True

    if include_tests_status:
        report_map[instance_id]['tests_status'] = report

    return report_map
