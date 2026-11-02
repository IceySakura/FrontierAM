import re

from typing import Any

from nexus.extensions.tasks.swe.swe_rebench_v2.log_parsers import get_log_parser_by_instance
from nexus.extensions.tasks.swe.t3build_repolaunch_midtrain.constants import (
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
from nexus.extensions.tasks.swe.t3build_repolaunch_midtrain.test_spec import TestSpec


# Strip trailing timing tokens like "[1.34 ms]" or "[123 ms]" from test names.
_TIMING_SUFFIX_RE = re.compile(r'\s*\[\s*\d+(?:\.\d+)?\s*(?:ms|s)\s*\]\s*$', re.IGNORECASE)


def _strip_timing(name: str) -> str:
    return _TIMING_SUFFIX_RE.sub('', name).strip()


# MARK: Utility functions
def test_passed(case: str, sm: dict[str, str], has_new_test_file: bool) -> bool:
    if case in sm:
        return sm[case] in [TestStatus.PASSED.value, TestStatus.XFAIL.value]

    # Try matching after stripping timing suffix (e.g. "[4.06 ms]") from reference name.
    case_stripped = _strip_timing(case)
    if case_stripped != case and case_stripped in sm:
        return sm[case_stripped] in [TestStatus.PASSED.value, TestStatus.XFAIL.value]

    if has_new_test_file and '::' in case:
        import re

        parts = case.split('::')
        if len(parts) >= 3:
            file_part = re.escape(parts[0])
            method_part = re.escape(parts[-1])
            pattern = re.compile(f'^{file_part}::.*::{method_part}$')

            for key in sm:
                if pattern.match(key):
                    return sm[key] in [TestStatus.PASSED.value, TestStatus.XFAIL.value]
    return False


def test_failed(case: str, sm: dict[str, str], has_new_test_file: bool) -> bool:
    if case in sm:
        return sm[case] in [TestStatus.FAILED.value, TestStatus.ERROR.value, TestStatus.SKIPPED.value]

    # Try matching after stripping timing suffix.
    case_stripped = _strip_timing(case)
    if case_stripped != case and case_stripped in sm:
        return sm[case_stripped] in [TestStatus.FAILED.value, TestStatus.ERROR.value, TestStatus.SKIPPED.value]

    if has_new_test_file and '::' in case:
        import re

        parts = case.split('::')
        if len(parts) >= 3:
            file_part = re.escape(parts[0])
            method_part = re.escape(parts[-1])
            pattern = re.compile(f'^{file_part}::.*::{method_part}$')

            for key in sm:
                if pattern.match(key):
                    return sm[key] in [TestStatus.FAILED.value, TestStatus.ERROR.value, TestStatus.SKIPPED.value]

    return True


# MARK: Evaluation report functions
def get_logs_eval(test_spec: TestSpec, log_fp: str) -> tuple[dict[str, str], bool]:
    """
    Retrieve evaluation results for a task instance from its corresponding log file.
    get_log_parser_by_instance always returns a wrapped (log, test_spec) -> dict parser.
    """
    log_parser = get_log_parser_by_instance(test_spec.instance)

    with open(log_fp) as f:
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

        content = content.split(START_TEST_OUTPUT)[1].split(END_TEST_OUTPUT)[0]
        return log_parser(content, test_spec), True


def get_eval_tests_report(
    eval_status_map: dict[str, str],
    gold_results: dict[str, str | list[str]],
    calculate_to_fail: bool = False,
    eval_type: EvalType = EvalType.PASS_AND_FAIL,
    has_new_test_file: bool = False,
) -> dict[str, dict[str, list[str]]]:
    """
    Create a report based on failure/pass change from gold results to eval results.
    """

    def check_pass_and_fail(test_case, eval_status_map, success, failed):
        if test_passed(test_case, eval_status_map, has_new_test_file):
            success.append(test_case)
        elif test_failed(test_case, eval_status_map, has_new_test_file):
            failed.append(test_case)

    def check_fail_only(test_case, eval_status_map, success, failed):
        if test_case in eval_status_map and eval_status_map[test_case] == TestStatus.FAILED.value:
            failed.append(test_case)
        else:
            success.append(test_case)

    check_test_case = check_pass_and_fail if eval_type == EvalType.PASS_AND_FAIL else check_fail_only

    f2p_success = []
    f2p_failure = []
    for test_case in gold_results[FAIL_TO_PASS]:
        check_test_case(test_case, eval_status_map, f2p_success, f2p_failure)

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
        for test_case in gold_results[FAIL_TO_FAIL]:
            check_test_case(test_case, eval_status_map, f2f_success, f2f_failure)

        for test_case in gold_results[PASS_TO_FAIL]:
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
    total = len(report[FAIL_TO_PASS]['success']) + len(report[FAIL_TO_PASS]['failure'])
    if total == 0:
        return 1
    return len(report[FAIL_TO_PASS]['success']) / total


def compute_pass_to_pass(report: dict[str, dict[str, Any]]) -> float:
    total = len(report[PASS_TO_PASS]['success']) + len(report[PASS_TO_PASS]['failure'])
    if total == 0:
        return 1
    return len(report[PASS_TO_PASS]['success']) / total


def get_resolution_status(report: dict[str, dict[str, Any]]) -> str:
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
    run_id: str = '',
    has_new_test_file: bool = False,
) -> dict[str, Any]:
    report_map = {}

    instance_id = prediction[KEY_INSTANCE_ID]
    report_map[instance_id] = {
        'patch_is_None': False,
        'patch_exists': False,
        'patch_successfully_applied': False,
        'resolved': False,
        'status': '',
    }

    if prediction[KEY_PREDICTION] is None:
        report_map[instance_id]['patch_is_None'] = True
        report_map[instance_id]['status'] = 'SYSTEM_ERROR'
        return report_map
    report_map[instance_id]['patch_exists'] = True

    if isinstance(prediction[KEY_PREDICTION], str) and prediction[KEY_PREDICTION].strip() == '':
        report_map[instance_id]['status'] = 'EMPTY_PATCH'
        return report_map

    eval_status_map_after, after_found = get_logs_eval(test_spec, test_log_path)

    if not after_found:
        report_map[instance_id]['status'] = 'ERROR_PATCH'
        return report_map

    report_map[instance_id]['patch_successfully_applied'] = True

    eval_type = EvalType.FAIL_ONLY if test_spec.repo in FAIL_ONLY_REPOS else EvalType.PASS_AND_FAIL

    eval_ref = {
        KEY_INSTANCE_ID: test_spec.instance_id,
        FAIL_TO_PASS: test_spec.FAIL_TO_PASS,
        PASS_TO_PASS: test_spec.PASS_TO_PASS,
    }
    report = get_eval_tests_report(
        eval_status_map_after, eval_ref, eval_type=eval_type, has_new_test_file=has_new_test_file
    )

    if get_resolution_status(report) == ResolvedStatus.FULL.value:
        report_map[instance_id]['resolved'] = True
        report_map[instance_id]['status'] = 'RESOLVED'
    else:
        report_map[instance_id]['status'] = 'UNSOLVED'

    if include_tests_status:
        report_map[instance_id]['tests_status'] = report

    return report_map
