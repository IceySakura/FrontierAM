import json

from pathlib import Path

import pytest

from nexus.extensions.tasks.swe.eval import _compute_partial_score
from nexus.extensions.tasks.swe.swe_bench_pro.dataset_handler import _enrich_test_output, get_eval_report_map


def _write_test_output(tmp_path: Path, tests: list[dict[str, str]]) -> Path:
    test_output_path = tmp_path / 'test_output.txt'
    test_output_path.write_text(json.dumps({'tests': tests}), encoding='utf-8')
    return test_output_path


def _report(instance: dict, test_output_path: Path, include_tests_status: bool = True) -> dict:
    report_map = get_eval_report_map(
        instance=instance,
        prediction={'instance_id': instance['instance_id'], 'model_patch': 'patch'},
        test_log_path=str(test_output_path),
        include_tests_status=include_tests_status,
    )
    return report_map[instance['instance_id']]


def test_pro_report_includes_tests_status_without_changing_resolved(tmp_path: Path) -> None:
    instance = {
        'instance_id': 'instance_repo__case',
        'FAIL_TO_PASS': "['TestFixed', 'TestStillBroken']",
        'PASS_TO_PASS': ['TestKept', 'TestLost', 'TestMissing'],
    }
    test_output_path = _write_test_output(
        tmp_path,
        [
            {'name': 'TestFixed', 'status': 'PASSED'},
            {'name': 'TestStillBroken', 'status': 'FAILED'},
            {'name': 'TestKept', 'status': 'PASSED'},
            {'name': 'TestLost', 'status': 'ERROR'},
        ],
    )

    report = _report(instance, test_output_path)

    assert report['resolved'] is False
    assert report['tests_status'] == {
        'FAIL_TO_PASS': {
            'success': ['TestFixed'],
            'failure': ['TestStillBroken'],
        },
        'PASS_TO_PASS': {
            'success': ['TestKept'],
            'failure': ['TestLost', 'TestMissing'],
        },
    }


def test_pro_report_omits_tests_status_when_not_requested(tmp_path: Path) -> None:
    instance = {
        'instance_id': 'instance_repo__case',
        'FAIL_TO_PASS': ['TestFixed'],
        'PASS_TO_PASS': [],
    }
    test_output_path = _write_test_output(tmp_path, [{'name': 'TestFixed', 'status': 'PASSED'}])

    report = _report(instance, test_output_path, include_tests_status=False)

    assert report == {'resolved': True}


def test_pro_report_preserves_fuzzy_gold_name_matching(tmp_path: Path) -> None:
    instance = {
        'instance_id': 'instance_repo__case',
        'FAIL_TO_PASS': "['test/i18n.js | i18n ']",
        'PASS_TO_PASS': [],
    }
    test_output_path = _write_test_output(
        tmp_path,
        [{'name': 'test/i18n.js | i18n "', 'status': 'PASSED'}],
    )

    report = _report(instance, test_output_path)

    assert report['resolved'] is True
    assert report['tests_status']['FAIL_TO_PASS'] == {
        'success': ['test/i18n.js | i18n '],
        'failure': [],
    }


def test_pro_report_can_drive_partial_score(tmp_path: Path) -> None:
    instance = {
        'instance_id': 'instance_repo__case',
        'FAIL_TO_PASS': ['TestFixed', 'TestStillBroken'],
        'PASS_TO_PASS': ['TestKept', 'TestLost', 'TestMissing'],
    }
    test_output_path = _write_test_output(
        tmp_path,
        [
            {'name': 'TestFixed', 'status': 'PASSED'},
            {'name': 'TestStillBroken', 'status': 'FAILED'},
            {'name': 'TestKept', 'status': 'PASSED'},
            {'name': 'TestLost', 'status': 'FAILED'},
        ],
    )

    report = _report(instance, test_output_path)

    assert _compute_partial_score(report) == pytest.approx(1 / 6)


def test_pro_go_enrichment_adds_passing_retry_without_overwriting_parser_failure() -> None:
    instance = {
        'instance_id': 'instance_gravitational__teleport-case',
        'repo_language': 'go',
    }
    test_output = {'tests': [{'name': 'TestProxySSHDial', 'status': 'FAILED'}]}
    stdout_log = """=== RUN   TestProxySSHDial
--- FAIL: TestProxySSHDial (0.42s)
=== RUN   TestProxySSHDial
--- PASS: TestProxySSHDial (0.38s)
PASS
"""

    enriched = _enrich_test_output(instance, test_output, stdout_log, '')

    assert enriched['tests'] == [
        {'name': 'TestProxySSHDial', 'status': 'FAILED'},
        {'name': 'TestProxySSHDial', 'status': 'PASSED'},
    ]
    assert enriched['pro_parser_fallback']['go_test_statuses'] == 2
    assert '--- FAIL: TestProxySSHDial' in enriched['pro_raw_logs']['stdout_tail']


def test_pro_go_enrichment_adds_failed_tests_missing_from_parser() -> None:
    instance = {
        'instance_id': 'instance_flipt-io__flipt-case',
        'repo_language': 'go',
    }
    test_output = {'tests': []}
    stdout_log = """=== RUN   Test_SourceSelfSignedSkipTLS
--- FAIL: Test_SourceSelfSignedSkipTLS (0.13s)
FAIL
"""

    enriched = _enrich_test_output(instance, test_output, stdout_log, '')

    assert enriched['tests'] == [{'name': 'Test_SourceSelfSignedSkipTLS', 'status': 'FAILED'}]
    assert enriched['pro_parser_fallback']['go_test_statuses'] == 1
    assert enriched['pro_raw_logs']['stdout_tail'].endswith('FAIL\n')


def test_pro_go_enrichment_preserves_duplicate_parser_statuses() -> None:
    instance = {
        'instance_id': 'instance_go__case',
        'repo_language': 'go',
    }
    test_output = {
        'tests': [
            {'name': 'TestFlaky', 'status': 'PASSED'},
            {'name': 'TestFlaky', 'status': 'FAILED'},
            {'name': 'TestOther', 'status': 'PASSED'},
        ],
    }
    stdout_log = """=== RUN   TestFlaky
--- FAIL: TestFlaky (0.01s)
"""

    enriched = _enrich_test_output(instance, test_output, stdout_log, '')

    assert enriched['tests'] == test_output['tests']
    assert enriched['pro_parser_fallback']['go_test_statuses'] == 1


def test_pro_go_enrichment_adds_raw_pass_without_replacing_parser_skip() -> None:
    instance = {
        'instance_id': 'instance_go__case',
        'repo_language': 'go',
    }
    test_output = {'tests': [{'name': 'TestRecovered', 'status': 'SKIPPED'}]}
    stdout_log = """=== RUN   TestRecovered
--- PASS: TestRecovered (0.01s)
"""

    enriched = _enrich_test_output(instance, test_output, stdout_log, '')

    assert enriched['tests'] == [
        {'name': 'TestRecovered', 'status': 'SKIPPED'},
        {'name': 'TestRecovered', 'status': 'PASSED'},
    ]


def test_pro_go_enrichment_does_not_duplicate_existing_parser_failure() -> None:
    instance = {
        'instance_id': 'instance_go__case',
        'repo_language': 'go',
    }
    test_output = {'tests': [{'name': 'TestBroken', 'status': 'FAILED'}]}
    stdout_log = """=== RUN   TestBroken
--- FAIL: TestBroken (0.01s)
"""

    enriched = _enrich_test_output(instance, test_output, stdout_log, '')

    assert enriched['tests'] == test_output['tests']
    assert enriched['pro_parser_fallback']['go_test_statuses'] == 1


def test_pro_go_status_parser_accepts_indented_subtests() -> None:
    instance = {
        'instance_id': 'instance_go__case',
        'repo_language': 'go',
    }
    test_output = {'tests': []}
    stdout_log = """=== RUN   TestParent/child
    --- PASS: TestParent/child (0.01s)
"""

    enriched = _enrich_test_output(instance, test_output, stdout_log, '')

    assert enriched['tests'] == [{'name': 'TestParent/child', 'status': 'PASSED'}]


def test_pro_go_enrichment_does_not_let_later_package_skip_override_pass() -> None:
    instance = {
        'instance_id': 'instance_flipt-io__flipt-dbe263961b187e1c5d7fe34c65b000985a2da5a0',
        'repo_language': 'go',
    }
    test_output = {
        'tests': [
            {'name': 'Test_Store_String', 'status': 'PASSED'},
            {'name': 'Test_Store', 'status': 'PASSED'},
            {'name': 'Test_Store_String', 'status': 'PASSED'},
        ],
    }
    stdout_log = """=== RUN   Test_Store_String
--- PASS: Test_Store_String (0.00s)
=== RUN   Test_Store
--- PASS: Test_Store (1.01s)
=== RUN   Test_Store_String
--- PASS: Test_Store_String (0.00s)
=== RUN   Test_Store
--- SKIP: Test_Store (0.00s)
"""

    enriched = _enrich_test_output(instance, test_output, stdout_log, '')

    assert enriched['tests'] == test_output['tests']
    assert enriched['pro_raw_logs']['stdout_tail'] == stdout_log


def test_pro_go_enrichment_does_not_let_flipt_package_skips_override_passes() -> None:
    instance = {
        'instance_id': 'instance_flipt-io__flipt-e5fe37c379e1eec2dd3492c5737c0be761050b26',
        'repo_language': 'go',
    }
    test_output = {
        'tests': [
            {'name': 'Test_SourceString', 'status': 'PASSED'},
            {'name': 'Test_SourceString', 'status': 'PASSED'},
            {'name': 'Test_SourceGet', 'status': 'PASSED'},
            {'name': 'Test_SourceSubscribe', 'status': 'PASSED'},
            {'name': 'Test_SourceString', 'status': 'PASSED'},
            {'name': 'Test_SourceGet', 'status': 'PASSED'},
            {'name': 'Test_SourceSubscribe', 'status': 'PASSED'},
            {'name': 'Test_SourceString', 'status': 'PASSED'},
        ],
    }
    stdout_log = """=== RUN   Test_SourceString
--- PASS: Test_SourceString (0.00s)
=== RUN   Test_SourceGet
--- SKIP: Test_SourceGet (0.00s)
=== RUN   Test_SourceSubscribe
--- SKIP: Test_SourceSubscribe (0.00s)
=== RUN   Test_SourceString
--- PASS: Test_SourceString (0.00s)
=== RUN   Test_SourceGet
--- PASS: Test_SourceGet (0.00s)
=== RUN   Test_SourceSubscribe
--- PASS: Test_SourceSubscribe (5.01s)
=== RUN   Test_SourceString
--- PASS: Test_SourceString (0.00s)
=== RUN   Test_SourceGet
--- PASS: Test_SourceGet (0.01s)
=== RUN   Test_SourceSubscribe
--- PASS: Test_SourceSubscribe (1.02s)
=== RUN   Test_SourceString
--- PASS: Test_SourceString (0.00s)
=== RUN   Test_SourceGet
--- SKIP: Test_SourceGet (0.00s)
=== RUN   Test_SourceSubscribe
--- SKIP: Test_SourceSubscribe (0.00s)
"""

    enriched = _enrich_test_output(instance, test_output, stdout_log, '')

    assert enriched['tests'] == test_output['tests']
    assert enriched['pro_raw_logs']['stdout_tail'] == stdout_log


def test_pro_go_enrichment_leaves_clean_passing_output_unlabeled() -> None:
    instance = {
        'instance_id': 'instance_go__case',
        'repo_language': 'go',
    }
    test_output = {'tests': [{'name': 'TestKept', 'status': 'PASSED'}]}
    stdout_log = """=== RUN   TestKept
--- PASS: TestKept (0.01s)
PASS
"""

    enriched = _enrich_test_output(instance, test_output, stdout_log, '')

    assert enriched['tests'] == [{'name': 'TestKept', 'status': 'PASSED'}]
    assert enriched['pro_raw_logs']['stdout_tail'] == stdout_log


def test_pro_go_enrichment_is_scoped_to_go_instances() -> None:
    instance = {
        'instance_id': 'instance_NodeBB__NodeBB-case',
        'repo_language': 'js',
    }
    test_output = {'tests': []}
    stdout_log = """=== RUN   TestThatLooksLikeGo
--- PASS: TestThatLooksLikeGo (0.01s)
"""

    enriched = _enrich_test_output(instance, test_output, stdout_log, '')

    assert enriched['tests'] == []
    assert enriched['pro_raw_logs']['stdout_tail'] == stdout_log
