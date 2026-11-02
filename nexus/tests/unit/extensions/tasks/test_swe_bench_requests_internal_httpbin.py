import copy
import dataclasses
import hashlib
import json

from typing import Any, cast

import pytest

from nexus.extensions.tasks.swe.constants import SWEbenchInstance
from nexus.extensions.tasks.swe.swe_bench import dataset_handler, requests_internal_httpbin
from nexus.extensions.tasks.swe.swe_bench.constants import (
    END_TEST_OUTPUT,
    SPECS_REQUESTS,
    START_TEST_OUTPUT,
)


_FIXTURE_MARKER = '# SWE-bench requests exact-case internal HTTPBin fixture'
_TEST_FILE = 'test_requests.py'
_TEST_PATCH = """diff --git a/test_requests.py b/test_requests.py
--- a/test_requests.py
+++ b/test_requests.py
@@ -1 +1,2 @@
 def test_placeholder():
+    pass
"""
_FAIL_TO_PASS = ['test_requests.py::RequestsTestCase::test_target']
_PASS_TO_PASS = ['test_requests.py::RequestsTestCase::test_regression']


def _canonical_status_sha256(tests: list[str]) -> str:
    payload = json.dumps(tests, separators=(',', ':'))
    return hashlib.sha256(payload.encode()).hexdigest()


def _instance(instance_id: str, *, lists_as_json: bool = False) -> SWEbenchInstance:
    case = requests_internal_httpbin._REQUESTS_INTERNAL_HTTPBIN_CASES[instance_id]
    instance: SWEbenchInstance = {
        'instance_id': instance_id,
        'repo': 'psf/requests',
        'dataset_type': 'swe-bench',
        'version': case.version,
        'base_commit': case.base_commit,
        'test_patch': _TEST_PATCH,
        'FAIL_TO_PASS': copy.deepcopy(_FAIL_TO_PASS),
        'PASS_TO_PASS': copy.deepcopy(_PASS_TO_PASS),
    }
    if lists_as_json:
        instance['FAIL_TO_PASS'] = json.dumps(instance['FAIL_TO_PASS'])
        instance['PASS_TO_PASS'] = json.dumps(instance['PASS_TO_PASS'])
    return instance


@pytest.fixture
def synthetic_case_specs(monkeypatch: pytest.MonkeyPatch) -> None:
    test_patch_sha256 = hashlib.sha256(_TEST_PATCH.encode()).hexdigest()
    fail_to_pass_sha256 = _canonical_status_sha256(_FAIL_TO_PASS)
    pass_to_pass_sha256 = _canonical_status_sha256(_PASS_TO_PASS)
    for instance_id, case in list(requests_internal_httpbin._REQUESTS_INTERNAL_HTTPBIN_CASES.items()):
        monkeypatch.setitem(
            requests_internal_httpbin._REQUESTS_INTERNAL_HTTPBIN_CASES,
            instance_id,
            dataclasses.replace(
                case,
                test_patch_sha256=test_patch_sha256,
                fail_to_pass_count=len(_FAIL_TO_PASS),
                fail_to_pass_sha256=fail_to_pass_sha256,
                pass_to_pass_count=len(_PASS_TO_PASS),
                pass_to_pass_sha256=pass_to_pass_sha256,
                post_test_file_sha256='0' * 64,
            ),
        )


@pytest.mark.parametrize(
    (
        'instance_id',
        'version',
        'base_commit',
        'test_patch_sha256',
        'f2p_count',
        'f2p_sha256',
        'p2p_count',
        'p2p_sha256',
    ),
    [
        (
            'psf__requests-1921',
            '2.3',
            '3c88e520da24ae6f736929a750876e7654accc3d',
            'a9542a0223dcb934125f74642468fb69c5cae8d5fa1602279a3745862db6252d',
            6,
            '62c407d559b8089a6cc3c3ee45e53ecede5d939683204f1a745d9a0263d6a168',
            107,
            '36d6ebd9ae18d706c395c0b815c8e5ca0d8d69dad086121d3a3cf943ea9b3d6d',
        ),
        (
            'psf__requests-2317',
            '2.4',
            '091991be0da19de9108dbe5e3752917fea3d7fdc',
            'ff283039874db19ea594bda54c2aa6812eb25fea9881e2517c4a143805cdb552',
            8,
            '36ad2bde8f7bd06bef39d763c8710605b8072eb2969ad3e5c37fc9be439e2c44',
            133,
            'cdce5ad360721a761c14706bd9038f4c0d3ec3b6619fc747099246ba2f851d10',
        ),
    ],
)
def test_requests_internal_httpbin_canonical_case_fingerprints_are_locked(
    instance_id: str,
    version: str,
    base_commit: str,
    test_patch_sha256: str,
    f2p_count: int,
    f2p_sha256: str,
    p2p_count: int,
    p2p_sha256: str,
) -> None:
    case = requests_internal_httpbin._REQUESTS_INTERNAL_HTTPBIN_CASES[instance_id]

    assert case.version == version
    assert case.base_commit == base_commit
    assert case.test_patch_sha256 == test_patch_sha256
    assert case.fail_to_pass_count == f2p_count
    assert case.fail_to_pass_sha256 == f2p_sha256
    assert case.pass_to_pass_count == p2p_count
    assert case.pass_to_pass_sha256 == p2p_sha256
    assert case.directives == (_TEST_FILE,)


def test_requests_internal_httpbin_post_patch_file_hashes_are_locked() -> None:
    cases = requests_internal_httpbin._REQUESTS_INTERNAL_HTTPBIN_CASES

    assert cases['psf__requests-1921'].post_test_file_sha256 == (
        '318da0521e0d056ccf97b64adee853e7d53244f7c82b4b8c7dd7c0ef72d50ae2'
    )
    assert cases['psf__requests-2317'].post_test_file_sha256 == (
        'fe816a0da3b4abd996014ef559a0ce7eeefcd0664d62a683924bd65840998b02'
    )


@pytest.mark.parametrize('instance_id', ['psf__requests-1921', 'psf__requests-2317'])
@pytest.mark.parametrize('lists_as_json', [False, True])
def test_requests_targets_use_exact_case_internal_httpbin_fixture(
    instance_id: str,
    lists_as_json: bool,
    synthetic_case_specs: None,
) -> None:
    script = dataset_handler.get_eval_script(_instance(instance_id, lists_as_json=lists_as_json))

    assert script.count(_FIXTURE_MARKER) == 1
    assert script.index('+    pass') < script.index(_FIXTURE_MARKER)
    assert script.index(_FIXTURE_MARKER) < script.index(f": '{START_TEST_OUTPUT}'")
    assert script.index(f": '{END_TEST_OUTPUT}'") < script.rindex(
        f'git checkout {requests_internal_httpbin._REQUESTS_INTERNAL_HTTPBIN_CASES[instance_id].base_commit} {_TEST_FILE}'
    )
    assert 'HTTPBIN_URL=http://hyrl-httpbin.woa.com/' in script
    assert 'REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt' in script
    assert f'pytest -rA {_TEST_FILE}' in script
    assert '-k ' not in script

    start_test_output = script.index(f": '{START_TEST_OUTPUT}'")
    end_test_output = script.index(f": '{END_TEST_OUTPUT}'")
    ca_cleanup = script.index(
        (
            f'git checkout {requests_internal_httpbin._REQUESTS_INTERNAL_HTTPBIN_CASES[instance_id].base_commit} '
            'requests/cacert.pem'
        ),
        end_test_output,
    )
    test_file_cleanup = script.rindex(
        f'git checkout {requests_internal_httpbin._REQUESTS_INTERNAL_HTTPBIN_CASES[instance_id].base_commit} {_TEST_FILE}'
    )
    assert end_test_output < ca_cleanup < test_file_cleanup

    test_command = script[start_test_output:end_test_output]
    assert 'sed -i' not in test_command
    assert 'env HTTPBIN_URL=' in test_command

    fixture = script[script.index(_FIXTURE_MARKER) : start_test_output]
    assert 'hashlib.sha256' in fixture
    assert 'os.replace' in fixture
    assert 'ca-certificates.crt' in fixture
    assert 'requests/cacert.pem' in fixture
    assert 'https://hyrl-httpbin.woa.com/status/301' in fixture
    assert 'https://httpbin.org/status/301' in fixture
    assert 'unexpected test_requests.py sha256' in fixture
    assert 'expected exactly one rewrite anchor' in fixture


def test_requests_1921_fixture_only_rewrites_the_https_redirect(
    synthetic_case_specs: None,
) -> None:
    script = dataset_handler.get_eval_script(_instance('psf__requests-1921'))
    fixture = script[script.index(_FIXTURE_MARKER) : script.index(f": '{START_TEST_OUTPUT}'")]

    assert 'https://hyrl-httpbin.woa.com/status/301' in fixture
    assert 'https://hyrl-httpbin.woa.com/redirect/5' not in fixture
    assert 'https://hyrl-httpbin.woa.com/delay/10' not in fixture
    assert 'socket.getaddrinfo' not in fixture
    assert 'test_headers_on_session_with_None_are_not_sent' not in fixture


def test_requests_2317_fixture_preserves_history_auth_timeout_and_encoded_method_semantics(
    synthetic_case_specs: None,
) -> None:
    script = dataset_handler.get_eval_script(_instance('psf__requests-2317'))
    fixture = script[script.index(_FIXTURE_MARKER) : script.index(f": '{START_TEST_OUTPUT}'")]

    assert 'https://hyrl-httpbin.woa.com/status/301' in fixture
    assert 'https://hyrl-httpbin.woa.com/redirect/5' in fixture
    assert 'https://hyrl-httpbin.woa.com/delay/10' in fixture
    assert 'socket.getaddrinfo' in fixture
    assert 'socket.AF_INET' in fixture
    assert 'http://{auth_target_ip}/get' in fixture
    assert "requests.request(b'GET', httpbin('get'))" not in fixture
    assert 'test_encoded_methods' not in fixture


@pytest.mark.parametrize(
    ('field', 'value'),
    [
        ('repo', 'django/django'),
        ('dataset_type', 'swe-bench-plus-plus'),
        ('version', '2.5'),
        ('base_commit', '0000000000000000000000000000000000000000'),
        ('test_patch', _TEST_PATCH + '\n'),
        ('FAIL_TO_PASS', []),
        ('FAIL_TO_PASS', [_FAIL_TO_PASS[0], _FAIL_TO_PASS[0]]),
        ('FAIL_TO_PASS', [1]),
        ('PASS_TO_PASS', []),
        ('PASS_TO_PASS', [_PASS_TO_PASS[0], _PASS_TO_PASS[0]]),
        ('PASS_TO_PASS', [1]),
        ('test_patch', _TEST_PATCH.replace('test_requests.py', 'test_other.py')),
    ],
)
def test_requests_target_metadata_drift_fails_closed(
    field: str,
    value: object,
    synthetic_case_specs: None,
) -> None:
    instance = _instance('psf__requests-2317')
    cast(dict[str, Any], instance)[field] = value

    with pytest.raises(ValueError, match='psf__requests-2317'):
        dataset_handler.get_eval_script(instance)


def test_requests_target_test_command_drift_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_case_specs: None,
) -> None:
    original_make_eval_script_list = dataset_handler.make_eval_script_list

    def make_drifted_eval_script_list(*args: object, **kwargs: object) -> list[str]:
        commands = original_make_eval_script_list(*args, **kwargs)
        end_test_output = commands.index(f": '{END_TEST_OUTPUT}'")
        commands[end_test_output - 1] = 'pytest -q test_requests.py'
        return commands

    monkeypatch.setattr(dataset_handler, 'make_eval_script_list', make_drifted_eval_script_list)

    with pytest.raises(ValueError, match='psf__requests-2317'):
        dataset_handler.get_eval_script(_instance('psf__requests-2317'))


@pytest.mark.parametrize(
    ('instance_id', 'version', 'base_commit'),
    [
        ('psf__requests-synthetic-23', '2.3', '0000000000000000000000000000000000000023'),
        ('psf__requests-synthetic-24', '2.4', '0000000000000000000000000000000000000024'),
        ('psf__requests-2931', '2.9', '5f7a3a74aab1625c2bb65f643197ee885e3da576'),
    ],
)
def test_requests_fixture_does_not_leak_to_non_target_cases(
    instance_id: str,
    version: str,
    base_commit: str,
    synthetic_case_specs: None,
) -> None:
    instance: SWEbenchInstance = {
        'instance_id': instance_id,
        'repo': 'psf/requests',
        'dataset_type': 'swe-bench',
        'version': version,
        'base_commit': base_commit,
        'test_patch': _TEST_PATCH,
        'FAIL_TO_PASS': copy.deepcopy(_FAIL_TO_PASS),
        'PASS_TO_PASS': copy.deepcopy(_PASS_TO_PASS),
    }

    before = dataset_handler.get_eval_script(instance)
    dataset_handler.get_eval_script(_instance('psf__requests-2317'))
    after = dataset_handler.get_eval_script(instance)

    assert before == after
    assert _FIXTURE_MARKER not in after
    assert 'HTTPBIN_URL=http://hyrl-httpbin.woa.com/' not in after
    assert 'REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt' not in after
    assert 'socket.getaddrinfo' not in after


def test_requests_20_legacy_workaround_is_unchanged() -> None:
    command = SPECS_REQUESTS['2.0']['test_cmd']

    assert 'http://hyrl-httpbin.woa.com/' in command
    assert 'requests/cacert.pem' in command
    assert 'pytest -rA' in command


@pytest.mark.parametrize('version', ['2.3', '2.4'])
def test_requests_23_and_24_no_longer_use_version_wide_httpbin_or_ca_workarounds(version: str) -> None:
    command = SPECS_REQUESTS[version]['test_cmd']

    assert 'hyrl-httpbin.woa.com' not in command
    assert 'requests/cacert.pem' not in command


@pytest.mark.parametrize('instance_id', ['psf__requests-1921', 'psf__requests-2317'])
def test_requests_fixture_generation_is_idempotent(
    instance_id: str,
    synthetic_case_specs: None,
) -> None:
    instance = _instance(instance_id)

    first = dataset_handler.get_eval_script(instance)
    second = dataset_handler.get_eval_script(instance)

    assert first == second
    assert first.count(_FIXTURE_MARKER) == 1


@pytest.mark.parametrize('instance_id', ['psf__requests-1921', 'psf__requests-2317'])
def test_requests_fixture_avoids_high_risk_global_network_changes(
    instance_id: str,
    synthetic_case_specs: None,
) -> None:
    script = dataset_handler.get_eval_script(_instance(instance_id))
    fixture = script[script.index(_FIXTURE_MARKER) : script.index(f": '{START_TEST_OUTPUT}'")]

    assert '/etc/hosts' not in fixture
    assert 'verify=False' not in fixture
    assert 'CERT_NONE' not in fixture
    assert 'Session.send' not in fixture
    assert 'HTTPAdapter.send' not in fixture
    assert 'monkeypatch' not in fixture
    assert 'socket.socket' not in fixture
    assert 'socket.getaddrinfo =' not in fixture
    assert 'pytest.skip' not in fixture
    assert 'xfail' not in fixture
    assert 'serve_forever' not in fixture
    assert 'subprocess.Popen' not in fixture
