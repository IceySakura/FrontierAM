from typing import Any, cast

import pytest

from nexus.extensions.tasks.swe.constants import SWEbenchInstance
from nexus.extensions.tasks.swe.swe_bench_multilingual import test_spec
from nexus.extensions.tasks.swe.swe_bench_multilingual.constants import START_TEST_OUTPUT


def _instance(repo: str, version: str) -> SWEbenchInstance:
    return {
        'instance_id': f'{repo.replace("/", "__")}-{version}',
        'repo': repo,
        'version': version,
        'base_commit': '0000000000000000000000000000000000000000',
        'test_patch': (
            'diff --git a/tests/example.txt b/tests/example.txt\n'
            '--- a/tests/example.txt\n'
            '+++ b/tests/example.txt\n'
            '@@ -1 +1,2 @@\n'
            ' existing\n'
            '+added\n'
        ),
        'FAIL_TO_PASS': [],
        'PASS_TO_PASS': [],
    }


@pytest.fixture
def temporary_specs(monkeypatch: pytest.MonkeyPatch) -> dict[str, dict[str, dict[str, Any]]]:
    specs: dict[str, dict[str, dict[str, Any]]] = {
        'example/install': {
            '1': {
                'install': 'install-one',
                'test_cmd': ['run-tests'],
            },
        },
        'example/build': {
            '1': {
                'build': ['configure-build', 'compile-build'],
                'test_cmd': ['run-tests'],
            },
        },
        'example/both': {
            '1': {
                'install': ['install-one', 'install-two'],
                'build': 'compile-build',
                'test_cmd': ['run-tests'],
            },
        },
        'example/neither': {
            '1': {
                'test_cmd': ['run-tests'],
            },
        },
    }
    monkeypatch.setattr(test_spec, 'MAP_REPO_VERSION_TO_SPECS', specs)
    return specs


def test_command_helpers_keep_install_and_build_as_separate_phases(temporary_specs: object) -> None:
    assert test_spec.get_install_cmds(_instance('example/install', '1')) == ['install-one']
    assert test_spec.get_build_cmds(_instance('example/install', '1')) == []

    assert test_spec.get_install_cmds(_instance('example/build', '1')) == []
    assert test_spec.get_build_cmds(_instance('example/build', '1')) == ['configure-build', 'compile-build']

    assert test_spec.get_install_cmds(_instance('example/both', '1')) == ['install-one', 'install-two']
    assert test_spec.get_build_cmds(_instance('example/both', '1')) == ['compile-build']

    assert test_spec.get_install_cmds(_instance('example/neither', '1')) == []
    assert test_spec.get_build_cmds(_instance('example/neither', '1')) == []


def test_eval_script_runs_install_then_build_before_tests(temporary_specs: object) -> None:
    commands = test_spec.make_eval_script_list(_instance('example/both', '1'), 'testbed', '/testbed')

    apply_patch = next(index for index, command in enumerate(commands) if command.startswith('git apply'))
    fail_fast = commands.index('set -e')
    install_one = commands.index('install-one')
    install_two = commands.index('install-two')
    build = commands.index('compile-build')
    stop_fail_fast = commands.index('set +e')
    start = commands.index(f": '{START_TEST_OUTPUT}'")
    test = commands.index('run-tests')

    assert apply_patch < install_one < install_two < fail_fast < build < stop_fail_fast < start < test


def test_real_build_only_fmt_case_rebuilds_before_ctest() -> None:
    instance = _instance('fmtlib/fmt', '1683')

    commands = test_spec.make_eval_script_list(instance, 'testbed', '/testbed')

    assert 'cmake -B build -S .' in commands
    assert 'cmake --build build --parallel $(nproc) --target printf-test' in commands
    assert commands.index('cmake --build build --parallel $(nproc) --target printf-test') < commands.index(
        'ctest --test-dir build -V -R printf-test'
    )


@pytest.mark.parametrize('version', ['2005', '2006'])
def test_immutable_build_does_not_run_networked_bundle_stats(version: str) -> None:
    build_commands = test_spec.get_build_cmds(_instance('immutable-js/immutable-js', version))

    assert build_commands == ['npm run build:dist']


def test_unknown_spec_has_no_install_or_build_commands(temporary_specs: object) -> None:
    instance = _instance('example/missing', '1')

    assert test_spec.get_install_cmds(instance) == []
    assert test_spec.get_build_cmds(instance) == []


def test_install_only_script_keeps_existing_non_fail_fast_behavior(temporary_specs: object) -> None:
    commands = test_spec.make_eval_script_list(_instance('example/install', '1'), 'testbed', '/testbed')

    assert 'install-one' in commands
    assert 'set -e' not in commands
    assert 'set +e' not in commands


def test_non_string_command_values_are_normalized(monkeypatch: pytest.MonkeyPatch) -> None:
    specs = {
        'example/types': {
            '1': {
                'install': 7,
                'build': [8, 'nine'],
                'test_cmd': ['run-tests'],
            },
        },
    }
    monkeypatch.setattr(test_spec, 'MAP_REPO_VERSION_TO_SPECS', specs)
    instance = _instance('example/types', '1')

    assert test_spec.get_install_cmds(instance) == ['7']
    assert test_spec.get_build_cmds(instance) == ['8', 'nine']


def test_valkey_928_exact_case_still_bypasses_generic_build(monkeypatch: pytest.MonkeyPatch) -> None:
    instance = _instance('valkey-io/valkey', '928')
    cast(dict[str, Any], instance)['instance_id'] = 'valkey-io__valkey-928'
    monkeypatch.setattr(test_spec.valkey_928, 'is_target_instance', lambda candidate: candidate is instance)
    monkeypatch.setattr(
        test_spec.valkey_928,
        'make_eval_script_list',
        lambda candidate, repo_directory: ['exact-case-lifecycle'],
    )

    assert test_spec.make_eval_script_list(instance, 'testbed', '/testbed') == ['exact-case-lifecycle']
