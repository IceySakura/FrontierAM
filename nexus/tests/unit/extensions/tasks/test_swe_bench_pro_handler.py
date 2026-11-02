import pytest

from nexus.extensions.tasks.swe.swe_bench_pro import dataset_handler


_FLIPT_INSTANCE_ID = 'instance_flipt-io__flipt-5aef5a14890aa145c22d864a834694bae3a6f112'
_FLIPT_TEST = 'Test_SourceSelfSignedSkipTLS'


def _instance(
    *,
    instance_id: str = _FLIPT_INSTANCE_ID,
    selected_tests: str | list[str] = f"['{_FLIPT_TEST}']",
) -> dict:
    return {
        'instance_id': instance_id,
        'repo': 'flipt-io/flipt',
        'base_commit': '3e8ab3fdbd7b3bf14e1db6443d78bc530743d8d0',
        'repo_language': 'go',
        'selected_test_files_to_run': selected_tests,
        'before_repo_set_cmd': (
            'git reset --hard 3e8ab3fdbd7b3bf14e1db6443d78bc530743d8d0\n'
            'git checkout 5aef5a14890aa145c22d864a834694bae3a6f112 -- internal/storage/fs/git/source_test.go'
        ),
        'spec_dict': {
            'run_script.sh': '#!/bin/bash\necho run tests\n',
            'parser.py': 'print("parse")\n',
        },
    }


@pytest.fixture
def dockerfiles(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dataset_handler, '_load_base_docker', lambda _: '')
    monkeypatch.setattr(dataset_handler, '_instance_docker', lambda _: '')


def test_flipt_self_signed_tls_mock_is_injected_for_exact_case_and_test(dockerfiles: None) -> None:
    instance = _instance()

    script = dataset_handler.get_eval_script(instance)
    workspace_files = dataset_handler._build_eval_workspace_files(instance, f'/tmp/swebench/{_FLIPT_INSTANCE_ID}')
    workspace_paths = {item.path for item in workspace_files}

    assert '# SWE-bench-pro flipt local self-signed TLS fixture' in script
    assert '127.0.0.1 self-signed.badssl.com' in script
    assert 'NO_PROXY' in script
    assert 'no_proxy' in script
    assert 'flipt_self_signed_tls_mock.go' in script
    assert f'bash /tmp/swebench/{_FLIPT_INSTANCE_ID}/run_script.sh {_FLIPT_TEST}' in script
    assert f'/tmp/swebench/{_FLIPT_INSTANCE_ID}/flipt_self_signed_tls_mock.go' in workspace_paths


def test_flipt_self_signed_tls_mock_is_not_injected_for_other_test(dockerfiles: None) -> None:
    instance = _instance(selected_tests="['TestOther']")

    script = dataset_handler.get_eval_script(instance)
    workspace_files = dataset_handler._build_eval_workspace_files(instance, f'/tmp/swebench/{_FLIPT_INSTANCE_ID}')

    assert 'flipt_self_signed_tls_mock.go' not in script
    assert all('flipt_self_signed' not in item.path for item in workspace_files)


def test_flipt_self_signed_tls_mock_is_not_injected_for_other_instance(dockerfiles: None) -> None:
    instance = _instance(instance_id='instance_flipt-io__flipt-another-case')

    script = dataset_handler.get_eval_script(instance)
    workspace_files = dataset_handler._build_eval_workspace_files(
        instance, '/tmp/swebench/instance_flipt-io__flipt-another-case'
    )

    assert 'flipt_self_signed_tls_mock.go' not in script
    assert all('flipt_self_signed' not in item.path for item in workspace_files)


def test_flipt_selected_tests_accept_real_lists(dockerfiles: None) -> None:
    instance = _instance(selected_tests=[_FLIPT_TEST])

    script = dataset_handler.get_eval_script(instance)

    assert 'flipt_self_signed_tls_mock.go' in script
    assert f'run_script.sh {_FLIPT_TEST}' in script
