import ast
import base64
import hashlib
import io
import json
import tarfile

from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from nexus.extensions.tasks.swe.constants import SWEbenchInstance
from nexus.extensions.tasks.swe.swe_bench_pro import dataset_handler, nodebb_registry_fixture


_TARGET_INSTANCE_ID = (
    'instance_NodeBB__NodeBB-76c6e30282906ac664f2c9278fc90999b27b1f48-vd59a5728dfc977f44533186ace531248c2917516'
)
_TARGET_IMAGE = (
    'mirrors.tencent.com/hunyuan_yanguan/'
    'sweap-images-instance_nodebb__nodebb-76c6e30282906ac664f2c9278fc90999b27b1f48-'
    'vd59a5728dfc977f44533186ace531248c2917516:latest'
)
_TEST_PATCH = """diff --git a/test/plugins.js b/test/plugins.js
index 1260f8fd7118..582823b33a09 100644
--- a/test/plugins.js
+++ b/test/plugins.js
@@ -262,6 +262,18 @@ describe('Plugins', () => {
+		it('should error if plugin id is invalid', async () => {
+			await assert.rejects(
+				plugins.toggleActive('\\t\\nnodebb-plugin'),
+				{ message: '[[error:invalid-plugin-id]]' }
+			);
+
+			await assert.rejects(
+				plugins.toggleActive('notaplugin'),
+				{ message: '[[error:invalid-plugin-id]]' }
+			);
+		});
"""
_BEFORE_REPO_SET_CMD = """git reset --hard a3e1a666b876e0b3ccbb5284dd826c8c90c113b4
git clean -fd
git checkout a3e1a666b876e0b3ccbb5284dd826c8c90c113b4
git checkout 76c6e30282906ac664f2c9278fc90999b27b1f48 -- test/plugins.js"""
_FAIL_TO_PASS = [
    'test/plugins.js | Plugins install/activate/uninstall should error if plugin id is invalid',
]
_PASS_TO_PASS = [
    'test/plugins.js | Plugins should get plugin data from nbbpm',
    'test/plugins.js | Plugins should get a list of plugins',
    'test/plugins.js | Plugins install/activate/uninstall should install a plugin',
    'test/plugins.js | Plugins install/activate/uninstall should activate plugin',
    'test/plugins.js | Plugins install/activate/uninstall should upgrade plugin',
    'test/plugins.js | Plugins install/activate/uninstall should uninstall a plugin',
]
_EXPECTED_BUNDLE_MEMBERS = {
    'ajv-6.15.0.tgz',
    'async-0.2.10.tgz',
    'colors-0.6.2.tgz',
    'nodebb-plugin-imgur-1.0.16.tgz',
    'nodebb-plugin-imgur-5.0.0.tgz',
    'pkginfo-0.3.1.tgz',
    'qs-6.5.5.tgz',
    'winston-0.8.3.tgz',
}


def _sha256(value: str | bytes) -> str:
    if isinstance(value, str):
        value = value.encode()
    return hashlib.sha256(value).hexdigest()


def _parse_tests(value: str | list[str]) -> list[str]:
    if isinstance(value, str):
        return [str(item) for item in ast.literal_eval(value)]
    return [str(item) for item in value]


def _canonical_test_hash(value: str | list[str]) -> str:
    payload = json.dumps(_parse_tests(value), ensure_ascii=False, separators=(',', ':'))
    return _sha256(payload)


def _typed(instance: dict) -> SWEbenchInstance:
    return cast(SWEbenchInstance, instance)


def _instance(*, instance_id: str = _TARGET_INSTANCE_ID) -> dict:
    return {
        'instance_id': instance_id,
        'repo': 'NodeBB/NodeBB',
        'dataset_type': 'swe-bench_pro',
        'base_commit': 'a3e1a666b876e0b3ccbb5284dd826c8c90c113b4',
        'workdir': '/app',
        'repo_language': 'js',
        'image': _TARGET_IMAGE,
        'patch': 'candidate patch must not be part of the fixture guard',
        'test_patch': _TEST_PATCH,
        'before_repo_set_cmd': _BEFORE_REPO_SET_CMD,
        'selected_test_files_to_run': ['test/i18n.js', 'test/plugins.js'],
        'FAIL_TO_PASS': repr(_FAIL_TO_PASS),
        'PASS_TO_PASS': json.dumps(_PASS_TO_PASS),
        'spec_dict': {
            'baseDockerfile': 'FROM node:22\nENV SETUP={}\n',
            'instanceDockerfile': 'FROM nodebb-base\nENV UV_HTTP_TIMEOUT=60\n',
            'parser.py': 'print("parse")\n',
            'run_script.sh': '#!/bin/bash\nset -e\necho run tests\n',
        },
    }


def _case_for(instance: dict) -> nodebb_registry_fixture.NodeBBRegistryCase:
    specs = instance['spec_dict']
    return nodebb_registry_fixture.NodeBBRegistryCase(
        instance_id=instance['instance_id'].lower(),
        repo=instance['repo'],
        dataset_type=instance['dataset_type'],
        base_commit=instance['base_commit'],
        workdir=instance['workdir'],
        repo_language=instance['repo_language'],
        image=instance['image'],
        selected_tests=tuple(instance['selected_test_files_to_run']),
        test_patch_sha256=_sha256(instance['test_patch']),
        before_repo_set_cmd_sha256=_sha256(instance['before_repo_set_cmd']),
        fail_to_pass_count=len(_parse_tests(instance['FAIL_TO_PASS'])),
        fail_to_pass_sha256=_canonical_test_hash(instance['FAIL_TO_PASS']),
        pass_to_pass_count=len(_parse_tests(instance['PASS_TO_PASS'])),
        pass_to_pass_sha256=_canonical_test_hash(instance['PASS_TO_PASS']),
        spec_sha256=tuple(sorted((name, _sha256(content)) for name, content in specs.items())),
        test_file_sha256='b9344d41e89dfc82b2e6e8e80b0c767a33b553fedf05ee16de0b0c0918ac1d5f',
        node_version='v22.16.0',
        npm_version='10.9.2',
        nodebb_version='3.6.3',
    )


@pytest.fixture
def target_case(monkeypatch: pytest.MonkeyPatch) -> dict:
    instance = _instance()
    monkeypatch.setattr(nodebb_registry_fixture, 'NODEBB_REGISTRY_CASE', _case_for(instance))
    return instance


@pytest.fixture
def dockerfiles(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dataset_handler, '_load_base_docker', lambda _: '')
    monkeypatch.setattr(dataset_handler, '_instance_docker', lambda _: '')


def test_nodebb_registry_fixture_is_exact_case_and_preserves_real_test_commands(
    target_case: dict,
    dockerfiles: None,
) -> None:
    path = f'/tmp/swebench/{_TARGET_INSTANCE_ID.lower()}'

    script = dataset_handler.get_eval_script(_typed(target_case))
    files = dataset_handler._build_eval_workspace_files(_typed(target_case), path)
    workspace_paths = {item.path for item in files}

    marker = nodebb_registry_fixture.NODEBB_REGISTRY_FIXTURE_MARKER
    checkout = 'git checkout 76c6e30282906ac664f2c9278fc90999b27b1f48 -- test/plugins.js'
    assert script.count(marker) == 1
    assert script.index(checkout) < script.index(marker)
    assert f'bash {path}/run_script.sh test/i18n.js,test/plugins.js' in script
    assert f'python {path}/parser.py {path}/stdout.log {path}/stderr.log {path}/test_output.txt' in script
    assert 'nodebb_test_rc=$?' in script
    assert 'NPM_CONFIG_USERCONFIG' in script
    assert 'npm_config_registry=' in script
    assert 'config.json.before-nodebb-registry' in script
    assert 'cp "$nodebb_registry_config_backup" config.json' in script
    assert '127.0.0.1' in script
    assert '--port-file' in script
    assert 'rm -f' in script
    assert 'test_output.txt' in script
    assert 'test-return-code' in script
    assert 'cleanup-ok' in script
    assert '/etc/hosts' not in script
    assert 'npm config set' not in script
    assert 'HTTPS_PROXY=' not in script
    assert 'pytest.skip' not in script
    assert 'xfail' not in script
    assert 'cat >> test/plugins.js' not in script
    assert 'sed -i' not in script
    assert 'python3 - test/plugins.js' not in script
    assert f'{path}/{nodebb_registry_fixture.NODEBB_REGISTRY_SERVER_FILENAME}' in workspace_paths
    assert f'{path}/{nodebb_registry_fixture.NODEBB_REGISTRY_MANIFEST_FILENAME}' in workspace_paths
    assert f'{path}/{nodebb_registry_fixture.NODEBB_REGISTRY_ASSET_BUNDLE_FILENAME}' in workspace_paths


@pytest.mark.parametrize(
    ('field', 'replacement'),
    [
        ('repo', 'NodeBB/Other'),
        ('dataset_type', 'swe-bench'),
        ('base_commit', '0' * 40),
        ('workdir', '/testbed'),
        ('repo_language', 'ts'),
        ('image', _TARGET_IMAGE.replace(':latest', ':changed')),
        ('selected_test_files_to_run', ['test/plugins.js']),
        ('test_patch', _TEST_PATCH + '\n'),
        ('before_repo_set_cmd', _BEFORE_REPO_SET_CMD.replace('test/plugins.js', 'test/other.js')),
        ('FAIL_TO_PASS', repr(_FAIL_TO_PASS + ['unexpected F2P'])),
        ('PASS_TO_PASS', json.dumps(_PASS_TO_PASS + ['unexpected P2P'])),
        ('spec_dict', {'baseDockerfile': 'changed'}),
    ],
)
def test_nodebb_registry_fixture_fails_closed_on_target_dataset_drift(
    target_case: dict,
    field: str,
    replacement: object,
) -> None:
    target_case[field] = replacement

    with pytest.raises(ValueError, match='NodeBB registry fixture'):
        nodebb_registry_fixture.needs_nodebb_registry_fixture(target_case)


def test_nodebb_registry_fixture_does_not_guard_candidate_patch(target_case: dict) -> None:
    assert nodebb_registry_fixture.needs_nodebb_registry_fixture(target_case)

    target_case['patch'] = 'different candidate or oracle patch'

    assert nodebb_registry_fixture.needs_nodebb_registry_fixture(target_case)


def test_nodebb_registry_fixture_does_not_leak_to_other_nodebb_instances(
    target_case: dict,
    dockerfiles: None,
) -> None:
    target_case['instance_id'] = 'instance_NodeBB__NodeBB-another-case-vd59a5728dfc977f44533186ace531248c2917516'
    path = f'/tmp/swebench/{target_case["instance_id"].lower()}'

    script = dataset_handler.get_eval_script(_typed(target_case))
    files = dataset_handler._build_eval_workspace_files(_typed(target_case), path)

    assert nodebb_registry_fixture.NODEBB_REGISTRY_FIXTURE_MARKER not in script
    assert all('nodebb_registry' not in item.path for item in files)


def test_nodebb_registry_fixture_case_object_is_immutable(target_case: dict) -> None:
    case = _case_for(target_case)

    with pytest.raises((AttributeError, TypeError)):
        setattr(case, ''.join(('re', 'po')), 'changed')

    assert replace(case, repo='changed').repo == 'changed'
    assert case.repo == 'NodeBB/NodeBB'


def test_nodebb_registry_assets_are_content_addressed_and_protocol_is_fail_closed(
    target_case: dict,
) -> None:
    path = f'/tmp/swebench/{_TARGET_INSTANCE_ID.lower()}'
    files = nodebb_registry_fixture.get_nodebb_registry_fixture_files(target_case, path)
    by_name = {item.path.rsplit('/', 1)[-1]: item for item in files}

    bundle_item = by_name[nodebb_registry_fixture.NODEBB_REGISTRY_ASSET_BUNDLE_FILENAME]
    assert isinstance(bundle_item.content, str)
    bundle = base64.b64decode(bundle_item.content)
    assert bundle_item.encoding == 'base64'
    assert _sha256(bundle) == nodebb_registry_fixture.NODEBB_REGISTRY_ASSET_BUNDLE_SHA256
    with tarfile.open(fileobj=io.BytesIO(bundle), mode='r:gz') as archive:
        assert set(archive.getnames()) == _EXPECTED_BUNDLE_MEMBERS

    manifest_content = by_name[nodebb_registry_fixture.NODEBB_REGISTRY_MANIFEST_FILENAME].content
    assert isinstance(manifest_content, str)
    manifest = json.loads(manifest_content)
    versions = manifest['versions']
    assert len(versions) == 58
    assert manifest['latest']['nodebb-plugin-imgur'] == '5.0.0'
    assert versions['nodebb-plugin-imgur@1.0.16']['source'] == 'bundle'
    assert versions['nodebb-plugin-imgur@5.0.0']['source'] == 'bundle'
    assert versions['sass@1.69.7']['source'] == 'cache'
    assert versions['sass-embedded-linux-x64@1.69.7']['source'] == 'cache'
    assert sum(item['source'] == 'bundle' for item in versions.values()) == len(_EXPECTED_BUNDLE_MEMBERS)

    server = by_name[nodebb_registry_fixture.NODEBB_REGISTRY_SERVER_FILENAME].content
    assert isinstance(server, str)
    assert "ThreadingHTTPServer(('127.0.0.1', 0)" in server
    assert "'/__health'" in server
    assert "'/__audit'" in server
    assert 'violations' in server
    assert 'unknown route' in server
    assert 'packages.nodebb.org' not in server
    assert 'registry.npmjs.org' not in server
    assert 'mirrors.tencent.com' not in server


def test_nodebb_registry_fixture_generation_is_idempotent(
    target_case: dict,
    dockerfiles: None,
) -> None:
    path = f'/tmp/swebench/{_TARGET_INSTANCE_ID.lower()}'

    first_script = dataset_handler.get_eval_script(_typed(target_case))
    second_script = dataset_handler.get_eval_script(_typed(target_case))
    first_files = dataset_handler._build_eval_workspace_files(_typed(target_case), path)
    second_files = dataset_handler._build_eval_workspace_files(_typed(target_case), path)

    assert first_script == second_script
    assert [item.model_dump() for item in first_files] == [item.model_dump() for item in second_files]


def test_nodebb_registry_audit_is_persisted_without_changing_test_statuses(
    target_case: dict,
    tmp_path: Path,
) -> None:
    runtime_dir = tmp_path / 'nodebb_registry_runtime'
    runtime_dir.mkdir()
    requests = [
        {'method': 'GET', 'path': '/api/v1/plugins/nodebb-plugin-markdown', 'query': ''},
        {'method': 'GET', 'path': '/nodebb-plugin-imgur', 'query': ''},
    ]
    (runtime_dir / 'audit.json').write_text(json.dumps({'requests': requests, 'violations': []}))
    (runtime_dir / 'test-return-code').write_text('93\n')
    (runtime_dir / 'cleanup-ok').write_text('ok\n')
    test_output = {'tests': [{'name': 'kept test', 'status': 'PASSED'}]}

    enriched = nodebb_registry_fixture.enrich_nodebb_registry_test_output(
        target_case,
        test_output,
        tmp_path,
    )

    assert test_output == {'tests': [{'name': 'kept test', 'status': 'PASSED'}]}
    assert enriched['tests'] == test_output['tests']
    assert enriched['pro_fixture_audit'] == {
        'name': 'nodebb_registry',
        'request_count': 2,
        'requests': requests,
        'violations': [],
        'test_return_code': 93,
        'cleanup': 'ok',
    }


@pytest.mark.parametrize(
    ('audit', 'return_code', 'cleanup'),
    [
        ({'requests': [], 'violations': [{'reason': 'unexpected route'}]}, '93\n', 'ok\n'),
        ({'requests': [], 'violations': []}, 'not-an-int\n', 'ok\n'),
        ({'requests': [], 'violations': []}, '93\n', 'missing\n'),
    ],
)
def test_nodebb_registry_audit_persistence_fails_closed(
    target_case: dict,
    tmp_path: Path,
    audit: dict,
    return_code: str,
    cleanup: str,
) -> None:
    runtime_dir = tmp_path / 'nodebb_registry_runtime'
    runtime_dir.mkdir()
    (runtime_dir / 'audit.json').write_text(json.dumps(audit))
    (runtime_dir / 'test-return-code').write_text(return_code)
    if cleanup != 'missing\n':
        (runtime_dir / 'cleanup-ok').write_text(cleanup)

    with pytest.raises(ValueError, match='NodeBB registry fixture audit'):
        nodebb_registry_fixture.enrich_nodebb_registry_test_output(
            target_case,
            {'tests': []},
            tmp_path,
        )
