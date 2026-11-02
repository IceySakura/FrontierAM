from types import SimpleNamespace
from typing import Any

import pytest

from pydantic import ValidationError

from nexus.core.agent import get_agent_gateway_scaffold_policy
from nexus.extensions.agents.workbuddy import agent, compat, installer


def test_workbuddy_gateway_auto_resolves_registered_workbuddy_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('NEXUS_GATEWAY_USE_GENERIC_POLICY', raising=False)
    assert get_agent_gateway_scaffold_policy('workbuddy', 'auto') == 'workbuddy'


def test_remote_package_requires_sha256() -> None:
    with pytest.raises(ValidationError, match='package_sha256 is required'):
        agent.WorkBuddyInput(
            user_prompt='fix moto',
            working_dir='/testbed',
            package_url='https://artifacts.example.test/workbuddy-5.7.0.tgz',
        )


@pytest.mark.asyncio
async def test_run_workbuddy_uploads_chunks_and_invokes(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    tarball = tmp_path / 'wb-5.7.0-linux-x64.tar'
    tarball.write_bytes(b'abcdefghij')
    uploaded: list[str] = []
    captured: dict[str, Any] = {}

    class Runtime:
        async def ws_connect(self) -> None:
            captured['connected'] = True

        async def upload_file(self, files, overwrite=True):
            uploaded.extend(item.path for item in files)
            return SimpleNamespace(results=[SimpleNamespace(success=True, message=None)])

    class Context:
        async def get_runtime(self):
            return Runtime()

    async def fake_call_runtime_agent(target, /, **kwargs):
        captured['target'] = target
        captured.update(kwargs)
        return {
            'success': True,
            'output': 'done',
            'stderr': '',
            'exit_code': 0,
            'execution_time': 1.5,
        }

    monkeypatch.setattr(agent, 'require_context', lambda: Context())
    monkeypatch.setattr(agent, 'iter_file_chunks', lambda path: installer.iter_file_chunks(path, chunk_bytes=4))
    monkeypatch.setattr(agent, 'call_runtime_agent', fake_call_runtime_agent)

    result = await agent.run_workbuddy(
        agent.WorkBuddyInput(
            user_prompt='fix moto',
            working_dir='/testbed',
            tarball_path=str(tarball),
        )
    )

    assert result.success is True
    assert result.output == 'done'
    assert captured['connected'] is True
    assert captured['target'].__name__ == 'setup_and_run_workbuddy'
    assert captured['working_dir'] == '/testbed'
    assert captured['effort'] == 'xhigh'
    assert captured['thinking'] is False
    assert captured['chat_template_kwargs'] is None
    assert captured['preset'] == 'code'
    assert captured['version'] == 'latest'
    assert captured['model_id'] == 'general-model'
    assert captured['json_output'] is True
    assert captured['remote_part_paths'] == [
        '/tmp/nexus/workbuddy/wb-linux-x64.tar.part000',
        '/tmp/nexus/workbuddy/wb-linux-x64.tar.part001',
        '/tmp/nexus/workbuddy/wb-linux-x64.tar.part002',
    ]
    assert uploaded == captured['remote_part_paths']
    # Routing vars are enforced runtime-side, so the host no longer seeds them here.
    assert captured['extra_env'] == {}


@pytest.mark.asyncio
async def test_run_workbuddy_uses_remote_package_without_host_upload(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class Runtime:
        async def ws_connect(self) -> None:
            captured['connected'] = True

        async def upload_file(self, files, overwrite=True):
            raise AssertionError('remote package installation must not upload host chunks')

    class Context:
        async def get_runtime(self):
            return Runtime()

    async def fake_call_runtime_agent(target, /, **kwargs):
        captured.update(kwargs)
        return {
            'success': True,
            'output': 'done',
            'stderr': '',
            'exit_code': 0,
            'execution_time': 1.0,
        }

    monkeypatch.delenv('WORKBUDDY_TARBALL', raising=False)
    monkeypatch.setattr(agent, 'require_context', lambda: Context())
    monkeypatch.setattr(agent, 'call_runtime_agent', fake_call_runtime_agent)

    result = await agent.run_workbuddy(
        agent.WorkBuddyInput(
            user_prompt='fix moto',
            working_dir='/testbed',
            package_url='https://artifacts.example.test/workbuddy-5.7.0.tgz',
            package_sha256='a' * 64,
        )
    )

    assert result.success is True
    assert captured['connected'] is True
    assert captured['remote_part_paths'] == []
    assert captured['package_url'] == 'https://artifacts.example.test/workbuddy-5.7.0.tgz'
    assert captured['package_sha256'] == 'a' * 64


@pytest.mark.asyncio
async def test_run_workbuddy_passes_host_npm_token_only_to_runtime_install(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class Runtime:
        async def ws_connect(self) -> None:
            return None

        async def upload_file(self, files, overwrite=True):
            raise AssertionError('npm installation must not upload the host tarball')

    class Context:
        async def get_runtime(self):
            return Runtime()

    async def fake_call_runtime_agent(target, /, **kwargs):
        captured.update(kwargs)
        return {'success': True, 'output': 'done', 'stderr': '', 'exit_code': 0, 'execution_time': 1.0}

    monkeypatch.setenv('WORKBUDDY_NPM_TOKEN', 'read-only-test-token')
    monkeypatch.setenv('WORKBUDDY_NPM_USERNAME', 'shared-reader')
    monkeypatch.setattr(agent, 'require_context', lambda: Context())
    monkeypatch.setattr(agent, 'call_runtime_agent', fake_call_runtime_agent)

    result = await agent.run_workbuddy(agent.WorkBuddyInput(user_prompt='ping', working_dir='/testbed'))

    assert result.success is True
    assert captured['npm_token'] == 'read-only-test-token'
    assert captured['npm_username'] == 'shared-reader'
    assert captured['remote_part_paths'] is None
    assert captured['extra_env'] == {}
    assert 'connected' not in captured


@pytest.mark.asyncio
async def test_run_workbuddy_rejects_empty_tarball(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    tarball = tmp_path / 'empty.tar'
    tarball.write_bytes(b'')

    class Runtime:
        async def ws_connect(self) -> None:
            return None

        async def upload_file(self, files, overwrite=True):
            raise AssertionError('empty tarball must not upload')

    class Context:
        async def get_runtime(self):
            return Runtime()

    monkeypatch.setattr(agent, 'require_context', lambda: Context())
    monkeypatch.delenv('WORKBUDDY_NPM_USERNAME', raising=False)
    monkeypatch.delenv('WORKBUDDY_NPM_TOKEN', raising=False)

    with pytest.raises(RuntimeError, match='empty'):
        await agent.run_workbuddy(
            agent.WorkBuddyInput(user_prompt='x', working_dir='/testbed', tarball_path=str(tarball))
        )


def _stub_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path, result: dict[str, Any]) -> None:
    tarball = tmp_path / 'wb-5.7.0-linux-x64.tar'
    tarball.write_bytes(b'abcd')

    class Runtime:
        async def ws_connect(self) -> None:
            return None

        async def upload_file(self, files, overwrite=True):
            return SimpleNamespace(results=[SimpleNamespace(success=True, message=None)])

    class Context:
        async def get_runtime(self):
            return Runtime()

    async def fake_call_runtime_agent(target, /, **kwargs):
        return result

    monkeypatch.setattr(agent, 'require_context', lambda: Context())
    monkeypatch.setattr(agent, 'call_runtime_agent', fake_call_runtime_agent)


@pytest.mark.asyncio
async def test_output_carries_the_turn_outcome(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    _stub_runtime(
        monkeypatch,
        tmp_path,
        {
            'success': True,
            'output': '{"stopReason":"end_turn"}',
            'stderr': '',
            'exit_code': 0,
            'execution_time': 2.0,
            'status': 'completed',
            'task_outcome': 'SUCCESS',
            'stop_reason': 'end_turn',
            'assistant_text': 'pong',
            'transcript_path': '/root/.workbuddy/projects/ws/abc.jsonl',
            'conversation_id': 'abc',
        },
    )

    result = await agent.run_workbuddy(
        agent.WorkBuddyInput(
            user_prompt='ping', working_dir='/testbed', tarball_path=str(tmp_path / 'wb-5.7.0-linux-x64.tar')
        )
    )

    assert result.success is True
    assert result.status == 'completed'
    assert result.task_outcome == 'SUCCESS'
    assert result.assistant_text == 'pong'
    assert result.conversation_id == 'abc'
    assert result.failure_kind is None


@pytest.mark.asyncio
async def test_runtime_failure_fields_reach_the_output(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    _stub_runtime(
        monkeypatch,
        tmp_path,
        {
            'success': False,
            'output': '',
            'stderr': 'boom',
            'exit_code': -1,
            'execution_time': 1.0,
            'status': 'error',
            'failure_kind': 'infrastructure',
            'failure_stage': 'install',
        },
    )

    result = await agent.run_workbuddy(
        agent.WorkBuddyInput(
            user_prompt='ping', working_dir='/testbed', tarball_path=str(tmp_path / 'wb-5.7.0-linux-x64.tar')
        )
    )

    assert result.success is False
    assert result.failure_kind == 'infrastructure'
    assert result.failure_stage == 'install'


@pytest.mark.asyncio
async def test_unclassified_failure_still_gets_attributed(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    _stub_runtime(
        monkeypatch,
        tmp_path,
        {'success': False, 'output': '', 'stderr': '', 'exit_code': 1, 'execution_time': 1.0},
    )

    result = await agent.run_workbuddy(
        agent.WorkBuddyInput(
            user_prompt='ping', working_dir='/testbed', tarball_path=str(tmp_path / 'wb-5.7.0-linux-x64.tar')
        )
    )

    assert result.failure_kind == 'agent'
    assert result.failure_stage == 'process'


@pytest.mark.parametrize('version', sorted(compat.VERIFIED_PERMISSION_MODE_PATCH_VERSIONS))
def test_permission_mode_patch_script_rewrites_verified_versions(tmp_path, version: str) -> None:
    file_patches = compat.PERMISSION_MODE_PATCH_SPECS[version]
    pkg_root = tmp_path / 'package'
    for relpath, rules in file_patches.items():
        target = pkg_root / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('\n'.join(old for _, old, _ in rules))

    patches = compat.build_compatibility_patches(version, 'fullAccess')
    assert patches['extra_cli_args'] == ['--permission-mode', 'fullAccess']
    script = patches['pre_start_scripts'][0]['code'].replace('/tmp/nexus/agents/workbuddy', str(tmp_path))
    exec(compile(script, f'<compat-{version}>', 'exec'), {})

    for relpath, rules in file_patches.items():
        patched = (pkg_root / relpath).read_text()
        for _, _, new in rules:
            assert new in patched


def test_permission_mode_patch_script_supports_regex_rules(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    pkg_root = tmp_path / 'package'
    (pkg_root / 'lib').mkdir(parents=True)
    target = pkg_root / 'lib/headless-main-AbCdEf12.js'
    target.write_text('const x = {cwd:process.cwd(),model:foo.model,};')

    monkeypatch.setitem(
        compat.PERMISSION_MODE_PATCH_SPECS,
        '9.9.9-test',
        {
            'lib/headless-main-*.js': [
                (
                    'regex',
                    r'cwd:process\.cwd\(\),model:([A-Za-z0-9_$]+)\.model,',
                    r'cwd:process.cwd(),permissionMode:\1.permissionMode,model:\1.model,',
                ),
            ],
        },
    )
    script = compat._permission_mode_patch_script('9.9.9-test').replace('/tmp/nexus/agents/workbuddy', str(tmp_path))
    exec(compile(script, '<compat-regex>', 'exec'), {})
    assert 'cwd:process.cwd(),permissionMode:foo.permissionMode,model:foo.model,' in target.read_text()


def test_build_compatibility_patches_rejects_unverified_version() -> None:
    with pytest.raises(ValueError, match='only been verified'):
        compat.build_compatibility_patches('9.9.9', 'fullAccess')


def test_patch_script_rejects_unverified_exact_version_with_guidance() -> None:
    with pytest.raises(ValueError, match='only been verified') as exc:
        compat._permission_mode_patch_script('9.9.9')
    assert 'pin WORKBUDDY_VERSION' in str(exc.value)


def _fake_install(tmp_path, version: str) -> None:
    pkg_root = tmp_path / 'package'
    pkg_root.mkdir(parents=True, exist_ok=True)
    (pkg_root / 'package.json').write_text(f'{{"version":"{version}"}}')
    for relpath, rules in compat.PERMISSION_MODE_PATCH_SPECS.get(version, {}).items():
        target = pkg_root / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('\n'.join(old for _, old, _ in rules))


def test_dist_tag_patch_picks_the_installed_release(tmp_path) -> None:
    _fake_install(tmp_path, '5.11.0-hotfix.202610272104')
    script = compat.build_compatibility_patches('latest', 'fullAccess')['pre_start_scripts'][0]['code']
    exec(compile(script.replace('/tmp/nexus/agents/workbuddy', str(tmp_path)), '<compat-latest>', 'exec'), {})
    assert 'permission-mode' in (tmp_path / 'package' / 'lib' / 'wb.js').read_text()


def test_dist_tag_patch_refuses_an_unverified_install(tmp_path) -> None:
    _fake_install(tmp_path, '9.9.9')
    script = compat.build_compatibility_patches('latest', 'fullAccess')['pre_start_scripts'][0]['code']
    with pytest.raises(RuntimeError, match="no verified spec for installed WorkBuddy '9.9.9'") as exc:
        exec(compile(script.replace('/tmp/nexus/agents/workbuddy', str(tmp_path)), '<compat-latest>', 'exec'), {})
    message = str(exc.value)
    assert 'pin WORKBUDDY_VERSION' in message
    assert all(version in message for version in compat.VERIFIED_PERMISSION_MODE_PATCH_VERSIONS)
