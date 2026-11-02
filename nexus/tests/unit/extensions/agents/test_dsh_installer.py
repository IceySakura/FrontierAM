from __future__ import annotations

import json

from pathlib import Path
from types import SimpleNamespace

import pytest

from nexus.extensions.agents.dsh import installer
from nexus.extensions.agents.dsh.prebuilt import PrebuiltIntegrityError, RuntimeTarget
from nexus.runtime.executors.command import CommandRunStatus


def _materialize_install(prefix: Path, *, dsh_version: str = '0.1.0-rc.6', acp_version: str = '0.1.0-rc.6') -> None:
    dsh_dir = prefix / 'node_modules' / '@deepseek-ai' / 'dsh'
    acp_dir = prefix / 'node_modules' / '@deepseek-ai' / 'dsh-acp'
    (dsh_dir / 'lib').mkdir(parents=True)
    acp_dir.mkdir(parents=True)
    (dsh_dir / 'lib' / 'bin.js').write_text('')
    (dsh_dir / 'package.json').write_text(json.dumps({'version': dsh_version, 'bin': {'dsh': 'lib/bin.js'}}))
    (acp_dir / 'package.json').write_text(json.dumps({'version': acp_version}))
    (prefix / 'node_modules' / 'node-pty').mkdir()
    (prefix / 'package-lock.json').write_text('{}\n')


@pytest.mark.asyncio
async def test_install_dsh_installs_exact_requested_packages_with_lifecycle_scripts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    commands: list[str] = []

    def fake_binary(name: str) -> str | None:
        return {
            'node': '/runtime/node',
            'npm': '/runtime/npm',
            'make': '/usr/bin/make',
            'g++': '/usr/bin/g++',
        }.get(name)

    async def fake_run_command(request):
        commands.append(request.command)
        if request.command.endswith('--version'):
            return SimpleNamespace(status=CommandRunStatus.Finished, return_code=0, stdout='v22.20.0\n', stderr='')
        if ' install ' in request.command:
            prefix = Path(request.cwd)
            dependencies = json.loads((prefix / 'package.json').read_text())['dependencies']
            _materialize_install(
                prefix,
                dsh_version=dependencies['@deepseek-ai/dsh'],
                acp_version=dependencies['@deepseek-ai/dsh-acp'],
            )
        return SimpleNamespace(status=CommandRunStatus.Finished, return_code=0, stdout='', stderr='')

    monkeypatch.setattr(installer, 'find_runtime_binary', fake_binary)
    monkeypatch.setattr(installer, 'find_nexus_runtime_binary', fake_binary)
    monkeypatch.setattr('nexus.runtime.executors.command.run_command', fake_run_command)

    entry = await installer.install_dsh(
        str(tmp_path),
        version='1.2.3',
        acp_version='4.5.6-rc.1',
        prebuilt_base_url=None,
    )

    install_command = next(command for command in commands if ' install ' in command)
    install_root = installer.get_installation_root(entry)
    assert entry == install_root / 'node_modules' / '@deepseek-ai' / 'dsh' / 'lib' / 'bin.js'
    assert install_root.parent == tmp_path / 'installations'
    assert 'npm install' in install_command or '/runtime/npm install' in install_command
    assert '--ignore-scripts' not in install_command
    assert any('rebuild node-pty --build-from-source=true' in command for command in commands)
    assert any('node-pty' in command for command in commands)
    manifest = json.loads((install_root / 'package.json').read_text())
    assert manifest['dependencies'] == {
        '@deepseek-ai/dsh': '1.2.3',
        '@deepseek-ai/dsh-acp': '4.5.6-rc.1',
    }
    assert (install_root / 'package-lock.json').is_file()


@pytest.mark.asyncio
async def test_install_dsh_prefers_prebuilt_without_compiler(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    commands: list[str] = []
    target = RuntimeTarget('linux', 'x64', 'glibc', '2.35', 22)

    def fake_binary(name: str) -> str | None:
        return {'node': '/runtime/node', 'npm': '/runtime/npm'}.get(name)

    async def fake_run_command(request):
        commands.append(request.command)
        if request.command.endswith('--version'):
            return SimpleNamespace(status=CommandRunStatus.Finished, return_code=0, stdout='v22.20.0\n', stderr='')
        return SimpleNamespace(status=CommandRunStatus.Finished, return_code=0, stdout='', stderr='')

    def fake_download(_manifest, *, archive_path: Path, install_path: Path, timeout: float = 300.0) -> None:
        del archive_path, timeout
        install_path.mkdir()
        _materialize_install(install_path)

    monkeypatch.setattr(installer, 'find_runtime_binary', fake_binary)
    monkeypatch.setattr(installer, 'find_nexus_runtime_binary', fake_binary)
    monkeypatch.setattr(installer, 'detect_runtime_target', lambda *, node_major: target)
    monkeypatch.setattr(
        installer,
        'fetch_prebuilt_manifest',
        lambda *_args, **_kwargs: SimpleNamespace(
            archive='runtime.tar.gz',
            archive_url='https://artifacts.test/runtime.tar.gz',
        ),
    )
    monkeypatch.setattr(installer, 'download_and_extract_prebuilt', fake_download)
    monkeypatch.setattr('nexus.runtime.executors.command.run_command', fake_run_command)

    entry = await installer.install_dsh(str(tmp_path), prebuilt_base_url='https://artifacts.test/v1')

    assert entry.is_file()
    assert not any(' npm ' in f' {command} ' or command.startswith('/runtime/npm ') for command in commands)
    assert installer.get_installed_versions(installer.get_installation_root(entry)) == (
        '0.1.0-rc.6',
        '0.1.0-rc.6',
    )


@pytest.mark.asyncio
async def test_install_dsh_falls_back_to_npm_when_prebuilt_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    commands: list[str] = []
    target = RuntimeTarget('linux', 'x64', 'glibc', '2.35', 22)

    def fake_binary(name: str) -> str | None:
        return {
            'node': '/runtime/node',
            'npm': '/runtime/npm',
            'make': '/usr/bin/make',
            'g++': '/usr/bin/g++',
        }.get(name)

    async def fake_run_command(request):
        commands.append(request.command)
        if request.command.endswith('--version'):
            return SimpleNamespace(status=CommandRunStatus.Finished, return_code=0, stdout='v22.20.0\n', stderr='')
        if ' install ' in request.command:
            _materialize_install(Path(request.cwd))
        return SimpleNamespace(status=CommandRunStatus.Finished, return_code=0, stdout='', stderr='')

    monkeypatch.setattr(installer, 'find_runtime_binary', fake_binary)
    monkeypatch.setattr(installer, 'find_nexus_runtime_binary', fake_binary)
    monkeypatch.setattr(installer, 'detect_runtime_target', lambda *, node_major: target)
    monkeypatch.setattr(installer, 'fetch_prebuilt_manifest', lambda *_args, **_kwargs: None)
    monkeypatch.setattr('nexus.runtime.executors.command.run_command', fake_run_command)

    await installer.install_dsh(str(tmp_path), prebuilt_base_url='https://artifacts.test/v1')

    assert any(' install ' in command for command in commands)


@pytest.mark.asyncio
async def test_install_dsh_does_not_hide_prebuilt_integrity_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    commands: list[str] = []
    target = RuntimeTarget('linux', 'x64', 'glibc', '2.35', 22)

    async def fake_run_command(request):
        commands.append(request.command)
        return SimpleNamespace(status=CommandRunStatus.Finished, return_code=0, stdout='v22.20.0\n', stderr='')

    def fail_manifest(*_args, **_kwargs):
        raise PrebuiltIntegrityError('checksum mismatch')

    monkeypatch.setattr(
        installer, 'find_nexus_runtime_binary', lambda name: '/runtime/node' if name == 'node' else None
    )
    monkeypatch.setattr(installer, 'detect_runtime_target', lambda *, node_major: target)
    monkeypatch.setattr(installer, 'fetch_prebuilt_manifest', fail_manifest)
    monkeypatch.setattr('nexus.runtime.executors.command.run_command', fake_run_command)

    with pytest.raises(PrebuiltIntegrityError, match='checksum mismatch'):
        await installer.install_dsh(str(tmp_path), prebuilt_base_url='https://artifacts.test/v1')

    assert not any('npm' in command for command in commands)


@pytest.mark.asyncio
async def test_install_dsh_rejects_unsupported_node(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    async def fake_run_command(_request):
        return SimpleNamespace(status=CommandRunStatus.Finished, return_code=0, stdout='v22.18.0\n', stderr='')

    monkeypatch.setattr(installer, 'find_nexus_runtime_binary', lambda _name: '/runtime/node')
    monkeypatch.setattr('nexus.runtime.executors.command.run_command', fake_run_command)

    with pytest.raises(RuntimeError, match=r'\^22\.19\.0'):
        await installer.install_dsh(str(tmp_path))


@pytest.mark.asyncio
async def test_install_dsh_rejects_dist_tags(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='exact package versions'):
        await installer.install_dsh(str(tmp_path), version='latest')


def test_validate_native_build_tools_reports_runtime_requirement(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(installer, 'find_runtime_binary', lambda _name: None)

    with pytest.raises(RuntimeError, match='build-essential'):
        installer._validate_native_build_tools()
