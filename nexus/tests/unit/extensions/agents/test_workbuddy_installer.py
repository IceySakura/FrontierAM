import hashlib

from pathlib import Path
from typing import Any

import pytest

from nexus.extensions.agents.workbuddy import installer
from nexus.runtime.executors import command as command_executor


def test_build_private_npmrc_uses_basic_auth_not_raw_token() -> None:
    npmrc = installer.build_private_npmrc(
        'http://mirrors.tencent.com/npm/',
        'g_reader',
        'secret-token',
    )
    assert 'registry=http://mirrors.tencent.com/npm/' in npmrc
    assert '//mirrors.tencent.com/npm/:_auth=' in npmrc
    assert 'secret-token' not in npmrc
    assert 'always-auth=true' in npmrc


def test_resolve_tarball_path_prefers_explicit_then_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    explicit = tmp_path / 'explicit.tar'
    env_file = tmp_path / 'env.tar'
    explicit.write_bytes(b'x')
    env_file.write_bytes(b'y')
    monkeypatch.setenv('WORKBUDDY_TARBALL', str(env_file))

    assert installer.resolve_tarball_path(str(explicit)) == explicit.resolve()
    assert installer.resolve_tarball_path(None) == env_file.resolve()


def test_resolve_tarball_path_missing_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('WORKBUDDY_TARBALL', raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(installer, 'DEFAULT_TARBALL_NAME', 'definitely-missing-wb.tar')
    with pytest.raises(FileNotFoundError, match='WORKBUDDY_TARBALL'):
        installer.resolve_tarball_path(str(tmp_path / 'missing.tar'))


def test_iter_file_chunks_and_part_names(tmp_path: Path) -> None:
    path = tmp_path / 'wb.tar'
    path.write_bytes(b'abcdefghi')
    chunks = list(installer.iter_file_chunks(path, chunk_bytes=4))
    assert chunks == [b'abcd', b'efgh', b'i']
    assert installer.remote_part_path(0) == '/tmp/nexus/workbuddy/wb-linux-x64.tar.part000'
    assert installer.remote_part_path(12) == '/tmp/nexus/workbuddy/wb-linux-x64.tar.part012'


def test_build_assemble_command_quotes_paths() -> None:
    cmd = installer.build_assemble_command(
        ['/tmp/a part.000', '/tmp/a part.001'],
        '/tmp/out tar',
        '/opt/work buddy',
    )
    assert "cat '/tmp/a part.000' '/tmp/a part.001' > '/tmp/out tar'" in cmd
    assert "tar -xf '/tmp/out tar' -C '/opt/work buddy'" in cmd


def test_sqlite_addon_present(tmp_path: Path) -> None:
    assert installer.sqlite_addon_present(str(tmp_path)) is False
    addon = tmp_path / 'package' / 'node_modules' / 'better-sqlite3'
    addon.mkdir(parents=True)
    assert installer.sqlite_addon_present(str(tmp_path)) is True

    npm_addon = tmp_path / 'node_modules' / 'better-sqlite3'
    npm_addon.mkdir(parents=True)
    assert installer.sqlite_addon_present(str(tmp_path)) is True


@pytest.mark.asyncio
async def test_install_from_private_npm_registry(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, Any] = {}
    npmrc = tmp_path / 'runtime-npmrc'
    monkeypatch.setattr(installer, 'REMOTE_NPMRC', str(npmrc))

    async def fake_run_command(request: command_executor.RunCommandRequest) -> command_executor.RunCommandResponse:
        captured['command'] = request.command
        captured['extra_env'] = request.extra_env
        userconfig = Path(request.extra_env['NPM_CONFIG_USERCONFIG'])
        captured['npmrc'] = userconfig.read_text()
        captured['npmrc_mode'] = userconfig.stat().st_mode & 0o777
        package = tmp_path / 'node_modules' / '@tencent' / 'nexus-wb-cli'
        (package / 'bin').mkdir(parents=True)
        (package / 'bin' / 'wb.js').write_text('#!/usr/bin/env node\n')
        (package / 'package.json').write_text('{"name":"@tencent/nexus-wb-cli","version":"5.7.0"}')
        (tmp_path / 'node_modules' / 'better-sqlite3').mkdir(parents=True)
        return command_executor.RunCommandResponse(
            status=command_executor.CommandRunStatus.Finished,
            execution_time=0.1,
            return_code=0,
            stdout='ok',
            stderr='',
        )

    monkeypatch.setattr(command_executor, 'run_command', fake_run_command)
    monkeypatch.setattr(installer, 'find_runtime_binary', lambda name: f'/usr/bin/{name}')

    result = await installer.install_workbuddy(
        str(tmp_path),
        version='5.7.0',
        npm_registry='https://npm.example.test/',
        npm_token='read-only-test-token',
        npm_username='shared-reader',
    )

    assert result == tmp_path / 'node_modules' / '@tencent' / 'nexus-wb-cli' / 'bin' / 'wb.js'
    assert '@tencent/nexus-wb-cli@5.7.0' in captured['command']
    assert captured['extra_env']['NPM_CONFIG_REGISTRY'] == 'https://npm.example.test/'
    assert captured['extra_env']['NPM_CONFIG_USERCONFIG'] == str(npmrc)
    assert '//npm.example.test/:_auth=' in captured['npmrc']
    assert 'read-only-test-token' not in captured['npmrc']
    assert captured['npmrc_mode'] == 0o600
    assert 'read-only-test-token' not in captured['command']
    assert not npmrc.exists()


def test_pinned_version_rejects_dist_tags() -> None:
    assert installer.DEFAULT_WORKBUDDY_VERSION == 'latest'
    assert installer.pinned_version('5.11.0-hotfix.202610272104') == '5.11.0-hotfix.202610272104'
    assert installer.pinned_version('latest') is None
    assert installer.pinned_version('hotfix') is None


@pytest.mark.asyncio
async def test_install_dist_tag_reinstalls_and_accepts_resolved_version(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    package = tmp_path / 'node_modules' / '@tencent' / 'nexus-wb-cli'
    (package / 'bin').mkdir(parents=True)
    (package / 'bin' / 'wb.js').write_text('#!/usr/bin/env node\n')
    (package / 'package.json').write_text('{"version":"5.9.0"}')
    (tmp_path / 'node_modules' / 'better-sqlite3').mkdir(parents=True)
    monkeypatch.setattr(installer, 'REMOTE_NPMRC', str(tmp_path / 'runtime-npmrc'))
    commands: list[str] = []

    async def fake_run_command(request: command_executor.RunCommandRequest) -> command_executor.RunCommandResponse:
        commands.append(request.command)
        (package / 'package.json').write_text('{"version":"5.11.0-hotfix.202610271047"}')
        return command_executor.RunCommandResponse(
            status=command_executor.CommandRunStatus.Finished,
            execution_time=0.1,
            return_code=0,
            stdout='ok',
            stderr='',
        )

    monkeypatch.setattr(command_executor, 'run_command', fake_run_command)
    monkeypatch.setattr(installer, 'find_runtime_binary', lambda name: f'/usr/bin/{name}')

    result = await installer.install_workbuddy(
        str(tmp_path),
        npm_registry='https://npm.example.test/',
        npm_token='read-only-test-token',
        npm_username='shared-reader',
    )

    assert result == package / 'bin' / 'wb.js'
    assert len(commands) == 1
    assert '@tencent/nexus-wb-cli@latest' in commands[0]
    assert installer.get_installed_version(str(tmp_path)) == '5.11.0-hotfix.202610271047'


@pytest.mark.asyncio
async def test_install_from_generic_url_verifies_sha256(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    archive_bytes = b'valid npm package bytes'
    commands: list[str] = []
    monkeypatch.setattr(installer, 'REMOTE_PACKAGE', str(tmp_path / 'download.tgz'))

    async def fake_run_command(request: command_executor.RunCommandRequest) -> command_executor.RunCommandResponse:
        commands.append(request.command)
        if 'curl' in request.command:
            archive = Path(installer.REMOTE_PACKAGE)
            archive.parent.mkdir(parents=True, exist_ok=True)
            archive.write_bytes(archive_bytes)
        else:
            package = tmp_path / 'node_modules' / '@tencent' / 'nexus-wb-cli'
            (package / 'bin').mkdir(parents=True)
            (package / 'bin' / 'wb.js').write_text('#!/usr/bin/env node\n')
            (package / 'package.json').write_text('{"name":"@tencent/nexus-wb-cli","version":"5.7.0"}')
            (tmp_path / 'node_modules' / 'better-sqlite3').mkdir(parents=True)
        return command_executor.RunCommandResponse(
            status=command_executor.CommandRunStatus.Finished,
            execution_time=0.1,
            return_code=0,
            stdout='ok',
            stderr='',
        )

    monkeypatch.setattr(command_executor, 'run_command', fake_run_command)
    monkeypatch.setattr(installer, 'find_runtime_binary', lambda name: name)

    result = await installer.install_workbuddy(
        str(tmp_path),
        version='5.7.0',
        package_url='https://artifacts.example.test/workbuddy-5.7.0.tgz',
        package_sha256=hashlib.sha256(archive_bytes).hexdigest(),
    )

    assert result.is_file()
    assert len(commands) == 2
    assert 'https://artifacts.example.test/workbuddy-5.7.0.tgz' in commands[0]
    assert installer.REMOTE_PACKAGE in commands[1]


@pytest.mark.asyncio
async def test_install_from_generic_url_rejects_bad_sha256(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(installer, 'REMOTE_PACKAGE', str(tmp_path / 'download.tgz'))

    async def fake_run_command(request: command_executor.RunCommandRequest) -> command_executor.RunCommandResponse:
        archive = Path(installer.REMOTE_PACKAGE)
        archive.parent.mkdir(parents=True, exist_ok=True)
        archive.write_bytes(b'tampered')
        return command_executor.RunCommandResponse(
            status=command_executor.CommandRunStatus.Finished,
            execution_time=0.1,
            return_code=0,
            stdout='ok',
            stderr='',
        )

    monkeypatch.setattr(command_executor, 'run_command', fake_run_command)

    with pytest.raises(RuntimeError, match='SHA256 mismatch'):
        await installer.install_workbuddy(
            str(tmp_path),
            package_url='https://artifacts.example.test/workbuddy-5.7.0.tgz',
            package_sha256='0' * 64,
        )


@pytest.mark.asyncio
async def test_install_skips_extract_when_wb_js_exists(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    wb_js = tmp_path / 'package' / 'bin' / 'wb.js'
    wb_js.parent.mkdir(parents=True)
    wb_js.write_text('#!/usr/bin/env node\n')
    (tmp_path / 'package' / 'package.json').write_text('{"version":"5.7.0"}')
    (tmp_path / 'package' / 'node_modules' / 'better-sqlite3').mkdir(parents=True)
    captured: list[str] = []

    async def fake_run_command(request: command_executor.RunCommandRequest) -> command_executor.RunCommandResponse:
        captured.append(request.command)
        return command_executor.RunCommandResponse(
            status=command_executor.CommandRunStatus.Finished,
            execution_time=0.01,
            return_code=0,
            stdout='ok',
            stderr='',
        )

    monkeypatch.setattr(command_executor, 'run_command', fake_run_command)
    result = await installer.install_workbuddy_from_parts(str(tmp_path), ['/tmp/part000'])
    assert result == wb_js
    assert captured == []


@pytest.mark.asyncio
async def test_install_assembles_then_npm(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, Any] = {}

    async def fake_run_command(request: command_executor.RunCommandRequest) -> command_executor.RunCommandResponse:
        captured.setdefault('commands', []).append(request.command)
        captured.setdefault('cwds', []).append(request.cwd)
        captured['extra_env'] = request.extra_env
        wb_js = tmp_path / 'package' / 'bin' / 'wb.js'
        wb_js.parent.mkdir(parents=True, exist_ok=True)
        wb_js.write_text('#!/usr/bin/env node\n')
        (tmp_path / 'package' / 'package.json').write_text('{"version":"5.7.0"}')
        if 'npm' in request.command:
            (tmp_path / 'package' / 'node_modules' / 'better-sqlite3').mkdir(parents=True, exist_ok=True)
        return command_executor.RunCommandResponse(
            status=command_executor.CommandRunStatus.Finished,
            execution_time=0.1,
            return_code=0,
            stdout='ok',
            stderr='',
        )

    monkeypatch.setattr(command_executor, 'run_command', fake_run_command)
    monkeypatch.setattr(installer, 'find_runtime_binary', lambda name: f'/usr/bin/{name}')

    result = await installer.install_workbuddy_from_parts(
        str(tmp_path),
        ['/tmp/part000', '/tmp/part001'],
        npm_registry='https://registry.example.test/',
    )

    assert result == tmp_path / 'package' / 'bin' / 'wb.js'
    assert any('cat' in cmd and 'tar -xf' in cmd for cmd in captured['commands'])
    assert any('better-sqlite3@12.8.0' in cmd for cmd in captured['commands'])
    assert any(cwd == str(tmp_path / 'package') for cwd in captured['cwds'])
    assert captured['extra_env']['NPM_CONFIG_REGISTRY'] == 'https://registry.example.test/'
