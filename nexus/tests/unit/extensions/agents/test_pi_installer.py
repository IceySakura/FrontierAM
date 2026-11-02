import json

from pathlib import Path
from typing import Any

import pytest

from nexus.extensions.agents.pi_agent import installer
from nexus.runtime.executors import command as command_executor


def test_parse_semver() -> None:
    assert installer.parse_semver('v22.19.0') == (22, 19, 0)
    assert installer.parse_semver('v24.19.0') == (24, 19, 0)


@pytest.mark.asyncio
async def test_prerequisites_reject_old_node(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(installer, 'find_nexus_runtime_binary', lambda name: f'/usr/bin/{name}')

    async def fake_run_checked_command(*args, **kwargs):
        return 'v22.18.0'

    monkeypatch.setattr(installer, '_run_checked_command', fake_run_checked_command)
    with pytest.raises(RuntimeError, match='Node >= 22.19.0'):
        await installer.ensure_runtime_prerequisites()


@pytest.mark.asyncio
async def test_prerequisites_accept_supported_node(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(installer, 'find_nexus_runtime_binary', lambda name: f'/usr/bin/{name}')

    async def fake_run_checked_command(*args, **kwargs):
        return 'v22.19.0'

    monkeypatch.setattr(installer, '_run_checked_command', fake_run_checked_command)
    assert await installer.ensure_runtime_prerequisites() == 'v22.19.0'


@pytest.mark.asyncio
async def test_prerequisites_require_node_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(installer, 'find_nexus_runtime_binary', lambda name: None)
    with pytest.raises(RuntimeError, match='node was not found'):
        await installer.ensure_runtime_prerequisites()


@pytest.mark.asyncio
async def test_prerequisites_prefer_runtime_node_over_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """A TypeScript testbed puts its own old Node on PATH; the runtime's must win."""
    monkeypatch.setattr(installer, 'find_nexus_runtime_binary', lambda _name: '/nix/runtime/nix-env/bin/node')
    monkeypatch.setattr(installer, 'find_runtime_binary', lambda _name: '/usr/local/bin/node')
    probed: list[str] = []

    async def fake_run_checked_command(command, *args, **kwargs):
        probed.append(command)
        return 'v22.19.0'

    monkeypatch.setattr(installer, '_run_checked_command', fake_run_checked_command)
    assert await installer.ensure_runtime_prerequisites() == 'v22.19.0'
    assert probed == ['/nix/runtime/nix-env/bin/node --version']


def _fake_package(tmp_path: Path, version: str) -> None:
    """Lay out the npm tree the way `npm install` leaves it."""
    package_dir = tmp_path / 'node_modules' / '@earendil-works' / 'pi-coding-agent'
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / 'package.json').write_text(json.dumps({'version': version, 'bin': {'pi': 'dist/cli.js'}}))
    entry = package_dir / 'dist' / 'cli.js'
    entry.parent.mkdir(parents=True, exist_ok=True)
    entry.write_text('#!/usr/bin/env node\n')


@pytest.mark.asyncio
async def test_install_pi_honors_registry_and_version(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, Any] = {}

    async def fake_prerequisites(*args, **kwargs):
        return 'v22.19.0'

    async def fake_run_command(request: command_executor.RunCommandRequest) -> command_executor.RunCommandResponse:
        captured['command'] = request.command
        captured['extra_env'] = request.extra_env
        _fake_package(tmp_path, '0.84.1')
        return command_executor.RunCommandResponse(
            status=command_executor.CommandRunStatus.Finished,
            execution_time=0.1,
            return_code=0,
            stdout='ok',
            stderr='',
        )

    monkeypatch.setattr(installer, 'ensure_runtime_prerequisites', fake_prerequisites)
    monkeypatch.setattr(installer, 'find_runtime_binary', lambda name: f'/usr/bin/{name}')
    monkeypatch.setattr(command_executor, 'run_command', fake_run_command)

    binary = await installer.install_pi_via_npm(
        install_dir=str(tmp_path),
        version='0.84.1',
        npm_registry='https://registry.example.test/',
    )

    assert binary == tmp_path / 'pi'
    assert binary.is_symlink()
    assert '@earendil-works/pi-coding-agent@0.84.1' in captured['command']
    assert 'https://registry.example.test/' in captured['command']
    assert captured['extra_env']['NPM_CONFIG_REGISTRY'] == 'https://registry.example.test/'


@pytest.mark.asyncio
async def test_install_pi_reuses_matching_install(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A pinned version that is already present must not re-run npm."""

    async def fake_prerequisites(*args, **kwargs):
        return 'v22.19.0'

    async def fail_run_command(request):  # pragma: no cover - must not be reached
        raise AssertionError('npm install must be skipped when the version already matches')

    _fake_package(tmp_path, '0.84.1')
    monkeypatch.setattr(installer, 'ensure_runtime_prerequisites', fake_prerequisites)
    monkeypatch.setattr(installer, 'find_runtime_binary', lambda name: f'/usr/bin/{name}')
    monkeypatch.setattr(command_executor, 'run_command', fail_run_command)

    binary = await installer.install_pi_via_npm(install_dir=str(tmp_path), version='0.84.1')

    assert binary == tmp_path / 'pi'


@pytest.mark.asyncio
async def test_install_pi_reinstalls_on_version_mismatch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[str] = []

    async def fake_prerequisites(*args, **kwargs):
        return 'v22.19.0'

    async def fake_run_command(request: command_executor.RunCommandRequest) -> command_executor.RunCommandResponse:
        calls.append(request.command)
        _fake_package(tmp_path, '0.84.1')
        return command_executor.RunCommandResponse(
            status=command_executor.CommandRunStatus.Finished,
            execution_time=0.1,
            return_code=0,
            stdout='ok',
            stderr='',
        )

    _fake_package(tmp_path, '0.80.0')
    monkeypatch.setattr(installer, 'ensure_runtime_prerequisites', fake_prerequisites)
    monkeypatch.setattr(installer, 'find_runtime_binary', lambda name: f'/usr/bin/{name}')
    monkeypatch.setattr(command_executor, 'run_command', fake_run_command)

    await installer.install_pi_via_npm(install_dir=str(tmp_path), version='0.84.1')

    assert len(calls) == 1


@pytest.mark.asyncio
async def test_install_pi_raises_when_npm_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    async def fake_prerequisites(*args, **kwargs):
        return 'v22.19.0'

    async def fake_run_command(request) -> command_executor.RunCommandResponse:
        return command_executor.RunCommandResponse(
            status=command_executor.CommandRunStatus.Finished,
            execution_time=0.1,
            return_code=1,
            stdout='',
            stderr='ENOTFOUND registry',
        )

    monkeypatch.setattr(installer, 'ensure_runtime_prerequisites', fake_prerequisites)
    monkeypatch.setattr(installer, 'find_runtime_binary', lambda name: f'/usr/bin/{name}')
    monkeypatch.setattr(command_executor, 'run_command', fake_run_command)

    with pytest.raises(RuntimeError, match='Failed to install pi'):
        await installer.install_pi_via_npm(install_dir=str(tmp_path), version='0.84.1')


def test_get_installed_version_handles_missing_and_corrupt(tmp_path: Path) -> None:
    assert installer.get_installed_version(str(tmp_path)) is None

    package_dir = tmp_path / 'node_modules' / '@earendil-works' / 'pi-coding-agent'
    package_dir.mkdir(parents=True)
    (package_dir / 'package.json').write_text('{not json')
    assert installer.get_installed_version(str(tmp_path)) is None
