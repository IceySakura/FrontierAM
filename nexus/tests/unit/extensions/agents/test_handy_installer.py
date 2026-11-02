import json

from pathlib import Path
from typing import Any

import pytest

from nexus.extensions.agents.handy import installer
from nexus.runtime.executors import command as command_executor


def test_parse_semver() -> None:
    assert installer.parse_semver('v22.19.0') == (22, 19, 0)


@pytest.mark.asyncio
async def test_prerequisites_reject_old_node(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(installer, 'find_nexus_runtime_binary', lambda name: f'/usr/bin/{name}')
    monkeypatch.setattr(installer, 'find_runtime_binary', lambda name: f'/usr/bin/{name}')

    async def fake_run_checked_command(*args, **kwargs):
        return 'v22.18.0'

    monkeypatch.setattr(installer, '_run_checked_command', fake_run_checked_command)
    with pytest.raises(RuntimeError, match='Node >= 22.19.0'):
        await installer.ensure_runtime_prerequisites()


@pytest.mark.asyncio
async def test_prerequisites_require_ripgrep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(installer, 'find_nexus_runtime_binary', lambda _name: '/usr/bin/node')
    monkeypatch.setattr(installer, 'find_runtime_binary', lambda _name: None)
    with pytest.raises(RuntimeError, match='ripgrep'):
        await installer.ensure_runtime_prerequisites()


@pytest.mark.asyncio
async def test_prerequisites_prefer_runtime_node_over_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """A TypeScript testbed puts its own old Node on PATH; the runtime's must win."""
    monkeypatch.setattr(installer, 'find_nexus_runtime_binary', lambda _name: '/nix/runtime/nix-env/bin/node')
    monkeypatch.setattr(installer, 'find_runtime_binary', lambda name: f'/usr/local/bin/{name}')
    probed: list[str] = []

    async def fake_run_checked_command(command, *args, **kwargs):
        probed.append(command)
        return 'v22.19.0'

    monkeypatch.setattr(installer, '_run_checked_command', fake_run_checked_command)
    assert await installer.ensure_runtime_prerequisites() == ('v22.19.0', '/usr/local/bin/rg')
    assert probed == ['/nix/runtime/nix-env/bin/node --version']


@pytest.mark.asyncio
async def test_install_handy_honors_registry(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, Any] = {}

    async def fake_prerequisites(*args, **kwargs):
        return 'v22.19.0', '/usr/bin/rg'

    async def fake_run_command(request: command_executor.RunCommandRequest) -> command_executor.RunCommandResponse:
        captured['command'] = request.command
        captured['extra_env'] = request.extra_env
        package_dir = tmp_path / 'node_modules' / '@tencent' / 'handy-agent'
        package_dir.mkdir(parents=True)
        (package_dir / 'package.json').write_text(json.dumps({'version': '0.1.14', 'bin': {'handy': 'bin/handy.mjs'}}))
        entry = package_dir / 'bin' / 'handy.mjs'
        entry.parent.mkdir(parents=True)
        entry.write_text('#!/usr/bin/env node\n')
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

    binary = await installer.install_handy_via_npm(
        install_dir=str(tmp_path),
        version='0.1.14',
        npm_registry='https://registry.example.test/',
    )

    assert binary == tmp_path / 'handy'
    assert binary.is_symlink()
    assert '@tencent/handy-agent@0.1.14' in captured['command']
    assert 'https://registry.example.test/' in captured['command']
    assert captured['extra_env']['NPM_CONFIG_REGISTRY'] == 'https://registry.example.test/'
