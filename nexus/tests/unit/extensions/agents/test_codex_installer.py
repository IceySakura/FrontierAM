import hashlib
import json

from pathlib import Path
from typing import Any

import pytest

from aiohttp import web

from nexus.extensions.agents import installed_common
from nexus.extensions.agents.codex import installer
from nexus.runtime.executors.models.command import CommandRunStatus, RunCommandResponse


@pytest.mark.parametrize('windows', [False, True])
async def test_binary_download_verifies_checksum_and_preserves_previous_install_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, windows: bool
) -> None:
    monkeypatch.setattr(installer, 'is_windows', lambda: windows)
    payload = b'codex binary payload'
    app = web.Application()

    async def download(_request: web.Request) -> web.Response:
        return web.Response(body=payload)

    app.router.add_get('/codex', download)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '127.0.0.1', 0)
    await site.start()
    url = f'http://127.0.0.1:{runner.addresses[0][1]}/codex?signature=private'
    try:
        binary = await installer.install_codex_from_url(url, str(tmp_path), hashlib.sha256(payload).hexdigest().upper())
        assert binary.name == ('codex.exe' if windows else 'codex')
        assert binary.read_bytes() == payload
        with pytest.raises(RuntimeError, match='SHA256 checksum'):
            await installer.install_codex_from_url(url, str(tmp_path), '0' * 64)
        assert binary.read_bytes() == payload
        with pytest.raises(RuntimeError, match='HTTP 404') as error:
            await installer.install_codex_from_url(url.replace('/codex?', '/missing?'), str(tmp_path))
        assert 'signature' not in str(error.value)
        assert binary.read_bytes() == payload
        assert list(tmp_path.glob('*.tmp')) == []
    finally:
        await runner.cleanup()


async def test_windows_npm_install_uses_powershell_and_cmd_shim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(installer, 'is_windows', lambda: True)
    monkeypatch.setattr(installed_common, 'is_windows', lambda: True)
    monkeypatch.setattr(
        installed_common,
        'find_runtime_binary',
        lambda name: r'C:\Program Files\nodejs\npm.cmd' if name == 'npm' else None,
    )
    install_dir = tmp_path / "directory with spaces and 'quote"
    calls = []

    async def run_command(request: Any) -> RunCommandResponse:
        calls.append(request)
        binary = install_dir / 'node_modules' / '.bin' / 'codex.cmd'
        binary.parent.mkdir(parents=True)
        binary.touch()
        metadata = install_dir / 'node_modules' / '@openai' / 'codex' / 'package.json'
        metadata.parent.mkdir(parents=True)
        metadata.write_text(json.dumps({'version': '0.146.1'}), encoding='utf-8')
        return RunCommandResponse(status=CommandRunStatus.Finished, return_code=0, execution_time=0.1)

    monkeypatch.setattr('nexus.runtime.executors.command.run_command', run_command)
    binary = await installer.install_codex_cli(str(install_dir), '0.146.1')
    assert binary.name == 'codex.cmd'
    assert calls[0].command.startswith("& 'C:\\Program Files\\nodejs\\npm.cmd' 'install'")
    assert "spaces and ''quote" in calls[0].command
    assert calls[0].command.endswith('; exit $LASTEXITCODE')
    assert await installer.install_codex_cli(str(install_dir), '0.146.1') == binary
    assert len(calls) == 1
