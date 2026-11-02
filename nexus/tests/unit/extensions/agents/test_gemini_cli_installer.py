from pathlib import Path

import pytest

from nexus.extensions.agents.gemini_cli import installer
from nexus.runtime.executors import command as command_executor


@pytest.mark.asyncio
async def test_install_gemini_cli_honors_npm_registry(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    captured: dict[str, object] = {}

    def fake_find_runtime_binary(name: str) -> str:
        return f'/usr/bin/{name}'

    async def fake_run_command(request: command_executor.RunCommandRequest) -> command_executor.RunCommandResponse:
        captured['command'] = request.command
        captured['cwd'] = request.cwd
        captured['extra_env'] = request.extra_env

        installation = installer.get_installation(version='0.33.1', install_root=str(tmp_path))
        installation.package_dir.mkdir(parents=True, exist_ok=True)
        installation.bin_path.parent.mkdir(parents=True, exist_ok=True)
        installation.bin_path.write_text('', encoding='utf-8')

        return command_executor.RunCommandResponse(
            status=command_executor.CommandRunStatus.Finished,
            execution_time=0.1,
            return_code=0,
            stdout='ok',
            stderr='',
        )

    monkeypatch.setattr(installer, 'find_runtime_binary', fake_find_runtime_binary)
    monkeypatch.setattr(command_executor, 'run_command', fake_run_command)

    installation = await installer.install_gemini_cli(
        version='0.33.1',
        install_root=str(tmp_path),
        npm_registry='https://registry.npmjs.org/',
    )

    assert installation.package_dir.exists()
    assert '--registry' in captured['command']
    assert 'https://registry.npmjs.org/' in captured['command']
    assert captured['extra_env']['NPM_CONFIG_REGISTRY'] == 'https://registry.npmjs.org/'
