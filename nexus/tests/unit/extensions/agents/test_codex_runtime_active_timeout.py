from pathlib import Path
from typing import Any

import pytest

from nexus.extensions.agents.codex import runtime_setup
from nexus.runtime.executors.models.command import CommandRunStatus, RunCommandResponse


class _RuntimeControl:
    sio = object()


class _ResponsesProxy:
    def __init__(self, captured: dict[str, Any], **kwargs: Any) -> None:
        captured['proxy_kwargs'] = kwargs

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass


@pytest.mark.parametrize('mode', ['exec_json', 'app_server'])
@pytest.mark.parametrize('parity_enabled', [False, True])
async def test_codex_runtime_passes_timeout_control_id_to_command(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    parity_enabled: bool,
    mode: str,
) -> None:
    js_entry = tmp_path / 'codex.js'
    js_entry.write_text('')
    captured: dict[str, Any] = {}

    async def install_codex_cli(**kwargs: Any) -> Path:
        binary = 'codex.cmd' if runtime_setup.is_windows() else 'codex'
        return Path(kwargs['install_dir']) / 'node_modules' / '.bin' / binary

    async def run_command(request: Any) -> RunCommandResponse:
        captured['request'] = request
        return RunCommandResponse(
            status=CommandRunStatus.Finished,
            execution_time=0.1,
            return_code=0,
            stdout='codex-cli 0.114.0\n',
            stderr='',
        )

    async def run_app_server(**kwargs: Any) -> dict[str, Any]:
        from types import SimpleNamespace

        captured['request'] = SimpleNamespace(timeout_control_id=kwargs['timeout_control_id'], extra_env=kwargs['env'])
        return {'success': True}

    monkeypatch.setattr('nexus.extensions.agents.codex.app_server_runtime.run_codex_app_server', run_app_server)

    def responses_proxy(**kwargs: Any) -> _ResponsesProxy:
        return _ResponsesProxy(captured, **kwargs)

    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setenv('USERPROFILE', str(tmp_path))
    monkeypatch.setattr('nexus.extensions.agents.codex.installer.install_codex_cli', install_codex_cli)
    monkeypatch.setattr('nexus.runtime.context.require_runtime_control', lambda: _RuntimeControl())
    monkeypatch.setattr('nexus.runtime.executors.command.run_command', run_command)
    monkeypatch.setattr('nexus.runtime.server.services.responses_proxy.ResponsesProxy', responses_proxy)
    monkeypatch.setattr(runtime_setup, 'find_free_port', lambda: 12345)
    monkeypatch.setattr(runtime_setup, 'find_nexus_runtime_binary', lambda _name: '/bin/node')
    monkeypatch.setattr(runtime_setup, 'resolve_npm_bin_entry', lambda *_args: js_entry)
    monkeypatch.setattr(runtime_setup, 'get_current_span_id', lambda: 'codex-runtime-timeout', raising=False)

    await runtime_setup.setup_and_run_codex(
        mode=mode,
        task='fix timeout',
        working_dir=str(tmp_path),
        timeout=10,
        codex_home_name='training-parity' if parity_enabled else None,
        normalize_exec_command_metadata=parity_enabled,
    )

    assert captured['request'].timeout_control_id == 'codex-runtime-timeout'
    assert captured['proxy_kwargs']['request_timeout'] is None

    assert captured['proxy_kwargs']['normalize_exec_command_metadata'] is parity_enabled
    codex_home = Path(captured['request'].extra_env['CODEX_HOME'])
    assert codex_home.parent == tmp_path / '.nexus-codex'
    assert (codex_home.name == 'training-parity') is parity_enabled
    assert (codex_home / 'config.toml').is_file()
