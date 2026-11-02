from __future__ import annotations

import base64

from pathlib import Path
from typing import Any

import pytest

from nexus.runtime.bridge.adapters.ags_waa import AGSWAABridgeAdapter
from nexus.runtime.bridge.adapters.base import BridgeLaunchOptions
from nexus.runtime.providers.ags.waa import AGSRawWindowsAgentArenaRuntime


class FakeHandle:
    headers = {'X-Access-Token': 'token'}

    def url(self, port: int, path: str = '') -> str:
        return f'https://{port}-instance.example.com{path}'


class FakeResponse:
    status_code = 200
    text = 'File Uploaded'


class FakeWAARuntime(AGSRawWindowsAgentArenaRuntime):
    handle = FakeHandle()
    server_port = 5000

    def __init__(self) -> None:
        self.scripts: list[str] = []
        self.requests: list[tuple[str, str, dict[str, Any]]] = []
        self.launches: list[tuple[str | list[str], bool]] = []

    async def execute(self, command: str | list[str], shell: bool = False) -> Any:
        assert isinstance(command, list)
        self.scripts.append(base64.b64decode(command[4]).decode('utf-16-le'))
        return {'returncode': 0, 'output': 'launched\n', 'error': ''}

    async def request(self, method: str, path: str, **kwargs: Any) -> FakeResponse:
        self.requests.append((method, path, kwargs))
        return FakeResponse()

    async def setup_launch(self, command: str | list[str], shell: bool = True) -> str:
        self.launches.append((command, shell))
        return 'launched'


@pytest.mark.asyncio
async def test_waa_bridge_upload_extracts_windows_bundle(tmp_path: Path) -> None:
    artifact = tmp_path / 'bridge.tar.gz'
    artifact.write_bytes(b'tar-data')
    runtime = FakeWAARuntime()
    adapter = AGSWAABridgeAdapter(runtime)

    await adapter.upload_artifact(artifact)

    assert runtime.requests[0][0:2] == ('POST', '/setup/upload')
    assert runtime.requests[0][2]['port'] == 5000
    assert any('tar.exe -xzf' in script and r'C:\Windows\Temp' in script for script in runtime.scripts)


@pytest.mark.asyncio
async def test_waa_bridge_launch_uses_mapped_port_and_options() -> None:
    runtime = FakeWAARuntime()
    adapter = AGSWAABridgeAdapter(runtime)

    launch = await adapter.launch_bridge(
        BridgeLaunchOptions(
            conversion_log_level='DEBUG',
            thinking_config='bedrock-claude',
            request_timeout=12.5,
            api_protocol='responses',
        )
    )

    assert launch.base_url == 'https://9281-instance.example.com'
    assert launch.local_base_url == 'http://127.0.0.1:9281'
    assert launch.headers == {'X-Access-Token': 'token'}
    command, shell = runtime.launches[-1]
    assert isinstance(command, list)
    assert command[:4] == ['cmd.exe', '/d', '/s', '/c']
    assert command[-6:-3] == ['9281', 'DEBUG', 'bedrock-claude']
    assert command[-3] == launch.launch_id
    assert command[-2] == '12.5'
    assert command[-1] == 'responses'
    assert launch.launch_id
    assert any(part.endswith('launch_bridge.cmd') for part in command)
    assert shell is False


@pytest.mark.asyncio
async def test_waa_bridge_stop_prefers_recorded_pid() -> None:
    runtime = FakeWAARuntime()
    adapter = AGSWAABridgeAdapter(runtime)

    await adapter.stop_bridge()

    script = runtime.scripts[-1]
    assert 'bridge.pid' in script
    assert 'taskkill.exe /PID $processId /T /F' in script
    assert 'Get-NetTCPConnection -LocalPort 9281' in script
    assert '} else {' not in script


def test_host_bridge_adapter_factory_builds_waa_adapter() -> None:
    from nexus.extensions.agents.installed_common.native_bridge import _build_adapter

    runtime = FakeWAARuntime()
    adapter = _build_adapter(runtime)

    assert isinstance(adapter, AGSWAABridgeAdapter)
    assert adapter.runtime is runtime
