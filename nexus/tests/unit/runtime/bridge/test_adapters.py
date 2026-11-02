from __future__ import annotations

from pathlib import Path

import aiohttp
import pytest

from nexus.runtime.bridge.adapters import devcloud_macos
from nexus.runtime.bridge.adapters.ags_osworld import AGSOSWorldBridgeAdapter
from nexus.runtime.bridge.adapters.base import BridgeLaunchOptions
from nexus.runtime.bridge.adapters.devcloud_macos import DevCloudMacOSBridgeAdapter
from nexus.runtime.executors.models.command import RunCommandResponse
from nexus.runtime.providers.ags.osworld import AGSOSWorldRuntime, AGSRawOSWorldRuntime
from nexus.runtime.providers.devcloud.client import DevCloudRuntime


class FakeHandle:
    headers = {'X-Access-Token': 'token'}

    def url(self, port: int, path: str = '') -> str:
        return f'https://{port}-iid.example.com{path}'


class FakeOSWorldRuntime(AGSOSWorldRuntime):
    handle = FakeHandle()
    server_port = 5000
    calls: list[tuple[str, dict]]

    def __init__(self) -> None:
        self.calls = []
        self.http_calls = []

    async def _http_request(self, method: str, url: str, **kwargs):
        self.http_calls.append((method, url, kwargs))

        class Response:
            status_code = 200
            text = 'File Uploaded'

        return Response()

    async def run_command(self, command: str, *, timeout: float = 180, **kwargs):
        self.calls.append(('run_command', {'command': command, 'timeout': timeout, **kwargs}))
        return RunCommandResponse(status='Finished', return_code=0, stdout='ok')


class FakeRawOSWorldRuntime(AGSRawOSWorldRuntime):
    def __init__(self, nexus_runtime: FakeOSWorldRuntime) -> None:
        self.nexus_runtime = nexus_runtime

    def to_nexus(self) -> FakeOSWorldRuntime:
        return self.nexus_runtime


class FakeDevCloudRuntime:
    base_url = 'http://9.1.2.3:5000'
    calls: list[tuple[str, object]]

    def __init__(self) -> None:
        self.calls = []

    async def run_command(self, command: str, *, timeout: float = 180, **kwargs):
        self.calls.append(('run_command', {'command': command, 'timeout': timeout}))
        return RunCommandResponse(status='Finished', return_code=0, stdout='ok')

    async def upload_file(self, local_path: str, remote_path: str):
        self.calls.append(('upload_file', (local_path, remote_path)))
        return {'ok': True}


def test_bridge_adapter_factory_infers_adapter_from_runtime_type() -> None:
    from nexus.extensions.agents.installed_common.native_bridge import _build_adapter

    osworld_runtime = FakeOSWorldRuntime()
    devcloud_runtime = DevCloudRuntime(base_url='http://9.1.2.3:5000')

    osworld_adapter = _build_adapter(FakeRawOSWorldRuntime(osworld_runtime))
    devcloud_adapter = _build_adapter(devcloud_runtime)

    assert isinstance(osworld_adapter, AGSOSWorldBridgeAdapter)
    assert osworld_adapter.runtime is osworld_runtime
    assert isinstance(devcloud_adapter, DevCloudMacOSBridgeAdapter)
    assert devcloud_adapter.runtime is devcloud_runtime


@pytest.mark.asyncio
async def test_osworld_adapter_converts_raw_runtime_to_nexus_runtime(tmp_path: Path) -> None:
    artifact = tmp_path / 'bridge.tar.gz'
    artifact.write_bytes(b'tar-data')
    nexus_rt = FakeOSWorldRuntime()
    adapter = AGSOSWorldBridgeAdapter(FakeRawOSWorldRuntime(nexus_rt))

    await adapter.upload_artifact(artifact)

    assert adapter.runtime is nexus_rt
    assert nexus_rt.calls


@pytest.mark.asyncio
async def test_osworld_upload_uses_setup_upload_with_extended_timeout(tmp_path: Path) -> None:
    artifact = tmp_path / 'bridge.tar.gz'
    artifact.write_bytes(b'tar-data')
    rt = FakeOSWorldRuntime()
    adapter = AGSOSWorldBridgeAdapter(rt)

    await adapter.upload_artifact(artifact)

    assert len(rt.http_calls) == 1
    method, url, kwargs = rt.http_calls[0]
    assert method == 'POST'
    assert url == 'https://5000-iid.example.com/setup/upload'
    assert kwargs['timeout'].total == 300.0
    commands = [arg['command'] for name, arg in rt.calls if name == 'run_command']
    assert not any('base64.b64decode' in command for command in commands)
    assert not any('sha256sum' in command or 'shasum' in command for command in commands)


@pytest.mark.asyncio
async def test_osworld_upload_retries_setup_upload_disconnect(tmp_path: Path) -> None:
    class FlakyUploadRuntime(FakeOSWorldRuntime):
        async def _http_request(self, method: str, url: str, **kwargs):
            self.http_calls.append((method, url, kwargs))
            if len(self.http_calls) == 1:
                raise aiohttp.ServerDisconnectedError()

            class Response:
                status_code = 200
                text = 'File Uploaded'

            return Response()

    artifact = tmp_path / 'bridge.tar.gz'
    artifact.write_bytes(b'tar-data')
    rt = FlakyUploadRuntime()
    adapter = AGSOSWorldBridgeAdapter(rt)

    await adapter.upload_artifact(artifact)

    assert [call[1] for call in rt.http_calls] == [
        'https://5000-iid.example.com/setup/upload',
        'https://5000-iid.example.com/setup/upload',
    ]
    commands = [arg['command'] for name, arg in rt.calls if name == 'run_command']
    assert not any('base64.b64decode' in command for command in commands)


@pytest.mark.asyncio
@pytest.mark.parametrize('platform', ['posix', 'waa'])
async def test_wait_ready_reaches_bridge_through_environment_proxy(monkeypatch, platform) -> None:
    from aiohttp import web
    from aiohttp.test_utils import TestServer

    from nexus.runtime.bridge.adapters.ags_waa import AGSWAABridgeAdapter
    from nexus.runtime.bridge.adapters.base import BridgeLaunchResult
    from nexus.utils.async_http import http_client_with_trust_env

    requests = []

    async def health(request):
        requests.append((request.host, request.headers.get('X-Access-Token')))
        return web.json_response({'bridge': 'nexus-llm-bridge', 'launch_id': 'test-launch'})

    app = web.Application()
    app.router.add_get('/health', health)
    async with TestServer(app) as proxy:
        monkeypatch.delenv('NEXUS_HOST_TRUST_ENV', raising=False)
        monkeypatch.setenv('HTTP_PROXY', str(proxy.make_url('/')))
        monkeypatch.setenv('http_proxy', str(proxy.make_url('/')))
        monkeypatch.setenv('NO_PROXY', '')
        monkeypatch.setenv('no_proxy', '')
        adapter = (
            DevCloudMacOSBridgeAdapter(FakeDevCloudRuntime())
            if platform == 'posix'
            else AGSWAABridgeAdapter.__new__(AGSWAABridgeAdapter)
        )
        launch = BridgeLaunchResult(
            base_url='http://bridge.invalid',
            local_base_url='http://127.0.0.1:8765',
            launch_id='test-launch',
            headers={'X-Access-Token': 'test-token'},
        )

        try:
            await adapter.wait_ready(launch, timeout=2)
        finally:
            await http_client_with_trust_env.close()

    assert requests == [('bridge.invalid', 'test-token')]


@pytest.mark.asyncio
async def test_wait_ready_raises_bridge_not_ready_not_timeout() -> None:
    from nexus.runtime.bridge.adapters.base import BridgeLaunchResult, BridgeNotReadyError

    rt = FakeDevCloudRuntime()
    adapter = DevCloudMacOSBridgeAdapter(rt)
    launch = BridgeLaunchResult(base_url='http://127.0.0.1:1', local_base_url='http://127.0.0.1:1')

    with pytest.raises(BridgeNotReadyError) as exc_info:
        await adapter.wait_ready(launch, timeout=0.0)

    # Must not be a TimeoutError, or the runner's timeout handler swallows it as a fake task timeout.
    assert not isinstance(exc_info.value, TimeoutError)
    assert 'not ready' in str(exc_info.value)


@pytest.mark.asyncio
async def test_osworld_launch_returns_connection_headers() -> None:
    rt = FakeOSWorldRuntime()
    adapter = AGSOSWorldBridgeAdapter(rt)

    launch = await adapter.launch_bridge(
        BridgeLaunchOptions(
            conversion_log_level='DEBUG',
            thinking_config='bedrock-claude',
            request_timeout=12.5,
            api_protocol='responses',
        )
    )

    assert launch.base_url == 'https://9281-iid.example.com'
    assert launch.headers == {'X-Access-Token': 'token'}
    launch_command = rt.calls[-1][1]['command']
    assert 'bin/launch_bridge.sh 9281 DEBUG bedrock-claude ' in launch_command
    assert launch_command.endswith(' 12.5 responses')
    assert launch.launch_id


def test_devcloud_adapter_keeps_devcloud_runtime() -> None:
    rt = FakeDevCloudRuntime()
    adapter = DevCloudMacOSBridgeAdapter(rt)

    assert adapter.runtime is rt


def test_devcloud_base_url_rewrites_port() -> None:
    rt = FakeDevCloudRuntime()
    adapter = DevCloudMacOSBridgeAdapter(rt)

    assert adapter._connection_info().base_url == 'http://9.1.2.3:8765'


@pytest.mark.asyncio
async def test_devcloud_launch_forwards_bridge_options() -> None:
    rt = FakeDevCloudRuntime()
    adapter = DevCloudMacOSBridgeAdapter(rt)

    await adapter.launch_bridge(
        BridgeLaunchOptions(
            conversion_log_level='DEBUG',
            thinking_config='bedrock-claude',
            request_timeout=12.5,
            api_protocol='responses',
        )
    )

    command = rt.calls[-1][1]['command']
    assert 'bin/launch_bridge.sh 8765 DEBUG bedrock-claude ' in command
    assert command.endswith(' 12.5 responses')


@pytest.mark.asyncio
async def test_posix_stop_cleans_listening_port_without_pid_file() -> None:
    rt = FakeDevCloudRuntime()
    adapter = DevCloudMacOSBridgeAdapter(rt)

    await adapter.stop_bridge()

    command = rt.calls[-1][1]['command']
    assert 'lsof -tiTCP:8765 -sTCP:LISTEN' in command
    assert 'fuser -k 8765/tcp' in command


@pytest.mark.asyncio
async def test_devcloud_upload_uses_setup_upload_multipart_without_checksum(monkeypatch, tmp_path: Path) -> None:
    artifact = tmp_path / 'bridge.tar.gz'
    artifact.write_bytes(b'tar-data')
    rt = FakeDevCloudRuntime()
    uploads: list[tuple[str, str, str]] = []

    async def fake_upload(base_url: str, local_path: Path, remote_path: str, timeout: float) -> None:
        uploads.append((base_url, str(local_path), remote_path))

    monkeypatch.setattr(devcloud_macos, '_upload_file_to_setup', fake_upload)
    adapter = DevCloudMacOSBridgeAdapter(rt)

    await adapter.upload_artifact(artifact)

    assert uploads == [('http://9.1.2.3:5000', str(artifact), '/tmp/nexus-llm-bridge.tar.gz')]
    assert not any(name == 'upload_file' for name, _ in rt.calls)
    commands = [arg['command'] for name, arg in rt.calls if name == 'run_command']
    assert not any('sha256sum' in command or 'shasum' in command for command in commands)


@pytest.mark.asyncio
@pytest.mark.parametrize('status', [200, 503])
async def test_devcloud_upload_uses_environment_proxy(monkeypatch, tmp_path: Path, status) -> None:
    from aiohttp import web
    from aiohttp.test_utils import TestServer

    from nexus.utils.async_http import http_client_with_trust_env

    uploaded = {}

    async def upload(request):
        fields = await request.post()
        uploaded['path'] = fields['file_path']
        uploaded['content'] = fields['file_data'].file.read()
        return web.Response(status=status, text='upload response')

    app = web.Application()
    app.router.add_post('/setup/upload', upload)
    artifact = tmp_path / 'bridge.tar.gz'
    artifact.write_bytes(b'bridge artifact')
    async with TestServer(app) as proxy:
        monkeypatch.delenv('NEXUS_HOST_TRUST_ENV', raising=False)
        monkeypatch.setenv('HTTP_PROXY', str(proxy.make_url('/')))
        monkeypatch.setenv('http_proxy', str(proxy.make_url('/')))
        monkeypatch.setenv('NO_PROXY', '')
        monkeypatch.setenv('no_proxy', '')
        try:
            if status == 200:
                await devcloud_macos._upload_file_to_setup('http://bridge.invalid', artifact, '/tmp/bridge.tar.gz', 2)
            else:
                with pytest.raises(RuntimeError, match='HTTP 503 upload response'):
                    await devcloud_macos._upload_file_to_setup(
                        'http://bridge.invalid', artifact, '/tmp/bridge.tar.gz', 2
                    )
        finally:
            await http_client_with_trust_env.close()
    assert uploaded == {'path': '/tmp/bridge.tar.gz', 'content': b'bridge artifact'}


@pytest.mark.asyncio
async def test_devcloud_shared_bridge_explicit_proxy_reaches_upload_health_and_host(monkeypatch, tmp_path):
    from aiohttp import web
    from aiohttp.test_utils import TestServer

    from nexus.utils.async_http import http_client_with_trust_env

    seen = []
    launch = None

    async def upload(request):
        fields = await request.post()
        assert fields['file_data'].file.read() == b'tar-data'
        seen.append(request.path)
        return web.Response()

    async def health(request):
        seen.append(request.path)
        return web.json_response({'bridge': 'nexus-llm-bridge', 'launch_id': launch.launch_id})

    app = web.Application()
    app.router.add_post('/setup/upload', upload)
    app.router.add_get('/health', health)
    artifact = tmp_path / 'bridge.tar.gz'
    artifact.write_bytes(b'tar-data')
    monkeypatch.setenv('NO_PROXY', '*')
    monkeypatch.setenv('no_proxy', '*')
    async with TestServer(app) as proxy:
        runtime = FakeDevCloudRuntime()
        runtime.base_url = 'http://bridge.invalid:5000'
        runtime.proxy_url = str(proxy.make_url('/'))
        adapter = DevCloudMacOSBridgeAdapter(runtime)
        try:
            await adapter.upload_artifact(artifact)
            launch = await adapter.launch_bridge()
            await adapter.wait_ready(launch, timeout=2)
            client = adapter.create_host_client(launch)
            assert client.proxy_url == runtime.proxy_url
            assert client.base_url == launch.base_url
        finally:
            await http_client_with_trust_env.close()
    assert seen == ['/setup/upload', '/health']
