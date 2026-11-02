import io
import tarfile

from pathlib import Path

import pytest

from nexus.runtime.bridge.artifacts import build_app_bundle
from nexus.runtime.executors.models.command import RunCommandResponse
from nexus.runtime.providers.macosworld_devcloud.bridge import MacOSWorldBridgeAdapter


class FakeDevCloudRuntime:
    proxy_url = None
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


@pytest.mark.asyncio
async def test_devcloud_explicit_proxy_covers_upload_and_readiness(monkeypatch, tmp_path: Path) -> None:
    from aiohttp import web
    from aiohttp.test_utils import TestServer

    from nexus.utils.async_http import http_client_with_trust_env

    seen = []
    launch = None

    async def upload(request):
        fields = await request.post()
        with tarfile.open(fileobj=io.BytesIO(fields['file_data'].file.read())) as archive:
            assert 'nexus-llm-bridge/src/macosworld_bridge/protocol.py' in archive.getnames()
        seen.append((request.host, request.path))
        return web.Response()

    async def health(request):
        seen.append((request.host, request.path))
        return web.json_response({'bridge': 'nexus-llm-bridge', 'launch_id': launch.launch_id})

    app = web.Application()
    app.router.add_post('/setup/upload', upload)
    app.router.add_get('/health', health)
    artifact = tmp_path / 'bridge.tar.gz'
    build_app_bundle(output=artifact, target='macos-x86_64-py311', default_port=8765, include_vendor=False)
    # Ambient proxy configuration must not accidentally make the test pass.
    monkeypatch.setenv('NO_PROXY', '*')
    monkeypatch.setenv('no_proxy', '*')
    async with TestServer(app) as proxy:
        runtime = FakeDevCloudRuntime()
        runtime.base_url = 'http://bridge.invalid:5000'
        runtime.proxy_url = str(proxy.make_url('/'))
        adapter = MacOSWorldBridgeAdapter(runtime)
        try:
            await adapter.upload_artifact(artifact)
            launch = await adapter.launch_bridge()
            assert launch.proxy_url == runtime.proxy_url
            assert launch.local_base_url == f'http://127.0.0.1:{adapter.bridge_port}'
            await adapter.wait_ready(launch, timeout=2)
        finally:
            await http_client_with_trust_env.close()
    assert seen == [('bridge.invalid', '/setup/upload'), ('bridge.invalid', '/health')]
