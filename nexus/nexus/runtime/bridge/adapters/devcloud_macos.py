from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlparse

import aiohttp

from nexus.runtime.bridge.adapters.base import (
    BasePosixBridgeAdapter,
    BridgeConnectionInfo,
    BridgeLaunchOptions,
    BridgeLaunchResult,
)
from nexus.runtime.providers.devcloud.client import DevCloudRuntime
from nexus.utils.async_http import http_client_with_trust_env as http_client


async def _upload_file_to_setup(
    base_url: str, local_path: Path, remote_path: str, timeout: float, *, proxy_url: str | None = None
) -> None:
    form = aiohttp.FormData()
    form.add_field('file_path', remote_path)
    with local_path.open('rb') as file_data:
        form.add_field(
            'file_data',
            file_data,
            filename=local_path.name,
            content_type='application/octet-stream',
        )
        resp = await http_client.post(
            f'{base_url.rstrip("/")}/setup/upload',
            data=form,
            timeout=aiohttp.ClientTimeout(total=timeout),
            **({'proxy': proxy_url} if proxy_url else {}),
        )
        if resp.status >= 400:
            raise RuntimeError(f'DevCloud macOS bridge artifact upload failed: HTTP {resp.status} {resp.text[:500]}')


@dataclass(frozen=True)
class DevCloudMacOSBridgeLaunchResult(BridgeLaunchResult):
    proxy_url: str | None = None


class DevCloudMacOSBridgeAdapter(BasePosixBridgeAdapter):
    target_label = 'DevCloud macOS'

    def __init__(self, runtime: DevCloudRuntime) -> None:
        super().__init__(runtime)
        self.runtime: DevCloudRuntime
        self.proxy_url = getattr(runtime, 'proxy_url', None)

    def _connection_info(self) -> BridgeConnectionInfo:
        parsed = urlparse(self.runtime.base_url or '')
        return BridgeConnectionInfo(base_url=f'{parsed.scheme}://{parsed.hostname}:{self.bridge_port}')

    async def _upload_artifact(self, artifact_path: Path, remote_artifact: str) -> None:
        await _upload_file_to_setup(
            self.runtime.base_url or '',
            artifact_path,
            remote_artifact,
            timeout=300.0,
            **self._upload_proxy_kwargs(),
        )

    def _upload_proxy_kwargs(self):
        return {'proxy_url': self.proxy_url} if self.proxy_url else {}

    async def launch_bridge(self, options: BridgeLaunchOptions | None = None) -> BridgeLaunchResult:
        launch = await super().launch_bridge(options)
        if self.proxy_url:
            return DevCloudMacOSBridgeLaunchResult(**asdict(launch), proxy_url=self.proxy_url)
        return launch

    def _health_request_kwargs(self, launch: BridgeLaunchResult):
        return {'proxy': self.proxy_url} if self.proxy_url else {}

    def create_host_client(self, launch: BridgeLaunchResult):
        from nexus.runtime.bridge.host_client import BridgeHostClient

        return BridgeHostClient(
            base_url=launch.base_url,
            headers=launch.headers,
            proxy_url=getattr(launch, 'proxy_url', None),
        )


__all__ = ['DevCloudMacOSBridgeAdapter']
