# /// script
# dependencies = []
#
# [tool.nexus]
# run_hooks = { gui_recording_cos = "nexus.runner.hooks.gui_recording_cos" }
# ///

"""Record a desktop and upload directly to a user-configured COS bucket."""

from __future__ import annotations

import base64
import json
import logging
import math
import shlex
import subprocess
import uuid

from dataclasses import replace
from datetime import datetime, timezone
from importlib.resources import files
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlsplit

from nexus.runner.hooks.base import RunHook, run_hook_registry
from nexus.runtime.bridge.runtime_kind import NativeBridgeRuntimeKind, classify_native_bridge_runtime
from nexus.runtime.executors.models.gui import GuiStartRecordingRequest
from nexus.utils.cos import AsyncCosClient, CosConfig, load_cos_credentials


if TYPE_CHECKING:
    from nexus.core.context import ExecutionContext
    from nexus.runner.single import RunOutput
    from nexus.runtime.client.base import Runtime


logger = logging.getLogger(__name__)


@run_hook_registry.register('gui_recording_cos')
class GuiRecordingCosHook(RunHook):
    def __init__(
        self,
        *,
        bucket: str,
        region: str = 'ap-guangzhou',
        key_prefix: str = 'gui-recordings',
        runtime_name: str = 'main',
        fps: int = 10,
        max_duration_seconds: int = 3600,
        output_path: str | None = None,
        upload_timeout: float = 600,
        url_expires_seconds: int = 3600,
    ) -> None:
        if not math.isfinite(upload_timeout) or upload_timeout <= 0:
            raise ValueError('upload_timeout must be finite and positive')
        if (
            isinstance(url_expires_seconds, bool)
            or not isinstance(url_expires_seconds, int)
            or url_expires_seconds <= 0
        ):
            raise ValueError('url_expires_seconds must be a positive integer')
        self.url_expires_seconds = url_expires_seconds
        self.config = CosConfig(bucket=bucket, region=region, use_internal_endpoint=True, scheme='https')
        self.request = GuiStartRecordingRequest(
            output_path=output_path, fps=fps, max_duration_seconds=max_duration_seconds
        )
        self.key = '/'.join(part for part in (key_prefix.strip('/'), f'{uuid.uuid4().hex}.mp4') if part)
        self.runtime_name = runtime_name
        self.upload_timeout = upload_timeout
        self._client: AsyncCosClient | None = None
        self._runtime: Runtime | None = None
        self._started = False
        self._stopped = False
        self._uploaded = False
        self._result: dict[str, Any] = {}

    async def on_context_ready(self, ctx: ExecutionContext) -> None:
        self._client = AsyncCosClient(self.config, load_cos_credentials())
        self._runtime = await ctx.get_runtime(self.runtime_name)
        response = await self._runtime.gui_start_recording(**self.request.model_dump())
        if not response.success:
            raise RuntimeError(response.error or 'Failed to start desktop recording')
        self._started = True
        self._result = {'run_id': ctx.run_id, 'runtime_name': self.runtime_name, **response.model_dump()}

    async def on_run_output(self, ctx: ExecutionContext, output: RunOutput) -> None:
        await self._finish()

    async def on_context_closing(self, ctx: ExecutionContext) -> None:
        await self._finish()

    async def _run_upload_script(self, **payload: Any) -> dict[str, Any]:
        assert self._runtime is not None
        source = files('nexus.runtime.utils').joinpath('_artifact_upload.py').read_bytes()
        encoded = base64.b64encode(source).decode('ascii')
        request = base64.b64encode(json.dumps(payload).encode()).decode('ascii')
        code = (
            f'import base64;ns={{}};exec(base64.b64decode({encoded!r}),ns);'
            f'ns["main"](base64.b64decode({request!r}).decode())'
        )
        if classify_native_bridge_runtime(self._runtime) == NativeBridgeRuntimeKind.AGS_WAA:
            command = subprocess.list2cmdline(['python', '-c', code])
        else:
            command = shlex.join(['python3', '-c', code])
        response = await self._runtime.run_command(command=command, timeout=self.upload_timeout + 10)
        if response.return_code != 0:
            raise RuntimeError(f'Sandbox COS command failed: {response.status}')
        for line in (response.stdout or '').splitlines():
            if line.startswith('NEXUS_ARTIFACT_UPLOAD_RESULT:'):
                result = json.loads(line.split(':', 1)[1])
                if not result['success']:
                    raise RuntimeError(result['error'])
                return result
        raise RuntimeError('Sandbox COS command returned no result')

    async def _finish(self) -> None:
        if not self._started or self._uploaded:
            return
        assert self._runtime is not None
        assert self._client is not None
        if not self._stopped:
            response = await self._runtime.gui_stop_recording()
            self._result.update(response.model_dump())
            if not response.success:
                raise RuntimeError(response.error or 'Failed to stop desktop recording')
            self._stopped = True
        path = self._result['output_path']
        metadata = await self._run_upload_script(action='inspect', path=path)
        if metadata['size_bytes'] != self._result['size_bytes']:
            raise RuntimeError('Recording file size changed after stop')
        headers = {'Content-Type': 'video/mp4', 'Content-MD5': metadata['content_md5']}
        url = self._client.presign_url(
            self.key, method='PUT', headers=headers, expires=max(3600, math.ceil(self.upload_timeout) + 60)
        )
        download_client = AsyncCosClient(replace(self.config, use_internal_endpoint=False), self._client.credentials)
        download_url = download_client.presign_url(self.key, method='GET', expires=self.url_expires_seconds)
        expires_at = int(parse_qs(urlsplit(download_url).query)['q-sign-time'][0].split(';')[1])
        uploaded = await self._run_upload_script(
            action='upload',
            path=path,
            url=url,
            method='PUT',
            headers=headers,
            length=metadata['size_bytes'],
            timeout=self.upload_timeout,
        )
        if not uploaded.get('etag'):
            raise RuntimeError('COS upload returned no ETag')
        self._result['cos'] = {
            'bucket': self.config.bucket,
            'region': self.config.region,
            'key': self.key,
            'uri': f'cos://{self.config.bucket}/{self.key}',
            'etag': uploaded['etag'],
            'url': download_url,
            'url_expires_at': datetime.fromtimestamp(expires_at, tz=timezone.utc).isoformat(),
        }
        self._result['source_deleted'] = True
        self._uploaded = True
        logger.info('Desktop recording uploaded to COS: %s', self._result['cos']['uri'])

    def get_result(self) -> dict[str, Any]:
        return dict(self._result)
