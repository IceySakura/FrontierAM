# /// script
# dependencies = []
#
# [tool.nexus]
# run_hooks = { gui_recording = "nexus.runner.hooks.gui_recording" }
# ///

"""Record a task's desktop and upload directly from the sandbox to COS."""

from __future__ import annotations

import base64
import json
import logging
import shlex
import subprocess

from importlib.resources import files
from typing import TYPE_CHECKING, Any

from nexus.runner.hooks.base import RunHook, run_hook_registry
from nexus.runtime.bridge.runtime_kind import NativeBridgeRuntimeKind, classify_native_bridge_runtime
from nexus.runtime.executors.models.gui import GuiStartRecordingRequest
from nexus.utils.artifact import ArtifactClient


if TYPE_CHECKING:
    from nexus.core.context import ExecutionContext
    from nexus.runner.single import RunOutput
    from nexus.runtime.client.base import Runtime


logger = logging.getLogger(__name__)


@run_hook_registry.register('gui_recording')
class GuiRecordingHook(RunHook):
    def __init__(
        self,
        *,
        runtime_name: str = 'main',
        fps: int = 10,
        max_duration_seconds: int = 3600,
        output_path: str | None = None,
        artifact_name: str | None = None,
        artifact_folder: str | None = None,
        storage_pool: str | None = None,
        upload_timeout: float = 600,
    ) -> None:
        if upload_timeout <= 0:
            raise ValueError('upload_timeout must be positive')
        self.runtime_name = runtime_name
        self.request = GuiStartRecordingRequest(
            output_path=output_path, fps=fps, max_duration_seconds=max_duration_seconds
        )
        self._upload_key: str | None = None
        self.artifact_name = artifact_name or ''
        self.artifact_folder = artifact_folder
        self._parent_folder_id: str | None = None
        self.storage_pool = storage_pool
        self.upload_timeout = upload_timeout
        self._runtime: Runtime | None = None
        self._started = False
        self._stopped = False
        self._uploaded = False
        self._upload_result: dict[str, Any] | None = None
        self._result: dict[str, Any] = {}

    async def on_context_ready(self, ctx: ExecutionContext) -> None:
        self._upload_key = f'gui-recording:{ctx.run_id}:{self.runtime_name}'
        self.artifact_name = self.artifact_name or f'recording-{ctx.run_id}.mp4'
        if self.artifact_folder and self.artifact_folder != '/':
            async with ArtifactClient() as client:
                self._parent_folder_id = await client.resolve_resource_id(self.artifact_folder, expect='folder')
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
        # run_command uses cmd.exe on WAA and a POSIX shell on OSWorld/macOS.
        if classify_native_bridge_runtime(self._runtime) == NativeBridgeRuntimeKind.AGS_WAA:
            command = subprocess.list2cmdline(['python', '-c', code])
        else:
            command = shlex.join(['python3', '-c', code])
        response = await self._runtime.run_command(command=command, timeout=self.upload_timeout + 10)
        if response.return_code != 0:
            raise RuntimeError(f'Sandbox artifact command failed: {response.status}')
        for line in (response.stdout or '').splitlines():
            if line.startswith('NEXUS_ARTIFACT_UPLOAD_RESULT:'):
                result = json.loads(line.split(':', 1)[1])
                if not result['success']:
                    raise RuntimeError(result['error'])
                return result
        raise RuntimeError('Sandbox artifact command returned no result')

    async def _finish(self) -> None:
        if not self._started or self._uploaded:
            return
        assert self._runtime is not None
        if not self._stopped:
            response = await self._runtime.gui_stop_recording()
            self._result.update(response.model_dump())
            if not response.success:
                raise RuntimeError(response.error or 'Failed to stop desktop recording')
            self._stopped = True

        async with ArtifactClient() as client:
            if self._upload_result is None:
                remote_path = self._result['output_path']
                metadata = await self._run_upload_script(action='inspect', path=remote_path)
                if metadata['size_bytes'] != self._result['size_bytes']:
                    raise RuntimeError('Recording file size changed after stop')
                session = await client.create_upload(
                    filename=self.artifact_name,
                    size_bytes=metadata['size_bytes'],
                    content_md5=metadata['content_md5'],
                    content_type='video/mp4',
                    parent_folder_id=self._parent_folder_id,
                    storage_pool=self.storage_pool,
                    idempotency_key=self._upload_key,
                )
                self._result['artifact'] = {'artifact_id': session.artifact_id, 'status': session.status}
                parts = None
                if session.is_ready:
                    await self._run_upload_script(action='upload', path=remote_path, already_uploaded=True)
                else:
                    if session.upload_id is None:
                        raise RuntimeError('Artifact upload session has no upload_id')
                    if session.is_multipart:
                        assert session.part_size is not None
                        expected_count = (metadata['size_bytes'] + session.part_size - 1) // session.part_size
                        if len(session.part_urls) != expected_count:
                            raise RuntimeError('Artifact upload session has an invalid part count')
                        parts = []
                        for index, url in enumerate(session.part_urls):
                            offset = index * session.part_size
                            uploaded = await self._run_upload_script(
                                action='upload',
                                path=remote_path,
                                url=url,
                                method='PUT',
                                headers={},
                                offset=offset,
                                length=min(session.part_size, metadata['size_bytes'] - offset),
                                delete_source=index == expected_count - 1,
                                timeout=self.upload_timeout,
                            )
                            if not uploaded.get('etag'):
                                raise RuntimeError('COS multipart upload returned no ETag')
                            parts.append({'part_number': index + 1, 'etag': uploaded['etag']})
                    else:
                        if not session.url:
                            raise RuntimeError('Artifact upload session has no URL')
                        await self._run_upload_script(
                            action='upload',
                            path=remote_path,
                            url=session.url,
                            method=session.method,
                            headers=session.headers,
                            length=metadata['size_bytes'],
                            timeout=self.upload_timeout,
                        )
                # COS has the file and the script has deleted the source. Keep
                # completion data so a host-side retry does not upload again.
                self._upload_result = {
                    'artifact_id': session.artifact_id,
                    'upload_id': None if session.is_ready else session.upload_id,
                    'parts': parts,
                }
                self._result['source_deleted'] = True
            if self._upload_result['upload_id'] is not None:
                await client.complete_upload(self._upload_result['upload_id'], parts=self._upload_result['parts'])
            artifact = await client.get_artifact(self._upload_result['artifact_id'])
            if artifact.status != 'ready':
                raise RuntimeError(f'Artifact is not ready: {artifact.status}')
            self._result['artifact'] = {
                'artifact_id': artifact.artifact_id,
                'name': artifact.name,
                'content_type': artifact.content_type,
                'size_bytes': artifact.size_bytes,
                'status': artifact.status,
                'parent_folder_id': artifact.parent_folder_id,
            }
            self._uploaded = True
            logger.info('Desktop recording uploaded: %s (%s)', artifact.artifact_id, artifact.name)

    def get_result(self) -> dict[str, Any]:
        return dict(self._result)
