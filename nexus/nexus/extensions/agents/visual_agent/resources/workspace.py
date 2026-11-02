"""Prepare the visual agent's workspace using existing runtime interfaces."""

import asyncio
import base64
import logging
import tempfile
import zipfile

from pathlib import Path, PurePosixPath

from pydantic import BaseModel, Field, field_validator

from nexus.extensions.agents.visual_agent.resources.media import read_media_file
from nexus.runtime.client.base import Runtime
from nexus.runtime.executors.models.file import FileItem


logger = logging.getLogger(__name__)


class WithWorkspace(BaseModel):
    """One sandbox workspace and an optional local directory or ZIP to populate it."""

    workspace: str = Field(default='/tmp', description='Absolute workspace directory inside the main runtime.')
    upload_workspace: str | None = Field(
        default=None, description='Local directory or ZIP whose contents populate workspace.'
    )

    @field_validator('workspace')
    @classmethod
    def validate_workspace(cls, value: str) -> str:
        path = PurePosixPath(value)
        if (
            not value.startswith('/')
            or value.startswith('//')
            or '..' in path.parts
            or '\x00' in value
            or '\\' in value
        ):
            raise ValueError('workspace must be an absolute POSIX directory path without .. or backslashes')
        return str(path)

    @field_validator('upload_workspace')
    @classmethod
    def validate_upload_workspace(cls, value: str | None) -> str | None:
        if value is not None and (not value.strip() or '\x00' in value):
            raise ValueError('upload_workspace must be a non-empty local path')
        return value


def _validate_source(source_path: str) -> Path:
    source = Path(source_path).expanduser()
    if source.is_symlink():
        raise ValueError(f'Workspace source symlinks are not supported: {source}')
    if not source.exists():
        raise FileNotFoundError(f'Workspace source does not exist: {source}')
    if source.is_dir():
        return source
    if source.is_file() and zipfile.is_zipfile(source):
        return source
    raise ValueError(f'upload_workspace must be a directory or a valid ZIP archive: {source}')


def _pack_directory(source: Path, archive_path: Path) -> None:
    with zipfile.ZipFile(archive_path, 'w', compression=zipfile.ZIP_DEFLATED, strict_timestamps=False) as archive:
        for path in sorted(source.rglob('*')):
            # A source such as /tmp can contain our own temporary archive.
            if path == archive_path.parent or archive_path.parent in path.parents:
                continue
            if path.is_symlink() or (not path.is_dir() and not path.is_file()):
                raise ValueError(f'Workspace only supports regular files and directories: {path}')
            archive.write(path, path.relative_to(source).as_posix())


async def _upload_archive_file(runtime: Runtime, source: Path, target: str) -> None:
    """Transfer the temporary ZIP; user-facing targets are always directories."""
    payload = await asyncio.to_thread(read_media_file, source)
    response = await runtime.upload_file(
        [FileItem(path=target, content=base64.b64encode(payload).decode('ascii'), encoding='base64')],
        overwrite=True,
    )
    if len(response.results) != 1 or not response.results[0].success:
        details = '; '.join(result.message or result.path for result in response.results)
        raise RuntimeError(f'Workspace upload failed: {source} -> {target}: {details or "missing result"}')


async def _upload_archive(runtime: Runtime, source: Path, target: str) -> None:
    staging_dir = await runtime.exec("import tempfile\nreturn tempfile.mkdtemp(prefix='nexus-workspace-')")
    try:
        remote_archive = f'{staging_dir}/workspace.zip'
        await _upload_archive_file(runtime, source, remote_archive)
        await runtime.invoke(
            'nexus.extensions.agents.visual_agent.resources.sandbox_workspace::extract_workspace_archive',
            remote_archive,
            target,
        )
    finally:
        try:
            await runtime.exec(f'import shutil\nshutil.rmtree({staging_dir!r})')
        except Exception:
            logger.warning('Could not remove workspace staging directory %s', staging_dir, exc_info=True)


async def upload_workspace(runtime: Runtime, source_path: str, workspace: str = '/tmp') -> str:
    """Copy directory contents or extract a ZIP into one sandbox workspace."""
    config = WithWorkspace(workspace=workspace, upload_workspace=source_path)
    source = await asyncio.to_thread(_validate_source, source_path)
    if source.is_dir():
        with tempfile.TemporaryDirectory(prefix='nexus-workspace-') as local_tmp:
            archive_path = Path(local_tmp) / 'workspace.zip'
            await asyncio.to_thread(_pack_directory, source, archive_path)
            await _upload_archive(runtime, archive_path, config.workspace)
    else:
        await _upload_archive(runtime, source, config.workspace)
    logger.info('Uploaded workspace %s -> %s', source, config.workspace)
    return config.workspace
