"""Task file access for dataset-backed runs."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import stat
import tempfile
import uuid

from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import aiohttp

from pydantic import BaseModel, Field, PrivateAttr

from nexus.dataset._format import format_bytes_detail
from nexus.dataset.access import DataAccessContext, current_data_access
from nexus.dataset.auth import (
    DatasetAuthenticator,
    TaiPatAuthenticator,
    WedataAuthenticator,
    WedataKey,
    authenticator_to_auth_payload,
    default_taihu_pat,
    load_wedata_key,
    tai_pat_from_env,
    wedata_key_available,
)
from nexus.dataset.cache import sha256_file
from nexus.dataset.exceptions import DatasetClientError, DatasetConfigurationError
from nexus.dataset.io import DEFAULT_TIMEOUT_SECONDS, DatasetIoClient
from nexus.dataset.item import CosFileRef, LocalFileRef, TaijiFileRef, TaskFilesSpec
from nexus.dataset.zip_index import (
    ZipEntry,
    ZipIndex,
    matches_path_pattern,
    read_entry_bytes,
    read_zip_index,
    read_zip_index_from_file,
    write_entry_to_file,
)


if TYPE_CHECKING:
    from nexus.runtime import Runtime


logger = logging.getLogger(__name__)

_GLOB_CHARS = frozenset('*?[')


class TaskFilesSession(BaseModel):
    """Runtime-side session containing a task-files spec and transient auth."""

    run_id: str | None = None
    task_files: TaskFilesSpec
    auth: dict[str, Any] | None = Field(default=None, repr=False)
    expires_at: int | None = None


class TaskFiles(BaseModel):
    """Host-side task-files access object."""

    model_config = {'arbitrary_types_allowed': True}

    spec: TaskFilesSpec
    run_id: str | None = None

    async def list(self, pattern: str = '**/*') -> Sequence[str]:
        raise NotImplementedError

    async def exists(self, path: str) -> bool:
        normalized = _normalize_task_path(path)
        return normalized in await self.list(normalized)

    async def read_bytes(self, path: str) -> bytes:
        raise NotImplementedError

    async def read_text(self, path: str, encoding: str = 'utf-8') -> str:
        if _has_glob(path):
            raise ValueError('read_text() only accepts a single file path, not a glob pattern')
        return (await self.read_bytes(path)).decode(encoding)

    async def download(self, path_or_pattern: str, to: str | Path, *, runtime: Runtime | None = None) -> None:
        if runtime is not None:
            await runtime.invoke(
                'nexus.dataset.task_files::runtime_download_task_files',
                await self.create_session(),
                path_or_pattern,
                str(to),
            )
            return
        await _download_to_host(self, path_or_pattern, Path(to))

    async def create_session(self, *, run_id: str | None = None) -> TaskFilesSession:
        return TaskFilesSession(run_id=run_id or self.run_id, task_files=self.spec)

    async def close(self) -> None:
        """Release resources owned by this accessor."""


class ZipTaskFiles(TaskFiles):
    """Nexus ZIP parsing shared by explicit local, Taiji and COS file references."""

    _context: DataAccessContext = PrivateAttr()
    _owns_context: bool = PrivateAttr(default=False)
    _blob: Any = PrivateAttr(default=None)
    _index: ZipIndex | None = PrivateAttr(default=None)

    async def _get_index(self) -> ZipIndex:
        from nexus.dataset.blob import FileResolver

        if self._index is None:
            self._blob = FileResolver.open(self.spec.file_ref(), self._context, size=self.spec.size)
            metadata = await self._blob.stat()
            if self.spec.size is not None and self.spec.size != metadata.size:
                raise ValueError('Task files size mismatch')
            source = self.spec.file_ref()
            if self.spec.sha256 and isinstance(source, CosFileRef) and source.etag is None:
                digest = hashlib.sha256()
                for start in range(0, metadata.size, 1024 * 1024):
                    digest.update(await self._blob.read_range(start, min(start + 1024 * 1024, metadata.size)))
                if digest.hexdigest() != self.spec.sha256:
                    raise ValueError('Task files SHA256 mismatch')
            self._index = await read_zip_index(metadata.size, self._blob.read_range)
            if any(entry.uncompressed_size > 512 * 1024 * 1024 for entry in self._index.entries.values()):
                raise ValueError('Task file exceeds the 512 MiB entry limit')
            if sum(entry.uncompressed_size for entry in self._index.entries.values()) > 4 * 1024**3:
                raise ValueError('Task files exceed the 4 GiB expanded bundle limit')
        return self._index

    async def list(self, pattern: str = '**/*') -> Sequence[str]:
        return (await self._get_index()).list(pattern)

    async def read_bytes(self, path: str) -> bytes:
        entry = (await self._get_index()).entries.get(_normalize_task_path(path))
        if entry is None:
            raise FileNotFoundError(path)
        return await read_entry_bytes(entry, self._blob.read_range)

    async def write_file(self, path: str, target: Path) -> None:
        entry = (await self._get_index()).entries.get(_normalize_task_path(path))
        if entry is None:
            raise FileNotFoundError(path)
        await write_entry_to_file(entry, self._blob.read_range, target)

    async def create_session(self, *, run_id: str | None = None) -> TaskFilesSession:
        source = self.spec.file_ref()
        auth: dict[str, Any] = {'kind': 'data_credentials', 'credentials': {}}
        if isinstance(source, CosFileRef) and 'cos' in self._context.credentials:
            auth['credentials']['cos'] = self._context.credentials['cos']
        if isinstance(source, TaijiFileRef):
            authenticator = self._context.dataset_authenticator()
            if authenticator is not None:
                auth['taiji_auth'] = authenticator_to_auth_payload(authenticator)
        return TaskFilesSession(
            run_id=run_id or self.run_id,
            task_files=self.spec,
            auth=auth,
        )

    async def close(self) -> None:
        if self._owns_context:
            await self._context.close()


class LocalDirectoryTaskFiles(TaskFiles):
    """Task files backed by a local directory."""

    root: Path

    async def list(self, pattern: str = '**/*') -> Sequence[str]:
        paths = [
            item.relative_to(self.root).as_posix()
            for item in self.root.rglob('*')
            if item.is_file() and matches_path_pattern(item.relative_to(self.root).as_posix(), pattern)
        ]
        return sorted(paths)

    async def read_bytes(self, path: str) -> bytes:
        normalized = _normalize_task_path(path)
        full_path = (self.root / normalized).resolve()
        if not _is_relative_to(full_path, self.root):
            raise ValueError(f'Task file path escapes root: {path}')
        if not full_path.is_file():
            raise FileNotFoundError(f'Task file not found: {normalized}')
        return full_path.read_bytes()


class LocalZipTaskFiles(TaskFiles):
    """Task files backed by a local Nexus ZIP archive."""

    zip_path: Path
    _index: ZipIndex | None = PrivateAttr(default=None)

    async def list(self, pattern: str = '**/*') -> Sequence[str]:
        return self._get_index().list(pattern)

    async def read_bytes(self, path: str) -> bytes:
        normalized = _normalize_task_path(path)
        index = self._get_index()
        entry = index.entries.get(normalized)
        if entry is None:
            raise FileNotFoundError(f'Task file not found: {normalized}')
        return await read_entry_bytes(entry, self._read_range)

    def _get_index(self) -> ZipIndex:
        if self._index is None:
            if not self.zip_path.is_file():
                raise FileNotFoundError(f'Task files zip not found: {self.zip_path}')
            if self.spec.sha256:
                actual = sha256_file(self.zip_path)
                if actual != self.spec.sha256:
                    raise ValueError(f'Task files zip sha256 mismatch: expected {self.spec.sha256}, got {actual}')
            self._index = read_zip_index_from_file(self.zip_path)
        return self._index

    async def _read_range(self, start: int, end: int) -> bytes:
        with open(self.zip_path, 'rb') as f:
            f.seek(start)
            return f.read(end - start)


class RemoteZipTaskFiles(TaskFiles):
    """Task files backed by a remote signed Nexus ZIP archive."""

    _index: ZipIndex | None = PrivateAttr(default=None)
    _signed_url: str | None = PrivateAttr(default=None)
    _authenticator: DatasetAuthenticator | None = PrivateAttr(default=None)

    def _auth_payload(self) -> dict[str, Any] | None:
        """Resolve the auth payload, reusing an injected authenticator if any.

        When the dataset client selected an authenticator (e.g. passed
        explicitly to ``DatasetClient``), reuse it so signing stays consistent
        with the metadata/items path. Otherwise fall back to auto-detection,
        whose precedence mirrors ``build_default_authenticator()``.
        """

        if self._authenticator is not None:
            return authenticator_to_auth_payload(self._authenticator)
        return _optional_task_files_auth_payload()

    async def list(self, pattern: str = '**/*') -> Sequence[str]:
        return (await self._get_index()).list(pattern)

    async def read_bytes(self, path: str) -> bytes:
        normalized = _normalize_task_path(path)
        index = await self._get_index()
        entry = index.entries.get(normalized)
        if entry is None:
            raise FileNotFoundError(f'Task file not found: {normalized}')
        _log_remote_entry_read('host', self.spec, entry)
        return await read_entry_bytes(entry, self._read_range)

    async def create_session(self, *, run_id: str | None = None) -> TaskFilesSession:
        return TaskFilesSession(
            run_id=run_id or self.run_id,
            task_files=self.spec,
            auth=self._auth_payload(),
        )

    async def _get_index(self) -> ZipIndex:
        if self._index is not None:
            logger.debug(
                'remote task_files index hit: side=host remote=%s zip=%s entries=%s',
                _remote_label(self.spec),
                self.spec.path,
                len(self._index.entries),
            )
            return self._index
        if self.spec.size is None:
            raise ValueError('Remote task files zip requires size metadata')
        self._index = await read_zip_index(self.spec.size, self._read_range)
        logger.debug(
            'remote task_files index loaded: side=host remote=%s zip=%s size=%s entries=%s',
            _remote_label(self.spec),
            self.spec.path,
            format_bytes_detail(self.spec.size),
            len(self._index.entries),
        )
        return self._index

    async def _read_range(self, start: int, end: int) -> bytes:
        signed_url = await self._get_signed_url()
        return await _http_range_read(signed_url, start, end)

    async def _get_signed_url(self) -> str:
        if self._signed_url is not None:
            return self._signed_url
        remote = self.spec.remote
        if remote is None:
            raise ValueError('Remote task files zip is missing remote dataset coordinates')
        signed = await _sign_remote_task_files(self.spec, auth=self._auth_payload())
        self._signed_url = _extract_signed_url(signed, self.spec.path)
        return self._signed_url


class RuntimeTaskFiles:
    """Runtime-side task-files accessor used through RuntimeControl."""

    def __init__(self, *, sio: Any = None) -> None:
        self.sio = sio
        self._session: TaskFilesSession | None = None
        self._delegate: _RuntimeTaskFilesDelegate | None = None

    def set_session(self, session: TaskFilesSession | dict[str, Any]) -> None:
        parsed = TaskFilesSession.model_validate(session)
        self._session = parsed
        self._delegate = _delegate_for_session(parsed, self)

    async def list(self, pattern: str = '**/*') -> Sequence[str]:
        return await (await self._get_delegate()).list(pattern)

    async def exists(self, path: str) -> bool:
        normalized = _normalize_task_path(path)
        return normalized in await self.list(normalized)

    async def read_bytes(self, path: str) -> bytes:
        if _has_glob(path):
            raise ValueError('read_bytes() only accepts a single file path, not a glob pattern')
        return await (await self._get_delegate()).read_bytes(path)

    async def read_text(self, path: str, encoding: str = 'utf-8') -> str:
        return (await self.read_bytes(path)).decode(encoding)

    async def download(self, path_or_pattern: str, to: str | Path) -> None:
        await (await self._get_delegate()).download(path_or_pattern, Path(to))

    async def request_host_session(self) -> TaskFilesSession:
        request_id = uuid.uuid4().hex
        response = await self._request_response(
            'task_files:session_request',
            f'task_files:session_response:{request_id}',
            {'request_id': request_id},
        )
        if not response.get('ok'):
            raise RuntimeError(response.get('error') or 'Host did not provide task_files session')
        session = TaskFilesSession.model_validate(response.get('session'))
        self.set_session(session)
        return session

    async def request_host_local_file(self, payload: dict[str, Any]) -> dict[str, Any]:
        request_id = uuid.uuid4().hex
        response = await self._request_response(
            'task_files:local_file_request',
            f'task_files:local_file_response:{request_id}',
            {'request_id': request_id, **payload},
        )
        if not response.get('ok'):
            raise RuntimeError(response.get('error') or 'Host failed to serve local task file')
        return response

    async def _get_delegate(self) -> _RuntimeTaskFilesDelegate:
        if self._delegate is None:
            await self.request_host_session()
        assert self._delegate is not None
        return self._delegate

    async def _get_session(self) -> TaskFilesSession:
        if self._session is None:
            await self.request_host_session()
        assert self._session is not None
        return self._session

    async def _request_response(self, event: str, response_event: str, payload: dict[str, Any]) -> dict[str, Any]:
        if self.sio is None:
            raise RuntimeError('Runtime task_files requires Socket.IO bridge')
        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()

        async def on_response(data: dict[str, Any]) -> None:
            if not future.done():
                future.set_result(data)

        self.sio.on(response_event, on_response)
        await self.sio.emit(event, payload)
        return await asyncio.wait_for(future, timeout=DEFAULT_TIMEOUT_SECONDS)


class _RuntimeTaskFilesDelegate:
    async def list(self, pattern: str = '**/*') -> Sequence[str]:
        raise NotImplementedError

    async def read_bytes(self, path: str) -> bytes:
        raise NotImplementedError

    async def download(self, path_or_pattern: str, target: Path) -> None:
        await _download_to_host(self, path_or_pattern, target)


class _RuntimeRemoteZipTaskFiles(_RuntimeTaskFilesDelegate):
    def __init__(self, session: TaskFilesSession) -> None:
        self.session = session
        self.spec = session.task_files
        self._index: ZipIndex | None = None
        self._signed_url: str | None = None

    async def list(self, pattern: str = '**/*') -> Sequence[str]:
        return (await self._get_index()).list(pattern)

    async def read_bytes(self, path: str) -> bytes:
        normalized = _normalize_task_path(path)
        entry = (await self._get_index()).entries.get(normalized)
        if entry is None:
            raise FileNotFoundError(f'Task file not found: {normalized}')
        _log_remote_entry_read('runtime', self.spec, entry)
        return await read_entry_bytes(entry, self._read_range)

    async def _get_index(self) -> ZipIndex:
        if self._index is not None:
            logger.debug(
                'remote task_files index hit: side=runtime remote=%s zip=%s entries=%s',
                _remote_label(self.spec),
                self.spec.path,
                len(self._index.entries),
            )
            return self._index
        if self.spec.size is None:
            raise ValueError('Remote task files zip requires size metadata')
        self._index = await read_zip_index(self.spec.size, self._read_range)
        logger.debug(
            'remote task_files index loaded: side=runtime remote=%s zip=%s size=%s entries=%s',
            _remote_label(self.spec),
            self.spec.path,
            format_bytes_detail(self.spec.size),
            len(self._index.entries),
        )
        return self._index

    async def _read_range(self, start: int, end: int) -> bytes:
        return await _http_range_read(await self._get_signed_url(), start, end)

    async def _get_signed_url(self) -> str:
        if self._signed_url is not None:
            return self._signed_url
        remote = self.spec.remote
        if remote is None:
            raise ValueError('Remote task_files zip is missing remote dataset coordinates')
        signed = await _sign_remote_task_files(self.spec, auth=self.session.auth)
        self._signed_url = _extract_signed_url(signed, self.spec.path)
        return self._signed_url


class _RuntimeLocalTaskFiles(_RuntimeTaskFilesDelegate):
    def __init__(self, owner: RuntimeTaskFiles) -> None:
        self.owner = owner

    async def list(self, pattern: str = '**/*') -> Sequence[str]:
        response = await self.owner.request_host_local_file({'op': 'list', 'pattern': pattern})
        paths = response.get('paths') or []
        if not isinstance(paths, list):
            raise RuntimeError('Invalid host task_files list response')
        return [str(path) for path in paths]

    async def read_bytes(self, path: str) -> bytes:
        normalized = _normalize_task_path(path)
        runtime_path = await self._materialize(normalized)
        return Path(runtime_path).read_bytes()

    async def download(self, path_or_pattern: str, target: Path) -> None:
        matches = await _resolve_matches(self, path_or_pattern)
        if len(matches) == 1 and not _has_glob(path_or_pattern):
            await self._download_one(matches[0], target)
            return
        if target.exists() and not target.is_dir():
            raise ValueError(f'Target must be a directory when downloading multiple task files: {target}')
        for rel_path in matches:
            await self._download_one(rel_path, target / _target_relative_path(rel_path, path_or_pattern))

    async def _materialize(self, rel_path: str) -> str:
        session = await self.owner._get_session()
        target = _runtime_materialized_path(session, rel_path)
        response = await self.owner.request_host_local_file({
            'op': 'materialize',
            'path': rel_path,
            'target': str(target),
        })
        runtime_path = response.get('path')
        if not isinstance(runtime_path, str) or not runtime_path:
            raise RuntimeError('Invalid host task_files materialize response')
        return runtime_path

    async def _download_one(self, rel_path: str, target: Path) -> None:
        response = await self.owner.request_host_local_file({
            'op': 'download',
            'path': rel_path,
            'target': str(target.expanduser()),
        })
        runtime_path = response.get('path')
        if not isinstance(runtime_path, str) or not runtime_path:
            raise RuntimeError('Invalid host task_files download response')


def task_files_from_spec(
    spec: TaskFilesSpec | dict[str, Any] | None,
    *,
    run_id: str | None = None,
    authenticator: DatasetAuthenticator | None = None,
    context: DataAccessContext | None = None,
) -> TaskFiles | None:
    """Build a host-side TaskFiles implementation from a serializable spec.

    ``authenticator`` lets callers reuse an already-selected authentication
    context (e.g. the one chosen by ``DatasetClient``) for remote signing,
    instead of re-reading process-global configuration. It only affects remote
    task_files; local specs ignore it.
    """

    if spec is None:
        return None
    parsed = TaskFilesSpec.model_validate(spec)
    if parsed.source is not None:
        if parsed.type == 'directory' and isinstance(parsed.source, LocalFileRef):
            root = Path(parsed.source.path).expanduser().resolve()
            return LocalDirectoryTaskFiles(spec=parsed, root=root, run_id=run_id)
        files = ZipTaskFiles(spec=parsed, run_id=run_id)
        bound = context or current_data_access()
        files._owns_context = bound is None
        files._context = bound or DataAccessContext.from_env()
        if authenticator is not None:
            files._context.authenticator = authenticator
        return files
    if not isinstance(parsed.path, str) or not parsed.path.strip():
        raise ValueError('Task files path must be non-empty')
    path = Path(parsed.path).expanduser()
    if parsed.remote is not None:
        if parsed.type != 'zip':
            raise ValueError(f'Remote task_files type is not supported: {parsed.type}')
        files = RemoteZipTaskFiles(spec=parsed, run_id=run_id)
        bound = context or current_data_access()
        if bound is not None and authenticator is None:
            authenticator = bound.dataset_authenticator()
            if authenticator is None:
                raise ValueError('Taiji credentials are missing for this execution')
        files._authenticator = authenticator
        return files
    if parsed.type == 'directory':
        root = path.resolve()
        return LocalDirectoryTaskFiles(spec=parsed.model_copy(update={'path': str(root)}), root=root, run_id=run_id)
    if parsed.type == 'zip':
        zip_path = path.resolve()
        return LocalZipTaskFiles(
            spec=parsed.model_copy(update={'path': str(zip_path)}), zip_path=zip_path, run_id=run_id
        )
    raise ValueError(f'Unsupported task_files type: {parsed.type}')


async def runtime_download_task_files(
    session: TaskFilesSession | dict[str, Any],
    path_or_pattern: str,
    to: str,
) -> None:
    """Runtime invoke helper used by host-commanded task file downloads."""

    from nexus.runtime.context import require_runtime_control

    rt = require_runtime_control()
    rt.task_files.set_session(TaskFilesSession.model_validate(session))
    await rt.task_files.download(path_or_pattern, to)


async def runtime_receive_task_file(content: bytes, target: str) -> dict[str, Any]:
    """Runtime invoke helper used by the host to place a task file in the runtime filesystem."""

    target_path = Path(target).expanduser()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_bytes(content)
    return {'path': str(target_path), 'size': len(content)}


async def runtime_read_task_file(path: str) -> bytes:
    """Read a task attachment in a runtime, using its host-provided session."""
    from nexus.runtime.context import require_runtime_control

    return await require_runtime_control().task_files.read_bytes(path)


async def _download_to_host(files: TaskFiles | _RuntimeTaskFilesDelegate, path_or_pattern: str, target: Path) -> None:
    matches = await _resolve_matches(files, path_or_pattern)
    if len(matches) == 1 and not _has_glob(path_or_pattern):
        await _write_single(files, matches[0], target)
        return
    if target.exists() and not target.is_dir():
        raise ValueError(f'Target must be a directory when downloading multiple task files: {target}')
    for rel_path in matches:
        await _write_single(files, rel_path, target / _target_relative_path(rel_path, path_or_pattern))


async def _resolve_matches(files: TaskFiles | _RuntimeTaskFilesDelegate, path_or_pattern: str) -> list[str]:
    if _has_glob(path_or_pattern):
        matches = await files.list(_normalize_pattern(path_or_pattern))
        if not matches:
            raise FileNotFoundError(f'No task files matched pattern: {path_or_pattern}')
        return list(matches)
    normalized = _normalize_task_path(path_or_pattern)
    if normalized not in await files.list(normalized):
        raise FileNotFoundError(f'Task file not found: {normalized}')
    return [normalized]


async def _write_single(files: TaskFiles | _RuntimeTaskFilesDelegate, rel_path: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(files, ZipTaskFiles):
        await files.write_file(rel_path, target)
        return
    if isinstance(files, (LocalZipTaskFiles, RemoteZipTaskFiles, _RuntimeRemoteZipTaskFiles)):
        index = files._get_index() if isinstance(files, LocalZipTaskFiles) else await files._get_index()
        await write_entry_to_file(index.entries[rel_path], files._read_range, target)
        return
    target.write_bytes(await files.read_bytes(rel_path))
    if isinstance(files, LocalDirectoryTaskFiles):
        os.chmod(target, stat.S_IMODE((files.root / rel_path).stat().st_mode))


def _delegate_for_session(session: TaskFilesSession, owner: RuntimeTaskFiles) -> _RuntimeTaskFilesDelegate:
    spec = session.task_files
    if spec.source is not None and spec.is_remote:
        return _RuntimeFileRefTaskFiles(session)
    if spec.remote is not None:
        if spec.type != 'zip':
            raise ValueError(f'Remote task_files type is not supported: {spec.type}')
        return _RuntimeRemoteZipTaskFiles(session)
    return _RuntimeLocalTaskFiles(owner)


class _RuntimeFileRefTaskFiles(_RuntimeTaskFilesDelegate):
    def __init__(self, session: TaskFilesSession) -> None:
        self.session = session

    async def _files(self) -> ZipTaskFiles:
        context = DataAccessContext(credentials=(self.session.auth or {}).get('credentials', {}))
        if auth := (self.session.auth or {}).get('taiji_auth'):
            context.authenticator = (
                _tai_pat_authenticator_from_auth(auth)
                if auth.get('kind') == 'tai_pat'
                else WedataAuthenticator(key=_wedata_key_from_auth(auth))
            )
        files = ZipTaskFiles(spec=self.session.task_files)
        files._context = context
        files._owns_context = True
        return files

    async def list(self, pattern: str = '**/*') -> Sequence[str]:
        files = await self._files()
        try:
            return await files.list(pattern)
        finally:
            await files.close()

    async def read_bytes(self, path: str) -> bytes:
        files = await self._files()
        try:
            return await files.read_bytes(path)
        finally:
            await files.close()

    async def download(self, path_or_pattern: str, target: Path) -> None:
        files = await self._files()
        try:
            await files.download(path_or_pattern, target)
        finally:
            await files.close()


def _wedata_key_auth_payload(key: WedataKey) -> dict[str, Any]:
    return {'kind': 'wedata_key', 'key': key.to_auth_payload()}


def _tai_pat_auth_payload(pat: str) -> dict[str, Any]:
    return {'kind': 'tai_pat', 'pat': pat}


def _optional_wedata_key_auth_payload() -> dict[str, Any] | None:
    try:
        return _wedata_key_auth_payload(load_wedata_key())
    except DatasetConfigurationError:
        if wedata_key_available():
            raise
        return None


def _optional_task_files_auth_payload() -> dict[str, Any] | None:
    """Build the transient auth payload for remote task_files signing.

    Mirrors ``build_default_authenticator()``'s precedence exactly so remote
    task_files signing matches the metadata/items path instead of maintaining a
    second, divergent order:

    1. ``TAI_PAT`` env var -> Taihu PAT.
    2. A configured Wedata key -> legacy TAuth flow.
    3. The default ``~/.nexus/auth/taihu-pat`` file -> Taihu PAT, so PAT-only
       environments can sign private task_files instead of silently falling
       back to anonymous access against the legacy endpoint.

    The credential is a long-lived secret and is passed to the runtime in the
    session payload, exactly like the existing Wedata key. Keep this in mind if
    the trust boundary between host and runtime ever tightens (host-side
    pre-signing would avoid shipping any credential).
    """

    env_pat = tai_pat_from_env()
    if env_pat:
        return _tai_pat_auth_payload(env_pat)
    wedata_payload = _optional_wedata_key_auth_payload()
    if wedata_payload is not None:
        return wedata_payload
    default_pat = default_taihu_pat()
    if default_pat:
        return _tai_pat_auth_payload(default_pat)
    return None


async def _sign_remote_task_files(spec: TaskFilesSpec, *, auth: dict[str, Any] | None) -> dict[str, Any]:
    remote = spec.remote
    if remote is None:
        raise ValueError('Remote task_files zip is missing remote dataset coordinates')
    timeout = aiohttp.ClientTimeout(total=DEFAULT_TIMEOUT_SECONDS)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        client = _dataset_io_client_for_task_files(session, auth=auth)
        try:
            return await client.sign_files(
                remote.namespace,
                remote.dataset,
                remote.version,
                [spec.path],
                split=remote.split,
            )
        except DatasetClientError as exc:
            if auth is None:
                raise DatasetClientError(
                    f'{exc}; no host Wedata key was available, and anonymous task_files signing failed '
                    f'for {_remote_label(spec)} zip={spec.path!r}',
                    trace_id=exc.trace_id,
                    code=exc.code,
                    http_status=exc.http_status,
                ) from exc
            raise


def _dataset_io_client_for_task_files(
    session: aiohttp.ClientSession,
    *,
    auth: dict[str, Any] | None,
) -> DatasetIoClient:
    if auth is None:
        return DatasetIoClient(session=session, allow_anonymous=True)
    kind = auth.get('kind')
    if kind == 'tai_pat':
        return DatasetIoClient(session=session, authenticator=_tai_pat_authenticator_from_auth(auth))
    return DatasetIoClient(session=session, authenticator=WedataAuthenticator(key=_wedata_key_from_auth(auth)))


def _tai_pat_authenticator_from_auth(auth: dict[str, Any]) -> TaiPatAuthenticator:
    pat = auth.get('pat')
    if not isinstance(pat, str) or not pat:
        raise ValueError('Remote task_files auth has invalid Taihu PAT payload')
    return TaiPatAuthenticator(pat=pat)


def _wedata_key_from_auth(auth: dict[str, Any]) -> WedataKey:
    if auth.get('kind') != 'wedata_key':
        raise ValueError('Remote task_files auth has unsupported kind')
    key_payload = auth.get('key')
    if not isinstance(key_payload, dict):
        raise ValueError('Remote task_files auth has invalid Wedata key payload')
    return WedataKey.from_mapping(key_payload, source='task_files session')


def _wedata_key_from_session(session: TaskFilesSession) -> WedataKey:
    auth = session.auth or {}
    if not auth:
        raise ValueError('Remote task_files session is missing Wedata key auth')
    return _wedata_key_from_auth(auth)


def _runtime_materialized_path(session: TaskFilesSession, rel_path: str) -> Path:
    run_id = _cache_component(session.run_id or 'default')
    source = f'{session.task_files.type}\0{session.task_files.path}\0{session.task_files.remote!r}'
    source_id = hashlib.sha256(source.encode('utf-8')).hexdigest()[:16]
    return (
        Path(tempfile.gettempdir()) / 'nexus' / 'task_files' / run_id / source_id / Path(*PurePosixPath(rel_path).parts)
    )


def _cache_component(value: str) -> str:
    return ''.join(ch if ch.isalnum() or ch in '._-' else '_' for ch in value)[:80] or 'default'


async def _http_range_read(url: str, start: int, end: int) -> bytes:
    if end < start:
        raise ValueError(f'Invalid byte range: {start}-{end}')
    if end == start:
        return b''
    headers = {'Range': f'bytes={start}-{end - 1}'}
    verify_ssl = _ssl_verify_for_url(url)
    host = _url_host(url)
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers, ssl=verify_ssl) as response:
            if response.status == 206:
                data = await response.read()
                logger.debug(
                    'task_files http range: host=%s range=%s-%s status=%s size=%s ssl_verify=%s',
                    host,
                    start,
                    end - 1,
                    response.status,
                    format_bytes_detail(len(data)),
                    verify_ssl,
                )
                return data
            if response.status == 200 and start == 0:
                data = await response.read()
                logger.debug(
                    'task_files http range: host=%s range=%s-%s status=%s size=%s truncated_to=%s ssl_verify=%s',
                    host,
                    start,
                    end - 1,
                    response.status,
                    format_bytes_detail(len(data)),
                    format_bytes_detail(end - start),
                    verify_ssl,
                )
                return data[: end - start]
            body = await response.read()
            text = _decode_http_body(body, response.charset)
            logger.debug(
                'task_files http range failed: host=%s range=%s-%s status=%s response_size=%s ssl_verify=%s',
                host,
                start,
                end - 1,
                response.status,
                format_bytes_detail(len(body)),
                verify_ssl,
            )
            raise RuntimeError(f'HTTP range read failed (HTTP {response.status}): {text[:300]}')


def _ssl_verify_for_url(url: str) -> bool:
    hostname = urlparse(url).hostname or ''
    return '.cos-internal.' not in hostname


def _url_host(url: str) -> str:
    return urlparse(url).hostname or '<unknown>'


def _decode_http_body(body: bytes, charset: str | None) -> str:
    encoding = charset or 'utf-8'
    try:
        return body.decode(encoding, errors='replace')
    except LookupError:
        return body.decode('utf-8', errors='replace')


def _remote_label(spec: TaskFilesSpec) -> str:
    remote = spec.remote
    if remote is None:
        return '<local>'
    label = f'{remote.namespace}/{remote.dataset}@{remote.version}'
    if remote.split:
        label = f'{label}?split={remote.split}'
    return label


def _log_remote_entry_read(side: str, spec: TaskFilesSpec, entry: ZipEntry) -> None:
    logger.debug(
        'remote task file read: side=%s remote=%s zip=%s path=%s method=%s range=%s-%s compressed=%s uncompressed=%s',
        side,
        _remote_label(spec),
        spec.path,
        entry.path,
        entry.compression_method,
        entry.data_offset,
        entry.data_offset + entry.compressed_size - 1,
        format_bytes_detail(entry.compressed_size),
        format_bytes_detail(entry.uncompressed_size),
    )


def _extract_signed_url(data: dict[str, Any], path: str) -> str:
    for key in ('urls', 'signed_urls', 'signedUrls'):
        value = data.get(key)
        if isinstance(value, dict) and path in value:
            return str(value[path])
        if isinstance(value, list):
            url = _extract_url_from_entries(value, path)
            if url:
                return url
    for key in ('files', 'items'):
        value = data.get(key)
        if isinstance(value, list):
            url = _extract_url_from_entries(value, path)
            if url:
                return url
    value = data.get(path)
    if isinstance(value, str):
        return value
    raise ValueError(f'files:sign response did not include a signed URL for {path!r}')


def _extract_url_from_entries(entries: list[Any], path: str) -> str | None:
    for item in entries:
        if not isinstance(item, dict):
            continue
        item_path = item.get('path') or item.get('file') or item.get('key')
        url = item.get('url') or item.get('signed_url') or item.get('signedUrl')
        if item_path == path and url:
            return str(url)
    return None


def _normalize_task_path(path: str) -> str:
    normalized = PurePosixPath(path.replace('\\', '/'))
    if normalized.is_absolute() or '..' in normalized.parts:
        raise ValueError(f'Invalid task file path: {path}')
    result = normalized.as_posix()
    while result.startswith('./'):
        result = result[2:]
    if not result or result == '.':
        raise ValueError('Task file path must not be empty')
    return result


def _normalize_pattern(pattern: str) -> str:
    normalized = pattern.replace('\\', '/')
    while normalized.startswith('./'):
        normalized = normalized[2:]
    if normalized.startswith('/') or '/../' in f'/{normalized}/':
        raise ValueError(f'Invalid task file pattern: {pattern}')
    return normalized


def _has_glob(value: str) -> bool:
    return any(char in value for char in _GLOB_CHARS)


def _target_relative_path(rel_path: str, pattern: str) -> str:
    prefix = _literal_glob_prefix(pattern)
    if prefix and rel_path == prefix:
        return PurePosixPath(rel_path).name
    if prefix and rel_path.startswith(prefix.rstrip('/') + '/'):
        return rel_path[len(prefix.rstrip('/')) + 1 :]
    return rel_path


def _literal_glob_prefix(pattern: str) -> str:
    normalized = _normalize_pattern(pattern)
    first_glob = min((normalized.find(char) for char in _GLOB_CHARS if char in normalized), default=-1)
    if first_glob < 0:
        return ''
    prefix = normalized[:first_glob]
    if '/' in prefix:
        return prefix.rsplit('/', 1)[0]
    return ''


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
