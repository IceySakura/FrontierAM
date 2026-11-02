"""Backend-neutral byte access for file bundles."""

from __future__ import annotations

import asyncio
import hashlib
import os
import uuid

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from nexus.dataset.access import DataAccessContext
from nexus.dataset.item import CosFileRef, LocalFileRef, TaijiFileRef, TaskFilesSpec


@dataclass(frozen=True)
class BlobStat:
    size: int
    identity: str


class BlobReader(Protocol):
    async def stat(self) -> BlobStat: ...

    async def read_range(self, start: int, end: int) -> bytes: ...

    async def download_to(self, target: Path, *, sha256: str | None = None) -> None: ...


class BaseBlobReader:
    async def stat(self) -> BlobStat:
        raise NotImplementedError

    async def read_range(self, start: int, end: int) -> bytes:
        raise NotImplementedError

    async def download_to(self, target: Path, *, sha256: str | None = None) -> None:
        metadata = await self.stat()
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f'.{target.name}.{uuid.uuid4().hex}.part')
        digest = hashlib.sha256()
        try:
            with temporary.open('xb') as output:
                for start in range(0, metadata.size, 1024 * 1024):
                    chunk = await self.read_range(start, min(start + 1024 * 1024, metadata.size))
                    output.write(chunk)
                    digest.update(chunk)
            if sha256 is not None and digest.hexdigest() != sha256:
                raise ValueError('Downloaded file SHA256 mismatch')
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)


class LocalBlobReader(BaseBlobReader):
    def __init__(self, source: LocalFileRef) -> None:
        self.path = Path(source.path)
        self._metadata: BlobStat | None = None

    async def stat(self) -> BlobStat:
        stat = self.path.stat()
        value = BlobStat(stat.st_size, f'{stat.st_dev}:{stat.st_ino}:{stat.st_mtime_ns}:{stat.st_ctime_ns}')
        if self._metadata is not None and self._metadata != value:
            raise ValueError('Local file changed while reading')
        self._metadata = value
        return value

    async def read_range(self, start: int, end: int) -> bytes:
        metadata = await self.stat()
        if start < 0 or end < start or end > metadata.size:
            raise ValueError('Invalid local byte range')

        def read() -> bytes:
            with self.path.open('rb') as stream:
                stream.seek(start)
                return stream.read(end - start)

        data = await asyncio.to_thread(read)
        await self.stat()
        if len(data) != end - start:
            raise ValueError('Short local file read')
        return data


class CosBlobReader(BaseBlobReader):
    def __init__(self, source: CosFileRef, context: DataAccessContext) -> None:
        self.source = source
        self.client = context.cos_client(source.bucket, source.region)
        self._metadata: BlobStat | None = None

    async def stat(self) -> BlobStat:
        if self._metadata is None:
            metadata = await self.client.stat(self.source.key)
            if not metadata.etag:
                raise ValueError('COS object has no ETag')
            if self.source.etag is not None and metadata.etag != self.source.etag:
                raise ValueError('COS object identity differs from the published reference')
            self._metadata = BlobStat(metadata.size, metadata.etag)
        return self._metadata

    async def read_range(self, start: int, end: int) -> bytes:
        metadata = await self.stat()
        return await self.client.read_range(self.source.key, start, end, etag=metadata.identity, size=metadata.size)


class TaijiBlobReader(BaseBlobReader):
    def __init__(self, source: TaijiFileRef, context: DataAccessContext, *, size: int | None) -> None:
        self.source = source
        self.context = context
        self.size = size

    async def stat(self) -> BlobStat:
        if self.size is None:
            raise ValueError('Taiji file reference requires size metadata')
        return BlobStat(self.size, f'{self.source.dataset!r}/{self.source.path}')

    async def read_range(self, start: int, end: int) -> bytes:
        from nexus.dataset.auth import authenticator_to_auth_payload
        from nexus.dataset.task_files import _extract_signed_url, _http_range_read, _sign_remote_task_files

        metadata = await self.stat()
        if start < 0 or end < start or end > metadata.size:
            raise ValueError('Invalid Taiji byte range')
        authenticator = self.context.dataset_authenticator()
        spec = TaskFilesSpec(type='zip', path=self.source.path, remote=self.source.dataset, size=self.size)
        signed = await _sign_remote_task_files(
            spec, auth=authenticator_to_auth_payload(authenticator) if authenticator else None
        )
        data = await _http_range_read(_extract_signed_url(signed, spec.path), start, end)
        if len(data) != end - start:
            raise ValueError('Short Taiji byte range response')
        return data


class FileResolver:
    @staticmethod
    def open(
        source: LocalFileRef | TaijiFileRef | CosFileRef,
        context: DataAccessContext,
        *,
        size: int | None = None,
    ) -> BlobReader:
        if isinstance(source, LocalFileRef):
            return LocalBlobReader(source)
        if isinstance(source, CosFileRef):
            return CosBlobReader(source, context)
        if isinstance(source, TaijiFileRef):
            return TaijiBlobReader(source, context, size=size)
        raise TypeError(f'Unsupported file reference: {type(source).__name__}')
