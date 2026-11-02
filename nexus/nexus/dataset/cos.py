"""Immutable COS dataset releases and streaming record reader."""

from __future__ import annotations

import asyncio
import hashlib
import json
import tempfile

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Literal
from urllib.parse import parse_qs, quote, unquote, urlencode, urlsplit

import aiohttp

from pydantic import BaseModel, ConfigDict, Field, model_validator

from nexus.dataset.access import DataAccessContext, current_data_access
from nexus.dataset.blob import CosBlobReader
from nexus.dataset.build import DatasetBuild, canonical_json, validate_component
from nexus.dataset.cache import sha256_file
from nexus.dataset.item import CosFileRef, DatasetItem, TaskFilesSpec
from nexus.dataset.reader import (
    BaseDatasetReader,
    DatasetDescriptor,
    DatasetReaderError,
    DatasetSplitDescriptor,
    _validate_query,
)
from nexus.utils.cos import AsyncCosClient, CosClientError, CosServiceError


MAX_MANIFEST_BYTES = 16 * 1024 * 1024
MAX_SHARD_BYTES = 32 * 1024 * 1024


class CosDatasetRef(BaseModel):
    model_config = ConfigDict(extra='forbid')
    manifest_uri: str
    region: str
    manifest_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    split: str | None = None

    @model_validator(mode='after')
    def validate_location(self) -> CosDatasetRef:
        parsed = urlsplit(self.manifest_uri)
        if parsed.scheme != 'cos' or parsed.query or parsed.fragment or not parsed.path.startswith('/'):
            raise ValueError('Invalid COS Manifest object URI')
        CosFileRef(bucket=parsed.netloc, region=self.region, key=unquote(parsed.path[1:]))
        if self.split is not None:
            validate_component(self.split)
        return self

    def file_ref(self) -> CosFileRef:
        parsed = urlsplit(self.manifest_uri)
        return CosFileRef(bucket=parsed.netloc, region=self.region, key=unquote(parsed.path[1:]))

    @property
    def data_uri(self) -> str:
        query = {'region': self.region}
        if self.split is not None:
            query['split'] = self.split
        return f'{self.manifest_uri}?{urlencode(query)}#sha256={self.manifest_sha256}'

    @classmethod
    def parse(cls, value: str) -> CosDatasetRef:
        parsed = urlsplit(value)
        query = parse_qs(parsed.query, keep_blank_values=True)
        if set(query) - {'region', 'split'} or any(len(v) != 1 for v in query.values()):
            raise ValueError('Invalid COS dataset URI query')
        if 'region' not in query or not parsed.fragment.startswith('sha256='):
            raise ValueError('COS dataset URI requires region and Manifest sha256')
        return cls(
            manifest_uri=f'{parsed.scheme}://{parsed.netloc}{parsed.path}',
            region=query['region'][0],
            manifest_sha256=parsed.fragment.removeprefix('sha256='),
            split=query.get('split', [None])[0],
        )


class RecordShard(BaseModel):
    model_config = ConfigDict(extra='forbid')
    source: CosFileRef
    sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    size: int = Field(ge=0, le=MAX_SHARD_BYTES)
    count: int = Field(ge=1)


class ReleaseManifest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    schema_version: Literal[1] = 1
    compiler_version: Literal[1] = 1
    namespace: str
    dataset: str
    version: str
    content_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    default_split: str
    splits: dict[str, list[RecordShard]]

    @model_validator(mode='after')
    def validate_names(self) -> ReleaseManifest:
        for value in (self.namespace, self.dataset, self.version, *self.splits):
            validate_component(value)
        if self.default_split not in self.splits or any(not shards for shards in self.splits.values()):
            raise ValueError('Manifest contains invalid splits')
        return self


class PublishedDatasetRef(BaseModel):
    backend: Literal['taiji', 'cos']
    data_uri: str
    resolved_version: str
    content_sha256: str
    item_count: int
    splits: dict[str, int]
    region: str | None = None
    manifest_sha256: str | None = None


class PublishOutcomeUnknownError(RuntimeError):
    """The release may exist; retry the same build to resolve its identity."""


async def _verify_object(client: AsyncCosClient, key: str, expected: str, size: int) -> str:
    metadata = await client.stat(key)
    if metadata.size != size or not metadata.etag:
        raise FileExistsError(f'COS object conflicts with build: {key}')
    digest = hashlib.sha256()
    for start in range(0, size, 1024 * 1024):
        digest.update(
            await client.read_range(key, start, min(start + 1024 * 1024, size), etag=metadata.etag, size=size)
        )
    if digest.hexdigest() != expected:
        raise FileExistsError(f'COS object conflicts with build: {key}')
    return metadata.etag


async def _put_immutable(client: AsyncCosClient, key: str, path: Path, *, expected_sha256: str | None = None) -> str:
    digest = await asyncio.to_thread(sha256_file, path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError('Frozen dataset attachment changed')
    try:
        await client.upload_file(path, key, headers={'x-cos-forbid-overwrite': 'true'})
    except CosServiceError as exc:
        if exc.status not in (409, 412):
            raise
    return await _verify_object(client, key, digest, path.stat().st_size)


class CosPublisher:
    def __init__(self, *, store: str, region: str) -> None:
        parsed = urlsplit(store)
        if parsed.scheme != 'cos' or parsed.query or parsed.fragment:
            raise ValueError('COS store must be cos://bucket/prefix')
        self.bucket = parsed.netloc
        self.prefix = unquote(parsed.path).strip('/')
        self.region = region
        CosFileRef(bucket=self.bucket, region=region, key=f'{self.prefix}/probe' if self.prefix else 'probe')

    async def publish(
        self,
        build: DatasetBuild,
        target: str,
        context: DataAccessContext,
    ) -> PublishedDatasetRef:
        from nexus.dataset.uri import parse_dataset_uri

        uri = parse_dataset_uri(target)
        await asyncio.to_thread(build.validate_records)
        for value in (uri.namespace, uri.dataset, uri.version):
            validate_component(value)
        if uri.version in ('latest', 'dev') or uri.split is not None:
            raise ValueError('COS publish requires an explicit version without a split selector')
        client = context.cos_client(self.bucket, self.region)
        if await client.bucket_versioning() is not None:
            raise ValueError('COS immutable releases require a bucket that has never enabled versioning')
        base = '/'.join(p for p in (self.prefix, uri.namespace, uri.dataset) if p)
        splits: dict[str, list[RecordShard]] = {}
        rendered_digest = hashlib.sha256(canonical_json({'default_split': build.default_split, 'format': 1}) + b'\n')
        with tempfile.TemporaryDirectory(prefix='nexus-cos-publish-') as work:
            directory = Path(work)
            for path in sorted((build.root / 'objects').glob('*.zip')):
                etag = await _put_immutable(client, f'{base}/objects/{path.name}', path, expected_sha256=path.stem)
                (directory / f'{path.stem}.etag').write_text(etag, encoding='utf-8')
            for name in sorted(build.splits):
                shards: list[RecordShard] = []
                buffer = bytearray()
                count = 0

                async def flush(name: str = name, shards: list[RecordShard] = shards) -> None:
                    nonlocal buffer, count
                    if not count:
                        return
                    key = f'{base}/snapshots/{build.content_sha256}/{name}/items-{len(shards):05d}.jsonl'
                    path = directory / 'shard.jsonl'
                    path.write_bytes(buffer)
                    await _put_immutable(client, key, path)
                    shards.append(
                        RecordShard(
                            source=CosFileRef(bucket=self.bucket, region=self.region, key=key),
                            sha256=hashlib.sha256(buffer).hexdigest(),
                            size=len(buffer),
                            count=count,
                        )
                    )
                    buffer = bytearray()
                    count = 0

                for record in build.iter_records(name):
                    rendered_digest.update(canonical_json({'split': name, 'record': record}) + b'\n')
                    if record.get('files') is not None:
                        files = TaskFilesSpec.model_validate(record['files'])
                        if (build.root / 'objects' / f'{files.sha256}.zip').stat().st_size != files.size:
                            raise ValueError('Frozen dataset attachment size changed')
                        record['files'] = files.model_copy(
                            update={
                                'source': CosFileRef(
                                    bucket=self.bucket,
                                    region=self.region,
                                    key=f'{base}/objects/{files.sha256}.zip',
                                    etag=(directory / f'{files.sha256}.etag').read_text(encoding='utf-8'),
                                ),
                            }
                        ).model_dump(exclude_none=True)
                    line = canonical_json(record) + b'\n'
                    if len(line) > MAX_SHARD_BYTES:
                        raise ValueError('Record exceeds COS shard size limit')
                    if len(buffer) + len(line) > MAX_SHARD_BYTES or count >= 1000:
                        await flush()
                    buffer.extend(line)
                    count += 1
                await flush()
                splits[name] = shards
            if rendered_digest.hexdigest() != build.content_sha256:
                raise ValueError('Frozen dataset records changed while publishing')
            manifest = ReleaseManifest(
                namespace=uri.namespace,
                dataset=uri.dataset,
                version=uri.version,
                content_sha256=build.content_sha256,
                default_split=build.default_split,
                splits=splits,
            )
            data = canonical_json(manifest.model_dump())
            if len(data) > MAX_MANIFEST_BYTES:
                raise ValueError('Release Manifest exceeds size limit')
            digest = hashlib.sha256(data).hexdigest()
            key = f'{base}/releases/{uri.version}/manifest.json'
            path = directory / 'manifest.json'
            path.write_bytes(data)
            try:
                await _put_immutable(client, key, path)
            except (aiohttp.ClientError, TimeoutError, CosClientError) as exc:
                try:
                    await _verify_object(client, key, digest, len(data))
                except FileExistsError:
                    raise
                except (aiohttp.ClientError, TimeoutError, CosClientError) as verify_exc:
                    raise PublishOutcomeUnknownError(
                        f'COS publish outcome unknown: {key}; expected sha256={digest}; retry the same build'
                    ) from verify_exc
                if isinstance(exc, CosServiceError) and exc.status == 403:
                    raise
        ref = CosDatasetRef(
            manifest_uri=f'cos://{self.bucket}/{quote(key, safe="/")}',
            region=self.region,
            manifest_sha256=digest,
        )
        return PublishedDatasetRef(
            backend='cos',
            data_uri=ref.data_uri,
            resolved_version=uri.version,
            content_sha256=build.content_sha256,
            item_count=build.item_count,
            splits=build.splits,
            region=self.region,
            manifest_sha256=digest,
        )


class CosDatasetReader(BaseDatasetReader):
    def __init__(self, reference: str, *, context: DataAccessContext | None = None) -> None:
        super().__init__()
        self.ref = CosDatasetRef.parse(reference)
        bound = context or current_data_access()
        self.context = bound or DataAccessContext.from_env()
        self._owns_context = bound is None
        self.manifest: ReleaseManifest | None = None

    async def _open(self) -> None:
        try:
            blob = CosBlobReader(self.ref.file_ref(), self.context)
            metadata = await blob.stat()
            if metadata.size > MAX_MANIFEST_BYTES:
                raise ValueError('Release Manifest exceeds size limit')
            raw = await blob.read_range(0, metadata.size)
            if hashlib.sha256(raw).hexdigest() != self.ref.manifest_sha256:
                raise ValueError('Release Manifest SHA256 mismatch')
            self.manifest = ReleaseManifest.model_validate_json(raw)
            if self.ref.split is not None and self.ref.split not in self.manifest.splits:
                raise DatasetReaderError('dataset_split_not_found', detail=self.ref.split)
        except BaseException:
            if self._owns_context:
                await self.context.close()
            raise

    async def _close(self, *_exc: object) -> None:
        if self._owns_context:
            await self.context.close()

    async def describe(self) -> DatasetDescriptor:
        self._require_open()
        assert self.manifest is not None
        return DatasetDescriptor(
            reference=self.ref.data_uri,
            resolved_reference=self.ref.data_uri,
            provider='cos',
            name=self.manifest.dataset,
            version=self.manifest.version,
            default_split=self.ref.split or self.manifest.default_split,
            splits=[
                DatasetSplitDescriptor(name=k, item_count=sum(s.count for s in v))
                for k, v in self.manifest.splits.items()
            ],
            metadata={
                'manifest_sha256': self.ref.manifest_sha256,
                'content_sha256': self.manifest.content_sha256,
                'region': self.ref.region,
            },
        )

    async def iter_item_ids(self, *, split: str | None = None, limit: int | None = None) -> AsyncIterator[str]:
        async for item in self.iter_items(split=split, limit=limit):
            yield item.id

    def iter_items(
        self,
        *,
        split: str | None = None,
        item_id: str | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[DatasetItem]:
        return self._scoped_items(self._iter_items(split=split, item_id=item_id, limit=limit))

    async def _iter_items(
        self,
        *,
        split: str | None = None,
        item_id: str | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[DatasetItem]:
        self._require_open()
        _validate_query(split=split, item_id=item_id, limit=limit)
        assert self.manifest is not None
        name = split or self.ref.split or self.manifest.default_split
        if name not in self.manifest.splits:
            raise DatasetReaderError('dataset_split_not_found', detail=name)
        if limit == 0:
            return
        count = 0
        for shard in self.manifest.splits[name]:
            blob = CosBlobReader(shard.source, self.context)
            if (await blob.stat()).size != shard.size:
                raise ValueError('Dataset record shard size mismatch')
            raw = await blob.read_range(0, shard.size)
            if hashlib.sha256(raw).hexdigest() != shard.sha256:
                raise ValueError('Dataset record shard SHA256 mismatch')
            lines = raw.splitlines()
            if len(lines) != shard.count:
                raise ValueError('Dataset record shard count mismatch')
            for line in lines:
                item = DatasetItem.from_mapping(json.loads(line))
                if item.files is not None and not item.files.is_remote:
                    raise ValueError('Published COS dataset contains local task files')
                if item_id is not None and item.id != item_id:
                    continue
                yield item
                count += 1
                if (limit is not None and count >= limit) or item_id is not None:
                    return
        if item_id is not None and count == 0:
            raise DatasetReaderError('dataset_item_not_found', detail=item_id)
