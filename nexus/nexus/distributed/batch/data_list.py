"""NXDL v1 binary container for local batch task-files ZIP archives."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import struct
import uuid

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, BinaryIO, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from nexus.dataset.zip_index import NexusZipError, read_zip_index_from_bytes, read_zip_index_from_file


NXDL_MAGIC = b'NXDL'
NXDL_VERSION = 1
NXDL_FILE_HEADER = struct.Struct('<4sHHQQQ')
NXDL_FRAME_HEADER = struct.Struct('<QQQ')
DEFAULT_MAX_DATA_FRAME_BYTES = 256 * 1024 * 1024
DEFAULT_MAX_DATA_LIST_BYTES = 4 * 1024 * 1024 * 1024
_COPY_CHUNK_SIZE = 1024 * 1024
_MAX_SIGNED_OFFSET = 2**63 - 1
_SHA256_PATTERN = r'^[0-9a-f]{64}$'


class DataListError(ValueError):
    """Stable, coded NXDL validation or limit failure."""

    def __init__(self, code: str, *, detail: str | None = None) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f'{code}:{detail}' if detail else code)


class DataRef(BaseModel):
    """Reference from one task-list v3 record to a complete NXDL frame."""

    model_config = ConfigDict(extra='forbid', frozen=True)

    version: Literal[1] = 1
    offset: int
    length: int
    sha256: str = Field(pattern=_SHA256_PATTERN)

    @field_validator('offset', 'length', mode='before')
    @classmethod
    def _strict_non_negative_integer(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > _MAX_SIGNED_OFFSET:
            raise ValueError('data-list offsets and lengths must be non-negative signed 64-bit integers')
        return value


@dataclass(frozen=True, slots=True)
class DataListManifest:
    path: Path
    sha256: str
    size_bytes: int
    frame_count: int


@dataclass(frozen=True, slots=True)
class DataListFrame:
    offset: int
    length: int
    sha256: str
    item_id: str
    metadata: dict[str, Any]
    metadata_size: int
    zip_offset: int
    zip_size: int
    zip_sha256: str
    zip_bytes: bytes | None = field(default=None, repr=False, compare=False)

    @property
    def ref(self) -> DataRef:
        return DataRef(version=1, offset=self.offset, length=self.length, sha256=self.sha256)


@dataclass(frozen=True, slots=True)
class DataListIndex:
    manifest: DataListManifest
    frames: tuple[DataListFrame, ...]
    by_offset: Mapping[int, DataListFrame] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, 'by_offset', MappingProxyType({frame.offset: frame for frame in self.frames}))


@dataclass(frozen=True, slots=True)
class IndexedFramePayload:
    encoded: bytes
    descriptor: DataListFrame


class DataListWriter:
    """Streaming, atomic NXDL v1 writer."""

    def __init__(
        self,
        output_path: str | Path,
        *,
        max_frame_bytes: int = DEFAULT_MAX_DATA_FRAME_BYTES,
        max_bytes: int = DEFAULT_MAX_DATA_LIST_BYTES,
    ) -> None:
        self.output_path = Path(output_path)
        self.max_frame_bytes = _positive_limit(max_frame_bytes, label='max_frame_bytes')
        self.max_bytes = _positive_limit(max_bytes, label='max_bytes')
        if self.max_bytes < NXDL_FILE_HEADER.size:
            raise DataListError('batch_data_list_exceeds_byte_limit')
        self.temporary = self.output_path.with_name(f'.{self.output_path.name}.{uuid.uuid4().hex}.tmp')
        self.stream: BinaryIO | None = None
        self.frame_count = 0
        self._manifest: DataListManifest | None = None

    def __enter__(self) -> DataListWriter:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.temporary.open('x+b')
        self.stream.write(NXDL_FILE_HEADER.pack(NXDL_MAGIC, NXDL_VERSION, 0, NXDL_FILE_HEADER.size, 0, 0))
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        completed = False
        try:
            if exc_type is None:
                self._finalize()
                completed = True
        finally:
            if self.stream is not None and not self.stream.closed:
                self.stream.close()
            self.stream = None
            try:
                if completed:
                    self.temporary.replace(self.output_path)
            finally:
                self.temporary.unlink(missing_ok=True)

    def write_frame(self, *, item_id: str, zip_path: str | Path, root_name: str | None = None) -> DataRef:
        stream = self._require_open()
        if not isinstance(item_id, str) or not item_id:
            raise DataListError('invalid_batch_data_list', detail='frame item_id must be a non-empty string')
        archive_path = Path(zip_path)
        try:
            archive_size, archive_sha256 = _file_identity_no_follow(archive_path)
            read_zip_index_from_file(archive_path)
        except (OSError, NexusZipError) as exc:
            raise DataListError('invalid_batch_data_list', detail=f'invalid ZIP for item {item_id}') from exc

        metadata: dict[str, Any] = {
            'item_id': item_id,
            'task_files': {
                'sha256': archive_sha256,
                'size_bytes': archive_size,
                'type': 'zip',
            },
            'version': 1,
        }
        if root_name is not None:
            from nexus.dataset.item import TaskFilesSpec

            TaskFilesSpec(type='zip', path='files.zip', root_name=root_name)
            metadata['task_files']['root_name'] = root_name
        encoded_metadata = _canonical_json_bytes(metadata)
        frame_size = NXDL_FRAME_HEADER.size + len(encoded_metadata) + archive_size
        if frame_size > self.max_frame_bytes:
            raise DataListError('batch_data_frame_exceeds_byte_limit')
        offset = stream.tell()
        if offset > _MAX_SIGNED_OFFSET or frame_size > _MAX_SIGNED_OFFSET:
            raise DataListError('invalid_batch_data_list', detail='frame offset or length exceeds signed 64-bit range')
        if offset + frame_size > self.max_bytes:
            raise DataListError('batch_data_list_exceeds_byte_limit')

        header = NXDL_FRAME_HEADER.pack(frame_size, len(encoded_metadata), archive_size)
        digest = hashlib.sha256()
        digest.update(header)
        digest.update(encoded_metadata)
        stream.write(header)
        stream.write(encoded_metadata)
        copied = 0
        archive_digest = hashlib.sha256()
        try:
            with _open_regular_no_follow(archive_path) as source:
                for chunk in iter(lambda: source.read(_COPY_CHUNK_SIZE), b''):
                    stream.write(chunk)
                    digest.update(chunk)
                    archive_digest.update(chunk)
                    copied += len(chunk)
        except OSError as exc:
            raise DataListError('invalid_batch_data_list', detail=f'could not read ZIP for item {item_id}') from exc
        if copied != archive_size or archive_digest.hexdigest() != archive_sha256:
            raise DataListError('invalid_batch_data_list', detail=f'ZIP changed while writing item {item_id}')

        self.frame_count += 1
        return DataRef(version=1, offset=offset, length=frame_size, sha256=digest.hexdigest())

    def result(self) -> DataListManifest:
        if self._manifest is None:
            raise RuntimeError('data-list writer has not completed successfully')
        return self._manifest

    def _require_open(self) -> BinaryIO:
        if self.stream is None or self.stream.closed or self._manifest is not None:
            raise RuntimeError('data-list writer is not open')
        return self.stream

    def _finalize(self) -> None:
        stream = self._require_open()
        if self.frame_count < 1:
            raise DataListError('invalid_batch_data_list', detail='data-list must contain at least one frame')
        size_bytes = stream.tell()
        if size_bytes > self.max_bytes:
            raise DataListError('batch_data_list_exceeds_byte_limit')
        stream.seek(0)
        stream.write(
            NXDL_FILE_HEADER.pack(
                NXDL_MAGIC,
                NXDL_VERSION,
                0,
                NXDL_FILE_HEADER.size,
                self.frame_count,
                0,
            )
        )
        stream.flush()
        os.fsync(stream.fileno())
        stream.seek(0)
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(_COPY_CHUNK_SIZE), b''):
            digest.update(chunk)
        self._manifest = DataListManifest(
            path=self.output_path,
            sha256=digest.hexdigest(),
            size_bytes=size_bytes,
            frame_count=self.frame_count,
        )


def scan_data_list(
    path: str | Path,
    *,
    max_frame_bytes: int = DEFAULT_MAX_DATA_FRAME_BYTES,
    max_bytes: int = DEFAULT_MAX_DATA_LIST_BYTES,
    max_items: int = 10_000,
) -> DataListIndex:
    """Sequentially validate an NXDL file while retaining only a bounded frame index."""

    archive_path = Path(path)
    max_frame_bytes = _positive_limit(max_frame_bytes, label='max_frame_bytes')
    max_bytes = _positive_limit(max_bytes, label='max_bytes')
    max_items = _positive_limit(max_items, label='max_items')
    try:
        file_stat = archive_path.lstat()
    except OSError as exc:
        raise DataListError('invalid_batch_data_list', detail='data-list file is unavailable') from exc
    if stat.S_ISLNK(file_stat.st_mode) or not stat.S_ISREG(file_stat.st_mode):
        raise DataListError('invalid_batch_data_list', detail='data-list path must be a regular file')
    size_bytes = file_stat.st_size
    if size_bytes > max_bytes:
        raise DataListError('batch_data_list_exceeds_byte_limit')
    if size_bytes < NXDL_FILE_HEADER.size:
        raise DataListError('invalid_batch_data_list', detail='truncated file header')

    frames: list[DataListFrame] = []
    file_digest = hashlib.sha256()
    try:
        with _open_regular_no_follow(archive_path) as stream:
            header = _read_exact(stream, NXDL_FILE_HEADER.size, detail='truncated file header')
            file_digest.update(header)
            magic, version, flags, header_size, frame_count, reserved = NXDL_FILE_HEADER.unpack(header)
            if (
                magic != NXDL_MAGIC
                or version != NXDL_VERSION
                or flags != 0
                or header_size != NXDL_FILE_HEADER.size
                or frame_count < 1
                or reserved != 0
            ):
                raise DataListError('invalid_batch_data_list', detail='invalid file header')
            if frame_count > max_items:
                raise DataListError('invalid_batch_data_list', detail='frame count exceeds item limit')

            cursor = NXDL_FILE_HEADER.size
            for _ in range(frame_count):
                frame_header = _read_exact(stream, NXDL_FRAME_HEADER.size, detail='truncated frame header')
                frame_size, metadata_size, zip_size = NXDL_FRAME_HEADER.unpack(frame_header)
                if frame_size != NXDL_FRAME_HEADER.size + metadata_size + zip_size:
                    raise DataListError('invalid_batch_data_list', detail='frame size fields do not match')
                if frame_size > max_frame_bytes:
                    raise DataListError('batch_data_frame_exceeds_byte_limit')
                if frame_size < NXDL_FRAME_HEADER.size or cursor + frame_size > size_bytes:
                    raise DataListError('invalid_batch_data_list', detail='frame range is outside data-list')
                body = _read_exact(
                    stream,
                    frame_size - NXDL_FRAME_HEADER.size,
                    detail='truncated frame body',
                )
                file_digest.update(frame_header)
                file_digest.update(body)
                encoded = frame_header + body
                validated = validate_frame(encoded, max_frame_bytes=max_frame_bytes)
                frames.append(replace(validated, offset=cursor, zip_bytes=None))
                cursor += frame_size
            if cursor != size_bytes:
                raise DataListError('invalid_batch_data_list', detail='trailing bytes after final frame')
    except DataListError:
        raise
    except OSError as exc:
        raise DataListError('invalid_batch_data_list', detail='could not read data-list') from exc

    manifest = DataListManifest(
        path=archive_path,
        sha256=file_digest.hexdigest(),
        size_bytes=size_bytes,
        frame_count=len(frames),
    )
    return DataListIndex(manifest=manifest, frames=tuple(frames))


def read_indexed_frame(
    path: str | Path,
    ref: DataRef,
    *,
    expected_frame: DataListFrame,
    expected_item_id: str,
    max_frame_bytes: int = DEFAULT_MAX_DATA_FRAME_BYTES,
) -> IndexedFramePayload:
    """Read one frame whose structure and ZIP were validated by an earlier index scan."""

    if ref != expected_frame.ref or expected_item_id != expected_frame.item_id:
        raise DataListError('batch_task_list_data_ref_invalid', detail='frame index identity mismatch')
    max_frame_bytes = _positive_limit(max_frame_bytes, label='max_frame_bytes')
    if ref.length > max_frame_bytes:
        raise DataListError('batch_data_frame_exceeds_byte_limit')
    archive_path = Path(path)
    try:
        file_size = archive_path.lstat().st_size
        if ref.offset < NXDL_FILE_HEADER.size or ref.length < NXDL_FRAME_HEADER.size:
            raise DataListError('batch_task_list_data_ref_invalid', detail='invalid frame range')
        end = ref.offset + ref.length
        if end < ref.offset or end > file_size:
            raise DataListError('batch_task_list_data_ref_invalid', detail='frame range is outside data-list')
        with _open_regular_no_follow(archive_path) as stream:
            stream.seek(ref.offset)
            encoded = _read_exact(stream, ref.length, detail='truncated referenced frame')
    except DataListError:
        raise
    except OSError as exc:
        raise DataListError('batch_task_list_data_ref_invalid', detail='could not read referenced frame') from exc
    if hashlib.sha256(encoded).hexdigest() != expected_frame.sha256:
        raise DataListError('batch_task_list_data_ref_invalid', detail='frame digest mismatch')
    frame_size = NXDL_FRAME_HEADER.unpack_from(encoded)[0]
    if frame_size != expected_frame.length:
        raise DataListError('batch_task_list_data_ref_invalid', detail='frame length mismatch')
    return IndexedFramePayload(encoded=encoded, descriptor=expected_frame)


def validate_frame(
    encoded: bytes,
    *,
    expected_ref: DataRef | None = None,
    expected_item_id: str | None = None,
    max_frame_bytes: int = DEFAULT_MAX_DATA_FRAME_BYTES,
) -> DataListFrame:
    """Validate one complete NXDL frame and return its ZIP payload and identity."""

    max_frame_bytes = _positive_limit(max_frame_bytes, label='max_frame_bytes')
    if len(encoded) > max_frame_bytes:
        raise DataListError('batch_data_frame_exceeds_byte_limit')
    if len(encoded) < NXDL_FRAME_HEADER.size:
        raise DataListError('invalid_batch_data_list', detail='truncated frame header')
    frame_size, metadata_size, zip_size = NXDL_FRAME_HEADER.unpack_from(encoded)
    if frame_size != NXDL_FRAME_HEADER.size + metadata_size + zip_size or frame_size != len(encoded):
        raise DataListError('invalid_batch_data_list', detail='frame size fields do not match')
    metadata_end = NXDL_FRAME_HEADER.size + metadata_size
    metadata_bytes = encoded[NXDL_FRAME_HEADER.size : metadata_end]
    zip_bytes = encoded[metadata_end:]
    try:
        metadata = json.loads(metadata_bytes.decode('utf-8'))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DataListError('invalid_batch_data_list', detail='frame metadata is not valid UTF-8 JSON') from exc
    if not isinstance(metadata, dict) or _canonical_json_bytes(metadata) != metadata_bytes:
        raise DataListError('invalid_batch_data_list', detail='frame metadata is not canonical JSON')
    item_id, declared_zip_size, declared_zip_sha256 = _validate_frame_metadata(metadata)
    if declared_zip_size != zip_size or len(zip_bytes) != zip_size:
        raise DataListError('invalid_batch_data_list', detail='ZIP size identity mismatch')
    actual_zip_sha256 = hashlib.sha256(zip_bytes).hexdigest()
    if actual_zip_sha256 != declared_zip_sha256:
        raise DataListError('invalid_batch_data_list', detail='ZIP sha256 identity mismatch')
    try:
        read_zip_index_from_bytes(zip_bytes)
    except NexusZipError as exc:
        raise DataListError('invalid_batch_data_list', detail='frame ZIP index is invalid') from exc

    digest = hashlib.sha256(encoded).hexdigest()
    if expected_ref is not None and (expected_ref.length != len(encoded) or expected_ref.sha256 != digest):
        raise DataListError('batch_task_list_data_ref_invalid', detail='frame reference identity mismatch')
    if expected_item_id is not None and expected_item_id != item_id:
        raise DataListError('batch_task_list_data_ref_invalid', detail='frame item identity mismatch')
    return DataListFrame(
        offset=expected_ref.offset if expected_ref is not None else 0,
        length=len(encoded),
        sha256=digest,
        item_id=item_id,
        metadata=metadata,
        metadata_size=metadata_size,
        zip_offset=NXDL_FRAME_HEADER.size + metadata_size,
        zip_size=zip_size,
        zip_sha256=actual_zip_sha256,
        zip_bytes=zip_bytes,
    )


def _validate_frame_metadata(metadata: dict[str, Any]) -> tuple[str, int, str]:
    version = metadata.get('version')
    if (
        set(metadata) != {'item_id', 'task_files', 'version'}
        or isinstance(version, bool)
        or not isinstance(version, int)
        or version != 1
    ):
        raise DataListError('invalid_batch_data_list', detail='invalid frame metadata schema')
    item_id = metadata.get('item_id')
    task_files = metadata.get('task_files')
    if not isinstance(item_id, str) or not item_id or not isinstance(task_files, dict):
        raise DataListError('invalid_batch_data_list', detail='invalid frame metadata identity')
    if set(task_files) - {'root_name'} != {'type', 'sha256', 'size_bytes'} or task_files.get('type') != 'zip':
        raise DataListError('invalid_batch_data_list', detail='invalid frame task_files schema')
    if 'root_name' in task_files:
        from nexus.dataset.item import TaskFilesSpec

        try:
            TaskFilesSpec(type='zip', path='files.zip', root_name=task_files['root_name'])
        except ValueError as exc:
            raise DataListError('invalid_batch_data_list', detail='invalid root_name') from exc
    size_bytes = task_files.get('size_bytes')
    sha256 = task_files.get('sha256')
    if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes < 0:
        raise DataListError('invalid_batch_data_list', detail='invalid frame ZIP size')
    if (
        not isinstance(sha256, str)
        or len(sha256) != 64
        or any(character not in '0123456789abcdef' for character in sha256)
    ):
        raise DataListError('invalid_batch_data_list', detail='invalid frame ZIP sha256')
    return item_id, size_bytes, sha256


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(',', ':'),
        ).encode('utf-8')
    except (TypeError, ValueError) as exc:
        raise DataListError('invalid_batch_data_list', detail='metadata is not JSON serializable') from exc


def _positive_limit(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise DataListError('invalid_batch_data_list', detail=f'{label} must be a positive integer')
    return value


def _read_exact(stream: BinaryIO, size: int, *, detail: str) -> bytes:
    data = stream.read(size)
    if len(data) != size:
        raise DataListError('invalid_batch_data_list', detail=detail)
    return data


def _open_regular_no_follow(path: Path):
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0)
    descriptor = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise OSError(f'path is not a regular file: {path}')
        return os.fdopen(descriptor, 'rb')
    except BaseException:
        os.close(descriptor)
        raise


def _file_identity_no_follow(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with _open_regular_no_follow(path) as stream:
        for chunk in iter(lambda: stream.read(_COPY_CHUNK_SIZE), b''):
            digest.update(chunk)
            size += len(chunk)
    return size, digest.hexdigest()


__all__ = [
    'DEFAULT_MAX_DATA_FRAME_BYTES',
    'DEFAULT_MAX_DATA_LIST_BYTES',
    'DataListError',
    'DataListFrame',
    'DataListIndex',
    'DataListManifest',
    'DataListWriter',
    'DataRef',
    'IndexedFramePayload',
    'NXDL_FILE_HEADER',
    'NXDL_FRAME_HEADER',
    'read_indexed_frame',
    'scan_data_list',
    'validate_frame',
]
