"""Nexus ZIP writer and central-directory range reader."""

from __future__ import annotations

import binascii
import fnmatch
import hashlib
import io
import logging
import os
import stat
import struct
import zipfile
import zlib

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final

from nexus.dataset._format import format_bytes_detail


NEXUS_EXTRA_ID: Final[int] = 0x4E58
NEXUS_EXTRA_VERSION: Final[int] = 1
NEXUS_EXTRA_DATA_SIZE: Final[int] = 46
NEXUS_EXTRA_SIZE: Final[int] = 4 + NEXUS_EXTRA_DATA_SIZE
_NEXUS_EXTRA_DATA_OFFSET: Final[int] = 4 + 1 + 1 + 4
_NEXUS_EXTRA_STRUCT: Final[str] = '<HHBBIQ32s'

_EOCD_SIGNATURE: Final[bytes] = b'PK\x05\x06'
_ZIP64_LOCATOR_SIGNATURE: Final[bytes] = b'PK\x06\x07'
_ZIP64_EOCD_SIGNATURE: Final[bytes] = b'PK\x06\x06'
_CENTRAL_DIR_SIGNATURE: Final[bytes] = b'PK\x01\x02'
_LOCAL_FILE_SIGNATURE: Final[bytes] = b'PK\x03\x04'
_ZIP64_EXTRA_ID: Final[int] = 0x0001
_MAX_EOCD_SEARCH: Final[int] = 22 + 65535
_SAMPLE_SIZE: Final[int] = 10 * 1024 * 1024
_CHUNK_SIZE: Final[int] = 1024 * 1024

logger = logging.getLogger(__name__)


class NexusZipError(ValueError):
    """Raised when a ZIP file is not a supported Nexus task-files archive."""


@dataclass(frozen=True)
class ZipEntry:
    """Central-directory metadata for one ZIP file entry."""

    path: str
    compression_method: int
    compressed_size: int
    uncompressed_size: int
    crc32: int
    data_offset: int
    mode: int
    sha256: str | None = None


@dataclass(frozen=True)
class ZipIndex:
    """In-memory ZIP index built from the central directory."""

    entries: dict[str, ZipEntry]

    def list(self, pattern: str = '**/*') -> Sequence[str]:
        return sorted(path for path in self.entries if matches_path_pattern(path, pattern))


def matches_path_pattern(path: str, pattern: str) -> bool:
    """Match task-file paths with a recursive glob that includes root files."""

    if pattern == '**/*' or fnmatch.fnmatchcase(path, pattern):
        return True
    return pattern.startswith('**/') and fnmatch.fnmatchcase(path, pattern[3:])


RangeReader = Callable[[int, int], Awaitable[bytes]]


def write_nexus_zip_from_directory(path: Path) -> bytes:
    """Return Nexus task-files ZIP bytes for all files under *path*."""

    files = _safe_directory_files(path)

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', allowZip64=True) as archive:
        for child in files:
            arcname = child.relative_to(path).as_posix()
            mode = stat.S_IMODE(child.lstat().st_mode)
            info = zipfile.ZipInfo.from_file(child, arcname)
            info.compress_type = _choose_compression(child)
            info.extra = _nexus_extra(mode=mode, data_offset=0, sha256=_sha256_file_digest(child))
            with _open_regular_file_no_follow(child) as src, archive.open(info, 'w', force_zip64=True) as dst:
                for chunk in iter(lambda: src.read(_CHUNK_SIZE), b''):
                    dst.write(chunk)

    data = bytearray(buffer.getvalue())
    _patch_central_directory_nexus_offsets(data)
    return bytes(data)


def write_nexus_zip_directory_to_file(path: Path, target: Path, *, deterministic: bool = False) -> None:
    """Write Nexus task-files ZIP bytes directly to *target*."""

    files = _safe_directory_files(path)
    before = {child: child.stat() for child in files} if deterministic else {}

    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, 'w+b') as raw:
        with zipfile.ZipFile(raw, 'w', allowZip64=True) as archive:
            for child in files:
                arcname = child.relative_to(path).as_posix()
                mode = stat.S_IMODE(child.lstat().st_mode)
                info = zipfile.ZipInfo.from_file(child, arcname)
                if deterministic:
                    info.date_time = (1980, 1, 1, 0, 0, 0)
                info.compress_type = _choose_compression(child)
                info.extra = _nexus_extra(mode=mode, data_offset=0, sha256=_sha256_file_digest(child))
                with _open_regular_file_no_follow(child) as src, archive.open(info, 'w', force_zip64=True) as dst:
                    written = 0
                    for chunk in iter(lambda: src.read(_CHUNK_SIZE), b''):
                        written += len(chunk)
                        if deterministic and written > before[child].st_size:
                            raise NexusZipError('Task file grew during build')
                        dst.write(chunk)
                    if deterministic and written != before[child].st_size:
                        raise NexusZipError('Task file changed during build')
    _patch_central_directory_nexus_offsets_in_file(target)
    if deterministic and (
        files != _safe_directory_files(path)
        or any(
            (old.st_ino, old.st_size, old.st_mtime_ns, old.st_ctime_ns)
            != (new.st_ino, new.st_size, new.st_mtime_ns, new.st_ctime_ns)
            for child, old in before.items()
            for new in [child.stat()]
        )
    ):
        target.unlink(missing_ok=True)
        raise NexusZipError('Task-files directory changed during build')


def _safe_directory_files(path: Path) -> list[Path]:
    try:
        root_stat = path.lstat()
    except FileNotFoundError:
        raise NotADirectoryError(f'Path is not a directory: {path}') from None
    if stat.S_ISLNK(root_stat.st_mode):
        raise NexusZipError(f'Task-files directory must not be a symlink: {path}')
    if not stat.S_ISDIR(root_stat.st_mode):
        raise NotADirectoryError(f'Path is not a directory: {path}')

    root = path.resolve(strict=True)
    files: list[Path] = []
    for current, directory_names, file_names in os.walk(path, topdown=True, followlinks=False):
        directory_names.sort()
        file_names.sort()
        current_path = Path(current)
        for name in [*directory_names, *file_names]:
            child = current_path / name
            child_stat = child.lstat()
            if stat.S_ISLNK(child_stat.st_mode):
                raise NexusZipError(f'Task-files directory contains a symlink: {child}')
            if stat.S_ISDIR(child_stat.st_mode):
                continue
            if not stat.S_ISREG(child_stat.st_mode):
                raise NexusZipError(f'Task-files directory contains an unsupported non-regular file: {child}')
            resolved = child.resolve(strict=True)
            try:
                resolved.relative_to(root)
            except ValueError as exc:
                raise NexusZipError(f'Task-files path escapes directory root: {child}') from exc
            files.append(child)
    return sorted(files, key=lambda child: child.relative_to(path).as_posix())


def _open_regular_file_no_follow(path: Path):
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0)
    descriptor = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise NexusZipError(f'Task-files path is not a regular file: {path}')
        return os.fdopen(descriptor, 'rb')
    except BaseException:
        os.close(descriptor)
        raise


def _sha256_file_digest(path: Path) -> bytes:
    digest = hashlib.sha256()
    with _open_regular_file_no_follow(path) as f:
        for chunk in iter(lambda: f.read(_CHUNK_SIZE), b''):
            digest.update(chunk)
    return digest.digest()


def _choose_compression(path: Path) -> int:
    size = path.lstat().st_size
    if size <= _SAMPLE_SIZE:
        return zipfile.ZIP_STORED
    with _open_regular_file_no_follow(path) as f:
        sample = f.read(_SAMPLE_SIZE)
    compressed = zlib.compress(sample, level=6)
    if len(compressed) <= len(sample) * 0.5:
        return zipfile.ZIP_DEFLATED
    return zipfile.ZIP_STORED


def _nexus_extra(*, mode: int, data_offset: int, sha256: bytes) -> bytes:
    return struct.pack(
        _NEXUS_EXTRA_STRUCT,
        NEXUS_EXTRA_ID,
        NEXUS_EXTRA_DATA_SIZE,
        NEXUS_EXTRA_VERSION,
        0,
        mode,
        data_offset,
        sha256,
    )


def _patch_central_directory_nexus_offsets(data: bytearray) -> None:
    eocd = _parse_eocd_from_bytes(bytes(data))
    cd_offset = eocd.central_directory_offset
    cd_size = eocd.central_directory_size
    position = cd_offset
    end = cd_offset + cd_size
    while position < end:
        if data[position : position + 4] != _CENTRAL_DIR_SIGNATURE:
            raise NexusZipError(f'Invalid central directory signature at offset {position}')
        name_len, extra_len, comment_len = struct.unpack_from('<HHH', data, position + 28)
        local_header_offset = struct.unpack_from('<I', data, position + 42)[0]
        name_start = position + 46
        extra_start = name_start + name_len
        extra_end = extra_start + extra_len
        name = bytes(data[name_start:extra_start]).decode('utf-8')
        if local_header_offset == 0xFFFFFFFF:
            local_header_offset = _zip64_local_header_offset(bytes(data[extra_start:extra_end]))
        data_offset = _local_data_offset(data, local_header_offset)
        nexus_offset = _find_extra_field(data, extra_start, extra_end, NEXUS_EXTRA_ID)
        if nexus_offset is None:
            raise NexusZipError(f'Nexus extra field missing in generated entry: {name}')
        struct.pack_into('<Q', data, nexus_offset + _NEXUS_EXTRA_DATA_OFFSET, data_offset)
        position = extra_end + comment_len

    if position != end:
        raise NexusZipError('Central directory parsing ended at unexpected offset')


def _patch_central_directory_nexus_offsets_in_file(path: Path) -> None:
    with open(path, 'r+b') as f:
        eocd = _read_central_directory_info_from_file(f)
        position = eocd.central_directory_offset
        end = eocd.central_directory_offset + eocd.central_directory_size
        while position < end:
            f.seek(position)
            header = f.read(46)
            if len(header) < 46 or header[:4] != _CENTRAL_DIR_SIGNATURE:
                raise NexusZipError(f'Invalid central directory signature at offset {position}')
            name_len, extra_len, comment_len = struct.unpack_from('<HHH', header, 28)
            local_header_offset = struct.unpack_from('<I', header, 42)[0]
            name = f.read(name_len).decode('utf-8')
            extra = f.read(extra_len)
            if local_header_offset == 0xFFFFFFFF:
                local_header_offset = _zip64_local_header_offset(extra)
            data_offset = _local_data_offset_in_file(f, local_header_offset)
            nexus_offset = _find_extra_field_bytes(extra, NEXUS_EXTRA_ID)
            if nexus_offset is None:
                raise NexusZipError(f'Nexus extra field missing in generated entry: {name}')
            f.seek(position + 46 + name_len + nexus_offset + _NEXUS_EXTRA_DATA_OFFSET)
            f.write(struct.pack('<Q', data_offset))
            position += 46 + name_len + extra_len + comment_len

        if position != end:
            raise NexusZipError('Central directory parsing ended at unexpected offset')


def _read_central_directory_info_from_file(f) -> _CentralDirectoryInfo:
    f.seek(0, os.SEEK_END)
    size = f.tell()
    tail_size = min(size, _MAX_EOCD_SEARCH)
    f.seek(size - tail_size)
    tail = f.read(tail_size)
    eocd_offset = tail.rfind(_EOCD_SIGNATURE)
    if eocd_offset < 0:
        raise NexusZipError('ZIP EOCD not found')
    absolute_eocd = size - tail_size + eocd_offset
    info = _parse_eocd_record(tail, eocd_offset)
    if not _needs_zip64(info):
        return info

    locator_offset = absolute_eocd - 20
    if locator_offset < 0:
        raise NexusZipError('Zip64 locator missing')
    f.seek(locator_offset)
    locator = f.read(20)
    if locator[:4] != _ZIP64_LOCATOR_SIGNATURE:
        raise NexusZipError('Zip64 locator signature missing')
    _sig, disk_no, zip64_eocd_offset, disk_count = struct.unpack('<4sIQI', locator)
    if disk_no != 0 or disk_count != 1:
        raise NexusZipError('Multi-disk Zip64 archives are not supported')
    f.seek(zip64_eocd_offset)
    header = f.read(56)
    if len(header) < 56 or header[:4] != _ZIP64_EOCD_SIGNATURE:
        raise NexusZipError('Zip64 EOCD signature missing')
    (
        _sig,
        _record_size,
        _version_made,
        _version_needed,
        zip64_disk_no,
        zip64_cd_disk,
        _entries_disk,
        entries_total,
        cd_size,
        cd_offset,
    ) = struct.unpack_from('<4sQHHIIQQQQ', header, 0)
    if zip64_disk_no != 0 or zip64_cd_disk != 0:
        raise NexusZipError('Multi-disk Zip64 archives are not supported')
    return _CentralDirectoryInfo(
        central_directory_offset=cd_offset,
        central_directory_size=cd_size,
        entry_count=entries_total,
    )


def _local_data_offset_in_file(f, local_header_offset: int) -> int:
    f.seek(local_header_offset)
    header = f.read(30)
    if len(header) < 30 or header[:4] != _LOCAL_FILE_SIGNATURE:
        raise NexusZipError(f'Invalid local file header at offset {local_header_offset}')
    name_len, extra_len = struct.unpack_from('<HH', header, 26)
    return local_header_offset + 30 + name_len + extra_len


def _local_data_offset(data: bytearray, local_header_offset: int) -> int:
    if data[local_header_offset : local_header_offset + 4] != _LOCAL_FILE_SIGNATURE:
        raise NexusZipError(f'Invalid local file header at offset {local_header_offset}')
    name_len, extra_len = struct.unpack_from('<HH', data, local_header_offset + 26)
    return local_header_offset + 30 + name_len + extra_len


def _zip64_local_header_offset(extra: bytes) -> int:
    fields = _parse_extra_fields(extra)
    zip64 = fields.get(_ZIP64_EXTRA_ID)
    if not zip64 or len(zip64) < 24:
        raise NexusZipError('Zip64 local header offset missing from central directory extra field')
    return struct.unpack_from('<Q', zip64, len(zip64) - 8)[0]


def _find_extra_field(data: bytearray, start: int, end: int, field_id: int) -> int | None:
    position = start
    while position + 4 <= end:
        current_id, size = struct.unpack_from('<HH', data, position)
        data_start = position + 4
        data_end = data_start + size
        if data_end > end:
            raise NexusZipError('Malformed ZIP extra field')
        if current_id == field_id:
            return position
        position = data_end
    return None


def _find_extra_field_bytes(extra: bytes, field_id: int) -> int | None:
    position = 0
    while position + 4 <= len(extra):
        current_id, size = struct.unpack_from('<HH', extra, position)
        data_start = position + 4
        data_end = data_start + size
        if data_end > len(extra):
            raise NexusZipError('Malformed ZIP extra field')
        if current_id == field_id:
            return position
        position = data_end
    return None


@dataclass(frozen=True)
class _CentralDirectoryInfo:
    central_directory_offset: int
    central_directory_size: int
    entry_count: int


@dataclass(frozen=True)
class _NexusExtra:
    mode: int
    data_offset: int
    sha256: str | None


async def read_zip_index(size: int, read_range: RangeReader) -> ZipIndex:
    """Read only the ZIP tail and central directory, then return a Nexus index."""

    if size < 22:
        raise NexusZipError('ZIP file is too small')
    tail_size = min(size, _MAX_EOCD_SEARCH)
    tail_offset = size - tail_size
    logger.debug(
        'zip index range: phase=eocd_tail archive=%s range=%s-%s length=%s',
        format_bytes_detail(size),
        tail_offset,
        size,
        format_bytes_detail(tail_size),
    )
    tail = await read_range(tail_offset, size)
    eocd_offset = tail.rfind(_EOCD_SIGNATURE)
    if eocd_offset < 0:
        raise NexusZipError('ZIP EOCD not found')

    eocd_absolute = tail_offset + eocd_offset
    info = _parse_eocd_record(tail, eocd_offset)
    if _needs_zip64(info):
        info = await _read_zip64_info(size=size, eocd_offset=eocd_absolute, read_range=read_range)
    _validate_central_directory_bounds(info, archive_size=size)

    logger.debug(
        'zip index range: phase=central_directory archive=%s range=%s-%s length=%s entries=%s',
        format_bytes_detail(size),
        info.central_directory_offset,
        info.central_directory_offset + info.central_directory_size,
        format_bytes_detail(info.central_directory_size),
        info.entry_count,
    )
    directory = await read_range(
        info.central_directory_offset,
        info.central_directory_offset + info.central_directory_size,
    )
    entries = _parse_central_directory(directory, info)
    _validate_entry_bounds(entries, archive_size=size, central_directory_offset=info.central_directory_offset)
    return ZipIndex(entries=entries)


def read_zip_index_from_file(path: Path) -> ZipIndex:
    with _open_regular_file_no_follow(path) as stream:
        stream.seek(0, os.SEEK_END)
        archive_size = stream.tell()
        info = _read_central_directory_info_from_file(stream)
        _validate_central_directory_bounds(info, archive_size=archive_size)
        stream.seek(info.central_directory_offset)
        directory = stream.read(info.central_directory_size)
    if len(directory) != info.central_directory_size:
        raise NexusZipError('Truncated central directory')
    entries = _parse_central_directory(directory, info)
    _validate_entry_bounds(
        entries,
        archive_size=archive_size,
        central_directory_offset=info.central_directory_offset,
    )
    return ZipIndex(entries=entries)


def read_zip_index_from_bytes(data: bytes) -> ZipIndex:
    eocd_offset = data.rfind(_EOCD_SIGNATURE)
    if eocd_offset < 0:
        raise NexusZipError('ZIP EOCD not found')
    info = _parse_eocd_record(data, eocd_offset)
    if _needs_zip64(info):
        info = _read_zip64_info_from_bytes(data, eocd_offset=eocd_offset)
    _validate_central_directory_bounds(info, archive_size=len(data))
    directory = data[info.central_directory_offset : info.central_directory_offset + info.central_directory_size]
    if len(directory) != info.central_directory_size:
        raise NexusZipError('Truncated central directory')
    entries = _parse_central_directory(directory, info)
    _validate_entry_bounds(entries, archive_size=len(data), central_directory_offset=info.central_directory_offset)
    return ZipIndex(entries=entries)


def _read_zip64_info_from_bytes(data: bytes, *, eocd_offset: int) -> _CentralDirectoryInfo:
    locator_offset = eocd_offset - 20
    if locator_offset < 0:
        raise NexusZipError('Zip64 locator missing')
    locator = data[locator_offset:eocd_offset]
    if locator[:4] != _ZIP64_LOCATOR_SIGNATURE:
        raise NexusZipError('Zip64 locator signature missing')
    _sig, disk_no, zip64_eocd_offset, disk_count = struct.unpack('<4sIQI', locator)
    if disk_no != 0 or disk_count != 1:
        raise NexusZipError('Multi-disk Zip64 archives are not supported')
    header = data[zip64_eocd_offset : zip64_eocd_offset + 56]
    if len(header) < 56 or header[:4] != _ZIP64_EOCD_SIGNATURE:
        raise NexusZipError('Zip64 EOCD signature missing')
    (
        _sig,
        _record_size,
        _version_made,
        _version_needed,
        zip64_disk_no,
        zip64_cd_disk,
        _entries_disk,
        entries_total,
        cd_size,
        cd_offset,
    ) = struct.unpack_from('<4sQHHIIQQQQ', header, 0)
    if zip64_disk_no != 0 or zip64_cd_disk != 0:
        raise NexusZipError('Multi-disk Zip64 archives are not supported')
    return _CentralDirectoryInfo(
        central_directory_offset=cd_offset,
        central_directory_size=cd_size,
        entry_count=entries_total,
    )


def _parse_eocd_from_bytes(data: bytes) -> _CentralDirectoryInfo:
    eocd_offset = data.rfind(_EOCD_SIGNATURE)
    if eocd_offset < 0:
        raise NexusZipError('ZIP EOCD not found')
    return _parse_eocd_record(data, eocd_offset)


def _parse_eocd_record(data: bytes, offset: int) -> _CentralDirectoryInfo:
    if len(data) < offset + 22:
        raise NexusZipError('Truncated ZIP EOCD')
    (
        _signature,
        disk_no,
        cd_disk_no,
        entry_count_disk,
        entry_count,
        cd_size,
        cd_offset,
        comment_len,
    ) = struct.unpack_from('<4sHHHHIIH', data, offset)
    if len(data) < offset + 22 + comment_len:
        raise NexusZipError('Truncated ZIP EOCD comment')
    if len(data) != offset + 22 + comment_len:
        raise NexusZipError('Trailing bytes after ZIP EOCD')
    if disk_no != 0 or cd_disk_no != 0:
        raise NexusZipError('Multi-disk ZIP archives are not supported')
    if entry_count_disk != entry_count and entry_count_disk != 0xFFFF and entry_count != 0xFFFF:
        raise NexusZipError('Multi-disk ZIP entry counts are not supported')
    return _CentralDirectoryInfo(
        central_directory_offset=cd_offset,
        central_directory_size=cd_size,
        entry_count=max(entry_count, entry_count_disk),
    )


def _needs_zip64(info: _CentralDirectoryInfo) -> bool:
    return (
        info.central_directory_offset == 0xFFFFFFFF
        or info.central_directory_size == 0xFFFFFFFF
        or info.entry_count == 0xFFFF
    )


async def _read_zip64_info(
    *,
    size: int,
    eocd_offset: int,
    read_range: RangeReader,
) -> _CentralDirectoryInfo:
    locator_offset = eocd_offset - 20
    if locator_offset < 0:
        raise NexusZipError('Zip64 locator missing')
    logger.debug(
        'zip index range: phase=zip64_locator archive=%s range=%s-%s length=%s',
        format_bytes_detail(size),
        locator_offset,
        eocd_offset,
        format_bytes_detail(eocd_offset - locator_offset),
    )
    locator = await read_range(locator_offset, eocd_offset)
    if locator[:4] != _ZIP64_LOCATOR_SIGNATURE:
        raise NexusZipError('Zip64 locator signature missing')
    _sig, disk_no, zip64_eocd_offset, disk_count = struct.unpack('<4sIQI', locator)
    if disk_no != 0 or disk_count != 1:
        raise NexusZipError('Multi-disk Zip64 archives are not supported')
    if zip64_eocd_offset >= size:
        raise NexusZipError('Zip64 EOCD offset is outside archive')

    logger.debug(
        'zip index range: phase=zip64_eocd archive=%s range=%s-%s length=%s',
        format_bytes_detail(size),
        zip64_eocd_offset,
        zip64_eocd_offset + 56,
        format_bytes_detail(56),
    )
    header = await read_range(zip64_eocd_offset, zip64_eocd_offset + 56)
    if header[:4] != _ZIP64_EOCD_SIGNATURE:
        raise NexusZipError('Zip64 EOCD signature missing')
    (
        _sig,
        _record_size,
        _version_made,
        _version_needed,
        zip64_disk_no,
        zip64_cd_disk,
        _entries_disk,
        entries_total,
        cd_size,
        cd_offset,
    ) = struct.unpack_from('<4sQHHIIQQQQ', header, 0)
    if zip64_disk_no != 0 or zip64_cd_disk != 0:
        raise NexusZipError('Multi-disk Zip64 archives are not supported')
    return _CentralDirectoryInfo(
        central_directory_offset=cd_offset,
        central_directory_size=cd_size,
        entry_count=entries_total,
    )


def _parse_central_directory(directory: bytes, info: _CentralDirectoryInfo) -> dict[str, ZipEntry]:
    entries: dict[str, ZipEntry] = {}
    position = 0
    parsed = 0
    while position < len(directory):
        if directory[position : position + 4] != _CENTRAL_DIR_SIGNATURE:
            raise NexusZipError(f'Invalid central directory signature at relative offset {position}')
        if len(directory) < position + 46:
            raise NexusZipError('Truncated central directory entry')
        flags = struct.unpack_from('<H', directory, position + 8)[0]
        method = struct.unpack_from('<H', directory, position + 10)[0]
        crc32 = struct.unpack_from('<I', directory, position + 16)[0]
        compressed_size = struct.unpack_from('<I', directory, position + 20)[0]
        uncompressed_size = struct.unpack_from('<I', directory, position + 24)[0]
        name_len, extra_len, comment_len = struct.unpack_from('<HHH', directory, position + 28)
        external_attr = struct.unpack_from('<I', directory, position + 38)[0]
        local_header_offset = struct.unpack_from('<I', directory, position + 42)[0]
        name_start = position + 46
        extra_start = name_start + name_len
        comment_start = extra_start + extra_len
        next_position = comment_start + comment_len
        if next_position > len(directory):
            raise NexusZipError('Truncated central directory entry payload')

        try:
            name = directory[name_start:extra_start].decode('utf-8')
        except UnicodeDecodeError as exc:
            raise NexusZipError('ZIP entry name is not valid UTF-8') from exc
        normalized_name = _validate_zip_entry_path(name)
        extra = directory[extra_start:comment_start]
        if flags & 0x1:
            raise NexusZipError(f'Encrypted ZIP entry is not supported: {name}')
        unix_mode = external_attr >> 16
        file_type = stat.S_IFMT(unix_mode)
        if file_type not in {0, stat.S_IFREG, stat.S_IFDIR}:
            raise NexusZipError(f'Unsupported ZIP entry file type: {name}')
        if name.endswith('/'):
            if file_type not in {0, stat.S_IFDIR}:
                raise NexusZipError(f'ZIP directory entry has invalid file type: {name}')
            position = next_position
            parsed += 1
            continue

        zip64_values = _parse_zip64_extra(
            extra,
            uncompressed_size=uncompressed_size,
            compressed_size=compressed_size,
            local_header_offset=local_header_offset,
        )
        uncompressed_size = zip64_values.get('uncompressed_size', uncompressed_size)
        compressed_size = zip64_values.get('compressed_size', compressed_size)
        local_header_offset = zip64_values.get('local_header_offset', local_header_offset)

        nexus = _parse_nexus_extra(extra, name)
        if method not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}:
            raise NexusZipError(f'Unsupported ZIP compression method {method} for entry {name}')

        if normalized_name in entries:
            raise NexusZipError(f'Duplicate ZIP entry path: {normalized_name}')
        entries[normalized_name] = ZipEntry(
            path=normalized_name,
            compression_method=method,
            compressed_size=compressed_size,
            uncompressed_size=uncompressed_size,
            crc32=crc32,
            data_offset=nexus.data_offset,
            mode=nexus.mode or ((external_attr >> 16) & 0o777),
            sha256=nexus.sha256,
        )
        _ = local_header_offset
        position = next_position
        parsed += 1

    if parsed != info.entry_count:
        raise NexusZipError(f'Central directory entry count mismatch: expected {info.entry_count}, got {parsed}')
    return entries


def _validate_zip_entry_path(name: str) -> str:
    directory = name.endswith('/')
    value = name[:-1] if directory else name
    if not value or '\x00' in value or '\\' in value:
        raise NexusZipError(f'Invalid ZIP entry path: {name!r}')
    path = PurePosixPath(value)
    if path.is_absolute() or value.startswith('/'):
        raise NexusZipError(f'Absolute ZIP entry path is not supported: {name}')
    if any(part in {'', '.', '..'} for part in value.split('/')):
        raise NexusZipError(f'ZIP entry path traversal is not supported: {name}')
    if path.parts and path.parts[0].endswith(':'):
        raise NexusZipError(f'Absolute ZIP entry path is not supported: {name}')
    normalized = path.as_posix()
    if normalized != value:
        raise NexusZipError(f'Non-canonical ZIP entry path is not supported: {name}')
    return normalized


def _validate_entry_bounds(
    entries: dict[str, ZipEntry],
    *,
    archive_size: int,
    central_directory_offset: int,
) -> None:
    if central_directory_offset < 0 or central_directory_offset > archive_size:
        raise NexusZipError('Central directory offset is outside archive')
    for entry in entries.values():
        end = entry.data_offset + entry.compressed_size
        if entry.data_offset < 0 or end < entry.data_offset or end > central_directory_offset:
            raise NexusZipError(f'ZIP entry data range is outside archive: {entry.path}')


def _validate_central_directory_bounds(info: _CentralDirectoryInfo, *, archive_size: int) -> None:
    end = info.central_directory_offset + info.central_directory_size
    if (
        info.central_directory_offset < 0
        or info.central_directory_size < 0
        or end < info.central_directory_offset
        or end > archive_size
    ):
        raise NexusZipError('Central directory range is outside archive')


def _parse_extra_fields(extra: bytes) -> dict[int, bytes]:
    fields: dict[int, bytes] = {}
    position = 0
    while position + 4 <= len(extra):
        field_id, size = struct.unpack_from('<HH', extra, position)
        data_start = position + 4
        data_end = data_start + size
        if data_end > len(extra):
            raise NexusZipError('Malformed ZIP extra field')
        fields[field_id] = extra[data_start:data_end]
        position = data_end
    if position != len(extra):
        raise NexusZipError('Trailing bytes in ZIP extra field')
    return fields


def _parse_zip64_extra(
    extra: bytes,
    *,
    uncompressed_size: int,
    compressed_size: int,
    local_header_offset: int,
) -> dict[str, int]:
    data = _parse_extra_fields(extra).get(_ZIP64_EXTRA_ID)
    if data is None:
        return {}
    position = 0
    result: dict[str, int] = {}
    if uncompressed_size == 0xFFFFFFFF:
        result['uncompressed_size'] = struct.unpack_from('<Q', data, position)[0]
        position += 8
    if compressed_size == 0xFFFFFFFF:
        result['compressed_size'] = struct.unpack_from('<Q', data, position)[0]
        position += 8
    if local_header_offset == 0xFFFFFFFF:
        result['local_header_offset'] = struct.unpack_from('<Q', data, position)[0]
    return result


def _parse_nexus_extra(extra: bytes, name: str) -> _NexusExtra:
    data = _parse_extra_fields(extra).get(NEXUS_EXTRA_ID)
    if data is None:
        raise NexusZipError(f'Nexus extra field missing for ZIP entry: {name}')
    if len(data) != NEXUS_EXTRA_DATA_SIZE:
        raise NexusZipError(f'Invalid Nexus extra size for ZIP entry {name}: {len(data)}')
    version, _flags, mode, data_offset, sha256_raw = struct.unpack('<BBIQ32s', data)
    if version != NEXUS_EXTRA_VERSION:
        raise NexusZipError(f'Unsupported Nexus ZIP extra version {version} for entry {name}')
    sha256 = sha256_raw.hex() if any(sha256_raw) else None
    return _NexusExtra(mode=mode, data_offset=data_offset, sha256=sha256)


async def read_entry_bytes(entry: ZipEntry, read_range: RangeReader) -> bytes:
    """Read, decompress, and validate one indexed ZIP entry."""

    if entry.uncompressed_size > 512 * 1024 * 1024 or entry.compressed_size > 512 * 1024 * 1024:
        raise NexusZipError(f'ZIP entry exceeds the 512 MiB limit: {entry.path}')
    compressed = (
        b''
        if entry.compressed_size == 0
        else await read_range(entry.data_offset, entry.data_offset + entry.compressed_size)
    )
    if entry.compression_method == zipfile.ZIP_STORED:
        data = compressed
    elif entry.compression_method == zipfile.ZIP_DEFLATED:
        decoder = zlib.decompressobj(-zlib.MAX_WBITS)
        data = decoder.decompress(compressed, entry.uncompressed_size + 1)
        if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
            raise NexusZipError(f'ZIP entry exceeds declared size or has invalid compressed data: {entry.path}')
    else:
        raise NexusZipError(f'Unsupported ZIP compression method {entry.compression_method}: {entry.path}')

    _validate_entry_data(entry, data)
    return data


async def write_entry_to_file(entry: ZipEntry, read_range: RangeReader, target: Path) -> None:
    """Read one indexed ZIP entry and write it to *target*."""

    target.parent.mkdir(parents=True, exist_ok=True)
    data = await read_entry_bytes(entry, read_range)
    target.write_bytes(data)
    if entry.mode:
        os.chmod(target, entry.mode)


def _validate_entry_data(entry: ZipEntry, data: bytes) -> None:
    if len(data) != entry.uncompressed_size:
        raise NexusZipError(
            f'ZIP entry size mismatch for {entry.path}: expected {entry.uncompressed_size}, got {len(data)}'
        )
    crc = binascii.crc32(data) & 0xFFFFFFFF
    if crc != entry.crc32:
        raise NexusZipError(f'ZIP entry CRC mismatch for {entry.path}: expected {entry.crc32}, got {crc}')
    if entry.sha256:
        digest = hashlib.sha256(data).hexdigest()
        if digest != entry.sha256:
            raise NexusZipError(f'ZIP entry sha256 mismatch for {entry.path}: expected {entry.sha256}, got {digest}')
