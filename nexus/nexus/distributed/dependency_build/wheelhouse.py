from __future__ import annotations

import hashlib
import re
import shutil
import stat
import tarfile
import uuid
import zipfile

from dataclasses import dataclass
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from typing import Any, Literal, cast

from packaging.utils import InvalidName, canonicalize_name
from packaging.version import InvalidVersion, Version
from pydantic import Field, field_validator, model_validator
from zstandard import ZstdCompressor, ZstdDecompressor, ZstdError

from nexus.core.module_dependencies import ModuleSelector
from nexus.distributed.artifacts.models import ArtifactRef
from nexus.distributed.environment.limits import SourcePackageLimitsSpec
from nexus.distributed.submission.specs import StrictModel
from nexus.distributed.utils.identity import canonical_sha256


_MAX_WHEEL_ENTRIES = 100_000
_MAX_WHEEL_MEMBER_BYTES = 512 * 1024 * 1024
_MAX_WHEEL_TOTAL_BYTES = 2 * 1024 * 1024 * 1024
_MAX_METADATA_BYTES = 4 * 1024 * 1024
_MAX_WHEEL_METADATA_BYTES = 64 * 1024
_MAX_WHEEL_PATH_BYTES = 4096
_MAX_ZSTD_WINDOW_BYTES = 128 * 1024 * 1024
_TAR_METADATA_ALLOWANCE_BYTES = 2 * 1024 * 1024
_TAR_METADATA_PER_ENTRY_BYTES = 8192
_WHEEL_TAG_PATTERN = re.compile(r'^[A-Za-z0-9_.]+-[A-Za-z0-9_.]+-[A-Za-z0-9_.]+$')
_WHEEL_FILENAME_PATTERN = re.compile(r'^[A-Za-z0-9_.+-]+\.whl$')


class WheelhouseBuildSpec(StrictModel):
    owner_principal_id: str = Field(min_length=1, max_length=64)
    environment_id: str = Field(min_length=1, max_length=64)
    source_package: ArtifactRef
    nexus_wheel: ArtifactRef
    uv_lock: ArtifactRef
    source_package_limits: SourcePackageLimitsSpec = Field(default_factory=SourcePackageLimitsSpec)
    extras: list[str] = Field(default_factory=list)
    module_selectors: list[ModuleSelector] = Field(default_factory=list)
    workload_image: str = Field(min_length=1, max_length=512)
    builder_image: str = Field(min_length=1, max_length=512)
    python_abi: Literal['cp313'] = 'cp313'
    platform: Literal['linux_x86_64'] = 'linux_x86_64'
    os_family: Literal['tlinux4'] = 'tlinux4'

    @field_validator('extras')
    @classmethod
    def normalize_extras(cls, values: list[str]) -> list[str]:
        normalized = sorted(set(values))
        try:
            for value in normalized:
                canonicalize_name(value, validate=True)
        except InvalidName as exc:
            raise ValueError('invalid wheelhouse extra') from exc
        return normalized

    @field_validator('module_selectors')
    @classmethod
    def normalize_module_selectors(cls, values: list[ModuleSelector]) -> list[ModuleSelector]:
        return sorted(
            set(values),
            key=lambda item: (item.kind, item.object_type or '', item.value),
        )

    @model_validator(mode='after')
    def validate_artifact_kinds(self) -> WheelhouseBuildSpec:
        expected = {
            'source_package': (self.source_package, 'source_package'),
            'nexus_wheel': (self.nexus_wheel, 'nexus_wheel'),
            'uv_lock': (self.uv_lock, 'uv_lock'),
        }
        for field_name, (ref, kind) in expected.items():
            if ref.kind != kind:
                raise ValueError(f'{field_name} must have kind={kind}')
        return self


class WheelhouseMetadata(StrictModel):
    schema_version: Literal[1, 2] = 1
    build_key: str = Field(pattern=r'^[0-9a-f]{64}$')
    nexus_wheel_path: str = Field(min_length=1, max_length=512)
    module_names: list[str] = Field(default_factory=list, exclude_if=lambda value: not value)
    module_paths: list[str] = Field(default_factory=list, exclude_if=lambda value: not value)
    module_host_requirements: list[str] = Field(default_factory=list, exclude_if=lambda value: not value)
    module_lock_path: str | None = Field(default=None, exclude_if=lambda value: value is None)
    module_lock_sha256: str | None = Field(
        default=None,
        pattern=r'^[0-9a-f]{64}$',
        exclude_if=lambda value: value is None,
    )

    @field_validator('nexus_wheel_path')
    @classmethod
    def validate_nexus_wheel_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if (
            path.is_absolute()
            or value != path.as_posix()
            or len(path.parts) != 2
            or path.parts[0] != 'wheels'
            or not _WHEEL_FILENAME_PATTERN.fullmatch(path.parts[1])
        ):
            raise ValueError('nexus_wheel_path must name a wheel directly under wheels/')
        return value

    @field_validator('module_names')
    @classmethod
    def normalize_module_names(cls, values: list[str]) -> list[str]:
        if any(not value for value in values):
            raise ValueError('module_names must not contain empty values')
        return sorted(set(values))

    @field_validator('module_paths')
    @classmethod
    def normalize_module_paths(cls, values: list[str]) -> list[str]:
        if any(not value or value.startswith('.') or value.endswith('.') or '..' in value for value in values):
            raise ValueError('module_paths must contain dotted module paths')
        return sorted(set(values))

    @field_validator('module_lock_path')
    @classmethod
    def validate_module_lock_path(cls, value: str | None) -> str | None:
        if value is not None and value != 'locks/module-host.lock':
            raise ValueError('module_lock_path must be locks/module-host.lock')
        return value

    @model_validator(mode='after')
    def validate_module_metadata(self) -> WheelhouseMetadata:
        has_lock = self.module_lock_path is not None or self.module_lock_sha256 is not None
        if self.schema_version == 1:
            if self.module_names or self.module_paths or self.module_host_requirements or has_lock:
                raise ValueError('wheelhouse metadata schema v1 cannot contain module metadata')
        elif self.module_host_requirements:
            if self.module_lock_path is None or self.module_lock_sha256 is None:
                raise ValueError('module Host requirements require a supplemental lock')
        elif has_lock:
            raise ValueError('supplemental lock requires module Host requirements')
        return self


@dataclass(frozen=True, slots=True)
class NexusWheelMetadata:
    version: str
    tags: tuple[str, ...]
    extras: frozenset[str]
    filename: str


class _LimitedReader:
    def __init__(self, raw: Any, *, max_bytes: int) -> None:
        self.raw = raw
        self.max_bytes = max_bytes
        self.bytes_read = 0

    def read(self, size: int = -1) -> bytes:
        remaining = self.max_bytes - self.bytes_read
        request_size = remaining + 1 if size < 0 else min(size, remaining + 1)
        data = self.raw.read(request_size)
        self.bytes_read += len(data)
        if self.bytes_read > self.max_bytes:
            raise ValueError('wheelhouse decompressed stream exceeds limit')
        return data


def wheelhouse_build_key(spec: WheelhouseBuildSpec, *, version: Literal[3, 4] = 4) -> str:
    if version not in (3, 4):
        raise ValueError('unsupported wheelhouse build key version')
    return canonical_sha256({
        'version': version,
        'owner_principal_id': spec.owner_principal_id,
        'environment_id': spec.environment_id,
        'source_package': _artifact_content_identity(spec.source_package),
        'nexus_wheel': _artifact_content_identity(spec.nexus_wheel),
        'uv_lock': _artifact_content_identity(spec.uv_lock),
        'source_package_limits': spec.source_package_limits.model_dump(mode='json'),
        'extras': spec.extras,
        'module_selectors': [selector.model_dump(mode='json') for selector in spec.module_selectors],
        'workload_image': spec.workload_image,
        'builder_image': spec.builder_image,
        'python_abi': spec.python_abi,
        'platform': spec.platform,
        'os_family': spec.os_family,
    })


def resolve_wheelhouse_build_key(spec: WheelhouseBuildSpec, expected_build_key: str | None = None) -> str:
    current = wheelhouse_build_key(spec)
    if expected_build_key is None or expected_build_key == current:
        return current
    # Older source-package Batch Managers create records using the v3 policy.
    if expected_build_key != wheelhouse_build_key(spec, version=3):
        raise ValueError('dependency build key does not match the build inputs')
    return expected_build_key


def _artifact_content_identity(ref: ArtifactRef) -> dict[str, str | int]:
    return {
        'kind': ref.kind,
        'sha256': ref.sha256,
        'size_bytes': ref.size_bytes,
    }


def inspect_nexus_wheel(path: Path) -> NexusWheelMetadata:
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            if len(infos) > _MAX_WHEEL_ENTRIES:
                raise ValueError('nexus-agent wheel file count exceeds limit')

            names: set[str] = set()
            total_bytes = 0
            metadata_info: zipfile.ZipInfo | None = None
            wheel_info: zipfile.ZipInfo | None = None
            for info in infos:
                name = _safe_wheel_member_name(info.filename, directory=info.is_dir())
                if name in names:
                    raise ValueError('nexus-agent wheel contains duplicate members')
                names.add(name)
                if info.flag_bits & 0x1:
                    raise ValueError('nexus-agent wheel contains encrypted members')
                mode = info.external_attr >> 16
                member_type = stat.S_IFMT(mode)
                if info.is_dir():
                    if member_type not in {0, stat.S_IFDIR}:
                        raise ValueError('nexus-agent wheel contains a non-directory member marked as a directory')
                    continue
                if member_type not in {0, stat.S_IFREG}:
                    raise ValueError('nexus-agent wheel contains a non-regular member')
                if info.file_size < 0 or info.file_size > _MAX_WHEEL_MEMBER_BYTES:
                    raise ValueError('nexus-agent wheel member exceeds limit')
                total_bytes += info.file_size
                if total_bytes > _MAX_WHEEL_TOTAL_BYTES:
                    raise ValueError('nexus-agent wheel unpacked size exceeds limit')

                member_path = PurePosixPath(name)
                if len(member_path.parts) != 2 or not member_path.parts[0].endswith('.dist-info'):
                    continue
                if member_path.parts[1] == 'METADATA':
                    if metadata_info is not None:
                        raise ValueError('nexus-agent wheel metadata layout is invalid')
                    metadata_info = info
                elif member_path.parts[1] == 'WHEEL':
                    if wheel_info is not None:
                        raise ValueError('nexus-agent wheel metadata layout is invalid')
                    wheel_info = info

            if metadata_info is None or wheel_info is None:
                raise ValueError('nexus-agent wheel metadata layout is invalid')
            metadata_dir = PurePosixPath(metadata_info.filename).parts[0]
            wheel_dir = PurePosixPath(wheel_info.filename).parts[0]
            if metadata_dir != wheel_dir:
                raise ValueError('nexus-agent wheel dist-info directories do not match')

            metadata_bytes = _read_bounded_zip_member(
                archive,
                metadata_info,
                limit=_MAX_METADATA_BYTES,
                label='METADATA',
            )
            wheel_bytes = _read_bounded_zip_member(
                archive,
                wheel_info,
                limit=_MAX_WHEEL_METADATA_BYTES,
                label='WHEEL',
            )
    except zipfile.BadZipFile as exc:
        raise ValueError('nexus-agent wheel is not a valid wheel archive') from exc
    except OSError as exc:
        raise ValueError('nexus-agent wheel is unreadable') from exc

    metadata = BytesParser().parsebytes(metadata_bytes)
    if canonicalize_name(str(metadata.get('Name') or '')) != 'nexus-agent':
        raise ValueError('source-package wheel distribution must be nexus-agent')
    version = str(metadata.get('Version') or '').strip()
    if not version:
        raise ValueError('nexus-agent wheel version is missing')
    try:
        Version(version)
    except InvalidVersion as exc:
        raise ValueError('nexus-agent wheel version is invalid') from exc
    normalized_version = re.sub(r'[^A-Za-z0-9.]+', '_', version)
    expected_metadata_dir = f'nexus_agent-{normalized_version}.dist-info'
    if metadata_dir != expected_metadata_dir:
        raise ValueError('nexus-agent wheel dist-info directory does not match METADATA')
    try:
        extras = frozenset(
            canonicalize_name(str(value).strip(), validate=True)
            for value in metadata.get_all('Provides-Extra', [])
            if str(value).strip()
        )
    except InvalidName as exc:
        raise ValueError('nexus-agent wheel extra is invalid') from exc

    wheel_metadata = BytesParser().parsebytes(wheel_bytes)
    tags = tuple(sorted({str(value).strip() for value in wheel_metadata.get_all('Tag', []) if str(value).strip()}))
    if not tags or any(_WHEEL_TAG_PATTERN.fullmatch(tag) is None for tag in tags):
        raise ValueError('nexus-agent wheel tag is invalid')
    filename = f'nexus_agent-{normalized_version}-{tags[0]}.whl'
    return NexusWheelMetadata(
        version=version,
        tags=tags,
        extras=extras,
        filename=filename,
    )


def write_wheelhouse_archive(
    wheels_dir: Path,
    spec: WheelhouseBuildSpec,
    output: Path,
    *,
    nexus_wheel_path: str,
    expected_build_key: str | None = None,
    module_names: list[str] | None = None,
    module_paths: list[str] | None = None,
    module_host_requirements: list[str] | None = None,
    module_lock: Path | None = None,
) -> WheelhouseMetadata:
    module_names = list(module_names or [])
    module_paths = list(module_paths or [])
    module_host_requirements = list(module_host_requirements or [])
    if bool(module_host_requirements) != bool(module_lock):
        raise ValueError('module Host requirements and supplemental lock must be provided together')
    if module_lock is not None and (not module_lock.is_file() or module_lock.is_symlink()):
        raise ValueError('module supplemental lock must be a regular file')
    metadata = WheelhouseMetadata(
        schema_version=2 if module_names or module_paths or module_host_requirements else 1,
        build_key=resolve_wheelhouse_build_key(spec, expected_build_key),
        nexus_wheel_path=nexus_wheel_path,
        module_names=module_names,
        module_paths=module_paths,
        module_host_requirements=module_host_requirements,
        module_lock_path='locks/module-host.lock' if module_lock is not None else None,
        module_lock_sha256=(hashlib.sha256(module_lock.read_bytes()).hexdigest() if module_lock is not None else None),
    )
    wheel_paths = sorted(wheels_dir.glob('*.whl'), key=lambda path: path.name)
    if not wheel_paths:
        raise ValueError('wheelhouse contains no wheels')
    archive_paths: set[str] = set()
    for path in wheel_paths:
        if not path.is_file() or path.is_symlink():
            raise ValueError('wheelhouse wheel must be a regular file')
        archive_path = f'wheels/{path.name}'
        _wheelhouse_member_path(archive_path)
        if archive_path in archive_paths:
            raise ValueError('wheelhouse contains duplicate wheel names')
        archive_paths.add(archive_path)
    if metadata.nexus_wheel_path not in archive_paths:
        raise ValueError('wheelhouse does not contain nexus_wheel_path')

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f'.{output.name}.{uuid.uuid4().hex}.tmp')
    try:
        with temporary.open('wb') as raw:
            compressor = ZstdCompressor(
                level=10,
                write_checksum=True,
                write_content_size=False,
                write_dict_id=False,
                threads=0,
            )
            with compressor.stream_writer(raw, closefd=False) as compressed:
                with tarfile.open(fileobj=compressed, mode='w|', format=tarfile.PAX_FORMAT) as archive:
                    for path in wheel_paths:
                        info = _tar_info(f'wheels/{path.name}', path.stat().st_size)
                        with path.open('rb') as stream:
                            archive.addfile(info, stream)
                    if module_lock is not None:
                        info = _tar_info('locks/module-host.lock', module_lock.stat().st_size)
                        with module_lock.open('rb') as stream:
                            archive.addfile(info, stream)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return metadata


def extract_wheelhouse_archive(
    archive: Path,
    target: Path,
    *,
    max_files: int,
    max_total_bytes: int,
) -> None:
    if max_files < 1 or max_total_bytes < 1:
        raise ValueError('invalid wheelhouse extraction limits')
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() or target.exists() and not target.is_dir():
        raise ValueError('wheelhouse target must be a directory')
    temporary = target.with_name(f'.{target.name}.{uuid.uuid4().hex}.extracting')
    backup = target.with_name(f'.{target.name}.{uuid.uuid4().hex}.backup')
    temporary.mkdir(parents=True)
    replaced = False
    try:
        _extract_wheelhouse_members(
            archive,
            temporary,
            max_files=max_files,
            max_total_bytes=max_total_bytes,
        )
        if target.exists():
            target.replace(backup)
        temporary.replace(target)
        replaced = True
        shutil.rmtree(backup, ignore_errors=True)
    finally:
        if not replaced and backup.exists() and not target.exists():
            backup.replace(target)
        shutil.rmtree(temporary, ignore_errors=True)
        shutil.rmtree(backup, ignore_errors=True)


def _extract_wheelhouse_members(
    archive_path: Path,
    target: Path,
    *,
    max_files: int,
    max_total_bytes: int,
) -> None:
    seen: set[str] = set()
    wheel_count = 0
    total_bytes = 0
    stream_limit = max_total_bytes + max_files * _TAR_METADATA_PER_ENTRY_BYTES + _TAR_METADATA_ALLOWANCE_BYTES
    try:
        with archive_path.open('rb') as raw:
            decompressor = ZstdDecompressor(max_window_size=_MAX_ZSTD_WINDOW_BYTES)
            with decompressor.stream_reader(raw, closefd=False) as decompressed:
                limited = _LimitedReader(decompressed, max_bytes=stream_limit)
                with tarfile.open(fileobj=cast(Any, limited), mode='r|') as archive:
                    while (member := archive.next()) is not None:
                        path = _wheelhouse_member_path(member.name)
                        if not member.isfile():
                            raise ValueError('wheelhouse member must be a regular file')
                        if member.name in seen:
                            raise ValueError('wheelhouse contains duplicate members')
                        seen.add(member.name)
                        if path.parts[0] == 'wheels':
                            wheel_count += 1
                        if len(seen) > max_files:
                            raise ValueError('wheelhouse file count exceeds limit')
                        if member.size < 0:
                            raise ValueError('wheelhouse member size is invalid')
                        total_bytes += member.size
                        if total_bytes > max_total_bytes:
                            raise ValueError('wheelhouse unpacked size exceeds limit')
                        source = archive.extractfile(member)
                        if source is None:
                            raise ValueError('wheelhouse member is unreadable')
                        destination = target.joinpath(*path.parts)
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        _copy_exact(source, destination, expected_size=member.size)
    except ValueError:
        raise
    except (OSError, tarfile.TarError, ZstdError) as exc:
        raise ValueError('invalid wheelhouse archive') from exc
    if wheel_count == 0:
        raise ValueError('wheelhouse contains no wheels')


def _safe_wheel_member_name(value: str, *, directory: bool) -> str:
    normalized = value[:-1] if directory and value.endswith('/') and not value.endswith('//') else value
    path = PurePosixPath(normalized)
    if (
        path.is_absolute()
        or '\\' in value
        or '\x00' in value
        or len(value.encode('utf-8')) > _MAX_WHEEL_PATH_BYTES
        or normalized != path.as_posix()
        or any(part in {'', '.', '..'} for part in path.parts)
    ):
        raise ValueError('nexus-agent wheel contains an unsafe member path')
    return normalized


def _read_bounded_zip_member(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    *,
    limit: int,
    label: str,
) -> bytes:
    if info.file_size > limit:
        raise ValueError(f'nexus-agent wheel {label} exceeds limit')
    with archive.open(info) as stream:
        content = stream.read(limit + 1)
    if len(content) > limit:
        raise ValueError(f'nexus-agent wheel {label} exceeds limit')
    return content


def _wheelhouse_member_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    is_wheel = (
        len(path.parts) == 2
        and path.parts[0] == 'wheels'
        and _WHEEL_FILENAME_PATTERN.fullmatch(path.parts[1]) is not None
    )
    is_module_lock = value == 'locks/module-host.lock'
    if path.is_absolute() or value != path.as_posix() or not (is_wheel or is_module_lock):
        raise ValueError('wheelhouse member path is invalid')
    return path


def _copy_exact(source, destination: Path, *, expected_size: int) -> None:
    written = 0
    with destination.open('xb') as output:
        while chunk := source.read(min(1024 * 1024, expected_size - written + 1)):
            written += len(chunk)
            if written > expected_size:
                raise ValueError('wheelhouse member size mismatch')
            output.write(chunk)
    if written != expected_size:
        raise ValueError('wheelhouse member size mismatch')


def _tar_info(name: str, size_bytes: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = size_bytes
    info.mode = 0o644
    info.uid = 0
    info.gid = 0
    info.uname = ''
    info.gname = ''
    info.mtime = 0
    return info


__all__ = [
    'NexusWheelMetadata',
    'WheelhouseBuildSpec',
    'WheelhouseMetadata',
    'extract_wheelhouse_archive',
    'inspect_nexus_wheel',
    'resolve_wheelhouse_build_key',
    'wheelhouse_build_key',
    'write_wheelhouse_archive',
]
