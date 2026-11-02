from __future__ import annotations

import hashlib
import io
import os
import shutil
import stat
import tarfile
import tempfile

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Literal, cast

from pathspec import GitIgnoreSpec
from zstandard import ZstdCompressor, ZstdDecompressor, ZstdError


_DEFAULT_EXCLUDES = (
    '.git/**',
    '.venv/**',
    '**/__pycache__/**',
    '**/.pytest_cache/**',
    '**/.ruff_cache/**',
    '**/node_modules/**',
    'outputs/**',
    'tests/e2e/*_output/**',
    '**/*.sqlite',
    '**/*.sqlite3',
    '.env',
    '.env.*',
    '**/.env',
    '**/.env.*',
    '**/id_rsa',
    '**/id_ed25519',
    '**/*.pem',
    '**/*.p12',
    '**/*.key',
)
_COPY_CHUNK_BYTES = 1024 * 1024
_MAX_SOURCE_PATH_BYTES = 4096
_MAX_ZSTD_WINDOW_BYTES = 128 * 1024 * 1024
_TAR_METADATA_ALLOWANCE_BYTES = 2 * 1024 * 1024
_TAR_METADATA_PER_ENTRY_BYTES = 8192


@dataclass(frozen=True, slots=True)
class PackageLimits:
    max_files: int = 100_000
    max_single_file_bytes: int = 256 * 1024 * 1024
    max_total_bytes: int = 2 * 1024 * 1024 * 1024
    max_archive_bytes: int = 1024 * 1024 * 1024

    def __post_init__(self) -> None:
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in (
                self.max_files,
                self.max_single_file_bytes,
                self.max_total_bytes,
                self.max_archive_bytes,
            )
        ):
            raise ValueError('source package limits must be positive integers')


@dataclass(frozen=True, slots=True)
class PackagedFile:
    path: Path
    sha256: str
    size_bytes: int
    file_count: int
    unpacked_size_bytes: int


@dataclass(frozen=True, slots=True)
class _SourceEntry:
    path: Path
    relative: str
    kind: Literal['directory', 'file', 'symlink']
    size_bytes: int = 0
    mode: int = 0
    link_target: str | None = None


@dataclass(frozen=True, slots=True)
class _PendingSymlink:
    path: Path
    link_target: str


class _IdentityWriter(io.RawIOBase):
    def __init__(self, raw: BinaryIO) -> None:
        self.raw = raw
        self.digest = hashlib.sha256()
        self.size_bytes = 0

    def writable(self) -> bool:
        return True

    def write(self, data: Any) -> int | None:
        written = self.raw.write(data)
        if written is not None and written > 0:
            self.digest.update(data[:written])
            self.size_bytes += written
        return written

    def flush(self) -> None:
        self.raw.flush()


class _LimitedReader(io.RawIOBase):
    def __init__(self, raw, *, max_bytes: int) -> None:
        self.raw = raw
        self.max_bytes = max_bytes
        self.bytes_read = 0

    def read(self, size: int = -1) -> bytes:
        remaining = self.max_bytes - self.bytes_read
        request_size = remaining + 1 if size < 0 else min(size, remaining + 1)
        data = self.raw.read(request_size)
        self.bytes_read += len(data)
        if self.bytes_read > self.max_bytes:
            raise ValueError('source package decompressed stream exceeds limit')
        return data

    def readable(self) -> bool:
        return True


def build_source_package(
    root: str | Path,
    output_path: str | Path,
    *,
    limits: PackageLimits = PackageLimits(),
    excluded_paths: Iterable[str | Path] = (),
) -> PackagedFile:
    root_path = Path(root).expanduser().resolve(strict=True)
    output = Path(output_path).expanduser().resolve(strict=False)
    if not root_path.is_dir():
        raise ValueError('source root must be a directory')
    mandatory_ignores = GitIgnoreSpec.from_lines(_DEFAULT_EXCLUDES)
    project_ignores = _read_ignore_spec(root_path / '.gitignore')
    excluded = _normalize_excluded_paths(excluded_paths)
    entries = _collect_entries(
        root_path,
        mandatory_ignores=mandatory_ignores,
        project_ignores=project_ignores,
        excluded=excluded,
        output=_absolute_without_symlinks(output),
        limits=limits,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f'.{output.name}.', dir=output.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    identity: _IdentityWriter | None = None
    try:
        with temporary.open('wb') as raw, _IdentityWriter(raw) as identity:
            compressor = ZstdCompressor(
                level=10,
                threads=0,
                write_checksum=True,
                write_content_size=False,
                write_dict_id=False,
            )
            with compressor.stream_writer(cast(BinaryIO, identity), closefd=False) as compressed:
                with tarfile.open(fileobj=compressed, mode='w|', format=tarfile.PAX_FORMAT) as archive:
                    for entry in entries:
                        _add_source_entry(archive, entry)
        assert identity is not None
        if identity.size_bytes > limits.max_archive_bytes:
            raise ValueError('source package exceeds archive size limit')
        temporary.replace(output)
        return PackagedFile(
            path=output,
            sha256=identity.digest.hexdigest(),
            size_bytes=identity.size_bytes,
            file_count=sum(entry.kind != 'directory' for entry in entries),
            unpacked_size_bytes=sum(entry.size_bytes for entry in entries),
        )
    finally:
        temporary.unlink(missing_ok=True)


def safe_extract_source_package(
    archive_path: str | Path,
    target_dir: str | Path,
    *,
    limits: PackageLimits = PackageLimits(),
) -> None:
    archive = Path(archive_path)
    if not archive.is_file() or archive.is_symlink():
        raise ValueError('source package must be a regular file')
    if archive.stat().st_size > limits.max_archive_bytes:
        raise ValueError('source package exceeds archive size limit')

    target_input = Path(target_dir).expanduser()
    target_parent = target_input.parent.resolve(strict=False)
    target_parent.mkdir(parents=True, exist_ok=True)
    target = target_parent / target_input.name
    if target.is_symlink():
        raise ValueError('source package target must not be a symlink')
    temporary = Path(tempfile.mkdtemp(prefix=f'.{target.name}.extracting-', dir=target_parent))
    try:
        _extract_source_archive(archive, temporary, limits=limits)
        _replace_directory(temporary, target)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def _collect_entries(
    root: Path,
    *,
    mandatory_ignores: GitIgnoreSpec,
    project_ignores: GitIgnoreSpec,
    excluded: set[Path],
    output: Path,
    limits: PackageLimits,
) -> list[_SourceEntry]:
    entries: list[_SourceEntry] = []
    total_size = 0

    def visit(directory: Path) -> None:
        nonlocal total_size
        with os.scandir(directory) as children:
            ordered = sorted(children, key=lambda child: child.name)
        for child in ordered:
            path = Path(child.path)
            relative = path.relative_to(root).as_posix()
            lexical_path = _absolute_without_symlinks(path)
            is_directory = child.is_dir(follow_symlinks=False)
            if (
                lexical_path == output
                or _is_forced_excluded(lexical_path, excluded)
                or _matches_ignore(relative, is_directory=is_directory, spec=mandatory_ignores)
                or _matches_ignore(relative, is_directory=is_directory, spec=project_ignores)
            ):
                continue
            if len(relative.encode('utf-8')) > _MAX_SOURCE_PATH_BYTES:
                raise ValueError(f'source path exceeds length limit: {relative}')
            metadata = child.stat(follow_symlinks=False)
            mode = metadata.st_mode
            if stat.S_ISLNK(mode):
                link_target = os.readlink(path)
                entries.append(
                    _SourceEntry(
                        path=path,
                        relative=relative,
                        kind='symlink',
                        mode=0o777,
                        link_target=link_target,
                    )
                )
            elif stat.S_ISDIR(mode):
                entries.append(_SourceEntry(path=path, relative=relative, kind='directory', mode=0o755))
                visit(path)
            elif stat.S_ISREG(mode):
                if metadata.st_nlink != 1:
                    raise ValueError(f'unsupported hardlinked source entry: {relative}')
                size = metadata.st_size
                if size > limits.max_single_file_bytes:
                    raise ValueError(f'source file exceeds size limit: {relative}')
                total_size += size
                if total_size > limits.max_total_bytes:
                    raise ValueError('source package exceeds unpacked size limit')
                normalized_mode = 0o755 if mode & 0o111 else 0o644
                entries.append(
                    _SourceEntry(
                        path=path,
                        relative=relative,
                        kind='file',
                        size_bytes=size,
                        mode=normalized_mode,
                    )
                )
            else:
                raise ValueError(f'unsupported source entry: {relative}')
            if len(entries) > limits.max_files:
                raise ValueError('source package exceeds entry count limit')

    visit(root)
    entries.sort(key=lambda entry: entry.relative)
    included = {entry.relative for entry in entries}
    for entry in entries:
        if entry.kind == 'symlink':
            _validate_source_symlink(
                entry,
                root=root,
                included=included,
                mandatory_ignores=mandatory_ignores,
                project_ignores=project_ignores,
                excluded=excluded,
                output=output,
            )
    return entries


def _validate_source_symlink(
    entry: _SourceEntry,
    *,
    root: Path,
    included: set[str],
    mandatory_ignores: GitIgnoreSpec,
    project_ignores: GitIgnoreSpec,
    excluded: set[Path],
    output: Path,
) -> None:
    assert entry.link_target is not None
    normalized_target = _normalize_link_target(entry.relative, entry.link_target)
    immediate_target = root if normalized_target is None else root.joinpath(*normalized_target.parts)
    try:
        resolved_target = entry.path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f'source symlink is dangling or cyclic: {entry.relative}') from exc
    if resolved_target != root and root not in resolved_target.parents:
        raise ValueError(f'source symlink escapes source root: {entry.relative}')

    target_relative = None if immediate_target == root else immediate_target.relative_to(root).as_posix()
    resolved_relative = None if resolved_target == root else resolved_target.relative_to(root).as_posix()
    if (
        _absolute_without_symlinks(immediate_target) == output
        or _is_forced_excluded(_absolute_without_symlinks(immediate_target), excluded)
        or _is_forced_excluded(_absolute_without_symlinks(resolved_target), excluded)
        or (
            target_relative is not None
            and (
                _matches_ignore(
                    target_relative,
                    is_directory=immediate_target.is_dir(),
                    spec=mandatory_ignores,
                )
                or _matches_ignore(
                    target_relative,
                    is_directory=immediate_target.is_dir(),
                    spec=project_ignores,
                )
            )
        )
        or (
            resolved_relative is not None
            and (
                _matches_ignore(
                    resolved_relative,
                    is_directory=resolved_target.is_dir(),
                    spec=mandatory_ignores,
                )
                or _matches_ignore(
                    resolved_relative,
                    is_directory=resolved_target.is_dir(),
                    spec=project_ignores,
                )
            )
        )
        or (target_relative is not None and target_relative not in included)
    ):
        raise ValueError(f'source symlink points to excluded target: {entry.relative}')


def _add_source_entry(archive: tarfile.TarFile, entry: _SourceEntry) -> None:
    info = tarfile.TarInfo(entry.relative)
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ''
    info.gname = ''
    info.mode = entry.mode
    info.pax_headers = {}
    if entry.kind == 'directory':
        info.type = tarfile.DIRTYPE
        archive.addfile(info)
        return
    if entry.kind == 'symlink':
        info.type = tarfile.SYMTYPE
        assert entry.link_target is not None
        info.linkname = entry.link_target
        archive.addfile(info)
        return
    info.type = tarfile.REGTYPE
    info.size = entry.size_bytes
    with entry.path.open('rb') as source:
        archive.addfile(info, source)


def _extract_source_archive(archive_path: Path, target: Path, *, limits: PackageLimits) -> None:
    stream_limit = (
        limits.max_total_bytes + limits.max_files * _TAR_METADATA_PER_ENTRY_BYTES + _TAR_METADATA_ALLOWANCE_BYTES
    )
    seen: dict[PurePosixPath, Literal['directory', 'file', 'symlink']] = {}
    paths_with_descendants: set[PurePosixPath] = set()
    pending_symlinks: list[_PendingSymlink] = []
    total_size = 0
    entry_count = 0
    try:
        with archive_path.open('rb') as raw:
            with ZstdDecompressor(max_window_size=_MAX_ZSTD_WINDOW_BYTES).stream_reader(
                raw, closefd=False
            ) as decompressed:
                limited = _LimitedReader(decompressed, max_bytes=stream_limit)
                with tarfile.open(fileobj=limited, mode='r|') as archive:
                    for member in archive:
                        relative = _safe_tar_path(member.name)
                        if member.isdir():
                            kind: Literal['directory', 'file', 'symlink'] = 'directory'
                        elif member.isreg():
                            kind = 'file'
                        elif member.issym():
                            kind = 'symlink'
                        else:
                            raise ValueError(f'unsupported source package entry: {member.name}')
                        _validate_path_type(
                            relative,
                            kind=kind,
                            seen=seen,
                            paths_with_descendants=paths_with_descendants,
                        )
                        seen[relative] = kind
                        paths_with_descendants.update(
                            parent for parent in relative.parents if parent != PurePosixPath('.')
                        )
                        entry_count += 1
                        if entry_count > limits.max_files:
                            raise ValueError('source package exceeds entry count limit')
                        output = target.joinpath(*relative.parts)
                        if kind == 'directory':
                            output.mkdir(parents=True, exist_ok=True)
                            output.chmod(0o755)
                            continue
                        output.parent.mkdir(parents=True, exist_ok=True)
                        if kind == 'symlink':
                            _normalize_link_target(relative.as_posix(), member.linkname)
                            pending_symlinks.append(_PendingSymlink(path=output, link_target=member.linkname))
                            continue
                        if member.size > limits.max_single_file_bytes:
                            raise ValueError(f'source file exceeds size limit: {member.name}')
                        total_size += member.size
                        if total_size > limits.max_total_bytes:
                            raise ValueError('source package exceeds unpacked size limit')
                        source = archive.extractfile(member)
                        if source is None:
                            raise ValueError(f'source package file is unreadable: {member.name}')
                        written = 0
                        with output.open('xb') as destination:
                            while chunk := source.read(_COPY_CHUNK_BYTES):
                                written += len(chunk)
                                if written > member.size:
                                    raise ValueError(f'source package file exceeds declared size: {member.name}')
                                destination.write(chunk)
                        if written != member.size:
                            raise ValueError(f'source package file is truncated: {member.name}')
                        output.chmod(0o755 if member.mode & 0o111 else 0o644)
    except (ZstdError, tarfile.TarError, EOFError) as exc:
        raise ValueError('source package archive is invalid') from exc

    for pending in pending_symlinks:
        pending.path.symlink_to(pending.link_target)
    root = target.resolve(strict=True)
    for pending in pending_symlinks:
        try:
            resolved = pending.path.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ValueError(f'source package symlink is dangling or cyclic: {pending.path.name}') from exc
        if resolved != root and root not in resolved.parents:
            raise ValueError(f'source package symlink escapes target: {pending.path.name}')


def _replace_directory(source: Path, target: Path) -> None:
    if not target.exists():
        source.replace(target)
        return
    if not target.is_dir() or target.is_symlink():
        raise ValueError('source package target must be a directory')
    backup = Path(tempfile.mkdtemp(prefix=f'.{target.name}.previous-', dir=target.parent))
    backup.rmdir()
    target.replace(backup)
    try:
        source.replace(target)
    except Exception:
        backup.replace(target)
        raise
    shutil.rmtree(backup, ignore_errors=True)


def _safe_tar_path(value: str) -> PurePosixPath:
    normalized = value.rstrip('/')
    if (
        not normalized
        or '\x00' in normalized
        or '\\' in normalized
        or len(normalized.encode('utf-8')) > _MAX_SOURCE_PATH_BYTES
    ):
        raise ValueError('invalid source package path')
    path = PurePosixPath(normalized)
    if path.is_absolute() or path.as_posix() != normalized or any(part in {'', '.', '..'} for part in path.parts):
        raise ValueError('invalid source package path')
    return path


def _normalize_link_target(link_path: str, link_target: str) -> PurePosixPath | None:
    if (
        not link_target
        or '\x00' in link_target
        or '\\' in link_target
        or len(link_target.encode('utf-8')) > _MAX_SOURCE_PATH_BYTES
    ):
        raise ValueError(f'invalid source symlink target: {link_path}')
    target = PurePosixPath(link_target)
    if target.is_absolute():
        raise ValueError(f'source symlink target must be relative: {link_path}')
    parts = list(PurePosixPath(link_path).parent.parts)
    for part in target.parts:
        if part in {'', '.'}:
            continue
        if part == '..':
            if not parts:
                raise ValueError(f'source symlink target escapes source root: {link_path}')
            parts.pop()
        else:
            parts.append(part)
    return PurePosixPath(*parts) if parts else None


def _validate_path_type(
    path: PurePosixPath,
    *,
    kind: Literal['directory', 'file', 'symlink'],
    seen: dict[PurePosixPath, Literal['directory', 'file', 'symlink']],
    paths_with_descendants: set[PurePosixPath],
) -> None:
    if path in seen:
        raise ValueError('source package contains duplicate paths')
    for parent in path.parents:
        if parent == PurePosixPath('.'):
            continue
        if parent in seen and seen[parent] != 'directory':
            raise ValueError('source package contains a path type conflict')
    if kind != 'directory' and path in paths_with_descendants:
        raise ValueError('source package contains a path type conflict')


def _read_ignore_spec(path: Path) -> GitIgnoreSpec:
    if not path.is_file():
        return GitIgnoreSpec.from_lines(())
    return GitIgnoreSpec.from_lines(path.read_text(encoding='utf-8').splitlines())


def _matches_ignore(relative: str, *, is_directory: bool, spec: GitIgnoreSpec) -> bool:
    candidate = f'{relative}/' if is_directory else relative
    return spec.match_file(candidate)


def _normalize_excluded_paths(paths: Iterable[str | Path]) -> set[Path]:
    excluded: set[Path] = set()
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        excluded.add(_absolute_without_symlinks(path))
        excluded.add(path.resolve(strict=False))
    return excluded


def _absolute_without_symlinks(path: Path) -> Path:
    return Path(os.path.abspath(path))


def _is_forced_excluded(path: Path, excluded: set[Path]) -> bool:
    return any(item == path or item in path.parents for item in excluded)


__all__ = ['PackageLimits', 'PackagedFile', 'build_source_package', 'safe_extract_source_package']
