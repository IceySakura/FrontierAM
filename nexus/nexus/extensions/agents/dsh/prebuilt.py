"""Download and safely materialize prebuilt DeepSeek Harness npm installations."""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import re
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


logger = logging.getLogger(__name__)

DEFAULT_DSH_PREBUILT_BASE_URL = 'https://mirrors.tencent.com/repository/generic/nexus-dsh/v1'
PREBUILT_MANIFEST_VERSION = 1
_MAX_MANIFEST_BYTES = 1024 * 1024
_MAX_ARCHIVE_BYTES = 1024 * 1024 * 1024
_MAX_DOWNLOAD_ATTEMPTS = 8
_DOWNLOAD_SOCKET_TIMEOUT = 30.0
_SHA256_PATTERN = re.compile(r'^[0-9a-f]{64}$')
_ARCHITECTURES = {
    'aarch64': 'arm64',
    'amd64': 'x64',
    'arm64': 'arm64',
    'x86_64': 'x64',
}
_ALLOWED_ARCHIVE_ROOTS = frozenset({'node_modules', 'package-lock.json', 'package.json'})


class PrebuiltUnavailableError(RuntimeError):
    """The prebuilt repository could not be consulted or downloaded."""


class PrebuiltIntegrityError(RuntimeError):
    """A published prebuilt artifact violates its immutable contract."""


class PrebuiltCompatibilityError(RuntimeError):
    """A valid artifact cannot run on the current runtime."""


@dataclass(frozen=True)
class RuntimeTarget:
    """Platform facts that participate in prebuilt artifact selection."""

    os: str
    architecture: str
    libc: str | None
    libc_version: str | None
    node_major: int

    @property
    def tag(self) -> str:
        """Return the stable repository path component for this target."""
        parts = [self.os, self.architecture]
        if self.libc is not None:
            parts.append(self.libc)
        parts.append(f'node{self.node_major}')
        return '-'.join(parts)


@dataclass(frozen=True)
class PrebuiltManifest:
    """Validated metadata for one immutable DSH installation archive."""

    manifest_url: str
    dsh_version: str
    acp_version: str
    target: str
    archive: str
    archive_sha256: str
    archive_size: int
    package_lock_sha256: str
    minimum_libc_version: str | None

    @property
    def archive_url(self) -> str:
        """Resolve the sibling archive without allowing path traversal."""
        return urllib.parse.urljoin(self.manifest_url, urllib.parse.quote(self.archive, safe=''))


def detect_runtime_target(*, node_major: int) -> RuntimeTarget | None:
    """Return the current supported artifact target, or ``None`` for unknown hosts."""
    architecture = _ARCHITECTURES.get(platform.machine().lower())
    if architecture is None:
        return None
    if sys.platform == 'linux':
        libc_name, libc_version = platform.libc_ver()
        normalized = libc_name.lower().replace('gnu libc', 'glibc')
        if normalized not in {'glibc', 'musl'}:
            return None
        return RuntimeTarget('linux', architecture, normalized, libc_version or None, node_major)
    if sys.platform == 'darwin':
        return RuntimeTarget('macos', architecture, None, None, node_major)
    if sys.platform == 'win32':
        return RuntimeTarget('windows', architecture, None, None, node_major)
    return None


def prebuilt_manifest_url(
    base_url: str,
    *,
    dsh_version: str,
    acp_version: str,
    target: RuntimeTarget,
) -> str:
    """Build the immutable manifest lookup URL for exact package versions."""
    root = base_url.rstrip('/') + '/'
    parts = (
        f'dsh-{dsh_version}',
        f'acp-{acp_version}',
        target.tag,
        'manifest.json',
    )
    return root + '/'.join(urllib.parse.quote(part, safe='') for part in parts)


def fetch_prebuilt_manifest(
    base_url: str,
    *,
    dsh_version: str,
    acp_version: str,
    target: RuntimeTarget,
    timeout: float = 15.0,
) -> PrebuiltManifest | None:
    """Fetch and validate a manifest; return ``None`` only for a clean 404."""
    url = prebuilt_manifest_url(
        base_url,
        dsh_version=dsh_version,
        acp_version=acp_version,
        target=target,
    )
    request = urllib.request.Request(url, headers={'User-Agent': 'nexus-dsh-installer/1'})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read(_MAX_MANIFEST_BYTES + 1)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise PrebuiltUnavailableError(f'prebuilt manifest request returned HTTP {exc.code}: {url}') from exc
    except (OSError, urllib.error.URLError) as exc:
        raise PrebuiltUnavailableError(f'prebuilt manifest request failed: {url}: {exc}') from exc
    if len(raw) > _MAX_MANIFEST_BYTES:
        raise PrebuiltIntegrityError(f'prebuilt manifest exceeds {_MAX_MANIFEST_BYTES} bytes: {url}')
    try:
        data = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PrebuiltIntegrityError(f'prebuilt manifest is not valid JSON: {url}') from exc
    return _parse_manifest(
        data,
        manifest_url=url,
        dsh_version=dsh_version,
        acp_version=acp_version,
        target=target,
    )


def _required(data: dict[str, Any], name: str, expected_type: type) -> Any:
    value = data.get(name)
    if not isinstance(value, expected_type) or isinstance(value, bool):
        raise PrebuiltIntegrityError(f'prebuilt manifest field {name!r} must be {expected_type.__name__}')
    return value


def _parse_manifest(
    raw: Any,
    *,
    manifest_url: str,
    dsh_version: str,
    acp_version: str,
    target: RuntimeTarget,
) -> PrebuiltManifest:
    if not isinstance(raw, dict):
        raise PrebuiltIntegrityError('prebuilt manifest root must be an object')
    version = _required(raw, 'manifest_version', int)
    if version != PREBUILT_MANIFEST_VERSION:
        raise PrebuiltIntegrityError(
            f'unsupported prebuilt manifest version {version}; expected {PREBUILT_MANIFEST_VERSION}'
        )
    actual_dsh = _required(raw, 'dsh_version', str)
    actual_acp = _required(raw, 'acp_version', str)
    actual_target = _required(raw, 'target', str)
    if (actual_dsh, actual_acp, actual_target) != (dsh_version, acp_version, target.tag):
        raise PrebuiltIntegrityError(
            'prebuilt manifest identity mismatch: '
            f'got {(actual_dsh, actual_acp, actual_target)!r}, '
            f'expected {(dsh_version, acp_version, target.tag)!r}'
        )
    archive = _required(raw, 'archive', str)
    archive_path = PurePosixPath(archive)
    if archive in {'', '.', '..'} or archive_path.name != archive or archive_path.is_absolute():
        raise PrebuiltIntegrityError(f'prebuilt archive must be a plain filename, got {archive!r}')
    archive_sha256 = _required(raw, 'archive_sha256', str)
    package_lock_sha256 = _required(raw, 'package_lock_sha256', str)
    if not _SHA256_PATTERN.fullmatch(archive_sha256):
        raise PrebuiltIntegrityError('prebuilt archive_sha256 must be 64 lowercase hexadecimal characters')
    if not _SHA256_PATTERN.fullmatch(package_lock_sha256):
        raise PrebuiltIntegrityError('prebuilt package_lock_sha256 must be 64 lowercase hexadecimal characters')
    archive_size = _required(raw, 'archive_size', int)
    if not 0 < archive_size <= _MAX_ARCHIVE_BYTES:
        raise PrebuiltIntegrityError(f'prebuilt archive_size must be between 1 and {_MAX_ARCHIVE_BYTES}')
    minimum_libc_version = raw.get('minimum_libc_version')
    if minimum_libc_version is not None and not isinstance(minimum_libc_version, str):
        raise PrebuiltIntegrityError('prebuilt minimum_libc_version must be a string or null')
    manifest = PrebuiltManifest(
        manifest_url=manifest_url,
        dsh_version=actual_dsh,
        acp_version=actual_acp,
        target=actual_target,
        archive=archive,
        archive_sha256=archive_sha256,
        archive_size=archive_size,
        package_lock_sha256=package_lock_sha256,
        minimum_libc_version=minimum_libc_version,
    )
    _validate_libc_compatibility(manifest, target)
    return manifest


def _version_tuple(value: str) -> tuple[int, ...]:
    if not re.fullmatch(r'\d+(?:\.\d+)*', value):
        raise PrebuiltIntegrityError(f'invalid minimum libc version in prebuilt manifest: {value!r}')
    return tuple(int(part) for part in value.split('.'))


def _validate_libc_compatibility(manifest: PrebuiltManifest, target: RuntimeTarget) -> None:
    minimum = manifest.minimum_libc_version
    if minimum is None or target.libc_version is None:
        return
    if _version_tuple(target.libc_version) < _version_tuple(minimum):
        raise PrebuiltCompatibilityError(
            f'prebuilt artifact requires {target.libc} >= {minimum}, runtime has {target.libc_version}'
        )


def download_and_extract_prebuilt(
    manifest: PrebuiltManifest,
    *,
    archive_path: Path,
    install_path: Path,
    timeout: float = 300.0,
) -> None:
    """Download, checksum, and safely extract one prebuilt installation."""
    size, actual_sha256 = _download_archive(manifest, archive_path=archive_path, timeout=timeout)
    if size != manifest.archive_size:
        raise PrebuiltIntegrityError(
            f'prebuilt archive size mismatch: expected {manifest.archive_size}, downloaded {size}'
        )
    if actual_sha256 != manifest.archive_sha256:
        raise PrebuiltIntegrityError(
            f'prebuilt archive SHA-256 mismatch: expected {manifest.archive_sha256}, got {actual_sha256}'
        )
    install_path.mkdir(parents=True, exist_ok=False)
    _safe_extract_tar(archive_path, install_path)
    lock_path = install_path / 'package-lock.json'
    if not lock_path.is_file():
        raise PrebuiltIntegrityError('prebuilt archive does not contain package-lock.json')
    lock_sha256 = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    if lock_sha256 != manifest.package_lock_sha256:
        raise PrebuiltIntegrityError(
            f'prebuilt package-lock.json SHA-256 mismatch: expected {manifest.package_lock_sha256}, got {lock_sha256}'
        )


def _download_archive(manifest: PrebuiltManifest, *, archive_path: Path, timeout: float) -> tuple[int, str]:
    """Download an archive with bounded, resumable requests."""
    deadline = time.monotonic() + timeout
    digest = hashlib.sha256()
    size = 0
    last_error: BaseException | None = None
    archive_path.unlink(missing_ok=True)
    for attempt in range(_MAX_DOWNLOAD_ATTEMPTS):
        if size >= manifest.archive_size:
            break
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        headers = {'User-Agent': 'nexus-dsh-installer/1'}
        if size:
            headers['Range'] = f'bytes={size}-'
        request = urllib.request.Request(manifest.archive_url, headers=headers)
        request_start = size
        try:
            with urllib.request.urlopen(
                request,
                timeout=min(_DOWNLOAD_SOCKET_TIMEOUT, remaining),
            ) as response:
                response_status = response.getcode()
                mode = 'ab'
                if request_start and response_status == 206:
                    content_range = response.headers.get('Content-Range', '')
                    if not content_range.startswith(f'bytes {request_start}-'):
                        raise PrebuiltUnavailableError(
                            f'prebuilt archive returned an invalid Content-Range: {content_range!r}'
                        )
                elif request_start and response_status == 200:
                    # Some redirecting proxies drop Range. Restart safely rather
                    # than appending a second full response to the partial file.
                    size = 0
                    digest = hashlib.sha256()
                    mode = 'wb'
                elif response_status not in {200, 206}:
                    raise PrebuiltUnavailableError(
                        f'prebuilt archive request returned HTTP {response_status}: {manifest.archive_url}'
                    )
                with archive_path.open(mode) as output:
                    while chunk := response.read(1024 * 1024):
                        size += len(chunk)
                        if size > manifest.archive_size or size > _MAX_ARCHIVE_BYTES:
                            raise PrebuiltIntegrityError(
                                f'prebuilt archive exceeds declared size {manifest.archive_size}'
                            )
                        digest.update(chunk)
                        output.write(chunk)
        except (PrebuiltIntegrityError, PrebuiltUnavailableError):
            raise
        except (OSError, urllib.error.HTTPError, urllib.error.URLError) as exc:
            last_error = exc
            logger.warning(
                'DeepSeek Harness prebuilt download request %d/%d was interrupted at %d/%d bytes; resuming: %s',
                attempt + 1,
                _MAX_DOWNLOAD_ATTEMPTS,
                size,
                manifest.archive_size,
                exc,
            )
            continue
    if size < manifest.archive_size:
        detail = f': {last_error}' if last_error is not None else ''
        raise PrebuiltUnavailableError(
            f'prebuilt archive download incomplete after {_MAX_DOWNLOAD_ATTEMPTS} requests '
            f'({size}/{manifest.archive_size} bytes): {manifest.archive_url}{detail}'
        ) from last_error
    return size, digest.hexdigest()


def _safe_extract_tar(archive_path: Path, destination: Path) -> None:
    seen: set[str] = set()
    try:
        with tarfile.open(archive_path, mode='r:gz') as archive:
            members = archive.getmembers()
            for member in members:
                _validate_tar_member(member, seen)
            archive.extractall(destination, members=members, filter='fully_trusted')
    except PrebuiltIntegrityError:
        raise
    except (OSError, tarfile.TarError) as exc:
        raise PrebuiltIntegrityError(f'prebuilt archive is not a valid tar.gz: {exc}') from exc


def _validate_tar_member(member: tarfile.TarInfo, seen: set[str]) -> None:
    path = PurePosixPath(member.name)
    if path.is_absolute() or not path.parts or '..' in path.parts:
        raise PrebuiltIntegrityError(f'unsafe path in prebuilt archive: {member.name!r}')
    if path.parts[0] not in _ALLOWED_ARCHIVE_ROOTS:
        raise PrebuiltIntegrityError(f'unexpected top-level path in prebuilt archive: {member.name!r}')
    canonical = str(path)
    if canonical in seen:
        raise PrebuiltIntegrityError(f'duplicate path in prebuilt archive: {member.name!r}')
    seen.add(canonical)
    if not (member.isfile() or member.isdir() or member.issym() or member.islnk()):
        raise PrebuiltIntegrityError(f'special file in prebuilt archive: {member.name!r}')
    if member.mode & 0o6000:
        raise PrebuiltIntegrityError(f'setuid/setgid file in prebuilt archive: {member.name!r}')
    if member.issym():
        link = PurePosixPath(member.linkname)
        resolved = PurePosixPath(path.parent, link)
        _validate_link_target(member.name, member.linkname, resolved)
    elif member.islnk():
        _validate_link_target(member.name, member.linkname, PurePosixPath(member.linkname))


def _validate_link_target(name: str, linkname: str, target: PurePosixPath) -> None:
    if target.is_absolute():
        raise PrebuiltIntegrityError(f'absolute link target in prebuilt archive: {name!r} -> {linkname!r}')
    depth = 0
    for part in target.parts:
        if part in {'', '.'}:
            continue
        if part == '..':
            depth -= 1
            if depth < 0:
                raise PrebuiltIntegrityError(f'escaping link target in prebuilt archive: {name!r} -> {linkname!r}')
        else:
            depth += 1


__all__ = [
    'DEFAULT_DSH_PREBUILT_BASE_URL',
    'PREBUILT_MANIFEST_VERSION',
    'PrebuiltCompatibilityError',
    'PrebuiltIntegrityError',
    'PrebuiltManifest',
    'PrebuiltUnavailableError',
    'RuntimeTarget',
    'detect_runtime_target',
    'download_and_extract_prebuilt',
    'fetch_prebuilt_manifest',
    'prebuilt_manifest_url',
]
