"""Content-verified task bundles; no dependency on a source checkout at execution."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import tarfile
import tempfile

from pathlib import Path, PurePosixPath

from pydantic import BaseModel, ConfigDict, Field

from nexus.dataset.auth import WedataAuthenticator, WedataKey
from nexus.dataset.task_files import task_files_from_spec
from nexus.extensions.tasks.os_world_v2.adapter.source import COMMIT, RELEASE, verify_files
from nexus.utils.paths import NEXUS_HOME


MANIFEST = 'bundle.json'
PART_BYTES = 256 * 1024 * 1024
ARCHIVES = {
    'source.tar.gz': 'repo',
    'shared-assets.tar.gz': 'assets',
    'task-assets.tar.gz': 'assets',
    'dependencies.tar.gz': 'dependencies',
}


class Digest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    size: int = Field(ge=0)


class Manifest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    schema_version: int = Field(default=1, ge=1, le=1)
    source_commit: str
    benchmark_release: str
    task_id: str
    archives: dict[str, Digest]
    archive_parts: dict[str, list[Digest]] = Field(default_factory=dict)
    missing_external_dependencies: list[str] = Field(default_factory=list)
    dependency_profile: str | None = None


def part_name(archive: str, index: int) -> str:
    return f'{archive}.part-{index:05d}'


def digest_file(path: Path) -> Digest:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return Digest(sha256=digest.hexdigest(), size=path.stat().st_size)


def authenticator(raw):
    if raw is None:
        return None
    value = json.dumps(raw) if isinstance(raw, dict) else raw
    return WedataAuthenticator(key=WedataKey.from_raw(value, source='V2 task_config.wedata_key'))


def extract_regular(archive: Path, target: Path) -> None:
    """Extract regular files/directories only. Reject links, devices and collisions."""
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, 'r:gz') as stream:
        for member in stream:
            name = PurePosixPath(member.name)
            if name.is_absolute() or '\\' in member.name or any(p in {'', '.', '..'} for p in member.name.split('/')):
                raise ValueError('Unsafe V2 archive path')
            path = target.joinpath(*name.parts)
            if not path.resolve().is_relative_to(target.resolve()):
                raise ValueError('V2 archive escapes destination')
            if member.isdir():
                path.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise ValueError('V2 archives may not contain links or special files')
            path.parent.mkdir(parents=True, exist_ok=True)
            source = stream.extractfile(member)
            if source is None:
                raise ValueError('Missing regular file content in V2 archive')
            with source, path.open('xb') as destination:
                shutil.copyfileobj(source, destination, 1024 * 1024)
            os.chmod(path, 0o700 if member.mode & 0o111 else 0o600)


async def _thread_complete(function, *args):
    # Cancellation must not leave an extraction/copy thread writing into a cleaned directory.
    work = asyncio.create_task(asyncio.to_thread(function, *args))
    try:
        return await asyncio.shield(work)
    except asyncio.CancelledError:
        while not work.done():
            try:
                await asyncio.shield(work)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        # Preserve cancellation rather than replacing it with a background I/O error.
        if not work.cancelled():
            work.exception()
        raise


async def materialize(
    files,
    *,
    expected_sha256: str,
    selected: str,
    destination: Path,
    raw_key=None,
    cache_root: Path | None = None,
    require_dependencies: bool = True,
):
    if raw_key is not None:
        files = task_files_from_spec(files.spec, authenticator=authenticator(raw_key))
    if files is None:
        raise ValueError('V2 task_files is required')
    raw = await files.read_bytes(MANIFEST)
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('V2 bundle manifest SHA256 mismatch')
    manifest = Manifest.model_validate_json(raw)
    if (manifest.source_commit, manifest.benchmark_release, manifest.task_id) != (COMMIT, RELEASE, selected):
        raise ValueError('V2 bundle identity mismatch')
    if manifest.dependency_profile:
        from nexus.extensions.tasks.os_world_v2.adapter.task033 import PROFILE

        if selected != '033' or manifest.dependency_profile != PROFILE:
            raise ValueError('Unsupported V2 dependency profile')
        if 'dependencies.tar.gz' not in manifest.archives or manifest.missing_external_dependencies:
            raise ValueError('V2 dependency profile requires complete dependency archives')
    if not {'source.tar.gz', 'shared-assets.tar.gz', 'task-assets.tar.gz'} <= manifest.archives.keys():
        raise ValueError('V2 bundle is missing required archives')
    if manifest.archives.keys() - ARCHIVES.keys():
        raise ValueError('Unknown archive in V2 bundle')
    if require_dependencies and manifest.missing_external_dependencies:
        raise ValueError(
            f'V2 task {selected} has unpublished external dependencies: '
            + ', '.join(manifest.missing_external_dependencies)
        )
    if manifest.archive_parts.keys() - manifest.archives.keys():
        raise ValueError('Unknown chunked archive in V2 bundle')
    expected_files = {MANIFEST}
    for name, digest in manifest.archives.items():
        if name in manifest.archive_parts:
            parts = manifest.archive_parts[name]
            if not parts or any(p.size <= 0 or p.size > PART_BYTES for p in parts):
                raise ValueError('Invalid V2 archive parts')
            if sum(p.size for p in parts) != digest.size:
                raise ValueError('V2 archive parts size mismatch')
            expected_files.update(part_name(name, i) for i in range(len(parts)))
        else:
            expected_files.add(name)
    if set(await files.list()) != expected_files:
        raise ValueError('V2 bundle has undeclared/missing files')
    cache = (cache_root or NEXUS_HOME / 'osworld-v2-inputs') / 'archives'
    cache.mkdir(parents=True, exist_ok=True)
    cached = {}
    for name, expected in manifest.archives.items():
        path = cache / expected.sha256
        if path.exists():
            if path.is_symlink() or await _thread_complete(digest_file, path) != expected:
                raise ValueError('V2 input cache digest mismatch; remove corrupted cache entry')
        else:
            # Dataset transport verifies its ZIP entry checksum, then we verify the pinned bundle checksum.
            fd, temporary = tempfile.mkstemp(prefix='.download-', dir=cache)
            try:
                with os.fdopen(fd, 'wb') as stream:
                    parts = manifest.archive_parts.get(name)
                    entries = [(part_name(name, i), p) for i, p in enumerate(parts)] if parts else [(name, expected)]
                    for filename, digest in entries:
                        data = await files.read_bytes(filename)
                        if len(data) != digest.size or hashlib.sha256(data).hexdigest() != digest.sha256:
                            raise ValueError(f'V2 archive digest mismatch: {filename}')
                        stream.write(data)
                        del data
                if await _thread_complete(digest_file, Path(temporary)) != expected:
                    raise ValueError(f'V2 reassembled archive digest mismatch: {name}')
                os.replace(temporary, path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        cached[name] = path
    destination.mkdir(parents=True, exist_ok=False)
    try:
        for name, archive in cached.items():
            await _thread_complete(extract_regular, archive, destination / ARCHIVES[name])
        repo, assets = destination / 'repo', destination / 'assets'
        await _thread_complete(verify_files, repo)
        # Official tasks 082/103/104 have canonical local fallback paths.
        # These links are created by us after verified extraction, never accepted from input archives.
        for link in (repo / 'evaluation_examples/task_assets', repo / 'cache/osworld_v2_assets'):
            if link.exists():
                raise ValueError('Unexpected packaged fallback asset directory')
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(assets, target_is_directory=True)
        return repo, assets, destination / 'dependencies', manifest
    except BaseException:
        shutil.rmtree(destination)
        raise
