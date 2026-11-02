"""Target-neutral streaming dataset build and attachment freezing."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shutil
import sqlite3
import stat
import tempfile
import zipfile

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from nexus.dataset.access import DataAccessContext
from nexus.dataset.blob import FileResolver, LocalBlobReader
from nexus.dataset.cache import sha256_file
from nexus.dataset.item import DatasetItem, LocalFileRef, TaskFilesSpec
from nexus.dataset.reader import DatasetReader
from nexus.dataset.zip_index import (
    NexusZipError,
    _safe_directory_files,
    read_entry_bytes,
    read_zip_index_from_file,
    write_nexus_zip_directory_to_file,
)


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def validate_component(value: str) -> str:
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_.-]{0,255}', value):
        raise ValueError(f'Invalid dataset path component: {value!r}')
    return value


@dataclass(frozen=True)
class DatasetBuild:
    root: Path
    splits: dict[str, int]
    default_split: str
    content_sha256: str

    @property
    def item_count(self) -> int:
        return sum(self.splits.values())

    def iter_records(self, split: str) -> Iterator[dict[str, Any]]:
        with (self.root / 'records' / f'{validate_component(split)}.jsonl').open(encoding='utf-8') as stream:
            for line in stream:
                yield json.loads(line)

    def validate_records(self) -> None:
        digest = hashlib.sha256(canonical_json({'default_split': self.default_split, 'format': 1}) + b'\n')
        for split, expected in sorted(self.splits.items()):
            count = 0
            for record in self.iter_records(split):
                digest.update(canonical_json({'split': split, 'record': record}) + b'\n')
                count += 1
            if count != expected:
                raise ValueError('Frozen dataset record count changed')
        if digest.hexdigest() != self.content_sha256:
            raise ValueError('Frozen dataset records changed')


class DatasetBuilder:
    def __init__(
        self,
        *,
        max_file_bytes: int = 512 * 1024 * 1024,
        max_bundle_bytes: int = 4 * 1024**3,
        files_to_cwd: bool = False,
    ) -> None:
        self.max_file_bytes = max_file_bytes
        self.max_bundle_bytes = max_bundle_bytes
        self.files_to_cwd = files_to_cwd

    async def build(
        self,
        reader: DatasetReader,
        output: Path,
        context: DataAccessContext,
        *,
        split: str | None = None,
    ) -> DatasetBuild:
        output = output.resolve()
        output.mkdir(parents=True, exist_ok=True)
        if any(output.iterdir()):
            raise ValueError('Dataset build output must be empty')
        (output / 'objects').mkdir()
        (output / 'records').mkdir()
        descriptor = await reader.describe()
        splits = [split] if split else [entry.name for entry in descriptor.splits]
        if not splits:
            splits = [descriptor.default_split or 'test']
        counts: dict[str, int] = {}
        digest = hashlib.sha256()
        default_split = split or descriptor.default_split or splits[0]
        digest.update(canonical_json({'default_split': default_split, 'format': 1}) + b'\n')
        db = sqlite3.connect(str(output / 'ids.sqlite'))
        try:
            db.execute('CREATE TABLE ids (split TEXT, id TEXT, PRIMARY KEY (split,id))')
            for name in sorted(splits):
                validate_component(name)
                count = 0
                with (output / 'records' / f'{name}.jsonl').open('wb') as stream:
                    async for item in reader.iter_items(split=name):
                        if self.files_to_cwd:
                            item = item.model_copy(deep=True)
                            if item.files is None:
                                raise ValueError('files_to_cwd requires task files')
                            item.body.setdefault('runner', {})['task_files_materialization'] = {
                                'mode': 'cwd',
                                'target': 'host',
                            }
                        validate_component(item.id)
                        try:
                            db.execute('INSERT INTO ids VALUES (?, ?)', (name, item.id))
                        except sqlite3.IntegrityError as exc:
                            raise ValueError(f'Duplicate item ID in {name}: {item.id}') from exc
                        record = item.model_dump(exclude_none=True)
                        if files := item.resolved_task_files():
                            record['files'] = await self._freeze_files(item, files, output, context)
                        data = canonical_json(record)
                        if len(data) > 16 * 1024 * 1024:
                            raise ValueError(f'Dataset record exceeds size limit: {item.id}')
                        stream.write(data + b'\n')
                        digest.update(canonical_json({'split': name, 'record': record}) + b'\n')
                        count += 1
                if not count:
                    raise ValueError(f'Cannot publish empty split: {name}')
                counts[name] = count
        finally:
            db.close()
            (output / 'ids.sqlite').unlink(missing_ok=True)
        return DatasetBuild(output, counts, default_split, digest.hexdigest())

    async def _freeze_files(
        self,
        item: DatasetItem,
        files: TaskFilesSpec,
        output: Path,
        context: DataAccessContext,
    ) -> dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix='bundle-', dir=output) as work:
            work_path = Path(work)
            archive = work_path / 'files.zip'
            root_name = files.root_name
            compatibility = item.body.get('runner', {}).get('task_files_materialization', {}).get('mode') == 'cwd'
            if files.type == 'directory':
                source = files.file_ref()
                if not isinstance(source, LocalFileRef):
                    raise ValueError('Directory files must be local')
                path = Path(source.path)
                if output.is_relative_to(path.resolve()):
                    raise ValueError('Dataset build output must not be inside an attached directory')
                await asyncio.to_thread(self._check_directory_limits, path)
                root_name = root_name or path.name
                await asyncio.to_thread(write_nexus_zip_directory_to_file, path, archive, deterministic=True)
            else:
                blob = FileResolver.open(files.file_ref(), context, size=files.size)
                metadata = await blob.stat()
                if metadata.size > self.max_bundle_bytes:
                    raise ValueError('ZIP exceeds bundle size limit')
                if files.size is not None and files.size != metadata.size:
                    raise ValueError('ZIP size mismatch')
                await blob.download_to(archive, sha256=files.sha256)
                try:
                    index = read_zip_index_from_file(archive)
                    if sum(e.uncompressed_size for e in index.entries.values()) > self.max_bundle_bytes:
                        raise ValueError('Expanded ZIP exceeds bundle size limit')
                    if any(e.uncompressed_size > self.max_file_bytes for e in index.entries.values()):
                        raise ValueError('ZIP entry exceeds file size limit')
                    if compatibility and root_name is None:
                        root_name = await asyncio.to_thread(self._normalize_zip, archive, work_path, True, None)
                except NexusZipError:
                    root_name = await asyncio.to_thread(
                        self._normalize_zip, archive, work_path, compatibility, root_name
                    )
            if archive.stat().st_size > self.max_bundle_bytes:
                raise ValueError('ZIP exceeds bundle size limit')
            index = read_zip_index_from_file(archive)
            if any(e.uncompressed_size > self.max_file_bytes for e in index.entries.values()):
                raise ValueError('ZIP entry exceeds file size limit')
            if sum(e.uncompressed_size for e in index.entries.values()) > self.max_bundle_bytes:
                raise ValueError('Expanded ZIP exceeds bundle size limit')
            verifier = LocalBlobReader(LocalFileRef(path=str(archive)))
            for entry in index.entries.values():
                await read_entry_bytes(entry, verifier.read_range)
            if compatibility and root_name is None:
                raise ValueError('files_to_cwd requires root_name for a ZIP without a unique top-level directory')
            digest = await asyncio.to_thread(sha256_file, archive)
            target = output / 'objects' / f'{digest}.zip'
            if not target.exists():
                os.replace(archive, target)
            return TaskFilesSpec(
                type='zip',
                source=LocalFileRef(path=f'objects/{digest}.zip'),
                root_name=root_name,
                size=target.stat().st_size,
                sha256=digest,
            ).model_dump(exclude_none=True)

    def _check_directory_limits(self, path: Path) -> None:
        total = 0
        for child in _safe_directory_files(path):
            size = child.stat().st_size
            if size > self.max_file_bytes:
                raise ValueError('Directory entry exceeds file size limit')
            total += size
            if total > self.max_bundle_bytes:
                raise ValueError('Directory exceeds expanded bundle size limit')

    def _normalize_zip(
        self,
        archive: Path,
        work: Path,
        compatibility: bool,
        root_name: str | None,
    ) -> str | None:
        extracted = work / 'expanded'
        extracted.mkdir()
        with zipfile.ZipFile(archive) as source:
            entries = source.infolist()
            if sum(e.file_size for e in entries) > self.max_bundle_bytes:
                raise ValueError('Expanded ZIP exceeds bundle size limit')
            seen: set[str] = set()
            roots = set()
            nested = True
            for entry in entries:
                path = PurePosixPath(entry.filename)
                parts = entry.filename.rstrip('/').split('/')
                if path.is_absolute() or '\\' in entry.filename or any(p in ('', '.', '..') for p in parts):
                    raise ValueError('Unsafe ZIP entry path')
                if entry.filename.rstrip('/') in seen:
                    raise ValueError('Duplicate ZIP entry')
                seen.add(entry.filename.rstrip('/'))
                mode = entry.external_attr >> 16
                if stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR):
                    raise ValueError('ZIP links and special files are not supported')
                if entry.is_dir():
                    continue
                if entry.file_size > self.max_file_bytes:
                    raise ValueError('ZIP entry exceeds file size limit')
                roots.add(parts[0])
                nested = nested and len(parts) > 1
                target = extracted.joinpath(*parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.open(entry) as src, target.open('xb') as dst:
                    shutil.copyfileobj(src, dst, length=1024 * 1024)
                target.chmod(stat.S_IMODE(mode) & 0o777 if mode else 0o644)
            bundle_root = extracted
            if compatibility and nested and len(roots) == 1:
                original_root = next(iter(roots))
                bundle_root = extracted / original_root
                root_name = root_name or original_root
            write_nexus_zip_directory_to_file(bundle_root, archive, deterministic=True)
        return root_name
