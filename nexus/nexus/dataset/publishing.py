"""Publisher interface and the existing Taiji service adapter."""

from __future__ import annotations

import asyncio
import hashlib
import os
import sqlite3
import tempfile

from pathlib import Path
from typing import Any, Protocol

import yaml

from nexus.dataset.access import DataAccessContext
from nexus.dataset.build import DatasetBuild, canonical_json
from nexus.dataset.cos import PublishedDatasetRef
from nexus.dataset.exceptions import UcAssetNotFoundError
from nexus.dataset.item import DatasetItem, TaskFilesSpec
from nexus.dataset.publisher import publish_local_dataset
from nexus.dataset.reader import _is_not_found_error, open_dataset
from nexus.dataset.uri import parse_dataset_uri


class DatasetPublisher(Protocol):
    async def publish(
        self,
        build: DatasetBuild,
        target: str,
        context: DataAccessContext,
    ) -> PublishedDatasetRef: ...


def export_local_build(build: DatasetBuild, output: Path, *, name: str) -> None:
    """Render a frozen build into the existing local DatasetReader layout."""
    output.mkdir(parents=True, exist_ok=True)
    splits: dict[str, Any] = {}
    for split in build.splits:
        items = []
        for record in build.iter_records(split):
            if record.get('files'):
                files = TaskFilesSpec.model_validate(record['files'])
                record['files'] = {
                    'type': 'zip',
                    'path': os.path.relpath(
                        build.root / f'objects/{files.sha256}.zip',
                        output / 'items' / split,
                    ),
                    'sha256': files.sha256,
                    'size': files.size,
                    **({'root_name': files.root_name} if files.root_name else {}),
                }
            relative = f'items/{split}/{record["id"]}.json'
            path = output / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(canonical_json(record))
            items.append(relative)
        splits[split] = {'items': items}
    (output / 'dataset.yaml').write_text(
        yaml.safe_dump({
            'name': name,
            'default_split': build.default_split,
            'splits': splits,
        }),
        encoding='utf-8',
    )


class TaijiPublisher:
    def __init__(self, **options: Any) -> None:
        self.options = options

    async def publish(
        self,
        build: DatasetBuild,
        target: str,
        context: DataAccessContext,
    ) -> PublishedDatasetRef:
        uri = parse_dataset_uri(target)
        await asyncio.to_thread(build.validate_records)
        if uri.version in ('latest', 'dev'):
            raise ValueError('Publish requires an explicit release version')
        if not await self._verify_existing(build, uri.display, context):
            with tempfile.TemporaryDirectory(prefix='nexus-taiji-publish-') as temporary:
                source = Path(temporary) / 'dataset'
                export_local_build(build, source, name=uri.dataset)
                try:
                    await publish_local_dataset(
                        source,
                        namespace=uri.namespace,
                        dataset_name=uri.dataset,
                        release_version=uri.version,
                        authenticator=context.dataset_authenticator(),
                        **self.options,
                    )
                except Exception:
                    # A commit response can be lost after the fixed release became visible.
                    if not await self._verify_existing(build, uri.display, context):
                        raise
            if not await self._verify_existing(build, uri.display, context):
                raise RuntimeError('Published Taiji release is not readable yet; retry the same build')
        return PublishedDatasetRef(
            backend='taiji',
            data_uri=uri.display,
            resolved_version=uri.version,
            content_sha256=build.content_sha256,
            item_count=build.item_count,
            splits=build.splits,
        )

    async def _verify_existing(self, build: DatasetBuild, reference: str, context: DataAccessContext) -> bool:
        exists = False
        try:
            async with open_dataset(reference, authenticator=context.dataset_authenticator()) as reader:
                descriptor = await reader.describe()
                exists = True
                if {s.name for s in descriptor.splits} != set(
                    build.splits
                ) or descriptor.default_split != build.default_split:
                    raise FileExistsError(f'Taiji release has different splits: {reference}')
                with tempfile.TemporaryDirectory(prefix='nexus-release-check-') as temporary:
                    db = sqlite3.connect(str(Path(temporary) / 'records.sqlite'))
                    try:
                        db.execute('CREATE TABLE records (id TEXT PRIMARY KEY, digest TEXT)')
                        for split in build.splits:
                            for record in build.iter_records(split):
                                db.execute('INSERT INTO records VALUES (?, ?)', (record['id'], _record_digest(record)))
                            async for item in reader.iter_items(split=split):
                                expected = db.execute('SELECT digest FROM records WHERE id=?', (item.id,)).fetchone()
                                if expected is None or expected[0] != _record_digest(
                                    item.model_dump(exclude_none=True)
                                ):
                                    raise FileExistsError(f'Taiji release has different content: {reference}')
                                db.execute('DELETE FROM records WHERE id=?', (item.id,))
                            if db.execute('SELECT count(*) FROM records').fetchone()[0]:
                                raise FileExistsError(f'Taiji release has missing records: {reference}')
                    finally:
                        db.close()
            return True
        except Exception as exc:
            if not exists and (isinstance(exc, UcAssetNotFoundError) or _is_not_found_error(exc)):
                return False
            raise


def _record_digest(record: dict[str, Any]) -> str:
    item = DatasetItem.model_validate(record)
    data = item.model_dump(exclude_none=True)
    if item.files is not None:
        data['files'] = {
            'type': item.files.type,
            'sha256': item.files.sha256,
            'size': item.files.size,
            'root_name': item.files.root_name,
        }
    return hashlib.sha256(canonical_json(data)).hexdigest()
