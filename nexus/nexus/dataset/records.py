"""Streaming record parsing and task-input adaptation."""

from __future__ import annotations

import copy
import hashlib
import json
import sqlite3
import tempfile

from collections.abc import AsyncIterator, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import yaml

from pydantic import BaseModel, ConfigDict, Field

from nexus.dataset.cache import sha256_file
from nexus.dataset.item import DatasetItem, TaskFilesSpec, resolve_task_files_input
from nexus.dataset.reader import (
    BaseDatasetReader,
    DatasetDescriptor,
    DatasetReaderError,
    DatasetSplitDescriptor,
    _validate_query,
)


@dataclass(frozen=True)
class RecordContext:
    source_path: Path
    line: int
    source_sha256: str | None = None


class RecordReader(Protocol):
    path: Path

    def iter_records(self, *, with_source_digest: bool = False) -> Iterator[tuple[dict[str, Any], RecordContext]]: ...


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f'{value} is not a JSON value')


class JsonlRecordReader:
    def __init__(self, path: str | Path, *, max_line_bytes: int = 16 * 1024 * 1024) -> None:
        self.path = Path(path).expanduser().resolve()
        self.max_line_bytes = max_line_bytes

    def iter_records(self, *, with_source_digest: bool = False) -> Iterator[tuple[dict[str, Any], RecordContext]]:
        before = self.path.stat()
        source_digest = sha256_file(self.path) if with_source_digest else None
        digest = hashlib.sha256()
        with self.path.open('rb') as stream:
            line = 0
            while raw := stream.readline(self.max_line_bytes + 1):
                line += 1
                digest.update(raw)
                if len(raw) > self.max_line_bytes:
                    raise ValueError(f'{self.path}:{line}: JSONL record exceeds size limit')
                try:
                    value = json.loads(raw, parse_constant=_reject_json_constant)
                    if not isinstance(value, dict):
                        raise ValueError('record must be an object')
                except (ValueError, UnicodeError) as exc:
                    raise ValueError(f'{self.path}:{line}: invalid JSONL object') from exc
                yield value, RecordContext(self.path, line, source_digest)
        after = self.path.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ) or (source_digest is not None and source_digest != digest.hexdigest()):
            raise ValueError(f'{self.path}: JSONL source changed while reading')


class ImportFiles(BaseModel):
    model_config = ConfigDict(extra='forbid')
    path_field: str = Field(min_length=1)
    type: Literal['directory', 'zip', 'auto'] = 'directory'
    root_name_field: str | None = Field(default=None, min_length=1)


class ImportCompat(BaseModel):
    model_config = ConfigDict(extra='forbid')
    files_to_cwd: bool = False


class ImportSpec(BaseModel):
    model_config = ConfigDict(extra='forbid')
    schema_version: Literal[1] = 1
    record_kind: Literal['auto', 'task_input', 'run_input', 'dataset_item'] = 'dataset_item'
    task_name: str | None = Field(default=None, min_length=1)
    id_field: str = Field(default='id', min_length=1)
    generate_ids: bool = False
    split: str = Field(default='test', pattern=r'^[a-zA-Z0-9][a-zA-Z0-9_.-]*$')
    files: ImportFiles | None = None
    compat: ImportCompat = Field(default_factory=ImportCompat)

    @classmethod
    def from_file(cls, path: str | Path) -> ImportSpec:
        with Path(path).open(encoding='utf-8') as stream:
            return cls.model_validate(yaml.safe_load(stream))


def resolve_import_spec(
    value: ImportSpec | Mapping[str, Any] | str | Path | None,
    *,
    id_field: str | None = None,
    files_field: str | None = None,
) -> ImportSpec:
    """Apply explicit field overrides without changing an existing spec's format."""
    if value is None:
        spec = ImportSpec(record_kind='auto', generate_ids=True)
    elif isinstance(value, (str, Path)):
        spec = ImportSpec.from_file(value)
    else:
        spec = ImportSpec.model_validate(value)
    data = spec.model_dump()
    if id_field is not None:
        data.update(id_field=id_field, generate_ids=False)
    if files_field is not None:
        data['files'] = {**(data['files'] or {'type': 'auto'}), 'path_field': files_field}
    resolved = ImportSpec.model_validate(data)
    if resolved.files is not None and resolved.files.path_field == resolved.id_field:
        raise ValueError('ID and attachment fields must be different')
    return resolved


class RecordAdapter:
    def __init__(self, spec: ImportSpec) -> None:
        self.spec = spec

    def convert(self, record: Mapping[str, Any], source_context: RecordContext) -> DatasetItem:
        data = copy.deepcopy(dict(record))
        kind = self.spec.record_kind
        if kind == 'auto':
            # The id/body envelope is reserved for standard DatasetItem records.
            envelope_fields = {'id', 'body', 'files', 'merge', 'metadata', self.spec.id_field}
            if self.spec.files is not None:
                envelope_fields.add(self.spec.files.path_field)
            is_item = (
                isinstance(data.get('body'), Mapping) and self.spec.id_field in data and data.keys() <= envelope_fields
            )
            kind = 'dataset_item' if is_item else 'task_input'
        item_id = data.get(self.spec.id_field)
        generated_id = self.spec.id_field not in data and self.spec.generate_ids
        if generated_id:
            item_id = f'line-{source_context.line:08d}'
        if not isinstance(item_id, str) or not item_id.strip():
            raise ValueError(f'record requires a non-empty string ID in field {self.spec.id_field!r}')
        raw_files = None
        if self.spec.files is not None:
            path = data.pop(self.spec.files.path_field, None)
            if path is None:
                raise ValueError(f'Missing file field {self.spec.files.path_field!r}')
            raw_files = {'path': path}
            if self.spec.files.type != 'auto':
                raw_files['type'] = self.spec.files.type
            if self.spec.files.root_name_field is not None:
                root_name = record.get(self.spec.files.root_name_field)
                if not isinstance(root_name, str) or not root_name:
                    raise ValueError(f'Missing or invalid root_name field {self.spec.files.root_name_field!r}')
                raw_files['root_name'] = root_name
            if self.spec.files.type == 'auto':
                raw_files = resolve_task_files_input(raw_files, base_dir=source_context.source_path.parent)
                raw_files['type'] = 'directory' if Path(raw_files['path']).is_dir() else 'zip'
        if kind == 'dataset_item':
            data['id'] = item_id
            if raw_files is not None:
                if data.get('files') is not None:
                    raise ValueError('Conflicting file declarations')
                data['files'] = raw_files
            item = DatasetItem.from_mapping(data, source_path=source_context.source_path)
        else:
            if kind == 'run_input':
                if self.spec.id_field == 'id':
                    data.pop('id', None)
                task_files = data.pop('task_files', None)
                if raw_files is not None and task_files is not None:
                    raise ValueError('Conflicting task_files declarations')
                raw_files = raw_files if raw_files is not None else task_files
                body = data
            else:
                body = {'task_config': data}
                if self.spec.task_name is not None:
                    body['task_name'] = self.spec.task_name
            files = (
                TaskFilesSpec.model_validate(
                    resolve_task_files_input(raw_files, base_dir=source_context.source_path.parent)
                )
                if raw_files is not None
                else None
            )
            item = DatasetItem(id=item_id, body=body, files=files, source_path=source_context.source_path)
        if item.files is not None and not item.files.is_remote:
            local = item.local_task_files_path()
            if local is None or not local.exists():
                raise ValueError('Declared files do not exist')
            if item.files.type == 'directory':
                if not local.is_dir():
                    raise ValueError('Declared directory is not a directory')
                if item.files.root_name is None:
                    item.files = TaskFilesSpec.model_validate({
                        **item.files.model_dump(exclude_none=True),
                        'root_name': local.name,
                    })
        if self.spec.compat.files_to_cwd:
            if item.files is None:
                raise ValueError('files_to_cwd requires task files')
            runner = item.body.setdefault('runner', {})
            runner['task_files_materialization'] = {'mode': 'cwd', 'target': 'host'}
        if generated_id:
            if source_context.source_sha256 is None:
                raise ValueError('Generated IDs require the source content digest')
            item.metadata['import_record'] = {
                'source_sha256': source_context.source_sha256,
                'line': source_context.line,
            }
        return item


class RecordDatasetReader(BaseDatasetReader):
    def __init__(self, record_reader: RecordReader, adapter: RecordAdapter) -> None:
        super().__init__()
        self.record_reader = record_reader
        self.adapter = adapter

    async def describe(self) -> DatasetDescriptor:
        self._require_open()
        path = self.record_reader.path
        return DatasetDescriptor(
            reference=str(path),
            resolved_reference=str(path),
            provider='jsonl',
            name=path.stem,
            default_split=self.adapter.spec.split,
            splits=[DatasetSplitDescriptor(name=self.adapter.spec.split)],
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
        if split is not None and split != self.adapter.spec.split:
            raise DatasetReaderError('dataset_split_not_found', detail=split)
        if limit == 0:
            return
        count = 0
        selected = None
        with tempfile.TemporaryDirectory(prefix='nexus-record-ids-') as directory:
            connection = sqlite3.connect(str(Path(directory) / 'ids.sqlite'))
            try:
                connection.execute('CREATE TABLE ids (id TEXT PRIMARY KEY)')
                for record, context in self.record_reader.iter_records(
                    with_source_digest=self.adapter.spec.generate_ids
                ):
                    try:
                        item = self.adapter.convert(record, context)
                        connection.execute('INSERT INTO ids VALUES (?)', (item.id,))
                    except sqlite3.IntegrityError as exc:
                        raise DatasetReaderError(
                            'dataset_duplicate_item_id', detail=f'{context.source_path}:{context.line}'
                        ) from exc
                    except (TypeError, ValueError) as exc:
                        raise ValueError(f'{context.source_path}:{context.line}: {exc}') from exc
                    if item_id is not None:
                        if item.id == item_id:
                            selected = item
                        continue
                    yield item
                    count += 1
                    if limit is not None and count >= limit:
                        return
            finally:
                connection.close()
        if item_id is not None:
            if selected is None:
                raise DatasetReaderError('dataset_item_not_found', detail=item_id)
            yield selected
