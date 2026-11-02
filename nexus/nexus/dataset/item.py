"""Dataset item schema."""

from __future__ import annotations

import json

from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml

from pydantic import BaseModel, Field, field_validator, model_validator


class TaijiRemoteDatasetRef(BaseModel):
    """Remote dataset coordinates for an item fetched from IO service."""

    namespace: str
    dataset: str
    version: str
    split: str | None = None


RemoteDatasetRef = TaijiRemoteDatasetRef


class LocalFileRef(BaseModel):
    kind: Literal['local'] = 'local'
    path: str = Field(min_length=1)


class TaijiFileRef(BaseModel):
    kind: Literal['taiji'] = 'taiji'
    dataset: TaijiRemoteDatasetRef
    path: str = Field(min_length=1)


class CosFileRef(BaseModel):
    kind: Literal['cos'] = 'cos'
    bucket: str = Field(pattern=r'^[a-zA-Z0-9][a-zA-Z0-9.-]*$')
    region: str = Field(pattern=r'^[a-z][a-z0-9-]+$')
    key: str = Field(min_length=1)
    etag: str | None = None

    @field_validator('key')
    @classmethod
    def validate_key(cls, value: str) -> str:
        if value.startswith('/') or '\\' in value or any(p in ('', '.', '..') for p in value.split('/')):
            raise ValueError('COS key must be a relative object path without traversal')
        return value


FileRef = Annotated[LocalFileRef | TaijiFileRef | CosFileRef, Field(discriminator='kind')]


class TaskFilesSpec(BaseModel):
    """Task file declaration for a dataset item or RunInput."""

    type: Literal['directory', 'zip']
    path: str = ''
    sha256: str | None = None
    size: int | None = None
    remote: RemoteDatasetRef | None = None
    source: FileRef | None = None
    root_name: str | None = None
    schema_version: Literal[2] | None = None

    @field_validator('root_name')
    @classmethod
    def validate_root_name(cls, value: str | None) -> str | None:
        if value is not None and (not value or value in ('.', '..') or any(c in value for c in '/\\\x00')):
            raise ValueError('root_name must be one safe directory name')
        return value

    @model_validator(mode='after')
    def validate_source(self) -> TaskFilesSpec:
        if self.source is not None:
            if self.remote is not None:
                raise ValueError('task_files source and legacy remote are mutually exclusive')
            if not isinstance(self.source, LocalFileRef) and self.type != 'zip':
                raise ValueError('Remote task_files must be a ZIP')
            self.schema_version = 2
        return self

    @property
    def is_remote(self) -> bool:
        return self.remote is not None or isinstance(self.source, (TaijiFileRef, CosFileRef))

    def file_ref(self) -> LocalFileRef | TaijiFileRef | CosFileRef:
        if self.source is not None:
            return self.source
        if self.remote is not None:
            return TaijiFileRef(dataset=self.remote, path=self.path)
        return LocalFileRef(path=self.path)


def resolve_task_files_input(value: Any, *, base_dir: str | Path) -> Any:
    """Normalize user-facing local task-files shorthand into a ``TaskFilesSpec`` mapping."""

    if value is None:
        return None
    if isinstance(value, TaskFilesSpec):
        value = value.model_dump(exclude_none=True)
    if isinstance(value, str):
        value = {'path': value}
    if not isinstance(value, Mapping):
        raise TypeError('task_files must be a path string or mapping')

    resolved = dict(value)
    if resolved.get('remote') is not None:
        return resolved

    if resolved.get('source') is not None:
        parsed = TaskFilesSpec.model_validate(resolved)
        if not isinstance(parsed.source, LocalFileRef):
            return resolved
        path = Path(parsed.source.path).expanduser()
        if not path.is_absolute():
            path = Path(base_dir) / path
        resolved['source'] = {'kind': 'local', 'path': str(path.resolve())}
        return resolved

    raw_path = resolved.get('path')
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError('Local task_files requires a non-empty path')
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = Path(base_dir) / path
    path = path.resolve()
    resolved['path'] = str(path)
    if 'type' not in resolved:
        resolved['type'] = 'zip' if path.is_file() or path.suffix.lower() == '.zip' else 'directory'
    return resolved


class DatasetItem(BaseModel):
    """A dataset item that contributes a partial ``RunInput``."""

    id: str
    body: dict[str, Any]
    merge: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    files: TaskFilesSpec | None = None
    source_path: Path | None = Field(default=None, exclude=True)
    remote_ref: RemoteDatasetRef | None = Field(default=None, exclude=True)

    @classmethod
    def from_mapping(
        cls,
        data: dict[str, Any],
        *,
        source_path: Path | None = None,
        remote_ref: RemoteDatasetRef | None = None,
    ) -> DatasetItem:
        item = cls.model_validate(data)
        item.source_path = source_path
        item.remote_ref = remote_ref
        return item

    @classmethod
    def from_file(cls, path: str | Path) -> DatasetItem:
        item_path = Path(path)
        with open(item_path, encoding='utf-8') as f:
            data = json.load(f) if item_path.suffix == '.json' else yaml.safe_load(f)
        if not isinstance(data, dict):
            raise TypeError(f'Dataset item must be a mapping: {item_path}')
        return cls.from_mapping(data, source_path=item_path)

    def local_task_files_path(self) -> Path | None:
        """Return the local task-files path if this item declares one."""

        if self.files is None or self.files.is_remote:
            return None
        source = self.files.file_ref()
        if not isinstance(source, LocalFileRef):
            return None
        path = source.path
        if self.source_path is None:
            return Path(path)
        return (self.source_path.parent / path).resolve()

    def resolved_task_files(self) -> TaskFilesSpec | None:
        """Return item files adjusted for the ``RunInput.task_files`` field."""

        if self.files is None:
            return None
        if self.files.is_remote:
            return self.files
        if self.remote_ref is not None and self.files.source is None:
            return self.files.model_copy(update={'remote': self.remote_ref})
        path = self.local_task_files_path()
        if path is None:
            return self.files
        if self.files.source is not None:
            return self.files.model_copy(update={'source': LocalFileRef(path=str(path))})
        return self.files.model_copy(update={'path': str(path)})
