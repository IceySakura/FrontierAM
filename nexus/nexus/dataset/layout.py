"""Local dataset layout loading and template generation."""

from __future__ import annotations

import json
import shutil

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from nexus.dataset.item import DatasetItem


DATASET_YAML = 'dataset.yaml'
SUPPORTED_TEMPLATES = {'generic', 'terminal', 'swe'}


@dataclass(frozen=True)
class LocalDataset:
    """A local authoring dataset rooted at a directory."""

    root: Path
    raw: dict[str, Any]

    @property
    def default_split(self) -> str | None:
        value = self.raw.get('default_split')
        return str(value) if value else None

    @property
    def splits(self) -> dict[str, Any]:
        splits = self.raw.get('splits') or {}
        if not isinstance(splits, dict):
            raise TypeError(f'{self.root / DATASET_YAML}: splits must be a mapping')
        return splits

    def resolve_split(self, split: str | None = None) -> str:
        if split:
            if split not in self.splits:
                raise ValueError(f'Split {split!r} not found in {self.root / DATASET_YAML}')
            return split
        if self.default_split:
            if self.default_split not in self.splits:
                raise ValueError(f'Default split {self.default_split!r} not found in {self.root / DATASET_YAML}')
            return self.default_split
        if len(self.splits) == 1:
            return next(iter(self.splits))
        available = ', '.join(sorted(self.splits)) or '<none>'
        raise ValueError(f'Split is required for {self.root}; available splits: {available}')

    def item_paths(self, split: str | None = None) -> list[Path]:
        split_name = self.resolve_split(split)
        split_data = self.splits[split_name] or {}
        if not isinstance(split_data, dict):
            raise TypeError(f'Split {split_name!r} in {self.root / DATASET_YAML} must be a mapping')
        items = split_data.get('items') or []
        if not isinstance(items, list):
            raise TypeError(f'Split {split_name!r} items must be a list')
        return [(self.root / str(item_path)).resolve() for item_path in items]

    def load_items(
        self,
        *,
        split: str | None = None,
        item_id: str | None = None,
        limit: int | None = None,
    ) -> list[DatasetItem]:
        matched: list[DatasetItem] = []
        for item_path in self.item_paths(split):
            item = DatasetItem.from_file(item_path)
            if item_id is not None and item.id != item_id:
                continue
            matched.append(item)
            if limit is not None and len(matched) >= limit:
                break
        if item_id is not None and not matched:
            raise KeyError(f'Dataset item not found: {item_id}')
        return matched


def load_local_dataset(path: str | Path) -> LocalDataset:
    root = Path(path).expanduser().resolve()
    dataset_path = root / DATASET_YAML
    if not dataset_path.exists():
        raise FileNotFoundError(f'Local dataset metadata not found: {dataset_path}')
    with open(dataset_path, encoding='utf-8') as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise TypeError(f'{dataset_path} must contain a mapping')
    return LocalDataset(root=root, raw=data)


def is_local_dataset_path(value: str | Path) -> bool:
    path = Path(value).expanduser()
    return path.exists() and path.is_dir() and (path / DATASET_YAML).exists()


def create_dataset_template(
    path: str | Path,
    *,
    template: str = 'terminal',
    item_id: str | None = None,
    name: str | None = None,
    force: bool = False,
) -> Path:
    """Create a local dataset authoring template."""

    if template not in SUPPORTED_TEMPLATES:
        raise ValueError(f'Unsupported dataset template {template!r}; choose one of {sorted(SUPPORTED_TEMPLATES)}')

    root = Path(path).expanduser().resolve()
    if root.exists():
        if not root.is_dir():
            raise FileExistsError(f'Dataset path exists and is not a directory: {root}')
        if any(root.iterdir()):
            if not force:
                raise FileExistsError(f'Dataset directory is not empty: {root}')
            shutil.rmtree(root)

    item_ids = [item_id] if item_id is not None else [f'case-{index:03d}' for index in range(1, 4)]

    dataset_yaml = {
        'schema_version': 1,
        'name': name or root.name,
        'authors': [],
        'maintainers': [],
        'default_split': 'test',
        'splits': {
            'test': {
                'items': [f'items/{current_item_id}/item.yaml' for current_item_id in item_ids],
            },
        },
    }
    _write_yaml(root / DATASET_YAML, dataset_yaml)

    for current_item_id in item_ids:
        _write_template_item(root, template, current_item_id)
    return root


def _write_template_item(root: Path, template: str, item_id: str) -> None:
    item_dir = root / 'items' / item_id
    files_dir = item_dir / 'files'
    files_dir.mkdir(parents=True, exist_ok=True)

    if template == 'terminal':
        item = _terminal_item(item_id)
        (files_dir / 'instruction.md').write_text(
            'Solve the task in this directory. Add tests or solution files as needed.\n',
            encoding='utf-8',
        )
        (files_dir / 'tests').mkdir(exist_ok=True)
        (files_dir / 'solution').mkdir(exist_ok=True)
    elif template == 'swe':
        item = _swe_item(item_id)
    else:
        item = _generic_item(item_id)
        readme = files_dir / 'README.md'
        readme.write_text('Place case files for this dataset item here.\n', encoding='utf-8')

    _write_yaml(item_dir / 'item.yaml', item)


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def _generic_item(item_id: str) -> dict[str, Any]:
    return {
        'id': item_id,
        'files': {'type': 'directory', 'path': 'files'},
        'body': {
            'task_name': 'echo',
            'task_config': {
                'message': 'Hello from a Nexus dataset item.',
                'run_oracle': True,
            },
        },
    }


def _terminal_item(item_id: str) -> dict[str, Any]:
    return {
        'id': item_id,
        'files': {'type': 'directory', 'path': 'files'},
        'body': {
            'task_name': 'terminal_task',
            'task_config': {
                'message': 'Solve the task described in task_files/instruction.md.',
                'run_oracle': False,
            },
            'runtime_configs': {
                'main': {
                    'image': 'ubuntu:24.04',
                    'working_dir': '/workspace',
                    'resource_spec': {'cpu': 2, 'memory': 8},
                },
            },
        },
    }


def _swe_item(item_id: str) -> dict[str, Any]:
    return {
        'id': item_id,
        'body': {
            'task_name': 'echo',
            'task_config': {
                'message': json.dumps({'instance_id': item_id}),
                'run_oracle': False,
            },
        },
    }
