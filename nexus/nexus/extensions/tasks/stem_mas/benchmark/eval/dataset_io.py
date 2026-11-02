"""Local Nexus dataset authoring helpers for STEM-MAS source adapters."""

from __future__ import annotations

import shutil

from collections import Counter
from pathlib import Path

import yaml

from nexus.dataset import DatasetItem


def write_local_nexus_dataset(
    items: list[DatasetItem],
    output_dir: str | Path,
    *,
    name: str,
    split: str = 'test',
    force: bool = False,
) -> Path:
    """Write items in the local authoring layout accepted by ``nexus dataset``."""

    if not items:
        raise ValueError('Cannot materialize an empty Nexus dataset')
    item_ids = [item.id for item in items]
    duplicate_ids = sorted(item_id for item_id, count in Counter(item_ids).items() if count > 1)
    if duplicate_ids:
        raise ValueError(f'Duplicate dataset item id: {duplicate_ids[0]}')

    output_path = Path(output_dir).expanduser()
    if output_path.is_symlink():
        raise FileExistsError(f'Dataset output path must not be a symlink: {output_path}')
    root = output_path.resolve()
    if root.exists():
        if not root.is_dir():
            raise FileExistsError(f'Dataset output path must be a directory, not a file or symlink: {root}')
        if any(root.iterdir()):
            if not force:
                raise FileExistsError(f'Dataset output directory is not empty: {root}')
            if not _is_local_nexus_dataset(root):
                raise FileExistsError(
                    f'Refusing to replace non-dataset directory: {root}. '
                    'Remove it explicitly or choose another output path.'
                )
            shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    item_paths: list[str] = []
    for index, item in enumerate(items):
        relative_path = Path('items') / f'{index:06d}' / 'item.yaml'
        _write_yaml(root / relative_path, item.model_dump(mode='json', exclude_none=True))
        item_paths.append(relative_path.as_posix())

    _write_yaml(
        root / 'dataset.yaml',
        {
            'schema_version': 1,
            'name': name,
            'authors': [],
            'maintainers': [],
            'default_split': split,
            'splits': {split: {'items': item_paths}},
        },
    )
    return root


def _write_yaml(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as file:
        yaml.safe_dump(value, file, sort_keys=False, allow_unicode=True)


def _is_local_nexus_dataset(path: Path) -> bool:
    manifest_path = path / 'dataset.yaml'
    if not manifest_path.is_file():
        return False
    try:
        manifest = yaml.safe_load(manifest_path.read_text(encoding='utf-8'))
    except (OSError, yaml.YAMLError, UnicodeError):
        return False
    return (
        isinstance(manifest, dict) and manifest.get('schema_version') == 1 and isinstance(manifest.get('splits'), dict)
    )


__all__ = ['write_local_nexus_dataset']
