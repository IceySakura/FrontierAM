"""Package an upstream macOSWorld checkout as a portable Nexus dataset.

Usage:
    python -m nexus.extensions.tasks.mac_os_world.dataset \
        workspace/macosworld workspace/datasets/macosworld

The dataset contains original task JSON and attribution, not macOS VM images or
benchmark_files. Supply the agent and a prepared macOS runtime separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess

from pathlib import Path
from typing import Any

import yaml


SOURCE_URL = 'https://github.com/showlab/macosworld'
LICENSE = 'CC-BY-NC-4.0'
AUTHORS = [{'name': name} for name in ['Pei Yang', 'Hai Ci', 'Mike Zheng Shou']]


def build_mac_os_world_dataset(source: Path, output: Path, *, source_revision: str | None = None) -> dict[str, int]:
    """Keep one item per upstream task; separate the safety subset from test."""
    source = source.resolve()
    output = output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f'Dataset output must be an empty directory: {output}')
    license_file = source / 'LICENSE'
    if not license_file.is_file():
        raise FileNotFoundError(f'Upstream license not found: {license_file}')
    task_paths = sorted((source / 'tasks').glob('*/*.json'))
    if not task_paths:
        raise ValueError(f'No macOSWorld task JSON found under {source / "tasks"}')

    tasks: list[tuple[Path, dict[str, Any], str]] = []
    seen: set[str] = set()
    for path in task_paths:
        data = json.loads(path.read_text(encoding='utf-8'))
        task_id = str(data.get('id') or path.stem)
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', task_id):
            raise ValueError(f'Invalid dataset item id: {task_id!r}')
        if task_id in seen:
            raise ValueError(f'Duplicate macOSWorld task id: {task_id}')
        seen.add(task_id)
        tasks.append((path, data, task_id))

    splits: dict[str, dict[str, list[str]]] = {}
    for path, data, task_id in tasks:
        category = path.parent.name
        split = 'safety' if category == 'safety' else 'test'
        item_dir = output / 'items' / task_id
        files_dir = item_dir / 'files'
        files_dir.mkdir(parents=True)
        shutil.copyfile(path, files_dir / 'task.json')
        shutil.copyfile(license_file, files_dir / 'LICENSE')
        attribution = {
            'source_url': SOURCE_URL,
            'source_revision': source_revision,
            'source_path': path.relative_to(source).as_posix(),
            'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'license': LICENSE,
            'authors': AUTHORS,
            'modifications': 'Original task JSON is unchanged; added Nexus dataset packaging.',
        }
        (files_dir / 'SOURCE.json').write_text(
            json.dumps(attribution, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
        )
        task_text = data.get('task')
        item = {
            'id': task_id,
            'files': {'type': 'directory', 'path': 'files'},
            'merge': {'imports': 'append_unique'},
            'body': {
                'imports': ['nexus.extensions.tasks.mac_os_world'],
                'task_name': 'mac_os_world',
                'task_config': {'task_file': 'task.json', 'category': category, 'task_id': task_id},
            },
            'metadata': {
                **attribution,
                'category': category,
                'instruction': str(task_text.get('en') or '') if isinstance(task_text, dict) else str(task_text or ''),
                'instructions': task_text if isinstance(task_text, dict) else {'en': str(task_text or '')},
                'languages': sorted(task_text) if isinstance(task_text, dict) else ['en'],
                'requires': 'Nexus mac_os_world task_files support and a prepared macOS benchmark environment.',
            },
        }
        (item_dir / 'item.yaml').write_text(yaml.safe_dump(item, allow_unicode=True, sort_keys=False), encoding='utf-8')
        splits.setdefault(split, {'items': []})['items'].append(f'items/{task_id}/item.yaml')

    descriptor = {
        'schema_version': 1,
        'name': 'macosworld',
        'authors': AUTHORS,
        'maintainers': [],
        'default_split': 'test' if 'test' in splits else 'safety',
        'splits': splits,
    }
    (output / 'dataset.yaml').write_text(
        yaml.safe_dump(descriptor, allow_unicode=True, sort_keys=False), encoding='utf-8'
    )
    return {name: len(spec['items']) for name, spec in splits.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, help='Upstream checkout containing tasks/ and LICENSE')
    parser.add_argument('output', type=Path, help='Empty local dataset directory')
    args = parser.parse_args()
    revision = subprocess.run(
        ['git', '-C', str(args.source), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=False
    )
    counts = build_mac_os_world_dataset(
        args.source, args.output, source_revision=revision.stdout.strip() if revision.returncode == 0 else None
    )
    print(json.dumps({'path': str(args.output.resolve()), 'splits': counts, 'total': sum(counts.values())}))


if __name__ == '__main__':
    main()
