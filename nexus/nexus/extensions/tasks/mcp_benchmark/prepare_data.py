from __future__ import annotations

import argparse
import subprocess
import sys

from pathlib import Path
from typing import TypedDict

from .constants import (
    ATLAS_DATASET_REPO_URL,
    ATLAS_DATASET_ROOT,
    CONVERTED_DATASETS_ROOT,
    TOOLATHLON_DATASET_REPO_URL,
    TOOLATHLON_DATASET_ROOT,
    UNIVERSE_DATASET_REPO_URL,
    UNIVERSE_DATASET_ROOT,
)
from .manifests import clear_manifest_caches, write_manifest


class _DatasetMeta(TypedDict):
    root: Path
    repo_url: str


_DATASET_META: dict[str, _DatasetMeta] = {
    'toolathlon': {'root': TOOLATHLON_DATASET_ROOT, 'repo_url': TOOLATHLON_DATASET_REPO_URL},
    'atlas': {'root': ATLAS_DATASET_ROOT, 'repo_url': ATLAS_DATASET_REPO_URL},
    'universe': {'root': UNIVERSE_DATASET_ROOT, 'repo_url': UNIVERSE_DATASET_REPO_URL},
}


def _run_git(args: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ['git', *args],
        cwd=str(cwd) if cwd is not None else None,
        check=True,
        text=True,
        capture_output=True,
    )


def _sync_repo(path: Path, repo_url: str) -> str:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        _run_git(['clone', '--depth', '1', repo_url, str(path)])
        return 'cloned'

    if not (path / '.git').exists():
        raise RuntimeError(f'{path} exists but is not a git repository')

    status = _run_git(['status', '--porcelain'], cwd=path)
    if status.stdout.strip():
        return 'dirty-skip-update'

    current_url = _run_git(['remote', 'get-url', 'origin'], cwd=path).stdout.strip()
    if current_url != repo_url:
        return f'remote-mismatch:{current_url}'

    _run_git(['fetch', '--prune', 'origin'], cwd=path)
    _run_git(['pull', '--ff-only'], cwd=path)
    return 'updated'


def _selected_datasets(raw: str) -> list[str]:
    requested = [item.strip() for item in raw.split(',') if item.strip()]
    invalid = sorted(set(requested) - set(_DATASET_META))
    if invalid:
        raise ValueError(f'Unsupported datasets: {", ".join(invalid)}')
    return [dataset for dataset in _DATASET_META if dataset in requested]


def _jsonl_row_count(path: Path) -> int:
    with path.open(encoding='utf-8') as handle:
        return sum(1 for line in handle if line.strip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description='Download MCP benchmark dataset repos and convert them into per-dataset JSONL manifests.'
    )
    parser.add_argument('--datasets', default='toolathlon,atlas,universe')
    parser.add_argument('--skip-sync', action='store_true', help='Use local dataset repos as-is without clone/pull.')
    args = parser.parse_args(argv)

    datasets = _selected_datasets(args.datasets)
    CONVERTED_DATASETS_ROOT.mkdir(parents=True, exist_ok=True)

    for dataset in datasets:
        root = _DATASET_META[dataset]['root']
        repo_url = _DATASET_META[dataset]['repo_url']
        if args.skip_sync:
            print(f'[{dataset}] skip sync, using {root}')
            continue
        result = _sync_repo(root, repo_url)
        print(f'[{dataset}] repo {result}: {root}')

    written: list[tuple[str, Path, int]] = []
    for dataset in datasets:
        path = write_manifest(dataset)
        row_count = _jsonl_row_count(path)
        written.append((dataset, path, row_count))

    clear_manifest_caches()

    for dataset, path, row_count in written:
        print(f'[{dataset}] wrote {row_count} rows -> {path}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
