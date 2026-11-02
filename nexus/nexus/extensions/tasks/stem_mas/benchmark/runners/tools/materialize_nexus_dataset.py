"""Materialize a STEM-MAS source adapter as a local Nexus Dataset."""

from __future__ import annotations

import argparse
import sys

from pathlib import Path


def ensure_repo_root_on_path() -> None:
    for parent in Path(__file__).resolve().parents:
        if (parent / 'pyproject.toml').exists() and (parent / 'nexus').is_dir():
            sys.path.insert(0, str(parent))
            return


ensure_repo_root_on_path()

from nexus.extensions.tasks.stem_mas.benchmark.eval.dataset_io import write_local_nexus_dataset  # noqa: E402
from nexus.extensions.tasks.stem_mas.benchmark.eval.profile import BenchmarkProfile  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path, help='STEM-MAS benchmark profile used to select a source')
    parser.add_argument('--input', help='Optional local dataset source override')
    parser.add_argument('--output', required=True, type=Path, help='Local Nexus dataset authoring directory')
    parser.add_argument('--split', default='test')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    profile = BenchmarkProfile.load(args.config)
    dataset = profile.build_dataset_adapter(args.input)
    items = dataset.load_items(limit=args.limit)
    output = write_local_nexus_dataset(
        items,
        args.output,
        name=dataset.name,
        split=args.split,
        force=args.force,
    )
    print(f'Materialized {len(items)} items for {dataset.name} at {output}')


if __name__ == '__main__':
    main()
