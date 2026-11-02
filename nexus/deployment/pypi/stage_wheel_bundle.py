from __future__ import annotations

import argparse
import shutil

from pathlib import Path, PurePosixPath


try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


BUNDLE_ROOT = PurePosixPath('nexus/_bundle')
RESOURCE_ROOTS = (BUNDLE_ROOT, PurePosixPath('nexus/gateway/trajectory_store_backends/redis_store'))
EXCLUDED_DIRECTORY_NAMES = {
    '.git',
    '.hatch',
    '.hg',
    '.mypy_cache',
    '.nox',
    '.pytest_cache',
    '.ruff_cache',
    '.tox',
    '.venv',
    '__pycache__',
}
EXCLUDED_FILE_NAMES = {'.DS_Store', '.git'}


def load_bundle_inclusion_map(project_root: Path) -> dict[Path, PurePosixPath]:
    pyproject_path = project_root / 'pyproject.toml'
    with pyproject_path.open('rb') as file:
        project = tomllib.load(file)

    force_include = project['tool']['hatch']['build']['targets']['wheel']['force-include']
    if not isinstance(force_include, dict):
        raise TypeError('Wheel force-include configuration must be a mapping')

    inclusion_map: dict[Path, PurePosixPath] = {}
    for source, destination in force_include.items():
        if not isinstance(source, str) or not isinstance(destination, str):
            raise TypeError('Wheel force-include sources and destinations must be strings')

        destination_path = PurePosixPath(destination)
        if not any(destination_path.is_relative_to(root) for root in RESOURCE_ROOTS):
            continue

        if destination_path in RESOURCE_ROOTS or destination_path.is_absolute() or '..' in destination_path.parts:
            raise ValueError(f'Invalid wheel bundle destination: {destination}')

        source_relative = Path(source)
        if source_relative.is_absolute() or '..' in source_relative.parts:
            raise ValueError(f'Wheel bundle source must stay inside the project: {source}')
        source_path = project_root / source_relative
        if not source_path.exists():
            raise FileNotFoundError(f'Wheel bundle source does not exist: {source_path}')

        inclusion_map[source_path] = destination_path

    if not inclusion_map:
        raise ValueError('Wheel force-include configuration contains no supported wheel resources')
    return inclusion_map


def _ignore_directory_entries(_directory: str, entries: list[str]) -> set[str]:
    return {entry for entry in entries if entry in EXCLUDED_DIRECTORY_NAMES or entry in EXCLUDED_FILE_NAMES}


def stage_wheel_bundle(project_root: Path, staging_dir: Path) -> list[Path]:
    inclusion_map = load_bundle_inclusion_map(project_root)
    for root in RESOURCE_ROOTS:
        resource_destination = staging_dir.joinpath(*root.parts)
        if resource_destination.exists():
            shutil.rmtree(resource_destination)

    staged_files: list[Path] = []
    for source, relative_destination in inclusion_map.items():
        destination = staging_dir.joinpath(*relative_destination.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.is_file():
            shutil.copy2(source, destination)
            staged_files.append(destination)
        elif source.is_dir():
            shutil.copytree(source, destination, ignore=_ignore_directory_entries)
            staged_files.extend(path for path in destination.rglob('*') if path.is_file())
        else:
            raise ValueError(f'Unsupported wheel bundle source: {source}')

    return sorted(staged_files)


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Stage runtime bundle and Redis resources from the Hatch wheel manifest.'
    )
    parser.add_argument(
        '--project-root',
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help='Nexus repository root. Defaults to the root containing this script.',
    )
    parser.add_argument('--staging-dir', type=Path, required=True, help='Root directory of the wheel staging tree.')
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    staging_dir = args.staging_dir.resolve()
    staged_files = stage_wheel_bundle(project_root, staging_dir)
    print(f'Staged {len(staged_files)} wheel resource files into {staging_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
