"""Canonical identity for one live T3-Harbor task."""

from __future__ import annotations

import hashlib
import re
import tomllib

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nexus.extensions.harbor_bridge.compose import compose_identity_paths


TASK_IDENTITY_VERSION = 't3h1'
_IGNORED_NAMES = frozenset({'__pycache__', '.DS_Store'})
_UNSAFE_RUN_ID_CHARS = re.compile(r'[^A-Za-z0-9_-]+')


@dataclass(frozen=True)
class TaskIdentity:
    """One canonical identity and its shared provenance metadata."""

    run_id: str
    content_sha256: str
    task_toml_sha256: str

    def metadata(
        self,
        *,
        task_dir: Path,
        config: dict[str, Any],
        include_source_path: bool = True,
    ) -> dict[str, str]:
        environment = config.get('environment', {}) or {}
        judge = config.get('judge', {}) or {}
        metadata = {
            'format': 't3-harbor',
            'task_identity_version': TASK_IDENTITY_VERSION,
            'task_content_sha256': self.content_sha256,
            'task_toml_sha256': self.task_toml_sha256,
            'image_digest': str(environment.get('image_digest', '')),
            'reward_kind': str(judge.get('reward_kind', '')),
        }
        if include_source_path:
            metadata['source_path'] = str(task_dir)
        return metadata


def build_task_identity(task_dir: Path, task_id: str) -> TaskIdentity:
    content_sha256 = task_content_sha256(task_dir)
    return TaskIdentity(
        run_id=task_run_id(task_id, content_sha256),
        content_sha256=content_sha256,
        task_toml_sha256=task_toml_sha256(task_dir),
    )


def task_content_sha256(task_dir: Path) -> str:
    """Hash canonical task inputs, including native Compose runtime files."""
    root = task_dir.resolve()
    digest = hashlib.sha256()
    digest.update(f'{TASK_IDENTITY_VERSION}\0'.encode())
    digest.update(_required_file_identity(root, root / 'task.toml'))
    digest.update(_required_file_identity(root, root / 'instruction.md'))

    tests_dir = root / 'tests'
    digest.update(b'directory\0tests\0')
    if tests_dir.exists():
        if not tests_dir.is_dir():
            raise NotADirectoryError(f'T3-Harbor tests path is not a directory: {tests_dir}')
        for path in sorted(tests_dir.rglob('*'), key=lambda item: item.relative_to(root).as_posix()):
            relative = path.relative_to(root)
            if _is_ignored(relative):
                continue
            relative_text = relative.as_posix()
            if path.is_symlink():
                digest.update(f'symlink\0{relative_text}\0{path.readlink()}\0'.encode())
            elif path.is_dir():
                digest.update(f'directory\0{relative_text}\0'.encode())
            elif path.is_file():
                digest.update(_file_identity(root, path))
            else:
                raise ValueError(f'Unsupported task identity path type: {path}')

    with (root / 'task.toml').open('rb') as task_file:
        config = tomllib.load(task_file)
    compose_paths = compose_identity_paths(root, config)
    if compose_paths:
        digest.update(b'compose-inputs-v2\0')
        for path in compose_paths:
            _update_path_identity(digest, root, path)
    return digest.hexdigest()


def task_toml_sha256(task_dir: Path) -> str:
    return hashlib.sha256((task_dir / 'task.toml').read_bytes()).hexdigest()


def task_run_id(task_id: str, content_sha256: str) -> str:
    component = _UNSAFE_RUN_ID_CHARS.sub('-', task_id).strip('-') or 'unnamed'
    return f't3-harbor__{component}__{content_sha256[:16]}'


def _required_file_identity(root: Path, path: Path) -> bytes:
    if not path.is_file():
        raise FileNotFoundError(f'Required T3-Harbor task file not found: {path}')
    return _file_identity(root, path)


def _file_identity(root: Path, path: Path) -> bytes:
    relative = path.relative_to(root).as_posix()
    file_digest = hashlib.sha256()
    with path.open('rb') as file:
        while chunk := file.read(1024 * 1024):
            file_digest.update(chunk)
    return f'file\0{relative}\0{file_digest.hexdigest()}\0'.encode()


def _update_path_identity(digest: Any, root: Path, path: Path) -> None:
    """Add one file or complete directory tree to a task identity digest."""
    relative = path.relative_to(root)
    if _is_ignored(relative):
        return
    if path.is_symlink():
        digest.update(f'symlink\0{relative.as_posix()}\0{path.readlink()}\0'.encode())
        return
    if path.is_file():
        digest.update(_compose_file_identity(root, path))
        return
    if not path.is_dir():
        raise ValueError(f'Unsupported Compose identity path type: {path}')

    digest.update(f'directory\0{relative.as_posix()}\0{path.stat().st_mode & 0o7777:o}\0'.encode())
    for child in sorted(path.rglob('*'), key=lambda item: item.relative_to(root).as_posix()):
        child_relative = child.relative_to(root)
        if _is_ignored(child_relative):
            continue
        if child.is_symlink():
            digest.update(f'symlink\0{child_relative.as_posix()}\0{child.readlink()}\0'.encode())
        elif child.is_dir():
            digest.update(f'directory\0{child_relative.as_posix()}\0{child.stat().st_mode & 0o7777:o}\0'.encode())
        elif child.is_file():
            digest.update(_compose_file_identity(root, child))
        else:
            raise ValueError(f'Unsupported Compose identity path type: {child}')


def _compose_file_identity(root: Path, path: Path) -> bytes:
    return f'mode\0{path.relative_to(root).as_posix()}\0{path.stat().st_mode & 0o7777:o}\0'.encode() + _file_identity(
        root, path
    )


def _is_ignored(relative: Path) -> bool:
    return any(part in _IGNORED_NAMES for part in relative.parts) or relative.suffix == '.pyc'


__all__ = [
    'TASK_IDENTITY_VERSION',
    'TaskIdentity',
    'build_task_identity',
    'task_content_sha256',
    'task_run_id',
    'task_toml_sha256',
]
