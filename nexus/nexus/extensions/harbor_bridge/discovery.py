"""Shared discovery for live T3-Harbor task directories."""

from __future__ import annotations

import logging
import os

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


logger = logging.getLogger(__name__)


def discover_task_dirs(
    roots: list[Path],
    *,
    flat: bool = False,
    workers: int = 1,
) -> list[Path]:
    """Discover visible task directories without descending into a task itself."""
    task_dirs: list[Path] = []
    frontier: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        if _has_task_toml(root):
            task_dirs.append(root)
        else:
            frontier.append(root)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        while frontier:
            candidates = [child for children in executor.map(_child_dirs, frontier) for child in children]
            is_task = list(executor.map(_has_task_toml, candidates))
            task_dirs.extend(path for path, present in zip(candidates, is_task, strict=True) if present)
            if flat:
                break
            frontier = [path for path, present in zip(candidates, is_task, strict=True) if not present]

    return sorted(set(task_dirs))


def _child_dirs(parent: Path) -> list[Path]:
    try:
        with os.scandir(parent) as entries:
            return [
                Path(entry.path)
                for entry in entries
                if not entry.name.startswith('.') and entry.is_dir(follow_symlinks=False)
            ]
    except OSError as exc:
        logger.warning('Skipping unreadable T3-Harbor directory %s: %s', parent, exc)
        return []


def _has_task_toml(path: Path) -> bool:
    return (path / 'task.toml').is_file()


__all__ = ['discover_task_dirs']
