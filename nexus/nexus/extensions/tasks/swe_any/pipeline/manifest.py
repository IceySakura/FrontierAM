"""JSON contracts shared by the three feature-level pipeline tasks."""

from __future__ import annotations

import contextlib
import json
import os
import tempfile

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from nexus.extensions.tasks.swe_any.utils import constants
from nexus.extensions.tasks.swe_any.utils.artifacts import write_jsonl


def manifest_run_id(instance_id: str, index: int) -> str:
    """Return the stable batch run id for one manifest row."""
    return f'{instance_id}#{constants.slice_dirname(index)}'


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield JSON objects from *path*, rejecting malformed row shapes."""
    for line_number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f'{path}:{line_number}: expected a JSON object')
        yield row


def read_manifest(path: Path) -> list[dict[str, Any]]:
    """Read and minimally validate a feature manifest."""
    rows = list(iter_jsonl(path))
    seen: set[int] = set()
    for row in rows:
        index = row.get('index')
        heuristic = row.get('heuristic_row')
        if not isinstance(index, int):
            raise ValueError(f'{path}: manifest row missing integer index')
        if index in seen:
            raise ValueError(f'{path}: duplicate feature index {index}')
        if not isinstance(heuristic, dict):
            raise ValueError(f'{path}: feature index {index} missing heuristic_row')
        seen.add(index)
    return sorted(rows, key=lambda row: row['index'])


def write_manifest(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Write manifest rows in stable feature-index order."""
    write_jsonl(sorted(rows, key=lambda row: int(row['index'])), path)


def write_feature_result(path: Path, result: dict[str, Any]) -> None:
    """Atomically write one feature's terminal pipeline result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as fh:
            json.dump(result, fh, indent=2, ensure_ascii=False)
            fh.write('\n')
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


def read_feature_result(path: Path) -> dict[str, Any]:
    """Read one feature result and validate its index/stage container."""
    result = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(result, dict) or not isinstance(result.get('index'), int):
        raise ValueError(f'{path}: invalid feature result')
    if not isinstance(result.get('stages'), dict):
        raise ValueError(f'{path}: feature result missing stages mapping')
    return result


__all__ = [
    'iter_jsonl',
    'manifest_run_id',
    'read_feature_result',
    'read_manifest',
    'write_feature_result',
    'write_manifest',
]
