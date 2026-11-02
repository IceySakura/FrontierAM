"""Host-side JSON artifact helpers shared by legacy tasks and feature runners."""

from __future__ import annotations

import json

from collections.abc import Sequence
from pathlib import Path
from typing import Any


def write_json_file(payload: dict | list, path: Path) -> None:
    """Write *payload* as pretty UTF-8 JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def write_jsonl(rows: Sequence[dict], path: Path) -> None:
    """Write one JSON object per line in UTF-8."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + '\n')


def ordered_rows(rows: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    """Return index-keyed rows ordered by their integer join key."""
    return [rows[index] for index in sorted(rows)]


__all__ = ['ordered_rows', 'write_json_file', 'write_jsonl']
