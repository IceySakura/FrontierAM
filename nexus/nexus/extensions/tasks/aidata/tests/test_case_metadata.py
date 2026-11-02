from __future__ import annotations

from pathlib import Path

from nexus.extensions.tasks.aidata import aidata
from nexus.extensions.tasks.aidata.case.spec import _parse_case_metadata
from nexus.extensions.tasks.aidata.generator import _build_metadata


def test_case_metadata_does_not_require_owner() -> None:
    metadata = _parse_case_metadata({'created_at': '2026-07-17'}, Path('task.yaml'))

    assert metadata.model_dump() == {
        'authors': [],
        'created_at': '2026-07-17',
        'tags': [],
    }


def test_cli_metadata_validation_does_not_require_owner() -> None:
    assert aidata._validate_metadata({'created_at': '2026-07-17'}, 'task.yaml') == []


def test_legacy_owner_is_not_emitted() -> None:
    raw = {
        'owner': 'yuanjuncai',
        'created_at': '2026-07-17',
        'tags': ['demo'],
    }

    parsed = _parse_case_metadata(raw, Path('task.yaml'))
    batch_metadata = _build_metadata('example', {'task_id': 'example', 'metadata': raw})

    assert 'owner' not in parsed.model_dump()
    assert 'owner' not in batch_metadata
