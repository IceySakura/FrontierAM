from __future__ import annotations

from pathlib import Path

import pytest

from nexus.dataset.item import TaskFilesSpec
from nexus.dataset.task_files import task_files_from_spec
from nexus.extensions.harbor_bridge.task import _materialize_task_directory


@pytest.mark.asyncio
async def test_materialize_task_directory_downloads_cloud_assignment_zip(tmp_path: Path) -> None:
    source = tmp_path / 'source'
    (source / 'environment').mkdir(parents=True)
    (source / 'tests').mkdir()
    (source / 'task.toml').write_text('[environment]\ndocker_image = "task:v1"\n', encoding='utf-8')
    (source / 'instruction.md').write_text('Solve it.\n', encoding='utf-8')
    (source / 'tests' / 'test.sh').write_text('#!/bin/sh\n', encoding='utf-8')

    from nexus.dataset.zip_index import write_nexus_zip_directory_to_file

    archive = tmp_path / 'files.zip'
    write_nexus_zip_directory_to_file(source, archive)
    files = task_files_from_spec(TaskFilesSpec(type='zip', path=str(archive)))
    assert files is not None

    destination = tmp_path / 'materialized'
    result = await _materialize_task_directory(files, serialized_task_dir=None, destination=destination)

    assert result == destination
    assert (destination / 'task.toml').read_text(encoding='utf-8') == (source / 'task.toml').read_text(encoding='utf-8')
    assert (destination / 'tests' / 'test.sh').is_file()


@pytest.mark.asyncio
async def test_materialize_task_directory_keeps_legacy_live_path(tmp_path: Path) -> None:
    source = tmp_path / 'source'
    source.mkdir()

    result = await _materialize_task_directory(None, serialized_task_dir=str(source), destination=tmp_path / 'unused')

    assert result == source
