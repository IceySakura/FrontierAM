from __future__ import annotations

import json

import pytest

from sqlalchemy.ext.asyncio import async_sessionmaker

from nexus.distributed.artifacts.models import ArtifactRef
from nexus.distributed.batch.control import BatchControlService, _materialize_generator_subprocess
from nexus.distributed.batch.materialize import materialize_items
from nexus.distributed.batch.task_list import (
    BATCH_ITEM_METADATA_BYTES_ENV,
    canonical_batch_item_metadata,
    canonical_batch_metadata,
)
from nexus.distributed.config import DistributedConfig
from nexus.distributed.config.views import instance_manager_config_view
from nexus.distributed.errors import DomainValidationError
from nexus.distributed.repositories.batch import MaterializeChunkSnapshot, read_representative_run_input
from nexus.runner.batch import BatchItem
from nexus.runner.single import RunInput


def _metadata(size_bytes: int) -> dict[str, str]:
    # Compact JSON adds eight bytes around this value.
    return {'x': 'a' * (size_bytes - 8)}


def test_local_metadata_default_accepts_256_kib_boundary(monkeypatch):
    monkeypatch.delenv(BATCH_ITEM_METADATA_BYTES_ENV, raising=False)
    assert len(canonical_batch_item_metadata(_metadata(256 * 1024))) == 256 * 1024
    with pytest.raises(ValueError, match='batch item metadata exceeds size limit'):
        canonical_batch_item_metadata(_metadata(256 * 1024 + 1))


@pytest.mark.parametrize('limit', [64 * 1024, 512 * 1024])
def test_local_metadata_limit_can_be_lowered_or_raised(monkeypatch, tmp_path, limit):
    monkeypatch.setenv(BATCH_ITEM_METADATA_BYTES_ENV, str(limit))
    item = BatchItem(config=RunInput(task_name='echo'), metadata=_metadata(limit), run_id='item-1')
    output = tmp_path / 'tasks.jsonl'
    materialize_items([item], output, task_list_version=2)
    assert json.loads(output.read_text())['metadata'] == item.metadata
    item.metadata = _metadata(limit + 1)
    with pytest.raises(ValueError, match='batch item metadata exceeds size limit'):
        materialize_items([item], output, task_list_version=2)


@pytest.mark.parametrize('value', ['0', '-1', 'abc', '1.5'])
def test_local_metadata_limit_rejects_invalid_environment(monkeypatch, value):
    monkeypatch.setenv(BATCH_ITEM_METADATA_BYTES_ENV, value)
    with pytest.raises(ValueError, match=f'{BATCH_ITEM_METADATA_BYTES_ENV} must be a positive integer'):
        canonical_batch_item_metadata({})


def test_batch_metadata_retains_64_kib_limit(monkeypatch):
    monkeypatch.setenv(BATCH_ITEM_METADATA_BYTES_ENV, str(512 * 1024))
    with pytest.raises(ValueError, match='batch metadata exceeds size limit'):
        canonical_batch_metadata(_metadata(64 * 1024 + 1))


def test_server_metadata_limit_is_projected_and_positive():
    assert DistributedConfig().batch_item_metadata_max_size_bytes == 128 * 1024
    config = DistributedConfig(batch_item_metadata_max_size_bytes=256 * 1024)
    assert instance_manager_config_view(config).batch_item_metadata_max_size_bytes == 256 * 1024
    with pytest.raises(ValueError, match='batch_item_metadata_max_size_bytes'):
        DistributedConfig(batch_item_metadata_max_size_bytes=0)


@pytest.mark.parametrize(('server_limit', 'accepted'), [(128 * 1024, False), (256 * 1024, True)])
async def test_server_materialization_uses_config_not_local_environment(monkeypatch, tmp_path, server_limit, accepted):
    monkeypatch.setenv(BATCH_ITEM_METADATA_BYTES_ENV, str(512 * 1024))
    item = BatchItem(config=RunInput(task_name='echo'), metadata=_metadata(200 * 1024), run_id='item-1')
    path = tmp_path / 'tasks.jsonl'
    generated = materialize_items([item], path, task_list_version=2)
    config = DistributedConfig(
        batch_item_metadata_max_size_bytes=server_limit,
        batch_control_spool_dir=str(tmp_path / 'spool'),
    )
    control = BatchControlService(session_factory=async_sessionmaker(), config=instance_manager_config_view(config))
    snapshot = MaterializeChunkSnapshot(
        generation_epoch=0,
        task_list_ref=ArtifactRef(
            artifact_id='art-metadata-test', kind='task_list', sha256=generated.sha256, size_bytes=generated.size_bytes
        ),
        record_version=2,
        cursor_bytes=0,
        materialized_count=0,
        total_items=1,
        generation_complete=False,
    )

    async def read_chunk():
        return await control._prepare_materialize_chunk(
            'exe-metadata-test',
            snapshot=snapshot,
            task_list_path=path,
            data_list_path=None,
            data_list_index=None,
            chunk_size=10,
            max_items=10,
            max_line_bytes=1024 * 1024,
            max_chunk_bytes=1024 * 1024,
            max_data_frame_bytes=1024 * 1024,
            max_task_record_bytes=1024 * 1024,
        )

    try:
        if accepted:
            result = await read_chunk()
            assert result.records[0][0].metadata == item.metadata
            assert (
                read_representative_run_input(path, max_line_bytes=1024 * 1024, max_metadata_bytes=server_limit).item_id
                == 'item-1'
            )
        else:
            with pytest.raises(DomainValidationError, match='invalid_task_list_metadata'):
                await read_chunk()
            with pytest.raises(DomainValidationError, match='invalid_task_list_metadata'):
                read_representative_run_input(path, max_line_bytes=1024 * 1024)
    finally:
        await control.close()


@pytest.mark.parametrize(('server_limit', 'accepted'), [(128 * 1024, False), (256 * 1024, True)])
async def test_remote_generator_overrides_inherited_local_limit(monkeypatch, tmp_path, server_limit, accepted):
    monkeypatch.setenv(BATCH_ITEM_METADATA_BYTES_ENV, str(1024 * 1024))
    generator = tmp_path / 'generator.py'
    generator.write_text(
        'from nexus.runner.batch import BatchItem\n'
        'from nexus.runner.single import RunInput\n'
        'def generate_configs():\n'
        '    yield BatchItem(config=RunInput(task_name="echo"), metadata={"x": "a" * 200000})\n'
    )
    kwargs = {
        'max_items': 10,
        'max_bytes': 1024 * 1024,
        'timeout': 20,
        'task_list_version': 2,
        'max_metadata_bytes': server_limit,
    }
    if accepted:
        result = await _materialize_generator_subprocess(generator, tmp_path / 'remote.jsonl', **kwargs)
        assert result.total_items == 1
    else:
        with pytest.raises(RuntimeError, match='batch item metadata exceeds size limit'):
            await _materialize_generator_subprocess(generator, tmp_path / 'remote.jsonl', **kwargs)
