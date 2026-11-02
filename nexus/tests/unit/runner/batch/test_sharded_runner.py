from __future__ import annotations

import sqlite3

from pathlib import Path

import pytest

from nexus.cli.commands.export import export_sharded
from nexus.runner.batch import runner as batch_runner
from nexus.runner.batch.batch_config import BatchConfig
from nexus.runner.batch.database import BatchDatabase


def _write_generator(path: Path, count: int) -> None:
    path.write_text(
        f"""
from nexus.runner.batch.batch_config import BatchItem
from nexus.runner.single import RunInput


def generate_configs():
    for index in range({count}):
        yield BatchItem(
            config=RunInput(task_name='noop', task_config={{'index': index}}),
            run_id=f'run-{{index}}',
            metadata={{'index': index}},
        )
"""
    )


def _count_runs(db_path: Path) -> int:
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute('SELECT COUNT(*) FROM runs').fetchone()[0]
    finally:
        conn.close()


async def _get_batch_info(db_path: Path, key: str):
    async with BatchDatabase(db_path) as db:
        return await db.get_batch_info(key)


async def _get_shard_configs(db_path: Path) -> list[dict]:
    shard_root = batch_runner._shard_root_path(db_path)
    shards = await _get_batch_info(db_path, 'shards')
    configs = []
    for shard in shards:
        async with BatchDatabase(shard_root / shard['database']) as db:
            configs.append(await db.get_batch_info('config'))
    return configs


@pytest.mark.asyncio
async def test_run_batch_with_worker_processes_creates_shards_and_aggregates(tmp_path, monkeypatch):
    generator_path = tmp_path / 'generator.py'
    db_path = tmp_path / 'batch.db'
    _write_generator(generator_path, 5)

    async def fake_run_worker_processes(specs):
        for spec in specs:
            async with BatchDatabase(spec.db_path) as db:
                for run in await db.get_all_runs():
                    await db.update_run_status(run['run_id'], 'success', start_time='start', end_time='end')

    monkeypatch.setattr(batch_runner, '_run_worker_processes', fake_run_worker_processes)

    summary = await batch_runner.run_batch(
        generator_path=generator_path,
        db_path=db_path,
        batch_config=BatchConfig(
            max_concurrent=5,
            worker_processes=3,
            enable_trace_db=False,
            enable_log_capture=False,
        ),
    )

    assert summary.total == 5
    assert summary.success == 5
    assert summary.failed == 0

    shard_root = batch_runner._shard_root_path(db_path)
    assert await _get_batch_info(db_path, 'sharded_batch') is True
    shards = await _get_batch_info(db_path, 'shards')
    assert [shard['max_concurrent'] for shard in shards] == [2, 2, 1]
    assert sorted(_count_runs(shard_root / shard['database']) for shard in shards) == [1, 2, 2]
    assert not list(shard_root.glob('*.json'))

    jsonl_path = tmp_path / 'results.jsonl'
    assert await export_sharded(db_path, 'jsonl', str(jsonl_path)) == 0
    assert len(jsonl_path.read_text().splitlines()) == 5


@pytest.mark.asyncio
async def test_run_batch_with_worker_processes_reuses_existing_sharded_database(tmp_path, monkeypatch):
    generator_path = tmp_path / 'generator.py'
    db_path = tmp_path / 'batch.db'
    _write_generator(generator_path, 4)

    async def fake_run_worker_processes(specs):
        for spec in specs:
            async with BatchDatabase(spec.db_path) as db:
                for run in await db.get_all_runs():
                    await db.update_run_status(run['run_id'], 'success', start_time='start', end_time='end')

    monkeypatch.setattr(batch_runner, '_run_worker_processes', fake_run_worker_processes)

    await batch_runner.run_batch(
        generator_path=generator_path,
        db_path=db_path,
        batch_config=BatchConfig(max_concurrent=4, worker_processes=2),
    )

    summary = await batch_runner.run_batch(
        generator_path=None,
        db_path=db_path,
        batch_config=BatchConfig(max_concurrent=6, worker_processes=2),
    )

    assert summary.total == 4
    assert summary.success == 4

    worker_configs = await _get_shard_configs(db_path)
    assert [config['max_concurrent'] for config in worker_configs] == [3, 3]
    assert [config['worker_processes'] for config in worker_configs] == [1, 1]


@pytest.mark.asyncio
async def test_resume_existing_shards_requires_at_least_one_concurrency_per_shard(tmp_path, monkeypatch):
    generator_path = tmp_path / 'generator.py'
    db_path = tmp_path / 'batch.db'
    _write_generator(generator_path, 4)

    async def fake_run_worker_processes(_specs):
        return None

    monkeypatch.setattr(batch_runner, '_run_worker_processes', fake_run_worker_processes)

    await batch_runner.run_batch(
        generator_path=generator_path,
        db_path=db_path,
        batch_config=BatchConfig(max_concurrent=4, worker_processes=4),
    )

    with pytest.raises(ValueError, match='--max-concurrent must be at least 4'):
        await batch_runner.run_batch(
            generator_path=None,
            db_path=db_path,
            batch_config=BatchConfig(max_concurrent=2, worker_processes=4),
        )


@pytest.mark.asyncio
async def test_single_worker_run_rejects_existing_sharded_batch(tmp_path, monkeypatch):
    generator_path = tmp_path / 'generator.py'
    db_path = tmp_path / 'batch.db'
    _write_generator(generator_path, 2)

    async def fake_run_worker_processes(_specs):
        return None

    monkeypatch.setattr(batch_runner, '_run_worker_processes', fake_run_worker_processes)

    await batch_runner.run_batch(
        generator_path=generator_path,
        db_path=db_path,
        batch_config=BatchConfig(max_concurrent=2, worker_processes=2),
    )

    with pytest.raises(ValueError, match='Existing sharded batch found'):
        await batch_runner.run_batch(
            generator_path=generator_path,
            db_path=db_path,
            batch_config=BatchConfig(max_concurrent=1, worker_processes=1),
        )
