"""Follow-up coverage for column-subset scheduling and full-format publication.

Covers:
- `_aggregate_shard_summaries` reads only `run_id` from shard databases;
- `run_batch` never falls back to materializing all rows (`get_all_runs`);
- `export_full` stages into a unique partial directory and keeps the previous
  target untouched when the export fails.
"""

from __future__ import annotations

import inspect
import os
import stat

from typing import Any, cast

import pytest

from nexus.cli.commands import export as cli_export
from nexus.cli.commands.export import AtomicDirectoryOutput, export_full
from nexus.runner.batch import runner as batch_runner
from nexus.runner.batch.runner import BatchShardSpec, _aggregate_shard_summaries


# --------------------------------------------------------------- #7 column reads


class _RecordingDb:
    """Stands in for BatchDatabase, recording iter_runs keyword arguments."""

    def __init__(self, calls: list[dict[str, Any]], rows: list[dict[str, Any]]):
        self.calls = calls
        self._rows = rows

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get_summary(self):
        return {'total': len(self._rows), 'failed': len(self._rows)}

    def iter_runs(self, status=None, fields=None):
        self.calls.append({'status': status, 'fields': tuple(fields) if fields else None})

        async def gen():
            for row in self._rows:
                yield row

        return gen()


async def test_aggregate_shard_summaries_reads_only_run_id(tmp_path, monkeypatch):
    calls: list[dict[str, Any]] = []
    db = _RecordingDb(calls, [{'run_id': 'r1', 'task_output': {'huge': 'x'}}])
    monkeypatch.setattr(batch_runner, 'BatchDatabase', lambda db_path: db)
    spec = BatchShardSpec(index=0, db_path=tmp_path / 'shard.db', max_concurrent=1, run_count=1)

    summary = await _aggregate_shard_summaries([spec], start_time_epoch=0.0)

    assert calls == [{'status': 'failed', 'fields': ('run_id',)}]
    assert summary.failed_runs == ['r1']
    assert summary.failed == 1


def test_run_batch_does_not_materialize_all_runs():
    """Tripwire: scheduling must stream with a column subset, never get_all_runs."""
    source = inspect.getsource(batch_runner.run_batch)
    assert 'get_all_runs' not in source
    assert 'fields=summary_columns' in source


# --------------------------------------------------------------- atomic full export


class _FullExportDb:
    def __init__(self, runs: list[dict[str, Any]]):
        self._runs = runs

    async def get_batch_info(self, key: str):
        return {'kind': key}

    def iter_runs(self):
        async def gen():
            for run in self._runs:
                yield run

        return gen()


def _run(run_id: str) -> dict[str, Any]:
    return {
        'run_id': run_id,
        'status': 'success',
        'config': {'model': 'test'},
        'metadata': {'row': run_id},
        'task_output': {'v': 1},
    }


async def test_export_full_publishes_tree_without_leftovers(tmp_path):
    target = tmp_path / 'export'
    code = await export_full(cast(Any, _FullExportDb([_run('r1'), _run('r2')])), target)

    assert code == 0
    assert (target / 'runs' / 'r1' / 'status.json').exists()
    assert (target / 'runs' / 'r2' / 'status.json').exists()
    assert (target / 'batch_metadata.json').exists()
    assert (target / 'batch_config.json').exists()
    assert not list(tmp_path.glob('export.partial-*'))
    assert not list(tmp_path.glob('export.old-*'))


async def test_export_full_failure_keeps_previous_target_intact(tmp_path, monkeypatch):
    target = tmp_path / 'export'
    (target / 'runs').mkdir(parents=True)
    (target / 'runs' / 'previous.json').write_text('keep me')

    real = cli_export.export_run_to_full

    def failing(run: dict, tree) -> None:
        if run['run_id'] == 'r2':
            raise RuntimeError('disk exploded')
        real(run, tree)

    monkeypatch.setattr(cli_export, 'export_run_to_full', failing)
    with pytest.raises(RuntimeError, match='disk exploded'):
        await export_full(cast(Any, _FullExportDb([_run('r1'), _run('r2')])), target)

    # previous target untouched; partial tree kept for diagnosis (the next run
    # removes it before rebuilding), and no swap debris anywhere
    assert (target / 'runs' / 'previous.json').read_text() == 'keep me'
    assert not (target / 'runs' / 'r1').exists()
    partial_dirs = list(tmp_path.glob('export.partial-*'))
    assert len(partial_dirs) == 1
    assert (partial_dirs[0] / 'runs' / 'r1').exists()
    assert not list(tmp_path.glob('export.old-*'))


async def test_export_full_replaces_previous_target(tmp_path):
    target = tmp_path / 'export'
    (target / 'runs' / 'stale-run').mkdir(parents=True)

    await export_full(cast(Any, _FullExportDb([_run('r1')])), target)

    assert (target / 'runs' / 'r1' / 'status.json').exists()
    assert not (target / 'runs' / 'stale-run').exists()
    assert not list(tmp_path.glob('export.partial-*'))
    assert not list(tmp_path.glob('export.old-*'))


async def test_export_full_preserves_symlink_and_target_mode(tmp_path):
    real_target = tmp_path / 'real-export'
    real_target.mkdir()
    os.chmod(real_target, 0o750)
    link = tmp_path / 'export'
    link.symlink_to(real_target, target_is_directory=True)

    await export_full(cast(Any, _FullExportDb([_run('r1')])), link)

    assert link.is_symlink()
    assert (real_target / 'runs' / 'r1' / 'status.json').exists()
    assert stat.S_IMODE(real_target.stat().st_mode) == 0o750


async def test_export_full_recovers_crash_swap_state(tmp_path):
    target = tmp_path / 'export'
    backup = tmp_path / 'export.old-crashed'
    partial = tmp_path / 'export.partial-crashed'
    (backup / 'runs').mkdir(parents=True)
    (backup / 'runs' / 'previous.json').write_text('keep until replacement')
    (partial / 'stale').mkdir(parents=True)

    await export_full(cast(Any, _FullExportDb([_run('r1')])), target)

    assert (target / 'runs' / 'r1' / 'status.json').exists()
    assert not backup.exists()
    assert not partial.exists()


def test_atomic_full_output_rejects_a_concurrent_writer(tmp_path):
    target = tmp_path / 'export'
    first = AtomicDirectoryOutput(target)
    first.open()
    try:
        with pytest.raises(RuntimeError, match='full export is already running'):
            AtomicDirectoryOutput(target).open()
    finally:
        first.abort()
