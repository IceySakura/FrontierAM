"""Tests for the export memory fixes (MR-1/MR-2/MR-3).

Covers: streaming local JSONL write (byte-identical with json.dumps), lone
surrogate handling, atomic local export (symlinks, permissions, aborted
incremental batches), iter_runs streaming/equivalence/peak memory, and the
cloud integrity validation (sha256 + line-head/tail comparison + deferred
semantic errors + index cap fallback).
"""

import asyncio
import contextlib
import json
import os
import stat
import tracemalloc

import pytest

from nexus.cli.commands.export import AtomicTextOutput, _atomic_text_output, export_run_to_full, write_run_to_jsonl
from nexus.distributed.export import writer as writer_module
from nexus.distributed.export.writer import AtomicJsonlOutput, ExportError, _has_artifacts
from nexus.runner.batch import runner as batch_runner
from nexus.runner.batch.batch_config import BatchConfig, BatchItem
from nexus.runner.batch.database import BatchDatabase
from nexus.runner.single import ErrorInfo, RunInput, RunOutput


def _record(run_id: str = 'item-001', **overrides):
    record = {
        'run_id': run_id,
        'status': 'success',
        'start_time': None,
        'end_time': None,
        'retry_count': 0,
        'error': None,
        'task_output': {'value': 1},
        'agent_output': None,
        'processor_results': {},
        'metadata': {},
    }
    record.update(overrides)
    return record


# ---------------------------------------------------------------- MR-1 (L0)


def test_write_run_to_jsonl_byte_identical_with_json_dumps(tmp_path):
    run = _record()
    run['task_output'] = {'value': '你好\U0001f600', 'nested': {'list': [1, 2.5, None]}}

    target = tmp_path / 'new.jsonl'
    with _atomic_text_output(target) as f:
        write_run_to_jsonl(run, f)

    legacy = tmp_path / 'legacy.jsonl'
    record = {
        'run_id': run['run_id'],
        'status': run['status'],
        'start_time': None,
        'end_time': None,
        'retry_count': 0,
        'error': None,
        'task_output': run['task_output'],
        'agent_output': None,
        'processor_results': {},
        'metadata': {},
    }
    legacy.write_text(json.dumps(record, ensure_ascii=False) + '\n', encoding='utf-8')

    assert target.read_bytes() == legacy.read_bytes()


def test_write_run_to_jsonl_lone_surrogate_round_trips(tmp_path):
    run = _record(run_id='surrogate')
    run['task_output'] = {'content': 'ok \ud83d tail'}

    target = tmp_path / 'out.jsonl'
    with _atomic_text_output(target) as f:
        write_run_to_jsonl(run, f)

    loaded = json.loads(target.read_text(encoding='utf-8'))
    assert loaded['task_output']['content'] == 'ok \ud83d tail'


def test_atomic_text_output_keeps_partial_on_error(tmp_path):
    target = tmp_path / 'out.jsonl'

    def boom():
        with _atomic_text_output(target) as f:
            f.write('{"partial": true}\n')
            raise RuntimeError('boom')

    with pytest.raises(RuntimeError, match='boom'):
        boom()
    assert not target.exists()
    assert target.with_name(target.name + '.partial').read_text() == '{"partial": true}\n'


def test_atomic_text_output_replaces_target(tmp_path):
    target = tmp_path / 'out.jsonl'
    with _atomic_text_output(target) as f:
        f.write('{"ok": 1}\n')
    assert target.read_text() == '{"ok": 1}\n'
    assert not target.with_name(target.name + '.partial').exists()


def test_atomic_text_output_follows_symlink_and_keeps_mode(tmp_path):
    real = tmp_path / 'shared' / 'results.jsonl'
    real.parent.mkdir()
    real.write_text('old\n')
    os.chmod(real, 0o640)
    link = tmp_path / 'results.jsonl'
    link.symlink_to(real)

    with _atomic_text_output(link) as f:
        f.write('new\n')

    assert link.is_symlink()
    assert real.read_text() == 'new\n'
    assert stat.S_IMODE(real.stat().st_mode) == 0o640
    assert not (real.parent / 'results.jsonl.partial').exists()


def test_atomic_text_output_rejects_directory_target(tmp_path):
    with pytest.raises(IsADirectoryError):
        AtomicTextOutput(tmp_path)


def test_atomic_text_output_discard_existing_removes_stale_target(tmp_path):
    target = tmp_path / 'results.jsonl'
    target.write_text('previous batch\n')

    output = AtomicTextOutput(target, discard_existing=True)
    output.file.write('half\n')
    output.abort()

    assert not target.exists()
    assert output.partial.read_text() == 'half\n'


def test_export_run_to_full_lone_surrogate(tmp_path):
    run = _record(run_id='surrogate', task_output={'content': 'ok \ud83d tail'})
    run['config'] = {'task_name': 'echo', 'note': 'cfg \ud83d'}

    export_run_to_full(run, tmp_path)

    loaded = json.loads((tmp_path / 'runs' / 'surrogate' / 'task_output.json').read_text(encoding='utf-8'))
    assert loaded == {'content': 'ok \ud83d tail'}
    assert (tmp_path / 'configs' / 'surrogate.yaml').exists()


def _success_items(count: int) -> list[BatchItem]:
    return [BatchItem(config=RunInput(task_name='unused'), run_id=f'r{i}') for i in range(count)]


async def test_incremental_export_publishes_on_completion(tmp_path, monkeypatch):
    async def fake_run_single(run_input):
        return RunOutput(success=True)

    monkeypatch.setattr(batch_runner, 'run_single', fake_run_single)
    target = tmp_path / 'results.jsonl'
    target.write_text('previous batch\n')
    os.chmod(target, 0o640)

    await batch_runner.run_batch(
        generator_path=None,
        db_path=':memory:',
        batch_config=BatchConfig(enable_trace_db=False, enable_log_capture=False),
        batch_items=_success_items(2),
        export_format='jsonl',
        export_path=target,
    )

    rows = [json.loads(line) for line in target.read_text(encoding='utf-8').splitlines()]
    assert sorted(row['run_id'] for row in rows) == ['r0', 'r1']
    assert stat.S_IMODE(target.stat().st_mode) == 0o640
    assert not target.with_name('results.jsonl.partial').exists()


async def test_incremental_export_abort_leaves_no_stale_target(tmp_path, monkeypatch):
    async def failing_run_single(run_input):
        return RunOutput(success=False, error=ErrorInfo(type='RuntimeError', message='boom'))

    monkeypatch.setattr(batch_runner, 'run_single', failing_run_single)
    target = tmp_path / 'results.jsonl'
    target.write_text('previous batch\n')

    with pytest.raises(RuntimeError, match='boom'):
        await batch_runner.run_batch(
            generator_path=None,
            db_path=':memory:',
            batch_config=BatchConfig(fail_fast=True, retry_failed=0, enable_trace_db=False, enable_log_capture=False),
            batch_items=_success_items(1),
            export_format='jsonl',
            export_path=target,
        )

    # The previous batch's complete-looking file must not survive an aborted run.
    assert not target.exists()
    partial_rows = target.with_name('results.jsonl.partial').read_text(encoding='utf-8').splitlines()
    assert [json.loads(line)['status'] for line in partial_rows] == ['failed']


# ---------------------------------------------------------------- MR-2 (iter_runs)


async def test_iter_runs_matches_get_all_runs(tmp_path):
    db_path = tmp_path / 'batch.db'
    async with BatchDatabase(db_path) as db:
        for i in range(5):
            await db.insert_run(f'r{i}', {'task_name': 'echo', 'task_config': {'i': i}}, {'case': i})
            await db.update_run_status(
                f'r{i}',
                'success',
                task_output={'score': i / 10},
                agent_output={'trajectory': [i]},
                processor_results={'chat': {'messages': [i]}},
            )

        streamed = [run async for run in db.iter_runs()]
        listed = await db.get_all_runs()

    assert streamed == listed
    assert [r['run_id'] for r in streamed] == [f'r{i}' for i in range(5)]


async def test_iter_runs_fields_subset_skips_large_columns(tmp_path):
    db_path = tmp_path / 'batch.db'
    async with BatchDatabase(db_path) as db:
        await db.insert_run('r0', {'task_name': 'echo', 'task_config': {}}, {})
        await db.update_run_status('r0', 'success', task_output={'score': 1})

        rows = [run async for run in db.iter_runs(fields=('run_id', 'status', 'retry_count', 'error'))]

    assert rows == [{'run_id': 'r0', 'status': 'success', 'retry_count': 0, 'error': None}]


async def test_iter_runs_rejects_unknown_column(tmp_path):
    db_path = tmp_path / 'batch.db'
    async with BatchDatabase(db_path) as db:
        with pytest.raises(ValueError, match='unknown run columns'):
            [run async for run in db.iter_runs(fields=('run_id', 'password_hash'))]  # noqa: S106


async def test_iter_runs_status_filter(tmp_path):
    db_path = tmp_path / 'batch.db'
    async with BatchDatabase(db_path) as db:
        await db.insert_run('ok', {'task_name': 'echo', 'task_config': {}}, {})
        await db.update_run_status('ok', 'success')
        await db.insert_run('bad', {'task_name': 'echo', 'task_config': {}}, {})
        await db.update_run_status('bad', 'failed')

        failed = [run['run_id'] async for run in db.iter_runs(status='failed')]

    assert failed == ['bad']


async def _peak_while_streaming(db: BatchDatabase, **kwargs) -> int:
    tracemalloc.start()
    try:
        async for run in db.iter_runs(**kwargs):
            del run
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


async def test_iter_runs_payload_columns_fetch_one_row_at_a_time(tmp_path):
    # aiosqlite prefetches 64 raw rows per thread hop by default; with large
    # payload columns that makes the peak ~64 rows instead of one.
    row_bytes = 1_000_000
    async with BatchDatabase(tmp_path / 'batch.db') as db:
        for i in range(32):
            await db.insert_run(f'r{i:02d}', {'task_name': 'echo', 'task_config': {}}, {})
            await db.update_run_status(f'r{i:02d}', 'success', agent_output={'text': 'x' * row_bytes})

        streamed_peak = await _peak_while_streaming(db)
        prefetched_peak = await _peak_while_streaming(db, prefetch=64)

    assert streamed_peak < 8 * row_bytes
    assert prefetched_peak > 24 * row_bytes


async def test_iter_runs_rejects_invalid_prefetch(tmp_path):
    async with BatchDatabase(tmp_path / 'batch.db') as db:
        with pytest.raises(ValueError, match='prefetch'):
            [run async for run in db.iter_runs(prefetch=0)]


# ---------------------------------------------------------------- MR-3 (cloud)


async def _write_three(tmp_path):
    writer = AtomicJsonlOutput(tmp_path / 'results.jsonl', force=False)
    rows = [_record('r0'), _record('r1', status='failed', error={'type': 'E'}), _record('r2')]
    await writer.open()
    await writer.write_records(rows)
    await writer.close_and_validate(expected_count=3, require_sorted=False)
    return writer, rows


async def test_default_validation_matches_strict(tmp_path):
    writer, _ = await _write_three(tmp_path)
    expected = writer.partial.read_bytes()
    await writer.publish()
    assert writer.output.read_bytes() == expected


async def test_semantic_error_deferred_to_close(tmp_path):
    writer = AtomicJsonlOutput(tmp_path / 'results.jsonl', force=False)
    await writer.open()
    # Must not raise at write time (established close-time contract) ...
    await writer.write_records([{'run_id': 'incomplete'}])
    with pytest.raises(Exception, match='unexpected JSONL fields'):
        # ... must raise here, keeping the same message as the strict path.
        await writer.close_and_validate(expected_count=1, require_sorted=True)
    assert not writer.output.exists()
    await writer.abort()


async def test_strict_and_default_agree_on_message(tmp_path):
    for kwargs in ({}, {'strict': True}):
        writer = AtomicJsonlOutput(tmp_path / 'results.jsonl', force=True)
        await writer.open()
        await writer.write_records([{'run_id': 'incomplete'}])
        with pytest.raises(Exception, match='unexpected JSONL fields'):
            await writer.close_and_validate(expected_count=1, require_sorted=True, **kwargs)
        await writer.abort()


@pytest.mark.parametrize(
    'corrupt',
    [
        'truncate-middle',
        'flip-byte',
        'inject-garbage',
        'merge-lines',
        'swap-lines',
        'duplicate-line',
        'append-line',
        'truncate-tail',
    ],
)
async def test_integrity_detects_corruption(tmp_path, corrupt):
    writer = AtomicJsonlOutput(tmp_path / 'results.jsonl', force=False)
    rows = [_record(f'r{i}') for i in range(3)]
    await writer.open()
    await writer.write_records(rows)
    # Simulate an external actor corrupting the finished partial: flush first so
    # every written byte is visible on disk, then tamper.
    writer._stream.flush()
    raw = writer.partial.read_bytes()
    lines = raw.splitlines(keepends=True)
    assert len(lines) == 3
    if corrupt == 'truncate-middle':
        raw = lines[0] + lines[1][: len(lines[1]) // 2] + lines[2]
    elif corrupt == 'flip-byte':
        raw = raw.replace(b'"r1"', b'"r@"', 1)
    elif corrupt == 'inject-garbage':
        raw = lines[0][:-2] + b', g}\n' + lines[1] + lines[2]
    elif corrupt == 'merge-lines':
        raw = lines[0][:-1] + lines[1] + lines[2]
    elif corrupt == 'swap-lines':
        raw = lines[0] + lines[2] + lines[1]
    elif corrupt == 'duplicate-line':
        raw = lines[0] + lines[0] + lines[2]
    elif corrupt == 'append-line':
        raw += lines[0]
    elif corrupt == 'truncate-tail':
        raw = raw[:-5]
    writer.partial.write_bytes(raw)

    with pytest.raises(ExportError):
        await writer.close_and_validate(expected_count=3, require_sorted=False)


async def test_integrity_passes_run_id_longer_than_any_fixed_head_window(tmp_path):
    writer = AtomicJsonlOutput(tmp_path / 'results.jsonl', force=False)
    long_id = 'run-' + '长' * 5000  # ~15 KB once encoded
    await writer.open()
    await writer.write_records([_record(long_id)])
    await writer.close_and_validate(expected_count=1, require_sorted=False)
    await writer.publish()
    assert json.loads(writer.output.read_text(encoding='utf-8'))['run_id'] == long_id


async def test_index_cap_falls_back_to_parse_per_line(tmp_path, monkeypatch):
    monkeypatch.setattr(writer_module, '_MAX_INDEX_ENTRIES', 2)
    writer = AtomicJsonlOutput(tmp_path / 'results.jsonl', force=False)
    rows = [_record('r0'), _record('r1', status='failed', task_output=None, processor_results=None), _record('r2')]
    await writer.open()
    await writer.write_records(rows)

    # The index is released while writing, not kept until close.
    assert not writer.index_available
    assert writer._index == []

    await writer.close_and_validate(expected_count=3, require_sorted=False)
    with pytest.raises(ExportError, match='index is unavailable'):
        await writer.filter_records_by_index(lambda entry: True)
    assert await writer.filter_records(lambda row: row['status'] == 'success') == 2


async def test_index_cap_still_reports_semantic_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(writer_module, '_MAX_INDEX_ENTRIES', 0)
    writer = AtomicJsonlOutput(tmp_path / 'results.jsonl', force=False)
    await writer.open()
    await writer.write_records([{'run_id': 'incomplete'}])
    with pytest.raises(ExportError, match='unexpected JSONL fields'):
        await writer.close_and_validate(expected_count=1, require_sorted=False)
    await writer.abort()


async def test_integrity_passes_non_ascii_run_id(tmp_path):
    writer = AtomicJsonlOutput(tmp_path / 'results.jsonl', force=False)
    rows = [_record('中文-run'), _record('emoji\U0001f600'), _record('quote"inside')]
    await writer.open()
    await writer.write_records(rows)
    # Must not raise: the line-head comparison must mirror iter_json_bytes escaping.
    await writer.close_and_validate(expected_count=3, require_sorted=False)
    await writer.publish()
    assert [json.loads(line)['run_id'] for line in writer.output.read_text(encoding='utf-8').splitlines()] == [
        '中文-run',
        'emoji\U0001f600',
        'quote"inside',
    ]


async def test_lone_surrogate_cloud_writer_round_trips(tmp_path):
    writer = AtomicJsonlOutput(tmp_path / 'results.jsonl', force=False)
    rows = [_record('r0', task_output={'content': 'ok \ud83d tail'})]
    await writer.open()
    await writer.write_records(rows)
    await writer.close_and_validate(expected_count=1, require_sorted=False)
    await writer.publish()
    loaded = json.loads(writer.output.read_text(encoding='utf-8'))
    assert loaded['task_output']['content'] == 'ok \ud83d tail'


async def test_filter_by_index_preserves_bytes_and_invalidates(tmp_path):
    writer, rows = await _write_three(tmp_path)
    expected = lines = writer.partial.read_bytes().splitlines(keepends=True)
    _ = lines

    written = await writer.filter_records_by_index(lambda entry: entry.status == 'success')
    assert written == 2
    assert writer.partial.read_bytes() == expected[0] + expected[2]
    assert [entry.run_id for entry in writer._index] == ['r0', 'r2']

    # A second by-index filter runs on the compacted layout written by the first.
    assert await writer.filter_records_by_index(lambda entry: entry.run_id == 'r2') == 1
    assert writer.partial.read_bytes() == expected[2]

    # The legacy path parses and rewrites the partial, invalidating the write-time
    # bookkeeping; the index-based filter must then refuse to run.
    await writer.filter_records(lambda row: row['status'] == 'success')
    with pytest.raises(ExportError, match='index is unavailable'):
        await writer.filter_records_by_index(lambda entry: True)


async def test_legacy_filter_records_still_works(tmp_path):
    writer, rows = await _write_three(tmp_path)
    expected = writer.partial.read_bytes().splitlines(keepends=True)
    written = await writer.filter_records(lambda row: row['status'] == 'success')
    assert written == 2
    assert writer.partial.read_bytes() == expected[0] + expected[2]


def test_has_artifacts_semantics():
    assert _has_artifacts({'task_output': {'score': 0}})
    assert _has_artifacts({'task_output': False})
    assert not _has_artifacts({'task_output': None})
    assert not _has_artifacts({'task_output': '', 'agent_output': [], 'processor_results': {}})
    assert _has_artifacts({'processor_results': {'chat': {}}})


# ---------------------------------------------------------------- partial 锁语义


async def test_stale_partial_is_reclaimed_without_force(tmp_path):
    """崩溃后残留的 partial（无活跃锁）应被下一次导出自动接管。"""
    stale = tmp_path / 'results.jsonl.partial'
    stale.write_bytes(b'{"run_id": "dead", "status": "failed"}\n' * 3)
    (tmp_path / 'results.jsonl.partial.lock').write_text('{"pid": 123456, "host": "old-box"}')

    writer = AtomicJsonlOutput(tmp_path / 'results.jsonl', force=False)
    await writer.open()  # 不得要求 --force
    await writer.write_records([_record('r0')])
    await writer.close_and_validate(expected_count=1, require_sorted=False)
    await writer.publish()

    assert json.loads(writer.output.read_text().splitlines()[0])['run_id'] == 'r0'
    assert not (tmp_path / 'results.jsonl.partial.lock').exists()


async def test_active_writer_conflict_is_reported(tmp_path):
    """另一个 writer 持锁时，新导出必须失败且错误信息可指导用户。"""
    first = AtomicJsonlOutput(tmp_path / 'results.jsonl', force=False)
    await first.open()
    await first.write_records([_record('r0')])

    second = AtomicJsonlOutput(tmp_path / 'results.jsonl', force=False)
    with pytest.raises(Exception, match='another export is actively writing'):
        await second.open()
    await first.abort()


async def test_force_overrides_stale_lock(tmp_path):
    stale = tmp_path / 'results.jsonl.partial'
    stale.write_bytes(b'garbage')
    (tmp_path / 'results.jsonl.partial.lock').write_text('{"pid": 1}')

    writer = AtomicJsonlOutput(tmp_path / 'results.jsonl', force=True)
    await writer.open()
    await writer.write_records([_record('fresh')])
    await writer.close_and_validate(expected_count=1, require_sorted=False)
    await writer.publish()
    assert 'fresh' in writer.output.read_text()


async def test_write_cancellation_keeps_partial_and_releases_lock(tmp_path):
    """Ctrl-C 语义：取消后保留 partial 供诊断、释放锁；重试自动接管。"""
    writer = AtomicJsonlOutput(tmp_path / 'results.jsonl', force=False)
    await writer.open()
    task = asyncio.create_task(writer.write_records([_record('big')]))
    await asyncio.sleep(0.01)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    await writer.abort()

    assert writer.partial.exists()  # 保留供诊断
    assert not writer.index_available  # 锁已释放，重试可接管

    retry = AtomicJsonlOutput(tmp_path / 'results.jsonl', force=False)
    await retry.open()  # 无需 --force 即可接管残留
    await retry.write_records([_record('fresh')])
    await retry.close_and_validate(expected_count=1, require_sorted=False)
    await retry.publish()
    assert json.loads(writer.output.read_text())['run_id'] == 'fresh'
