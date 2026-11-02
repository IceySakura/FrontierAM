from __future__ import annotations

import asyncio
import json
import os
import threading

from hashlib import sha256
from pathlib import Path
from typing import Any, BinaryIO, cast

import pytest

from nexus.distributed.export_output import JsonlEventWriter
from nexus.distributed.export_types import ExportError, TraceStreamKey


TABLE = 'trajectory_span_events_pr_0123456789abcdef0123456789abcdef'


def _source(*, execution_id: str = 'batch-1', spec_type: str = 'nexus_batch') -> dict[str, object]:
    return {
        'control_url': 'https://control.example.invalid/',
        'execution_id': execution_id,
        'spec_type': spec_type,
        'db_host': 'trace-db.example.invalid',
        'db_port': 3306,
        'db_name': 'trace_example',
        'db_user': 'rotatable-reader',
        'db_password': 'must-not-persist',
        'taihu_token': 'must-not-persist-either',
        'table': TABLE,
        'storage_scope': 'principal_v1',
        'content': 'traces',
        'schema_version': 1,
    }


def _stream(run_id: str = 'child-1', run_span_id: str = 'span-child-1') -> TraceStreamKey:
    return TraceStreamKey('principal_v1', TABLE, run_id, run_span_id)


def _event(
    record_id: int = 1,
    *,
    run_id: str = 'child-1',
    run_span_id: str = 'span-child-1',
    parent_execution_id: str | None = 'batch-1',
) -> dict[str, object]:
    return {
        'id': 9_223_372_036_854_775_000 + record_id,
        'run_id': run_id,
        'run_span_id': run_span_id,
        'parent_execution_id': parent_execution_id,
        'record_id': record_id,
        'event_type': 'update',
        'time_unix_nano': 1_789_084_800_000_000_000 + record_id,
        'trace_id': 'trace-实时',
        'span_id': 'step-span',
        'parent_span_id': None,
        'event_data': '{"attributes":{"text":"实时 ✓","step":42}}',
        'created_at': '2026-09-11T00:00:00.123456',
    }


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding='utf-8'))
    assert isinstance(value, dict)
    return value


async def test_large_record_partial_write_resumes_without_truncating_content(tmp_path, monkeypatch):
    from nexus.distributed.export import writer as writer_module

    output = tmp_path / 'large.jsonl'
    writer = JsonlEventWriter(output, source=_source())
    await writer.open()
    await writer.commit_page(_stream(), expected_cursor=0, rows=[_event(1)])
    committed = writer.committed_bytes
    large = _event(2)
    large['event_data'] = json.dumps({'text': '中' * 300_000}, ensure_ascii=False)
    original_write = writer_module._write_all
    chunks = []

    def interrupted(stream, chunk):
        chunks.append(len(chunk))
        original_write(stream, chunk)
        if sum(chunks) > 128 * 1024:
            raise OSError('injected large-record disk failure')

    monkeypatch.setattr(writer_module, '_write_all', interrupted)
    with pytest.raises(ExportError):
        await writer.commit_page(_stream(), expected_cursor=1, rows=[large])
    assert writer.cursor_for(_stream()) == 1
    assert writer.committed_bytes == committed
    assert max(chunks) <= 64 * 1024
    await writer.close(reason='error')
    monkeypatch.setattr(writer_module, '_write_all', original_write)

    resumed = JsonlEventWriter(output, source=_source(), resume=True)
    await resumed.open()
    assert output.stat().st_size == committed
    assert resumed.cursor_for(_stream()) == 1
    await resumed.commit_page(_stream(), expected_cursor=1, rows=[large, _event(3)])
    await resumed.close(reason='interrupted')
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert rows == [_event(1), large, _event(3)]
    checkpoint = _read_json(resumed.checkpoint)
    assert checkpoint['version'] == 1
    assert checkpoint['committed_sha256'] == sha256(output.read_bytes()).hexdigest()


class _CloseFailingRealFile:
    def __init__(self, stream: BinaryIO) -> None:
        self._stream = stream

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)

    def close(self) -> None:
        self._stream.close()
        raise OSError('injected close failure')


async def test_follow_writes_unicode_and_independent_stream_cursors(tmp_path: Path) -> None:
    output = tmp_path / 'events.jsonl'
    writer = JsonlEventWriter(output, source=_source())
    await writer.open()

    first = [_event(1), _event(7)]
    second = [_event(1, run_id='child-2', run_span_id='span-child-2')]
    await writer.commit_page(_stream(), expected_cursor=0, rows=first)
    await writer.commit_page(_stream('child-2', 'span-child-2'), expected_cursor=0, rows=second)
    checkpoint_before_observation = writer.checkpoint.read_bytes()
    await writer.update_observation({'execution_status': 'running', 'discovery_round': 2})
    assert writer.checkpoint.read_bytes() == checkpoint_before_observation
    await writer.close(reason='interrupted')
    await writer.close(reason='error')

    encoded = output.read_bytes()
    rows = [json.loads(line) for line in encoded.splitlines()]
    assert rows == first + second
    assert '实时 ✓'.encode() in encoded
    assert rows[0]['id'] == 9_223_372_036_854_775_001
    assert writer.cursor_for(_stream()) == 7
    assert writer.cursor_for(_stream('child-2', 'span-child-2')) == 1
    assert writer.committed_records == 3
    assert writer.committed_bytes == len(encoded)
    assert writer.stream_count == 2

    checkpoint = _read_json(writer.checkpoint)
    assert checkpoint['committed_sha256'] == sha256(encoded).hexdigest()
    assert checkpoint['committed_records'] == 3
    assert checkpoint['committed_bytes'] == len(encoded)
    assert checkpoint['streams'] == [
        {'run_id': 'child-1', 'run_span_id': 'span-child-1', 'last_record_id': 7},
        {'run_id': 'child-2', 'run_span_id': 'span-child-2', 'last_record_id': 1},
    ]
    assert checkpoint['source'] == {
        'control_url': 'https://control.example.invalid',
        'execution_id': 'batch-1',
        'spec_type': 'nexus_batch',
        'db_host': 'trace-db.example.invalid',
        'db_port': 3306,
        'db_name': 'trace_example',
        'table': TABLE,
        'storage_scope': 'principal_v1',
        'content': 'traces',
        'schema_version': 1,
    }
    manifest = _read_json(writer.manifest)
    assert manifest == {
        'schema_version': 1,
        'export_id': checkpoint['export_id'],
        'source': checkpoint['source'],
        'mode': 'follow',
        'status': 'interrupted',
        'started_at': checkpoint['started_at'],
        'last_committed_at': checkpoint['last_committed_at'],
        'committed_records': 3,
        'committed_bytes': len(encoded),
        'stream_count': 2,
        'observed': {'execution_status': 'running', 'discovery_round': 2},
    }
    for path in (output, writer.checkpoint, writer.manifest):
        assert path.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize(
    ('stream', 'row_update'),
    [
        (_stream(), {'parent_execution_id': 'other-batch'}),
        (_stream(), {'run_id': 'other-child'}),
        (_stream(), {'run_span_id': 'other-span'}),
        (_stream(), {'record_id': 0}),
        (_stream(), {'record_id': True}),
        (_stream(), {'id': True}),
        (_stream(), {'time_unix_nano': False}),
        (_stream(), {'event_data': '[]'}),
        (_stream(), {'event_data': '{"bad":NaN}'}),
        (_stream(), {'created_at': 1}),
    ],
)
async def test_invalid_page_is_rejected_without_state_change(
    tmp_path: Path, stream: TraceStreamKey, row_update: dict[str, object]
) -> None:
    output = tmp_path / 'events.jsonl'
    writer = JsonlEventWriter(output, source=_source())
    await writer.open()
    before_data = output.read_bytes()
    before_checkpoint = writer.checkpoint.read_bytes()
    row = {**_event(), **row_update}

    with pytest.raises(ExportError):
        await writer.commit_page(stream, expected_cursor=0, rows=[row])

    assert output.read_bytes() == before_data
    assert writer.checkpoint.read_bytes() == before_checkpoint
    assert writer.cursor_for(stream) == 0
    assert writer.committed_records == 0
    await writer.close(reason='error')


async def test_unknown_stream_and_stale_cursor_fail_before_writes(tmp_path: Path) -> None:
    output = tmp_path / 'events.jsonl'
    writer = JsonlEventWriter(output, source=_source())
    await writer.open()
    wrong_source_stream = TraceStreamKey(
        'principal_v1', 'trajectory_span_events_pr_fedcba9876543210fedcba9876543210', 'child-1', 'span-child-1'
    )
    before = output.read_bytes(), writer.checkpoint.read_bytes()

    with pytest.raises(ExportError):
        await writer.commit_page(wrong_source_stream, expected_cursor=0, rows=[_event()])
    with pytest.raises(ExportError):
        await writer.commit_page(_stream(), expected_cursor=1, rows=[_event(2)])

    assert (output.read_bytes(), writer.checkpoint.read_bytes()) == before
    await writer.close(reason='error')


async def test_single_and_live_bind_stream_run_id_to_execution(tmp_path: Path) -> None:
    for spec_type in ('nexus_single', 'nexus_live'):
        output = tmp_path / f'{spec_type}.jsonl'
        writer = JsonlEventWriter(output, source=_source(execution_id='execution-1', spec_type=spec_type))
        await writer.open()
        before = output.read_bytes(), writer.checkpoint.read_bytes()

        with pytest.raises(ExportError):
            await writer.commit_page(_stream(), expected_cursor=0, rows=[_event(parent_execution_id=None)])

        assert (output.read_bytes(), writer.checkpoint.read_bytes()) == before
        await writer.close(reason='error')


async def test_second_writer_cannot_acquire_output_lock(tmp_path: Path) -> None:
    output = tmp_path / 'events.jsonl'
    first = JsonlEventWriter(output, source=_source())
    await first.open()
    second = JsonlEventWriter(output, source=_source(), resume=True)

    with pytest.raises(ExportError, match='locked'):
        await second.open()

    await first.close(reason='interrupted')


async def test_stream_close_failure_still_releases_lock_for_resume(tmp_path: Path) -> None:
    output = tmp_path / 'events.jsonl'
    first = JsonlEventWriter(output, source=_source())
    await first.open()
    assert first._stream is not None
    first._stream = cast(BinaryIO, _CloseFailingRealFile(first._stream))

    with pytest.raises(ExportError, match='close event output'):
        await first.close(reason='interrupted')

    resumed = JsonlEventWriter(output, source=_source(), resume=True)
    try:
        assert first._stream is None
        assert first._lock_descriptor is None
        await resumed.open()
    finally:
        if first._lock_descriptor is not None:
            os.close(first._lock_descriptor)
            first._lock_descriptor = None
    await resumed.close(reason='interrupted')


async def test_resume_with_different_source_leaves_files_intact(tmp_path: Path) -> None:
    output = tmp_path / 'events.jsonl'
    first = JsonlEventWriter(output, source=_source())
    await first.open()
    await first.commit_page(_stream(), expected_cursor=0, rows=[_event()])
    await first.close(reason='interrupted')
    before = output.read_bytes(), first.checkpoint.read_bytes(), first.manifest.read_bytes()

    resumed = JsonlEventWriter(output, source=_source(execution_id='other-batch'), resume=True)
    with pytest.raises(ExportError, match='source'):
        await resumed.open()

    assert (output.read_bytes(), first.checkpoint.read_bytes(), first.manifest.read_bytes()) == before


async def test_resume_ignores_rotated_credentials(tmp_path: Path) -> None:
    output = tmp_path / 'events.jsonl'
    first = JsonlEventWriter(output, source=_source())
    await first.open()
    await first.close(reason='interrupted')
    rotated = _source()
    rotated.update(db_user='new-reader', db_password='new-secret', taihu_token='new-token')

    resumed = JsonlEventWriter(output, source=rotated, resume=True)
    await resumed.open()
    await resumed.close(reason='interrupted')


# --- the backend belongs to the identity ---------------------------------------------
# One principal table can be published by two backends on different schemas, so resuming
# across them would splice both shapes into one file. The identity carries the backend to
# make that impossible -- but only if the canonical form keeps the field, which it did not:
# the source is rebuilt key by key, and an omitted key is dropped in silence.


async def test_checkpoint_keeps_the_backend_it_was_given(tmp_path: Path) -> None:
    output = tmp_path / 'events.jsonl'
    writer = JsonlEventWriter(output, source=_source() | {'trace_backend': 'tdsql-boundless'})
    await writer.open()
    await writer.close(reason='interrupted')

    stored = json.loads(writer.checkpoint.read_text(encoding='utf-8'))

    # Read from the file rather than the writer: persisting it is the point, and dropping it
    # here is what made the comparison below unreachable.
    assert stored['source']['trace_backend'] == 'tdsql-boundless'


async def test_resume_under_another_backend_is_refused(tmp_path: Path) -> None:
    output = tmp_path / 'events.jsonl'
    first = JsonlEventWriter(output, source=_source() | {'trace_backend': 'tdsql-boundless'})
    await first.open()
    await first.commit_page(_stream(), expected_cursor=0, rows=[_event()])
    await first.close(reason='interrupted')
    before = output.read_bytes(), first.checkpoint.read_bytes(), first.manifest.read_bytes()

    resumed = JsonlEventWriter(output, source=_source() | {'trace_backend': 'tdsql'}, resume=True)
    with pytest.raises(ExportError, match='source'):
        await resumed.open()

    assert (output.read_bytes(), first.checkpoint.read_bytes(), first.manifest.read_bytes()) == before


async def test_resume_that_drops_the_backend_is_refused(tmp_path: Path) -> None:
    """Omitting the name must not bypass the check that naming it enables."""
    output = tmp_path / 'events.jsonl'
    first = JsonlEventWriter(output, source=_source() | {'trace_backend': 'tdsql-boundless'})
    await first.open()
    await first.close(reason='interrupted')

    resumed = JsonlEventWriter(output, source=_source(), resume=True)
    with pytest.raises(ExportError, match='source'):
        await resumed.open()


async def test_a_checkpoint_without_a_backend_still_resumes(tmp_path: Path) -> None:
    """Checkpoints predate this field, so absence has to stay a valid identity.

    Recorded only when named, which is why the key is absent rather than null: a null would
    make an older checkpoint unequal to a run that names no backend, stranding it.
    """
    output = tmp_path / 'events.jsonl'
    first = JsonlEventWriter(output, source=_source())
    await first.open()
    await first.close(reason='interrupted')
    stored = json.loads(first.checkpoint.read_text(encoding='utf-8'))
    assert 'trace_backend' not in stored['source']

    resumed = JsonlEventWriter(output, source=_source(), resume=True)
    await resumed.open()
    await resumed.close(reason='interrupted')


@pytest.mark.parametrize('value', ['', 17, None, ['tdsql']])
def test_an_unusable_backend_name_is_rejected(tmp_path: Path, value: Any) -> None:
    """An empty or non-string name cannot identify a backend, so it is refused at once.

    Raised from the constructor, where the source is canonicalised -- before any file is
    touched, which is why this needs no open() and no cleanup.
    """
    with pytest.raises(ExportError, match='trace_backend'):
        JsonlEventWriter(tmp_path / 'events.jsonl', source=_source() | {'trace_backend': value})


async def test_resume_truncates_data_fsynced_before_failed_checkpoint_and_retries_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from nexus.distributed.export import writer as export_output

    output = tmp_path / 'events.jsonl'
    writer = JsonlEventWriter(output, source=_source())
    await writer.open()
    initial_checkpoint = writer.checkpoint.read_bytes()
    real_replace = export_output.os.replace

    def fail_checkpoint_replace(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        if Path(dst) == writer.checkpoint:
            raise OSError('injected checkpoint replace failure')
        real_replace(src, dst)

    monkeypatch.setattr(export_output.os, 'replace', fail_checkpoint_replace)
    with pytest.raises(ExportError, match='checkpoint'):
        await writer.commit_page(_stream(), expected_cursor=0, rows=[_event()])
    uncommitted_tail = output.read_bytes()
    assert uncommitted_tail.endswith(b'\n')
    assert len(uncommitted_tail) > 0
    assert writer.checkpoint.read_bytes() == initial_checkpoint
    assert writer.committed_records == 0
    monkeypatch.setattr(export_output.os, 'replace', real_replace)
    await writer.close(reason='error')

    resumed = JsonlEventWriter(output, source=_source(), resume=True)
    await resumed.open()
    assert output.read_bytes() == b''
    await resumed.commit_page(_stream(), expected_cursor=0, rows=[_event()])
    await resumed.close(reason='interrupted')

    assert [json.loads(line)['record_id'] for line in output.read_bytes().splitlines()] == [1]
    assert _read_json(resumed.checkpoint)['committed_records'] == 1


async def test_manifest_failure_after_checkpoint_stops_writer_but_resume_keeps_committed_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from nexus.distributed.export import writer as export_output

    output = tmp_path / 'events.jsonl'
    writer = JsonlEventWriter(output, source=_source())
    await writer.open()
    real_replace = export_output.os.replace

    def fail_manifest_replace(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        if Path(dst) == writer.manifest:
            raise OSError('injected manifest replace failure')
        real_replace(src, dst)

    monkeypatch.setattr(export_output.os, 'replace', fail_manifest_replace)
    with pytest.raises(ExportError, match='manifest after checkpoint'):
        await writer.commit_page(_stream(), expected_cursor=0, rows=[_event()])
    assert _read_json(writer.checkpoint)['committed_records'] == 1
    assert writer.committed_records == 1
    assert [json.loads(line)['record_id'] for line in output.read_bytes().splitlines()] == [1]
    with pytest.raises(ExportError, match='I/O failure'):
        await writer.commit_page(_stream(), expected_cursor=1, rows=[_event(2)])
    monkeypatch.setattr(export_output.os, 'replace', real_replace)
    await writer.close(reason='error')

    resumed = JsonlEventWriter(output, source=_source(), resume=True)
    await resumed.open()
    assert resumed.cursor_for(_stream()) == 1
    await resumed.commit_page(_stream(), expected_cursor=1, rows=[_event(2)])
    await resumed.close(reason='interrupted')
    assert [json.loads(line)['record_id'] for line in output.read_bytes().splitlines()] == [1, 2]


@pytest.mark.parametrize('damage', ['short', 'hash', 'cursor'])
async def test_resume_refuses_corrupt_committed_state_without_truncating(tmp_path: Path, damage: str) -> None:
    output = tmp_path / f'{damage}.jsonl'
    first = JsonlEventWriter(output, source=_source())
    await first.open()
    await first.commit_page(_stream(), expected_cursor=0, rows=[_event()])
    await first.close(reason='interrupted')
    checkpoint = _read_json(first.checkpoint)
    if damage == 'short':
        output.write_bytes(output.read_bytes()[:-1])
    elif damage == 'hash':
        data = bytearray(output.read_bytes())
        data[0] ^= 1
        output.write_bytes(data)
    else:
        checkpoint['streams'] = {'child-1': 1}
        first.checkpoint.write_text(json.dumps(checkpoint), encoding='utf-8')
    before = output.read_bytes()

    resumed = JsonlEventWriter(output, source=_source(), resume=True)
    with pytest.raises(ExportError):
        await resumed.open()

    assert output.read_bytes() == before


async def test_resume_validates_rows_and_cursors_even_when_hash_matches(tmp_path: Path) -> None:
    output = tmp_path / 'events.jsonl'
    first = JsonlEventWriter(output, source=_source())
    await first.open()
    await first.commit_page(_stream(), expected_cursor=0, rows=[_event()])
    await first.close(reason='interrupted')
    row = json.loads(output.read_text(encoding='utf-8'))
    row['record_id'] = 2
    encoded = (json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n').encode()
    output.write_bytes(encoded)
    checkpoint = _read_json(first.checkpoint)
    checkpoint['committed_bytes'] = len(encoded)
    checkpoint['committed_sha256'] = sha256(encoded).hexdigest()
    first.checkpoint.write_text(json.dumps(checkpoint), encoding='utf-8')

    resumed = JsonlEventWriter(output, source=_source(), resume=True)
    with pytest.raises(ExportError, match='cursor'):
        await resumed.open()


async def test_cancellation_waits_for_inflight_fsync_before_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from nexus.distributed.export import writer as export_output

    output = tmp_path / 'events.jsonl'
    writer = JsonlEventWriter(output, source=_source())
    await writer.open()
    data_inode = output.stat().st_ino
    entered = threading.Event()
    release = threading.Event()
    real_fsync = export_output.os.fsync
    armed = True

    def blocking_data_fsync(fd: int) -> None:
        if armed and os.fstat(fd).st_ino == data_inode:
            entered.set()
            assert release.wait(5), 'test did not release blocked data fsync'
        real_fsync(fd)

    monkeypatch.setattr(export_output.os, 'fsync', blocking_data_fsync)
    commit = asyncio.create_task(writer.commit_page(_stream(), expected_cursor=0, rows=[_event()]))
    assert await asyncio.to_thread(entered.wait, 2)
    commit.cancel()
    close = asyncio.create_task(writer.close(reason='interrupted'))
    await asyncio.sleep(0.05)
    assert not commit.done()
    assert not close.done()
    armed = False
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await commit
    await close

    assert _read_json(writer.checkpoint)['committed_records'] == 1
    assert _read_json(writer.manifest)['status'] == 'interrupted'


async def test_completed_snapshot_publishes_data_then_manifest_without_checkpoint(tmp_path: Path) -> None:
    output = tmp_path / 'events.jsonl'
    writer = JsonlEventWriter(output, source=_source(), follow=False)
    await writer.open()
    partial = output.with_name(f'{output.name}.partial')
    assert partial.exists()
    assert not output.exists()
    assert not writer.checkpoint.exists()
    assert not writer.manifest.exists()
    await writer.commit_page(_stream(), expected_cursor=0, rows=[_event()])
    await writer.update_observation({'execution_status': 'running'})
    await writer.close(reason='snapshot')

    assert output.exists()
    assert not partial.exists()
    assert not writer.checkpoint.exists()
    manifest = _read_json(writer.manifest)
    assert manifest['mode'] == 'snapshot'
    assert manifest['status'] == 'snapshot'
    assert manifest['committed_records'] == 1
    assert manifest['observed'] == {'execution_status': 'running'}


@pytest.mark.parametrize('reason', ['interrupted', 'error'])
async def test_incomplete_snapshot_retains_partial_and_publishes_nothing(tmp_path: Path, reason: str) -> None:
    output = tmp_path / f'{reason}.jsonl'
    writer = JsonlEventWriter(output, source=_source(), follow=False)
    await writer.open()
    await writer.commit_page(_stream(), expected_cursor=0, rows=[_event()])
    await writer.close(reason=reason)

    assert output.with_name(f'{output.name}.partial').exists()
    assert not output.exists()
    assert not writer.checkpoint.exists()
    assert not writer.manifest.exists()


async def test_snapshot_manifest_failure_keeps_published_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from nexus.distributed.export import writer as export_output

    output = tmp_path / 'events.jsonl'
    writer = JsonlEventWriter(output, source=_source(), follow=False)
    await writer.open()
    await writer.commit_page(_stream(), expected_cursor=0, rows=[_event()])
    real_replace = export_output.os.replace

    def fail_manifest_replace(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        if Path(dst) == writer.manifest:
            raise OSError('injected manifest replace failure')
        real_replace(src, dst)

    monkeypatch.setattr(export_output.os, 'replace', fail_manifest_replace)
    with pytest.raises(ExportError, match='manifest'):
        await writer.close(reason='snapshot')

    assert output.exists()
    assert not output.with_name(f'{output.name}.partial').exists()
    assert not writer.manifest.exists()


async def test_snapshot_with_failed_data_fsync_cannot_be_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from nexus.distributed.export import writer as export_output

    output = tmp_path / 'events.jsonl'
    writer = JsonlEventWriter(output, source=_source(), follow=False)
    await writer.open()
    partial = output.with_name(f'{output.name}.partial')
    data_inode = partial.stat().st_ino
    real_fsync = export_output.os.fsync
    fail_once = True

    def fail_data_fsync(fd: int) -> None:
        nonlocal fail_once
        if fail_once and os.fstat(fd).st_ino == data_inode:
            fail_once = False
            raise OSError('injected data fsync failure')
        real_fsync(fd)

    monkeypatch.setattr(export_output.os, 'fsync', fail_data_fsync)
    with pytest.raises(ExportError, match='commit event page'):
        await writer.commit_page(_stream(), expected_cursor=0, rows=[_event()])
    monkeypatch.setattr(export_output.os, 'fsync', real_fsync)

    await writer.close(reason='snapshot')

    assert partial.exists()
    assert not output.exists()
    assert not writer.manifest.exists()


async def test_multi_megabyte_raw_payload_round_trips_across_recovery_chunks(tmp_path):
    output = tmp_path / 'large-events.jsonl'
    raw = '{ "text": "' + ('实时✓' * 300_000) + '", "integer": 9223372036854775806, "nullable": null }'
    assert len(raw.encode()) > 2 * 1024 * 1024
    first = {**_event(1), 'event_data': raw, 'created_at': None}
    writer = JsonlEventWriter(output, source=_source())
    await writer.open()
    await writer.commit_page(_stream(), expected_cursor=0, rows=[first])
    await writer.close(reason='interrupted')
    prefix = output.read_bytes()
    checkpoint = json.loads(writer.checkpoint.read_text())
    assert checkpoint['committed_bytes'] == len(prefix)
    assert checkpoint['committed_sha256'] == sha256(prefix).hexdigest()
    assert json.loads(prefix) == first
    with output.open('ab') as stream:
        stream.write(b'{"uncommitted":"' + b'x' * (1024 * 1024 + 7))
    resumed = JsonlEventWriter(output, source=_source(), resume=True)
    await resumed.open()
    try:
        assert resumed.cursor_for(_stream()) == 1
        assert output.read_bytes() == prefix
        second = {**_event(7), 'created_at': None}
        await resumed.commit_page(_stream(), expected_cursor=1, rows=[second])
    finally:
        await resumed.close(reason='interrupted')
    encoded = output.read_bytes()
    assert encoded.startswith(prefix)
    assert [json.loads(line) for line in encoded.splitlines()] == [first, second]
    final = json.loads(resumed.checkpoint.read_text())
    assert final['committed_records'] == 2
    assert final['committed_bytes'] == len(encoded)
    assert final['committed_sha256'] == sha256(encoded).hexdigest()
