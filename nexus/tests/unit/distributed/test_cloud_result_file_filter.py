import asyncio
import json
import threading

import pytest

from nexus.distributed.export import writer as writer_module
from nexus.distributed.export.results import make_jsonl_record
from nexus.distributed.export.types import ExportError
from nexus.distributed.export.writer import AtomicJsonlOutput


def record(run_id, output=None):
    return make_jsonl_record(
        run_id=run_id,
        distributed_status='failed',
        start_time=None,
        end_time=None,
        retry_count=0,
        fallback_error={'type': 'SavedError'},
        output={'success': False, 'task_output': output},
        metadata={'label': '中文 ☃'},
    )


async def test_file_filter_requires_successful_validation(tmp_path):
    writer = AtomicJsonlOutput(tmp_path / 'results.jsonl', force=False)

    def unexpected_predicate(row):
        pytest.fail('unvalidated output must not be read')

    with pytest.raises(ExportError, match='not been validated'):
        await writer.filter_records(unexpected_predicate)
    await writer.open()
    try:
        await writer.write_records([record('item')])
        with pytest.raises(ExportError, match='not been validated'):
            await writer.filter_records(unexpected_predicate)
        with pytest.raises(ExportError, match='count mismatch'):
            await writer.close_and_validate(expected_count=2, require_sorted=False)
        with pytest.raises(ExportError, match='not been validated'):
            await writer.filter_records(unexpected_predicate)
    finally:
        await writer.abort()
    assert not writer.output.exists()


async def test_file_filter_preserves_original_line_bytes_order_and_scalar_types(tmp_path):
    writer = AtomicJsonlOutput(tmp_path / 'results.jsonl', force=False)
    rows = [record('z', 0), record('middle'), record('a', False)]
    await writer.open()
    await writer.write_records(rows)
    await writer.close_and_validate(expected_count=3, require_sorted=False)
    # Vary only the whitespace and Unicode encoding of already validated records.
    lines = [
        (' \t' + json.dumps(rows[0], ensure_ascii=False, separators=(', ', ' : ')) + ' \r\n').encode(),
        (json.dumps(rows[1]) + '\n').encode(),
        ('\t' + json.dumps(rows[2], ensure_ascii=True, separators=(',', ':')) + '\t\n').encode(),
    ]
    writer.partial.write_bytes(b''.join(lines))
    visited = []

    def keep(row):
        visited.append(row)
        return row['task_output'] is not None

    assert await writer.filter_records(keep) == 2
    assert visited == rows
    assert type(visited[0]['task_output']) is int
    assert type(visited[2]['task_output']) is bool
    assert writer.partial.read_bytes() == lines[0] + lines[2]
    assert not writer.output.exists()
    await writer.publish()
    assert writer.output.read_bytes() == lines[0] + lines[2]
    assert not writer.partial.exists()
    assert set(tmp_path.iterdir()) == {writer.output}


@pytest.mark.parametrize('rows', [[], [record('item')]], ids=['empty-input', 'all-removed'])
async def test_file_filter_can_publish_empty_output(tmp_path, rows):
    writer = AtomicJsonlOutput(tmp_path / 'results.jsonl', force=False)
    await writer.open()
    await writer.write_records(rows)
    await writer.close_and_validate(expected_count=len(rows), require_sorted=False)
    visited = []

    def remove(row):
        visited.append(row)
        return False

    assert await writer.filter_records(remove) == 0
    assert visited == rows
    assert writer.partial.read_bytes() == b''
    await writer.publish()
    assert writer.output.read_bytes() == b''
    assert set(tmp_path.iterdir()) == {writer.output}


async def test_file_filter_replace_failure_keeps_original_partial_and_output(tmp_path, monkeypatch):
    writer = AtomicJsonlOutput(tmp_path / 'results.jsonl', force=True)
    writer.output.write_bytes(b'previous output\n')
    await writer.open()
    await writer.write_records([record('first'), record('last', 0)])
    await writer.close_and_validate(expected_count=2, require_sorted=False)
    original = writer.partial.read_bytes()

    def fail_replace(source, destination):
        assert destination == writer.partial
        assert source.parent == writer.partial.parent
        assert json.loads(source.read_bytes())['run_id'] == 'last'
        assert writer.partial.read_bytes() == original
        raise OSError('replace failed')

    monkeypatch.setattr(writer_module.os, 'replace', fail_replace)
    with pytest.raises(OSError, match='replace failed'):
        await writer.filter_records(lambda row: row['run_id'] == 'last')
    await writer.abort()
    assert writer.partial.read_bytes() == original
    assert writer.output.read_bytes() == b'previous output\n'
    assert set(tmp_path.iterdir()) == {writer.output, writer.partial}


async def test_cancelled_file_filter_waits_for_thread_and_does_not_publish(tmp_path):
    writer = AtomicJsonlOutput(tmp_path / 'results.jsonl', force=True)
    writer.output.write_bytes(b'previous output\n')
    await writer.open()
    await writer.write_records([record('item', False)])
    await writer.close_and_validate(expected_count=1, require_sorted=False)
    original = writer.partial.read_bytes()
    entered = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()

    def blocked_keep(row):
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(5)
        return True

    async def filter_and_publish():
        await writer.filter_records(blocked_keep)
        await writer.publish()

    task = asyncio.create_task(filter_and_publish())
    try:
        await asyncio.wait_for(entered.wait(), 2)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        assert writer.partial.read_bytes() == original
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await writer.abort()
    assert writer.partial.read_bytes() == original
    assert writer.output.read_bytes() == b'previous output\n'
    assert set(tmp_path.iterdir()) == {writer.output, writer.partial}
