"""Persistent trajectory exports use typed columns and never fetch DU blobs."""

import asyncio
import json
import os
import pickle
import struct
import subprocess
import sys
import threading

from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from tests.unit.gateway.redis_store.fake_redis import FakeRedis
from tests.unit.gateway.redis_store.file_storage.test_publication import MemoryFS

from nexus.gateway.trajectory_store_backends.redis import HybridNodeRef, NodeRecord, NodeRef, RedisTrajectoryRecordStore
from nexus.gateway.trajectory_store_backends.redis_store.checkpoint import dump as dump_module
from nexus.gateway.trajectory_store_backends.redis_store.checkpoint import iter_payloads, read_dump
from nexus.gateway.trajectory_store_backends.redis_store.types import RedisClientError
from nexus.gateway.trajectory_store_backends.types import NodePayloadRecord, SegmentDataRecord


class DumpFS(MemoryFS):
    def __init__(self, raw):
        super().__init__()
        self.raw = raw
        self.prepared = set()
        self.calls = []
        self.failed = False
        self.prepare_failed = False
        self.finished = False

    def get(self, url):
        raise AssertionError('dump must not fetch DU blobs')

    def prepare_dump_sources(self, dump_id, urls):
        assert self.raw._dump_protections
        self.calls.append(('prepare', dump_id))
        self.prepared.update(urls)
        if self.prepare_failed:
            raise RedisClientError('prepare failed', code='io_error')

    def start_dump_copy(self, dump_id, dump_dir):
        assert not self.raw._dump_protections
        self.calls.append(('start', dump_id))
        if not self.failed:
            directory = Path(dump_dir) / 'data'
            directory.mkdir(exist_ok=True)
            for url in self.prepared:
                (directory / (url.rsplit('/', 1)[-1] + '.blob')).write_bytes(self.blobs[url])

    def get_dump_status(self, dump_id):
        return {'du': {'state': 'failed' if self.failed else 'complete', 'error': 'copy failed'}}

    def finalize_dump(self, dump_id):
        self.finished = True


@pytest.fixture
def make_store():
    with ExitStack() as resources:

        def create():
            raw = FakeRedis()
            storage = DumpFS(raw)
            config = {'redis_client': raw, 'fs_client': storage}
            store = resources.enter_context(RedisTrajectoryRecordStore('run', config))
            return store, storage

        yield create


@pytest.mark.parametrize(
    ('name', 'value'), [('routed_experts', b''), ('routed_experts', []), ('multi_modal_items', {})]
)
async def test_selected_inline_blob_is_rejected_but_numeric_projection_is_readable(tmp_path, make_store, name, value):
    store, fs = make_store()
    ref = NodeRef(store.namespace, 'legacy')
    record = NodeRecord('t', 1, np.array([1], dtype=np.int32), **{name: value})
    store._client._r.set(store._client._node_key(ref.record_id), record.serialize())
    _, writing = store.dump_payloads([ref], dump_dir=str(tmp_path / 'selected'))
    with pytest.raises(ValueError, match=name):
        await writing
    assert not (tmp_path / 'selected' / 'trajectory.parquet').exists()
    assert not fs.prepared
    dumped, writing = store.dump_payloads([ref], dump_dir=str(tmp_path / 'numeric'), fields=['token_ids'])
    await writing
    fields = read_dump(tmp_path / 'numeric', dumped)[0]['fields']
    assert set(fields) == {'token_ids'}
    np.testing.assert_array_equal(fields['token_ids'], record.token_ids)


@pytest.mark.parametrize(
    ('blob', 'error'),
    [(b'x', struct.error), (bytes(8), EOFError), (b'\x01' + bytes(7) + b'x', pickle.UnpicklingError)],
)
async def test_offline_reader_reports_corrupt_blob_without_breaking_metadata_projection(
    tmp_path, make_store, blob, error
):
    store, _ = make_store()
    refs = await store.put_many([NodePayloadRecord('t', 1, np.array([1], dtype=np.int32), routed_experts=[2])])
    dumped, writing = store.dump_payloads(refs, dump_dir=str(tmp_path))
    await writing
    path = list(iter_payloads(tmp_path, load_blobs=False))[0]['fields']['routed_experts']
    (tmp_path / path.removeprefix('ckpt://')).write_bytes(blob)
    with pytest.raises(error):
        read_dump(tmp_path, dumped)
    assert list(iter_payloads(tmp_path, load_blobs=False))[0]['fields']['routed_experts'] == path


@pytest.mark.parametrize(
    ('name', 'value'),
    [
        ('token_ids', [1]),
        ('token_ids', np.array([1], dtype=np.int64)),
        ('token_ids', np.array([1], dtype='>i4')),
        ('topk_indices', np.array([1], dtype=np.uint32)),
        ('topk_indices', np.array([True])),
        ('logprobs', np.array([-0.5], dtype=np.float64)),
        ('logprobs', np.array([1], dtype=np.int32)),
        ('topk_values', np.array([0.5], dtype=np.float16)),
    ],
)
async def test_dump_rejects_noncanonical_array_without_publishing_checkpoint(tmp_path, make_store, name, value):
    store, _ = make_store()
    payload = replace(NodePayloadRecord('t', 1, np.array([1], dtype=np.int32)), **{name: value})
    refs = await store.put_many([payload])
    _, writing = store.dump_payloads(refs, dump_dir=str(tmp_path))
    with pytest.raises(TypeError, match=name):
        await writing
    assert not (tmp_path / 'trajectory.parquet').exists()
    assert not store._client._r._dump_protections


async def test_fixed_arrays_preserve_shape_noncontiguous_storage_and_external_blob(tmp_path, make_store, monkeypatch):
    import pyarrow.parquet as pq

    store, _ = make_store()
    payload = NodePayloadRecord(
        'traj',
        1,
        np.array([1, 2], dtype=np.int32),
        logprobs=np.array([-0.1, -np.inf], dtype=np.float32),
        topk_indices=np.arange(6, dtype=np.int32).reshape(3, 2).T,
        topk_values=np.arange(6, dtype=np.float32).reshape(2, 3),
        routed_experts=np.arange(8, dtype=np.uint8).reshape(2, 4),
    )
    empty = NodePayloadRecord(
        'traj',
        2,
        np.empty(0, dtype=np.int32),
        topk_indices=np.empty((2, 0), dtype=np.int32),
        topk_values=np.empty((0, 3), dtype=np.float32),
    )
    refs = await store.put_many([payload, empty])
    _, write = store.dump_payloads(refs, dump_dir=str(tmp_path))
    await write
    rows = list(iter_payloads(tmp_path))
    fields = rows[0]['fields']
    for name in ('token_ids', 'logprobs', 'topk_indices', 'topk_values', 'routed_experts'):
        np.testing.assert_array_equal(fields[name], getattr(payload, name))
        assert fields[name].dtype == getattr(payload, name).dtype
        assert fields[name].shape == getattr(payload, name).shape
    for name in ('token_ids', 'topk_indices', 'topk_values'):
        actual, expected = rows[1]['fields'][name], getattr(empty, name)
        assert isinstance(actual, np.ndarray)
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        np.testing.assert_array_equal(actual, expected)
    columns = []
    original = pq.ParquetFile.iter_batches

    def batches(self, **kwargs):
        columns.extend(kwargs['columns'])
        return original(self, **kwargs)

    monkeypatch.setattr(pq.ParquetFile, 'iter_batches', batches)
    summary = list(iter_payloads(tmp_path, fields=['topk_values'], summaries=True))[0]
    assert summary['fields']['topk_values'] == {'dtype': '<f4', 'shape': [2, 3], 'elements': 6}
    assert 'topk_values.shape' in columns
    assert not any(name.endswith('.values') or name == 'topk_values' for name in columns)


@pytest.mark.parametrize('cleanup_failed', [False, True])
async def test_hybrid_uses_authoritative_urls_no_blob_fetch_and_releases_protection_before_copy(
    tmp_path, monkeypatch, caplog, cleanup_failed, make_store
):
    store, fs = make_store()
    refs = await store.put_many([NodePayloadRecord('traj', 1, np.array([1], dtype=np.int32), routed_experts=[4, 5])])
    original = store._client.get_node_blobs([refs[0].record_id])
    stale = HybridNodeRef('run', refs[0].record_id, routed_experts_url='hykvfs://stale/unused')
    if cleanup_failed:

        def finish(dump_id):
            raise RedisClientError('cleanup failed', code='io_error')

        monkeypatch.setattr(fs, 'finalize_dump', finish)
    dumped, write = store.dump_payloads([stale], dump_dir=str(tmp_path))
    await write
    assert fs.prepared == set(fs.blobs)
    if cleanup_failed:
        assert (tmp_path / 'trajectory.parquet').exists()
        assert 'cleanup failed' in caplog.text
        assert dumped[0].dump_id in caplog.text
    else:
        assert fs.finished
    assert [item[0] for item in fs.calls] == ['prepare', 'start']
    assert store._client.get_node_blobs([refs[0].record_id]) == original
    assert list(iter_payloads(tmp_path))[0]['fields']['routed_experts'] == [4, 5]
    assert list(iter_payloads(tmp_path, load_blobs=False))[0]['fields']['routed_experts'].startswith('ckpt://data/')


async def test_projection_skips_unselected_blobs_and_segment_fields(tmp_path, make_store):
    store, fs = make_store()
    refs = await store.put_many([
        NodePayloadRecord(
            'traj',
            1,
            np.array([1], dtype=np.int32),
            topk_values=np.array([[1.5]], dtype=np.float32),
            routed_experts=[3],
        ),
        SegmentDataRecord('traj', 's', {'reward': 4, 'unused': b'large'}),
    ])
    _, write = store.dump_payloads(refs, dump_dir=str(tmp_path), fields=['token_ids', 'topk_values', 'reward'])
    await write
    assert not fs.prepared
    rows = list(iter_payloads(tmp_path))
    assert set(rows[0]['fields']) == {'token_ids', 'topk_values'}
    np.testing.assert_array_equal(rows[0]['fields']['token_ids'], np.array([1], dtype=np.int32))
    np.testing.assert_array_equal(rows[0]['fields']['topk_values'], np.array([[1.5]], dtype=np.float32))
    assert rows[1]['fields'] == {'reward': 4}


async def test_snapshot_queries_are_bounded_and_records_deduplicated(tmp_path, monkeypatch, make_store):
    store, _ = make_store()
    refs = await store.put_many([NodePayloadRecord('t', i, np.array([i], dtype=np.int32)) for i in range(35)])
    sizes = []
    original = store._client.get_node_blobs

    def read(ids):
        sizes.append(len(ids))
        return original(ids)

    monkeypatch.setattr(store._client, 'get_node_blobs', read)
    _, write = store.dump_payloads([*refs, *refs], dump_dir=str(tmp_path))
    await write
    assert sizes == [16, 16, 3]


@pytest.mark.parametrize('with_blob', [False, True])
async def test_prepared_batch_releases_protection_before_parquet_encoding(tmp_path, monkeypatch, make_store, with_blob):
    store, fs = make_store()
    refs = await store.put_many([
        NodePayloadRecord('t', i, np.array([i], dtype=np.int32), routed_experts=[i] if with_blob else None)
        for i in range(2)
    ])
    monkeypatch.setattr(dump_module, 'PARQUET_BATCH_SIZE', 1)
    original, written = dump_module._write_parquet_batch, []

    def write(writer, schema, rows):
        index = rows[0]['entry_id']
        assert {key.rsplit(b':node:', 1)[-1].decode() for key in store._client._r._dump_protections} == {
            ref.record_id for ref in refs[index + 1 :]
        }
        if with_blob:
            assert refs[index].routed_experts_url in fs.prepared
        written.append(index)
        return original(writer, schema, rows)

    monkeypatch.setattr(dump_module, '_write_parquet_batch', write)
    _, writing = store.dump_payloads(refs, dump_dir=str(tmp_path))
    await writing
    assert written == [0, 1]


async def test_later_prepare_unknown_abort_releases_all_records(tmp_path, monkeypatch, make_store):
    store, fs = make_store()
    refs = await store.put_many([
        NodePayloadRecord('t', i, np.array([i], dtype=np.int32), routed_experts=[i]) for i in range(2)
    ])
    monkeypatch.setattr(dump_module, 'PARQUET_BATCH_SIZE', 1)
    prepare = fs.prepare_dump_sources

    def fail_later(dump_id, urls):
        prepare(dump_id, urls)
        if refs[1].routed_experts_url in urls:
            assert {key.rsplit(b':node:', 1)[-1].decode() for key in store._client._r._dump_protections} == {
                refs[1].record_id
            }
            raise RedisClientError('second prepare failed', code='io_error')

    def abort(dump_id):
        raise RedisClientError('abort result unknown', code='timeout')

    monkeypatch.setattr(fs, 'prepare_dump_sources', fail_later)
    monkeypatch.setattr(fs, 'finalize_dump', abort)
    dumped, writing = store.dump_payloads(refs, dump_dir=str(tmp_path))
    with pytest.raises(RedisClientError, match='abort result unknown') as caught:
        await writing
    assert caught.value.body['dump_id'] == dumped[0].dump_id
    assert not store._client._r._dump_protections
    assert not (tmp_path / 'trajectory.parquet').exists()


async def test_cancel_later_prepare_drains_without_protecting_again_completed_batch(tmp_path, monkeypatch, make_store):
    store, fs = make_store()
    refs = await store.put_many([
        NodePayloadRecord('t', i, np.array([i], dtype=np.int32), routed_experts=[i]) for i in range(2)
    ])
    monkeypatch.setattr(dump_module, 'PARQUET_BATCH_SIZE', 1)
    started, finish = threading.Event(), threading.Event()
    prepare = fs.prepare_dump_sources

    def slow_later(dump_id, urls):
        prepare(dump_id, urls)
        if refs[1].routed_experts_url in urls:
            started.set()
            assert finish.wait(5)

    monkeypatch.setattr(fs, 'prepare_dump_sources', slow_later)
    _, writing = store.dump_payloads(refs, dump_dir=str(tmp_path))
    task = asyncio.create_task(writing)
    try:
        assert await asyncio.to_thread(started.wait, 2)
        assert {key.rsplit(b':node:', 1)[-1].decode() for key in store._client._r._dump_protections} == {
            refs[1].record_id
        }
        task.cancel()
        await asyncio.sleep(0.05)
        assert not task.done()
        assert not fs.finished
        assert {key.rsplit(b':node:', 1)[-1].decode() for key in store._client._r._dump_protections} == {
            refs[1].record_id
        }
    finally:
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert fs.finished
    assert not store._client._r._dump_protections
    assert not (tmp_path / 'trajectory.parquet').exists()


@pytest.mark.parametrize('abort_mode', ['normal', 'busy', 'unknown'])
async def test_prepare_failure_releases_protection_before_abort(tmp_path, monkeypatch, make_store, abort_mode):
    store, fs = make_store()
    refs = await store.put_many([NodePayloadRecord('t', 1, np.array([1], dtype=np.int32), routed_experts=[3])])
    fs.prepare_failed = True
    original, attempts = fs.finalize_dump, 0

    def abort(dump_id):
        nonlocal attempts
        attempts += 1
        assert not store._client._r._dump_protections
        if abort_mode == 'unknown':
            raise RedisClientError('abort result unknown', code='timeout')
        if abort_mode == 'busy' and attempts == 1:
            raise RedisClientError('prepare still running', code='dump_busy')
        return original(dump_id)

    monkeypatch.setattr(fs, 'finalize_dump', abort)
    _, write = store.dump_payloads(refs, dump_dir=str(tmp_path))
    with pytest.raises(RedisClientError, match='abort result unknown' if abort_mode == 'unknown' else 'prepare failed'):
        await write
    assert attempts == (2 if abort_mode == 'busy' else 1)
    assert not (tmp_path / 'trajectory.parquet').exists()
    assert not store._client._r._dump_protections
    if abort_mode != 'unknown':
        assert fs.finished


async def test_cancel_waits_for_inflight_copy_before_releasing_pins(tmp_path, monkeypatch, make_store):
    store, fs = make_store()
    started, finish = threading.Event(), threading.Event()
    try:
        refs = await store.put_many([NodePayloadRecord('t', 1, np.array([1], dtype=np.int32), routed_experts=[3])])
        original = fs.start_dump_copy

        def start(dump_id, directory):
            original(dump_id, directory)
            started.set()

        def status(dump_id):
            return {'du': {'state': 'complete' if finish.is_set() else 'running'}}

        monkeypatch.setattr(fs, 'start_dump_copy', start)
        monkeypatch.setattr(fs, 'get_dump_status', status)
        _, write = store.dump_payloads(refs, dump_dir=str(tmp_path))
        task = asyncio.create_task(write)
        assert await asyncio.to_thread(started.wait, 2)
        task.cancel()
        await asyncio.sleep(0.05)
        assert not task.done()
        assert not fs.finished
        assert not store._client._r._dump_protections
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert fs.finished
        assert not (tmp_path / 'trajectory.parquet').exists()
    finally:
        finish.set()


async def test_reader_preserves_ref_order_identity_and_skips_unselected_blob(tmp_path, make_store):
    store, _ = make_store()
    refs = await store.put_many([
        NodePayloadRecord('t', 1, np.array([1], dtype=np.int32), routed_experts=[3]),
        NodePayloadRecord('t', 2, np.array([2], dtype=np.int32), routed_experts=[4]),
    ])
    dumped, write = store.dump_payloads(refs, dump_dir=str(tmp_path))
    await write
    rows = read_dump(tmp_path, [dumped[1], dumped[0], dumped[1]], fields=['token_ids'])
    np.testing.assert_array_equal([row['fields']['token_ids'] for row in rows], [[2], [1], [2]])
    with pytest.raises(ValueError, match='different checkpoint'):
        read_dump(tmp_path, [replace(dumped[0], dump_id='other')])
    metadata = list(iter_payloads(tmp_path, load_blobs=False))
    (tmp_path / metadata[1]['fields']['routed_experts'].removeprefix('ckpt://')).unlink()
    assert read_dump(tmp_path, [dumped[0]])[0]['fields']['routed_experts'] == [3]


@pytest.mark.parametrize('failure', ['du_copy', 'metadata_copy', 'status'])
async def test_copy_or_status_failure_retains_sources_and_reports_dump_id(tmp_path, monkeypatch, failure, make_store):
    store, fs = make_store()
    refs = await store.put_many([NodePayloadRecord('t', 1, np.array([1], dtype=np.int32), routed_experts=[3])])

    if failure == 'du_copy':
        fs.failed = True
    elif failure == 'metadata_copy':

        def copy(source, destination):
            raise OSError('metadata copy failed')

        monkeypatch.setattr(dump_module.shutil, 'copyfile', copy)
    else:

        def status(dump_id):
            raise RedisClientError('status unavailable', code='timeout')

        monkeypatch.setattr(fs, 'get_dump_status', status)
    dumped, write = store.dump_payloads(refs, dump_dir=str(tmp_path))
    with pytest.raises(RedisClientError) as caught:
        await write
    assert caught.value.body['dump_id'] == dumped[0].dump_id
    assert fs.prepared
    assert not fs.finished
    assert not store._client._r._dump_protections
    if failure == 'du_copy':
        assert 'copy failed' in str(caught.value)
    assert not (tmp_path / 'trajectory.parquet').exists()


async def test_checkpoint_module_cli_projects_fields_and_returns_offline_summaries(tmp_path, make_store):
    store, _ = make_store()
    refs = await store.put_many([
        NodePayloadRecord(
            'trajectory',
            1,
            np.array([1, 2], dtype=np.int32),
            topk_values=np.array([[0.5]], dtype=np.float32),
            routed_experts=[3],
        )
    ])
    _, writing = store.dump_payloads(refs, dump_dir=str(tmp_path))
    await writing
    [blob_path] = (tmp_path / 'data').glob('*.blob')
    blob_reference = f'ckpt://data/{blob_path.name}'
    blob_path.unlink()
    root = Path(__file__).resolve().parents[4]
    process = subprocess.run(
        [
            sys.executable,
            '-m',
            'nexus.gateway.trajectory_store_backends.redis_store.checkpoint',
            str(tmp_path),
            '--fields',
            'token_ids',
            'routed_experts',
        ],
        cwd=root,
        env={**os.environ, 'PYTHONPATH': str(root)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    [row] = [json.loads(line) for line in process.stdout.splitlines()]
    assert row['fields'] == {
        'token_ids': {'dtype': '<i4', 'shape': [2], 'elements': 2},
        'routed_experts': blob_reference,
    }
