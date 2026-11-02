"""Checkpoint restoration crosses Redis, DU and shared-file ownership boundaries."""

import asyncio
import os
import pickle
import shutil
import subprocess
import sys

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from threading import Event, Lock
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pyarrow.parquet as pq
import pytest

from nexus.gateway.trajectory_store_backends.redis import HybridNodeRef, RedisTrajectoryRecordStore
from nexus.gateway.trajectory_store_backends.redis_store.checkpoint import read_dump
from nexus.gateway.trajectory_store_backends.redis_store.client import connect_redis
from nexus.gateway.trajectory_store_backends.redis_store.file_storage.blob_codec import encode_blob
from nexus.gateway.trajectory_store_backends.redis_store.file_storage.client import FSClient
from nexus.gateway.trajectory_store_backends.redis_store.file_storage.controller import ControllerService
from nexus.gateway.trajectory_store_backends.redis_store.file_storage.data_unit import DataUnitService
from nexus.gateway.trajectory_store_backends.redis_store.file_storage.types import DataUnitRegistration, Topology
from nexus.gateway.trajectory_store_backends.types import NodePayloadRecord, SegmentDataRecord


@pytest.fixture
def target_services(redis_backend_config, tmp_path):
    with ExitStack() as resources:
        redis = connect_redis(redis_backend_config['url'])
        resources.callback(redis.close)
        units = []
        for index in range(2):
            unit = DataUnitService(f'target-{index}', ssd_offload_dir=tmp_path / f'ssd-{index}')
            resources.callback(unit.stop)
            unit.start()
            units.append(unit)
        topology = Topology.create([DataUnitRegistration(unit.data_unit_id, unit.endpoint) for unit in units])
        controller = ControllerService(redis, topology).start()
        resources.callback(controller.stop)
        for unit in units:
            unit.configure(topology, controller.endpoint)
        fs = FSClient(controller.endpoint, timeout=5, io_threads=1)
        resources.callback(fs.close)
        yield SimpleNamespace(fs=fs, units=units, topology=topology)


def make_store(redis_backend_config, fs):
    config = {**redis_backend_config, 'fs_client': fs, 'key_prefix': 'checkpoint-load-' + uuid4().hex}
    return RedisTrajectoryRecordStore('restored-' + uuid4().hex, config)


def assert_payload_equal(actual, expected):
    if isinstance(expected, np.ndarray):
        assert isinstance(actual, np.ndarray)
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for name, value in expected.items():
            assert_payload_equal(actual[name], value)
    elif isinstance(expected, list | tuple):
        assert type(actual) is type(expected)
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            assert_payload_equal(left, right)
    else:
        assert actual == expected


def node(index):
    return NodePayloadRecord(
        'trajectory',
        index,
        np.arange(5, dtype=np.int32),
        logprobs=np.linspace(-1, 0, 5, dtype=np.float32),
        topk_indices=np.arange(15, dtype=np.int32).reshape(5, 3),
        topk_values=np.linspace(-3, 0, 15, dtype=np.float32).reshape(5, 3),
        routed_experts=np.arange(40, dtype=np.uint16).reshape(5, 8),
        multi_modal_items={'image': [{'raw': b'\x00\xff', 'pixels': np.arange(6, dtype=np.uint8).reshape(2, 3)}]},
    )


async def test_load_roundtrip_fresh_refs_after_sources_stop(
    services, target_services, redis_backend_config, tmp_path, caplog
):
    with make_store(redis_backend_config, services.fs) as source:
        refs = await source.put_many([
            node(7),
            SegmentDataRecord('trajectory', 'segment', {'score': 1.25, 'raw': b'abc', 'array': np.arange(3)}),
            NodePayloadRecord(
                'trajectory',
                8,
                np.array(10, dtype=np.int32),
                topk_values=np.empty((2, 0), dtype=np.float32),
                routed_experts=b'',
            ),
        ])
        ordered = [refs[1], refs[0], refs[2], refs[0]]
        expected = await source.get_many(ordered)
        original = source._client.get_node_blobs([ref.record_id for ref in refs])
        directory = tmp_path / 'checkpoint'
        dumped, writing = source.dump_payloads(ordered, dump_dir=str(directory))
        assert not directory.exists()
        assert [ref.kind for ref in dumped] == ['segment', 'node', 'node', 'node']
        assert dumped[1] == dumped[3]
        assert dumped[1] is not dumped[3]
        assert not hasattr(dumped[1], 'record_id')
        assert pickle.loads(pickle.dumps(dumped)) == dumped
        await writing
        metadata = pq.read_table(directory / 'trajectory.parquet', columns=['routed_experts', 'multi_modal_items'])
        for name in metadata.column_names:
            assert all(value.startswith('ckpt://data/') for value in metadata[name].to_pylist() if value is not None)
        assert source._client.get_node_blobs([ref.record_id for ref in refs]) == original
        await source.release_many(refs)
        assert source._client.get_node_blobs([ref.record_id for ref in refs]) == [None] * len(refs)
    for unit in services.units:
        unit.stop()

    directory = directory.rename(tmp_path / 'moved-checkpoint')
    assert len(list((directory / 'data').glob('*.blob'))) == 3
    assert set(read_dump(directory, [dumped[1]], fields=['topk_values'])[0]['fields']) == {'topk_values'}
    script = """
import sys
sys.modules.update(dict.fromkeys(('redis', 'ray')))
from nexus.gateway.trajectory_store_backends.redis_store.checkpoint import read_dump
rows = read_dump(sys.argv[1], fields=['topk_values'])
assert rows[1]['fields']['topk_values'].shape == (5, 3)
assert str(rows[1]['fields']['topk_values'].dtype) == 'float32'
"""
    root = Path(__file__).resolve().parents[5]
    offline = subprocess.run(
        [sys.executable, '-c', script, str(directory)],
        cwd=root,
        env={**os.environ, 'PYTHONPATH': str(root)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert offline.returncode == 0, offline.stdout + offline.stderr
    with make_store(redis_backend_config, target_services.fs) as target:
        caplog.clear()
        restored = await target.load_payloads(dumped, dump_dir=str(directory), exclude_node_id='ignored-ray-node')
        assert 'exclude_node_id' in caplog.text
        assert 'ignor' in caplog.text.lower()
        assert sum('exclude_node_id' in record.message for record in caplog.records) == 1
        assert len(restored) == len(ordered)
        assert restored[1].record_id == restored[3].record_id
        assert restored[1] is not restored[3]
        assert all(ref.namespace == target.namespace for ref in restored)
        assert {ref.record_id for ref in restored}.isdisjoint(ref.record_id for ref in refs)
        assert_payload_equal(await target.get_many(restored), expected)
        assert isinstance(restored[1], HybridNodeRef)
        assert all(
            url.startswith(f'hykvfs://{target_services.topology.cluster_id}/')
            for ref in restored
            if isinstance(ref, HybridNodeRef)
            for url in ref.blob_urls.values()
        )
        caplog.clear()
        again = await target.load_payloads(dumped, dump_dir=str(directory))
        assert not any('exclude_node_id' in record.message for record in caplog.records)
        assert {ref.record_id for ref in again}.isdisjoint(ref.record_id for ref in restored)
        assert set(again[1].blob_urls.values()).isdisjoint(restored[1].blob_urls.values())
        assert_payload_equal(await target.get_many(again), expected)
        await target.release_many(restored + again)
    assert_payload_equal([row['fields'] for row in read_dump(directory, dumped)], expected)


def route_allocations_to_one_du(fs, monkeypatch):
    original = fs.allocate_url
    owner = fs.topology.data_units[0].data_unit_id

    def allocate():
        while True:
            url = original()
            if fs.topology.owner(url).data_unit_id == owner:
                return url

    monkeypatch.setattr(fs, 'allocate_url', allocate)


@pytest.mark.parametrize('direction', ['export', 'import'])
async def test_multiple_ceph_files_copy_concurrently_on_the_same_du(
    services, target_services, redis_backend_config, tmp_path, monkeypatch, direction
):
    if direction == 'export':
        route_allocations_to_one_du(services.fs, monkeypatch)
    route_allocations_to_one_du(target_services.fs, monkeypatch)
    with make_store(redis_backend_config, services.fs) as source:
        refs = await source.put_many([
            NodePayloadRecord(
                'trajectory', index, np.array([index], dtype=np.int32), routed_experts=np.arange(64) + index
            )
            for index in range(4)
        ])
        directory = tmp_path / 'parallel'
        dumped, writing = source.dump_payloads(refs, dump_dir=str(directory))
        if direction == 'import':
            await writing
        source_roots = {unit._ssd_dir for unit in services.units} if direction == 'export' else {directory / 'data'}
        entered, release = Event(), Event()
        counter_lock = Lock()
        active = maximum = 0
        original_copy = shutil.copyfile

        def copy(source_path, destination, *args, **kwargs):
            nonlocal active, maximum
            if Path(source_path).parent not in source_roots:
                return original_copy(source_path, destination, *args, **kwargs)
            with counter_lock:
                active += 1
                maximum = max(maximum, active)
                if active >= 2:
                    entered.set()
            try:
                assert release.wait(timeout=15), 'test did not release parallel file copies'
                return original_copy(source_path, destination, *args, **kwargs)
            finally:
                with counter_lock:
                    active -= 1

        monkeypatch.setattr(shutil, 'copyfile', copy)
        with make_store(redis_backend_config, target_services.fs) as target:
            operation = writing if direction == 'export' else target.load_payloads(dumped, dump_dir=str(directory))
            task = asyncio.create_task(operation)
            try:
                assert await asyncio.to_thread(entered.wait, 5), 'one DU did not overlap two file copies'
            finally:
                release.set()
                result = await task
            assert maximum >= 2
            if direction == 'import':
                assert_payload_equal(await target.get_many(result), await source.get_many(refs))
                await target.release_many(result)
        assert len(read_dump(directory, dumped)) == 4
        await source.release_many(refs)


async def test_cancel_load_drains_imports_and_reclaims_acknowledged_targets(
    target_services, redis_backend_config, tmp_path, monkeypatch
):
    directory = tmp_path / 'cancel'
    with make_store(redis_backend_config, target_services.fs) as source:
        refs = await source.put_many([
            NodePayloadRecord(
                'trajectory', index, np.array([index], dtype=np.int32), routed_experts=np.arange(64) + index
            )
            for index in range(4)
        ])
        dumped, writing = source.dump_payloads(refs, dump_dir=str(directory))
        await writing
        await source.release_many(refs)
    entered, release = Event(), Event()
    original_copy = shutil.copyfile

    def copy(source, destination, *args, **kwargs):
        if Path(source).parent == directory / 'data':
            entered.set()
            assert release.wait(timeout=15), 'test did not release the import copy'
        return original_copy(source, destination, *args, **kwargs)

    monkeypatch.setattr(shutil, 'copyfile', copy)
    with make_store(redis_backend_config, target_services.fs) as target:
        loading = asyncio.create_task(target.load_payloads(dumped, dump_dir=str(directory)))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            loading.cancel()
            await asyncio.sleep(0)
            assert not loading.done(), 'cancellation must drain actual file I/O'
        finally:
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await loading
        assert not list(target._client._r.scan_iter(match=f'{target._client.key_prefix}:node:*'))
        assert all(unit._snapshot_stats()['objects'] == 0 for unit in target_services.units)
        assert all(not list(unit._ssd_dir.iterdir()) for unit in target_services.units)
    assert len(read_dump(directory, dumped)) == 4


def test_stop_waits_for_actual_path_import_before_removing_local_ssd(target_services, tmp_path, monkeypatch):
    url = target_services.fs.allocate_url()
    owner = target_services.topology.owner(url).data_unit_id
    unit = next(unit for unit in target_services.units if unit.data_unit_id == owner)
    source = tmp_path / 'shared.blob'
    source.write_bytes(encode_blob(np.arange(64)))
    entered, release, stopping_started = Event(), Event(), Event()
    original_copy = shutil.copyfile

    def copy(source_path, destination, *args, **kwargs):
        if Path(source_path) == source:
            entered.set()
            assert release.wait(timeout=15), 'test did not release the import copy'
        return original_copy(source_path, destination, *args, **kwargs)

    def stop():
        stopping_started.set()
        unit.stop()

    monkeypatch.setattr(shutil, 'copyfile', copy)
    with ThreadPoolExecutor(max_workers=2) as executor:
        importing = executor.submit(target_services.fs.import_file_to_ssd, url, str(source))
        stopping = None
        try:
            assert entered.wait(timeout=5)
            stopping = executor.submit(stop)
            assert stopping_started.wait(timeout=5)
            assert not stopping.done()
        finally:
            release.set()
            importing.result(timeout=5)
            if stopping is not None:
                stopping.result(timeout=5)
    assert source.exists()
    assert unit._ssd_dir is None
