"""Parquet exports remain readable after their live Redis/DU sources are deleted."""

import asyncio
import shutil

from pathlib import Path
from threading import Event
from uuid import uuid4

import numpy as np
import pytest

from nexus.gateway.trajectory_store_backends.redis import RedisTrajectoryRecordStore
from nexus.gateway.trajectory_store_backends.redis_store.checkpoint import dump as dump_module
from nexus.gateway.trajectory_store_backends.redis_store.checkpoint import read_dump
from nexus.gateway.trajectory_store_backends.redis_store.dump_protection import deferred_record_key, dump_owners_key
from nexus.gateway.trajectory_store_backends.redis_store.types import RedisClientError
from nexus.gateway.trajectory_store_backends.types import NodePayloadRecord


def store(services, redis_backend_config):
    return RedisTrajectoryRecordStore(
        'checkpoint-test',
        {**redis_backend_config, 'fs_client': services.fs, 'key_prefix': 'checkpoint-' + uuid4().hex},
    )


@pytest.mark.parametrize('prepare_first', [False, True], ids=['delete-first', 'prepare-first'])
async def test_unknown_dump_preparation_unprotects_delete_and_leaves_pin_cleanup_to_du(
    services, redis_backend_config, tmp_path, monkeypatch, prepare_first
):
    original_prepare, original_finish = services.fs.prepare_dump_sources, services.fs.finalize_dump
    pending = []

    def prepare(dump_id, urls):
        pending.append((dump_id, urls))
        if prepare_first:
            original_prepare(dump_id, urls)
        raise RedisClientError('prepare reply is unknown', code='timeout')

    def finish(dump_id):
        raise RedisClientError('finish reply is unknown', code='timeout')

    monkeypatch.setattr(services.fs, 'prepare_dump_sources', prepare)
    monkeypatch.setattr(services.fs, 'finalize_dump', finish)
    with store(services, redis_backend_config) as writer:
        with RedisTrajectoryRecordStore('peer', writer.config) as peer:
            [ref] = await writer.put_many([
                NodePayloadRecord('trajectory', 1, np.array([1], dtype=np.int32), routed_experts=np.arange(128))
            ])
            directory = tmp_path / 'export'
            _, writing = writer.dump_payloads([ref], dump_dir=str(directory))
            guard_key = dump_owners_key(writer._client._node_key(ref.record_id))
            try:
                with pytest.raises(RedisClientError, match='reply is unknown'):
                    await writing
                assert not writer._client._r.exists(guard_key), 'failed dump must release its deletion protection'
                assert not (directory / 'trajectory.parquet').exists()
                [(dump_id, urls)] = pending
                unit_id = services.fs.topology.owner(ref.routed_experts_url).data_unit_id
                unit = next(unit for unit in services.units if unit.data_unit_id == unit_id)
                if prepare_first:
                    with unit._metadata_lock:
                        entry = unit._store[ref.routed_experts_url]
                        source = entry.disk_path
                        assert entry.dump_pin_count == 1
                await asyncio.wait_for(peer.release_many([ref]), timeout=3)
                assert not writer._client.has_node(ref.record_id)[0]
                if prepare_first:
                    assert source.exists(), 'unknown finish must preserve the pinned source'
                    original_finish(dump_id)
                    assert not source.exists()
                    assert entry.dump_pin_count == 0
                else:
                    with pytest.raises(RedisClientError) as error:
                        original_prepare(dump_id, urls)
                    assert error.value.code == 'not_found'
            finally:
                # A failing regression must not leave a deletion guard or a DU pin behind.
                writer._client._r.delete(guard_key)
                for dump_id, _ in pending:
                    original_finish(dump_id)


async def test_prepared_batch_delete_before_parquet_write_preserves_pinned_source(
    services, redis_backend_config, tmp_path, monkeypatch
):
    entered, resume = Event(), Event()
    original_write = dump_module._write_parquet_batch

    def write(writer, schema, rows):
        if rows[0]['entry_id'] == 0:
            entered.set()
            assert resume.wait(timeout=15), 'test must release the Parquet writer'
        return original_write(writer, schema, rows)

    monkeypatch.setattr(dump_module, 'PARQUET_BATCH_SIZE', 1)
    monkeypatch.setattr(dump_module, '_write_parquet_batch', write)
    with store(services, redis_backend_config) as writer:
        with RedisTrajectoryRecordStore('peer', writer.config) as peer:
            refs = await writer.put_many([
                NodePayloadRecord('trajectory', i, np.array([i], dtype=np.int32), routed_experts=np.arange(128) + i)
                for i in range(2)
            ])
            directory = tmp_path / 'export'
            dumped, writing = writer.dump_payloads(refs, dump_dir=str(directory))
            task = asyncio.create_task(writing)
            deletion = None
            try:
                assert await asyncio.to_thread(entered.wait, 10), 'first batch must reach Parquet encoding'
                assert writer._client._r.exists(dump_owners_key(writer._client._node_key(refs[1].record_id)))
                url = refs[0].routed_experts_url
                owner = services.fs.topology.owner(url).data_unit_id
                unit = next(unit for unit in services.units if unit.data_unit_id == owner)
                with unit._metadata_lock:
                    entry = unit._store[url]
                    source = entry.disk_path
                    assert entry.dump_pin_count == 1
                # Threaded facade cancellation drains work, so let the writer resume on timeout.
                deletion = asyncio.create_task(peer.release_many([refs[0]]))
                done, _ = await asyncio.wait([deletion], timeout=3)
                assert deletion in done, 'prepared batch must be unprotected before Parquet encoding'
                await deletion
                assert source.exists(), 'Pin must preserve the deleted source before Ceph copy starts'
                assert not writer._client.has_node(refs[0].record_id)[0]
                assert not (directory / 'trajectory.parquet').exists()
            finally:
                resume.set()
                await asyncio.gather(*([task] if deletion is None else [task, deletion]))
            assert not source.exists()
            rows = read_dump(directory, dumped)
            for index, row in enumerate(rows):
                np.testing.assert_array_equal(row['fields']['routed_experts'], np.arange(128) + index)


async def test_ceph_copy_does_not_hold_deletion_protection_or_delay_delete(
    services, redis_backend_config, tmp_path, monkeypatch
):
    entered, release = Event(), Event()
    sources = []
    original_copy = shutil.copyfile
    local_roots = {unit._ssd_dir for unit in services.units}

    def slow_copy(source, destination, *args, **kwargs):
        if Path(source).parent in local_roots:
            sources.append(Path(source))
            entered.set()
            assert release.wait(timeout=15), 'test must release the copy worker'
        return original_copy(source, destination, *args, **kwargs)

    monkeypatch.setattr(shutil, 'copyfile', slow_copy)
    with store(services, redis_backend_config) as writer:
        peer = RedisTrajectoryRecordStore('other-client', writer.config)
        try:
            [ref] = await writer.put_many([
                NodePayloadRecord('trajectory', 1, np.array([1], dtype=np.int32), routed_experts=np.arange(128))
            ])
            dump_dir = tmp_path / 'export'
            dumped, writing = writer.dump_payloads([ref], dump_dir=str(dump_dir))
            task = asyncio.create_task(writing)
            try:
                assert await asyncio.to_thread(entered.wait, 10), 'DU must start its copy'
                await asyncio.wait_for(peer.release_many([ref]), timeout=3)
                assert all(path.exists() for path in sources), 'Pin must preserve logically deleted SSD sources'
                assert not (dump_dir / 'trajectory.parquet').exists()
                [unrelated] = await asyncio.wait_for(
                    peer.put_many([NodePayloadRecord('other', 1, np.array([2], dtype=np.int32))]), timeout=3
                )
                await peer.release_many([unrelated])
            finally:
                release.set()
                await task
            assert all(not path.exists() for path in sources)
            np.testing.assert_array_equal(read_dump(dump_dir, dumped)[0]['fields']['routed_experts'], np.arange(128))
        finally:
            peer.close()


@pytest.mark.parametrize('before_read', [False, True], ids=['before-pin', 'before-read'])
async def test_release_returns_with_deferred_metadata_until_dump_has_pinned_source(
    services, redis_backend_config, tmp_path, monkeypatch, before_read
):
    preparing, resume, deleting_blob = Event(), Event(), Event()
    original_prepare, original_delete = services.fs.prepare_dump_sources, services.fs.delete
    original_snapshot = dump_module._read_dump_batch

    def prepare(dump_id, urls):
        preparing.set()
        assert resume.wait(15), 'test must allow SSD preparation'
        return original_prepare(dump_id, urls)

    def delete(url):
        deleting_blob.set()
        return original_delete(url)

    def snapshot(*args):
        preparing.set()
        assert resume.wait(15), 'test must allow metadata capture'
        return original_snapshot(*args)

    if before_read:
        monkeypatch.setattr(dump_module, '_read_dump_batch', snapshot)
    else:
        monkeypatch.setattr(services.fs, 'prepare_dump_sources', prepare)
    monkeypatch.setattr(services.fs, 'delete', delete)
    with store(services, redis_backend_config) as writer:
        with RedisTrajectoryRecordStore('peer', writer.config) as peer:
            [ref] = await writer.put_many([
                NodePayloadRecord(
                    'trajectory', 1, np.array([1], dtype=np.int32), routed_experts=np.arange(128), multi_modal_items={}
                )
            ])
            url = ref.routed_experts_url
            unit_id = services.fs.topology.owner(url).data_unit_id
            unit = next(unit for unit in services.units if unit.data_unit_id == unit_id)
            dump_dir = tmp_path / 'export'
            key = writer._client._node_key(ref.record_id)
            [raw] = writer._client.get_node_blobs([ref.record_id])
            dumped, writing = writer.dump_payloads(
                [ref], dump_dir=str(dump_dir), fields=['token_ids', 'routed_experts']
            )
            task, deletion = asyncio.create_task(writing), None
            try:
                assert await asyncio.to_thread(preparing.wait, 10)
                deletion = asyncio.create_task(peer.release_many([ref]))
                done, _ = await asyncio.wait([deletion], timeout=3)
                assert deletion in done, 'logical release must not wait for dump preparation'
                await deletion
                assert not deleting_blob.is_set(), 'DU deletion must wait for the last dump preparation'
                assert not writer._client.has_node(ref.record_id)[0]
                assert writer._client._r.get(deferred_record_key(key)) == raw
                with unit._metadata_lock:
                    assert url in unit._store
                    assert not unit._store[url].logically_deleted
                    assert unit._store[url].dump_pin_count == 0
                peer.close()
            finally:
                resume.set()
                await asyncio.gather(*([task] if deletion is None else [task, deletion]))
            assert deleting_blob.is_set()
            assert not writer._client.has_node(ref.record_id)[0]
            assert not writer._client._r.exists(deferred_record_key(key))
            for blob_url in (ref.routed_experts_url, ref.multi_modal_items_url):
                with pytest.raises(RedisClientError) as error:
                    services.fs.get(blob_url)
                assert error.value.code == 'not_found'
            np.testing.assert_array_equal(read_dump(dump_dir, dumped)[0]['fields']['routed_experts'], np.arange(128))
