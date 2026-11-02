"""Logical deletion preserves a dump's protected metadata until preparation finishes."""

import asyncio
import threading

from binascii import crc_hqx
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack

import pytest

from tests.unit.gateway.redis_store.fake_redis import FakeRedis
from tests.unit.gateway.redis_store.file_storage.test_publication import MemoryFS, record
from tests.unit.gateway.redis_store.payloads import SamplePayload

from nexus.gateway.trajectory_store_backends.redis import RedisTrajectoryRecordStore
from nexus.gateway.trajectory_store_backends.redis_store import checkpoint
from nexus.gateway.trajectory_store_backends.redis_store.checkpoint import dump as checkpoint_dump
from nexus.gateway.trajectory_store_backends.redis_store.client import RedisClient
from nexus.gateway.trajectory_store_backends.redis_store.dump_protection import (
    deferred_record_key,
    dump_owners_key,
    protect_dump_records,
)
from nexus.gateway.trajectory_store_backends.redis_store.types import RedisClientError
from nexus.gateway.trajectory_store_backends.types import SegmentDataRecord


@pytest.fixture
def client():
    with RedisClient(redis_client=FakeRedis(), fs_client=MemoryFS()) as instance:
        for node_id in ('a', 'b'):
            instance.put_node(SamplePayload(text=node_id), node_id=node_id)
        yield instance


@pytest.mark.parametrize(
    'record_key',
    [
        b'prefix:node:a',
        b'{namespace}:node:a',
        b'prefix:{nested{tag}:node:a',
        b'{}:node:a',
        b'open{:node:a',
        b'}:node:a',
    ],
)
def test_dump_owners_key_has_same_cluster_slot_and_keeps_record_identity(record_key):
    def slot(key):
        start = key.find(b'{')
        end = key.find(b'}', start + 1) if start >= 0 else -1
        return crc_hqx(key[start + 1 : end] if end > start + 1 else key, 0) % 16384

    assert slot(dump_owners_key(record_key)) == slot(deferred_record_key(record_key)) == slot(record_key)
    assert dump_owners_key(record_key) != dump_owners_key(record_key + b'b')
    assert deferred_record_key(record_key) != deferred_record_key(record_key + b'b')
    assert deferred_record_key(record_key) != dump_owners_key(record_key)


def test_protection_is_shared_between_dump_owners_and_partial_release_is_idempotent(client):
    raw = client._r
    key_a, key_b = (dump_owners_key(client._node_key(node_id)) for node_id in ('a', 'b'))
    node_ids = ['b', 'a', 'a']
    with protect_dump_records(client, node_ids, 'first') as first:
        node_ids.clear()
        with protect_dump_records(client, ['a'], 'second'):
            assert raw._dump_protections == {key_a: {'first', 'second'}, key_b: {'first'}}
            first.release_records(['a', 'a', 'unheld'])
            first.release_records(['a'])
            assert raw._dump_protections == {key_a: {'second'}, key_b: {'first'}}
        assert raw._dump_protections == {key_b: {'first'}}
    assert raw._dump_protections == {}


def test_missing_record_drains_successful_registration_siblings(client):
    with pytest.raises(RedisClientError, match='missing'):
        with protect_dump_records(client, ['a', 'missing', 'b'], 'dump'):
            pytest.fail('missing source cannot enter dump')
    assert client._r._dump_protections == {}


def test_final_release_pipeline_lost_ack_still_releases_later_batches(client, monkeypatch):
    node_ids = [f'node-{index}' for index in range(515)]
    for node_id in node_ids:
        client._r.set(client._node_key(node_id), b'payload')
    execute = client._publication.execute_pipeline
    sizes = []

    def lose_first_release_reply(pipe, **options):
        releasing = pipe._ops[0][1][1].startswith('-- nexus:dump-release:1')
        sizes.append(len(pipe._ops))
        result = execute(pipe, **options)
        if releasing and len(sizes) == 4:
            raise ConnectionError('first release batch ACK lost')
        return result

    monkeypatch.setattr(client._publication, 'execute_pipeline', lose_first_release_reply)
    with ExitStack() as stack:
        lease = stack.enter_context(protect_dump_records(client, node_ids, 'dump'))
        assert len(client._r._dump_protections) == len(node_ids)
        assert sizes == [256, 256, 3]
        with pytest.raises(ConnectionError, match='first release batch ACK lost'):
            stack.close()
    assert sizes == [256, 256, 3, 256, 256, 3]
    assert lease._pending_release_keys == set(sorted(client._node_key(node_id) for node_id in node_ids)[:256])
    assert client._r._dump_protections == {}
    lease.release_records(node_ids)
    assert not lease._pending_release_keys
    assert sizes == [256, 256, 3, 256, 256, 3, 256]


def test_registration_pipeline_lost_ack_cleans_every_attempted_key(client, monkeypatch):
    execute = client._publication.execute_pipeline
    calls = 0

    def lose_reply(pipe, **options):
        nonlocal calls
        calls += 1
        result = execute(pipe, **options)
        if calls == 1:
            raise ConnectionError('registration ACK lost')
        return result

    monkeypatch.setattr(client._publication, 'execute_pipeline', lose_reply)
    with pytest.raises(ConnectionError, match='registration ACK lost'):
        with protect_dump_records(client, ['a', 'b'], 'dump'):
            pytest.fail('unknown registration cannot enter dump')
    assert calls == 2
    assert client._r._dump_protections == {}


def test_lost_release_ack_retains_only_unknown_keys_and_cannot_release_new_owner(client, monkeypatch):
    raw = client._r
    command = raw.execute_command
    key_a = dump_owners_key(client._node_key('a'))
    lost = False

    def lose_reply(name, *arguments):
        nonlocal lost
        result = command(name, *arguments)
        if name == 'EVAL' and arguments[0].startswith('-- nexus:dump-release:1') and arguments[2] == key_a and not lost:
            lost = True
            raise ConnectionError('release ACK lost')
        return result

    monkeypatch.setattr(raw, 'execute_command', lose_reply)
    with ExitStack() as second:
        with protect_dump_records(client, ['a', 'b'], 'first') as lease:
            with pytest.raises(ConnectionError, match='release ACK lost'):
                lease.release_records(['a', 'b'])
            assert lease._pending_release_keys == {client._node_key('a')}
            second.enter_context(protect_dump_records(client, ['a'], 'second'))
        assert raw._dump_protections == {key_a: {'second'}}
    assert raw._dump_protections == {}


def test_protected_delete_returns_before_dump_releases_metadata(client):
    with ThreadPoolExecutor(max_workers=1) as pool:
        with protect_dump_records(client, ['a'], 'dump'):
            pending = pool.submit(client.delete_multi_node, ['a'])
            completed = threading.Event()
            pending.add_done_callback(lambda _: completed.set())
            returned_while_protected = completed.wait(1)
        pending.result(timeout=2)
    assert returned_while_protected
    assert client.get_node_blobs(['a']) == [None]


@pytest.mark.parametrize('use_hint', [False, True])
def test_only_last_owner_reclaims_metadata_and_all_canonical_blobs(client, use_hint):
    result = client._publication.put([record()])[0]
    node_id = result.node_id
    key = client._node_key(node_id)
    original = client._r.get(key)
    urls = tuple(result.blob_urls.values())
    options = {'cleanup_urls_by_record': {node_id: urls}} if use_hint else {}
    with protect_dump_records(client, [node_id], 'first') as first:
        with protect_dump_records(client, [node_id], 'second') as second:
            client.delete_multi_node([node_id, node_id])
            assert not client.has_node(node_id)[0]
            assert client._r.get(deferred_record_key(key)) == original
            assert set(urls) <= client._publication._fs.blobs.keys()
            with pytest.raises(RedisClientError, match='not found'):
                with protect_dump_records(client, [node_id], 'late'):
                    pytest.fail('a new dump cannot claim logically deleted metadata')
            first.release_records([node_id], **options)
            assert client._r.get(deferred_record_key(key)) == original
            assert set(urls) <= client._publication._fs.blobs.keys()
            second.release_records([node_id], **options)
            assert client._r.get(deferred_record_key(key)) is None
            assert not client._publication._fs.blobs
            second.release_records([node_id], **options)


def test_failed_cleanup_keeps_deferred_record_for_explicit_release_retry(client, monkeypatch):
    result = client._publication.put([record()])[0]
    fs = client._publication._fs
    original_delete = fs.delete
    attempted = []

    def fail(url):
        attempted.append(url)
        raise OSError('SSD deletion failed')

    with protect_dump_records(client, [result.node_id], 'dump') as lease:
        client.delete_multi_node([result.node_id])
        monkeypatch.setattr(fs, 'delete', fail)
        with pytest.raises(OSError, match='SSD deletion failed'):
            lease.release_records([result.node_id])
        assert not lease._pending_release_keys
        assert client._r.get(deferred_record_key(client._node_key(result.node_id))) is not None
    assert attempted == [result.blob_urls['routed_experts']]
    monkeypatch.setattr(fs, 'delete', original_delete)
    client.delete_multi_node([result.node_id])
    assert not fs.blobs
    assert client._r.get(deferred_record_key(client._node_key(result.node_id))) is None
    client.delete_multi_node([result.node_id])


@pytest.mark.parametrize('phase', ['delete', 'release'])
def test_unknown_ack_retains_deferred_metadata_until_release_is_confirmed(client, monkeypatch, phase):
    result = client._publication.put([record()])[0]
    key = client._node_key(result.node_id)
    command = client._r.execute_command
    lost = False

    def lose_reply(name, *arguments):
        nonlocal lost
        answer = command(name, *arguments)
        release = name == 'EVAL' and arguments[0].startswith('-- nexus:dump-release:1')
        deletion = name == 'EVAL' and arguments[1] == 3
        if not lost and (release if phase == 'release' else deletion):
            lost = True
            raise ConnectionError(f'{phase} ACK lost')
        return answer

    with protect_dump_records(client, [result.node_id], 'dump') as lease:
        monkeypatch.setattr(client._r, 'execute_command', lose_reply)
        if phase == 'release':
            client.delete_multi_node([result.node_id])
        operation = lease.release_records if phase == 'release' else client.delete_multi_node
        with pytest.raises(ConnectionError, match=f'{phase} ACK lost'):
            operation([result.node_id])
        assert client._r.get(deferred_record_key(key)) is not None
        assert set(result.blob_urls.values()) <= client._publication._fs.blobs.keys()
        assert lease._pending_release_keys == {key}
    assert client._r.get(deferred_record_key(key)) is None
    assert not client._publication._fs.blobs


def test_lost_final_unlink_ack_is_not_replayed_and_retry_preserves_new_records(client, monkeypatch):
    [old] = client._publication.put([record()])
    command = client._r.execute_command
    unlinked = []

    def lose_reply(name, *arguments):
        answer = command(name, *arguments)
        if name == 'UNLINK':
            unlinked.append(arguments[0])
            raise ConnectionError('UNLINK ACK lost')
        return answer

    with protect_dump_records(client, [old.node_id], 'dump') as lease:
        client.delete_multi_node([old.node_id])
        monkeypatch.setattr(client._r, 'execute_command', lose_reply)
        with pytest.raises(ConnectionError, match='UNLINK ACK lost'):
            lease.release_records([old.node_id])
        assert not lease._pending_release_keys
    [new] = client._publication.put([record()])
    client.delete_multi_node([old.node_id])
    assert unlinked == [deferred_record_key(client._node_key(old.node_id))]
    assert set(new.blob_urls.values()) == client._publication._fs.blobs.keys()
    assert client.has_node(new.node_id)[0]


@pytest.mark.parametrize('forged_first', [True, False])
def test_deferred_reference_validation_precedes_explicit_cleanup(client, forged_first):
    left, right = client._publication.put([record(), record()])
    with protect_dump_records(client, [left.node_id], 'dump'):
        client.delete_multi_node([left.node_id])
        references = [right.blob_urls, left.blob_urls] if forged_first else [left.blob_urls, right.blob_urls]
        with pytest.raises(RedisClientError) as caught:
            client._publication.delete([left.node_id, left.node_id], reference_urls=references)
        assert caught.value.code == 'invalid_reference'
        assert len(client._publication._fs.blobs) == 4
        client.delete_multi_node([left.node_id])
        assert len(client._publication._fs.blobs) == 4
    assert set(right.blob_urls.values()) == client._publication._fs.blobs.keys()


async def test_snapshot_reads_deferred_metadata_and_cleans_unselected_blob(client):
    with RedisTrajectoryRecordStore('run', {'redis_client': client._r, 'fs_client': client._publication._fs}) as store:
        [ref] = await store.put_many([record()])
        with protect_dump_records(store._client, [ref.record_id], 'dump'):
            await store.release_many([ref])
            snapshot = checkpoint_dump._read_dump_batch(store._client, [ref], 0, {'token_ids', 'routed_experts'})
            rows, urls = snapshot[:2]
            assert rows[0]['token_ids'] == [1, 2]
            assert rows[0]['routed_experts'].startswith('ckpt://data/')
            assert 'multi_modal_items' not in rows[0]
            assert urls == {ref.routed_experts_url}
            assert len(client._publication._fs.blobs) == 2
        assert not client._publication._fs.blobs


async def test_segment_cas_can_advance_after_dump_captures_old_snapshot(tmp_path, monkeypatch):
    raw = FakeRedis()
    with RedisTrajectoryRecordStore('run', {'redis_client': raw, 'fs_client': MemoryFS()}) as store:
        [ref] = await store.put_many([SegmentDataRecord('run', 'segment', {'old': 1})])
        snapshot_ready, resume_dump = threading.Event(), threading.Event()
        snapshot = checkpoint_dump._read_dump_batch

        def capture(*arguments):
            result = snapshot(*arguments)
            snapshot_ready.set()
            assert resume_dump.wait(5)
            return result

        monkeypatch.setattr(checkpoint_dump, '_read_dump_batch', capture)
        dumped, writing = store.dump_payloads([ref], dump_dir=str(tmp_path))
        task = asyncio.create_task(writing)
        try:
            assert await asyncio.to_thread(snapshot_ready.wait, 2)
            await store.put_fields_many({ref: {'late': 2}})
            assert await store.get_many([ref]) == [{'old': 1, 'late': 2}]
        finally:
            resume_dump.set()
            await task
        assert checkpoint.read_dump(tmp_path, dumped)[0]['fields'] == {'old': 1}
    assert raw._dump_protections == {}
