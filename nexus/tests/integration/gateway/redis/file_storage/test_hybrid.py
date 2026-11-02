"""Hybrid records over real Redis/Redis Cluster and TCP blob services."""

import asyncio
import json
import pickle
import threading

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from uuid import uuid4

import numpy as np
import pytest

from nexus.gateway.trajectory_store_backends.redis import (
    HybridNodeRef,
    NodeRecord,
    RedisTrajectoryRecordStore,
    SegmentRecord,
)
from nexus.gateway.trajectory_store_backends.redis_store.client import RedisClient
from nexus.gateway.trajectory_store_backends.redis_store.file_storage.publication import StoredNode
from nexus.gateway.trajectory_store_backends.redis_store.types import RedisClientError


def record():
    return NodeRecord('run', 1, [1, 2], routed_experts=b'large', multi_modal_items={})


def test_inline_record_bytes_and_offloaded_roundtrip(client):
    inline = replace(record(), routed_experts=None, multi_modal_items=None)
    old = uuid4().hex
    client._r.set(client._node_key(old), inline.serialize())
    new = client.put_node(inline)
    assert isinstance(new, str)
    for node_id in (old, new):
        restored = client.get_node(node_id, payload_type=NodeRecord)
        assert type(restored) is NodeRecord
        np.testing.assert_equal(vars(restored), vars(inline))
    client.delete_multi_node([old, new])
    node_id = client.put_node(record())
    raw = StoredNode.deserialize(client.get_node_blobs([node_id])[0])
    assert NodeRecord.deserialize(raw.record).routed_experts is None
    assert NodeRecord.deserialize(raw.record).multi_modal_items is None
    restored = client.get_node(pickle.loads(pickle.dumps(node_id)), payload_type=NodeRecord)
    assert type(restored) is NodeRecord
    np.testing.assert_equal(vars(restored), vars(record()))
    client.delete_multi_node([node_id])
    client._r.set(client._node_key(old), b'undecodable')
    client.delete_node(old)


def test_plain_redis_url_supports_hybrid_publication_and_release(client, redis_backend_config):
    async def run():
        config = {
            **redis_backend_config,
            'url': redis_backend_config['url'].replace('redis+cluster://', 'redis://', 1),
            'fs_client': client._publication._fs,
            'key_prefix': client.key_prefix,
        }
        with RedisTrajectoryRecordStore('run', config) as store:
            ref = (await store.put_many([record()]))[0]
            assert isinstance(ref, HybridNodeRef)
            assert ref.routed_experts_url is not None
            assert (await store.get_many([ref]))[0]['routed_experts'] == b'large'
            await store.release_many([ref])
            assert not store._client.has_node(ref.record_id)[0]

    asyncio.run(run())


def test_redis_and_blobs_get_start_in_parallel(client, monkeypatch):
    payload = replace(record(), multi_modal_items=None)
    result = client._publication.put([payload])[0]
    redis_call = client.get_node_blobs
    fs_call = client._publication._fs.get
    barrier = threading.Barrier(2, timeout=5)

    def redis_wait(*args, **kwargs):
        barrier.wait()
        return redis_call(*args, **kwargs)

    def fs_wait(*args, **kwargs):
        barrier.wait()
        return fs_call(*args, **kwargs)

    monkeypatch.setattr(client, 'get_node_blobs', redis_wait)
    monkeypatch.setattr(client._publication._fs, 'get', fs_wait)
    (restored,) = client._publication.get([(result.node_id, NodeRecord)], reference_urls=[result.blob_urls])
    assert type(restored) is NodeRecord
    np.testing.assert_equal(vars(restored), vars(payload))


def test_blob_delete_waits_for_redis_deletion_ack(client, monkeypatch):
    result = client._publication.put([record()])[0]
    publisher = client._publication
    redis_deleted, release_ack, redis_acked, blob_started = (threading.Event() for _ in range(4))
    redis_call, fs_call = publisher.execute_pipeline, publisher._fs.delete
    deleted_urls = set()

    def redis_wait(*args, **kwargs):
        outcomes = redis_call(*args, **kwargs)
        redis_deleted.set()
        assert release_ack.wait(5)
        redis_acked.set()
        return outcomes

    def fs_delete(url):
        blob_started.set()
        assert redis_acked.is_set(), 'DU deletion must wait for the Redis deletion ACK'
        deleted_urls.add(url)
        return fs_call(url)

    monkeypatch.setattr(publisher, 'execute_pipeline', redis_wait)
    monkeypatch.setattr(publisher._fs, 'delete', fs_delete)
    with ThreadPoolExecutor(max_workers=1) as pool:
        deletion = pool.submit(client.delete_multi_node, [result.node_id])
        try:
            assert redis_deleted.wait(5)
            assert not client.has_node(result.node_id)[0]
            assert not blob_started.wait(0.05)
            assert not deletion.done()
        finally:
            release_ack.set()
        deletion.result(timeout=5)
    assert deleted_urls == set(result.blob_urls.values())


def test_trajectory_projection_append_and_release(client, redis_backend_config, monkeypatch):
    async def run():
        config = {**redis_backend_config, 'fs_client': client._publication._fs, 'key_prefix': client.key_prefix}
        with RedisTrajectoryRecordStore('run', config) as store:
            node, segment = await store.put_many([record(), SegmentRecord('run', 's', {'reward': 1})])
            assert isinstance(node, HybridNodeRef)
            node = HybridNodeRef(**json.loads(json.dumps(asdict(node))))
            assert (await store.get_many([node]))[0]['routed_experts'] == b'large'

            def forbidden(*args):
                raise AssertionError('projection must not read the blob')

            with monkeypatch.context() as patch:
                patch.setattr(store._client._publication._fs, 'get', forbidden)
                assert await store.get_many([node], fields=['token_ids']) == [{'token_ids': [1, 2]}]
            await store.put_fields_many({segment: {'advantage': [2]}})
            assert await store.get_many([segment]) == [{'reward': 1, 'advantage': [2]}]
            await store.release_many([node, segment, node])
            await store.release_many([node, segment])

    asyncio.run(run())


def test_writing_record_cannot_be_read_or_deleted(client, monkeypatch, redis_backend_config):
    started, release = threading.Event(), threading.Event()
    original_put = client._publication._fs.put
    node_id = uuid4().hex

    def blocked_put(*args):
        started.set()
        assert release.wait(5)
        return original_put(*args)

    monkeypatch.setattr(client._publication._fs, 'put', blocked_put)
    with RedisClient(**redis_backend_config, fs_client=client._publication._fs, key_prefix=client.key_prefix) as other:
        with ThreadPoolExecutor(max_workers=1) as pool:
            writing = pool.submit(client.put_node, record(), node_id=node_id)
            try:
                assert started.wait(5)
                raw = StoredNode.deserialize(other.get_node_blobs([node_id])[0])
                assert raw.state == 'writing'
                for urls in (None, [raw.blob_urls]):
                    for operation in (
                        lambda urls=urls: other._publication.get([(node_id, NodeRecord)], reference_urls=urls),
                        lambda urls=urls: other._publication.delete([node_id], reference_urls=urls),
                    ):
                        with pytest.raises(RedisClientError) as error:
                            operation()
                        assert error.value.code == 'write_in_progress'
                assert other.has_node(node_id)[0]
            finally:
                release.set()
            assert writing.result(timeout=5) == node_id
        restored = other.get_node(node_id, payload_type=NodeRecord)
        assert type(restored) is NodeRecord
        np.testing.assert_equal(vars(restored), vars(record()))
        other.delete_multi_node([node_id])


def test_early_missing_blob_is_reread_after_publication(client, monkeypatch):
    result = client._publication.put([record()])[0]
    original = client._publication._fs.get
    calls = set()

    def early(url):
        if url not in calls:
            calls.add(url)
            raise RedisClientError('early read', code='not_found')
        return original(url)

    monkeypatch.setattr(client._publication._fs, 'get', early)
    (restored,) = client._publication.get([(result.node_id, NodeRecord)], reference_urls=[result.blob_urls])
    assert type(restored) is NodeRecord
    np.testing.assert_equal(vars(restored), vars(record()))
    client.delete_multi_node([result.node_id])


def test_redis_command_error_preserves_other_record(client, monkeypatch):
    a, b = uuid4().hex, uuid4().hex
    invalid = client._node_key(a)
    client._r.set(invalid, b'not-a-hash')
    pipeline = client._r.pipeline

    def broken_pipeline(*args, **kwargs):
        pipe = pipeline(*args, **kwargs)
        original_set = pipe.set

        def set_or_fail(key, value, **options):
            return pipe.hset(key, 'field', 'value') if key == invalid else original_set(key, value, **options)

        pipe.set = set_or_fail
        return pipe

    with monkeypatch.context() as patch:
        patch.setattr(client._r, 'pipeline', broken_pipeline)
        with pytest.raises(RedisClientError) as caught:
            client.put_multi_node([record(), SegmentRecord('run', 's', {'reward': 1})], node_ids=[a, b])
    assert caught.value.body['partial_ids'] == [None, b]
    assert 'WRONGTYPE' in caught.value.body['errors'][0]
    assert isinstance(client.get_node(b, payload_type=SegmentRecord), SegmentRecord)
    assert all(conn.request('stats')[0]['objects'] == 0 for conn in client._publication._fs._units.values())
    client._r.delete(invalid)
    client.delete_multi_node([b])


@pytest.mark.parametrize('stage', ['reservation', 'commit', 'delete'])
def test_lost_redis_reply_is_not_replayed_by_driver(client, monkeypatch, stage):
    """Exercise redis-py's own retry loop, after the server applied the command."""
    import redis

    from redis.client import Pipeline
    from redis.cluster import NodeCommands

    node_id = uuid4().hex
    if stage == 'delete':
        client.put_node(record(), node_id=node_id)
        blob_urls = StoredNode.deserialize(client.get_node_blobs([node_id])[0]).blob_urls
    key = client._node_key(node_id)
    lost = []

    def matches(args):
        command = 'SET' if stage == 'reservation' else 'EVAL'
        return args[0] == command and key in args

    if client.url.startswith('redis+cluster://'):
        original = NodeCommands.read

        def read(commands):
            original(commands)
            for command in commands.commands:
                if matches(command.args) and not lost:
                    lost.append(True)
                    command.result = redis.ConnectionError('lost reply after application')

        monkeypatch.setattr(NodeCommands, 'read', read)
    else:
        method = '_execute_transaction' if stage == 'reservation' else '_execute_pipeline'
        original = getattr(Pipeline, method)

        def execute(pipe, connection, commands, raise_on_error):
            result = original(pipe, connection, commands, raise_on_error)
            if any(matches(args) for args, _ in commands) and not lost:
                lost.append(True)
                raise redis.ConnectionError('lost reply after application')
            return result

        monkeypatch.setattr(Pipeline, method, execute)
    if stage == 'delete':
        with pytest.raises(redis.ConnectionError, match='lost reply after application'):
            client.delete_node(node_id)
        assert client.get_node_blobs([node_id]) == [None]
        assert all(client._publication._fs.get(url) for url in blob_urls.values())
    else:
        with pytest.raises(RedisClientError) as caught:
            client.put_node(record(), node_id=node_id)
        assert caught.value.code == 'publication_failed'
        assert 'lost reply after application' in caught.value.body['errors'][0]
        stored = StoredNode.deserialize(client.get_node_blobs([node_id])[0])
        assert stored.state == ('writing' if stage == 'reservation' else 'committed')
    assert lost == [True]
