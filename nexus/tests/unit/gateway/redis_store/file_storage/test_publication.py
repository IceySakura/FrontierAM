"""Publication failures retain per-input results and never guess a rollback."""

import asyncio
import threading

from dataclasses import replace
from functools import partial
from unittest.mock import Mock
from uuid import uuid4

import numpy as np
import pytest

from tests.unit.gateway.redis_store.fake_redis import FakeRedis

from nexus.gateway.trajectory_store_backends.redis import HybridNodeRef, NodeRecord, RedisTrajectoryRecordStore
from nexus.gateway.trajectory_store_backends.redis_store.client import RedisClient
from nexus.gateway.trajectory_store_backends.redis_store.file_storage.blob_codec import decode_blob
from nexus.gateway.trajectory_store_backends.redis_store.file_storage.publication import StoredNode
from nexus.gateway.trajectory_store_backends.redis_store.types import CompareAndSetResult, RedisClientError


class MemoryFS:
    def __init__(self):
        self.blobs = {}
        self.closed = False

    def allocate_url(self):
        return 'hykvfs://test/' + uuid4().hex

    def put(self, url, blob):
        assert url not in self.blobs
        self.blobs[url] = blob

    def get(self, url):
        if url not in self.blobs:
            raise RedisClientError('blob missing', code='not_found')
        return self.blobs[url]

    def delete(self, url):
        return self.blobs.pop(url, None) is not None

    def close(self):
        self.closed = True


@pytest.fixture
def client():
    with RedisClient(redis_client=FakeRedis(), fs_client=MemoryFS()) as client:
        yield client


def record(value=b'large'):
    return NodeRecord('run', 1, [1, 2], routed_experts=value, multi_modal_items={})


def test_only_non_none_large_fields_are_external(client):
    node_id = client.put_node(record(b''))
    assert isinstance(node_id, str)
    stored = StoredNode.deserialize(client.get_node_blobs([node_id])[0])
    assert stored.state == 'committed'
    assert stored.write_id
    assert NodeRecord.deserialize(stored.record).routed_experts is None
    assert len(stored.blob_urls) == len(client._publication._fs.blobs) == 2
    restored = client.get_node(node_id, payload_type=NodeRecord)
    assert type(restored) is NodeRecord
    np.testing.assert_equal(vars(restored), vars(record(b'')))
    inline = replace(record(), routed_experts=None, multi_modal_items=None)
    inline_id = client.put_node(inline)
    assert client.get_node_blobs([inline_id]) == [inline.serialize()]


def test_reservation_conflict_does_not_upload_and_preserves_other_success(client):
    original = client.put_node(record(), node_id='taken')
    with pytest.raises(RedisClientError) as caught:
        client.put_multi_node([record(), record()], node_ids=['taken', 'new'])
    assert caught.value.code == 'publication_failed'
    assert caught.value.body['partial_ids'] == [None, 'new']
    assert set(caught.value.body['errors']) == {0}
    assert len(client._publication._fs.blobs) == 4
    restored = client.get_multi_node([(original, NodeRecord), ('new', NodeRecord)])
    for actual, expected in zip(restored, [record(), record()], strict=True):
        assert type(actual) is NodeRecord
        np.testing.assert_equal(vars(actual), vars(expected))


@pytest.mark.parametrize('failure', ['upload', 'cas_mismatch', 'cas_unknown'])
def test_partial_publication_keeps_writing_record_and_successful_sibling(client, monkeypatch, failure):
    if failure == 'upload':
        original = client._publication._fs.put

        def upload(url, blob):
            if decode_blob(blob) == b'fail':
                raise RuntimeError('upload failed')
            return original(url, blob)

        monkeypatch.setattr(client._publication._fs, 'put', upload)
    else:
        original = client.compare_and_set_many

        def commit(updates):
            if updates[0][0] == 'bad':
                if failure == 'cas_unknown':
                    original(updates)
                    raise ConnectionError('reply lost after commit')
                return [CompareAndSetResult.MISMATCH]
            return original(updates)

        monkeypatch.setattr(client, 'compare_and_set_many', commit)
    with pytest.raises(RedisClientError) as caught:
        client.put_multi_node([record(b'fail'), record(b'good')], node_ids=['bad', 'good'])
    assert caught.value.body['partial_ids'] == [None, 'good']
    restored = client.get_node('good', payload_type=NodeRecord)
    assert type(restored) is NodeRecord
    np.testing.assert_equal(vars(restored), vars(record(b'good')))
    stored = StoredNode.deserialize(client.get_node_blobs(['bad'])[0])
    assert stored.state == ('committed' if failure == 'cas_unknown' else 'writing')
    assert stored.blob_urls
    if failure != 'cas_unknown':
        for operation in (lambda: client.get_node('bad', payload_type=NodeRecord), lambda: client.delete_node('bad')):
            with pytest.raises(RedisClientError) as error:
                operation()
            assert error.value.code == 'write_in_progress'


def test_unknown_reservation_is_not_replayed_or_uploaded(client, monkeypatch):
    pipeline = client._r.pipeline
    attempts = []

    def fail_after_write(*args, **kwargs):
        pipe = pipeline(*args, **kwargs)
        execute = pipe.execute

        def lost_reply(**options):
            attempts.append(True)
            execute(**options)
            raise ConnectionError('reply lost')

        pipe.execute = lost_reply
        return pipe

    monkeypatch.setattr(client._r, 'pipeline', fail_after_write)
    with pytest.raises(RedisClientError) as caught:
        client.put_multi_node([record(), record()], node_ids=['a', 'b'])
    assert attempts == [True]
    assert caught.value.body['partial_ids'] == [None, None]
    assert client._publication._fs.blobs == {}
    assert all(StoredNode.deserialize(blob).state == 'writing' for blob in client.get_node_blobs(['a', 'b']))


def test_store_builds_partial_refs_from_publication_results_without_reread(client, monkeypatch):
    with RedisTrajectoryRecordStore('run', {'redis_client': client._r, 'fs_client': client._publication._fs}) as store:
        original = store._client._publication._commit_write

        def commit(plan):
            if NodeRecord.deserialize(StoredNode.deserialize(plan.pending).record).node_id == 1:
                raise RuntimeError('commit failed')
            return original(plan)

        monkeypatch.setattr(store._client._publication, '_commit_write', commit)
        monkeypatch.setattr(store._client, 'get_node_blobs', Mock(side_effect=AssertionError('unexpected reread')))
        with pytest.raises(RedisClientError) as caught:
            asyncio.run(store.put_many([record(), replace(record(), node_id=2)]))
        assert caught.value.code == 'publication_failed'
        refs = caught.value.body['partial_refs']
        assert len(refs) == 2
        assert refs[0] is None
        assert isinstance(refs[1], HybridNodeRef)
        assert refs[1].routed_experts_url in client._publication._fs.blobs
        assert refs[1].multi_modal_items_url in client._publication._fs.blobs
        assert set(caught.value.body['errors']) == {0}


def test_forged_reference_is_rejected_before_any_delete(client, monkeypatch):
    left, right = client._publication.put([record(), record()])
    urls = {**left.blob_urls, 'routed_experts': right.blob_urls['routed_experts']}
    delete = Mock(wraps=client._publication._fs.delete)
    monkeypatch.setattr(client._publication._fs, 'delete', delete)
    for operation in (
        lambda: client._publication.get([(left.node_id, NodeRecord)], reference_urls=[urls]),
        lambda: client._publication.delete([left.node_id], reference_urls=[urls]),
    ):
        with pytest.raises(RedisClientError) as caught:
            operation()
        assert caught.value.code == 'invalid_reference'
    delete.assert_not_called()
    restored = client.get_node(left.node_id, payload_type=NodeRecord)
    assert type(restored) is NodeRecord
    np.testing.assert_equal(vars(restored), vars(record()))


@pytest.mark.parametrize('failure', ['redis_conflict', 'fs_failure'])
def test_partial_delete_keeps_replacement_and_does_not_retry_orphan(client, monkeypatch, failure):
    result = client._publication.put([record()])[0]
    publisher = client._publication
    if failure == 'redis_conflict':
        original = publisher._delete_if_unchanged
        replacement = NodeRecord('other', 2, [3])

        def replace_first(snapshot):
            client._r.set(client._node_key(result.node_id), replacement.serialize())
            return original(snapshot)

        monkeypatch.setattr(publisher, '_delete_if_unchanged', replace_first)
        with pytest.raises(RedisClientError) as caught:
            client.delete_multi_node([result.node_id])
        assert caught.value.code == 'conflict'
        assert set(result.blob_urls.values()) <= publisher._fs.blobs.keys()
        restored = client.get_node(result.node_id, payload_type=NodeRecord)
        assert type(restored) is NodeRecord
        np.testing.assert_equal(vars(restored), vars(replacement))
    else:
        delete = Mock(side_effect=RuntimeError('injected delete failure'))
        monkeypatch.setattr(publisher._fs, 'delete', delete)
        with pytest.raises(RuntimeError, match='injected delete failure'):
            client.delete_multi_node([result.node_id])
        count = delete.call_count
        client.delete_multi_node([result.node_id])
        assert delete.call_count == count
        assert publisher._fs.get(result.blob_urls['routed_experts'])


def test_preallocated_fs_ids_are_immutable_even_for_inline_records(client):
    node_id = client.put_node(record(), node_id='immutable')
    inline = replace(record(), routed_experts=None, multi_modal_items=None)
    with pytest.raises(RedisClientError, match='already exists'):
        client.put_node(inline, node_id=node_id)
    with pytest.raises(ValueError, match='immutable'):
        client.put_node(inline, node_id=node_id, overwrite=True)
    restored = client.get_node(node_id, payload_type=NodeRecord)
    assert type(restored) is NodeRecord
    np.testing.assert_equal(vars(restored), vars(record()))


@pytest.mark.parametrize('forged_first', [True, False])
def test_duplicate_store_refs_are_all_validated_before_release(client, forged_first):
    config = {'redis_client': client._r, 'fs_client': client._publication._fs}
    with RedisTrajectoryRecordStore('run', config) as store:
        valid, other = asyncio.run(store.put_many([record(), record()]))
        forged = replace(valid, routed_experts_url=other.routed_experts_url)
        refs = [forged, valid] if forged_first else [valid, forged]
        with pytest.raises(RedisClientError) as caught:
            asyncio.run(store.release_many(refs))
        assert caught.value.code == 'invalid_reference'
        assert len(client._publication._fs.blobs) == 4


def test_parallel_refills_any_finished_slot_and_keeps_input_order(client):
    client._publication._io_workers = 2
    unblock = threading.Event()

    def call(index):
        if index == 0:
            assert unblock.wait(5)  # blocks one slot; the other must keep serving calls 1..3
        if index == 3:
            unblock.set()
        return index

    assert client._publication._run_parallel([partial(call, i) for i in range(5)]) == [0, 1, 2, 3, 4]


@pytest.mark.parametrize('outcome', [-1, ConnectionError('reply lost')])
def test_delete_only_cleans_blobs_of_acknowledged_records(client, monkeypatch, outcome):
    failed, deleted = client._publication.put([record(), record()])
    publisher = client._publication
    execute = publisher.execute_pipeline

    def partial_delete(pipe, **kwargs):
        results = execute(pipe, **kwargs)
        results[0] = outcome
        return results

    monkeypatch.setattr(publisher, 'execute_pipeline', partial_delete)
    with pytest.raises(RedisClientError if outcome == -1 else ConnectionError) as caught:
        client.delete_multi_node([failed.node_id, deleted.node_id])
    if outcome == -1:
        assert caught.value.code == 'conflict'
    else:
        assert caught.value is outcome
    assert set(failed.blob_urls.values()) <= publisher._fs.blobs.keys()
    assert set(deleted.blob_urls.values()).isdisjoint(publisher._fs.blobs)


def test_unprotected_delete_keeps_one_snapshot_and_one_delete_pipeline(client, monkeypatch):
    [result] = client._publication.put([record()])
    read = Mock(wraps=client._r.mget)
    execute = Mock(wraps=client._publication.execute_pipeline)
    monkeypatch.setattr(client._r, 'mget', read)
    monkeypatch.setattr(client._publication, 'execute_pipeline', execute)
    client.delete_multi_node([result.node_id])
    read.assert_called_once_with([client._node_key(result.node_id)])
    execute.assert_called_once()
    assert not client._publication._fs.blobs


def test_mixed_protected_delete_does_not_replay_or_clean_deferred_blobs(client, monkeypatch):
    from nexus.gateway.trajectory_store_backends.redis_store.dump_protection import (
        deferred_record_key,
        protect_dump_records,
    )

    busy, deleted = client._publication.put([record(), record()])
    publisher = client._publication
    busy_key, deleted_key = client._node_key(busy.node_id), client._node_key(deleted.node_id)
    with protect_dump_records(client, [busy.node_id], 'dump'):
        execute = publisher.execute_pipeline
        calls = []

        def capture(pipe, **kwargs):
            calls.append([args[3] for _, args, _ in pipe._ops])
            return execute(pipe, **kwargs)

        monkeypatch.setattr(publisher, 'execute_pipeline', capture)
        client.delete_multi_node([busy.node_id, deleted.node_id])
        assert calls == [[busy_key, deleted_key]]
        assert client.get_node_blobs([busy.node_id, deleted.node_id]) == [None, None]
        assert client._r.get(deferred_record_key(busy_key)) is not None
        assert set(busy.blob_urls.values()) <= publisher._fs.blobs.keys()
        assert set(deleted.blob_urls.values()).isdisjoint(publisher._fs.blobs)
    assert not publisher._fs.blobs
