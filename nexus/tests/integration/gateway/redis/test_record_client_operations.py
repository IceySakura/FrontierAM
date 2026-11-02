import asyncio
import pickle
import threading

from collections import Counter
from uuid import uuid4

import pytest

from nexus.gateway.trajectory_store_backends.redis import (
    NodeRecord,
    RedisTrajectoryRecordStore,
    SegmentRecord,
)
from nexus.gateway.trajectory_store_backends.redis_store.types import (
    CompareAndSetResult,
    RedisClientError,
    Serializable,
)
from tests.unit.gateway.redis_store.payloads import SamplePayload


pytestmark = pytest.mark.integration


@pytest.fixture
def redis_client(backend_config):
    from nexus.gateway.trajectory_store_backends.redis_store.client import RedisClient

    with RedisClient(**backend_config, key_prefix=uuid4().hex) as client:
        yield client


def test_redis_uses_batch_reads_and_pipelines(redis_client, monkeypatch):
    client = redis_client
    counts = Counter()
    original_pipeline = client._r.pipeline
    execute_pipeline = client._publication.execute_pipeline
    read_name = 'mget_nonatomic' if callable(getattr(client._r, 'mget_nonatomic', None)) else 'mget'
    original_read = getattr(client._r, read_name)

    def read(keys):
        counts['batch_read'] += 1
        assert len(keys) == 6
        return original_read(keys)

    def pipeline(*args, **kwargs):
        pipe = original_pipeline(*args, **kwargs)
        execute_command = pipe.execute_command

        def command(*args, **kwargs):
            counts[args[0]] += 1
            return execute_command(*args, **kwargs)

        pipe.execute_command = command
        return pipe

    def run(pipe, **kwargs):
        counts['record_pipeline'] += 1
        return execute_pipeline(pipe, **kwargs)

    monkeypatch.setattr(client._r, read_name, read)
    monkeypatch.setattr(client._r, 'pipeline', pipeline)
    monkeypatch.setattr(client._publication, 'execute_pipeline', run)
    assert client.compare_and_set_many([]) == []
    assert client.get_node_blobs([]) == []
    assert counts == {}
    payloads = [SamplePayload(i) for i in range(6)]
    refs = client.put_multi_node(payloads)
    assert all(isinstance(ref, str) for ref in refs)
    blobs = client.get_node_blobs(refs)
    assert (
        client.compare_and_set_many([(ref, blob, b'new') for ref, blob in zip(refs, blobs, strict=True)])
        == [CompareAndSetResult.UPDATED] * 6
    )
    assert client.get_node_blobs(refs) == [b'new'] * 6
    client.delete_multi_node(refs)
    client.delete_multi_node(refs)
    assert counts['batch_read'] == 4
    assert counts['record_pipeline'] == 3
    assert counts['SET'] == 6
    assert counts['EVAL'] == 12


def test_record_publication_uses_one_pipeline_without_prereading(redis_client, monkeypatch):
    counts = Counter()
    pipeline = redis_client._r.pipeline
    execute_pipeline = redis_client._publication.execute_pipeline

    def make_pipeline(*args, **kwargs):
        counts['pipeline'] += 1
        pipe = pipeline(*args, **kwargs)
        set_value = pipe.set

        def put(*args, **kwargs):
            counts['set'] += 1
            return set_value(*args, **kwargs)

        pipe.set = put
        return pipe

    def run(pipe, **kwargs):
        counts['execute'] += 1
        return execute_pipeline(pipe, **kwargs)

    def unexpected_read(*args, **kwargs):
        raise AssertionError('publication must not pre-read existing records')

    monkeypatch.setattr(redis_client._r, 'pipeline', make_pipeline)
    with RedisTrajectoryRecordStore(
        'pipeline',
        {
            'url': redis_client.url,
            'redis_client': redis_client._r,
            'fs_client': redis_client._publication._fs,
            'key_prefix': redis_client.key_prefix,
        },
    ) as store:
        monkeypatch.setattr(store._client._publication, 'execute_pipeline', run)
        node = NodeRecord('t', 1, [1])
        with monkeypatch.context() as patch:
            for name in ('get', 'mget', 'mget_nonatomic'):
                if callable(getattr(redis_client._r, name, None)):
                    patch.setattr(redis_client._r, name, unexpected_read)
            refs = store._put_many([node, SegmentRecord('t', 's', {}), node])
        assert len(set(refs)) == 3
        assert counts == {'pipeline': 1, 'set': 3, 'execute': 1}
        assert store._get_many(refs, None) == [{'token_ids': [1]}, {}, {'token_ids': [1]}]
        store._release_many(refs)


@pytest.fixture
def record_store(redis_client):
    with RedisTrajectoryRecordStore(
        'cas',
        {
            'url': redis_client.url,
            'redis_client': redis_client._r,
            'key_prefix': redis_client.key_prefix,
            'fs_client': redis_client._publication._fs,
        },
    ) as store:
        yield store


def test_raw_cas_preserves_order_and_does_not_decode(redis_client, monkeypatch):
    from redis.cluster import key_slot

    client = redis_client
    refs = [f'{i}{{slot-{i}}}' for i in range(4)]
    assert len({key_slot(client._node_key(ref)) for ref in refs}) == len(refs)
    for ref, blob in zip(refs[:3], [b'old', b'', b'unchanged'], strict=True):
        client._r.set(client._node_key(ref), blob)

    def no_decode(blob):
        raise AssertionError('raw primitives must not decode payloads')

    monkeypatch.setattr(Serializable, 'deserialize', staticmethod(no_decode))
    assert client.get_node_blobs([refs[3], refs[1], refs[0], refs[2]]) == [None, b'', b'old', b'unchanged']
    assert client.compare_and_set_many([
        (refs[3], b'old', b'must not create'),
        (refs[1], b'', b'zero'),
        (refs[2], b'wrong', b'must not overwrite'),
        (refs[0], b'old', b'new'),
    ]) == [
        CompareAndSetResult.MISSING,
        CompareAndSetResult.UPDATED,
        CompareAndSetResult.MISMATCH,
        CompareAndSetResult.UPDATED,
    ]
    assert client.get_node_blobs(refs) == [b'new', b'zero', b'unchanged', None]
    with pytest.raises(TypeError, match='node_id must be a string'):
        client.compare_and_set_many([(refs[0], b'new', b'invalid batch'), (None, b'', b'')])
    with pytest.raises(TypeError, match='node_id must be a string'):
        client.get_node_blobs([refs[0], None])
    assert client.get_node_blobs(refs[:1]) == [b'new']
    client.delete_multi_node(refs)


@pytest.mark.parametrize('conflicts', [1, 7])
def test_store_cas_retries_only_contenders_without_lost_fields(record_store, monkeypatch, conflicts):
    store = record_store
    refs = store._put_many([SegmentRecord('t', str(i), {}) for i in range(2)])
    original = store._client.get_node_blobs
    append = store._append_fields
    reads = 0

    def read(raw_refs):
        nonlocal reads
        assert raw_refs == [ref.record_id for ref in (refs if reads == 0 else refs[:1])]
        blobs = original(raw_refs)
        reads += 1
        if reads <= conflicts:
            payload = SegmentRecord.deserialize(blobs[0]).with_fields({f'other-{reads}': pickle.dumps('other-value')})
            store._client._r.set(store._client._node_key(refs[0].record_id), payload.serialize())
        return blobs

    def observe(encoded):
        with monkeypatch.context() as patch:
            patch.setattr(store._client, 'get_node_blobs', read)
            append(encoded)

    monkeypatch.setattr(store, '_append_fields', observe)
    asyncio.run(store.put_fields_many({ref: {'mine': 'my-value'} for ref in refs}))
    assert reads == conflicts + 1
    assert store._get_many(refs, None) == [
        {**{f'other-{i}': 'other-value' for i in range(1, conflicts + 1)}, 'mine': 'my-value'},
        {'mine': 'my-value'},
    ]
    store._release_many(refs)


def test_store_cas_bounds_retries_and_does_not_resurrect(record_store, monkeypatch):
    store = record_store
    (ref,) = store._put_many([SegmentRecord('t', 's', {})])
    original = store._client.get_node_blobs
    append = store._append_fields
    reads = 0

    def race(raw_refs):
        nonlocal reads
        blobs = original(raw_refs)
        reads += 1
        payload = SegmentRecord.deserialize(blobs[0]).with_fields({str(reads): pickle.dumps('competitor')})
        store._client._r.set(store._client._node_key(ref.record_id), payload.serialize())
        return blobs

    reader = race

    def observe(encoded):
        with monkeypatch.context() as patch:
            patch.setattr(store._client, 'get_node_blobs', reader)
            append(encoded)

    monkeypatch.setattr(store, '_append_fields', observe)
    with pytest.raises(RedisClientError) as error:
        asyncio.run(store.put_fields_many({ref: {'mine': 'value'}}))
    assert error.value.code == 'contention'
    assert reads == 8
    assert 'mine' not in store._get_many([ref], None)[0]

    def delete(raw_refs):
        blobs = original(raw_refs)
        store._client._r.delete(store._client._node_key(raw_refs[0]))
        return blobs

    reader = delete
    with pytest.raises(KeyError) as error:
        asyncio.run(store.put_fields_many({ref: {'mine': 'value'}}))
    assert error.value.__cause__.code == 'not_found'
    assert not store._client.has_node(ref.record_id)[0]


@pytest.mark.parametrize('competitor_value', ['same', 'different'])
def test_store_rechecks_fields_after_cas_mismatch(record_store, monkeypatch, competitor_value):
    store = record_store
    (ref,) = store._put_many([SegmentRecord('t', 's', {})])
    cas = store._client.compare_and_set_many
    calls = 0

    def compete(updates):
        nonlocal calls
        calls += 1
        raw_ref, old, _ = updates[0]
        payload = SegmentRecord.deserialize(old).with_fields({
            'mine': pickle.dumps(competitor_value, protocol=pickle.HIGHEST_PROTOCOL)
        })
        store._client._r.set(store._client._node_key(raw_ref), payload.serialize())
        return cas(updates)

    monkeypatch.setattr(store._client, 'compare_and_set_many', compete)
    if competitor_value == 'same':
        asyncio.run(store.put_fields_many({ref: {'mine': 'same'}}))
        asyncio.run(store.put_fields_many({ref: {'mine': 'same'}}))
    else:
        with pytest.raises(ValueError, match='conflicting record field'):
            asyncio.run(store.put_fields_many({ref: {'mine': 'same'}}))
    assert calls == 1
    assert store._get_many([ref], None) == [{'mine': competitor_value}]
    store._release_many([ref])


def test_store_cas_compares_original_bytes(record_store):
    store = record_store
    (ref,) = store._put_many([SegmentRecord('t', 's', {})])
    # A different valid pickle encoding must not cause spurious CAS mismatches.
    blob = pickle.dumps(SegmentRecord('t', 's', {}), protocol=0)
    store._client._r.set(store._client._node_key(ref.record_id), blob)
    asyncio.run(store.put_fields_many({ref: {'late': [2]}}))
    assert store._get_many([ref], None) == [{'late': [2]}]
    store._release_many([ref])


def test_cancel_drains_all_cas_rounds_in_one_worker(record_store, monkeypatch):
    store = record_store
    (ref,) = store._put_many([SegmentRecord('t', 's', {})])
    started = threading.Event()
    proceed = threading.Event()
    workers = []
    cas = store._client.compare_and_set_many

    def blocked_cas(updates):
        workers.append(threading.get_ident())
        if len(workers) == 1:
            started.set()
            assert proceed.wait(5)
            return [CompareAndSetResult.MISMATCH]
        return cas(updates)

    monkeypatch.setattr(store._client, 'compare_and_set_many', blocked_cas)

    async def exercise():
        task = asyncio.create_task(store.put_fields_many({ref: {'late': 1}}))
        try:
            assert await asyncio.to_thread(started.wait, 3)
            for index in range(2):
                task.cancel(f'cancel-{index}')
                await asyncio.sleep(0)
                assert not task.done()
        finally:
            proceed.set()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert len(workers) == 2
        assert workers[0] == workers[1] != threading.get_ident()
        assert await store.get_many([ref]) == [{'late': 1}]
        await store.release_many([ref])

    asyncio.run(exercise())


def test_redis_payload_paths_use_shared_serialization(redis_client, monkeypatch):
    counts = Counter()
    serialize = Serializable.serialize
    deserialize = Serializable.deserialize

    def encode(payload):
        counts['serialize'] += 1
        return serialize(payload)

    def decode(blob):
        counts['deserialize'] += 1
        return deserialize(blob)

    monkeypatch.setattr(Serializable, 'serialize', encode)
    monkeypatch.setattr(Serializable, 'deserialize', staticmethod(decode))
    client = redis_client
    payload = SamplePayload([2])
    ref = client.put_node(payload)
    assert client.get_node(ref, payload_type=SamplePayload) == payload
    records = [SamplePayload([1]), payload]
    refs = client.put_multi_node(records)
    assert client.get_multi_node([(ref, SamplePayload) for ref in refs]) == records
    assert counts == {'serialize': 3, 'deserialize': 3}
    client.delete_multi_node([ref, *refs])
