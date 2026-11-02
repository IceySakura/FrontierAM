"""Tests for the Redis payload and Tensor client."""

from __future__ import annotations

import asyncio
import pickle
import threading

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tests.unit.gateway.redis_store.fake_redis import FakeRedis
from tests.unit.gateway.redis_store.file_storage.test_publication import MemoryFS
from tests.unit.gateway.redis_store.payloads import SamplePayload

from nexus.gateway.trajectory_store_backends.redis_store.client import (
    AsyncRedisClient,
    RedisClient,
    parse_cluster_url,
    parse_sentinel_url,
)
from nexus.gateway.trajectory_store_backends.redis_store.types import (
    RedisClientError,
    Serializable,
)


@pytest.fixture
def redis_client():
    with RedisClient(url='redis://fake', redis_client=FakeRedis(), key_prefix='t', fs_client=MemoryFS()) as client:
        yield client


class FakeClusterRedis(FakeRedis):
    def __init__(self) -> None:
        super().__init__()
        self.used_nonatomic_mget = False
        self.pipeline_transaction = None

    def mget_nonatomic(self, keys):
        self.used_nonatomic_mget = True
        return self.mget(keys)

    def pipeline(self, transaction=None):
        self.pipeline_transaction = transaction
        pipe = super().pipeline()
        pipe._execution_strategy = SimpleNamespace(command_queue=pipe._ops)
        pipe.send_cluster_commands = lambda queue, **kwargs: pipe.execute(raise_on_error=kwargs['raise_on_error'])
        pipe.reset = lambda: pipe._ops.clear()
        return pipe


def test_preallocated_ids_keep_string_api_and_nx_conflicts(redis_client):
    payload = SamplePayload(text='fixed')
    assert redis_client.put_node(payload, node_id='a') == 'a'
    assert redis_client.put_multi_node([payload, payload], node_ids=['b', 'c']) == ['b', 'c']
    with pytest.raises(RedisClientError, match='already exists') as caught:
        redis_client.put_multi_node([payload, payload], node_ids=['a', 'd'], overwrite=False)
    assert caught.value.body['partial_ids'] == [None, 'd']
    assert redis_client.get_node('d', payload_type=SamplePayload) == payload


@pytest.mark.parametrize('node_ids', [['same', 'same'], ['only-one'], ['', 'ok'], [1, 'ok'], 'ab', b'ab'])
def test_invalid_preallocated_ids_fail_before_io(node_ids):
    raw = Mock(spec=FakeRedis)
    client = RedisClient(redis_client=raw, fs_client=MemoryFS())
    with pytest.raises((ValueError, TypeError)):
        client.put_multi_node([SamplePayload(), SamplePayload()], node_ids=node_ids)
    assert raw.mock_calls == []


def test_parse_sentinel_url():
    master, sentinels, db = parse_sentinel_url(
        'redis+sentinel://redis-master/2?sentinels=10.0.0.1:26379,10.0.0.2:26380'
    )
    assert master == 'redis-master'
    assert sentinels == [('10.0.0.1', 26379), ('10.0.0.2', 26380)]
    assert db == 2


def test_parse_cluster_url():
    url = 'redis+cluster://10.0.0.1:16379/0?nodes=10.0.0.1:16379,10.0.0.2:16379'
    assert parse_cluster_url(url) == [('10.0.0.1', 16379), ('10.0.0.2', 16379)]
    with pytest.raises(ValueError, match='database 0'):
        parse_cluster_url('redis+cluster://10.0.0.1:16379/1')


def test_put_get_payload_only(redis_client: RedisClient):
    ref = redis_client.put_node(SamplePayload(text='hello'))
    assert isinstance(ref, str)

    got = redis_client.get_node(ref, payload_type=SamplePayload)
    assert got.text == 'hello'
    assert redis_client.has_node(ref)[0]


def test_put_node_ignores_unknown_kwargs(redis_client: RedisClient):
    ref = redis_client.put_node(SamplePayload(text='x'), node=object(), wait_confirm=True)
    assert isinstance(ref, str)
    got = redis_client.get_node(ref, payload_type=SamplePayload)
    assert got.text == 'x'


def test_put_get_tensor_roundtrip(redis_client: RedisClient):
    torch = pytest.importorskip('torch')
    source = torch.arange(24, dtype=torch.float32).reshape(4, 6).T
    expected = source.clone()

    ref = redis_client.put_tensor(source)
    source.add_(100)
    restored = redis_client.get_tensor(ref)

    assert isinstance(ref, str)
    assert f'{redis_client.key_prefix}:tensor:{ref}'.encode() in redis_client._r._kv
    assert restored.shape == expected.shape
    assert restored.dtype == expected.dtype
    assert torch.equal(restored, expected)


def test_put_tensor_stores_one_variable_length_record(redis_client: RedisClient):
    torch = pytest.importorskip('torch')
    source = torch.arange(8, dtype=torch.int64).reshape(2, 4)

    ref = redis_client.put_tensor(source)
    record = bytes(redis_client._r.get(f'{redis_client.key_prefix}:tensor:{ref}'.encode()))
    meta_length = int.from_bytes(record[:4], byteorder='big')
    payload_offset = 4 + meta_length

    assert pickle.loads(record[4:payload_offset]) == ((2, 4), torch.int64)
    assert record[payload_offset:] == bytes(source.untyped_storage())
    assert len(redis_client._r._kv) == 1


def test_put_get_empty_tensor(redis_client: RedisClient):
    torch = pytest.importorskip('torch')
    source = torch.empty((2, 0), dtype=torch.float16)

    restored = redis_client.get_tensor(redis_client.put_tensor(source))

    assert restored.shape == source.shape
    assert restored.dtype == source.dtype
    assert restored.numel() == 0


def test_put_get_multi_payloads(redis_client: RedisClient):
    payloads = [SamplePayload(text=f'p{i}') for i in range(3)]
    refs = redis_client.put_multi_node(payloads)
    assert len(refs) == 3
    assert all(isinstance(r, str) for r in refs)
    assert len(set(refs)) == 3

    got = redis_client.get_multi_node([(ref, SamplePayload) for ref in refs])
    assert [g.text for g in got] == ['p0', 'p1', 'p2']


def test_cluster_multi_uses_cross_slot_safe_commands():
    raw = FakeClusterRedis()
    client = RedisClient(url='redis+cluster://fake:16379/0', redis_client=raw, key_prefix='c', fs_client=MemoryFS())
    payloads = [SamplePayload(text=f'c{i}') for i in range(3)]
    refs = client.put_multi_node(payloads)
    got = client.get_multi_node([(ref, SamplePayload) for ref in refs])
    assert raw.pipeline_transaction is False
    assert raw.used_nonatomic_mget
    assert not raw._ints  # Cluster mode must not allocate IDs from a replicated counter.
    assert len(set(refs)) == len(refs)
    assert [item.text for item in got] == ['c0', 'c1', 'c2']


def test_put_multi_ignores_nodes_kwarg(redis_client: RedisClient):
    payloads = [SamplePayload(text='a'), SamplePayload(text='b')]
    refs = redis_client.put_multi_node(payloads, nodes=[object(), object()])
    assert all(isinstance(r, str) for r in refs)
    got = redis_client.get_multi_node([(ref, SamplePayload) for ref in refs])
    assert [g.text for g in got] == ['a', 'b']


def test_ids_are_unique(redis_client: RedisClient):
    refs = [redis_client.put_node(SamplePayload(text=str(i))) for i in range(5)]
    assert len(set(refs)) == 5


def test_async_redis_client_runs_sync_api_in_worker_threads(monkeypatch):
    torch = pytest.importorskip('torch')
    client = AsyncRedisClient(url='redis://fake', redis_client=FakeRedis(), key_prefix='async', fs_client=MemoryFS())
    event_loop_thread = threading.get_ident()
    worker_threads = []

    def track_thread(method):
        def wrapped(*args, **kwargs):
            worker_threads.append(threading.get_ident())
            return method(*args, **kwargs)

        return wrapped

    for name in ('put_node', 'get_node', 'put_tensor', 'get_tensor'):
        monkeypatch.setattr(client, name, track_thread(getattr(client, name)))

    async def run():
        payload = SamplePayload(text='async-node')
        node_ref = await client.async_put_node(payload)
        assert (await client.async_get_node(node_ref, payload_type=SamplePayload)) == payload

        tensor = torch.arange(8, dtype=torch.float32)
        tensor_ref = await client.async_put_tensor(tensor)
        assert torch.equal(await client.async_get_tensor(tensor_ref), tensor)

    try:
        asyncio.run(run())
    finally:
        client.close()

    assert len(worker_threads) == 4
    assert all(thread_id != event_loop_thread for thread_id in worker_threads)


def test_serialization_failure_keeps_the_entire_batch_unpublished(redis_client, monkeypatch):
    serialize = Serializable.serialize

    def encode(payload):
        if payload.text == 'fail':
            raise ValueError('injected serialization failure')
        return serialize(payload)

    monkeypatch.setattr(Serializable, 'serialize', encode)
    with pytest.raises(ValueError, match='injected serialization failure'):
        redis_client.put_multi_node([SamplePayload(text='ok'), SamplePayload(text='fail')])
    assert redis_client._r._kv == {}


@pytest.mark.parametrize('multi', [False, True])
def test_deserialization_preserves_payload_type_validation(redis_client, multi):
    ref = 'invalid'
    key = redis_client._node_key(ref)
    redis_client._r.set(key, pickle.dumps({'not': 'a payload'}))
    with pytest.raises(RedisClientError, match='invalid payload'):
        redis_client.get_multi_node([(ref, SamplePayload)]) if multi else redis_client.get_node(
            ref, payload_type=SamplePayload
        )
    redis_client._r.set(key, b'!invalid-pickle')
    with pytest.raises(pickle.UnpicklingError):
        redis_client.get_multi_node([(ref, SamplePayload)]) if multi else redis_client.get_node(
            ref, payload_type=SamplePayload
        )


@pytest.mark.parametrize('multi', [False, True])
@pytest.mark.parametrize('source', ['payload', 'codec', 'network'])
def test_write_errors_propagate_before_returning_ids(redis_client, monkeypatch, multi, source):
    payload = SamplePayload(text='value')
    failure = RuntimeError('injected write failure')
    if source == 'payload':
        payload = object()
    elif source == 'codec':
        monkeypatch.setattr(SamplePayload, 'serialize', Mock(side_effect=failure))
    else:
        monkeypatch.setattr(redis_client._r, 'set', Mock(side_effect=failure))
    with pytest.raises(TypeError if source == 'payload' else RuntimeError) as caught:
        redis_client.put_multi_node([payload]) if multi else redis_client.put_node(payload)
    if source == 'network':
        assert caught.value.code == 'publication_failed'
        assert caught.value.body['partial_ids'] == [None]
        assert 'injected write failure' in caught.value.body['errors'][0]
    elif source == 'codec':
        assert caught.value is failure
    assert redis_client._r._kv == {}


@pytest.mark.parametrize('multi', [False, True])
def test_nx_conflict_raises_and_preserves_successful_writes(redis_client, monkeypatch, multi):
    monkeypatch.setattr(
        'nexus.gateway.trajectory_store_backends.redis_store.client.new_node_id',
        iter(['taken', 'fresh', 'taken'] if multi else ['taken', 'taken']).__next__,
    )
    redis_client.put_node(SamplePayload(text='original'))
    put = redis_client.put_multi_node if multi else redis_client.put_node
    payloads = [SamplePayload(text='new'), SamplePayload(text='conflict')] if multi else SamplePayload(text='conflict')
    with pytest.raises(RedisClientError, match='already exists.*taken') as caught:
        put(payloads, overwrite=False)
    assert caught.value.body['partial_ids'] == (['fresh', None] if multi else [None])
    assert redis_client.get_node('taken', payload_type=SamplePayload).text == 'original'
    if multi:
        assert redis_client.get_node('fresh', payload_type=SamplePayload).text == 'new'
