import pickle
import sys

from unittest.mock import Mock

import numpy as np
import pytest

from tests.unit.gateway.redis_store.fake_redis import FakeRedis
from tests.unit.gateway.redis_store.file_storage.test_publication import MemoryFS
from tests.unit.gateway.redis_store.payloads import BinaryPayload, JsonPayload, SamplePayload

from nexus.gateway.trajectory_store_backends.redis import NodeRecord
from nexus.gateway.trajectory_store_backends.redis_store.client import AsyncRedisClient, RedisClient
from nexus.gateway.trajectory_store_backends.redis_store.types import RedisClientError, Serializable


class EncodedNode(NodeRecord):
    def serialize(self):
        return b'custom:' + pickle.dumps(vars(self))

    @staticmethod
    def deserialize(blob):
        assert blob.startswith(b'custom:')
        return EncodedNode(**pickle.loads(blob[7:]))


@pytest.mark.parametrize('token_ids', [[2], np.array([2], dtype=np.int64)], ids=['list', 'numpy'])
def test_fs_keeps_selected_codecs_for_inline_and_offloaded_payloads(token_ids):
    pytest.importorskip('zmq')
    from nexus.gateway.trajectory_store_backends.redis_store.file_storage.publication import StoredNode

    fs = MemoryFS()
    with RedisClient(redis_client=FakeRedis(), fs_client=fs) as client:
        payloads = [JsonPayload('json'), BinaryPayload(42), EncodedNode('run', 1, token_ids, routed_experts=b'large')]
        ids = client.put_multi_node(payloads)
        for node_id, payload in zip(ids, payloads, strict=True):
            restored = client.get_node(node_id, payload_type=type(payload))
            assert type(restored) is type(payload)
            np.testing.assert_equal(vars(restored), vars(payload))
        order = (2, 0, 1, 2)
        restored = client.get_multi_node([(ids[i], type(payloads[i])) for i in order])
        for actual, expected in zip(restored, [payloads[i] for i in order], strict=True):
            assert type(actual) is type(expected)
            np.testing.assert_equal(vars(actual), vars(expected))
        assert StoredNode.deserialize(client.get_node_blobs([ids[2]])[0]).record.startswith(b'custom:')
        client.delete_multi_node(ids)
        assert fs.blobs == {}


@pytest.mark.parametrize('payload', [JsonPayload('你好'), BinaryPayload(-123), SamplePayload(text='pickle')])
def test_single_read_uses_selected_payload_codec(payload):
    client = RedisClient(redis_client=FakeRedis(), fs_client=MemoryFS())
    ref = client.put_node(payload)
    assert isinstance(ref, str)
    assert client.get_node(ref, payload_type=type(payload)) == payload
    assert client.get_node_blobs([ref]) == [payload.serialize()]


def test_mixed_batch_keeps_order_and_duplicates_in_one_read(monkeypatch):
    raw = FakeRedis()
    client = RedisClient(redis_client=raw, fs_client=MemoryFS())
    payloads = [JsonPayload('json'), BinaryPayload(42), SamplePayload(text='pickle')]
    refs = client.put_multi_node(payloads)
    read = Mock(wraps=raw.mget)
    monkeypatch.setattr(raw, 'mget', read)
    order = [2, 0, 1, 0]
    requests = [(refs[i], type(payloads[i])) for i in order]
    assert client.get_multi_node(requests) == [payloads[i] for i in order]
    read.assert_called_once_with([client._node_key(refs[i]) for i in order])


@pytest.mark.parametrize('payload', [JsonPayload('async'), BinaryPayload(42)])
async def test_async_read_forwards_payload_type(payload):
    client = AsyncRedisClient(redis_client=FakeRedis(), fs_client=MemoryFS())
    ref = await client.async_put_node(payload)
    assert await client.async_get_node(ref, payload_type=type(payload)) == payload


def test_reads_require_explicit_types_and_empty_batch_does_no_io():
    raw = Mock(spec=FakeRedis)
    client = AsyncRedisClient(redis_client=raw, fs_client=MemoryFS())
    ref = 'id'
    with pytest.raises(TypeError, match='payload_type'):
        client.get_node(ref)
    with pytest.raises(TypeError, match='payload_type'):
        client.async_get_node(ref)
    with pytest.raises(TypeError, match='payload_type'):
        client.get_multi_node([ref])
    assert client.get_multi_node([]) == []
    assert raw.mock_calls == []


@pytest.mark.parametrize('payload_type', [None, dict, Serializable(), 'JsonPayload'])
@pytest.mark.parametrize('multi', [False, True])
def test_invalid_payload_type_rejected_before_io(payload_type, multi):
    raw = Mock(spec=FakeRedis)
    client = RedisClient(redis_client=raw, fs_client=MemoryFS())
    ref = 'valid'
    with pytest.raises(TypeError, match='payload_type'):
        client.get_multi_node([(ref, JsonPayload), (ref, payload_type)]) if multi else client.get_node(
            ref, payload_type=payload_type
        )
    assert raw.mock_calls == []


@pytest.mark.parametrize('node_id', [None, 1])
@pytest.mark.parametrize('multi', [False, True])
def test_invalid_id_rejected_before_io(multi, node_id, monkeypatch):
    monkeypatch.setitem(sys.modules, 'torch', None)
    raw = Mock(spec=FakeRedis)
    client = RedisClient(redis_client=raw, fs_client=MemoryFS())
    with pytest.raises(TypeError, match='node_id must be a string'):
        client.get_multi_node([
            ('valid', JsonPayload),
            (node_id, BinaryPayload),
        ]) if multi else client.get_node(node_id, payload_type=JsonPayload)
    with pytest.raises(TypeError, match='node_id must be a string'):
        client.get_tensor(node_id)
    assert raw.mock_calls == []


@pytest.mark.parametrize('decoded', [BinaryPayload(7), {'wrong': 'type'}])
@pytest.mark.parametrize('multi', [False, True])
def test_decoder_return_type_is_checked(decoded, multi, monkeypatch):
    client = RedisClient(redis_client=FakeRedis(), fs_client=MemoryFS())
    ref = client.put_node(JsonPayload('value'))
    monkeypatch.setattr(JsonPayload, 'deserialize', staticmethod(lambda blob: decoded))
    with pytest.raises(RedisClientError, match='invalid payload') as caught:
        client.get_multi_node([(ref, JsonPayload)]) if multi else client.get_node(ref, payload_type=JsonPayload)
    assert ref in str(caught.value)
    assert 'JsonPayload' in str(caught.value)
    assert type(decoded).__name__ in str(caught.value)


@pytest.mark.parametrize('source', ['decoder', 'network'])
@pytest.mark.parametrize('multi', [False, True])
def test_read_errors_propagate(source, multi, monkeypatch):
    raw = FakeRedis()
    client = RedisClient(redis_client=raw, fs_client=MemoryFS())
    ref = client.put_node(JsonPayload('value'))
    failure = ValueError('custom decoder failed') if source == 'decoder' else ConnectionError('read failed')
    if source == 'decoder':
        monkeypatch.setattr(JsonPayload, 'deserialize', staticmethod(Mock(side_effect=failure)))
    else:
        monkeypatch.setattr(raw, 'mget', Mock(side_effect=failure))
    with pytest.raises(type(failure), match=str(failure)) as caught:
        client.get_multi_node([(ref, JsonPayload)]) if multi else client.get_node(ref, payload_type=JsonPayload)
    assert caught.value is failure


@pytest.mark.parametrize('multi', [False, True])
def test_typed_read_retains_not_found_code(multi):
    client = RedisClient(redis_client=FakeRedis(), fs_client=MemoryFS())
    ref = 'missing'
    with pytest.raises(RedisClientError, match='not found') as caught:
        client.get_multi_node([(ref, JsonPayload)]) if multi else client.get_node(ref, payload_type=JsonPayload)
    assert caught.value.code == 'not_found'


def test_explicit_base_class_keeps_default_pickle_decoder():
    client = RedisClient(redis_client=FakeRedis(), fs_client=MemoryFS())
    payload = SamplePayload(text='default')
    ref = client.put_node(payload)
    assert client.get_node(ref, payload_type=Serializable) == payload


def test_empty_string_id_is_still_addressable():
    client = RedisClient(redis_client=FakeRedis(), fs_client=MemoryFS())
    payload = JsonPayload('empty-id')
    client._r.set(client._node_key(''), payload.serialize())
    assert client.get_node('', payload_type=JsonPayload) == payload
    assert client.get_multi_node([('', JsonPayload)]) == [payload]
    assert client.get_node_blobs(['']) == [payload.serialize()]
