import asyncio
import json
import pickle
import sys

from dataclasses import fields, replace
from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import UUID, uuid4

import numpy as np
import pytest

from tests.unit.gateway.redis_store.fake_redis import FakeRedis
from tests.unit.gateway.redis_store.file_storage.test_publication import MemoryFS


pytestmark = pytest.mark.integration

from nexus.gateway.trajectory_store_backends.redis import (
    NodeRecord,
    NodeRef,
    RedisTrajectoryRecordStore,
    SegmentRecord,
    SegmentRef,
)
from nexus.gateway.trajectory_store_backends.redis_store.types import Serializable
from nexus.gateway.trajectory_store_backends.redis_store.file_storage.publication import (
    PublicationResult,
    RedisPublication,
)


@pytest.fixture
def store(backend_config):
    with RedisTrajectoryRecordStore(uuid4().hex, backend_config) as store:
        yield store


@pytest.mark.parametrize(
    'values',
    [
        {'token_ids': []},
        {
            'token_ids': [10, 11],
            'logprobs': [],
            'topk_indices': [],
            'topk_values': [],
            'routed_experts': [],
            'multi_modal_items': {},
        },
        {
            'token_ids': [10, 11],
            'logprobs': [-0.1],
            'topk_indices': [[11, 12]],
            'topk_values': [[-0.1, -0.2]],
            'routed_experts': [[1, 2]],
            'multi_modal_items': {'images': [{'pixel_values': [[1.0, 2.0]]}]},
        },
    ],
    ids=['sparse', 'empty-optional-fields', 'all-fields'],
)
def test_node_record_payload_stores_training_fields_directly(values):
    record = NodeRecord('trajectory', 1, **values)
    assert isinstance(record, Serializable)
    with RedisTrajectoryRecordStore(
        'run', {'url': 'redis://127.0.0.1:16379/0', 'redis_client': Mock(spec=FakeRedis), 'fs_client': MemoryFS()}
    ) as store:
        store._client._publication.put = Mock(return_value=[PublicationResult('id')])
        store._put_many([record])
        assert store._client._publication.put.call_args.args[0][0] is record
    restored = NodeRecord.deserialize(record.serialize())
    assert type(restored) is NodeRecord
    np.testing.assert_equal(vars(restored), vars(record))
    for name, value in values.items():
        assert getattr(restored, name) == value
    assert set(vars(record)) == {item.name for item in fields(record)}


def test_segment_payload_preserves_identity_on_append():
    original = {'reward': None, 'binary': b'raw'}
    record = SegmentRecord('trajectory', 'main@1', original)
    with RedisTrajectoryRecordStore(
        'run', {'url': 'redis://127.0.0.1:16379/0', 'redis_client': Mock(spec=FakeRedis), 'fs_client': MemoryFS()}
    ) as store:
        store._client._publication.put = Mock(return_value=[PublicationResult('id')])
        store._put_many([record])
        payload = store._client._publication.put.call_args.args[0][0]
    assert isinstance(payload, Serializable)
    assert type(payload) is SegmentRecord
    assert record.fields is original
    assert original == {'reward': None, 'binary': b'raw'}
    payload = SegmentRecord.deserialize(payload.serialize())
    merged = payload.with_fields({'late': pickle.dumps([1])})
    assert (merged.trajectory_id, merged.segment_id) == ('trajectory', 'main@1')
    assert merged.fields['reward'] == payload.fields['reward']
    assert merged.fields['binary'] == payload.fields['binary']
    assert {name: pickle.loads(value) for name, value in merged.fields.items()} == {
        'reward': None,
        'binary': b'raw',
        'late': [1],
    }


def test_config_and_empty_batches():
    raw = Mock(spec=FakeRedis)
    store = RedisTrajectoryRecordStore(
        'run', {'url': 'redis://127.0.0.1:16379/0', 'redis_client': raw, 'fs_client': MemoryFS()}
    )
    assert store.namespace == 'run'

    async def exercise():
        assert await store.put_many([]) == []
        assert await store.get_many([]) == []
        assert await store.put_fields_many({}) is None
        assert await store.release_many([]) is None

    with store:
        asyncio.run(exercise())
        assert raw.mock_calls == []
    raw.close.assert_called_once_with()


def test_default_url_discovers_ray_head_at_construction(monkeypatch):
    connect = Mock()
    monkeypatch.setattr('nexus.gateway.trajectory_store_backends.redis_store.client.connect_redis', connect)
    monkeypatch.setenv('CHIEF_IP', '192.0.2.99')
    for host in ('192.0.2.10', '192.0.2.11'):
        monkeypatch.setitem(
            sys.modules,
            'ray',
            SimpleNamespace(
                is_initialized=lambda: True,
                nodes=lambda host=host: [
                    {'Alive': True, 'Resources': {}, 'NodeManagerAddress': '192.0.2.20'},
                    {'Alive': True, 'Resources': {'node:__internal_head__': 1}, 'NodeManagerAddress': host},
                ],
            ),
        )
        with RedisTrajectoryRecordStore('run', {'fs_client': MemoryFS()}) as store:
            assert store._client.url == f'redis://{host}:16379/0'
            assert store.config['url'] == store._client.url
            connect.assert_called_with(store._client.url)


def test_default_url_requires_initialized_ray(monkeypatch):
    monkeypatch.setenv('CHIEF_IP', '192.0.2.99')
    monkeypatch.setitem(sys.modules, 'ray', SimpleNamespace(is_initialized=lambda: False))
    connect = Mock()
    monkeypatch.setattr('nexus.gateway.trajectory_store_backends.redis_store.client.connect_redis', connect)
    with pytest.raises(RuntimeError, match='initialized Ray'):
        RedisTrajectoryRecordStore('run', {'fs_client': MemoryFS()})
    connect.assert_not_called()


@pytest.mark.parametrize(
    'ray_module', [None, SimpleNamespace(is_initialized=lambda: False)], ids=['absent', 'uninitialized']
)
def test_explicit_url_does_not_need_ray(monkeypatch, ray_module):
    monkeypatch.setenv('CHIEF_IP', '192.0.2.99')
    monkeypatch.setitem(sys.modules, 'ray', ray_module)
    connect = Mock()
    monkeypatch.setattr('nexus.gateway.trajectory_store_backends.redis_store.client.connect_redis', connect)
    config = {'url': 'redis://192.0.2.20:6390/2', 'fs_client': MemoryFS()}
    with RedisTrajectoryRecordStore('run', config) as store:
        assert store._client.url == config['url']
        assert store.config == config
        connect.assert_called_once_with(config['url'])


# Pickles captured before simplifying the shared ref base.
@pytest.mark.parametrize(
    ('ref_type', 'previous_pickle'),
    [
        (
            NodeRef,
            b'\x80\x04\x95h\x00\x00\x00\x00\x00\x00\x00\x8c-nexus.gatew'
            b'ay.trajectory_store_back'
            b'ends.redis\x94\x8c\x07NodeRef\x94\x93\x94)'
            b'\x81\x94}\x94(\x8c\tnamespace\x94\x8c\x03run\x94\x8c'
            b'\trecord_id\x94\x8c\x02id\x94ub.',
        ),
        (
            SegmentRef,
            b'\x80\x04\x95k\x00\x00\x00\x00\x00\x00\x00\x8c-nexus.gatew'
            b'ay.trajectory_store_back'
            b'ends.redis\x94\x8c\nSegmentRef\x94'
            b'\x93\x94)\x81\x94}\x94(\x8c\tnamespace\x94\x8c\x03ru'
            b'n\x94\x8c\trecord_id\x94\x8c\x02id\x94ub.',
        ),
    ],
)
def test_ref_identity_and_serialization(ref_type, previous_pickle):
    ref = ref_type('run', 'id')
    assert [item.name for item in fields(ref)] == ['namespace', 'record_id']
    assert pickle.loads(pickle.dumps(ref)) == ref
    restored = pickle.loads(previous_pickle)
    assert type(restored) is ref_type
    assert restored == ref
    assert hash(restored) == hash(ref)
    assert hash(ref_type('run', 'id')) == hash(ref)
    assert ref_type('other', 'id') != ref
    assert ref_type('run', 'other') != ref
    assert NodeRef('run', 'id') != SegmentRef('run', 'id')
    for name, value in [('backend', 'redis'), ('data_unit_id', 'redis'), ('endpoint', None)]:
        with pytest.raises(TypeError):
            ref_type(**vars(ref), **{name: value})


@pytest.mark.parametrize(
    'selected_fields,expected',
    [
        (
            None,
            [{'token_ids': [1, 2], 'logprobs': [-0.1]}, {'token_ids': [3]}, {'reward': None, 'binary': b'raw'}],
        ),
        (
            ['logprobs', 'missing', 'token_ids', 'reward'],
            [{'logprobs': [-0.1], 'token_ids': [1, 2]}, {'token_ids': [3]}, {'reward': None}],
        ),
        (
            {'logprobs', 'missing', 'token_ids', 'reward'},
            [{'logprobs': [-0.1], 'token_ids': [1, 2]}, {'token_ids': [3]}, {'reward': None}],
        ),
        (('missing',), [{}, {}, {}]),
        ([], [{}, {}, {}]),
    ],
    ids=['all', 'partial-list', 'partial-set', 'missing-only', 'empty'],
)
async def test_get_many_omits_missing_fields_and_preserves_ref_order(store, selected_fields, expected):
    refs = await store.put_many([
        NodeRecord('trajectory', 0, [1, 2], logprobs=[-0.1]),
        NodeRecord('trajectory', 1, [3]),
        SegmentRecord('trajectory', 'main', {'reward': None, 'binary': b'raw'}),
    ])
    assert await store.get_many([*refs, refs[0]], fields=selected_fields) == [*expected, expected[0]]
    assert await store.get_many([], fields=selected_fields) == []
    await store.release_many(refs)
    with pytest.raises(KeyError):
        await store.get_many(refs, fields=selected_fields)


async def test_lifecycle(store):
    node = NodeRecord('trajectory', 1, [10, 11], logprobs=[-0.1])
    segment = SegmentRecord('trajectory', 'main@1', {'reward': None})
    refs = await store.put_many([node, segment, node])
    assert len(set(refs)) == 3
    assert all(UUID(ref.record_id).version == 4 and len(ref.record_id) == 32 for ref in refs)
    assert isinstance(refs[0], NodeRef)
    assert isinstance(refs[1], SegmentRef)
    replay = await store.put_many([node, segment])
    assert not set(replay).intersection(refs)
    assert await store.get_many(refs) == [
        {'token_ids': [10, 11], 'logprobs': [-0.1]},
        {'reward': None},
        {'token_ids': [10, 11], 'logprobs': [-0.1]},
    ]
    assert await store.get_many([refs[0]], fields=['token_ids']) == [{'token_ids': [10, 11]}]
    assert await store.get_many(refs, fields=[]) == [{}, {}, {}]
    assert await store.get_many([refs[0]], fields=['routed_experts']) == [{}]
    await store.put_fields_many({refs[1]: {'value': [1, 2], 'reward': None}})
    await store.put_fields_many({refs[1]: {'value': [1, 2]}})
    assert await store.get_many([refs[1]]) == [{'reward': None, 'value': [1, 2]}]
    (initial_snapshot,) = await store.put_many([segment])
    assert await store.get_many([initial_snapshot]) == [{'reward': None}]
    with pytest.raises(ValueError):
        await store.put_fields_many({refs[1]: {'new': 1, 'value': [3]}})
    assert await store.get_many([refs[1]], fields=['new']) == [{}]
    with pytest.raises(TypeError):
        await store.put_fields_many({refs[0]: {'value': 1}})
    await store.release_many([refs[0]])
    assert await store.get_many([refs[2]]) == [{'token_ids': [10, 11], 'logprobs': [-0.1]}]
    await store.release_many([*refs, *replay, initial_snapshot])
    await store.release_many(refs)
    with pytest.raises(KeyError):
        await store.get_many([refs[0]])
    with pytest.raises(KeyError):
        await store.put_fields_many({refs[1]: {'other': 1}})
    recreated = await store.put_many([node, segment])
    assert not set(recreated).intersection(refs)
    await store.release_many(recreated)


async def test_preflight_and_identity(store):
    original = NodeRecord('a', 1, [1])
    changed = replace(original, token_ids=[2])
    fresh = NodeRecord('a', 2, [3])
    distinct = await store.put_many([original, changed])
    assert distinct[0] != distinct[1]
    assert await store.get_many(distinct) == [{'token_ids': [1]}, {'token_ids': [2]}]
    (ref,) = await store.put_many([original])
    fresh_refs = await store.put_many([fresh, changed])
    other = RedisTrajectoryRecordStore(store.namespace + '-other', store.config)
    (other_ref,) = await other.put_many([original])
    assert other_ref.record_id != ref.record_id
    assert await other.get_many([ref]) == [{'token_ids': [1]}]
    refs = await store.put_many([replace(original, trajectory_id='b'), SegmentRecord('a', '1', {})])
    assert len({ref.record_id, *(r.record_id for r in refs)}) == 3
    s1, s2 = await store.put_many([SegmentRecord('a', 's1', {}), SegmentRecord('a', 's2', {'x': 1})])
    with pytest.raises(ValueError):
        await store.put_fields_many({s1: {'y': 1}, s2: {'x': 2}})
    assert await store.get_many([s1]) == [{}]
    await store.release_many([*distinct, ref, *fresh_refs, *refs, s1, s2])
    await other.release_many([other_ref])


async def test_concurrent_create_and_append(store):
    record = SegmentRecord('a', 's', {'start': 0})
    results = await asyncio.gather(*(store.put_many([record]) for _ in range(8)))
    ref = results[0][0]
    published = [result[0] for result in results]
    assert len(set(published)) == 8
    await asyncio.gather(*(store.put_fields_many({ref: {f'f{i}': i}}) for i in range(6)))
    assert await store.get_many([ref]) == [{'start': 0, **{f'f{i}': i for i in range(6)}}]
    results = await asyncio.gather(
        *(store.put_fields_many({ref: {'same': i}}) for i in range(2)), return_exceptions=True
    )
    assert sum(result is None for result in results) == 1
    assert sum(isinstance(result, ValueError) for result in results) == 1
    await store.release_many(published)


async def test_cpu_payloads(store):
    np = pytest.importorskip('numpy')
    torch = pytest.importorskip('torch')

    array = np.arange(6).reshape(2, 3)
    tensor = torch.arange(12, dtype=torch.float32, requires_grad=True).reshape(3, 4).t()
    node = NodeRecord('a', 1, [1, 2], routed_experts=array, multi_modal_items={'image': [{'pixel_values': tensor}]})
    (ref,) = await store.put_many([node])
    (got,) = await store.get_many([ref])
    np.testing.assert_array_equal(got['routed_experts'], array)
    assert got['routed_experts'].dtype == array.dtype
    assert torch.equal(got['multi_modal_items']['image'][0]['pixel_values'], tensor)
    segment = SegmentRecord('a', 's', {'array': array, 'tensor': tensor, 'binary': b'raw'})
    (segment_ref,) = await store.put_many([segment])
    read_segment = partial(store._client.get_node, payload_type=SegmentRecord)
    before = await store._run(read_segment, segment_ref.record_id)
    await store.put_fields_many({segment_ref: {'late': []}})
    after = await store._run(read_segment, segment_ref.record_id)
    assert all(after.fields[name] == blob for name, blob in before.fields.items())
    (segment_values,) = await store.get_many([segment_ref])
    np.testing.assert_array_equal(segment_values['array'], array)
    assert segment_values['array'].dtype == array.dtype
    assert torch.equal(segment_values['tensor'], tensor)
    assert segment_values['tensor'].dtype == tensor.dtype
    assert segment_values['tensor'].shape == tensor.shape
    assert segment_values['binary'] == b'raw'
    assert segment.fields['tensor'] is tensor
    node.token_ids.append(99)
    array[:] = -1
    assert await store.get_many([ref], fields=['token_ids']) == [{'token_ids': [1, 2]}]
    (stored,) = await store.get_many([ref])
    np.testing.assert_array_equal(stored['routed_experts'], np.arange(6).reshape(2, 3))
    await store.release_many([ref, segment_ref])


def _fail_on_unpickle():
    raise RuntimeError('unrequested field was decoded')


class UnreadableField:
    def __reduce__(self):
        return _fail_on_unpickle, ()


async def test_projection_only_unpickles_selected_fields(store):
    (ref,) = await store.put_many([SegmentRecord('projection', 's', {'good': [1], 'bad': UnreadableField()})])
    assert await store.get_many([ref], fields=['good']) == [{'good': [1]}]
    assert await store.get_many([ref], fields=['missing', 'good']) == [{'good': [1]}]
    assert await store.get_many([ref], fields=['missing']) == [{}]
    assert await store.get_many([ref], fields=[]) == [{}]
    with pytest.raises(RuntimeError, match='unrequested field'):
        await store.get_many([ref])
    await store.release_many([ref])


async def test_node_projection_decodes_the_whole_payload(store):
    (ref,) = await store.put_many([NodeRecord('projection', 1, [1], logprobs=UnreadableField())])
    with pytest.raises(RuntimeError, match='unrequested field'):
        await store.get_many([ref], fields=['token_ids'])
    await store.release_many([ref])


async def test_payload_metadata_and_empty_optional_fields_survive_storage(store):
    refs = await store.put_many([
        NodeRecord('identity', 1, [], logprobs=[], topk_indices=[], topk_values=[], multi_modal_items={}),
        SegmentRecord('identity', 'main@1', {'reward': None}),
    ])
    await store.put_fields_many({refs[1]: {'late': []}})
    payloads = await store._run(
        store._client.get_multi_node,
        [
            (refs[0].record_id, NodeRecord),
            (refs[1].record_id, SegmentRecord),
        ],
    )
    assert type(payloads[0]) is NodeRecord
    assert type(payloads[1]) is SegmentRecord
    assert payloads[0].node_id == 1
    assert payloads[1].segment_id == 'main@1'
    for payload in payloads:
        assert payload.trajectory_id == 'identity'
        assert set(vars(payload)) == {item.name for item in fields(payload)}
    assert payloads[0].multi_modal_items == {}
    assert await store.get_many(refs) == [
        {'token_ids': [], 'logprobs': [], 'topk_indices': [], 'topk_values': [], 'multi_modal_items': {}},
        {'reward': None, 'late': []},
    ]
    await store.release_many(refs)


async def test_hash_references_can_address_new_records(store, monkeypatch):
    records = [
        NodeRecord('identity', 1, [1]),
        SegmentRecord('identity', 'main@1', {'initial': None}),
    ]
    with monkeypatch.context() as patch:
        patch.setattr(
            'nexus.gateway.trajectory_store_backends.redis_store.client.new_node_id',
            iter(['a' * 64, 'b' * 64]).__next__,
        )
        refs = await store.put_many(records)
    assert [ref.record_id for ref in refs] == ['a' * 64, 'b' * 64]
    assert await store.get_many(refs[:1], fields=['token_ids']) == [{'token_ids': [1]}]
    await store.put_fields_many({refs[1]: {'late': [2]}})
    assert await store.get_many(refs[1:]) == [{'initial': None, 'late': [2]}]
    await store.release_many(refs)
    with pytest.raises(KeyError):
        await store.get_many(refs[:1])


async def test_invalid_batch_does_not_publish_earlier_records(store, monkeypatch):
    from redis.client import Pipeline
    from redis.cluster import ClusterPipeline

    writes = []

    def execute(*args, **kwargs):
        writes.append('execute')
        raise AssertionError('invalid batch executed a pipeline')

    monkeypatch.setattr(Pipeline, 'execute', execute)
    monkeypatch.setattr(ClusterPipeline, 'execute', execute)

    valid = NodeRecord('invalid-batch', 1, [1])
    for invalid in (
        object(),
        SegmentRecord('t', 's', {1: 'bad name'}),
        SegmentRecord('t', 's', {'unpickleable': lambda: None}),
        NodeRecord('t', 2, [2], routed_experts=lambda: None),
    ):
        with pytest.raises((TypeError, AttributeError)):
            await store.put_many([valid, invalid])
    assert writes == []


async def test_concurrent_publish_with_same_business_identity_creates_independent_records(store):
    results = await asyncio.gather(
        store.put_many([NodeRecord('conflict', 1, [1])]),
        store.put_many([NodeRecord('conflict', 1, [2])]),
    )
    refs = [result[0] for result in results]
    assert refs[0] != refs[1]
    assert await store.get_many(refs) == [{'token_ids': [1]}, {'token_ids': [2]}]
    await store.release_many(refs)


@pytest.mark.parametrize(
    'record,ref_type,expected',
    [
        (NodeRecord('process', 1, [10, 11]), 'NodeRef', {'token_ids': [10, 11]}),
        (SegmentRecord('process', 'main@1', {'reward': 1.0}), 'SegmentRef', {'reward': 1.0}),
    ],
)
def test_ref_can_be_read_in_an_independent_process(store, record, ref_type, expected):
    import os
    import subprocess
    import sys

    from pathlib import Path

    (ref,) = asyncio.run(store.put_many([record]))
    config = dict(store.config)
    code = """
import asyncio, json, pickle, sys
from nexus.gateway.trajectory_store_backends import redis as record_store
data = json.load(sys.stdin)
store = record_store.RedisTrajectoryRecordStore(data['namespace'], data['config'])
pickled_ref = pickle.loads(bytes.fromhex(data['pickled_ref']))
assert type(pickled_ref) is getattr(record_store, data['ref_type'])
print(json.dumps(asyncio.run(store.get_many([pickled_ref]))))
"""
    env = {**os.environ, 'PYTHONPATH': str(Path(__file__).resolve().parents[4])}
    result = subprocess.run(
        [sys.executable, '-c', code],
        input=json.dumps({
            'namespace': store.namespace,
            'config': config,
            'ref_type': ref_type,
            'pickled_ref': pickle.dumps(ref).hex(),
        }),
        env=env,
        text=True,
        capture_output=True,
        timeout=15,
        check=True,
    )
    assert json.loads(result.stdout) == [expected]
    asyncio.run(store.release_many([ref]))


async def test_real_backend_partial_operations_can_be_retried(store, monkeypatch):
    from nexus.gateway.trajectory_store_backends.redis_store.client import RedisClient
    from nexus.gateway.trajectory_store_backends.redis_store.types import RedisClientError

    client_type = RedisClient
    create = RedisPublication.put
    append = client_type.compare_and_set_many
    delete = RedisPublication.delete
    created = []

    def partial_create(self, payloads):
        created.extend(result.node_id for result in create(self, payloads[:1]))
        raise RedisClientError('injected partial create', code='unavailable')

    def partial_append(self, updates):
        append(self, updates[:1])
        raise RedisClientError('injected partial append', code='unavailable')

    def partial_delete(self, refs, *, reference_urls):
        delete(self, refs[:1], reference_urls=reference_urls[:1])
        raise RedisClientError('injected partial delete', code='unavailable')

    records = [SegmentRecord('partial', str(i), {'initial': i}) for i in range(3)]
    monkeypatch.setattr(RedisPublication, 'put', partial_create)
    with pytest.raises(RedisClientError, match='partial create'):
        await store.put_many(records)
    monkeypatch.setattr(RedisPublication, 'put', create)
    refs = await store.put_many(records)
    assert all(ref.record_id != created[0] for ref in refs)
    orphan = SegmentRef(store.namespace, created[0])
    assert await store.get_many([orphan]) == [{'initial': 0}]
    await store.release_many([orphan])

    updates = {ref: {'late': i} for i, ref in enumerate(refs)}
    monkeypatch.setattr(client_type, 'compare_and_set_many', partial_append)
    with pytest.raises(RedisClientError, match='partial append'):
        await store.put_fields_many(updates)
    monkeypatch.setattr(client_type, 'compare_and_set_many', append)
    assert await store.get_many(refs) == [{'initial': 0, 'late': 0}, {'initial': 1}, {'initial': 2}]
    await store.put_fields_many(updates)
    assert await store.get_many(refs) == [{'initial': i, 'late': i} for i in range(3)]

    monkeypatch.setattr(RedisPublication, 'delete', partial_delete)
    with pytest.raises(RedisClientError, match='partial delete'):
        await store.release_many(refs)
    monkeypatch.setattr(RedisPublication, 'delete', delete)
    with pytest.raises(KeyError):
        await store.get_many(refs[:1])
    assert len(await store.get_many(refs[1:])) == 2
    await store.release_many(refs)
    for ref in refs:
        with pytest.raises(KeyError):
            await store.get_many([ref])


async def test_namespace_labels_do_not_restrict_record_access(store):
    node_ref, segment_ref = await store.put_many([
        NodeRecord('labels', 1, [1]),
        SegmentRecord('labels', 's', {'initial': 1}),
    ])
    other = RedisTrajectoryRecordStore(store.namespace + '-other', store.config)
    alias = replace(segment_ref, namespace='label')
    assert await other.get_many([node_ref, alias]) == [{'token_ids': [1]}, {'initial': 1}]
    await other.put_fields_many({segment_ref: {'late': 2}})
    assert await store.get_many([alias]) == [{'initial': 1, 'late': 2}]
    await other.release_many([node_ref, segment_ref, alias])
    for ref in [node_ref, segment_ref]:
        with pytest.raises(KeyError):
            await store.get_many([ref], fields=[])
