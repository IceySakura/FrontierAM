import asyncio
import copy
import importlib.util
import json
import os
import pickle
import sys
import threading
import weakref

from collections.abc import Collection, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest

from tests.unit.gateway.redis_store.fake_redis import FakeRedis
from tests.unit.gateway.redis_store.file_storage.test_publication import MemoryFS

from nexus.gateway.trajectory_store import (
    GarnetTrajectoryRecordStore,
    InMemoryTrajectoryRecordStore,
    NodePayloadRecord,
    NodePayloadRef,
    RayTrajectoryRecordStore,
    RedisTrajectoryRecordStore,
    SegmentDataRecord,
    SegmentDataRef,
    TrajectoryRecordStore,
    build_trajectory_record_store,
    launch_servers,
    stop_servers,
)
from nexus.gateway.trajectory_store_backends.redis_store.file_storage.types import Endpoint


@pytest.fixture(scope='module')
def ray_runtime(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Any]:
    ray = pytest.importorskip('ray')
    ray.init(
        address='local',
        namespace='trajectory-store-tests',
        num_cpus=2,
        num_gpus=0,
        include_dashboard=False,
        object_store_memory=128 * 1024 * 1024,
        _node_ip_address='127.0.0.1',
        _temp_dir=str(tmp_path_factory.mktemp('ray')),
    )
    try:
        yield ray
    finally:
        ray.shutdown()


@pytest.fixture(scope='module')
def garnet_config():
    command = os.environ.get('GARNET_SERVER_COMMAND')
    if not command:
        pytest.skip('Set GARNET_SERVER_COMMAND to test the real Garnet backend')
    from nexus.gateway.trajectory_store_backends.garnet_server import GarnetServer

    server = GarnetServer({'server_command': json.loads(command), 'memory': '1g'}, '127.0.0.1')
    try:
        yield {'servers': [server.start()]}
    finally:
        server.stop()


@pytest.fixture(params=['memory', 'ray', 'redis', 'garnet'])
def store(request) -> TrajectoryRecordStore:
    namespace = f'contract-{request.node.name}'
    if request.param == 'memory':
        yield InMemoryTrajectoryRecordStore(namespace)
        return

    if request.param == 'garnet':
        with build_trajectory_record_store({
            'backend': 'garnet', 'namespace': namespace, 'garnet': request.getfixturevalue('garnet_config'),
        }) as store:
            yield store
        return

    if request.param == 'redis':
        with RedisTrajectoryRecordStore(namespace, request.getfixturevalue('redis_config')) as store:
            yield store
        return

    request.getfixturevalue('ray_runtime')
    yield RayTrajectoryRecordStore(namespace)


class TestTrajectoryRecordStoreContract:
    async def test_complete_snapshots_preserve_sparse_node_fields(self, store):
        complete_ref, token_only_ref = await store.put_many(
            [
                NodePayloadRecord(
                    trajectory_id='trajectory-1',
                    node_id=1,
                    token_ids=np.array([10, 11], dtype=np.int32),
                    logprobs=np.array([-0.1], dtype=np.float32),
                    routed_experts=[[4, 5]],
                ),
                NodePayloadRecord(trajectory_id='trajectory-1', node_id=2, token_ids=np.array([20], dtype=np.int32)),
            ]
        )

        np.testing.assert_equal(
            await store.get_many([complete_ref, token_only_ref]),
            [
                {'token_ids': [10, 11], 'logprobs': np.array([-0.1], dtype=np.float32), 'routed_experts': [[4, 5]]},
                {'token_ids': [20]},
            ],
        )

    async def test_create_select_append_and_release(self, store: TrajectoryRecordStore) -> None:
        node_ref, segment_ref = await store.put_many(
            [
                NodePayloadRecord(
                    trajectory_id='trajectory-1',
                    node_id=3,
                    token_ids=np.array([10, 11], dtype=np.int32),
                    logprobs=np.array([-0.1], dtype=np.float32),
                    topk_indices=np.array([[1, 2]], dtype=np.int32),
                    topk_values=np.array([[-0.2, -0.3]], dtype=np.float32),
                    routed_experts=[[4, 5]],
                ),
                SegmentDataRecord('trajectory-1', 'main@3', {'old_log_probs': [0.1]}),
            ]
        )

        np.testing.assert_equal(await store.get_many([node_ref], fields=['token_ids']), [{'token_ids': [10, 11]}])
        np.testing.assert_equal(
            await store.get_many(
                [node_ref],
                fields=['logprobs', 'topk_indices', 'topk_values', 'routed_experts'],
            ),
            [
                {
                    'logprobs': np.array([-0.1], dtype=np.float32),
                    'topk_indices': [[1, 2]],
                    'topk_values': np.array([[-0.2, -0.3]], dtype=np.float32),
                    'routed_experts': [[4, 5]],
                }
            ],
        )
        await store.put_fields_many({segment_ref: {'ref_log_prob': [0.2]}})
        assert await store.get_many([segment_ref], fields=['ref_log_prob']) == [{'ref_log_prob': [0.2]}]
        if not isinstance(store, GarnetTrajectoryRecordStore):
            if isinstance(store, RedisTrajectoryRecordStore):
                await store.put_fields_many({segment_ref: {'ref_log_prob': [0.2]}})
            else:
                with pytest.raises(ValueError, match='conflicting trajectory record fields'):
                    await store.put_fields_many({segment_ref: {'ref_log_prob': [0.2]}})
            with pytest.raises(ValueError, match='conflicting'):
                await store.put_fields_many({segment_ref: {'ref_log_prob': [9.9]}})

        await store.release_many([node_ref, node_ref])
        if isinstance(store, RedisTrajectoryRecordStore):
            with pytest.raises(KeyError):
                await store.get_many([node_ref], fields=['token_ids'])

    @pytest.mark.parametrize(
        ('fields', 'expected'),
        [
            (None, [{'token_ids': [1, 2], 'logprobs': np.array([-0.1], dtype=np.float32)}, {'token_ids': [3]}, {'ref_log_prob': [0.2]}]),
            (['logprobs', 'missing', 'token_ids'], [{'logprobs': np.array([-0.1], dtype=np.float32), 'token_ids': [1, 2]}, {'token_ids': [3]}, {}]),
            ({'logprobs', 'missing', 'token_ids'}, [{'logprobs': np.array([-0.1], dtype=np.float32), 'token_ids': [1, 2]}, {'token_ids': [3]}, {}]),
            (('missing',), [{}, {}, {}]),
            ([], [{}, {}, {}]),
        ],
        ids=['all', 'partial-list', 'partial-set', 'missing-only', 'empty'],
    )
    async def test_get_many_returns_requested_fields_present_in_each_ref(
        self, store: TrajectoryRecordStore, fields: Collection[str] | None, expected: list[dict[str, Any]]
    ) -> None:
        refs = await store.put_many(
            [
                NodePayloadRecord(
                    'trajectory-1', 0, np.array([1, 2], dtype=np.int32), logprobs=np.array([-0.1], dtype=np.float32)
                ),
                NodePayloadRecord('trajectory-1', 1, np.array([3], dtype=np.int32)),
                SegmentDataRecord('trajectory-1', 'main', {'ref_log_prob': [0.2]}),
            ]
        )

        np.testing.assert_equal(await store.get_many([*refs, refs[0]], fields=fields), [*expected, expected[0]])
        assert await store.get_many([], fields=fields) == []

    async def test_republishing_business_identity_returns_independent_records(
        self, store: TrajectoryRecordStore
    ) -> None:
        original = NodePayloadRecord('trajectory-1', 3, np.array([10], dtype=np.int32))
        first, duplicate, changed = await store.put_many(
            [
                original,
                original,
                NodePayloadRecord('trajectory-1', 3, np.array([20], dtype=np.int32)),
            ]
        )
        [replayed] = await store.put_many([original])
        assert len({first, duplicate, changed, replayed}) == 4
        assert await store.get_many([first, duplicate, changed, replayed]) == [
            {'token_ids': [10]},
            {'token_ids': [10]},
            {'token_ids': [20]},
            {'token_ids': [10]},
        ]
        await store.release_many([first])
        assert await store.get_many([duplicate, replayed]) == [{'token_ids': [10]}, {'token_ids': [10]}]
        await store.release_many([duplicate, changed, replayed])

    async def test_media_records_carry_processor_tensors(self, store: TrajectoryRecordStore) -> None:
        media_record_ref, text_record_ref = await store.put_many(
            [
                NodePayloadRecord(
                    'trajectory-1',
                    0,
                    np.array([1, 2], dtype=np.int32),
                    multi_modal_items={'images': ['tensor']},
                ),
                NodePayloadRecord('trajectory-1', 1, np.array([3, 4], dtype=np.int32)),
            ]
        )

        [image_payload, text_payload] = await store.get_many([media_record_ref, text_record_ref])

        assert image_payload['multi_modal_items'] == {'images': ['tensor']}
        np.testing.assert_equal(text_payload, {'token_ids': [3, 4]})

    async def test_ref_copy_and_pickle_preserve_type_identity_and_fields(self, store: TrajectoryRecordStore) -> None:
        refs = await store.put_many(
            [
                NodePayloadRecord(
                    'roundtrip', 0, np.array([1, 2], dtype=np.int32), routed_experts=np.array([[1, 2]], dtype=np.int32)
                ),
                SegmentDataRecord('roundtrip', 'main', {'values': [3, 4]}),
            ]
        )
        for ref, ref_type in zip(refs, [NodePayloadRef, SegmentDataRef], strict=True):
            restored_refs = [copy.deepcopy(ref), pickle.loads(pickle.dumps(ref))]
            for restored in restored_refs:
                assert isinstance(restored, ref_type)
                assert type(restored) is type(ref)
                if isinstance(store, (RayTrajectoryRecordStore, GarnetTrajectoryRecordStore)):
                    assert restored != ref, 'Field snapshot refs use object identity after copying or unpickling'
                    assert len({ref, restored}) == 2, 'Independent field snapshots must remain distinct set entries'
                    if isinstance(store, GarnetTrajectoryRecordStore):
                        assert restored.field_references == ref.field_references, 'Keep field locations and sizes'
                        assert all(location.nbytes > 0 for location in restored.field_references.values())
                else:
                    assert restored == ref
                    assert hash(restored) == hash(ref)
                    assert {ref: 'found'}[restored] == 'found'
                [payload] = await store.get_many([restored])
                if isinstance(ref, NodePayloadRef):
                    assert payload['token_ids'].tolist() == [1, 2]
                    np.testing.assert_array_equal(payload['routed_experts'], [[1, 2]])
                else:
                    assert payload == {'values': [3, 4]}

    async def test_returned_fields_survive_later_reads_release_and_client_close(self, store: TrajectoryRecordStore) -> None:
        refs = await store.put_many([
            SegmentDataRecord('buffer-lifetime', 'first', {
                'array': np.arange(1024, dtype=np.int32),
                'bytes': b'\x00first\xff',
                'empty': b'',
            }),
            SegmentDataRecord('buffer-lifetime', 'second', {'array': np.arange(8, dtype=np.float64)}),
        ])
        [retained] = await store.get_many(refs[:1])
        np.testing.assert_array_equal((await store.get_many(refs[1:]))[0]['array'], np.arange(8, dtype=np.float64))
        await store.release_many(refs)
        if hasattr(store, 'close'):
            store.close()
        np.testing.assert_array_equal(retained['array'], np.arange(1024, dtype=np.int32))
        assert type(retained['bytes']) is bytes and retained['bytes'] == b'\x00first\xff'
        assert type(retained['empty']) is bytes and retained['empty'] == b''

    async def test_many_selected_fields_preserve_order_across_batches(self, store: TrajectoryRecordStore) -> None:
        refs = await store.put_many([
            SegmentDataRecord('batch-order', str(index), {
                'array': np.full(8, index, dtype=np.int32), 'bytes': bytes([index]), 'other': index + 100,
            })
            for index in range(65)
        ])
        order = list(reversed(range(len(refs))))
        payloads = await store.get_many([refs[index] for index in order], fields=['bytes', 'array', 'missing'])
        for index, payload in zip(order, payloads, strict=True):
            assert set(payload) == {'bytes', 'array'}
            assert payload['bytes'] == bytes([index])
            np.testing.assert_array_equal(payload['array'], np.full(8, index, dtype=np.int32))
        assert await store.get_many(refs, fields=[]) == [{} for _ in refs]
        await store.release_many(refs)


class TestInMemoryTrajectoryRecordStore:
    async def test_put_get_share_fields_without_serialization(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fail_pickle(*args, **kwargs) -> None:
            raise AssertionError('inline put/get must not pickle')

        monkeypatch.setattr(pickle, 'dumps', fail_pickle)
        monkeypatch.setattr(pickle, 'loads', fail_pickle)
        tokens = np.array([1, 2], dtype=np.int32)
        experts = np.ones((2, 2), dtype=np.int32)
        store = InMemoryTrajectoryRecordStore('inline')
        [ref] = await store.put_many([NodePayloadRecord('trajectory', 0, tokens, routed_experts=experts)])
        [payload] = await store.get_many([ref])
        assert payload['token_ids'] is tokens
        assert payload['routed_experts'] is experts
        assert (await store.get_many([ref], fields=['token_ids']))[0]['token_ids'] is tokens
        assert await store.get_many([ref], fields=[]) == [{}]

    async def test_each_put_is_independent_and_refs_own_payload_lifetime(self) -> None:
        store = InMemoryTrajectoryRecordStore('inline')
        experts = np.ones((2, 2))
        weak_experts = weakref.ref(experts)
        record = NodePayloadRecord('trajectory', 0, np.array([1, 2], dtype=np.int32), routed_experts=experts)
        [ref] = await store.put_many([record])
        [other] = await store.put_many([record])
        assert ref != other
        assert len({ref, other}) == 2
        await store.put_many([NodePayloadRecord('trajectory', 0, np.array([99], dtype=np.int32))])
        del record, experts, other
        await store.release_many([ref, ref])
        del store
        assert weak_experts() is not None
        reader = InMemoryTrajectoryRecordStore('inline')
        assert (await reader.get_many([ref]))[0]['token_ids'].tolist() == [1, 2]
        del ref
        assert weak_experts() is None

    async def test_segment_append_is_local_and_rejects_duplicate_fields(self) -> None:
        store = InMemoryTrajectoryRecordStore('inline')
        first, second = await store.put_many(
            [
                SegmentDataRecord('trajectory', 'first', {'values': [1]}),
                SegmentDataRecord('trajectory', 'second', {'values': [2]}),
            ]
        )
        copied = copy.deepcopy(first)
        original_hash = hash(first)
        await store.put_fields_many({first: {'experts': np.ones((2, 2))}})
        with pytest.raises(ValueError, match='conflicting'):
            await store.put_fields_many({first: {'experts': np.ones((2, 2))}})
        assert hash(first) == original_hash
        assert await store.get_many([copied], fields=['experts']) == [{}]
        with pytest.raises(ValueError, match='conflicting'):
            await store.put_fields_many({first: {'new': [3]}, second: {'values': [99]}})

    async def test_empty_batches_and_wrong_refs(self) -> None:
        store = InMemoryTrajectoryRecordStore('inline')
        assert await store.put_many([]) == []
        assert await store.get_many([]) == []
        await store.release_many([])
        await store.put_fields_many({})
        [ref] = await store.put_many([NodePayloadRecord('trajectory', 0, np.array([1], dtype=np.int32))])
        with pytest.raises(TypeError, match='segment'):
            await store.put_fields_many({ref: {'new': [1]}})


class TestTrajectoryRecordStoreFactory:
    @pytest.mark.parametrize('backend', ['memory', 'ray', 'redis'])
    def test_launch_servers_returns_client_config_without_handler(self, backend: str) -> None:
        config = {'backend': backend, 'namespace': 'launch-test'}
        client_config, handler = launch_servers(config)
        assert client_config is not config
        assert client_config == config
        assert handler is None

    def test_launch_servers_rejects_unknown_backend(self) -> None:
        with pytest.raises(ValueError, match='unsupported trajectory_store.backend'):
            launch_servers({'backend': 'unknown'})

    def test_stop_servers_calls_handler_and_accepts_none(self) -> None:
        handler = Mock()

        stop_servers(handler)
        stop_servers(None)

        handler.stop.assert_called_once_with()

    def test_factory_selects_backend(self):
        store = build_trajectory_record_store({'backend': 'memory', 'namespace': 'run-4'})
        assert isinstance(store, InMemoryTrajectoryRecordStore)


class TestRedisTrajectoryRecordStore:
    @pytest.mark.parametrize(
        'config', [{}, {'url': None}, {'url': 'redis://explicit:16379/0'}], ids=['default', 'none', 'explicit']
    )
    def test_store_delegates_url_resolution_to_client(self, monkeypatch, config):
        monkeypatch.setitem(sys.modules, 'ray', None)
        client = Mock(url=config.get('url') or 'redis://192.0.2.10:16379/0')
        factory = Mock(return_value=client)
        monkeypatch.setattr('nexus.gateway.trajectory_store_backends.redis.RedisClient', factory)
        with RedisTrajectoryRecordStore('run', config) as store:
            factory.assert_called_once_with(**config)
            assert store.config == {'url': client.url, **config}
        client.close.assert_called_once_with()

    def test_launch_fs_configuration_does_not_start_services(self, monkeypatch):
        monkeypatch.setitem(sys.modules, 'ray', None)
        monkeypatch.setitem(sys.modules, 'zmq', None)
        config = {
            'backend': 'redis',
            'namespace': 'run',
            'redis': {
                'url': 'redis://127.0.0.1:16379/0',
                'controller_endpoint': {'host': '127.0.0.1', 'port': 17000},
                'fs_timeout': None,
                'io_workers': 8,
            },
        }
        client_config, handler = launch_servers(config)
        assert client_config == config
        assert handler is None

    def test_factory_selects_redis_backend(self, monkeypatch) -> None:
        pytest.importorskip('redis')
        connect = Mock()
        fs = Mock()
        create_fs = Mock(return_value=fs)
        monkeypatch.setattr('nexus.gateway.trajectory_store_backends.redis_store.client.connect_redis', connect)
        monkeypatch.setattr(
            'nexus.gateway.trajectory_store_backends.redis_store.file_storage.client.FSClient', create_fs
        )
        monkeypatch.setenv('CHIEF_IP', '192.0.2.99')
        monkeypatch.setitem(
            sys.modules,
            'ray',
            SimpleNamespace(
                is_initialized=lambda: True,
                nodes=lambda: [
                    {'Alive': True, 'Resources': {}, 'NodeManagerAddress': '192.0.2.20'},
                    {
                        'Alive': True,
                        'Resources': {'node:__internal_head__': 1},
                        'NodeManagerAddress': '192.0.2.10',
                    },
                ],
            ),
        )
        with build_trajectory_record_store({'backend': 'redis', 'namespace': 'run-1'}) as store:
            assert isinstance(store, RedisTrajectoryRecordStore)
            assert store.namespace == 'run-1'
            assert store.config == {'url': 'redis://192.0.2.10:16379/0'}
            assert store._client.url == 'redis://192.0.2.10:16379/0'
            create_fs.assert_called_once_with(Endpoint('192.0.2.10', 17000), timeout=120, io_threads=32)
        connect.assert_called_once_with('redis://192.0.2.10:16379/0')
        fs.close.assert_called_once_with()
        connect.return_value.close.assert_called_once_with()

    def test_factory_keeps_optional_redis_config(self, monkeypatch) -> None:
        pytest.importorskip('redis')
        connect = Mock()
        fs = Mock()
        create_fs = Mock(return_value=fs)
        monkeypatch.setattr('nexus.gateway.trajectory_store_backends.redis_store.client.connect_redis', connect)
        monkeypatch.setattr(
            'nexus.gateway.trajectory_store_backends.redis_store.file_storage.client.FSClient', create_fs
        )
        monkeypatch.setitem(sys.modules, 'ray', None)
        config = {
            'url': 'redis://127.0.0.1:16379/0',
            'controller_endpoint': {'host': '127.0.0.1', 'port': 17100},
        }
        with build_trajectory_record_store({'backend': 'redis', 'namespace': 'run-1', 'redis': config}) as store:
            assert store.config == config
            assert store.namespace == 'run-1'
            create_fs.assert_called_once_with(Endpoint('127.0.0.1', 17100), timeout=120, io_threads=32)
        connect.assert_called_once_with('redis://127.0.0.1:16379/0')
        fs.close.assert_called_once_with()
        connect.return_value.close.assert_called_once_with()

    def test_factory_requires_initialized_ray_for_default_controller(self, monkeypatch) -> None:
        monkeypatch.delenv('CHIEF_IP', raising=False)
        monkeypatch.setitem(sys.modules, 'ray', SimpleNamespace(is_initialized=lambda: False))
        raw = Mock(spec=FakeRedis)
        create_fs = Mock(side_effect=AssertionError('head discovery must fail before FS connects'))
        monkeypatch.setattr(
            'nexus.gateway.trajectory_store_backends.redis_store.file_storage.client.FSClient', create_fs
        )
        with pytest.raises(RuntimeError, match='initialized Ray'):
            build_trajectory_record_store({'backend': 'redis', 'namespace': 'run-1', 'redis': {'redis_client': raw}})
        create_fs.assert_not_called()
        raw.close.assert_called_once_with()

    def test_factory_borrowed_connections_need_no_ray(self, monkeypatch) -> None:
        monkeypatch.setitem(sys.modules, 'ray', None)
        raw = Mock(spec=FakeRedis)
        config = {'redis_client': raw, 'fs_client': MemoryFS()}
        with build_trajectory_record_store({'backend': 'redis', 'namespace': 'run-1', 'redis': config}) as store:
            assert store.config == config
            assert store._client._r is raw
        raw.close.assert_called_once_with()

    def test_factory_rejects_empty_namespace(self) -> None:
        with pytest.raises(ValueError, match='namespace'):
            build_trajectory_record_store({'backend': 'redis', 'namespace': ''})

    async def test_rejects_non_redis_segment_ref_for_late_fields(self) -> None:
        pytest.importorskip('redis')
        with RedisTrajectoryRecordStore('run-1', {'redis_client': FakeRedis(), 'fs_client': MemoryFS()}) as store:
            with pytest.raises(TypeError, match='late fields'):
                await store.put_fields_many({SegmentDataRef(): {'ref_log_prob': [0.2]}})


@pytest.mark.skipif(importlib.util.find_spec('ray') is None, reason='ray is not installed')
class TestRayTrajectoryRecordStore:
    async def test_dump_payloads_preserves_independent_refs_and_shared_field_snapshots(
        self, ray_runtime: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        store = RayTrajectoryRecordStore('dump-snapshots')
        record = NodePayloadRecord(
            'trajectory', 0, np.array([10, 11], dtype=np.int32), logprobs=np.array([-0.1], dtype=np.float32)
        )
        node_ref, segment_ref, reissued = await store.put_many([
            record,
            SegmentDataRecord('trajectory', 'main', {'values': [1]}),
            record,
        ])
        copied = copy.deepcopy(segment_ref)
        pickled = pickle.loads(pickle.dumps(node_ref))
        original_refs = [node_ref, segment_ref, copied, pickled, reissued]
        expected = await store.get_many(original_refs)
        assert len(set(original_refs)) == 5, 'Copied and republished refs must retain independent identities'

        with monkeypatch.context() as context:
            for name in ('put', 'get', 'get_actor'):
                context.setattr(ray_runtime, name, Mock(side_effect=AssertionError('dump must not fetch payloads')))
            dumped, write_payloads = store.dump_payloads(original_refs, dump_dir=str(tmp_path / "trajectory_store"))
            await write_payloads

        await store.put_fields_many({segment_ref: {'ref_log_prob': [0.2]}})
        np.testing.assert_equal(
            await store.get_many([node_ref, copied, pickled, reissued]),
            [expected[0], expected[2], expected[3], expected[4]],
            err_msg='Export must preserve original, copied, pickled, and republished fields',
        )
        assert await store.get_many([segment_ref]) == [{'values': [1], 'ref_log_prob': [0.2]}], (
            'Export must leave late-field writes functional'
        )
        await store.release_many(original_refs)
        np.testing.assert_equal(
            await store.get_many([node_ref, segment_ref]),
            [expected[0], {'values': [1], 'ref_log_prob': [0.2]}],
            err_msg='Export must preserve the direct backend release no-op',
        )
        assert [ref.field_references for ref in dumped] == [
            {'token_ids': 0, 'logprobs': 1},
            {'values': 2},
            {'values': 2},
            {'token_ids': 0, 'logprobs': 1},
            {'token_ids': 3, 'logprobs': 4},
        ], 'Deduplicate shared ObjectRefs while keeping separate publications distinct'
        assert len({id(ref) for ref in dumped}) == 5, 'Each input occurrence must have its own dump ref'

        restored = await store.load_payloads(dumped, dump_dir=str(tmp_path / 'trajectory_store'))
        np.testing.assert_equal(
            await store.get_many(restored), expected, err_msg='Load must recover the saved field snapshots'
        )
        assert len(set(restored)) == len(restored), 'Reconstructed record refs remain independent'
        assert restored[0].field_references['token_ids'] == restored[3].field_references['token_ids'], (
            'Saved shared fields must reuse their new ObjectRef'
        )
        assert restored[0].field_references['token_ids'] != restored[4].field_references['token_ids'], (
            'Distinct saved fields with equal values must remain distinct publications'
        )
        await store.put_fields_many({restored[1]: {'ref_log_prob': [0.3]}})
        assert await store.get_many([restored[1], restored[2]]) == [
            {'values': [1], 'ref_log_prob': [0.3]}, {'values': [1]}
        ], 'Restored segment copies keep independent late-field snapshots'

    async def test_put_many_offloads_one_batch_without_blocking_event_loop(
        self, ray_runtime: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = RayTrajectoryRecordStore('threaded-put')
        loop = asyncio.get_running_loop()
        loop_thread = threading.get_ident()
        put_field = ray_runtime.put
        thread_call = Mock(wraps=asyncio.to_thread)
        monkeypatch.setattr(asyncio, 'to_thread', thread_call)

        def checked_put(value: Any) -> Any:
            assert threading.get_ident() != loop_thread, 'Ray writes must run outside the event-loop thread'
            loop_progress = threading.Event()
            loop.call_soon_threadsafe(loop_progress.set)
            assert loop_progress.wait(timeout=5), 'The event loop must keep running during Ray writes'
            return put_field(value)

        monkeypatch.setattr(ray_runtime, 'put', checked_put)
        refs = await store.put_many(
            [
                NodePayloadRecord(
                    'trajectory', 0, np.array([1, 2], dtype=np.int32), logprobs=np.array([-0.1], dtype=np.float32)
                ),
                SegmentDataRecord('trajectory', 'main', {'values': [3]}),
            ]
        )

        thread_call.assert_called_once()
        np.testing.assert_equal(
            await store.get_many(refs),
            [
                {'token_ids': [1, 2], 'logprobs': np.array([-0.1], dtype=np.float32)},
                {'values': [3]},
            ],
            err_msg='Threaded publication must preserve record order and field values',
        )

    async def test_put_many_propagates_threaded_write_errors(
        self, ray_runtime: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = RayTrajectoryRecordStore('failed-put')
        error = RuntimeError('field publication failed')
        monkeypatch.setattr(ray_runtime, 'put', Mock(side_effect=error))

        with pytest.raises(RuntimeError, match='field publication failed') as raised:
            await store.put_many([NodePayloadRecord('trajectory', 0, np.array([1], dtype=np.int32))])

        assert raised.value is error, 'Threaded publication must propagate the original write error'

    async def test_fields_are_put_directly_without_creating_actors(
        self, ray_runtime: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ray = ray_runtime
        put_field = Mock(wraps=ray.put)
        monkeypatch.setattr(ray, 'put', put_field)
        monkeypatch.setattr(ray, 'remote', Mock(side_effect=AssertionError('store must not create actors')))
        monkeypatch.setattr(ray, 'get_actor', Mock(side_effect=AssertionError('store must not look up actors')))
        monkeypatch.setattr(ray, 'get_runtime_context', Mock(side_effect=AssertionError('store must not locate nodes')))
        store = RayTrajectoryRecordStore('direct-put')
        node_ref, segment_ref = await store.put_many(
            [
                NodePayloadRecord(
                    'trajectory', 0, np.array([1, 2], dtype=np.int32), logprobs=np.array([-0.1], dtype=np.float32)
                ),
                SegmentDataRecord('trajectory', 'main', {'values': [3]}),
            ]
        )
        await store.put_fields_many({segment_ref: {'ref_log_prob': [0.2]}})

        np.testing.assert_equal(
            [call.args for call in put_field.call_args_list],
            [
                ([1, 2],),
                (np.array([-0.1], dtype=np.float32),),
                ([3],),
                ([0.2],),
            ],
            err_msg='Each initial or appended field must be published by its own ray.put call',
        )
        for ref in (node_ref, segment_ref):
            assert set(vars(ref)) == {'field_references'}, 'Refs must only carry the field ObjectRefs'
            assert all(isinstance(value, ray.ObjectRef) for value in ref.field_references.values()), (
                'Each stored field must be a direct Ray ObjectRef'
            )
        np.testing.assert_equal(
            await store.get_many([node_ref, segment_ref]),
            [
                {'token_ids': [1, 2], 'logprobs': np.array([-0.1], dtype=np.float32)},
                {'values': [3], 'ref_log_prob': [0.2]},
            ],
            err_msg='Reads must resolve fields without contacting a store actor',
        )
        with pytest.raises(ValueError, match='conflicting trajectory record fields'):
            await store.put_fields_many({segment_ref: {'ref_log_prob': [0.2]}})
        assert put_field.call_count == 4, 'A duplicate field must be rejected before ray.put'
        await store.release_many([node_ref, segment_ref])

    async def test_copied_refs_keep_independent_field_snapshots(self, ray_runtime: Any) -> None:
        store = RayTrajectoryRecordStore('field-snapshots')
        [ref] = await store.put_many([SegmentDataRecord('trajectory', 'main', {'values': [1]})])
        copied = copy.deepcopy(ref)
        original_hash = hash(ref)
        await store.put_fields_many({ref: {'ref_log_prob': [0.2]}})
        assert hash(ref) == original_hash, 'Appending fields must not change a ref used as a dictionary key'
        assert await store.get_many([copied]) == [{'values': [1]}], 'Copies must retain their field snapshots'
        await store.put_fields_many({copied: {'ref_log_prob': [0.3]}})
        assert await store.get_many([ref, copied]) == [
            {'values': [1], 'ref_log_prob': [0.2]},
            {'values': [1], 'ref_log_prob': [0.3]},
        ], 'Duplicate field checks must only inspect the supplied ref'
        with pytest.raises(ValueError, match='ref_log_prob'):
            await store.put_fields_many({copied: {'ref_log_prob': [0.3]}})
        await store.release_many([ref, ref])
        assert set(ref.field_references) == {'values', 'ref_log_prob'}, 'Release must leave supplied refs unchanged'
        del ref
        assert await store.get_many([copied]) == [{'values': [1], 'ref_log_prob': [0.3]}], (
            'Dropping one ref must not invalidate a surviving copy'
        )
        await store.release_many([copied])

    async def test_empty_segment_refs_remain_distinct_dictionary_keys(self, ray_runtime: Any) -> None:
        store = RayTrajectoryRecordStore('empty-segments')
        record = SegmentDataRecord('trajectory', 'main', {})
        first, second = await store.put_many([record, record])
        assert len({first, second}) == 2, 'Empty refs for the same record must have independent identities'
        await store.put_fields_many({first: {'values': [1]}, second: {'values': [2]}})
        assert await store.get_many([first, second]) == [{'values': [1]}, {'values': [2]}], (
            'Both empty refs must receive their own field updates'
        )
        assert await store.put_many([]) == [], 'An empty put must return no refs'
        assert await store.get_many([]) == [], 'An empty get must return no payloads'
        await store.put_fields_many({})
        await store.release_many([])

    async def test_clients_read_object_refs_independently_of_namespace(self, ray_runtime: Any) -> None:
        config = {'backend': 'ray', 'namespace': 'shared-client'}
        first = build_trajectory_record_store(config)
        second = build_trajectory_record_store(config)
        other = RayTrajectoryRecordStore('other-client')
        [ref] = await first.put_many([NodePayloadRecord('trajectory', 0, np.array([1], dtype=np.int32))])
        [other_ref] = await other.put_many([NodePayloadRecord('trajectory', 0, np.array([2], dtype=np.int32))])
        assert await second.get_many([ref, other_ref]) == [{'token_ids': [1]}, {'token_ids': [2]}], (
            'ObjectRefs must carry everything a separate store client needs to read'
        )
        await first.release_many([ref])
        await other.release_many([other_ref])

    @pytest.mark.parametrize('backend', ['memory', 'ray'])
    async def test_record_is_visible_from_an_independent_ray_worker(self, backend: str, ray_runtime: Any) -> None:
        ray = ray_runtime

        def exercise_from_worker(config, node_ref, segment_ref):
            async def exercise():
                store = build_trajectory_record_store(config)
                node = await store.get_many([node_ref], fields=['token_ids'])
                await store.put_fields_many({segment_ref: {'ref_log_prob': [0.2]}})
                segment = await store.get_many([segment_ref], fields=['ref_log_prob'])
                await store.release_many([node_ref, segment_ref])
                return node, segment

            return asyncio.run(exercise())

        config = {'backend': backend, 'namespace': 'worker-contract'}
        writer = build_trajectory_record_store(config)
        node_ref, segment_ref = await writer.put_many(
            [
                NodePayloadRecord('trajectory-1', 1, np.array([10, 11], dtype=np.int32)),
                SegmentDataRecord('trajectory-1', 'main@1', {'old_log_probs': [0.1]}),
            ]
        )
        worker = ray.remote(exercise_from_worker)
        result = await asyncio.to_thread(
            ray.get,
            worker.remote(config, node_ref, segment_ref),
        )
        np.testing.assert_equal(result, ([{'token_ids': [10, 11]}], [{'ref_log_prob': [0.2]}]))
        np.testing.assert_equal(await writer.get_many([node_ref], fields=['token_ids']), [{'token_ids': [10, 11]}])
        assert await writer.get_many([segment_ref], fields=['ref_log_prob']) == [{}]
        await writer.release_many([node_ref, segment_ref])
