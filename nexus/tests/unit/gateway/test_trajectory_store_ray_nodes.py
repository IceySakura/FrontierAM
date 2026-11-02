"""Read field ObjectRefs published on two local CPU-only Ray nodes."""

from __future__ import annotations

import asyncio
import copy
import gc
import json
import pickle
import tempfile

from pathlib import Path
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest

from nexus.gateway.trajectory_store import (
    NodePayloadRecord,
    RayTrajectoryRecordStore,
    build_trajectory_record_store,
)
from nexus.gateway.trajectory_store_backends import ray as ray_backend
from nexus.gateway.trajectory_store_backends.ray import _RayDumpRef, _RayNodePayloadRef, _RaySegmentDataRef


def test_dump_and_read_records_published_on_distinct_nodes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ray = pytest.importorskip('ray')
    from ray.cluster_utils import Cluster
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    cluster = Cluster()
    ray_directory = tempfile.mkdtemp(prefix='ray-nodes-')
    monkeypatch.setenv('RAY_TMPDIR', ray_directory)
    try:
        cluster.add_node(
            temp_dir=f'{ray_directory}/ray',
            num_cpus=1,
            num_gpus=0,
            include_dashboard=False,
            object_store_memory=128 * 1024 * 1024,
        )
        cluster.add_node(num_cpus=1, num_gpus=0, object_store_memory=128 * 1024 * 1024)
        ray.init(address=cluster.address, namespace='trajectory-store-node-tests')

        @ray.remote(num_cpus=0)
        class Publisher:
            def __init__(self, config: dict[str, Any]) -> None:
                self.store = build_trajectory_record_store(config)

            async def publish(self, tokens: list[int]) -> tuple[Any, str]:
                [ref] = await self.store.put_many([
                    NodePayloadRecord('trajectory', 0, np.array(tokens, dtype=np.int32))
                ])
                return ref, ray.get_runtime_context().get_node_id()

        config = {'backend': 'ray', 'namespace': 'node-lifetime'}
        publishers = []
        refs = []
        node_ids = [node['NodeID'] for node in ray.nodes() if node['Alive']]
        for index, node_id in enumerate(node_ids):
            publisher = Publisher.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(node_id, soft=False)
            ).remote(config)
            publishers.append(publisher)
            ref, actual_node = ray.get(publisher.publish.remote([index]), timeout=30)
            assert actual_node == node_id, 'Each publisher must execute on its assigned node'
            refs.append(ref)

        assert len({ref.field_references['token_ids'] for ref in refs}) == 2, (
            'The same record published on different nodes must produce independent ObjectRefs'
        )

        extra_publisher = Publisher.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_ids[0], soft=False)
        ).remote(config)
        publishers.append(extra_publisher)
        extra_ref, extra_node = ray.get(extra_publisher.publish.remote([2]), timeout=30)
        assert extra_node == node_ids[0], 'Two publishers must share the first node'

        def checked_dump(indexed_refs: dict[int, Any], shard_path: str) -> None:
            from nexus.gateway.trajectory_store_backends.ray import _dump_payloads_on_node, _get_owner_node_id

            assert not ray.get_runtime_context().get_actor_id(), 'Dump must run as a global task, not in an Actor'
            assert all(isinstance(ref, ray.ObjectRef) for ref in indexed_refs.values()), (
                'Ray must preserve nested ObjectRefs without replacing them with payloads'
            )
            assert all(
                _get_owner_node_id(ref) == ray.get_runtime_context().get_node_id() for ref in indexed_refs.values()
            ), 'Hard affinity must route each field to its original publishing node'
            _dump_payloads_on_node(indexed_refs, shard_path)

        remote_task = ray.remote(num_cpus=0, max_retries=0)(checked_dump)
        calls: list[tuple[str, tuple[int, ...]]] = []

        class RecordingTask:
            def options(self, *, scheduling_strategy: Any) -> Any:
                assert not scheduling_strategy.soft, 'Unavailable nodes must not silently reroute'
                target = remote_task.options(scheduling_strategy=scheduling_strategy)

                class Submission:
                    def remote(self, indexed_refs: dict[int, Any], shard_path: str) -> Any:
                        calls.append((scheduling_strategy.node_id, tuple(indexed_refs)))
                        return target.remote(indexed_refs, shard_path)

                return Submission()

        monkeypatch.setattr(ray_backend, '_get_dump_payloads_task', lambda: RecordingTask())

        async def read_and_release() -> None:
            store = build_trajectory_record_store(config)
            original_refs = [refs[1], refs[0], copy.deepcopy(refs[1]), extra_ref]
            with monkeypatch.context() as context:
                for name in ('get', 'put', 'get_actor'):
                    context.setattr(ray, name, Mock(side_effect=AssertionError('Controller must not fetch payloads')))
                dumped, write_payloads = store.dump_payloads(original_refs, dump_dir=str(tmp_path / 'trajectory_store'))
                await write_payloads
            assert [ref.field_references for ref in dumped] == [
                {'token_ids': 0},
                {'token_ids': 1},
                {'token_ids': 0},
                {'token_ids': 2},
            ], 'Fields from different publishing nodes must use one per-call numbering table'
            assert calls == [(node_ids[1], (0,)), (node_ids[0], (1, 2))], (
                'Shared references are sent once and publishers on the same node share one task'
            )
            directory = tmp_path / 'trajectory_store'
            index = json.loads((directory / 'index.json').read_text())
            saved_payloads = {}
            assert len(index['shards']) == 2, 'One file per publishing node'
            for shard in index['shards']:
                with (directory / shard['file']).open('rb') as stream:
                    header = pickle.load(stream)
                    entries = [pickle.load(stream) for _ in range(header['field_count'])]
                    assert [entry[0] for entry in entries] == shard['field_ids'], 'Match the shared index'
                    saved_payloads.update(entries)
            np.testing.assert_equal(
                saved_payloads,
                {0: [1], 1: [0], 2: [2]},
                err_msg='Save each unique field once across both nodes',
            )
            np.testing.assert_equal(
                await store.get_many(refs, fields=['token_ids']), [{'token_ids': [0]}, {'token_ids': [1]}]
            )
            await store.release_many(refs)
            refs.clear()

            calls.clear()
            [local_ref] = await store.put_many([NodePayloadRecord('controller', 0, np.array([3], dtype=np.int32))])
            local_node = ray.get_runtime_context().get_node_id()
            _, write_payloads = store.dump_payloads([local_ref], dump_dir=str(tmp_path / 'local_trajectory_store'))
            await write_payloads
            assert calls == [(local_node, (0,))], 'Controller-owned objects must also use a remote task'

        asyncio.run(asyncio.wait_for(read_and_release(), timeout=30))
    finally:
        ray.shutdown()
        cluster.shutdown()


@pytest.mark.parametrize('remote_caller', [False, True])
def test_round_robin_load_excludes_controller_and_keeps_returned_refs_alive(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, remote_caller: bool
) -> None:
    ray = pytest.importorskip('ray')
    from ray.cluster_utils import Cluster
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    cluster = Cluster()
    ray_directory = tempfile.mkdtemp(prefix='ray-load-nodes-')
    monkeypatch.setenv('RAY_TMPDIR', ray_directory)
    try:
        cluster.add_node(
            temp_dir=f'{ray_directory}/ray',
            num_cpus=1,
            num_gpus=0,
            include_dashboard=False,
            object_store_memory=128 * 1024 * 1024,
            _system_config={
                'num_workers_soft_limit': 0,
                'idle_worker_killing_time_threshold_ms': 1000,
                'kill_idle_workers_interval_ms': 200,
            },
        )
        for _ in range(2):
            cluster.add_node(num_cpus=1, num_gpus=0, object_store_memory=128 * 1024 * 1024)
        ray.init(address=cluster.address, namespace='trajectory-load-node-tests')
        controller_node = ray.get_runtime_context().get_node_id()
        target_nodes = sorted(node['NodeID'] for node in ray.nodes() if node['NodeID'] != controller_node)
        shards = []
        dumped = []
        expected_fields = {}
        for index in range(5):
            filename = f'old-node-{4 - index}.pkl'
            fields = {2 * index: [index], 2 * index + 1: [-index]}
            with (tmp_path / filename).open('wb') as stream:
                pickle.dump({'format_version': 1, 'field_count': 2}, stream)
                for item in fields.items():
                    pickle.dump(item, stream)
            shards.append({'file': filename, 'field_ids': list(fields)})
            dumped.append(_RayDumpRef('node', {'token_ids': 2 * index, 'logprobs': 2 * index + 1}))
            expected_fields.update(fields)
        (tmp_path / 'index.json').write_text(json.dumps({'format_version': 1, 'shards': shards}))
        dumped.extend([_RayDumpRef('segment', {'token_ids': 0, 'routed_experts': 9}), dumped[0]])
        store = RayTrajectoryRecordStore('round-robin-load')

        @ray.remote(num_cpus=0)
        class RolloutController:
            async def load(self, dump_refs: list[Any], dump_dir: str, ppo_node_id: str) -> list[Any]:
                assert ray.get_runtime_context().get_node_id() != ppo_node_id, 'Run outside the PPO controller node'
                return await RayTrajectoryRecordStore('remote-load').load_payloads(
                    dump_refs, dump_dir=dump_dir, exclude_node_id=ppo_node_id
                )

        caller = (
            RolloutController.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(target_nodes[0], soft=False)
            ).remote()
            if remote_caller
            else None
        )

        async def load_and_wait() -> list[Any]:
            if caller is not None:
                restored = await caller.load.remote(dumped, str(tmp_path), controller_node)
            else:
                restored = await store.load_payloads(dumped, dump_dir=str(tmp_path))
            # NOTE(jerryhrzhou): Only nested field refs remain after load returns and task results leave scope.
            gc.collect()
            await asyncio.sleep(3)
            return restored

        with monkeypatch.context() as context:
            context.setattr(
                ray_backend, '_load_payloads_from_shard', Mock(side_effect=AssertionError('Controller read shard'))
            )
            for name in ('get', 'put'):
                context.setattr(ray, name, Mock(side_effect=AssertionError('Controller touched payload')))
            restored = asyncio.run(load_and_wait())

        assert [type(ref) for ref in restored] == [_RayNodePayloadRef] * 5 + [_RaySegmentDataRef, _RayNodePayloadRef]
        assert restored[0].field_references['token_ids'] == restored[5].field_references['token_ids'], (
            'Shared field IDs must retain one ObjectRef across record types'
        )
        assert restored[0] is not restored[6], 'Repeated records keep independent ref wrappers'
        for index, ref in enumerate(restored[:5]):
            assert {ray_backend._get_owner_node_id(value) for value in ref.field_references.values()} == {
                target_nodes[index % 2]
            }, 'Both fields of each shard must be published on its round-robin node'

        @ray.remote(num_cpus=0, max_retries=0)
        def read_fields(refs: list[Any]) -> list[dict[str, Any]]:
            assert not ray.get_runtime_context().get_actor_id(), 'Read as a global task'
            return [{name: ray.get(value) for name, value in ref.field_references.items()} for ref in refs]

        consumer = read_fields.options(scheduling_strategy=NodeAffinitySchedulingStrategy(target_nodes[0], soft=False))
        payloads = ray.get(consumer.remote(restored), timeout=30)
        assert payloads == [
            {name: expected_fields[field_id] for name, field_id in ref.field_references.items()} for ref in dumped
        ], 'Another worker must read the restored fields after publishing tasks become idle'

        with monkeypatch.context() as context:
            for name in ('get', 'put'):
                context.setattr(ray, name, Mock(side_effect=AssertionError('Controller touched payload')))
            resaved, write_payloads = store.dump_payloads(restored, dump_dir=str(tmp_path / 'resaved'))
            asyncio.run(write_payloads)
        resaved_index = json.loads((tmp_path / 'resaved/index.json').read_text())
        assert {shard['file'] for shard in resaved_index['shards']} == {f'{node_id}.pkl' for node_id in target_nodes}, (
            'Subsequent saves must run on the remote publishing nodes'
        )
        reloaded = asyncio.run(store.load_payloads(resaved, dump_dir=str(tmp_path / 'resaved')))
        assert ray.get(consumer.remote(reloaded), timeout=30) == payloads, 'Preserve values across another save/load'
    finally:
        ray.shutdown()
        cluster.shutdown()
