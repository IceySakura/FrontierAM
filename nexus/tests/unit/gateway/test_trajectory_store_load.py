"""Restore field snapshots from shards without collecting their payloads."""

from __future__ import annotations

import asyncio
import json
import pickle
import weakref

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest

from nexus.gateway.trajectory_store import RayTrajectoryRecordStore
from nexus.gateway.trajectory_store_backends import ray as ray_backend
from nexus.gateway.trajectory_store_backends.ray import _RayDumpRef, _RayNodePayloadRef, _RaySegmentDataRef


@pytest.fixture
def ray_store(monkeypatch: pytest.MonkeyPatch) -> RayTrajectoryRecordStore:
    ray = pytest.importorskip('ray')
    monkeypatch.setattr(ray, 'is_initialized', lambda: True)
    monkeypatch.setattr(ray, 'get_runtime_context', lambda: SimpleNamespace(get_node_id=lambda: 'controller'))
    monkeypatch.setattr(ray, 'nodes', lambda: [{'NodeID': 'controller', 'Alive': True}])
    return RayTrajectoryRecordStore('load-tests')


def _write_shards(directory: Path, shards: dict[str, dict[int, Any]]) -> None:
    for filename, fields in shards.items():
        with (directory / filename).open('wb') as stream:
            pickle.dump({'format_version': 1, 'field_count': len(fields)}, stream)
            for item in fields.items():
                pickle.dump(item, stream)
    (directory / 'index.json').write_text(
        json.dumps({
            'format_version': 1,
            'shards': [{'file': filename, 'field_ids': list(fields)} for filename, fields in shards.items()],
        })
    )


async def test_load_preserves_record_order_types_and_shared_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ray_store: RayTrajectoryRecordStore
) -> None:
    ray = pytest.importorskip('ray')
    _write_shards(tmp_path, {'first.pkl': {0: [7, 8], 1: [-0.25]}, 'second.pkl': {2: [[1, 2]]}})
    handles = [object(), object(), object()]
    put = Mock(side_effect=handles)
    monkeypatch.setattr(ray, 'put', put)
    node = _RayDumpRef('node', {'token_ids': 0, 'logprobs': 1})
    dumped = [node, _RayDumpRef('segment', {'token_ids': 0, 'routed_experts': 2}), node]

    restored = await ray_store.load_payloads(dumped, dump_dir=str(tmp_path))

    assert [type(ref) for ref in restored] == [_RayNodePayloadRef, _RaySegmentDataRef, _RayNodePayloadRef]
    assert [ref.field_references for ref in restored] == [
        {'token_ids': handles[0], 'logprobs': handles[1]},
        {'token_ids': handles[0], 'routed_experts': handles[2]},
        {'token_ids': handles[0], 'logprobs': handles[1]},
    ], 'Shared IDs must reuse a single publication across records'
    assert [call.args[0] for call in put.call_args_list] == [[7, 8], [-0.25], [[1, 2]]]
    assert len({id(ref) for ref in restored}) == 3, 'Each input occurrence gets an independent ref'
    restored[0].field_references.clear()
    assert restored[2].field_references['token_ids'] is handles[0], 'Field snapshots must be independent'
    assert node.field_references == {'token_ids': 0, 'logprobs': 1}, 'Loading must leave dump refs reusable'


async def test_load_reads_each_needed_shard_once_and_puts_only_requested_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ray_store: RayTrajectoryRecordStore
) -> None:
    ray = pytest.importorskip('ray')
    _write_shards(tmp_path, {'needed.pkl': {0: [1], 1: [2]}, 'unused.pkl': {2: [3]}})
    (tmp_path / 'unused.pkl').unlink()
    put = Mock(return_value=object())
    monkeypatch.setattr(ray, 'put', put)
    read_shard = Mock(wraps=ray_backend._load_payloads_from_shard)
    monkeypatch.setattr(ray_backend, '_load_payloads_from_shard', read_shard)

    await ray_store.load_payloads([_RayDumpRef('node', {'token_ids': 1})] * 2, dump_dir=str(tmp_path))

    read_shard.assert_called_once_with(str(tmp_path / 'needed.pkl'), {1})
    put.assert_called_once_with([2])


async def test_load_publishes_and_releases_each_payload_before_reading_next(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ray_store: RayTrajectoryRecordStore
) -> None:
    ray = pytest.importorskip('ray')
    _write_shards(tmp_path, {'shard.pkl': {index: np.array([index]) for index in range(3)}})
    previous: list[Any] = []
    events: list[tuple[str, int]] = []
    original_load = pickle.load

    def read(stream: Any) -> Any:
        assert not previous or previous[-1]() is None, 'Release the previous Python payload before reading more'
        item = original_load(stream)
        if isinstance(item, tuple):
            field_id, payload = item
            previous.append(weakref.ref(payload))
            events.append(('read', field_id))
        return item

    def put(payload: Any) -> Any:
        events.append(('put', int(payload[0])))
        return object()

    monkeypatch.setattr(pickle, 'load', read)
    monkeypatch.setattr(ray, 'put', put)
    await ray_store.load_payloads([_RayDumpRef('segment', {'first': 0, 'last': 2})], dump_dir=str(tmp_path))

    assert events == [('read', 0), ('put', 0), ('read', 1), ('read', 2), ('put', 2)]
    assert previous[-1]() is None, 'Do not retain the last payload after load returns'


async def test_load_empty_inputs_and_field_snapshots_need_no_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ray_store: RayTrajectoryRecordStore
) -> None:
    monkeypatch.setattr(ray_store._ray, 'nodes', Mock(side_effect=AssertionError('Empty refs need no dispatch')))
    assert await ray_store.load_payloads([], dump_dir=str(tmp_path)) == []
    restored = await ray_store.load_payloads(
        [_RayDumpRef('node', {}), _RayDumpRef('segment', {})], dump_dir=str(tmp_path)
    )
    assert [type(ref) for ref in restored] == [_RayNodePayloadRef, _RaySegmentDataRef]
    assert all(not ref.field_references for ref in restored)


@pytest.mark.parametrize('failure', ['missing_file', 'truncated_file', 'put'])
async def test_load_propagates_shard_read_and_publication_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str, ray_store: RayTrajectoryRecordStore
) -> None:
    ray = pytest.importorskip('ray')
    _write_shards(tmp_path, {'shard.pkl': {0: [1]}})
    error = OSError('publication failed')
    monkeypatch.setattr(ray, 'put', Mock(side_effect=error))
    if failure == 'missing_file':
        (tmp_path / 'shard.pkl').unlink()
    elif failure == 'truncated_file':
        (tmp_path / 'shard.pkl').write_bytes(b'')
    expected_error = {'missing_file': FileNotFoundError, 'truncated_file': EOFError, 'put': OSError}[failure]
    with pytest.raises(expected_error):
        await ray_store.load_payloads([_RayDumpRef('node', {'token_ids': 0})], dump_dir=str(tmp_path))


@pytest.mark.parametrize('shard_count', [1, 5])
@pytest.mark.parametrize('remote_caller', [False, True])
async def test_load_round_robins_needed_shards_across_sorted_non_controller_nodes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ray_store: RayTrajectoryRecordStore,
    shard_count: int,
    remote_caller: bool,
) -> None:
    shards = {f'{index}.pkl': {index: [index]} for index in range(shard_count)}
    shards = {'unused.pkl': {99: [99]}, **shards}
    _write_shards(tmp_path, shards)
    for filename in shards:
        (tmp_path / filename).unlink()
    nodes = Mock(
        return_value=[
            {'NodeID': 'b' * 56, 'Alive': True},
            {'NodeID': 'controller', 'Alive': True},
            {'NodeID': 'dead', 'Alive': False},
            {'NodeID': 'a' * 56, 'Alive': True},
        ]
    )
    monkeypatch.setattr(ray_store._ray, 'nodes', nodes)
    if remote_caller:
        monkeypatch.setattr(
            ray_store._ray, 'get_runtime_context', lambda: SimpleNamespace(get_node_id=lambda: 'a' * 56)
        )
    monkeypatch.setattr(ray_store._ray, 'put', Mock(side_effect=AssertionError('Controller must not publish')))
    submissions: dict[str, list[tuple[str, set[int]]]] = {}
    handles = {index: object() for index in range(shard_count)}

    def options(*, scheduling_strategy: Any) -> Any:
        assert not scheduling_strategy.soft, 'Load tasks must stay on the assigned nodes'

        def submit(shard_batch: list[tuple[str, set[int]]]) -> Any:
            submissions[scheduling_strategy.node_id] = shard_batch

            async def execute() -> dict[int, Any]:
                await asyncio.sleep(0)
                assert len(submissions) == min(2, shard_count), 'Dispatch nodes concurrently'
                return {index: handles[index] for _, field_ids in shard_batch for index in field_ids}

            return execute()

        return SimpleNamespace(remote=submit)

    monkeypatch.setattr(ray_backend, '_get_load_payloads_task', lambda: SimpleNamespace(options=options))
    dumped = [_RayDumpRef('node', {'token_ids': index}) for index in reversed(range(shard_count))]
    restored = await ray_store.load_payloads(
        dumped, dump_dir=str(tmp_path), exclude_node_id='controller' if remote_caller else None
    )

    nodes.assert_called_once()
    assert submissions == {
        node_id * 56: [(str(tmp_path / f'{index}.pkl'), {index}) for index in range(offset, shard_count, 2)]
        for offset, node_id in enumerate(('a', 'b')[: min(2, shard_count)])
    }, 'Only needed shards participate in round robin; controller and dead nodes must be excluded'
    assert [ref.field_references['token_ids'] for ref in restored] == [
        handles[index] for index in reversed(range(shard_count))
    ], 'Return refs in input order independently of shard assignment'


@pytest.mark.parametrize('failure', ['submit', 'execute'])
async def test_remote_load_failure_waits_for_siblings_without_local_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ray_store: RayTrajectoryRecordStore, failure: str
) -> None:
    _write_shards(tmp_path, {'first.pkl': {0: [0]}, 'second.pkl': {1: [1]}})
    monkeypatch.setattr(
        ray_store._ray,
        'nodes',
        lambda: [
            {'NodeID': 'a' * 56, 'Alive': True},
            {'NodeID': 'b' * 56, 'Alive': True},
            {'NodeID': 'controller', 'Alive': True},
        ],
    )
    monkeypatch.setattr(
        ray_backend, '_load_payloads_from_shard', Mock(side_effect=AssertionError('No Controller fallback'))
    )
    completed: list[str] = []

    def options(*, scheduling_strategy: Any) -> Any:
        node_id = scheduling_strategy.node_id

        def submit(shards: list[tuple[str, set[int]]]) -> Any:
            if node_id == 'a' * 56 and failure == 'submit':
                raise OSError('load failed')

            async def execute() -> dict[int, Any]:
                if node_id == 'a' * 56:
                    raise OSError('load failed')
                await asyncio.sleep(0)
                completed.append(node_id)
                return {1: object()}

            return execute()

        return SimpleNamespace(remote=submit)

    monkeypatch.setattr(ray_backend, '_get_load_payloads_task', lambda: SimpleNamespace(options=options))
    with pytest.raises(OSError, match='load failed'):
        await ray_store.load_payloads([_RayDumpRef('segment', {'first': 0, 'second': 1})], dump_dir=str(tmp_path))
    assert completed == ['b' * 56], 'Wait for every submitted node before propagating failure'
