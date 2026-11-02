"""Verify field numbering and node dispatch without reading payloads."""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import pickle
import subprocess
import sys
import weakref

from collections.abc import Collection
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from nexus.gateway.trajectory_store import (
    DumpRef,
    InMemoryTrajectoryRecordStore,
    NodePayloadRef,
    RayTrajectoryRecordStore,
    SegmentDataRef,
    TrajectoryRecordStore,
)
from nexus.gateway.trajectory_store_backends import ray as ray_backend
from nexus.gateway.trajectory_store_backends.ray import _RayNodePayloadRef, _RaySegmentDataRef


class _UnreadableObjectRef:
    def __init__(self, node_id: str = 'a' * 56) -> None:
        address_type = pytest.importorskip('ray.core.generated.common_pb2').Address
        self.address = address_type(node_id=bytes.fromhex(node_id)).SerializeToString()

    def owner_address(self) -> bytes:
        return self.address

    def __await__(self) -> Any:
        raise AssertionError('Export must not fetch payloads')

    def __reduce_ex__(self, protocol: int) -> Any:
        raise AssertionError('Export must not serialize ObjectRefs')


class _RecordingDumpTask:
    def __init__(self) -> None:
        self.submissions: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []

    def options(self, *, scheduling_strategy: Any) -> Any:
        assert not scheduling_strategy.soft, 'Dump tasks must not move to another node'

        def submit(indexed_refs: dict[int, Any], shard_path: str) -> Any:
            self.submissions.append((
                scheduling_strategy.node_id,
                tuple(indexed_refs),
                tuple(id(ref) for ref in indexed_refs.values()),
            ))

            async def execute() -> None:
                Path(shard_path).write_bytes(b'completed test shard')

            return execute()

        return SimpleNamespace(remote=submit)


@pytest.fixture
def dump_task(monkeypatch: pytest.MonkeyPatch) -> _RecordingDumpTask:
    task = _RecordingDumpTask()
    monkeypatch.setattr(ray_backend, '_get_dump_payloads_task', lambda: task)
    return task


@pytest.fixture
def ray_store(dump_task: _RecordingDumpTask) -> RayTrajectoryRecordStore:
    # NOTE(jerryhrzhou): The fake task replaces only transport; owner parsing and grouping use real code.
    return RayTrajectoryRecordStore.__new__(RayTrajectoryRecordStore)


@pytest.fixture
def real_refs() -> list[NodePayloadRef | SegmentDataRef]:
    node = _RayNodePayloadRef(
        {'logprobs': _UnreadableObjectRef(), 'token_ids': _UnreadableObjectRef()},
    )
    segment = _RaySegmentDataRef(
        {'old_log_probs': _UnreadableObjectRef()},
    )
    return [segment, node, segment]


async def test_dump_payloads_preserves_record_types_order_and_duplicates(
    ray_store: RayTrajectoryRecordStore, real_refs: list[NodePayloadRef | SegmentDataRef], tmp_path: Path
) -> None:
    dumped, write_payloads = ray_store.dump_payloads(real_refs, dump_dir=str(tmp_path / 'trajectory_store'))
    await write_payloads

    assert len(dumped) == len(real_refs), 'Every input occurrence must have an output'
    assert all(isinstance(ref, DumpRef) for ref in dumped), 'Export only persistent refs'
    assert not any(isinstance(ref, NodePayloadRef | SegmentDataRef) for ref in dumped), (
        'Persistent refs must not masquerade as live refs'
    )
    expected_segment = {
        'format_version': 1,
        'kind': 'segment',
        'field_references': {'old_log_probs': 0},
    }
    expected_node = {
        'format_version': 1,
        'kind': 'node',
        'field_references': {'logprobs': 1, 'token_ids': 2},
    }
    assert [asdict(ref) for ref in dumped] == [expected_segment, expected_node, expected_segment], (
        'Preserve record types, field order, and duplicate positions without business keys'
    )


@pytest.mark.parametrize('fields', [None, ['token_ids'], {'token_ids', 'missing'}, []])
async def test_dump_filters_fields_before_numbering_and_dispatch(
    ray_store: RayTrajectoryRecordStore,
    dump_task: _RecordingDumpTask,
    tmp_path: Path,
    fields: Collection[str] | None,
) -> None:
    shared = _UnreadableObjectRef('b' * 56)
    dropped = _UnreadableObjectRef('a' * 56)
    refs = [
        _RayNodePayloadRef({'logprobs': dropped, 'token_ids': shared}),
        _RaySegmentDataRef({'token_ids': shared}),
    ]
    original = [ref.field_references.copy() for ref in refs]
    dumped, write_payloads = ray_store.dump_payloads(refs, dump_dir=str(tmp_path), fields=fields)
    await write_payloads

    if fields is None:
        expected = [{'logprobs': 0, 'token_ids': 1}, {'token_ids': 1}]
        nodes = ['a' * 56, 'b' * 56]
    elif fields:
        expected = [{'token_ids': 0}, {'token_ids': 0}]
        nodes = ['b' * 56]
    else:
        expected = [{}, {}]
        nodes = []
    assert [ref.field_references for ref in dumped] == expected, (
        'Number only selected fields and deduplicate shared refs'
    )
    assert [node for node, _, _ in dump_task.submissions] == nodes, 'Do not dispatch nodes owning only excluded fields'
    assert [ref.field_references for ref in refs] == original, 'Filtering must leave borrowed refs unchanged'
    index = json.loads((tmp_path / 'index.json').read_text())
    assert {field_id for shard in index['shards'] for field_id in shard['field_ids']} == {
        field_id for ref in expected for field_id in ref.values()
    }, 'The shard index must contain exactly the selected field IDs'


async def test_dump_payloads_keeps_live_refs_and_captures_independent_field_snapshots(
    ray_store: RayTrajectoryRecordStore,
    tmp_path: Path,
) -> None:
    fields = {'token_ids': _UnreadableObjectRef()}
    ref = _RayNodePayloadRef(fields)
    original_fields = fields.copy()
    original_hash = hash(ref)
    [before], write_payloads = ray_store.dump_payloads([ref], dump_dir=str(tmp_path / 'trajectory_store'))
    await write_payloads

    assert ref.field_references is fields, 'Export must keep the live field dictionary'
    assert fields == original_fields, 'Export must not clear or replace runtime ObjectRefs'
    assert hash(ref) == original_hash, 'Export must not change runtime identity'

    fields['routed_experts'] = _UnreadableObjectRef()
    [after], write_payloads = ray_store.dump_payloads([ref], dump_dir=str(tmp_path / 'trajectory_store'))
    await write_payloads
    fields.clear()

    assert before.field_references == {'token_ids': 0}, 'Later writes must not alter a saved snapshot'
    assert after.field_references == {'token_ids': 0, 'routed_experts': 1}, (
        'Each export captures currently visible fields'
    )


async def test_dump_payloads_reuses_indices_for_shared_fields_across_records(
    ray_store: RayTrajectoryRecordStore,
    tmp_path: Path,
) -> None:
    shared = _UnreadableObjectRef()
    other = _UnreadableObjectRef()
    node = _RayNodePayloadRef({'token_ids': shared, 'logprobs': other})
    segment = _RaySegmentDataRef({'prompt_ids': shared, 'values': _UnreadableObjectRef()})
    dumped, write_payloads = ray_store.dump_payloads([node, segment, node], dump_dir=str(tmp_path / 'trajectory_store'))
    await write_payloads

    assert [ref.field_references for ref in dumped] == [
        {'token_ids': 0, 'logprobs': 1},
        {'prompt_ids': 0, 'values': 2},
        {'token_ids': 0, 'logprobs': 1},
    ], 'Number unique ObjectRefs across the whole batch, regardless of record or field name'
    assert all(type(index) is int for ref in dumped for index in ref.field_references.values()), (
        'Persistent field references must contain integers only'
    )
    dumped[0].field_references.clear()
    assert dumped[2].field_references == {'token_ids': 0, 'logprobs': 1}, (
        'Duplicate records must have independent field mappings'
    )


async def test_dump_payloads_distinguishes_reissued_fields_and_restarts_indices_each_call(
    ray_store: RayTrajectoryRecordStore,
    tmp_path: Path,
) -> None:
    first = _RayNodePayloadRef({'token_ids': _UnreadableObjectRef()})
    reissued = _RayNodePayloadRef({'token_ids': _UnreadableObjectRef()})
    dumped, write_payloads = ray_store.dump_payloads([first, reissued], dump_dir=str(tmp_path / 'trajectory_store'))
    await write_payloads

    assert [ref.field_references for ref in dumped] == [{'token_ids': 0}, {'token_ids': 1}], (
        'Different ObjectRefs must not merge even when their field names match'
    )
    [next_dump], write_payloads = ray_store.dump_payloads([reissued], dump_dir=str(tmp_path / 'trajectory_store'))
    await write_payloads
    assert next_dump.field_references == {'token_ids': 0}, 'Integer IDs belong to one dump invocation'


async def test_dump_payloads_does_not_retain_object_refs_in_index_table(
    ray_store: RayTrajectoryRecordStore, tmp_path: Path
) -> None:
    object_ref = _UnreadableObjectRef()
    weak_object_ref = weakref.ref(object_ref)
    ref = _RayNodePayloadRef({'token_ids': object_ref})

    [dumped], write_payloads = ray_store.dump_payloads([ref], dump_dir=str(tmp_path / 'trajectory_store'))
    await write_payloads
    del ref, object_ref

    assert weak_object_ref() is None, 'Neither the dump refs nor the Store may keep the numbering table alive'
    assert dumped.field_references == {'token_ids': 0}, 'Persistent field IDs must outlive runtime references'


async def test_empty_inputs_and_empty_field_snapshots_preserve_cardinality(
    ray_store: RayTrajectoryRecordStore,
    dump_task: _RecordingDumpTask,
    tmp_path: Path,
) -> None:
    dumped, write_payloads = ray_store.dump_payloads([], dump_dir=str(tmp_path / 'trajectory_store'))
    assert dumped == [], 'An empty batch has no persistent refs'
    await write_payloads
    [dumped], write_payloads = ray_store.dump_payloads(
        [_RaySegmentDataRef({})], dump_dir=str(tmp_path / 'trajectory_store')
    )
    await write_payloads
    assert dumped.field_references == {}, 'Keep an empty field snapshot instead of dropping its record'
    assert dump_task.submissions == [], 'Empty field snapshots must not launch tasks'
    assert json.loads((tmp_path / 'trajectory_store/index.json').read_text()) == {'format_version': 1, 'shards': []}, (
        'An empty dump still publishes an empty index'
    )


async def test_dump_payloads_groups_unique_fields_by_owner_node(
    ray_store: RayTrajectoryRecordStore, dump_task: _RecordingDumpTask, tmp_path: Path
) -> None:
    shared = _UnreadableObjectRef('a' * 56)
    second_node = _UnreadableObjectRef('b' * 56)
    first_node = _UnreadableObjectRef('a' * 56)
    refs = [
        _RayNodePayloadRef({'tokens': shared, 'experts': second_node}),
        _RaySegmentDataRef({'shared': shared, 'values': first_node}),
    ]

    dumped, write_payloads = ray_store.dump_payloads(refs, dump_dir=str(tmp_path / 'trajectory_store'))
    await write_payloads

    assert dump_task.submissions == [
        ('a' * 56, (0, 2), (id(shared), id(first_node))),
        ('b' * 56, (1,), (id(second_node),)),
    ], 'Use each field owner and send one task per node with each ObjectRef once'
    assert [ref.field_references for ref in dumped] == [{'tokens': 0, 'experts': 1}, {'shared': 0, 'values': 2}], (
        'Dispatched indices must match persisted field references'
    )


@pytest.mark.parametrize('failure_phase', ['submission', 'execution'])
async def test_failed_task_waits_for_other_submissions_before_propagating(
    ray_store: RayTrajectoryRecordStore, monkeypatch: pytest.MonkeyPatch, failure_phase: str, tmp_path: Path
) -> None:
    failed = asyncio.Event()
    started = asyncio.Event()
    finish = asyncio.Event()
    completed = asyncio.Event()
    error = RuntimeError('node unavailable')

    def options(*, scheduling_strategy: Any) -> Any:
        def submit(indexed_refs: dict[int, Any], shard_path: str) -> Any:
            if scheduling_strategy.node_id == 'a' * 56 and failure_phase == 'submission':
                failed.set()
                raise error

            async def execute() -> None:
                if scheduling_strategy.node_id == 'a' * 56:
                    failed.set()
                    raise error
                started.set()
                await finish.wait()
                completed.set()

            return execute()

        return SimpleNamespace(remote=submit)

    monkeypatch.setattr(ray_backend, '_get_dump_payloads_task', lambda: SimpleNamespace(options=options))
    refs = [_RayNodePayloadRef({'first': _UnreadableObjectRef(), 'second': _UnreadableObjectRef('b' * 56)})]
    _, write_payloads = ray_store.dump_payloads(refs, dump_dir=str(tmp_path / 'trajectory_store'))
    pending = asyncio.create_task(write_payloads)
    try:
        await asyncio.wait_for(asyncio.gather(failed.wait(), started.wait()), timeout=5)
        assert not pending.done(), 'A failed submission must not abandon other submitted tasks'
    finally:
        finish.set()
        with pytest.raises(RuntimeError, match='node unavailable') as raised:
            await pending
    assert raised.value is error, 'Propagate the original failure'
    assert completed.is_set(), 'Wait for the other node before returning the failure'
    assert not (tmp_path / 'trajectory_store/index.json').exists(), 'A failed dump must not publish its index'


def test_dump_returns_numbered_refs_before_running_any_io(
    ray_store: RayTrajectoryRecordStore, dump_task: _RecordingDumpTask, tmp_path: Path
) -> None:
    directory = tmp_path / 'payloads'
    shared = _UnreadableObjectRef()
    ref = _RayNodePayloadRef({'token_ids': shared, 'shared': shared})
    dumped, write_payloads = ray_store.dump_payloads([ref], dump_dir=str(directory))
    try:
        assert inspect.iscoroutine(write_payloads), 'Return the remaining work as a coroutine'
        assert dumped[0].field_references == {'token_ids': 0, 'shared': 0}, 'Number refs synchronously'
        assert dump_task.submissions == [], 'Do not submit remote writes until awaited'
        assert not directory.exists(), 'Synchronous preparation must not create files or directories'
        ref.field_references.clear()
        asyncio.run(write_payloads)
    finally:
        write_payloads.close()
    assert dump_task.submissions == [('a' * 56, (0,), (id(shared),))], 'Capture refs before callers replace them'
    assert (directory / 'index.json').exists(), 'Awaiting finishes the payload save'


def test_node_dump_reads_and_writes_each_field_before_fetching_the_next(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ray = pytest.importorskip('ray')
    numpy = pytest.importorskip('numpy')
    torch = pytest.importorskip('torch')
    values = [
        [1, 2],
        [-0.2],
        numpy.arange(4, dtype=numpy.int32),
        torch.arange(6, dtype=torch.bfloat16),
        {'images': [{'pixel_values': torch.ones(1, 2, dtype=torch.float32)}]},
    ]
    events = []
    original_dump = pickle.dump

    def get_field(object_ref: int) -> Any:
        events.append(('get', object_ref))
        return values[object_ref]

    def write_field(value: Any, stream: Any, **kwargs: Any) -> None:
        if isinstance(value, tuple):
            events.append(('write', value[0]))
        original_dump(value, stream, **kwargs)

    monkeypatch.setattr(ray, 'get', get_field)
    monkeypatch.setattr(pickle, 'dump', write_field)
    path = tmp_path / 'node.pkl'
    ray_backend._dump_payloads_on_node({index: index for index in reversed(range(len(values)))}, str(path))

    assert events == [event for index in range(len(values)) for event in (('get', index), ('write', index))], (
        'Write each field in integer order without fetching the next field first'
    )
    with path.open('rb') as stream:
        assert pickle.load(stream) == {'format_version': 1, 'field_count': len(values)}, 'Persist shard header'
        restored = dict(pickle.load(stream) for _ in values)
        assert stream.read() == b'', 'Persist exactly the declared fields'
    assert restored[0] == values[0], 'Preserve tokens'
    assert restored[1] == values[1], 'Preserve logprobs'
    numpy.testing.assert_array_equal(restored[2], values[2])
    assert restored[2].dtype == values[2].dtype, 'Preserve routing dtype'
    assert torch.equal(restored[3], values[3]), 'Preserve tensors'
    assert restored[3].dtype == values[3].dtype, 'Preserve tensor dtype'
    assert torch.equal(restored[4]['images'][0]['pixel_values'], values[4]['images'][0]['pixel_values']), (
        'Preserve nested multimodal tensors'
    )


def test_node_dump_propagates_file_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ray = pytest.importorskip('ray')
    monkeypatch.setattr(ray, 'get', lambda ref: [1])
    with pytest.raises(IsADirectoryError):
        ray_backend._dump_payloads_on_node({0: object()}, str(tmp_path))


def test_dump_task_wrapper_is_lazy_and_disables_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    ray = pytest.importorskip('ray')
    calls = []
    task = object()

    def remote(**options: Any) -> Any:
        def wrap(function: Any) -> Any:
            calls.append((options, function))
            return task

        return wrap

    monkeypatch.setattr(ray_backend, '_DUMP_PAYLOADS_TASK', None)
    monkeypatch.setattr(ray, 'remote', remote)
    assert ray_backend._get_dump_payloads_task() is task, 'Build a Ray task wrapper when first needed'
    assert ray_backend._get_dump_payloads_task() is task, 'Reuse the wrapper without retaining payload refs'
    assert calls == [({'num_cpus': 0, 'max_retries': 0}, ray_backend._dump_payloads_on_node)], (
        'No CPU reservations or automatic retries for the future file-writing task'
    )


async def test_independent_empty_refs_and_shared_snapshots_keep_separate_output_entries(
    ray_store: RayTrajectoryRecordStore,
    tmp_path: Path,
) -> None:
    shared = _UnreadableObjectRef()
    empty_refs = [_RaySegmentDataRef({}), _RaySegmentDataRef({})]
    shared_refs = [_RayNodePayloadRef({'token_ids': shared}), _RayNodePayloadRef({'token_ids': shared})]
    inputs = [*empty_refs, *shared_refs]
    assert len(set(inputs)) == 4, 'The MR treats runtime refs as independent Python objects'

    dumped, write_payloads = ray_store.dump_payloads(inputs, dump_dir=str(tmp_path / 'trajectory_store'))
    await write_payloads

    assert len(dumped) == 4, 'Deduplicate field ObjectRefs without collapsing records'
    assert len({id(ref) for ref in dumped}) == 4, 'Allocate one dump ref per input occurrence'
    assert len({id(ref.field_references) for ref in dumped}) == 4, 'Keep independent field mappings'
    assert [ref.field_references for ref in dumped] == [{}, {}, {'token_ids': 0}, {'token_ids': 0}], (
        'Shared field objects must reuse their integer across independent records'
    )


async def test_dump_refs_roundtrip_files_without_importing_ray(
    ray_store: RayTrajectoryRecordStore,
    real_refs: list[NodePayloadRef | SegmentDataRef],
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip('torch')
    dumped, write_payloads = ray_store.dump_payloads(real_refs, dump_dir=str(tmp_path / 'trajectory_store'))
    await write_payloads
    pickle_path = tmp_path / 'refs.pkl'
    pickle_path.write_bytes(pickle.dumps(dumped))
    torch_path = tmp_path / 'refs.pt'
    torch.save(dumped, torch_path)
    script = """
import importlib.abc
import json
import pickle
import sys
from dataclasses import asdict

class BlockRay(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'ray' or fullname.startswith('ray.'):
            raise AssertionError('Persistent refs must load without Ray')

sys.meta_path.insert(0, BlockRay())
import torch
from nexus.gateway.trajectory_store import DumpRef

with open(sys.argv[1], 'rb') as stream:
    pickled = pickle.load(stream)
loaded = torch.load(sys.argv[2], weights_only=False)
assert pickled == loaded, 'Both formats must preserve ref type and metadata'
assert all(isinstance(ref, DumpRef) for ref in loaded), 'Keep the public ref contract'
assert 'ray' not in sys.modules, 'Loading persistent refs must not initialize Ray'
print(json.dumps([asdict(ref) for ref in loaded]))
"""
    environment = os.environ.copy()
    repository_root = Path(__file__).resolve().parents[3]
    environment['PYTHONPATH'] = os.pathsep.join([str(repository_root), environment.get('PYTHONPATH', '')])
    result = subprocess.run(
        [sys.executable, '-c', script, str(pickle_path), str(torch_path)],
        cwd=repository_root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, f'Offline ref loading failed:\n{result.stdout}\n{result.stderr}'
    assert json.loads(result.stdout) == json.loads(json.dumps([asdict(ref) for ref in dumped])), (
        'Every persistent metadata field must survive a fresh process'
    )


@pytest.mark.parametrize('store_type', [TrajectoryRecordStore, InMemoryTrajectoryRecordStore])
async def test_unsupported_backend_dump_raises(store_type: type[TrajectoryRecordStore], tmp_path: Path) -> None:
    store = store_type.__new__(store_type)
    with pytest.raises(NotImplementedError, match='dump_payloads'):
        store.dump_payloads([], dump_dir=str(tmp_path / 'trajectory_store'))


@pytest.mark.parametrize(
    'store_type',
    [TrajectoryRecordStore, InMemoryTrajectoryRecordStore],
)
async def test_load_payloads_explicitly_reports_unimplemented_recovery(
    store_type: type[TrajectoryRecordStore], ray_store: RayTrajectoryRecordStore, tmp_path: Path
) -> None:
    store = store_type.__new__(store_type)
    dumped, write_payloads = ray_store.dump_payloads(
        [_RayNodePayloadRef({})], dump_dir=str(tmp_path / 'trajectory_store')
    )
    await write_payloads
    for refs in ([], dumped):
        with pytest.raises(NotImplementedError, match='load_payloads'):
            await store.load_payloads(refs, dump_dir=str(tmp_path / 'trajectory_store'))
