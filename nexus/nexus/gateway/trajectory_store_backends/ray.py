"""Distributed Ray Object Store-backed trajectory record store."""

from __future__ import annotations

import asyncio
import importlib
import json
import pickle

from collections.abc import Collection, Coroutine, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, cast

from .types import (
    DumpRef,
    NodePayloadRecord,
    NodePayloadRef,
    Record,
    RecordRef,
    SegmentDataRef,
    TrajectoryRecordStore,
)
from .utils import _record_fields


if TYPE_CHECKING:
    from ray import ObjectRef


@dataclass(eq=False)
class _RayNodePayloadRef(NodePayloadRef):
    field_references: dict[str, Any]


@dataclass(eq=False)
class _RaySegmentDataRef(SegmentDataRef):
    field_references: dict[str, Any]


@dataclass(frozen=True)
class _RayDumpRef(DumpRef):
    """Capture a typed field snapshot using integer IDs local to one dump call.

    Runtime refs carry only field ObjectRefs, so the dump retains the record type
    and field mapping without introducing business keys or owner identities.
    """

    kind: Literal['node', 'segment']
    field_references: dict[str, int] = field(hash=False)
    format_version: int = field(default=1, init=False)


_RayRef: TypeAlias = _RayNodePayloadRef | _RaySegmentDataRef


class RayTrajectoryRecordStore(TrajectoryRecordStore):
    """Store each payload field directly in Ray and carry its ObjectRef in the record ref.

    Refs carry field snapshots. Late-field writes update only the supplied refs;
    copies already sent to another process retain their original field snapshots.
    Release is a no-op; callers discard refs after their final consumer finishes.
    The processes publishing fields must stay alive while their ObjectRefs are used.
    """

    def __init__(self, namespace: str) -> None:
        if not namespace:
            raise ValueError('trajectory store namespace must not be empty')
        try:
            ray = importlib.import_module('ray')
        except ImportError as exc:
            raise RuntimeError('RayTrajectoryRecordStore requires ray') from exc
        if not ray.is_initialized():
            raise RuntimeError('RayTrajectoryRecordStore requires an initialized Ray runtime')
        self._ray = ray
        self.namespace = namespace

    def dump_payloads(
        self,
        real_refs: list[RecordRef],
        *,
        dump_dir: str,
        fields: Collection[str] | None = None,
    ) -> tuple[list[DumpRef], Coroutine[Any, Any, None]]:
        """Number field references and return a coroutine for writing their node shards.

        Args:
            real_refs: Borrowed Ray node or segment refs; export does not mutate or release them.
            dump_dir: Shared directory for node shards and index.json, written once per checkpoint.
            fields: Payload fields to export; None selects all fields. Filtering precedes numbering.

        Returns:
            One independent metadata snapshot per input occurrence, in input order,
            and the remaining payload write coroutine. No I/O starts until it is awaited.
            Unique ObjectRefs are numbered from zero in first-seen order across the
            whole call. Equal ObjectRefs share an index, including across independent
            record refs. The coroutine retains the source ObjectRefs until its writes
            finish, so callers can replace their refs before awaiting it. It publishes
            the index after all node files close. The caller reads no payloads.
        """
        dump_refs: list[DumpRef] = []
        object_ref_indices: dict[ObjectRef, int] = {}
        for ref in real_refs:
            ray_ref = cast(_RayRef, ref)
            dump_refs.append(
                _RayDumpRef(
                    kind='node' if isinstance(ray_ref, _RayNodePayloadRef) else 'segment',
                    field_references={
                        name: object_ref_indices.setdefault(object_ref, len(object_ref_indices))
                        for name, object_ref in ray_ref.field_references.items()
                        if fields is None or name in fields
                    },
                )
            )
        indexed_refs_by_node: dict[str, dict[int, ObjectRef]] = {}
        for object_ref, index in object_ref_indices.items():
            node_id = _get_owner_node_id(object_ref)
            indexed_refs_by_node.setdefault(node_id, {})[index] = object_ref

        directory = Path(dump_dir).absolute()

        async def write_payloads() -> None:
            directory.mkdir(parents=True, exist_ok=True)
            if indexed_refs_by_node:
                # NOTE(jerryhrzhou): Keep Ray scheduling imports out of offline metadata loading.
                from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

                dump_task = _get_dump_payloads_task()

                async def dispatch_node(node_id: str, indexed_refs: dict[int, ObjectRef]) -> None:
                    # NOTE(jerryhrzhou): Submit inside gather so submission failures also wait for sibling tasks.
                    await dump_task.options(
                        scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)
                    ).remote(indexed_refs, str(directory / f'{node_id}.pkl'))

                results = await asyncio.gather(
                    *(dispatch_node(node_id, refs) for node_id, refs in indexed_refs_by_node.items()),
                    return_exceptions=True,
                )
                for result in results:
                    if isinstance(result, BaseException):
                        raise result
            with (directory / 'index.json').open('w', encoding='utf-8') as stream:
                json.dump(
                    {
                        'format_version': 1,
                        'shards': [
                            {'file': f'{node_id}.pkl', 'field_ids': list(refs)}
                            for node_id, refs in indexed_refs_by_node.items()
                        ],
                    },
                    stream,
                )

        return dump_refs, write_payloads()

    async def load_payloads(
        self, dump_refs: list[DumpRef], *, dump_dir: str, exclude_node_id: str | None = None
    ) -> list[RecordRef]:
        """Restore field snapshots while excluding one node from shard loading.

        Args:
            dump_refs: Persistent node or segment refs from one dump, in the desired output order.
            dump_dir: Directory containing index.json and its payload shards.
            exclude_node_id: Node to exclude. None excludes the caller's node; an explicit
                node replaces that default, so a remote caller's node remains eligible.

        Returns:
            Independent runtime refs in input order, sharing ObjectRefs for shared field IDs.
            Shards are assigned round robin to sorted live node IDs, excluding only the
            selected node. Each node task reads its files sequentially and returns only
            ObjectRefs. With no eligible live node, the caller reads the shards directly.
        """
        ray_dump_refs = cast(list[_RayDumpRef], dump_refs)
        field_ids = {index for ref in ray_dump_refs for index in ref.field_references.values()}

        def read_shards() -> list[tuple[str, set[int]]]:
            directory = Path(dump_dir).absolute()
            with (directory / 'index.json').open(encoding='utf-8') as stream:
                index = json.load(stream)
            return [
                (str(directory / shard['file']), needed_ids)
                for shard in index['shards']
                if (needed_ids := field_ids.intersection(shard['field_ids']))
            ]

        indexed_refs: dict[int, ObjectRef] = {}
        if field_ids:
            shards = read_shards()
            if exclude_node_id is None:
                exclude_node_id = self._ray.get_runtime_context().get_node_id()
            node_ids = sorted(
                node['NodeID'] for node in self._ray.nodes() if node['Alive'] and node['NodeID'] != exclude_node_id
            )
            if not node_ids:
                indexed_refs = _load_payloads_on_node(shards)
            else:
                from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

                shards_by_node: dict[str, list[tuple[str, set[int]]]] = {}
                for index, shard in enumerate(shards):
                    shards_by_node.setdefault(node_ids[index % len(node_ids)], []).append(shard)
                load_task = _get_load_payloads_task()

                async def dispatch_node(node_id: str, shard_batch: list[tuple[str, set[int]]]) -> dict[int, ObjectRef]:
                    return await load_task.options(
                        scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)
                    ).remote(shard_batch)

                results = await asyncio.gather(
                    *(dispatch_node(node_id, batch) for node_id, batch in shards_by_node.items()),
                    return_exceptions=True,
                )
                for result in results:
                    if isinstance(result, BaseException):
                        raise result
                    indexed_refs.update(result)
        return [
            (_RayNodePayloadRef if ref.kind == 'node' else _RaySegmentDataRef)({
                name: indexed_refs[index] for name, index in ref.field_references.items()
            })
            for ref in ray_dump_refs
        ]

    async def put_many(self, records: Sequence[Record]) -> list[RecordRef]:
        """Publish one batch in a worker thread without blocking the caller's event loop."""

        def put_records() -> list[RecordRef]:
            return [
                (_RayNodePayloadRef if isinstance(record, NodePayloadRecord) else _RaySegmentDataRef)({
                    name: self._ray.put(value) for name, value in _record_fields(record).items()
                })
                for record in records
            ]

        return await asyncio.to_thread(put_records)

    async def get_many(
        self,
        refs: Sequence[RecordRef],
        *,
        fields: Collection[str] | None = None,
    ) -> list[dict[str, Any]]:
        requested_fields = tuple(fields) if fields is not None else None
        object_refs: list[Any] = []
        field_names_by_ref: list[tuple[str, ...]] = []
        for ref in refs:
            ray_ref = cast(_RayRef, ref)
            field_names = (
                tuple(ray_ref.field_references)
                if requested_fields is None
                else tuple(name for name in requested_fields if name in ray_ref.field_references)
            )
            field_names_by_ref.append(field_names)
            object_refs.extend(ray_ref.field_references[name] for name in field_names)

        values = await asyncio.gather(*object_refs)
        payloads: list[dict[str, Any]] = []
        value_index = 0
        for field_names in field_names_by_ref:
            payload = {}
            for field_name in field_names:
                payload[field_name] = values[value_index]
                value_index += 1
            payloads.append(payload)
        return payloads

    async def put_fields_many(self, updates: Mapping[SegmentDataRef, Mapping[str, Any]]) -> None:
        """Append new fields directly to the supplied refs' field snapshots."""

        def put_fields() -> None:
            for ref, fields in updates.items():
                ray_ref = cast(_RaySegmentDataRef, ref)
                for name, value in fields.items():
                    # NOTE(jerryhrzhou): This validation-only path tolerates concurrent same-field writes
                    # racing between the duplicate check and publication.
                    if name in ray_ref.field_references:
                        raise ValueError(f'conflicting trajectory record fields: {name}')
                    ray_ref.field_references[name] = self._ray.put(value)

        await asyncio.to_thread(put_fields)

    async def release_many(self, refs: Sequence[RecordRef]) -> None:
        pass


_DUMP_PAYLOADS_TASK: Any = None
_LOAD_PAYLOADS_TASK: Any = None


def _get_owner_node_id(object_ref: ObjectRef) -> str:
    """Decode the publishing node from Ray's runtime-specific ownership metadata."""
    # Keep the low-level Ray protobuf dependency out of metadata-only imports.
    from ray.core.generated.common_pb2 import Address

    owner = Address()
    owner.ParseFromString(object_ref.owner_address())
    return owner.node_id.hex()


def _dump_payloads_on_node(indexed_refs: dict[int, ObjectRef], shard_path: str) -> None:
    """Write one pickle stream without retaining previously written payloads."""
    # Only the node task resolves payloads; the controller handles refs and paths.
    import ray

    with open(shard_path, 'wb') as stream:
        pickle.dump({'format_version': 1, 'field_count': len(indexed_refs)}, stream, protocol=pickle.HIGHEST_PROTOCOL)
        for index in sorted(indexed_refs):
            payload = ray.get(indexed_refs[index])
            pickle.dump((index, payload), stream, protocol=pickle.HIGHEST_PROTOCOL)
            del payload


def _load_payloads_from_shard(shard_path: str, field_ids: set[int]) -> dict[int, ObjectRef]:
    """Publish requested fields from one shard, returning only their runtime handles."""
    # NOTE(jerryhrzhou): Keep Ray optional when importing persistent ref metadata offline.
    import ray

    indexed_refs: dict[int, ObjectRef] = {}
    with open(shard_path, 'rb') as stream:
        header = pickle.load(stream)
        for _ in range(header['field_count']):
            index, payload = pickle.load(stream)
            if index in field_ids:
                indexed_refs[index] = ray.put(payload)
            del payload
    return indexed_refs


def _load_payloads_on_node(shards: list[tuple[str, set[int]]]) -> dict[int, ObjectRef]:
    """Read a node's assigned shards sequentially, retaining only their published refs."""
    indexed_refs: dict[int, ObjectRef] = {}
    for shard_path, field_ids in shards:
        indexed_refs.update(_load_payloads_from_shard(shard_path, field_ids))
    return indexed_refs


def _get_load_payloads_task() -> Any:
    """Keep publishing workers alive through the returned nested ObjectRefs, without an Actor."""
    import ray

    global _LOAD_PAYLOADS_TASK
    if _LOAD_PAYLOADS_TASK is None:
        _LOAD_PAYLOADS_TASK = ray.remote(num_cpus=0, max_retries=0)(_load_payloads_on_node)
    return _LOAD_PAYLOADS_TASK


def _get_dump_payloads_task() -> Any:
    """Create the remote wrapper only when a nonempty dump needs node dispatch."""
    import ray

    global _DUMP_PAYLOADS_TASK
    if _DUMP_PAYLOADS_TASK is None:
        _DUMP_PAYLOADS_TASK = ray.remote(num_cpus=0, max_retries=0)(_dump_payloads_on_node)
    return _DUMP_PAYLOADS_TASK
