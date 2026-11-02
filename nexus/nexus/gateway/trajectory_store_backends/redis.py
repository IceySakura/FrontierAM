"""Trajectory records backed by the synchronous Redis client.

Each publication gets a fresh client-generated UUID. Writes are atomic per
record only; retrying a failed batch can leave copies from the previous attempt.
"""

from __future__ import annotations

import asyncio
import logging
import pickle

from collections.abc import Callable, Collection, Coroutine, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal
from uuid import uuid4

from .redis_store.client import RedisClient
from .redis_store.types import CompareAndSetResult, NodeId, RedisClientError, Serializable
from .types import (
    DumpRef,
    NodePayloadRecord,
    NodePayloadRef,
    Record,
    RecordRef,
    SegmentDataRecord,
    SegmentDataRef,
    TrajectoryRecordStore,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True, eq=False)
class NodeRecord(Serializable, NodePayloadRecord):
    """Node training data stored directly as a KV payload."""


@dataclass(frozen=True)
class SegmentRecord(Serializable, SegmentDataRecord):
    """Segment fields: source objects on input, field pickle bytes in storage."""

    def with_fields(self, updates: Mapping[str, bytes]) -> SegmentRecord:
        """Validate and merge encoded fields on a stored segment copy."""
        for name, value in updates.items():
            if not isinstance(name, str) or not isinstance(value, bytes):
                raise ValueError('record fields must map strings to pickled bytes')
            if name in self.fields and self.fields[name] != value:
                raise ValueError(f'conflicting record field: {name!r}')
        return replace(self, fields={**self.fields, **updates})


@dataclass(frozen=True)
class NodeRef(NodePayloadRef):
    """Serializable node handle tagged with its publication namespace."""

    namespace: str
    record_id: NodeId


@dataclass(frozen=True)
class HybridNodeRef(NodeRef):
    """Node identity plus optional external locations for parallel reads."""

    routed_experts_url: str | None = None
    multi_modal_items_url: str | None = None

    @property
    def blob_urls(self) -> dict[str, str]:
        return {
            name: url
            for name in ('routed_experts', 'multi_modal_items')
            if (url := getattr(self, name + '_url')) is not None
        }


@dataclass(frozen=True)
class SegmentRef(SegmentDataRef):
    """Serializable segment handle, also usable as an update mapping key."""

    namespace: str
    record_id: NodeId


@dataclass(frozen=True)
class _RedisDumpRef(DumpRef):
    """Locate one persistent row without retaining an online record identity."""

    dump_id: str
    entry_id: int
    kind: Literal['node', 'segment']
    fields: tuple[str, ...] | None


def _redis_refs(refs: Sequence[RecordRef]) -> list[NodeRef | SegmentRef]:
    out = []
    for ref in refs:
        if not isinstance(ref, (NodeRef, SegmentRef)):
            raise TypeError(f'unsupported trajectory record ref type: {type(ref)!r}')
        out.append(ref)
    return out


def _serialize_fields(fields: Mapping[str, Any]) -> dict[str, bytes]:
    if not isinstance(fields, Mapping) or any(not isinstance(name, str) for name in fields):
        raise TypeError('record fields must be a mapping with string keys')
    return {name: pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL) for name, value in fields.items()}


class RedisTrajectoryRecordStore(TrajectoryRecordStore):
    """UUID-addressed node payloads and append-only segment fields."""

    def __init__(self, namespace: str, config: Mapping[str, Any] | None = None) -> None:
        if not namespace:
            raise ValueError('trajectory store namespace must not be empty')
        self.namespace = namespace
        self.config = dict(config or {})
        self._client = RedisClient(**self.config)
        if 'url' not in self.config and self._client.url is not None:
            self.config['url'] = self._client.url

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> RedisTrajectoryRecordStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def dump_payloads(
        self,
        real_refs: list[RecordRef],
        *,
        dump_dir: str,
        fields: Collection[str] | None = None,
    ) -> tuple[list[DumpRef], Coroutine[Any, Any, None]]:
        """Prepare typed persistent refs; defer all remote I/O until awaited."""
        from .redis_store.checkpoint.dump import dump_checkpoint

        selected_fields = None if fields is None else tuple(fields)
        if selected_fields is not None and any(not isinstance(name, str) for name in selected_fields):
            raise TypeError('field names must be strings')
        refs = _redis_refs(real_refs)
        if (
            selected_fields is not None
            and 'token_ids' not in selected_fields
            and any(isinstance(ref, NodeRef) for ref in refs)
        ):
            raise ValueError('Node checkpoint fields must include token_ids')
        unique = {}
        dump_id = uuid4().hex
        dumped: list[DumpRef] = []
        for ref in refs:
            kind = 'node' if isinstance(ref, NodeRef) else 'segment'
            key = (kind, ref.namespace, ref.record_id)
            entry_id = unique.setdefault(key, (len(unique), ref))[0]
            dumped.append(_RedisDumpRef(dump_id, entry_id, kind, selected_fields))
        source_refs = [ref for _, ref in unique.values()]
        return dumped, dump_checkpoint(self, source_refs, dump_id, dump_dir, selected_fields)

    async def load_payloads(
        self, dump_refs: list[DumpRef], *, dump_dir: str, exclude_node_id: str | None = None
    ) -> list[RecordRef]:
        """Restore fresh Redis records, importing external files directly on each DU."""
        if exclude_node_id is not None:
            logger.warning('Redis load ignores Ray-specific exclude_node_id=%r', exclude_node_id)
        if not dump_refs:
            return []
        from .redis_store.checkpoint.load import load_checkpoint

        return await load_checkpoint(self, list(dump_refs), dump_dir)

    async def _run(self, operation: Callable[..., Any], *args: Any) -> Any:
        task = asyncio.create_task(asyncio.to_thread(operation, *args))
        completion = asyncio.gather(task, return_exceptions=True)
        cancelled = None
        # Drain synchronous I/O before propagating even repeated cancellation.
        while not completion.done():
            try:
                await asyncio.shield(completion)
            except asyncio.CancelledError as exc:
                cancelled = cancelled or exc
        if cancelled is not None:
            error = task.exception()
            if error is not None:
                # Keep I/O failures visible when cancellation takes precedence.
                asyncio.get_running_loop().call_exception_handler({
                    'message': 'RecordStore I/O failed while draining cancellation',
                    'exception': error,
                    'task': task,
                })
            raise cancelled from error
        try:
            return task.result()
        except RedisClientError as exc:
            if exc.code == 'not_found':
                raise KeyError(str(exc)) from exc
            raise

    async def put_many(self, records: Sequence[Record]) -> list[RecordRef]:
        if not records:
            return []
        return await self._run(self._put_many, list(records))

    def _put_many(self, records: Sequence[Record]) -> list[NodeRef | SegmentRef]:
        payloads = []
        for record in records:
            if isinstance(record, SegmentDataRecord):
                payloads.append(
                    SegmentRecord(record.trajectory_id, record.segment_id, _serialize_fields(record.fields))
                )
            elif isinstance(record, NodeRecord):
                payloads.append(record)
            elif isinstance(record, NodePayloadRecord):
                payloads.append(NodeRecord(**vars(record)))
            else:
                raise TypeError(f'unsupported trajectory record type: {type(record)!r}')
        from .redis_store.file_storage.publication import publication_errors

        results = self._client._publication.put(payloads)
        partial_refs = []
        for payload, result in zip(payloads, results, strict=True):
            if result.error is not None:
                partial_refs.append(None)
            elif result.blob_urls:
                partial_refs.append(
                    HybridNodeRef(
                        self.namespace,
                        result.node_id,
                        **{name + '_url': url for name, url in result.blob_urls.items()},
                    )
                )
            else:
                ref_type = NodeRef if isinstance(payload, NodeRecord) else SegmentRef
                partial_refs.append(ref_type(self.namespace, result.node_id))
        errors = publication_errors(results)
        if errors:
            raise RedisClientError(
                f'record publication failed: {errors}',
                code='publication_failed',
                body={'partial_refs': partial_refs, 'errors': errors},
            )
        return partial_refs

    async def get_many(
        self, refs: Sequence[RecordRef], *, fields: Collection[str] | None = None
    ) -> list[dict[str, Any]]:
        if not refs:
            return []
        return await self._run(self._get_many, _redis_refs(refs), None if fields is None else tuple(fields))

    def _get_many(self, refs: Sequence[NodeRef | SegmentRef], fields: Collection[str] | None) -> list[dict[str, Any]]:
        requests = [(ref.record_id, NodeRecord if isinstance(ref, NodeRef) else SegmentRecord) for ref in refs]
        if fields is not None and any(not isinstance(name, str) for name in fields):
            raise TypeError('field names must be strings')
        payloads = self._client._publication.get(
            requests,
            fields=fields,
            reference_urls=[ref.blob_urls if isinstance(ref, HybridNodeRef) else None for ref in refs],
        )
        out = []
        for payload in payloads:
            if isinstance(payload, NodeRecord):
                values = {
                    'token_ids': payload.token_ids,
                    'logprobs': payload.logprobs,
                    'topk_indices': payload.topk_indices,
                    'topk_values': payload.topk_values,
                    'routed_experts': payload.routed_experts,
                    'multi_modal_items': payload.multi_modal_items,
                }
                values = {name: value for name, value in values.items() if name == 'token_ids' or value is not None}
                out.append({name: values[name] for name in (values if fields is None else fields) if name in values})
            else:
                names = payload.fields if fields is None else fields
                out.append({name: pickle.loads(payload.fields[name]) for name in names if name in payload.fields})
        return out

    async def put_fields_many(self, updates: Mapping[SegmentDataRef, Mapping[str, Any]]) -> None:
        if updates:
            await self._run(self._put_fields_many, dict(updates))

    def _put_fields_many(self, updates: Mapping[SegmentDataRef, Mapping[str, Any]]) -> None:
        encoded = []
        for ref, fields in updates.items():
            if not isinstance(ref, SegmentRef):
                raise TypeError('late fields can only be appended to segment records')
            encoded.append((ref.record_id, _serialize_fields(fields)))
        with self._client._publication.operation():
            payloads = self._client.get_multi_node([(node_id, SegmentRecord) for node_id, _ in encoded])
            for payload, (_, fields) in zip(payloads, encoded, strict=True):
                payload.with_fields(fields)
            self._append_fields(encoded)

    def _append_fields(self, encoded: list[tuple[NodeId, dict[str, bytes]]]) -> None:
        pending = encoded
        for _ in range(8):
            blobs = self._client.get_node_blobs([node_id for node_id, _ in pending])
            replacements = []
            for (node_id, fields), blob in zip(pending, blobs, strict=True):
                if blob is None:
                    raise RedisClientError(f'payload not found for node_id={node_id}', code='not_found')
                payload = SegmentRecord.deserialize(blob)
                if not isinstance(payload, SegmentRecord):
                    raise ValueError('expected SegmentRecord')
                merged = payload.with_fields(fields)
                if merged.fields != payload.fields:
                    replacements.append((node_id, fields, blob, merged.serialize()))
            if not replacements:
                return
            # Compare the bytes actually read, not a re-encoded version of the record.
            results = self._client.compare_and_set_many([(node_id, old, new) for node_id, _, old, new in replacements])
            pending = []
            for (node_id, fields, _, _), result in zip(replacements, results, strict=True):
                if result is CompareAndSetResult.MISSING:
                    raise RedisClientError(f'payload not found for node_id={node_id}', code='not_found')
                if result is CompareAndSetResult.MISMATCH:
                    pending.append((node_id, fields))
            if not pending:
                return
        raise RedisClientError('record append contention exceeded 8 attempts', code='contention')

    async def release_many(self, refs: Sequence[RecordRef]) -> None:
        if refs:
            await self._run(self._release_many, _redis_refs(refs))

    def _release_many(self, refs: Sequence[NodeRef | SegmentRef]) -> None:
        # Validate every locator before deletion snapshots deduplicate IDs.
        self._client._publication.delete(
            [ref.record_id for ref in refs],
            reference_urls=[ref.blob_urls if isinstance(ref, HybridNodeRef) else None for ref in refs],
        )
