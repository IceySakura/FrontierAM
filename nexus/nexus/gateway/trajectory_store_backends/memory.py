"""Inline in-memory trajectory record store."""

from __future__ import annotations

import uuid

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeAlias, cast

from .types import NodePayloadRecord, NodePayloadRef, Record, RecordRef, SegmentDataRef, TrajectoryRecordStore
from .utils import _record_fields


@dataclass(frozen=True)
class _MemoryNodePayloadRef(NodePayloadRef):
    namespace: str
    record_id: str
    fields: dict[str, Any] = field(compare=False, hash=False, repr=False)


@dataclass(frozen=True)
class _MemorySegmentDataRef(SegmentDataRef):
    namespace: str
    record_id: str
    fields: dict[str, Any] = field(compare=False, hash=False, repr=False)


_MemoryRef: TypeAlias = _MemoryNodePayloadRef | _MemorySegmentDataRef


class InMemoryTrajectoryRecordStore(TrajectoryRecordStore):
    """Carry payloads in refs without a server or copying field values.

    Callers must treat published values and read results as read-only. Releasing
    refs is a no-op; their payloads live until all references are dropped.
    """

    def __init__(self, namespace: str) -> None:
        if not namespace:
            raise ValueError('trajectory store namespace must not be empty')
        self.namespace = namespace

    async def put_many(self, records: Sequence[Record]) -> list[RecordRef]:
        return [
            (_MemoryNodePayloadRef if isinstance(record, NodePayloadRecord) else _MemorySegmentDataRef)(
                self.namespace,
                uuid.uuid4().hex,
                _record_fields(record),
            )
            for record in records
        ]

    async def get_many(
        self,
        refs: Sequence[RecordRef],
        *,
        fields: Collection[str] | None = None,
    ) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = [cast(_MemoryRef, ref).fields for ref in refs]
        if fields is None:
            return payloads
        return [{name: payload[name] for name in fields if name in payload} for payload in payloads]

    async def put_fields_many(self, updates: Mapping[SegmentDataRef, Mapping[str, Any]]) -> None:
        for ref, fields in updates.items():
            if not isinstance(ref, _MemorySegmentDataRef):
                raise TypeError('late fields can only be appended to segment records')
            conflicts = sorted(set(ref.fields).intersection(fields))
            if conflicts:
                raise ValueError(f'conflicting trajectory record fields for {ref.record_id}: {conflicts}')
            ref.fields.update(fields)

    async def release_many(self, refs: Sequence[RecordRef]) -> None:
        pass
