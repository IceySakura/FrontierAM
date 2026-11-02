"""Trajectory record store types shared by backends."""

from __future__ import annotations

from collections.abc import Collection, Coroutine, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np


class NodePayloadRef:
    """Node reference; callers pass it intact to the store."""


class SegmentDataRef:
    """Segment reference; callers pass it intact to the store."""


class DumpRef:
    """Represent persistent payload metadata without retaining runtime references.

    Backend-specific subclasses describe saved references. They are not live
    NodePayloadRef or SegmentDataRef instances and cannot be passed to get_many.
    """


@dataclass(frozen=True, eq=False)
class NodePayloadRecord:
    """One node's producer-local payload, published at seal time.

    Everything a message contributes travels together: its tokens, the sampled spans'
    logprobs and routing, and the media this message introduced. Segment-level shapes are
    derived from these per-node records by the consumer.

    Tokens travel in the training view: the expanded span when the message carries images,
    the raw span otherwise, so a media-free payload is byte-identical to what the engine saw.

    Media travels as the processor tensors the training side reads. The source bytes stay
    out: no consumer decodes them, and a screenshot-sized payload would dominate the
    store while being read by nobody.
    """

    trajectory_id: str
    node_id: int
    token_ids: np.ndarray
    """This message's tokens in the training view."""
    logprobs: np.ndarray | None = None
    topk_indices: np.ndarray | None = None
    topk_values: np.ndarray | None = None
    routed_experts: Any | None = None
    multi_modal_items: dict[str, list[dict[str, Any]]] | None = None
    """Per-item processor tensors for the media this message introduced, keyed by modality."""


@dataclass(frozen=True)
class SegmentDataRecord:
    trajectory_id: str
    segment_id: str
    fields: Mapping[str, Any]


RecordRef: TypeAlias = NodePayloadRef | SegmentDataRef
Record: TypeAlias = NodePayloadRecord | SegmentDataRecord


class TrajectoryRecordStore:
    """Define the operations for trajectory store backends."""

    def dump_payloads(
        self,
        real_refs: list[RecordRef],
        *,
        dump_dir: str,
        fields: Collection[str] | None = None,
    ) -> tuple[list[DumpRef], Coroutine[Any, Any, None]]:
        """Prepare persistent references and return the coroutine that saves their payloads.

        Args:
            real_refs: Borrowed runtime references; export must not mutate or release them.
            dump_dir: Shared directory in which this dump writes its payload files and index.
            fields: Payload fields to export; None selects all fields. Missing fields are omitted.

        Returns:
            Persistent refs in input order, preserving duplicates, and a coroutine
            the caller must await to save the payload files and index. Preparation
            performs no file I/O and does not start background work.

        Raises:
            NotImplementedError: The backend does not support payload reference export.
        """
        raise NotImplementedError(f'{type(self).__name__} does not implement dump_payloads')

    async def load_payloads(
        self, dump_refs: list[DumpRef], *, dump_dir: str, exclude_node_id: str | None = None
    ) -> list[RecordRef]:
        """Restore runtime references from persistent references in input order.

        Args:
            dump_refs: Persistent references returned by dump_payloads.
            dump_dir: Directory containing the payload files and index for these refs.
            exclude_node_id: Ray node to exclude from shard loading. None excludes the caller's node.

        Returns:
            One runtime reference for each input occurrence, preserving duplicates.

        Raises:
            NotImplementedError: The backend does not support payload restoration.
        """
        raise NotImplementedError(f'{type(self).__name__} does not implement load_payloads')

    async def put_many(self, records: Sequence[Record]) -> list[RecordRef]:
        raise NotImplementedError('Store backend must implement put_many')

    async def get_many(
        self,
        refs: Sequence[RecordRef],
        *,
        fields: Collection[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Return payloads in ref order, omitting missing fields; None selects all fields."""
        raise NotImplementedError('Store backend must implement get_many')

    async def put_fields_many(
        self,
        updates: Mapping[SegmentDataRef, Mapping[str, Any]],
    ) -> None:
        raise NotImplementedError('Store backend must implement put_fields_many')

    async def release_many(self, refs: Sequence[RecordRef]) -> None:
        raise NotImplementedError('Store backend must implement release_many')
