"""Private scalar facts for direct Trajectory assembly."""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass, field
from typing import Any

from nexus.gateway.trajectory import MessageId, NodeId, SegmentBoundary


@dataclass(frozen=True)
class PendingBoundary:
    boundary: SegmentBoundary
    claims: tuple[tuple[str, Hashable], ...] = ()


@dataclass(frozen=True)
class RuntimeFactOccurrence:
    sequence: int
    fact: Any


@dataclass(frozen=True)
class NodeRuntimeFacts:
    identity: Hashable
    alias_identity: Hashable = None  # how its own chain first resent an LLM Node, set at most once
    tools_identity: Hashable = None
    context_key: Hashable | None = None
    boundary_claims: tuple[tuple[str, Hashable], ...] = ()
    occurrences: tuple[RuntimeFactOccurrence, ...] = ()

    def facts_of_type(self, fact_type: type) -> list[tuple[int, Any]]:
        return [
            (occurrence.sequence, occurrence.fact)
            for occurrence in self.occurrences
            if isinstance(occurrence.fact, fact_type)
        ]


@dataclass
class RuntimeFactsLedger:
    by_node: dict[NodeId, NodeRuntimeFacts] = field(default_factory=dict)


@dataclass
class RuntimeState:
    facts: RuntimeFactsLedger = field(default_factory=RuntimeFactsLedger)
    # Append-only: the LLM Node behind each logical message, which resent copies share.
    llm_by_message_id: dict[MessageId, NodeId] = field(default_factory=dict)
    next_node_id: int = 0
    next_message_id: int = 0
    next_request_sequence: int = 0
    next_fact_sequence: int = 0
    pending_boundary: PendingBoundary | None = None
