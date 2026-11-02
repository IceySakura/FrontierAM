from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable


ContextT = TypeVar('ContextT')
EventT = TypeVar('EventT')
MetadataT = TypeVar('MetadataT')


@runtime_checkable
class TraceSink(Protocol[ContextT, EventT]):
    """Protocol for writing trace-derived events to a backend."""

    async def start(self, context: ContextT) -> None:
        """Prepare the sink for one trace stream."""

    async def on_event(self, context: ContextT, event: EventT) -> None:
        """Persist or forward one trace-derived event."""

    async def close(self, context: ContextT) -> None:
        """Flush and close the sink for one trace stream."""


@runtime_checkable
class TraceSource(Protocol[MetadataT, EventT]):
    """Protocol for reading persisted trace-derived events."""

    async def read_events_after(self, metadata: MetadataT, cursor: int) -> list[EventT]:
        """Read persisted trace-derived events after the cursor."""
