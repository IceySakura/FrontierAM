from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from nexus.trace import TraceSink, TraceSource


class DictTraceSink:
    async def start(self, context: Mapping[str, Any]) -> None:
        pass

    async def on_event(self, context: Mapping[str, Any], event: dict[str, Any]) -> None:
        pass

    async def close(self, context: Mapping[str, Any]) -> None:
        pass


class DictTraceSource:
    async def read_events_after(self, metadata: Mapping[str, Any], cursor: int) -> list[dict[str, Any]]:
        return [{'cursor': cursor, **dict(metadata)}]


@pytest.mark.asyncio
async def test_trace_sink_and_source_protocols_are_trace_neutral() -> None:
    sink = DictTraceSink()
    source = DictTraceSource()

    assert isinstance(sink, TraceSink)
    assert isinstance(source, TraceSource)
    assert await source.read_events_after({'trace_id': 'trace-1'}, 3) == [{'trace_id': 'trace-1', 'cursor': 3}]
