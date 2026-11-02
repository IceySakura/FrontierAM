from __future__ import annotations

import importlib.util

import pytest

from nexus.extensions.agents.main.processor import MainAgentMetricsProcessor
from nexus.extensions.tasks.claw_eval.token_processor import TokenUsageProcessor
from nexus.runner.batch.trace_processor import DatabaseSpanProcessor
from nexus.runner.trace import RunTraceProcessor
from nexus.runtime.server.worker.sdk import _SIOSpanProcessor
from nexus.trace.processors.chat_tracker import ChatTrackerProcessor
from nexus.trace.processors.chrome_trace import ChromeTraceProcessor
from nexus.trace.processors.hytuner import HytunerProcessor
from nexus.trace.processors.json_line import JsonLineSpanProcessor
from nexus.trace.processors.print import PrintSpanProcessor
from nexus.trace.processors.socketio import SocketIOSpanBroadcaster
from nexus.trace.processors.token_counter import TokenCounterProcessor
from nexus.utils.truncate import DEFAULT_MARKER, DEFAULT_MAX_BYTES


def test_socketio_span_processors_choose_inline_dispatch() -> None:
    assert SocketIOSpanBroadcaster.dispatch_mode == 'inline'
    assert _SIOSpanProcessor.dispatch_mode == 'inline'


def test_socketio_span_broadcaster_default_queue_capacity_is_2000() -> None:
    class Sio:
        async def emit(self, _event, _data):  # noqa: ANN001
            return None

    broadcaster = SocketIOSpanBroadcaster(Sio())  # type: ignore[arg-type]

    assert broadcaster._emitter._queue.maxsize == 2000


@pytest.mark.asyncio
async def test_socketio_span_broadcaster_flushes_emitter(monkeypatch) -> None:
    class Sio:
        async def emit(self, _event, _data):  # noqa: ANN001
            return None

    broadcaster = SocketIOSpanBroadcaster(Sio())  # type: ignore[arg-type]
    flushed = False

    async def flush() -> None:
        nonlocal flushed
        flushed = True

    monkeypatch.setattr(broadcaster._emitter, 'flush', flush)

    await broadcaster.flush()

    assert flushed is True


def test_socketio_span_broadcaster_enqueue_event_truncates_attributes(monkeypatch) -> None:
    class Sio:
        async def emit(self, _event, _data):  # noqa: ANN001
            return None

    broadcaster = SocketIOSpanBroadcaster(Sio())  # type: ignore[arg-type]
    accepted: list[dict] = []

    def enqueue(data: dict) -> bool:
        accepted.append(data)
        return True

    monkeypatch.setattr(broadcaster._emitter, 'enqueue', enqueue)

    assert broadcaster.enqueue_event({'type': 'UPDATE', 'attributes': {'payload': 'x' * 300_000}}) is True
    payload = accepted[0]['attributes']['payload']
    assert DEFAULT_MARKER in payload
    assert len(payload.encode('utf-8')) <= DEFAULT_MAX_BYTES


def test_known_processors_use_inline_dispatch_for_now() -> None:
    processors = [
        ChatTrackerProcessor,
        ChromeTraceProcessor,
        DatabaseSpanProcessor,
        HytunerProcessor,
        JsonLineSpanProcessor,
        RunTraceProcessor,
        MainAgentMetricsProcessor,
        PrintSpanProcessor,
        SocketIOSpanBroadcaster,
        TokenUsageProcessor,
        _SIOSpanProcessor,
        TokenCounterProcessor,
    ]
    if importlib.util.find_spec('langfuse') is not None:
        from nexus.trace.processors.langfuse_tracker import LangfuseTrackerProcessor

        processors.append(LangfuseTrackerProcessor)

    assert {processor.dispatch_mode for processor in processors} == {'inline'}
