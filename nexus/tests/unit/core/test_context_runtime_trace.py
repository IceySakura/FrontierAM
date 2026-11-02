from __future__ import annotations

import asyncio
import logging

from typing import Any

import pytest

from nexus.core.context import ExecutionContext
from nexus.runtime import RuntimeProvider


class _RuntimeProvider(RuntimeProvider):
    def __init__(self, events: list[str]) -> None:
        super().__init__()
        self._events = events

    async def acquire(self, name: str, **params: Any) -> Any:  # pragma: no cover - not used by these tests
        raise NotImplementedError

    async def _release_impl(self, name: str) -> None:  # pragma: no cover - release_all is overridden
        raise NotImplementedError

    async def _release_all_impl(self) -> None:
        self._events.append('release')


@pytest.mark.asyncio
async def test_context_drains_runtime_trace_before_release() -> None:
    events: list[str] = []

    class Runtime:
        async def drain_trace(self) -> None:
            events.append('drain')

    context = ExecutionContext(runtime_provider=_RuntimeProvider(events))
    context._runtimes['main'] = Runtime()  # type: ignore[assignment]

    async with context:
        pass

    assert events == ['drain', 'release']


@pytest.mark.asyncio
async def test_context_releases_runtime_when_trace_drain_fails(caplog: pytest.LogCaptureFixture) -> None:
    events: list[str] = []

    class Runtime:
        async def drain_trace(self) -> None:
            raise RuntimeError('trace barrier unavailable')

    context = ExecutionContext(runtime_provider=_RuntimeProvider(events))
    context._runtimes['main'] = Runtime()  # type: ignore[assignment]

    async with context:
        pass

    assert events == ['release']
    assert 'Failed to drain runtime trace for main' in caplog.text


@pytest.mark.asyncio
async def test_context_ignores_trace_drain_child_cancellation(caplog: pytest.LogCaptureFixture) -> None:
    events: list[str] = []

    class Runtime:
        async def drain_trace(self) -> None:
            raise asyncio.CancelledError

    context = ExecutionContext(runtime_provider=_RuntimeProvider(events))
    context._runtimes['main'] = Runtime()  # type: ignore[assignment]

    with caplog.at_level(logging.WARNING, logger='nexus.core.context'):
        async with context:
            pass

    assert events == ['release']
    assert '[nexus-context-trace-drain-child-cancellation]' in caplog.text
    assert 'task_cancelling=0' in caplog.text


@pytest.mark.asyncio
async def test_context_releases_runtime_before_propagating_external_cancellation() -> None:
    events: list[str] = []
    drain_started = asyncio.Event()

    class Runtime:
        async def drain_trace(self) -> None:
            drain_started.set()
            await asyncio.Event().wait()

    context = ExecutionContext(runtime_provider=_RuntimeProvider(events))
    context._runtimes['main'] = Runtime()  # type: ignore[assignment]

    async def use_context() -> None:
        async with context:
            pass

    context_task = asyncio.create_task(use_context())
    await drain_started.wait()
    context_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await context_task

    assert events == ['release']


@pytest.mark.asyncio
async def test_context_preserves_primary_error_when_trace_drain_child_is_cancelled(
    caplog: pytest.LogCaptureFixture,
) -> None:
    events: list[str] = []

    class Runtime:
        async def drain_trace(self) -> None:
            raise asyncio.CancelledError('trace drain cancelled')

    context = ExecutionContext(
        runtime_provider=_RuntimeProvider(events),
        task_name='search_stateful',
        agent_name='search_agent',
        run_id='run-123',
    )
    context._runtimes['main'] = Runtime()  # type: ignore[assignment]

    with caplog.at_level(logging.WARNING, logger='nexus.core.context'):
        with pytest.raises(ValueError, match='primary task failure'):
            async with context:
                raise ValueError('primary task failure')

    assert events == ['release']
    assert '[nexus-context-trace-drain-child-cancellation]' in caplog.text
    assert 'primary_error_type=ValueError' in caplog.text
    assert 'cleanup_error_type=CancelledError' in caplog.text
    assert 'cleanup_stage=drain_trace:main' in caplog.text
    assert 'task_name=search_stateful' in caplog.text
    assert 'agent_name=search_agent' in caplog.text
    assert 'run_id=run-123' in caplog.text
    assert 'task_cancelling=0' in caplog.text
