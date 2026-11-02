from __future__ import annotations

import asyncio

from concurrent.futures import Future

import pytest

from nexus.distributed.utils.threadsafe import await_threadsafe


async def test_await_threadsafe_returns_completed_result() -> None:
    future: Future[str] = Future()
    future.set_result('completed')

    assert await await_threadsafe(future, timeout=1.0) == 'completed'


async def test_await_threadsafe_cancels_pending_future_after_deadline() -> None:
    future: Future[str] = Future()

    with pytest.raises(TimeoutError):
        await await_threadsafe(future, timeout=0.01)

    assert future.cancelled()


async def test_await_threadsafe_cancels_pending_future_when_caller_is_cancelled() -> None:
    future: Future[str] = Future()
    task = asyncio.create_task(await_threadsafe(future, timeout=60.0))
    await asyncio.sleep(0)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert future.cancelled()
