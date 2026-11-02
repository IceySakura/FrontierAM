from __future__ import annotations

import asyncio

import pytest

from nexus.extensions.harbor_bridge import task as task_module


@pytest.fixture(autouse=True)
def _fast_watchdog_pause_poll(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(task_module, '_AGENT_WATCHDOG_PAUSE_POLL_SECONDS', 0.01)


async def _pause_runner_deadline(task_timeout_handle: asyncio.Timeout, pause_seconds: float) -> None:
    """Mirror how the abort gate parks the runner deadline and pushes it out on resume."""
    deadline = task_timeout_handle.when()
    task_timeout_handle.reschedule(None)
    await asyncio.sleep(pause_seconds)
    task_timeout_handle.reschedule(deadline + pause_seconds)


@pytest.mark.asyncio
async def test_watchdog_excludes_runner_deadline_pause() -> None:
    async def agent_run(task_timeout_handle: asyncio.Timeout) -> str:
        await asyncio.sleep(0.05)
        await _pause_runner_deadline(task_timeout_handle, 0.3)
        await asyncio.sleep(0.05)
        return 'done'

    async with asyncio.timeout(10.0) as task_timeout_handle:
        result = await task_module._run_agent_with_watchdog(agent_run(task_timeout_handle), 0.2, task_timeout_handle)

    assert result == 'done'


@pytest.mark.asyncio
async def test_watchdog_follows_runner_deadline_credit() -> None:
    async def agent_run(task_timeout_handle: asyncio.Timeout) -> str:
        # Credit the runner deadline the way non-inference LLM time is credited.
        task_timeout_handle.reschedule(task_timeout_handle.when() + 0.3)
        await asyncio.sleep(0.3)
        return 'done'

    async with asyncio.timeout(10.0) as task_timeout_handle:
        result = await task_module._run_agent_with_watchdog(agent_run(task_timeout_handle), 0.2, task_timeout_handle)

    assert result == 'done'


@pytest.mark.asyncio
async def test_watchdog_honors_pause_that_starts_just_before_expiry() -> None:
    async def agent_run(task_timeout_handle: asyncio.Timeout) -> str:
        await asyncio.sleep(0.25)
        await _pause_runner_deadline(task_timeout_handle, 0.4)
        return 'done'

    async with asyncio.timeout(10.0) as task_timeout_handle:
        result = await task_module._run_agent_with_watchdog(agent_run(task_timeout_handle), 0.3, task_timeout_handle)

    assert result == 'done'


@pytest.mark.asyncio
async def test_watchdog_honors_extension_that_lands_just_before_expiry() -> None:
    async def agent_run(task_timeout_handle: asyncio.Timeout) -> str:
        await asyncio.sleep(0.25)
        task_timeout_handle.reschedule(task_timeout_handle.when() + 0.4)
        await asyncio.sleep(0.25)
        return 'done'

    async with asyncio.timeout(10.0) as task_timeout_handle:
        result = await task_module._run_agent_with_watchdog(agent_run(task_timeout_handle), 0.3, task_timeout_handle)

    assert result == 'done'


@pytest.mark.asyncio
async def test_watchdog_propagates_agent_exception() -> None:
    async def agent_run() -> None:
        await asyncio.sleep(0.01)
        raise ValueError('agent failed')

    async with asyncio.timeout(10.0) as task_timeout_handle:
        with pytest.raises(ValueError, match='agent failed'):
            await task_module._run_agent_with_watchdog(agent_run(), 10.0, task_timeout_handle)


@pytest.mark.asyncio
async def test_watchdog_times_out_without_runner_deadline_pause() -> None:
    agent_cancelled = asyncio.Event()

    async def agent_run() -> None:
        try:
            await asyncio.sleep(10.0)
        except asyncio.CancelledError:
            agent_cancelled.set()
            raise

    async with asyncio.timeout(10.0) as task_timeout_handle:
        with pytest.raises(TimeoutError):
            await task_module._run_agent_with_watchdog(agent_run(), 0.1, task_timeout_handle)

    assert agent_cancelled.is_set()


@pytest.mark.asyncio
async def test_watchdog_still_times_out_after_pause_is_credited() -> None:
    async def agent_run(task_timeout_handle: asyncio.Timeout) -> None:
        await _pause_runner_deadline(task_timeout_handle, 0.1)
        await asyncio.sleep(10.0)

    loop = asyncio.get_running_loop()
    started_at = loop.time()
    async with asyncio.timeout(10.0) as task_timeout_handle:
        with pytest.raises(TimeoutError):
            await task_module._run_agent_with_watchdog(agent_run(task_timeout_handle), 0.2, task_timeout_handle)

    assert loop.time() - started_at >= 0.3


@pytest.mark.asyncio
async def test_watchdog_without_runner_deadline_uses_wall_clock() -> None:
    with pytest.raises(TimeoutError):
        await task_module._run_agent_with_watchdog(asyncio.sleep(10.0), 0.05, None)

    async with asyncio.timeout(None) as task_timeout_handle:
        with pytest.raises(TimeoutError):
            await task_module._run_agent_with_watchdog(asyncio.sleep(10.0), 0.05, task_timeout_handle)


@pytest.mark.asyncio
async def test_runner_deadline_timeout_cancels_agent_and_propagates() -> None:
    agent_cancelled = asyncio.Event()

    async def agent_run() -> None:
        try:
            await asyncio.sleep(10.0)
        except asyncio.CancelledError:
            agent_cancelled.set()
            raise

    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.05) as task_timeout_handle:
            await task_module._run_agent_with_watchdog(agent_run(), 10.0, task_timeout_handle)

    assert agent_cancelled.is_set()
