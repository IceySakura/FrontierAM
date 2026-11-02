import asyncio
import gc

from types import SimpleNamespace

import pytest

from nexus.runtime.server import rpc_replay_guard
from nexus.runtime.server.rpc_replay_guard import RPCReplayGuard


@pytest.mark.asyncio
async def test_rpc_replay_guard_deduplicates_inflight_requests() -> None:
    guard = RPCReplayGuard()
    handler_started = asyncio.Event()
    release_handler = asyncio.Event()
    call_count = 0

    async def handler() -> dict[str, int]:
        nonlocal call_count
        call_count += 1
        handler_started.set()
        await release_handler.wait()
        return {'call_count': call_count}

    first_task = asyncio.create_task(guard.run('rpc:call_function', 'req-1', handler))
    await handler_started.wait()
    second_task = asyncio.create_task(guard.run('rpc:call_function', 'req-1', handler))

    await asyncio.sleep(0)
    release_handler.set()

    first_result, second_result = await asyncio.gather(first_task, second_task)

    assert first_result == {'call_count': 1}
    assert second_result == {'call_count': 1}
    assert call_count == 1


@pytest.mark.asyncio
async def test_rpc_replay_guard_replays_completed_results() -> None:
    guard = RPCReplayGuard()
    call_count = 0

    async def handler() -> dict[str, int]:
        nonlocal call_count
        call_count += 1
        return {'call_count': call_count}

    first_result = await guard.run('rpc:call_tool', 'req-2', handler)
    second_result = await guard.run('rpc:call_tool', 'req-2', handler)

    assert first_result == {'call_count': 1}
    assert second_result == {'call_count': 1}
    assert call_count == 1


@pytest.mark.asyncio
async def test_rpc_replay_guard_ttl_starts_when_the_call_completes(monkeypatch) -> None:
    clock = [0.0]
    monkeypatch.setattr(rpc_replay_guard, 'time', SimpleNamespace(monotonic=lambda: clock[0]))
    guard = RPCReplayGuard(ttl_seconds=300.0)
    call_count = 0

    async def slow_handler() -> int:
        nonlocal call_count
        call_count += 1
        clock[0] += 310.0
        return call_count

    assert await guard.run('llm_request', 'req-slow', slow_handler) == 1
    clock[0] += 30.0
    assert await guard.run('llm_request', 'req-slow', slow_handler) == 1
    clock[0] += 300.0
    assert await guard.run('llm_request', 'req-slow', slow_handler) == 2


@pytest.mark.asyncio
async def test_rpc_replay_guard_discards_completed_results() -> None:
    guard = RPCReplayGuard()
    call_count = 0

    async def handler() -> dict[str, int]:
        nonlocal call_count
        call_count += 1
        return {'call_count': call_count}

    assert await guard.run('llm_request', 'req-delivered', handler) == {'call_count': 1}

    assert await guard.discard('llm_request', 'req-delivered') is True
    assert await guard.discard('llm_request', 'req-delivered') is False
    assert await guard.run('llm_request', 'req-delivered', handler) == {'call_count': 2}


@pytest.mark.asyncio
async def test_rpc_replay_guard_discards_completed_results_after_delay() -> None:
    guard = RPCReplayGuard()
    call_count = 0

    async def handler() -> dict[str, int]:
        nonlocal call_count
        call_count += 1
        return {'call_count': call_count}

    assert await guard.run('llm_request', 'req-delayed', handler) == {'call_count': 1}
    assert await guard.discard('llm_request', 'req-delayed', delay=0.01) is True

    assert await guard.run('llm_request', 'req-delayed', handler) == {'call_count': 1}
    await asyncio.sleep(0.02)
    assert await guard.run('llm_request', 'req-delayed', handler) == {'call_count': 2}


@pytest.mark.asyncio
async def test_rpc_replay_guard_delayed_discard_does_not_remove_replacement() -> None:
    guard = RPCReplayGuard()
    call_count = 0

    async def handler() -> dict[str, int]:
        nonlocal call_count
        call_count += 1
        return {'call_count': call_count}

    assert await guard.run('llm_request', 'req-replaced', handler) == {'call_count': 1}
    assert await guard.discard('llm_request', 'req-replaced', delay=0.01) is True
    assert await guard.discard('llm_request', 'req-replaced') is True
    assert await guard.run('llm_request', 'req-replaced', handler) == {'call_count': 2}

    await asyncio.sleep(0.02)
    assert await guard.run('llm_request', 'req-replaced', handler) == {'call_count': 2}


@pytest.mark.asyncio
async def test_rpc_replay_guard_does_not_discard_inflight_results() -> None:
    guard = RPCReplayGuard()
    handler_started = asyncio.Event()
    release_handler = asyncio.Event()

    async def handler() -> str:
        handler_started.set()
        await release_handler.wait()
        return 'done'

    task = asyncio.create_task(guard.run('llm_request', 'req-inflight', handler))
    await handler_started.wait()

    assert await guard.discard('llm_request', 'req-inflight') is False
    assert 'llm_request:req-inflight' in guard._entries

    release_handler.set()
    assert await task == 'done'


@pytest.mark.asyncio
async def test_rpc_replay_guard_cleans_up_cancelled_owner() -> None:
    guard = RPCReplayGuard()
    handler_started = asyncio.Event()
    call_count = 0

    async def blocked_handler() -> dict[str, int]:
        nonlocal call_count
        call_count += 1
        handler_started.set()
        await asyncio.Future()
        return {'call_count': call_count}

    owner_task = asyncio.create_task(guard.run('rpc:call_function', 'req-3', blocked_handler))
    await handler_started.wait()

    duplicate_task = asyncio.create_task(guard.run('rpc:call_function', 'req-3', blocked_handler))
    await asyncio.sleep(0)

    owner_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await owner_task
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(duplicate_task, timeout=0.5)

    async def retry_handler() -> dict[str, int]:
        nonlocal call_count
        call_count += 1
        return {'call_count': call_count}

    retry_result = await guard.run('rpc:call_function', 'req-3', retry_handler)

    assert retry_result == {'call_count': 2}
    assert call_count == 2


@pytest.mark.asyncio
async def test_rpc_replay_guard_failed_owner_does_not_leak_future_exception() -> None:
    """A failing owner with no duplicate must not leave an unretrieved future.

    Reproduces the "Future exception was never retrieved" log noise: the owner
    stores the exception on its replay future and re-raises, but never awaits that
    future itself. Without the retrieving done-callback the pruned/GC'd future logs
    a spurious asyncio error for every failed request.
    """

    class BoomError(RuntimeError):
        pass

    async def failing_handler() -> dict[str, int]:
        raise BoomError('context length exceeded')

    guard = RPCReplayGuard()

    # Owner runs and raises; the exception must propagate to the caller unchanged.
    with pytest.raises(BoomError):
        await guard.run('rpc:call_function', 'req-boom', failing_handler)

    # Done-callbacks are scheduled via loop.call_soon; let them run.
    await asyncio.sleep(0)

    # Grab the stored future before pruning removes it, then confirm its exception
    # was already retrieved by the done-callback (so GC would not warn).
    entry = guard._entries.get('rpc:call_function:req-boom')
    assert entry is not None
    stored_future = entry.future
    assert stored_future.done()
    assert stored_future._log_traceback is False  # asyncio's "needs retrieval" flag cleared

    # Dropping and collecting the future must not raise or warn.
    del entry
    del stored_future
    gc.collect()
