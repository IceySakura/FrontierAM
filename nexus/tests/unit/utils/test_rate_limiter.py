from __future__ import annotations

import asyncio
import math
import threading
import time

import pytest

from nexus.utils.rate_limiter import (
    QpsLimiter,
    get_qps_limiter,
    get_rate_limiter,
    reset_all_qps_limiters,
    reset_all_rate_limiters,
    reset_qps_limiter,
)


@pytest.fixture(autouse=True)
def reset_registries() -> None:
    reset_all_qps_limiters()
    reset_all_rate_limiters()
    yield
    reset_all_qps_limiters()
    reset_all_rate_limiters()


@pytest.mark.asyncio
async def test_qps_limiter_spaces_concurrent_acquisitions() -> None:
    limiter = QpsLimiter('paced', qps=40)
    issued_at: list[float] = []

    async def acquire() -> None:
        await limiter.acquire()
        issued_at.append(time.monotonic())

    await asyncio.gather(*(acquire() for _ in range(4)))

    minimum_interval = 1 / limiter.qps
    intervals = [later - earlier for earlier, later in zip(issued_at, issued_at[1:], strict=False)]
    assert all(interval >= minimum_interval * 0.9 for interval in intervals)
    assert limiter.snapshot()['issued'] == 4
    assert limiter.snapshot()['waiting'] == 0


@pytest.mark.asyncio
async def test_qps_limiter_first_acquisition_is_immediate() -> None:
    limiter = QpsLimiter('immediate', qps=1)

    assert await limiter.acquire() < 50


@pytest.mark.asyncio
async def test_cancelled_waiters_do_not_create_scheduling_debt() -> None:
    limiter = QpsLimiter('cancelled', qps=20)
    await limiter.acquire()
    waiters = [asyncio.create_task(limiter.acquire()) for _ in range(10)]
    await asyncio.sleep(0)
    assert limiter.snapshot()['waiting'] == 10

    for waiter in waiters:
        waiter.cancel()
    await asyncio.gather(*waiters, return_exceptions=True)

    started_at = time.monotonic()
    await limiter.acquire()
    elapsed = time.monotonic() - started_at
    assert elapsed < 0.2
    assert limiter.snapshot()['waiting'] == 0
    assert limiter.snapshot()['issued'] == 2


@pytest.mark.asyncio
async def test_cancelled_middle_waiter_moves_later_reservations_forward() -> None:
    limiter = QpsLimiter('cancelled-middle', qps=10)
    await limiter.acquire()
    waiters = [asyncio.create_task(limiter.acquire()) for _ in range(3)]
    await asyncio.sleep(0)

    started_at = time.monotonic()
    waiters[1].cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiters[1]
    await asyncio.gather(waiters[0], waiters[2])

    assert time.monotonic() - started_at < 0.27
    assert limiter.snapshot()['waiting'] == 0
    assert limiter.snapshot()['issued'] == 3


@pytest.mark.asyncio
async def test_partial_cancellation_of_large_queue_does_not_strand_slots() -> None:
    # A large queue with most of its middle cancelled must not leave the survivors
    # pacing behind empty slots the cancelled waiters would have consumed.
    limiter = QpsLimiter('partial-cancel', qps=100)
    await limiter.acquire()
    waiters = [asyncio.create_task(limiter.acquire()) for _ in range(40)]
    await asyncio.sleep(0)
    assert limiter.snapshot()['waiting'] == 40

    survivors = waiters[:5] + waiters[-5:]
    cancelled = waiters[5:-5]
    for waiter in cancelled:
        waiter.cancel()
    await asyncio.gather(*cancelled, return_exceptions=True)

    started_at = time.monotonic()
    await asyncio.gather(*survivors)
    elapsed = time.monotonic() - started_at

    # Ten survivors at a 10ms interval pace in ~0.1s; inheriting the 30 cancelled
    # slots' debt would push this past 0.3s.
    assert elapsed < 0.25
    assert limiter.snapshot()['waiting'] == 0
    assert limiter.snapshot()['issued'] == 11


@pytest.mark.asyncio
async def test_context_error_does_not_block_later_acquisition() -> None:
    limiter = QpsLimiter('context-error', qps=1000)

    with pytest.raises(RuntimeError, match='request failed'):
        async with limiter:
            raise RuntimeError('request failed')

    await asyncio.wait_for(limiter.acquire(), timeout=0.1)
    assert limiter.snapshot()['issued'] == 2


@pytest.mark.parametrize('qps', [0, -1, math.nan, math.inf, -math.inf, 5e-324])
def test_qps_limiter_rejects_invalid_qps(qps: float) -> None:
    with pytest.raises(ValueError, match='(qps|interval) must be a finite positive number'):
        QpsLimiter('invalid', qps)


def test_qps_limiter_paces_across_event_loops() -> None:
    limiter = QpsLimiter('cross-loop', qps=20)
    issued_at: list[float] = []
    issued_lock = threading.Lock()

    async def run_batch() -> None:
        async def one() -> None:
            await limiter.acquire()
            with issued_lock:
                issued_at.append(time.monotonic())

        await asyncio.gather(*(one() for _ in range(5)))

    threads = [threading.Thread(target=lambda: asyncio.run(run_batch())) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(issued_at) == 10
    issued_at.sort()
    minimum_interval = 1 / limiter.qps
    intervals = [later - earlier for earlier, later in zip(issued_at, issued_at[1:], strict=False)]
    assert all(interval >= minimum_interval * 0.9 for interval in intervals)
    assert limiter.snapshot()['issued'] == 10
    assert limiter.snapshot()['waiting'] == 0


def test_qps_limiter_rechecks_pacing_when_older_reservation_commits_late() -> None:
    limiter = QpsLimiter('out-of-order', qps=20)
    asyncio.run(limiter.acquire())

    older_reserved = threading.Event()
    newer_issued = threading.Event()
    older_issued_at: list[float] = []

    async def acquire_older() -> None:
        task = asyncio.create_task(limiter.acquire())
        await asyncio.sleep(0)
        older_reserved.set()

        # Block only this event loop so a later reservation on another loop can issue first.
        assert newer_issued.wait(timeout=1)
        await task
        older_issued_at.append(time.monotonic())

    older_thread = threading.Thread(target=lambda: asyncio.run(acquire_older()))
    older_thread.start()
    assert older_reserved.wait(timeout=1)

    asyncio.run(limiter.acquire())
    newer_issued_at = time.monotonic()
    newer_issued.set()
    older_thread.join(timeout=1)

    assert not older_thread.is_alive()
    assert len(older_issued_at) == 1
    assert older_issued_at[0] - newer_issued_at >= limiter._interval * 0.9
    assert limiter.snapshot()['issued'] == 3
    assert limiter.snapshot()['waiting'] == 0


def test_named_qps_limiter_reuses_matching_configuration_and_rejects_conflicts() -> None:
    qps_limiter = get_qps_limiter('backend', 10)
    concurrency_limiter = get_rate_limiter('backend', 2)

    assert get_qps_limiter('backend', 10) is qps_limiter
    assert get_rate_limiter('backend', 2) is concurrency_limiter
    with pytest.raises(ValueError, match='already configured with qps=10'):
        get_qps_limiter('backend', 20)


def test_qps_limiter_reset_one_and_all() -> None:
    first = get_qps_limiter('first', 10)
    reset_qps_limiter('missing')
    reset_qps_limiter('first')
    assert get_qps_limiter('first', 10) is not first

    second = get_qps_limiter('second', 10)
    reset_all_qps_limiters()
    assert get_qps_limiter('second', 10) is not second
