from __future__ import annotations

import asyncio

import pytest

from nexus.extensions.projects.hy_live.episode_clock import EpisodeClock
from nexus.utils.timeout import async_timeout


TICK_S = 0.02


def _running_clock(abort_event: asyncio.Event, background: list[bool]) -> EpisodeClock:
    clock = EpisodeClock(tick_s=TICK_S)
    clock.bind(abort_event=abort_event, background_working=lambda: background[0])
    clock.start()
    return clock


@pytest.mark.asyncio
async def test_parked_cerebellum_with_idle_background_freezes_the_episode() -> None:
    abort_event = asyncio.Event()
    clock = _running_clock(abort_event, [False])

    with clock.cerebellum_call():
        await asyncio.sleep(0.2)
    await clock.stop()

    assert clock.cerebellum_paused_s == pytest.approx(0.2, abs=0.06)
    assert clock.frozen_s == pytest.approx(0.2, abs=0.06)
    assert clock.elapsed() == pytest.approx(clock.wall_s() - clock.frozen_s)
    assert clock.elapsed() < 0.06


@pytest.mark.asyncio
async def test_background_work_keeps_time_running_while_the_cerebellum_is_parked() -> None:
    abort_event = asyncio.Event()
    clock = _running_clock(abort_event, [True])

    with clock.cerebellum_call():
        await asyncio.sleep(0.2)
    await clock.stop()

    assert clock.cerebellum_paused_s == pytest.approx(0.2, abs=0.06), 'The pause itself is still measured'
    assert clock.frozen_s == 0.0, 'CodeBuddy or the midbrain working means the episode is not frozen'
    assert clock.elapsed() == pytest.approx(0.2, abs=0.06)


@pytest.mark.asyncio
async def test_paused_trainer_does_not_freeze_a_cerebellum_outside_a_call() -> None:
    abort_event = asyncio.Event()
    clock = _running_clock(abort_event, [False])

    await asyncio.sleep(0.1)
    abort_event.set()
    with clock.cerebellum_call():
        await asyncio.sleep(0.1)
    await clock.stop()

    assert clock.cerebellum_paused_s == 0.0, 'Nothing waits on the trainer until the cerebellum makes a call'
    assert clock.frozen_s == 0.0


@pytest.mark.asyncio
async def test_unbound_clock_never_freezes() -> None:
    clock = EpisodeClock(tick_s=TICK_S)
    clock.start()

    with clock.cerebellum_call():
        await asyncio.sleep(0.05)
    await clock.stop()

    assert clock.summary()['frozen_s'] == 0.0


@pytest.mark.asyncio
async def test_held_timeout_is_pushed_out_by_frozen_time() -> None:
    abort_event = asyncio.Event()
    clock = _running_clock(abort_event, [False])
    loop = asyncio.get_running_loop()
    started = loop.time()

    async def park_the_cerebellum() -> None:
        with clock.cerebellum_call():
            await asyncio.sleep(0.3)

    async def wait_under_held_cap() -> None:
        async with async_timeout(0.2) as scope:
            with clock.hold(scope):
                await asyncio.sleep(10)

    parking = asyncio.create_task(park_the_cerebellum())
    with pytest.raises(TimeoutError):
        await wait_under_held_cap()
    await parking
    await clock.stop()

    assert loop.time() - started == pytest.approx(0.5, abs=0.1), (
        'A 0.2 s cap that overlaps 0.3 s of frozen time fires after 0.5 s of wall time'
    )
