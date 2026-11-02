"""Episode time that leaves out the stretches a paused cerebellum freezes the whole episode.

A HyTuner partial rollout parks the policy, the cerebellum here, at its abort gate while the trainer
runs a step. Background work does not stop: CodeBuddy and the midbrain call their own models and keep
going, and the time they spend is real work. The episode is frozen only while the cerebellum is parked
*and* nothing in the background is working, because then every timer is just waiting on the trainer.
The delivery deadline and CodeBuddy's wall-clock cap are measured on this clock, so they skip exactly
those stretches.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging

from collections.abc import Callable, Iterator
from typing import Any


logger = logging.getLogger(__name__)

_TICK_S = 0.5


class EpisodeClock:
    """Accumulate how long the cerebellum was parked and how long the episode was frozen.

    State is sampled every ``tick_s``: a change of state is attributed to the tick it is seen in, so
    each pause is measured to within one tick. An unbound clock never freezes, as for a policy client
    with no abort gate.
    """

    def __init__(self, *, tick_s: float = _TICK_S) -> None:
        self.tick_s = tick_s
        self.cerebellum_paused_s = 0.0
        """Time the cerebellum spent inside a model call that the trainer had paused."""
        self.frozen_s = 0.0
        """Part of ``cerebellum_paused_s`` during which no background work ran either."""
        self._abort_event: asyncio.Event | None = None
        self._background_working: Callable[[], bool] = lambda: False
        self._calls_in_progress = 0
        self._started_at: float | None = None
        self._stopped_at: float | None = None
        self._last_tick_at = 0.0
        self._held_scopes: set[Any] = set()
        self._ticker: asyncio.Task[None] | None = None

    def bind(self, *, abort_event: asyncio.Event | None, background_working: Callable[[], bool]) -> None:
        """Watch ``abort_event`` (set while generation may run, cleared while paused) and the background."""
        self._abort_event = abort_event
        self._background_working = background_working

    @property
    def cerebellum_paused(self) -> bool:
        return self._calls_in_progress > 0 and self._abort_event is not None and not self._abort_event.is_set()

    @property
    def frozen(self) -> bool:
        return self.cerebellum_paused and not self._background_working()

    def start(self) -> None:
        loop = asyncio.get_running_loop()
        self._started_at = loop.time()
        self._last_tick_at = self._started_at
        self._ticker = asyncio.create_task(self._tick_forever(), name='hy-live-episode-clock')

    async def stop(self) -> None:
        ticker, self._ticker = self._ticker, None
        if ticker is not None:
            ticker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await ticker
            self._tick()
            self._stopped_at = self._last_tick_at

    def elapsed(self) -> float:
        """Seconds since ``start`` that the episode was not frozen."""
        return self.wall_s() - self.frozen_s

    def wall_s(self) -> float:
        if self._started_at is None:
            raise RuntimeError('EpisodeClock read before start()')
        end = self._stopped_at if self._stopped_at is not None else asyncio.get_running_loop().time()
        return end - self._started_at

    @contextlib.contextmanager
    def cerebellum_call(self) -> Iterator[None]:
        """Mark one cerebellum model call; the trainer can only park the cerebellum inside one."""
        self._calls_in_progress += 1
        try:
            yield
        finally:
            self._calls_in_progress -= 1

    @contextlib.contextmanager
    def hold(self, scope: Any) -> Iterator[None]:
        """Push an entered ``asyncio.timeout`` scope out by every frozen stretch while it is open."""
        self._held_scopes.add(scope)
        try:
            yield
        finally:
            self._held_scopes.discard(scope)

    def summary(self) -> dict[str, float]:
        if self._started_at is None:
            return {}
        return {
            'wall_s': round(self.wall_s(), 3),
            'cerebellum_paused_s': round(self.cerebellum_paused_s, 3),
            'frozen_s': round(self.frozen_s, 3),
        }

    async def _tick_forever(self) -> None:
        while True:
            await asyncio.sleep(self.tick_s)
            self._tick()

    def _tick(self) -> None:
        now = asyncio.get_running_loop().time()
        delta, self._last_tick_at = now - self._last_tick_at, now
        if not self.cerebellum_paused:
            return
        self.cerebellum_paused_s += delta
        if self._background_working():
            return
        self.frozen_s += delta
        for scope in list(self._held_scopes):
            deadline = scope.when()
            # A deadline already passed has fired; its scope is unwinding and cannot be moved.
            if deadline is None or deadline <= now:
                continue
            scope.reschedule(deadline + delta)


__all__ = ['EpisodeClock']
