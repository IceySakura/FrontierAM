"""Minimal asyncio event loop lag monitor with optional watchdog thread."""

from __future__ import annotations

import asyncio
import collections
import contextlib
import logging
import threading
import time

from collections.abc import Callable


logger = logging.getLogger(__name__)


class EventLoopMonitor:
    """Lightweight event loop scheduling latency probe.

    Periodically schedules a callback `interval` seconds in the future and
    measures how late it actually fires. The delay is the event loop
    scheduling lag — large values mean something blocked the loop.

    Optionally runs a watchdog thread that periodically checks the stats and
    logs a warning when lag exceeds a threshold. The watchdog lives in a
    separate OS thread on purpose: when the event loop is fully blocked, an
    asyncio task could not run, but the thread still fires the alert.

    Thread-safety: the probe runs on the event loop thread and the watchdog on
    a separate OS thread; both touch the running stats, so a lock guards them.
    start()/stop() are expected to be called from the loop thread, but stop()
    is safe to call from any thread.

    Usage:
        monitor = EventLoopMonitor()
        monitor.start()                              # probe only
        # or with watchdog:
        monitor.start(watchdog_interval=1.0, lag_threshold=0.1)
        # ... application runs ...
        stats = monitor.stats  # => {max, avg, min, samples}
        monitor.stop()
    """

    def __init__(self):
        self._loop: asyncio.AbstractEventLoop | None = None
        self._interval: float = 0.0
        self._expected: float = 0.0
        self._handle: asyncio.TimerHandle | None = None
        self._running = False
        # Guards the running stats below, shared between the probe thread and
        # the watchdog thread.
        self._lock = threading.Lock()
        self._samples = 0
        self._total = 0.0
        self._lag_max = 0.0
        self._lag_min = float('inf')
        # Wall-clock (monotonic) timestamp of the last probe run, readable from
        # the watchdog thread to detect a stalled loop.
        self._last_probe_monotonic: float = 0.0
        # Watchdog
        self._watchdog_interval: float = 0.0
        self._lag_threshold: float = 0.0
        self._window_buckets: int = 0
        self._on_sample: Callable[[float], None] | None = None
        self._on_alert: Callable[[str, float, dict], None] | None = None
        self._watchdog_thread: threading.Thread | None = None
        # Set on stop() to wake the watchdog immediately instead of waiting out
        # its sleep — keeps stop() from blocking the event loop thread.
        self._wakeup = threading.Event()

    def start(
        self,
        interval: float = 0.01,
        *,
        watchdog_interval: float | None = None,
        lag_threshold: float = 0.1,
        window_seconds: float | None = None,
        on_alert: Callable[[str, float, dict], None] | None = None,
        on_sample: Callable[[float], None] | None = None,
    ):
        """Start the probe. Must be called from within a running event loop.

        Args:
            interval: probe frequency in seconds (default 10ms).
            watchdog_interval: if set, spawn a watchdog thread that checks the
                stats every `watchdog_interval` seconds. None disables it.
            lag_threshold: alert when lag reaches this many seconds (default
                100ms). Applied to BOTH the sliding-window max (spike) and the
                sliding-window avg (sustained).
            window_seconds: length of the sliding window for alerting. Defaults
                to 5 × watchdog_interval. The window slides one watchdog tick at
                a time, so a one-off spike ages out and alerts auto-clear.
            on_alert: callback(reason, value, stats). `reason` is one of
                "spike", "sustained", "stall". Defaults to logging a warning.
        """
        if self._running:
            return
        # A previous watchdog that wouldn't die (e.g. a hung on_alert) must not
        # be silently leaked into a second concurrent thread.
        if self._watchdog_thread is not None and self._watchdog_thread.is_alive():
            raise RuntimeError('previous watchdog thread is still alive; cannot start')

        # Resolve the loop first; if there's no running loop this raises
        # RuntimeError before we mutate any state, leaving the object clean.
        loop = asyncio.get_running_loop()
        self._loop = loop
        self._on_sample = on_sample
        self._interval = interval
        self._expected = loop.time() + interval
        self._last_probe_monotonic = time.monotonic()
        # Clear stats so a restart doesn't inherit a stale peak that would fire
        # a spurious spike on the first watchdog tick.
        with self._lock:
            self._samples = 0
            self._total = 0.0
            self._lag_max = 0.0
            self._lag_min = float('inf')
        self._running = True
        self._handle = loop.call_later(interval, self._probe)

        if watchdog_interval is not None:
            self._watchdog_interval = watchdog_interval
            self._lag_threshold = lag_threshold
            window = window_seconds if window_seconds is not None else watchdog_interval * 5
            self._window_buckets = max(1, round(window / watchdog_interval))
            self._on_alert = on_alert or self._default_alert
            self._wakeup.clear()
            self._watchdog_thread = threading.Thread(
                target=self._watchdog_loop,
                name='EventLoopMonitor-watchdog',
                daemon=True,
            )
            self._watchdog_thread.start()

    def stop(self):
        """Stop the probe and watchdog. Safe to call from any thread."""
        self._running = False

        # Cancelling a TimerHandle touches loop-owned state, which is not
        # thread-safe. If we're not on the loop thread, hand it to the loop;
        # otherwise the probe also self-terminates on its next tick via the
        # _running flag, so a missed cancel is harmless.
        handle, self._handle = self._handle, None
        if handle is not None:
            if self._on_loop_thread():
                handle.cancel()
            else:
                # loop already closed → probe can't run anyway, so a missed cancel is harmless
                loop = self._loop
                if loop is not None:
                    with contextlib.suppress(RuntimeError):
                        loop.call_soon_threadsafe(handle.cancel)

        if self._watchdog_thread is not None:
            # Wake the watchdog immediately instead of letting it sleep out the
            # full interval, so stop() returns promptly without blocking the loop.
            self._wakeup.set()
            self._watchdog_thread.join(timeout=1.0)
            if self._watchdog_thread.is_alive():
                # Most likely a hung on_alert callback. Keep the reference so a
                # later start() refuses rather than spawning a second thread.
                logger.warning('EventLoopMonitor watchdog did not stop within 1s (on_alert callback may be blocking)')
            else:
                self._watchdog_thread = None

    def _on_loop_thread(self) -> bool:
        try:
            return asyncio.get_running_loop() is self._loop
        except RuntimeError:
            return False

    def reset(self):
        """Clear accumulated stats. The probe keeps running.

        Useful for periodic reporting: read `stats`, then `reset()` to start
        a fresh measurement window without restarting the probe.
        """
        with self._lock:
            self._samples = 0
            self._total = 0.0
            self._lag_max = 0.0
            self._lag_min = float('inf')

    @property
    def stats(self) -> dict:
        """Return current {max, avg, min, samples} snapshot."""
        with self._lock:
            if self._samples == 0:
                return {'max': 0.0, 'avg': 0.0, 'min': 0.0, 'samples': 0}
            return {
                'max': self._lag_max,
                'avg': self._total / self._samples,
                'min': self._lag_min,
                'samples': self._samples,
            }

    def _snapshot_and_reset(self) -> dict:
        """Atomically read the current window's stats and clear them.

        Done under one lock acquisition so the read and the reset can't be
        split by a concurrent probe update (which would lose samples).
        """
        with self._lock:
            if self._samples == 0:
                snap = {'max': 0.0, 'avg': 0.0, 'min': 0.0, 'samples': 0, 'sum': 0.0}
            else:
                snap = {
                    'max': self._lag_max,
                    'avg': self._total / self._samples,
                    'min': self._lag_min,
                    'samples': self._samples,
                    'sum': self._total,
                }
            self._samples = 0
            self._total = 0.0
            self._lag_max = 0.0
            self._lag_min = float('inf')
            return snap

    def _probe(self):
        if not self._running:
            return
        loop = self._loop
        if loop is None:
            return
        now = loop.time()
        try:
            lag = now - self._expected
            if lag < 0:  # fired early (clock granularity); not a real delay
                lag = 0.0

            with self._lock:
                self._samples += 1
                self._total += lag
                if lag > self._lag_max:
                    self._lag_max = lag
                if lag < self._lag_min:
                    self._lag_min = lag
            self._last_probe_monotonic = time.monotonic()
            if self._on_sample is not None:
                self._on_sample(lag)
        except Exception:
            logger.warning('EventLoopMonitor probe error', exc_info=True)
        finally:
            # Re-anchor to the ACTUAL wake time, not the old grid. Otherwise,
            # after a long block, _expected lags far behind real time and every
            # subsequent probe keeps reporting the stale delay forever.
            # call_later schedules on loop.time(), so we measure on loop.time()
            # too — single clock source (matters under uvloop).
            # The reschedule lives in finally so an error above can't stop the loop.
            self._expected = now + self._interval
            if self._running:
                self._handle = loop.call_later(self._interval, self._probe)

    def _watchdog_loop(self):
        """Runs in a separate OS thread; alerts on a sliding window.

        Each tick the per-tick stats become one bucket. We keep the last N
        buckets (the sliding window) and aggregate across them. Alerting is
        level-triggered: any tick whose window breaches the threshold fires an
        alert, so a lingering condition keeps alerting every watchdog_interval.
        Downstream alert systems are expected to dedupe/group repeats.
        """
        # Each bucket: (max, sum, count). deque drops the oldest automatically.
        buckets: collections.deque = collections.deque(maxlen=self._window_buckets)

        while self._running:
            # wait() returns True if stop() set the event — exit promptly;
            # returns False on timeout — time for a periodic check.
            if self._wakeup.wait(self._watchdog_interval):
                break
            try:
                # Atomically snapshot this tick's stats and start a fresh bucket.
                snap = self._snapshot_and_reset()
                buckets.append((snap['max'], snap['sum'], snap['samples']))

                # Aggregate the sliding window.
                win_max = 0.0
                win_sum = 0.0
                win_count = 0
                for b_max, b_sum, b_count in buckets:
                    if b_max > win_max:
                        win_max = b_max
                    win_sum += b_sum
                    win_count += b_count
                win_avg = win_sum / win_count if win_count else 0.0
                win_stats = {
                    'max': win_max,
                    'avg': win_avg,
                    'samples': win_count,
                    'window_seconds': len(buckets) * self._watchdog_interval,
                }

                # Stall: probe hasn't run for over a full interval ⇒ loop is
                # blocked/dead. Only the thread can catch this.
                stalled_for = time.monotonic() - self._last_probe_monotonic
                if stalled_for > self._watchdog_interval:
                    self._fire('stall', stalled_for, win_stats)

                # Spike: window max crosses threshold.
                if win_count > 0 and win_max >= self._lag_threshold:
                    self._fire('spike', win_max, win_stats)

                # Sustained: window avg crosses threshold.
                if win_count > 0 and win_avg >= self._lag_threshold:
                    self._fire('sustained', win_avg, win_stats)
            except Exception:
                logger.warning('EventLoopMonitor watchdog error', exc_info=True)

    def _fire(self, reason: str, value: float, stats: dict):
        """Invoke the alert callback, isolating it from raising into the loop."""
        if self._on_alert is None:
            return
        try:
            self._on_alert(reason, value, stats)
        except Exception:
            logger.warning('EventLoopMonitor on_alert error', exc_info=True)

    @staticmethod
    def _default_alert(reason: str, value: float, stats: dict):
        window = 'window[max={:.0f}ms avg={:.1f}ms samples={:d} over {:.1f}s]'.format(
            stats['max'] * 1000,
            stats['avg'] * 1000,
            stats['samples'],
            stats['window_seconds'],
        )
        if reason == 'stall':
            logger.warning(
                'event loop STALLED: probe has not fired for %.0fms (loop likely blocked) %s',
                value * 1000,
                window,
            )
        elif reason == 'spike':
            logger.warning('event loop lag SPIKE %.0fms %s', value * 1000, window)
        else:  # sustained
            logger.warning('event loop SUSTAINED lag avg=%.0fms %s', value * 1000, window)
