import asyncio
import time

import pytest

from nexus.utils.eventloop_monitor import EventLoopMonitor


@pytest.mark.asyncio
async def test_idle_loop_minimal_lag():
    """On an idle loop, lag should be near zero."""
    monitor = EventLoopMonitor()
    monitor.start(interval=0.005)
    await asyncio.sleep(0.1)
    monitor.stop()
    s = monitor.stats
    assert s['samples'] > 0
    assert s['max'] < 0.02  # well under 20ms on an idle loop


@pytest.mark.asyncio
async def test_detects_blocking():
    """Blocking the loop should produce measurable lag."""
    monitor = EventLoopMonitor()
    monitor.start(interval=0.005)
    await asyncio.sleep(0.02)
    time.sleep(0.1)  # block the event loop
    await asyncio.sleep(0.05)
    monitor.stop()
    s = monitor.stats
    assert s['max'] >= 0.05  # at least 50ms lag detected


@pytest.mark.asyncio
async def test_stop_clean():
    monitor = EventLoopMonitor()
    monitor.start()
    await asyncio.sleep(0.01)
    monitor.stop()


@pytest.mark.asyncio
async def test_double_start_is_safe():
    monitor = EventLoopMonitor()
    monitor.start(interval=0.005, watchdog_interval=0.1)
    thread1 = monitor._watchdog_thread
    monitor.start()  # should no-op, not spawn a second watchdog
    assert monitor._watchdog_thread is thread1
    await asyncio.sleep(0.01)
    monitor.stop()


@pytest.mark.asyncio
async def test_stats_before_start():
    monitor = EventLoopMonitor()
    s = monitor.stats
    assert s['samples'] == 0
    assert s['max'] == 0.0


@pytest.mark.asyncio
async def test_lag_never_negative():
    monitor = EventLoopMonitor()
    monitor.start(interval=0.005)
    await asyncio.sleep(0.1)
    monitor.stop()
    s = monitor.stats
    assert s['min'] >= 0.0


def test_start_without_running_loop_stays_clean():
    """start() outside a loop raises but leaves the object unstarted."""
    monitor = EventLoopMonitor()
    with pytest.raises(RuntimeError):
        monitor.start()
    assert monitor._running is False
    assert monitor.stats['samples'] == 0


@pytest.mark.asyncio
async def test_low_cpu_when_idle():
    """The probe must not busy-spin: sample count over 0.2s at 10ms interval
    should be roughly ~20, not tens of thousands."""
    monitor = EventLoopMonitor()
    monitor.start(interval=0.01)
    await asyncio.sleep(0.2)
    monitor.stop()
    s = monitor.stats
    # ~20 expected; allow generous upper bound but catch busy-spin (would be 100k+)
    assert s['samples'] < 200


@pytest.mark.asyncio
async def test_recovers_after_blocking():
    """After a block clears, lag must return to near zero (no stale drift).

    Regression: previously _expected += interval let the grid fall behind real
    time after a block, so every later probe kept reporting the stale delay.
    """
    monitor = EventLoopMonitor()
    monitor.start(interval=0.005)
    await asyncio.sleep(0.02)
    time.sleep(0.3)  # one big block
    await asyncio.sleep(0.02)
    monitor.reset()  # discard the block window
    await asyncio.sleep(0.1)  # measure recovery only
    monitor.stop()
    s = monitor.stats
    assert s['samples'] > 0
    assert s['max'] < 0.02, f'recovery lag should be ~0, got {s["max"] * 1000:.1f}ms'


@pytest.mark.asyncio
async def test_watchdog_alerts_on_lag():
    """Watchdog should fire a 'spike' alert when lag exceeds the threshold."""
    alerts = []

    def on_alert(reason, value, stats):
        alerts.append((reason, value))

    monitor = EventLoopMonitor()
    monitor.start(
        interval=0.005,
        watchdog_interval=0.1,
        lag_threshold=0.05,
        on_alert=on_alert,
    )
    await asyncio.sleep(0.02)
    time.sleep(0.15)  # block beyond threshold
    await asyncio.sleep(0.35)  # let watchdog observe + check
    monitor.stop()

    reasons = [r for r, _ in alerts]
    assert 'spike' in reasons or 'stall' in reasons, f'expected an alert, got {alerts}'


@pytest.mark.asyncio
async def test_watchdog_spike_repeats_then_clears():
    """Level-triggered: a spike re-alerts each tick while it sits in the window,
    then stops once it ages out.

    With a 0.3s window (3 buckets at 0.1s), one spike should alert roughly 3
    times (while its bucket lingers), then go silent — not forever.
    """
    spikes = []
    monitor = EventLoopMonitor()
    monitor.start(
        interval=0.005,
        watchdog_interval=0.1,
        lag_threshold=0.05,
        window_seconds=0.3,  # 3 buckets
        on_alert=lambda r, v, s: spikes.append(r) if r == 'spike' else None,
    )
    await asyncio.sleep(0.02)
    time.sleep(0.15)  # one spike
    await asyncio.sleep(0.6)  # several ticks: spike lingers then ages out
    monitor.stop()

    # Repeats while in window (more than once), but bounded by window size — it
    # must clear after the bucket ages out, not alert on every one of ~6 ticks.
    n = spikes.count('spike')
    assert 1 <= n <= 4, f'expected a few repeats then clear, got {n}: {spikes}'


@pytest.mark.asyncio
async def test_watchdog_silent_when_healthy():
    """Watchdog should not alert on an idle, healthy loop."""
    alerts = []

    monitor = EventLoopMonitor()
    monitor.start(
        interval=0.005,
        watchdog_interval=0.1,
        lag_threshold=0.05,
        on_alert=lambda r, v, s: alerts.append((r, v)),
    )
    await asyncio.sleep(0.35)  # several watchdog cycles, no blocking
    monitor.stop()

    assert alerts == [], f'healthy loop should not alert, got {alerts}'


@pytest.mark.asyncio
async def test_watchdog_default_alert_logs(caplog):
    """Without a custom callback, watchdog logs a warning."""
    import logging

    monitor = EventLoopMonitor()
    monitor.start(interval=0.005, watchdog_interval=0.1, lag_threshold=0.05)
    await asyncio.sleep(0.02)
    with caplog.at_level(logging.WARNING):
        time.sleep(0.15)
        await asyncio.sleep(0.35)
    monitor.stop()
    assert any('event loop' in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_stop_joins_watchdog_thread():
    """After stop(), the watchdog thread should be gone."""
    monitor = EventLoopMonitor()
    monitor.start(interval=0.005, watchdog_interval=0.1)
    await asyncio.sleep(0.05)
    monitor.stop()
    assert monitor._watchdog_thread is None


@pytest.mark.asyncio
async def test_stop_returns_promptly_with_long_watchdog():
    """stop() must not block for a full watchdog interval.

    Uses a long watchdog_interval; stop() should return well before it elapses
    because the wakeup event interrupts the wait immediately.
    """
    monitor = EventLoopMonitor()
    monitor.start(interval=0.005, watchdog_interval=5.0)
    await asyncio.sleep(0.05)

    t0 = time.monotonic()
    monitor.stop()
    elapsed = time.monotonic() - t0

    assert elapsed < 1.0, f'stop() blocked for {elapsed:.2f}s (should be ~0)'
    assert monitor._watchdog_thread is None


@pytest.mark.asyncio
async def test_watchdog_sustained_alert():
    """When nearly every probe sample is delayed, the window AVG crosses the
    threshold and a 'sustained' alert fires.

    Pattern: yield once (so the probe can be scheduled) then block — so almost
    every sample lands in a blocked window, keeping the average high rather
    than being diluted by idle samples.
    """
    sustained = []
    monitor = EventLoopMonitor()
    monitor.start(
        interval=0.005,
        watchdog_interval=0.1,
        lag_threshold=0.03,
        on_alert=lambda r, v, s: sustained.append(v) if r == 'sustained' else None,
    )
    for _ in range(60):
        await asyncio.sleep(0)  # let the probe run once, then block again
        time.sleep(0.04)  # 40ms block > 30ms threshold
    await asyncio.sleep(0.4)
    monitor.stop()
    assert len(sustained) >= 1, 'sustained avg breach should alert'


@pytest.mark.asyncio
async def test_stall_alert_fires_while_blocked():
    """While the loop is blocked, the watchdog thread (a separate OS thread)
    keeps detecting the stall and alerts on each tick (level-triggered)."""
    stalls = []
    monitor = EventLoopMonitor()
    monitor.start(
        interval=0.005,
        watchdog_interval=0.05,
        lag_threshold=0.5,  # high, so only stall (not spike) fires
        on_alert=lambda r, v, s: stalls.append(r) if r == 'stall' else None,
    )
    await asyncio.sleep(0.02)
    time.sleep(0.4)  # block across many watchdog ticks
    await asyncio.sleep(0.2)
    monitor.stop()
    # Level-triggered: a long block across ~8 ticks (0.4s / 0.05s) should fire
    # multiple stall alerts, not just one.
    assert stalls.count('stall') >= 2, f'expected repeated stalls, got {stalls}'


@pytest.mark.asyncio
async def test_restart_does_not_inherit_old_peak():
    """After stop()+start(), a prior lag peak must not trigger a spurious spike."""
    monitor = EventLoopMonitor()
    monitor.start(interval=0.005)
    await asyncio.sleep(0.02)
    time.sleep(0.2)  # big lag recorded in the first run
    await asyncio.sleep(0.02)
    monitor.stop()
    old_peak = monitor.stats['max']
    assert old_peak >= 0.05, f'first run should have recorded the block, got {old_peak}'

    # Restart with a watchdog; the old 200ms peak must not leak in.
    alerts = []
    monitor.start(
        interval=0.005,
        watchdog_interval=0.1,
        lag_threshold=0.05,
        on_alert=lambda r, v, s: alerts.append((r, v)),
    )
    # The restart itself must drop the carried-over stats. Checked directly because it
    # is the regression: were the peak kept, the first watchdog tick would replay it.
    assert monitor.stats == {'max': 0.0, 'avg': 0.0, 'min': 0.0, 'samples': 0}
    await asyncio.sleep(0.35)
    monitor.stop()
    # A loaded machine can block the loop for real, so a fresh alert is not a failure.
    # An inherited peak is, and it is recognisable: it replays the previous run's value.
    assert all(value != old_peak for _, value in alerts), f'restart inherited old peak: {alerts}'


@pytest.mark.asyncio
async def test_on_alert_exception_does_not_crash_watchdog():
    """If on_alert raises, the watchdog must keep running and keep alerting on
    later ticks rather than dying."""
    calls = []

    def flaky(reason, value, stats):
        calls.append(reason)
        raise RuntimeError('boom')

    monitor = EventLoopMonitor()
    monitor.start(
        interval=0.005,
        watchdog_interval=0.1,
        lag_threshold=0.05,
        window_seconds=10.0,  # wide window: spike stays breached for a while
        on_alert=flaky,
    )
    await asyncio.sleep(0.02)
    time.sleep(0.15)  # one spike
    await asyncio.sleep(0.45)  # several ticks; spike stays in the wide window
    monitor.stop()
    # The callback raises every time, yet the watchdog survives and keeps
    # firing (level-triggered): several spike attempts, not zero (crash) and
    # not stuck.
    assert calls.count('spike') >= 2, f'watchdog should survive errors, got {calls}'


@pytest.mark.asyncio
async def test_stop_from_other_thread_is_safe():
    """stop() called off the event loop thread must not touch loop state directly."""
    import threading

    monitor = EventLoopMonitor()
    monitor.start(interval=0.005, watchdog_interval=0.1)
    await asyncio.sleep(0.05)

    done = threading.Event()
    error = []

    def stopper():
        try:
            monitor.stop()
        except Exception as e:  # noqa: BLE001
            error.append(e)
        finally:
            done.set()

    t = threading.Thread(target=stopper)
    t.start()
    # Keep the loop alive so call_soon_threadsafe can be serviced.
    for _ in range(20):
        if done.is_set():
            break
        await asyncio.sleep(0.01)
    t.join(timeout=1.0)

    assert not error, f'stop() from another thread raised: {error}'
    assert monitor._running is False
