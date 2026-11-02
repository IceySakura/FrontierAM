from __future__ import annotations

import logging

from collections.abc import Callable

from nexus.distributed.instance_manager.self_fence import HardExitGuard


class FakeTimer:
    def __init__(self, delay: float, callback: Callable[[], None]) -> None:
        self.delay = delay
        self.callback = callback
        self.daemon = False
        self.started = False
        self.cancelled = False

    def start(self) -> None:
        self.started = True

    def cancel(self) -> None:
        self.cancelled = True


def test_hard_exit_guard_arms_one_daemon_timer_and_exits_after_deadline() -> None:
    timers: list[FakeTimer] = []
    exits: list[int] = []

    def timer_factory(delay: float, callback: Callable[[], None]) -> FakeTimer:
        timer = FakeTimer(delay, callback)
        timers.append(timer)
        return timer

    guard = HardExitGuard(
        timeout=5.0,
        exit_func=exits.append,
        timer_factory=timer_factory,
    )

    guard.arm('executor_loop_hang')
    guard.arm('duplicate_reason')

    assert len(timers) == 1
    assert timers[0].delay == 5.0
    assert timers[0].daemon is True
    assert timers[0].started is True
    timers[0].callback()
    assert exits == [70]


def test_hard_exit_guard_disarm_cancels_pending_exit() -> None:
    timers: list[FakeTimer] = []

    def timer_factory(delay: float, callback: Callable[[], None]) -> FakeTimer:
        timer = FakeTimer(delay, callback)
        timers.append(timer)
        return timer

    guard = HardExitGuard(timeout=5.0, exit_func=lambda _: None, timer_factory=timer_factory)
    guard.arm('heartbeat_timeout')

    guard.disarm()

    assert timers[0].cancelled is True


def test_hard_exit_guard_does_not_log_before_process_exit(monkeypatch) -> None:
    timers: list[FakeTimer] = []
    exits: list[int] = []

    def timer_factory(delay: float, callback: Callable[[], None]) -> FakeTimer:
        timer = FakeTimer(delay, callback)
        timers.append(timer)
        return timer

    monkeypatch.setattr(
        logging.getLogger('nexus.distributed.instance_manager.self_fence'),
        'critical',
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('logging is blocked')),
    )
    guard = HardExitGuard(timeout=5.0, exit_func=exits.append, timer_factory=timer_factory)
    guard.arm('manager_loop_stall')

    timers[0].callback()

    assert exits == [70]
