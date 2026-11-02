from __future__ import annotations

import os
import threading

from collections.abc import Callable
from typing import Protocol


_HARD_EXIT_CODE = 70


class _Timer(Protocol):
    daemon: bool

    def start(self) -> None: ...

    def cancel(self) -> None: ...


class HardExitGuard:
    def __init__(
        self,
        *,
        timeout: float,
        exit_func: Callable[[int], object] = os._exit,
        timer_factory: Callable[[float, Callable[[], None]], _Timer] = threading.Timer,
    ) -> None:
        if timeout <= 0:
            raise ValueError('hard exit timeout must be positive')
        self._timeout = timeout
        self._exit_func = exit_func
        self._timer_factory = timer_factory
        self._lock = threading.Lock()
        self._timer: _Timer | None = None
        self._reason: str | None = None

    def arm(self, reason: str) -> None:
        with self._lock:
            if self._timer is not None:
                return
            self._reason = reason
            timer = self._timer_factory(self._timeout, self._hard_exit)
            timer.daemon = True
            self._timer = timer
            timer.start()

    def disarm(self) -> None:
        with self._lock:
            timer, self._timer = self._timer, None
            self._reason = None
        if timer is not None:
            timer.cancel()

    def _hard_exit(self) -> None:
        with self._lock:
            if self._timer is None:
                return
            self._timer = None
            self._reason = None
        self._exit_func(_HARD_EXIT_CODE)


__all__ = ['HardExitGuard']
