from __future__ import annotations

import asyncio

from collections.abc import Awaitable, Callable
from contextlib import suppress


class LeaseLostError(RuntimeError):
    """The operation no longer owns its fenced lease."""


class PeriodicLease:
    def __init__(
        self,
        *,
        renew: Callable[[], Awaitable[bool]],
        interval: float,
        lost_code: str,
        error_type: type[LeaseLostError] = LeaseLostError,
    ) -> None:
        if interval <= 0:
            raise ValueError('lease renewal interval must be positive')
        self._renew = renew
        self.interval = interval
        self.lost_code = lost_code
        self.error_type = error_type
        self._lock = asyncio.Lock()
        self._task: asyncio.Task[None] | None = None
        self._error: BaseException | None = None

    async def __aenter__(self) -> PeriodicLease:
        await self.renew_now()
        self._task = asyncio.create_task(self._run(), name=f'fenced-lease:{self.lost_code}')
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        await self.close()

    async def renew_now(self) -> None:
        await self.ensure_owned()
        async with self._lock:
            await self.ensure_owned()
            try:
                renewed = await self._renew()
            except BaseException as exc:
                self._error = exc
                raise
            if not renewed:
                self._error = self.error_type(self.lost_code)
                raise self._error

    async def ensure_owned(self) -> None:
        if self._error is None:
            return
        if isinstance(self._error, LeaseLostError):
            raise self._error
        raise self.error_type(self.lost_code) from self._error

    async def close(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        with suppress(asyncio.CancelledError):
            await self._task
        self._task = None

    async def _run(self) -> None:
        while True:
            await asyncio.sleep(self.interval)
            try:
                await self.renew_now()
            except BaseException as exc:
                if isinstance(exc, asyncio.CancelledError):
                    raise
                if self._error is None:
                    self._error = exc
                return


__all__ = ['LeaseLostError', 'PeriodicLease']
