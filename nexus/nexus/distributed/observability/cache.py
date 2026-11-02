"""Bounded request coalescing and stale fallbacks for inventory queries."""

from __future__ import annotations

import asyncio
import copy
import logging
import time

from collections import OrderedDict
from collections.abc import Awaitable, Callable
from typing import Any


logger = logging.getLogger(__name__)


class LiveStatusCache:
    def __init__(
        self, *, ttl: float = 1, capacity: int = 128, query_timeout: float = 0.45, max_concurrent: int = 8
    ) -> None:
        self.ttl = max(0, ttl)
        self.capacity = capacity
        self.query_timeout = query_timeout
        self._entries: OrderedDict[tuple, tuple[float, dict[str, Any]]] = OrderedDict()
        self._pending: dict[tuple, asyncio.Task] = {}
        self._slots = asyncio.Semaphore(max_concurrent)

    async def get(self, key: tuple, load: Callable[[], Awaitable[dict[str, Any]]]) -> dict[str, Any]:
        entry = self._entries.get(key)
        if entry is not None and time.monotonic() - entry[0] <= self.ttl:
            self._entries.move_to_end(key)
            return self._response(entry)
        task = self._pending.get(key)
        if task is None:
            if len(self._pending) >= self.capacity:
                raise TimeoutError('live_status_query_capacity')
            task = asyncio.create_task(self._load(key, load))
            self._pending[key] = task
            task.add_done_callback(lambda done: self._finish(key, done))
        try:
            return self._response(await asyncio.shield(task))
        except Exception:
            if entry is None:
                raise
            value = self._response(entry)
            value['data_status'] = 'stale'
            value['error_code'] = 'live_query_failed'
            return value

    async def _load(self, key, load):
        async def run():
            async with self._slots:
                queried_at = time.monotonic()
                value = await load()
                return queried_at, value

        started = time.monotonic()
        try:
            entry = await asyncio.wait_for(run(), timeout=self.query_timeout)
        except TimeoutError:
            logger.warning(
                'Inventory cache refresh timed out: key=%r budget_seconds=%s elapsed_seconds=%.3f',
                key,
                self.query_timeout,
                time.monotonic() - started,
            )
            raise
        except Exception:
            logger.exception(
                'Inventory cache refresh failed: key=%r elapsed_seconds=%.3f', key, time.monotonic() - started
            )
            raise
        self._entries[key] = entry
        self._entries.move_to_end(key)
        while len(self._entries) > self.capacity:
            self._entries.popitem(last=False)
        return entry

    def _finish(self, key, task):
        if self._pending.get(key) is task:
            self._pending.pop(key, None)
        if not task.cancelled():
            task.exception()

    @staticmethod
    def _response(entry):
        value = copy.deepcopy(entry[1])
        value['cache_age_seconds'] = max(0, time.monotonic() - entry[0])
        return value

    async def close(self):
        pending = tuple(self._pending.values())
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        self._entries.clear()
