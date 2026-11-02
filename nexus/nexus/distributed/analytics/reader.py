"""Separate statistics pool, bounded cache, and cancellation-aware query sharing."""

from __future__ import annotations

import asyncio
import copy
import json
import time
import uuid

from collections import OrderedDict
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from nexus.distributed.analytics.query import StatisticsQuery, load_statistics
from nexus.distributed.config import DistributedConfig
from nexus.distributed.persistence.db import create_engine, create_session_factory


class StatisticsBusyError(RuntimeError):
    pass


@dataclass
class Pending:
    task: asyncio.Task
    waiters: int = 0


class BusinessStatisticsReader:
    def __init__(self, config: DistributedConfig, *, session_factory):
        self.config = config.business_statistics
        self.engine = None
        self.factory = session_factory
        database_url = config.read_database_url()
        # In-memory SQLite tests must share the original database. Deployed databases get their own pool.
        if self.config.enabled and database_url and ':memory:' not in database_url:
            settings = config.model_copy(update={'db_pool_size': 1, 'db_max_overflow': 0, 'db_pool_timeout': 1})
            self.engine = create_engine(settings)
            if self.engine.dialect.name == 'mysql':
                self.engine = self.engine.execution_options(isolation_level='REPEATABLE READ')
            self.factory = create_session_factory(self.engine)
        self._slots = asyncio.Semaphore(1)
        self._pending: dict[str, Pending] = {}
        self._cache: OrderedDict[str, tuple[float, dict]] = OrderedDict()

    async def get(self, query: StatisticsQuery, *, principal_id: str) -> dict:
        key = json.dumps([1, principal_id, query.filters()], sort_keys=True)
        entry = self._cache.get(key)
        if entry is not None and time.monotonic() - entry[0] < self.config.cache_seconds:
            self._cache.move_to_end(key)
            return self._response(entry, cached=True)
        pending = self._pending.get(key)
        if pending is None:
            if len(self._pending) >= 8:
                raise StatisticsBusyError('statistics_busy')
            task = asyncio.create_task(self._load(key, query))
            pending = Pending(task)
            self._pending[key] = pending
            task.add_done_callback(lambda done: self._finish(key, done))
        pending.waiters += 1
        try:
            return self._response(await asyncio.shield(pending.task), cached=False)
        finally:
            pending.waiters -= 1
            if not pending.waiters and not pending.task.done():
                if self._pending.get(key) is pending:
                    self._pending.pop(key, None)
                pending.task.cancel()
                await asyncio.gather(pending.task, return_exceptions=True)

    async def _load(self, key, query):
        try:
            await asyncio.wait_for(self._slots.acquire(), self.config.queue_timeout_seconds)
        except TimeoutError as exc:
            raise StatisticsBusyError('statistics_busy') from exc
        try:
            started = time.monotonic()
            async with self.factory() as session:
                marker = uuid.uuid4().hex
                connection_id = None
                interrupted = False

                async def read():
                    nonlocal connection_id
                    if session.get_bind().dialect.name == 'sqlite':
                        await session.execute(text('BEGIN'))
                    elif session.get_bind().dialect.name == 'mysql':
                        connection = await session.connection()
                        raw = await connection.get_raw_connection()
                        connection_id = raw.driver_connection.thread_id()
                        session.info['business_statistics_query_marker'] = marker
                    # MySQL uses a dedicated REPEATABLE READ pool: all aggregates share the first read snapshot.
                    return await load_statistics(session, query, self.config)

                try:
                    value = await asyncio.wait_for(read(), self.config.request_timeout_seconds)
                except (TimeoutError, asyncio.CancelledError):
                    interrupted = True
                    raise
                finally:
                    await session.rollback()
                    if interrupted and connection_id is not None:
                        await self._stop_query(connection_id, marker)
            entry = (started, value)
            if value['data_status'] == 'ready':
                self._cache[key] = entry
                self._cache.move_to_end(key)
                while len(self._cache) > self.config.cache_entries:
                    self._cache.popitem(last=False)
            return entry
        finally:
            self._slots.release()

    async def _stop_query(self, connection_id: int, marker: str):
        # Closing a MySQL socket does not interrupt CPU work or SLEEP. The old connection has
        # been returned/invalidated above; reuse this bounded pool to stop only our marked SELECT.
        async def stop():
            async with self.factory() as session:
                info = await session.scalar(
                    text('SELECT INFO FROM information_schema.PROCESSLIST WHERE ID=:id AND DB=DATABASE()'),
                    {'id': connection_id},
                )
                if (
                    info
                    and str(info).lstrip().upper().startswith('SELECT ')
                    and f'/* nexus_statistics:{marker} */' in info
                ):
                    try:
                        await session.execute(text(f'KILL QUERY {int(connection_id)}'))
                    except OperationalError as exc:
                        # The disconnected query may finish between PROCESSLIST and KILL.
                        if exc.orig is None or not exc.orig.args or exc.orig.args[0] != 1094:
                            raise

        try:
            await asyncio.wait_for(stop(), self.config.queue_timeout_seconds + 1)
        except Exception:
            # Do not admit a new query before the old SELECT's server-side execution cap.
            await asyncio.sleep(self.config.query_timeout_seconds)
            raise

    def _finish(self, key, task):
        pending = self._pending.get(key)
        if pending is not None and pending.task is task:
            self._pending.pop(key, None)
        if not task.cancelled():
            task.exception()

    @staticmethod
    def _response(entry, *, cached):
        value = copy.deepcopy(entry[1])
        value.update(cache_age_seconds=max(0, time.monotonic() - entry[0]), cached=cached)
        return value

    async def close(self):
        tasks = [item.task for item in self._pending.values()]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._cache.clear()
        if self.engine is not None:
            await self.engine.dispose()
