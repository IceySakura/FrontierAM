from __future__ import annotations

import asyncio
import hashlib
import math
import os
import random
import time

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import select, text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import async_sessionmaker

from nexus.distributed.persistence.db import is_retryable_database_error
from nexus.distributed.persistence.orm import InstanceManagerLivenessRow
from nexus.distributed.repositories.manager import ManagerRepository
from nexus.distributed.utils.clock import utcnow


DATABASE_COUNTERS = {
    'Aborted_connects',
    'Com_commit',
    'Com_rollback',
    'Created_tmp_disk_tables',
    'Innodb_row_lock_time',
    'Innodb_row_lock_waits',
    'Questions',
    'Slow_queries',
}
DATABASE_GAUGES = {'Threads_connected', 'Threads_running'}


@dataclass(frozen=True, slots=True)
class LogicalManagerHeartbeatTarget:
    manager_id: str
    session_id: str
    capacity_total: int


class ManagerHeartbeatLoad:
    """Run production-shaped heartbeat transactions for logical Managers."""

    def __init__(
        self,
        *,
        session_factory: async_sessionmaker,
        targets: list[LogicalManagerHeartbeatTarget],
        interval_seconds: float,
        healthy_ttl: timedelta,
        self_fence_timeout: timedelta,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError('heartbeat interval must be positive')
        if not targets:
            raise ValueError('at least one heartbeat target is required')
        self.session_factory = session_factory
        self.targets = list(targets)
        self.interval_seconds = interval_seconds
        self.healthy_ttl = healthy_ttl
        self.self_fence_timeout = self_fence_timeout.total_seconds()
        self._stop = asyncio.Event()
        self._ready = [asyncio.Event() for _ in self.targets]
        self._tasks: list[asyncio.Task[None]] = []
        self._successes = 0
        self._transient_failures = 0
        self._heartbeat_timeouts = 0
        self._fatal_errors: dict[str, str] = {}
        self._fenced_manager_ids: set[str] = set()
        self._self_fenced_manager_ids: set[str] = set()
        self._latencies: list[float] = []
        self._success_gaps: list[float] = []
        self._last_success: dict[str, float] = {}

    async def start(self) -> None:
        if self._tasks:
            raise RuntimeError('Manager heartbeat load is already started')
        for index, target in enumerate(self.targets):
            self._tasks.append(
                asyncio.create_task(
                    self._run_target(index, target),
                    name=f'capacity-heartbeat:{target.manager_id}',
                )
            )
        await asyncio.gather(*(ready.wait() for ready in self._ready))
        if self._fatal_errors or self._fenced_manager_ids or self._self_fenced_manager_ids:
            report = await self.stop()
            raise RuntimeError(f'Manager heartbeat load failed to start: {report}')

    async def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._tasks:
            pending = [task for task in self._tasks if not task.done()]
            if pending:
                _, pending_after_grace = await asyncio.wait(
                    pending,
                    timeout=min(5.0, max(0.1, self.interval_seconds)),
                )
                for task in pending_after_grace:
                    task.cancel()
            results = await asyncio.gather(*self._tasks, return_exceptions=True)
            for target, result in zip(self.targets, results, strict=True):
                if isinstance(result, BaseException) and not isinstance(result, asyncio.CancelledError):
                    self._fatal_errors.setdefault(target.manager_id, f'{type(result).__name__}: {result}')
        return await self._report()

    async def _run_target(self, index: int, target: LogicalManagerHeartbeatTarget) -> None:
        phase_delay = self.interval_seconds * index / len(self.targets)
        if await self._wait_for_stop(phase_delay):
            self._ready[index].set()
            return
        last_confirmed = time.monotonic()
        failure_count = 0
        jitter = random.Random(f'{target.manager_id}:{target.session_id}:capacity-heartbeat')
        try:
            while not self._stop.is_set():
                started = time.perf_counter()
                deadline = last_confirmed + self.self_fence_timeout
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._self_fenced_manager_ids.add(target.manager_id)
                    return
                try:
                    ok = await asyncio.wait_for(self._heartbeat_once(target), timeout=remaining)
                except asyncio.CancelledError:
                    raise
                except (DBAPIError, TimeoutError) as error:
                    if isinstance(error, DBAPIError) and not is_retryable_database_error(error):
                        self._fatal_errors[target.manager_id] = f'{type(error).__name__}: {error}'
                        return
                    self._transient_failures += 1
                    self._heartbeat_timeouts += int(isinstance(error, TimeoutError))
                    failure_count += 1
                    if time.monotonic() >= deadline:
                        self._self_fenced_manager_ids.add(target.manager_id)
                        return
                    base = min(0.25 * (2 ** min(failure_count - 1, 5)), self.interval_seconds)
                    delay = min(base * jitter.uniform(0.75, 1.25), max(0.0, deadline - time.monotonic()))
                    if await self._wait_for_stop(delay):
                        return
                    continue
                except Exception as error:
                    self._fatal_errors[target.manager_id] = f'{type(error).__name__}: {error}'
                    return
                if not ok:
                    self._fenced_manager_ids.add(target.manager_id)
                    return
                confirmed = time.monotonic()
                previous = self._last_success.get(target.manager_id)
                if previous is not None:
                    self._success_gaps.append(confirmed - previous)
                self._last_success[target.manager_id] = confirmed
                self._latencies.append(time.perf_counter() - started)
                self._successes += 1
                failure_count = 0
                last_confirmed = confirmed
                self._ready[index].set()
                if await self._wait_for_stop(self.interval_seconds):
                    return
        finally:
            self._ready[index].set()

    async def _heartbeat_once(self, target: LogicalManagerHeartbeatTarget) -> bool:
        async with self.session_factory() as session:
            ok = await ManagerRepository(session, healthy_ttl=self.healthy_ttl).heartbeat(
                manager_id=target.manager_id,
                session_id=target.session_id,
                capacity_total=target.capacity_total,
                observed_capacity_used=0,
                now=utcnow(),
            )
            await session.commit()
        return ok

    async def _wait_for_stop(self, delay: float) -> bool:
        if delay <= 0:
            return self._stop.is_set()
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=delay)
        except TimeoutError:
            return False
        return True

    async def _report(self) -> dict[str, Any]:
        manager_ids = [target.manager_id for target in self.targets]
        healthy_since = utcnow() - self.healthy_ttl
        async with self.session_factory() as session:
            rows = (
                await session.execute(
                    select(
                        InstanceManagerLivenessRow.manager_id,
                        InstanceManagerLivenessRow.last_heartbeat_at,
                        InstanceManagerLivenessRow.fenced_at,
                    ).where(InstanceManagerLivenessRow.manager_id.in_(manager_ids))
                )
            ).all()
        database_fenced = {manager_id for manager_id, _, fenced_at in rows if fenced_at is not None}
        stale_managers = (
            len(manager_ids) - len(rows) + sum(last_heartbeat_at < healthy_since for _, last_heartbeat_at, _ in rows)
        )
        fenced_managers = self._fenced_manager_ids | database_fenced
        return {
            'manager_count': len(self.targets),
            'interval_seconds': self.interval_seconds,
            'successes': self._successes,
            'transient_failures': self._transient_failures,
            'heartbeat_timeouts': self._heartbeat_timeouts,
            'fatal_failures': len(self._fatal_errors),
            'fatal_error_types': sorted({error.partition(':')[0] for error in self._fatal_errors.values()}),
            'fenced_managers': len(fenced_managers),
            'self_fenced_managers': len(self._self_fenced_manager_ids),
            'stale_managers': stale_managers,
            'latency_samples': len(self._latencies),
            'latency_p50_seconds': percentile(self._latencies, 0.50),
            'latency_p95_seconds': percentile(self._latencies, 0.95),
            'latency_p99_seconds': percentile(self._latencies, 0.99),
            'latency_max_seconds': round(max(self._latencies), 6) if self._latencies else None,
            'success_gap_max_seconds': round(max(self._success_gaps), 6) if self._success_gaps else None,
        }


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * quantile) - 1)
    return round(ordered[index], 6)


async def retry_tick(tick: Callable[[], Awaitable[int]]) -> int:
    for attempt in range(10):
        try:
            return await tick()
        except DBAPIError as error:
            if not is_retryable_database_error(error) or attempt == 9:
                raise
            await asyncio.sleep(min(0.01 * (2**attempt), 0.5))
    raise AssertionError('unreachable')


class ProcessSampler:
    def __init__(self, pids: list[int]) -> None:
        self.pids = list(pids)
        self.clock_ticks = int(os.sysconf('SC_CLK_TCK'))
        self.page_size = int(os.sysconf('SC_PAGE_SIZE'))
        self._previous_at: float | None = None
        self._previous_ticks: dict[int, int] = {}
        self._cpu_seconds = 0.0
        self._wall_seconds = 0.0
        self.cpu_peak_percent = 0.0
        self.rss_peak_bytes = 0
        self.samples = 0

    def replace_pid(self, old_pid: int, new_pid: int) -> None:
        try:
            index = self.pids.index(old_pid)
        except ValueError as exc:
            raise ValueError(f'process sampler does not track pid {old_pid}') from exc
        self.pids[index] = new_pid

    def sample(self) -> None:
        now = time.monotonic()
        current_ticks: dict[int, int] = {}
        total_rss = 0
        for pid in self.pids:
            try:
                stat = Path(f'/proc/{pid}/stat').read_text()
                _, _, fields = stat.partition(') ')
                values = fields.split()
                current_ticks[pid] = int(values[11]) + int(values[12])
                statm = Path(f'/proc/{pid}/statm').read_text().split()
                total_rss += int(statm[1]) * self.page_size
            except (FileNotFoundError, IndexError, ValueError):
                continue
        if not current_ticks:
            return
        if self._previous_at is not None and now > self._previous_at:
            elapsed = now - self._previous_at
            tick_delta = sum(
                max(0, ticks - self._previous_ticks[pid])
                for pid, ticks in current_ticks.items()
                if pid in self._previous_ticks
            )
            cpu_seconds = tick_delta / self.clock_ticks
            cpu_percent = cpu_seconds / elapsed * 100
            self._cpu_seconds += cpu_seconds
            self._wall_seconds += elapsed
            self.cpu_peak_percent = max(self.cpu_peak_percent, cpu_percent)
        self._previous_at = now
        self._previous_ticks = current_ticks
        self.rss_peak_bytes = max(self.rss_peak_bytes, total_rss)
        self.samples += 1

    def report(self) -> dict[str, Any]:
        average = self._cpu_seconds / self._wall_seconds * 100 if self._wall_seconds else 0.0
        return {
            'processes': len(self.pids),
            'samples': self.samples,
            'cpu_average_percent': round(average, 2),
            'cpu_peak_percent': round(self.cpu_peak_percent, 2),
            'rss_peak_bytes': self.rss_peak_bytes,
        }


async def database_status(session_factory: async_sessionmaker) -> dict[str, int]:
    names = sorted(DATABASE_COUNTERS | DATABASE_GAUGES)
    query = 'SHOW GLOBAL STATUS WHERE Variable_name IN (' + ','.join(f"'{name}'" for name in names) + ')'
    async with session_factory() as session:
        rows = (await session.execute(text(query))).all()
    result: dict[str, int] = {}
    for name, value in rows:
        try:
            result[str(name)] = int(value)
        except (TypeError, ValueError):
            continue
    return result


async def database_identity(session_factory: async_sessionmaker) -> dict[str, str]:
    async with session_factory() as session:
        version, database = (await session.execute(text('SELECT VERSION(), DATABASE()'))).one()
    return {
        'dialect': 'mysql',
        'version': str(version),
        'database': str(database),
        'status_scope': 'mysql_global_dedicated_server_required',
    }


def source_identity(root: Path, relative_paths: list[str] | tuple[str, ...]) -> dict[str, Any]:
    files = sorted(set(relative_paths))
    digest = hashlib.sha256()
    for relative_path in files:
        digest.update(relative_path.encode())
        digest.update(b'\0')
        digest.update((root / relative_path).read_bytes())
        digest.update(b'\0')
    return {'sha256': digest.hexdigest(), 'files': files}


def database_status_report(
    *,
    before: dict[str, int],
    after: dict[str, int],
    gauge_peaks: dict[str, int],
) -> dict[str, Any]:
    return {
        'counter_delta': {name: after.get(name, 0) - before.get(name, 0) for name in sorted(DATABASE_COUNTERS)},
        'gauges': {
            name: {
                'before': before.get(name),
                'after': after.get(name),
                'peak': gauge_peaks.get(name),
            }
            for name in sorted(DATABASE_GAUGES)
        },
    }


def join_processes(processes: list[Any], *, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    for process in processes:
        process.join(max(0, deadline - time.monotonic()))
    alive = [process for process in processes if process.is_alive()]
    if not alive:
        return
    for process in alive:
        process.terminate()
    for process in alive:
        process.join(5)
    stubborn = [process for process in alive if process.is_alive()]
    for process in stubborn:
        process.kill()
    for process in stubborn:
        process.join(5)
    still_alive = [process for process in stubborn if process.is_alive()]
    if still_alive:
        raise TimeoutError(f'Capacity worker processes did not stop: {[process.pid for process in still_alive]}')


__all__ = [
    'DATABASE_COUNTERS',
    'DATABASE_GAUGES',
    'LogicalManagerHeartbeatTarget',
    'ManagerHeartbeatLoad',
    'ProcessSampler',
    'database_status',
    'database_identity',
    'database_status_report',
    'join_processes',
    'percentile',
    'retry_tick',
    'source_identity',
]
