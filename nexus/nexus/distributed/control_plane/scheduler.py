from __future__ import annotations

import asyncio
import contextlib
import logging
import time

from datetime import datetime

import aiohttp

from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from nexus.distributed.config.views import SchedulerConfigView
from nexus.distributed.control_plane.leases import ShardLeaseCoordinator
from nexus.distributed.observability import (
    ControlLoopOutcome,
    ControlLoopSample,
    ControlPlaneMetrics,
    OpenTelemetryControlPlaneMetrics,
)
from nexus.distributed.persistence.db import is_database_transaction_conflict
from nexus.distributed.policies.executor import PolicyExecutor
from nexus.distributed.repositories.policy import PolicyRepository
from nexus.distributed.repositories.scheduler import SchedulerClaimStats, SchedulerRepository
from nexus.distributed.repositories.shard_lease import ShardLeaseRepository
from nexus.distributed.utils.clock import utcnow


logger = logging.getLogger(__name__)


class _SchedulerShardLeaseRepository(ShardLeaseRepository):
    def __init__(
        self,
        session: AsyncSession,
        *,
        scheduler_repository: SchedulerRepository,
        shard_count: int,
    ) -> None:
        super().__init__(session, shard_count=shard_count)
        self.scheduler_repository = scheduler_repository

    async def renew(self, **kwargs) -> list[int]:
        return await self.scheduler_repository.renew_shard_leases(**kwargs)


class SchedulerService:
    def __init__(
        self,
        *,
        session_factory: async_sessionmaker,
        config: SchedulerConfigView,
        owner_id: str,
        shard_group: str = 'execution-scheduler',
        metrics: ControlPlaneMetrics | None = None,
    ) -> None:
        self.session_factory = session_factory
        self.config = config
        self.owner_id = owner_id
        self.shard_group = shard_group
        self.metrics = metrics if metrics is not None else OpenTelemetryControlPlaneMetrics()
        self._stopping = asyncio.Event()
        self._shard_leases = ShardLeaseCoordinator(
            session_factory=session_factory,
            shard_count=config.shard_count,
            owner_id=owner_id,
            shard_group=shard_group,
            lease_ttl=config.scheduler_lease_ttl,
            renew_interval=config.scheduler_lease_renew_interval,
            ownership_limit=config.scheduler_shard_ownership_limit,
            discover_interval=config.scheduler_shard_discover_interval,
            discover_jitter=config.scheduler_shard_discover_jitter,
            repository_factory=self._lease_repository,
        )
        self._http_client: aiohttp.ClientSession | None = None
        self._wakeup_tasks: dict[str, asyncio.Task] = {}
        self._policy_executor = (
            PolicyExecutor(
                session_factory=session_factory,
                config=config.policy_executor,
                shard_count=config.shard_count,
                owner_id=owner_id,
            )
            if config.policy_executor.bindings
            else None
        )
        self._last_policy_demand: dict[str, float] = {}
        self._last_assignment_stats = SchedulerClaimStats()

    async def run_forever(self) -> None:
        if self._policy_executor is not None:
            await self._policy_executor.initialize()
        assignment_task = asyncio.create_task(
            self._run_assignment_loop(), name=f'dist-scheduler-assign:{self.owner_id}'
        )
        lease_task = asyncio.create_task(self._run_lease_heartbeat(), name=f'dist-scheduler-lease:{self.owner_id}')
        tasks = {assignment_task, lease_task}
        if self._policy_executor is not None:
            tasks.add(asyncio.create_task(self._policy_executor.run_forever(), name=f'policy-executor:{self.owner_id}'))
        try:
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
            for task in done:
                if error := task.exception():
                    raise error
            await asyncio.gather(*pending)
        finally:
            self._stopping.set()
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await self.close()

    async def _run_assignment_loop(self) -> None:
        next_due = time.monotonic()
        while not self._stopping.is_set():
            started = time.monotonic()
            lag = max(0.0, started - next_due)
            changed = 0
            outcome: ControlLoopOutcome = 'success'
            backoff = 0.0
            self._last_assignment_stats = SchedulerClaimStats()
            try:
                changed = await self.tick_once()
            except DBAPIError as exc:
                if not is_database_transaction_conflict(exc):
                    self._record_loop(
                        loop='assignment',
                        outcome='fatal_error',
                        started=started,
                        lag=lag,
                    )
                    raise
                outcome = 'transient_error'
                backoff = self.config.scheduler_poll_interval
                logger.warning(
                    'Scheduler loop retry: loop=assignment owner=%s shard_group=%s retry_reason=db_lock',
                    self.owner_id,
                    self.shard_group,
                    exc_info=True,
                )
            except Exception:
                self._record_loop(
                    loop='assignment',
                    outcome='fatal_error',
                    started=started,
                    lag=lag,
                )
                raise
            self._record_loop(
                loop='assignment',
                outcome=outcome,
                started=started,
                lag=lag,
                changed=changed,
                backoff=backoff,
                stats=self._last_assignment_stats,
            )
            if outcome == 'success':
                self.metrics.set_scheduler_owned_shards(self._shard_leases.owned_shard_count)
                self.metrics.record_scheduler_assignments(changed)
                self.metrics.record_scheduler_lock_contention(self._last_assignment_stats.lock_contention)
                self.metrics.record_scheduler_no_capacity(self._last_assignment_stats.no_capacity)
                self.metrics.record_scheduler_candidate_queries(self._last_assignment_stats.candidate_queries)
            next_due = time.monotonic() + self.config.scheduler_poll_interval
            await self._wait_for_stop(self.config.scheduler_poll_interval)

    async def _run_lease_heartbeat(self) -> None:
        interval = max(0.01, self.config.scheduler_lease_renew_interval / 2)
        next_due = time.monotonic()
        while not self._stopping.is_set():
            started = time.monotonic()
            lag = max(0.0, started - next_due)
            outcome: ControlLoopOutcome = 'success'
            backoff = 0.0
            try:
                await self._owned_shards_for_tick(utcnow())
            except DBAPIError as exc:
                if not is_database_transaction_conflict(exc):
                    self._record_loop(
                        loop='lease',
                        outcome='fatal_error',
                        started=started,
                        lag=lag,
                    )
                    raise
                outcome = 'transient_error'
                backoff = interval
                logger.warning(
                    'Scheduler loop retry: loop=lease owner=%s shard_group=%s retry_reason=db_lock',
                    self.owner_id,
                    self.shard_group,
                    exc_info=True,
                )
            except Exception:
                self._record_loop(
                    loop='lease',
                    outcome='fatal_error',
                    started=started,
                    lag=lag,
                )
                raise
            self._record_loop(
                loop='lease',
                outcome=outcome,
                started=started,
                lag=lag,
                backoff=backoff,
            )
            next_due = time.monotonic() + interval
            await self._wait_for_stop(interval)

    async def _wait_for_stop(self, timeout: float) -> None:
        if timeout <= 0:
            await asyncio.sleep(0)
            return
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(self._stopping.wait(), timeout=timeout)

    def stop(self) -> None:
        self._stopping.set()
        if self._policy_executor is not None:
            self._policy_executor.stopping.set()

    async def close(self) -> None:
        if self._policy_executor is not None:
            await self._policy_executor.close()
        self.metrics.set_scheduler_owned_shards(0)
        with contextlib.suppress(Exception):
            await self._shard_leases.release()
        for task in list(self._wakeup_tasks.values()):
            task.cancel()
        if self._wakeup_tasks:
            await asyncio.gather(*self._wakeup_tasks.values(), return_exceptions=True)
            self._wakeup_tasks.clear()
        if self._http_client is not None:
            await self._http_client.close()
            self._http_client = None

    async def drain_wakeups(self, *, timeout: float | None = None) -> None:
        tasks = [task for task in self._wakeup_tasks.values() if not task.done()]
        if not tasks:
            return
        await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=timeout)

    async def tick_once(self) -> int:
        now = utcnow()
        owned = await self._owned_shards_for_tick(now)
        if not owned:
            self._last_assignment_stats = SchedulerClaimStats()
            return 0
        transaction_started = time.monotonic()
        async with self.session_factory() as session:
            repo = self._repository(session)
            assignments = await repo.claim_and_assign_batch(shard_ids=owned, now=now)
            self._last_assignment_stats = repo.last_claim_stats
            if repo.lease_fence_lost:
                self._invalidate_shard_lease_cache()
                return 0
            if not await repo.revalidate_shard_lease_fence(now=utcnow()):
                self._invalidate_shard_lease_cache()
                return 0
            if repo.policy_demands and not repo.policy_snapshots_valid(now=utcnow()):
                await session.rollback()
                return 0
            policy_demands = set(repo.policy_demands)
            manager_ids = sorted({assignment.manager_id for assignment in assignments})
            manager_endpoints = await repo.manager_endpoints_by_id(manager_ids)
            await session.commit()
        self.metrics.record_scheduler_transaction_duration(
            duration_seconds=max(0.0, time.monotonic() - transaction_started)
        )

        pending = [key for key in policy_demands if time.monotonic() - self._last_policy_demand.get(key, 0) >= 10]
        if pending:
            try:
                async with self.session_factory() as session:
                    await PolicyRepository(session, shard_count=self.config.shard_count).demand(pending, now=utcnow())
                    await session.commit()
                self._last_policy_demand.update({key: time.monotonic() for key in pending})
            except Exception as exc:
                logger.warning('Policy demand refresh failed: error_type=%s', type(exc).__name__)
        for manager_id, endpoint_url in manager_endpoints.items():
            self._schedule_manager_wakeup(manager_id, endpoint_url)
        return len(assignments)

    async def _owned_shards_for_tick(self, now: datetime) -> list[int]:
        owned = await self._shard_leases.refresh(now)
        self.metrics.set_scheduler_owned_shards(len(owned))
        return owned

    def _record_loop(
        self,
        *,
        loop: str,
        outcome: ControlLoopOutcome,
        started: float,
        lag: float,
        changed: int = 0,
        backoff: float = 0,
        stats: SchedulerClaimStats | None = None,
    ) -> None:
        stats = stats or SchedulerClaimStats()
        self.metrics.record_loop(
            ControlLoopSample(
                component='scheduler',
                loop=loop,
                outcome=outcome,
                duration_seconds=max(0.0, time.monotonic() - started),
                lag_seconds=lag,
                scanned=stats.scanned,
                claimed=stats.claimed,
                changed=changed,
                backoff_seconds=backoff,
            )
        )

    def _repository(self, session) -> SchedulerRepository:
        return SchedulerRepository(
            session,
            shard_count=self.config.shard_count,
            manager_healthy_ttl=self.config.manager_healthy_ttl,
            no_capacity_backoff=self.config.no_capacity_backoff,
            default_user_quota=self.config.default_user_quota,
            scan_limit=self.config.scheduler_scan_limit,
            max_assignments=self.config.scheduler_max_assignments,
            manager_lock_limit=self.config.scheduler_manager_lock_limit,
            manager_lock_contention_backoff=self.config.scheduler_manager_lock_contention_backoff,
            lease_owner_id=self.lease_owner_id,
            lease_shard_group=self.shard_group,
            policy_config=self.config.policy_executor,
        )

    def _lease_repository(self, session: AsyncSession) -> ShardLeaseRepository:
        return _SchedulerShardLeaseRepository(
            session,
            scheduler_repository=self._repository(session),
            shard_count=self.config.shard_count,
        )

    def _schedule_manager_wakeup(self, manager_id: str, endpoint_url: str) -> None:
        existing = self._wakeup_tasks.get(manager_id)
        if existing is not None and not existing.done():
            return
        task = asyncio.create_task(
            self._wakeup_manager(manager_id, endpoint_url.rstrip('/')), name=f'dist-scheduler-wakeup:{manager_id}'
        )
        self._wakeup_tasks[manager_id] = task

        def _discard(done_task: asyncio.Task) -> None:
            if self._wakeup_tasks.get(manager_id) is done_task:
                self._wakeup_tasks.pop(manager_id, None)
            with contextlib.suppress(BaseException):
                done_task.result()

        task.add_done_callback(_discard)

    async def _wakeup_manager(self, manager_id: str, endpoint_url: str) -> None:
        headers = {}
        internal_token = self.config.read_internal_token()
        if internal_token:
            headers['X-Nexus-Internal-Token'] = internal_token
            if self.config.auth_mode == 'disabled':
                headers['Authorization'] = f'Bearer {internal_token}'
        try:
            session = await self._http_session()
            await session.post(
                f'{endpoint_url}/v1/internal/wakeup', headers=headers, timeout=aiohttp.ClientTimeout(total=2)
            )
        except Exception:
            logger.debug('Failed to wake manager %s', manager_id, exc_info=True)

    async def _http_session(self) -> aiohttp.ClientSession:
        if self._http_client is None or self._http_client.closed:
            self._http_client = aiohttp.ClientSession()
        return self._http_client

    def _invalidate_shard_lease_cache(self) -> None:
        self._shard_leases.invalidate()

    @property
    def lease_owner_id(self) -> str:
        return self._shard_leases.owner_id
