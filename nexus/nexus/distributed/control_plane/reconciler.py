from __future__ import annotations

import asyncio
import contextlib
import logging
import time

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import async_sessionmaker

from nexus.distributed.artifacts.maintenance import ArtifactMaintenance, ArtifactMaintenanceReconciler
from nexus.distributed.artifacts.run_result import TransientResultPublishError
from nexus.distributed.artifacts.store import ArtifactStoreTransientError
from nexus.distributed.batch.lifecycle import BatchLifecycle, BatchLifecycleReconciler
from nexus.distributed.batch.task_store import create_batch_task_store
from nexus.distributed.config import DistributedConfig
from nexus.distributed.config.views import (
    ReconcilerConfigView,
    artifact_config_view,
    batch_lifecycle_config_view,
    dependency_build_config_view,
    environment_config_view,
    reconciler_config_view,
)
from nexus.distributed.control_plane.leases import ShardLeaseCoordinator
from nexus.distributed.control_plane.terminal_purge import (
    BoundlessPartitionPurgeReconciler,
    TerminalPurgeTransientError,
)
from nexus.distributed.dependency_build.controller import DependencyBuildController
from nexus.distributed.environment.controller import EnvironmentController
from nexus.distributed.environment.providers import WorkloadClientResolver
from nexus.distributed.observability import (
    ControlLoopOutcome,
    ControlLoopSample,
    ControlLoopWork,
    ControlPlaneMetrics,
    OpenTelemetryControlPlaneMetrics,
)
from nexus.distributed.persistence.db import is_retryable_database_error
from nexus.distributed.ports import ReconcileController
from nexus.distributed.repositories.quota import QuotaRepository
from nexus.distributed.repositories.reconciler import ReconcilerRepository
from nexus.distributed.repositories.shard_lease import ShardLeaseFenceStatus, ShardLeaseRepository
from nexus.distributed.utils.clock import utcnow
from nexus.runner.trace.tdsql.partition_maintenance import (
    TDSQLPartitionMaintenance,
    TDSQLPartitionMaintenanceError,
)


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from nexus.distributed.observability.sampler import StatusSampler


class ReconcilerService:
    def __init__(
        self,
        *,
        session_factory: async_sessionmaker,
        config: ReconcilerConfigView,
        owner_id: str = 'reconciler',
        environment_controller: ReconcileController,
        dependency_build_controller: ReconcileController,
        workload_client_resolver: WorkloadClientResolver,
        batch_lifecycle: BatchLifecycle,
        artifact_maintenance: ArtifactMaintenance | None = None,
        metrics: ControlPlaneMetrics | None = None,
        shard_group: str = 'execution-reconciler',
        status_sampler: StatusSampler | None = None,
        tdsql_partition_maintenance: TDSQLPartitionMaintenance | None = None,
        terminal_purge: BoundlessPartitionPurgeReconciler | None = None,
    ) -> None:
        self.session_factory = session_factory
        self.config = config
        self.owner_id = owner_id
        self.shard_group = shard_group
        self.metrics = metrics if metrics is not None else OpenTelemetryControlPlaneMetrics()
        self._loop_work: dict[str, ControlLoopWork] = {}
        self._stopping = asyncio.Event()
        self._core_operation_cursor = 0
        self._shard_leases = ShardLeaseCoordinator(
            session_factory=session_factory,
            shard_count=config.shard_count,
            owner_id=owner_id,
            shard_group=shard_group,
            lease_ttl=config.reconciler_lease_ttl,
            renew_interval=config.reconciler_lease_renew_interval,
            ownership_limit=config.reconciler_shard_ownership_limit,
            discover_interval=config.reconciler_shard_discover_interval,
            discover_jitter=config.reconciler_shard_discover_jitter,
        )
        self.workload_client_resolver = workload_client_resolver
        self.environment_controller = environment_controller
        self.dependency_build_controller = dependency_build_controller
        self.batch_lifecycle = batch_lifecycle
        self.artifact_maintenance = artifact_maintenance
        self.status_sampler = status_sampler
        self.tdsql_partition_maintenance = tdsql_partition_maintenance
        self.terminal_purge = terminal_purge

    async def run_forever(self) -> None:
        tasks = {
            asyncio.create_task(
                self._run_loop(
                    loop_name='core',
                    tick=self._core_tick_once,
                    poll_interval=self.config.reconciler_poll_interval,
                    timeout=self.config.reconciler_core_timeout,
                ),
                name='dist-reconciler:core',
            ),
            asyncio.create_task(
                self._run_lease_heartbeat(),
                name=f'dist-reconciler:lease:{self.owner_id}',
            ),
        }
        if self.tdsql_partition_maintenance is not None:
            tasks.add(
                asyncio.create_task(
                    self._run_loop(
                        loop_name='tdsql-partition',
                        tick=self._tdsql_partition_tick_once,
                        poll_interval=self.config.tdsql_partition_reconcile_interval,
                        timeout=self.config.tdsql_partition_reconcile_timeout,
                    ),
                    name='dist-reconciler:tdsql-partition',
                )
            )
        tasks.update(
            self._worker_tasks(
                loop_name='environment',
                tick=self._environment_tick_once,
                worker_count=self.config.environment_reconcile_worker_count,
                poll_interval=self.config.reconciler_poll_interval,
                timeout=self.config.environment_reconcile_timeout,
            )
        )
        tasks.update(
            self._worker_tasks(
                loop_name='dependency-build',
                tick=self._dependency_build_tick_once,
                worker_count=self.config.dependency_build_reconcile_worker_count,
                poll_interval=self.config.dependency_build_poll_interval.total_seconds(),
                timeout=self.config.dependency_build_reconcile_timeout,
            )
        )
        if self.artifact_maintenance is not None:
            tasks.update(
                self._worker_tasks(
                    loop_name='artifact-maintenance',
                    tick=self._artifact_maintenance_tick_once,
                    worker_count=self.config.artifact_maintenance_worker_count,
                    poll_interval=self.config.reconciler_poll_interval,
                    timeout=self.config.artifact_maintenance_reconcile_timeout,
                )
            )
        tasks.update(
            self._worker_tasks(
                loop_name='batch',
                tick=self._batch_tick_once,
                worker_count=self.config.batch_lifecycle_worker_count,
                poll_interval=self.config.batch_lifecycle_poll_interval,
                timeout=self.config.batch_lifecycle_reconcile_timeout,
            )
        )
        if self.status_sampler is not None:
            tasks.add(asyncio.create_task(self.status_sampler.run_forever(), name='dist-reconciler:status-sampler'))
        if self.terminal_purge is not None:
            tasks.add(
                asyncio.create_task(
                    self._run_loop(
                        loop_name='terminal-purge',
                        tick=self._terminal_purge_tick_once,
                        poll_interval=self.config.terminal_purge_reconcile_interval,
                        timeout=self.config.terminal_purge_reconcile_timeout,
                    ),
                    name='dist-reconciler:terminal-purge',
                )
            )
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

    def _worker_tasks(
        self,
        *,
        loop_name: str,
        tick: Callable[[], Awaitable[int]],
        worker_count: int,
        poll_interval: float,
        timeout: float,
    ) -> set[asyncio.Task]:
        return {
            asyncio.create_task(
                self._run_loop(
                    loop_name=loop_name,
                    tick=tick,
                    poll_interval=poll_interval,
                    timeout=timeout,
                ),
                name=(
                    f'dist-reconciler:{loop_name}'
                    if worker_index == 0
                    else f'dist-reconciler:{loop_name}:{worker_index}'
                ),
            )
            for worker_index in range(worker_count)
        }

    async def _run_loop(
        self,
        *,
        loop_name: str,
        tick: Callable[[], Awaitable[int]],
        poll_interval: float,
        timeout: float,
    ) -> None:
        next_due = time.monotonic()
        while not self._stopping.is_set():
            started = time.monotonic()
            lag = max(0.0, started - next_due)
            changed = 0
            outcome: ControlLoopOutcome = 'success'
            backoff = 0.0
            self._loop_work.pop(loop_name, None)
            try:
                changed = await asyncio.wait_for(tick(), timeout=timeout)
            except Exception as exc:
                if not _is_transient_loop_error(exc):
                    self._record_loop(
                        loop=loop_name,
                        outcome='fatal_error',
                        started=started,
                        lag=lag,
                    )
                    raise
                outcome = 'transient_error'
                backoff = self.config.reconciler_failure_backoff
                logger.warning(
                    'Reconciler loop retry: loop=%s owner=%s retry_reason=%s',
                    loop_name,
                    self.owner_id,
                    type(exc).__name__,
                    exc_info=True,
                )
            self._record_loop(
                loop=loop_name,
                outcome=outcome,
                started=started,
                lag=lag,
                changed=changed,
                backoff=backoff,
            )
            wait_interval = backoff if outcome == 'transient_error' else poll_interval
            next_due = time.monotonic() + wait_interval
            await self._wait_for_stop(wait_interval)

    async def _run_lease_heartbeat(self) -> None:
        interval = max(0.01, self.config.reconciler_lease_renew_interval / 2)
        while not self._stopping.is_set():
            try:
                owned = await self._shard_leases.refresh(utcnow())
                self.metrics.set_reconciler_owned_shards(len(owned))
            except Exception as exc:
                if not _is_transient_loop_error(exc):
                    raise
                logger.warning(
                    'Reconciler lease heartbeat retry: owner=%s shard_group=%s retry_reason=%s',
                    self.owner_id,
                    self.shard_group,
                    type(exc).__name__,
                    exc_info=True,
                )
            await self._wait_for_stop(interval)

    def _record_loop(
        self,
        *,
        loop: str,
        outcome: ControlLoopOutcome,
        started: float,
        lag: float,
        changed: int = 0,
        backoff: float = 0,
    ) -> None:
        work = self._loop_work.get(loop)
        self.metrics.record_loop(
            ControlLoopSample(
                component='reconciler',
                loop=loop,
                outcome=outcome,
                duration_seconds=max(0.0, time.monotonic() - started),
                lag_seconds=lag,
                scanned=work.scanned if work is not None else 0,
                claimed=work.claimed if work is not None else 0,
                changed=work.changed if work is not None else changed,
                backoff_seconds=backoff,
            )
        )

    async def _wait_for_stop(self, timeout: float) -> None:
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._stopping.wait(), timeout=timeout)

    def stop(self) -> None:
        self._stopping.set()

    async def close(self) -> None:
        if self.status_sampler is not None:
            await self.status_sampler.close()
        self.metrics.set_reconciler_owned_shards(0)
        with contextlib.suppress(Exception):
            await self._shard_leases.release()
        await self.batch_lifecycle.close()
        await self.environment_controller.close()
        await self.dependency_build_controller.close()
        if self.artifact_maintenance is not None:
            await self.artifact_maintenance.close()
        await self.workload_client_resolver.close()

    async def tick_once(self) -> int:
        changed = await self._core_tick_once()
        changed += await self._batch_tick_once()
        changed += await self._dependency_build_tick_once()
        changed += await self._environment_tick_once()
        changed += await self._artifact_maintenance_tick_once()
        changed += await self._tdsql_partition_tick_once()
        changed += await self._terminal_purge_tick_once()
        return changed

    async def _core_tick_once(self) -> int:
        now = utcnow()
        owned_shards = await self._shard_leases.refresh(now)
        self.metrics.set_reconciler_owned_shards(len(owned_shards))
        if not owned_shards:
            self._loop_work['core'] = ControlLoopWork()
            return 0
        changed = 0
        scanned = 0
        claimed = 0
        operations = (
            'reconcile_manager_dead',
            'reconcile_session_mismatch',
            'reconcile_pending_timeouts',
            'reconcile_starting_timeouts',
            'expire_queued',
            'request_ttl_stops',
            'repair_active_assignments',
            'reconcile_stopping_timeouts',
            'cleanup_terminal',
        )
        operation_batches = dict.fromkeys(operations, 0)
        available_operations = set(operations)
        operation_index = self._core_operation_cursor % len(operations)
        transaction_budget = len(operations) * self.config.reconciler_core_max_batches_per_tick
        deadline = time.monotonic() + self.config.reconciler_core_time_budget
        transactions = 0
        fence_failed = False
        while available_operations and transactions < transaction_budget:
            operation = operations[operation_index]
            operation_index = (operation_index + 1) % len(operations)
            self._core_operation_cursor = operation_index
            if operation not in available_operations:
                continue
            async with self.session_factory() as session:
                repo = ReconcilerRepository(
                    session,
                    manager_dead_ttl=self.config.manager_dead_ttl,
                    assign_ack_timeout=self.config.assign_ack_timeout,
                    start_timeout=self.config.start_timeout,
                    batch_size=self.config.reconciler_batch_size,
                    shard_ids=owned_shards,
                    shard_count=self.config.shard_count,
                    lease_owner_id=self._shard_leases.owner_id,
                    lease_shard_group=self.shard_group,
                )
                operation_changed = await getattr(repo, operation)(now=now)
                operation_stats = repo.last_stats
                if not await repo.finalize_shard_fence(now=utcnow()):
                    if repo.lease_fence_lost:
                        self._shard_leases.invalidate()
                    fence_failed = True
                    break
                changed += operation_changed
                scanned += operation_stats.scanned
                claimed += operation_stats.claimed
                transactions += 1
                operation_batches[operation] += 1
                if operation_stats.overdue_lag_seconds:
                    self.metrics.record_reconciler_overdue_lag(
                        operation=operation,
                        lag_seconds=operation_stats.overdue_lag_seconds,
                    )
                if operation_changed:
                    self.metrics.record_reconciler_repair(
                        reason=operation,
                        count=operation_changed,
                    )
                await session.commit()
            if (
                operation_stats.scanned < self.config.reconciler_batch_size
                or operation_batches[operation] >= self.config.reconciler_core_max_batches_per_tick
            ):
                available_operations.discard(operation)
            if transactions > 0 and time.monotonic() >= deadline:
                break

        if not fence_failed and 0 in owned_shards and time.monotonic() < deadline:
            async with self.session_factory() as session:
                quota_scanned = await QuotaRepository(
                    session,
                    default_limits=self.config.default_user_quota,
                ).repair_all(now=now, limit=self.config.reconciler_batch_size)
                fence_status = await ShardLeaseRepository(
                    session,
                    shard_count=self.config.shard_count,
                ).fence(
                    owner_id=self._shard_leases.owner_id,
                    shard_group=self.shard_group,
                    shard_ids=[0],
                    now=utcnow(),
                )
                if fence_status is ShardLeaseFenceStatus.VALID:
                    await session.commit()
                    scanned += quota_scanned
                    claimed += quota_scanned
                else:
                    await session.rollback()
                    if fence_status is ShardLeaseFenceStatus.LOST:
                        self._shard_leases.invalidate()
        self._loop_work['core'] = ControlLoopWork(
            scanned=scanned,
            claimed=claimed,
            changed=changed,
        )
        return changed

    async def _dependency_build_tick_once(self) -> int:
        changed = await self.dependency_build_controller.reconcile_once()
        stats = getattr(self.dependency_build_controller, 'last_stats', None)
        if isinstance(stats, ControlLoopWork):
            self._loop_work['dependency-build'] = stats
        return changed

    async def _environment_tick_once(self) -> int:
        changed = await self.environment_controller.reconcile_once()
        stats = getattr(self.environment_controller, 'last_environment_stats', None)
        if isinstance(stats, ControlLoopWork):
            self._loop_work['environment'] = stats
        return changed

    async def _artifact_maintenance_tick_once(self) -> int:
        if self.artifact_maintenance is None:
            return 0
        changed = await self.artifact_maintenance.reconcile_once()
        stats = getattr(self.artifact_maintenance, 'last_stats', None)
        if isinstance(stats, ControlLoopWork):
            self._loop_work['artifact-maintenance'] = stats
        return changed

    async def _batch_tick_once(self) -> int:
        set_shard_ids = getattr(self.batch_lifecycle, 'set_shard_ids', None)
        if set_shard_ids is not None:
            set_shard_ids(self._shard_leases.snapshot())
        changed = await self.batch_lifecycle.reconcile_once()
        stats = getattr(self.batch_lifecycle, 'last_stats', None)
        if isinstance(stats, ControlLoopWork):
            self._loop_work['batch'] = stats
        return changed

    async def _tdsql_partition_tick_once(self) -> int:
        if self.tdsql_partition_maintenance is None:
            return 0
        changed = await self.tdsql_partition_maintenance.reconcile_once()
        stats = self.tdsql_partition_maintenance.last_stats
        if isinstance(stats, ControlLoopWork):
            self._loop_work['tdsql-partition'] = stats
        return changed

    async def _terminal_purge_tick_once(self) -> int:
        if self.terminal_purge is None:
            return 0
        # Use shard 0 as the reconciler-wide singleton lease.  This keeps the purge loop
        # single-writer when multiple reconciler replicas share the same control database.
        if 0 not in self._shard_leases.snapshot():
            return 0
        return await self.terminal_purge.reconcile_once()


def _is_transient_loop_error(error: BaseException) -> bool:
    if isinstance(error, DBAPIError):
        return is_retryable_database_error(error)
    return isinstance(
        error,
        (
            TimeoutError,
            ArtifactStoreTransientError,
            TransientResultPublishError,
            TDSQLPartitionMaintenanceError,
            TerminalPurgeTransientError,
        ),
    )


def create_reconciler_service(
    *,
    session_factory: async_sessionmaker,
    config: DistributedConfig,
    owner_id: str = 'reconciler',
    metrics: ControlPlaneMetrics | None = None,
) -> ReconcilerService:
    from nexus.distributed.observability.sampler import StatusSampler

    resolved_metrics = metrics if metrics is not None else OpenTelemetryControlPlaneMetrics()
    environment_config = environment_config_view(config)
    resolver = WorkloadClientResolver(config=environment_config)
    artifact_maintenance = (
        ArtifactMaintenanceReconciler(
            session_factory=session_factory,
            config=artifact_config_view(config),
            owner_id=owner_id,
        )
        if config.artifact_maintenance_enabled
        else None
    )
    tdsql_partition_maintenance = None
    if config.tdsql_partition_reconcile_enabled:
        tdsql_partition_maintenance = TDSQLPartitionMaintenance.from_config(
            config,
            horizon_days=config.tdsql_partition_reconcile_horizon_days,
        )
        if not tdsql_partition_maintenance.targets:
            tdsql_partition_maintenance = None
    # Precreation and purge are independent loops with independent switches: one keeps
    # partitions ahead of the write frontier, the other reclaims terminated ones.
    terminal_purge = (
        BoundlessPartitionPurgeReconciler(session_factory=session_factory, config=config)
        if config.terminal_data_purge_enabled
        else None
    )
    return ReconcilerService(
        session_factory=session_factory,
        config=reconciler_config_view(config),
        owner_id=owner_id,
        environment_controller=EnvironmentController(
            session_factory=session_factory,
            config=environment_config,
            client_resolver=resolver,
        ),
        dependency_build_controller=DependencyBuildController(
            session_factory=session_factory,
            config=dependency_build_config_view(config),
            client_resolver=resolver,
            owner_id=f'{owner_id}:dependency-build',
        ),
        workload_client_resolver=resolver,
        batch_lifecycle=BatchLifecycleReconciler(
            session_factory=session_factory,
            config=batch_lifecycle_config_view(config),
            owner_id=owner_id,
            task_store=create_batch_task_store(config),
            metrics=resolved_metrics,
        ),
        artifact_maintenance=artifact_maintenance,
        metrics=resolved_metrics,
        status_sampler=StatusSampler(
            config,
            session_factory=session_factory
            if not config.read_database_url() or config.read_database_url().startswith('sqlite')
            else None,
            owner_id=f'{owner_id}:status',
        )
        if config.status_sampling.enabled and session_factory is not None
        else None,
        tdsql_partition_maintenance=tdsql_partition_maintenance,
        terminal_purge=terminal_purge,
    )


__all__ = ['ReconcilerService', 'create_reconciler_service']
