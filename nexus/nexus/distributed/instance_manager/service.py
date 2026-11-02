from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import random
import tempfile
import threading
import time

from collections.abc import Callable
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import async_sessionmaker

from nexus.distributed.artifacts.gateway import create_application_gateway_artifact_transport
from nexus.distributed.batch.control import BatchControlService
from nexus.distributed.batch.task_store import BatchTaskStore, create_batch_task_store
from nexus.distributed.config.views import InstanceManagerConfigView
from nexus.distributed.environment.models import (
    DEFAULT_ENVIRONMENT_ID,
    DEFAULT_POOL_ID,
)
from nexus.distributed.execution.executors.asyncio_executor import AsyncioExecutor
from nexus.distributed.execution.executors.base import (
    ExecutionDriver,
    Executor,
    ExecutorCapacitySnapshot,
    ExecutorHealthSnapshot,
    QuarantinedAssignment,
)
from nexus.distributed.execution.models import ExecutionMode
from nexus.distributed.execution.runtime_info import manager_capability_snapshot
from nexus.distributed.execution.trace_backends import (
    RunTraceBackendRegistry,
    create_run_trace_backend_registry,
)
from nexus.distributed.instance_manager.assignments import InstanceManagerAssignmentsMixin
from nexus.distributed.instance_manager.loops import InstanceManagerLoopPort, InstanceManagerLoopsMixin
from nexus.distributed.instance_manager.reporting import ManagerServicePort, RunningAssignment
from nexus.distributed.instance_manager.request_budget import RequestBudgetControl
from nexus.distributed.instance_manager.self_fence import HardExitGuard
from nexus.distributed.observability.event_loop import AiomonitorHandle, LoopAiomonitorConfig, start_aiomonitor_for_loop
from nexus.distributed.policies.models import PolicyExecutorConfig, RequestBudgetConfig, RequestPolicySelection
from nexus.distributed.policies.request_rate import validate_manager_request_policy
from nexus.distributed.repositories.manager import ManagerRepository
from nexus.distributed.utils.clock import utcnow
from nexus.distributed.utils.identity import new_id
from nexus.distributed.utils.threadsafe import await_threadsafe
from nexus.utils.platform_metrics import record


DriverFactory = Callable[..., ExecutionDriver]
logger = logging.getLogger(__name__)


_EXECUTOR_PROBE_TIMEOUT_SECONDS = 10.0


class InstanceManagerService(
    InstanceManagerAssignmentsMixin,
    InstanceManagerLoopsMixin,
    InstanceManagerLoopPort,
    ManagerServicePort,
):
    def __init__(
        self,
        *,
        session_factory: async_sessionmaker,
        config: InstanceManagerConfigView,
        manager_id: str,
        endpoint_url: str,
        capacity_total: int,
        driver_factories: dict[str, DriverFactory] | None = None,
        executor: Executor | None = None,
        force_replace: bool = False,
        environment_id: str | None = None,
        pool_id: str | None = None,
        workload: str | None = None,
        labels: dict[str, str] | None = None,
        metadata: dict[str, Any] | None = None,
        draining: bool = False,
        bootstrap_version: int | None = None,
        artifact_ref_version: int | None = None,
        owner_principal_id: str | None = None,
        allocation_id: str | None = None,
        allocation_spec_hash: str | None = None,
        request_policy: RequestPolicySelection | dict[str, Any] | None = None,
        capability_snapshot: dict[str, Any] | None = None,
        trace_registry: RunTraceBackendRegistry | None = None,
        batch_task_store: BatchTaskStore | None = None,
        hard_exit_guard: HardExitGuard | None = None,
    ) -> None:
        self.session_factory = session_factory
        self.config = config
        self.manager_id = manager_id
        self.session_id = new_id('session')
        self.endpoint_url = endpoint_url.rstrip('/')
        self.capacity_total = capacity_total
        self.force_replace = force_replace
        self.labels = dict(labels or {})
        self.metadata = dict(metadata or {})
        self.log_path: str | None = None
        self.draining = draining
        self.owner_principal_id = owner_principal_id or os.getenv('NEXUS_DIST_MANAGER_OWNER_PRINCIPAL_ID') or None
        self.environment_id = environment_id or os.getenv('NEXUS_DIST_MANAGER_ENVIRONMENT_ID') or DEFAULT_ENVIRONMENT_ID
        self.pool_id = pool_id or os.getenv('NEXUS_DIST_MANAGER_POOL_ID') or DEFAULT_POOL_ID
        self.allocation_id = allocation_id or os.getenv('NEXUS_DIST_MANAGER_ALLOCATION_ID') or None
        configured_workload = workload or os.getenv('NEXUS_DIST_MANAGER_WORKLOAD') or self.labels.get('workload')
        self.workload = configured_workload or 'execution'
        if self.workload not in {'execution', 'orchestration'}:
            raise ValueError(f'unsupported workload: {self.workload}')
        self.allocation_spec_hash = allocation_spec_hash or os.getenv('NEXUS_DIST_MANAGER_ALLOCATION_SPEC_HASH') or None
        raw_request_policy = (
            request_policy
            if request_policy is not None
            else (getattr(config, 'request_policy', None) if self.workload != 'orchestration' else None)
        )
        policy_executor_cfg = getattr(config, 'policy_executor', None) or PolicyExecutorConfig()
        budget_cfg = getattr(config, 'request_budget', None) or RequestBudgetConfig()
        self.request_policy = validate_manager_request_policy(
            policy_executor_cfg,
            raw_request_policy,
            environment_id=self.environment_id,
            owner_principal_id=self.owner_principal_id,
        )
        self.request_budget_control = (
            RequestBudgetControl(
                session_factory=session_factory,
                manager_id=self.manager_id,
                manager_session_id=self.session_id,
                environment_id=self.environment_id,
                owner_principal_id=self.owner_principal_id,
                selection=self.request_policy,
                policy_config=policy_executor_cfg,
                budget_config=budget_cfg,
                shard_count=getattr(config, 'shard_count', 64),
            )
            if self.request_policy is not None
            else None
        )
        self.bootstrap_version = bootstrap_version
        self.artifact_ref_version = artifact_ref_version
        self._artifact_transport = create_application_gateway_artifact_transport(config)
        self.batch_task_store = batch_task_store if batch_task_store is not None else create_batch_task_store(config)
        self.batch_control = BatchControlService(
            session_factory=session_factory,
            config=config,
            artifact_transport=self._artifact_transport,
            task_store=self.batch_task_store,
            on_blocking_timeout=self._handle_batch_blocking_timeout,
        )
        self._assignment_files_temporary_root = tempfile.TemporaryDirectory(
            prefix='assignment-files-',
            dir=config.batch_control_spool_dir,
        )
        self._assignment_files_root = Path(self._assignment_files_temporary_root.name)
        os.chmod(self._assignment_files_root, 0o700)
        self.trace_registry = trace_registry or create_run_trace_backend_registry(
            config,
            artifact_transport=self._artifact_transport,
        )
        aiomonitor_config = LoopAiomonitorConfig.from_distributed_config(config)
        self.executor = executor or AsyncioExecutor(capacity_total=capacity_total, aiomonitor_config=aiomonitor_config)
        self.manager_loop: asyncio.AbstractEventLoop | None = None
        self._aiomonitor_handle: AiomonitorHandle | None = None
        self._stopping = asyncio.Event()
        self._poll_wakeup_event = asyncio.Event()
        self._command_wakeup_event = asyncio.Event()
        self._heartbeat_lock = asyncio.Lock()
        self._heartbeat_task: asyncio.Task | None = None
        self._poll_task: asyncio.Task | None = None
        self._command_task: asyncio.Task | None = None
        self._local_reconcile_task: asyncio.Task | None = None
        self._executor_watchdog_task: asyncio.Task | None = None
        self._heartbeat_backoff_random = random.Random(f'{manager_id}:{self.session_id}:heartbeat')
        self._poll_lock = asyncio.Lock()
        self._command_lock = asyncio.Lock()
        self._manager_loop_futures_lock = threading.Lock()
        self._manager_loop_futures: set[Any] = set()
        # TODO: Guard _running with a small async lock/helper layer before adding more local registry users.
        self._running: dict[str, RunningAssignment] = {}
        self._executor_health = ExecutorHealthSnapshot(responsive=True, effective_capacity=capacity_total)
        self._executor_unresponsive_since: float | None = None
        self._quarantined_worker_ids: set[str] = set()
        self._hard_exit_guard = hard_exit_guard or HardExitGuard(
            timeout=config.manager_hard_exit_timeout.total_seconds()
        )
        self._hard_exit_enabled = False
        if driver_factories is not None:
            self.driver_factories = driver_factories
        elif self.workload == 'orchestration':
            self.driver_factories = {'nexus_batch': self._create_batch_driver}
        elif self.workload == 'execution':
            self.driver_factories = {
                'nexus_single': self._create_single_driver,
                'nexus_live': self._create_live_driver,
            }
        else:
            raise ValueError(f'unsupported workload: {self.workload}')
        self.supported_spec_versions = {name: [1] for name in self.driver_factories}
        self.capability_snapshot = dict(
            capability_snapshot
            or manager_capability_snapshot(
                supported_modes=[ExecutionMode.JOB, ExecutionMode.LIVE],
                supported_spec_types=sorted(self.driver_factories),
                supported_instance_modes=[self.executor.instance_mode],
                supported_spec_versions=self.supported_spec_versions,
                bootstrap_version=self.bootstrap_version,
                artifact_ref_version=self.artifact_ref_version,
            )
        )

    @property
    def supported_spec_types(self) -> list[str]:
        return sorted(self.driver_factories)

    async def start(self) -> None:
        self.manager_loop = asyncio.get_running_loop()
        self._aiomonitor_handle = start_aiomonitor_for_loop(
            self.manager_loop,
            config=LoopAiomonitorConfig.from_distributed_config(self.config),
            name=f'instance-manager:{self.manager_id}:main',
            port_offset=0,
        )
        try:
            await self.executor.start()
            registered = await self._register_session()
            if not registered:
                raise RuntimeError(f'healthy manager already exists: {self.manager_id}')
            self._hard_exit_enabled = True
        except Exception:
            with contextlib.suppress(Exception):
                await self.executor.close()
            if self._aiomonitor_handle is not None:
                self._aiomonitor_handle.close()
                self._aiomonitor_handle = None
            raise

    async def run_forever(self) -> None:
        await self.start()
        self.start_background_tasks()
        await self._stopping.wait()

    async def stop(self) -> None:
        self._stopping.set()
        try:
            background_tasks = (
                self._heartbeat_task,
                self._poll_task,
                self._command_task,
                self._local_reconcile_task,
                self._executor_watchdog_task,
            )
            for task in background_tasks:
                if task is not None:
                    task.cancel()
            for task in background_tasks:
                if task is not None:
                    with contextlib.suppress(asyncio.CancelledError):
                        await task
            if self.request_budget_control is not None:
                await self.request_budget_control.close()
            for assignment_id in list(self._running):
                try:
                    await self.executor.stop_assignment(assignment_id, timeout=1.0)
                finally:
                    await self._release_running_assignment(assignment_id)
            await self.executor.close()
            await self.batch_control.close()
            if self.batch_task_store is not None:
                await self.batch_task_store.close()
            await self.trace_registry.close()
            if self._artifact_transport is not None:
                await self._artifact_transport.close()
            await self._drain_manager_loop_futures(timeout=2.0)
        finally:
            for assignment_id in list(self._running):
                await self._release_running_assignment(assignment_id)
            if self._aiomonitor_handle is not None:
                try:
                    self._aiomonitor_handle.close()
                except Exception:
                    logger.warning('Failed to close instance-manager aiomonitor handle', exc_info=True)
                self._aiomonitor_handle = None
            self._hard_exit_enabled = False
            try:
                self._hard_exit_guard.disarm()
            except Exception:
                logger.warning('Failed to disarm instance-manager hard-exit guard', exc_info=True)
            try:
                await asyncio.to_thread(self._assignment_files_temporary_root.cleanup)
            except Exception:
                logger.warning('Failed to clean manager assignment task-files root', exc_info=True)

    async def wait_for_idle(self, *, timeout: float) -> None:
        deadline = asyncio.get_running_loop().time() + timeout
        while self._running and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.01)
        if self._running:
            raise TimeoutError('instance manager did not become idle')

    async def wait_stopped(self) -> None:
        await self._stopping.wait()

    async def _handle_batch_blocking_timeout(self, operation: str) -> None:
        await self._self_fence(f'batch_blocking_timeout:{operation}')

    async def call_manager_loop(self, coroutine, *, timeout: float = 30.0):
        if timeout <= 0:
            if hasattr(coroutine, 'close'):
                coroutine.close()
            raise TimeoutError
        if self.manager_loop is None or self.manager_loop is asyncio.get_running_loop():
            return await asyncio.wait_for(coroutine, timeout=timeout)
        future = asyncio.run_coroutine_threadsafe(coroutine, self.manager_loop)
        with self._manager_loop_futures_lock:
            self._manager_loop_futures.add(future)

        def discard_done(done_future) -> None:
            with contextlib.suppress(BaseException):
                done_future.result()
            with self._manager_loop_futures_lock:
                self._manager_loop_futures.discard(done_future)

        future.add_done_callback(discard_done)
        return await await_threadsafe(future, timeout=timeout)

    async def _drain_manager_loop_futures(self, *, timeout: float) -> None:
        with self._manager_loop_futures_lock:
            futures = [future for future in self._manager_loop_futures if not future.done()]
        if not futures:
            return
        try:
            await asyncio.wait_for(
                asyncio.gather(*(asyncio.wrap_future(future) for future in futures), return_exceptions=True),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            for future in futures:
                future.cancel()
            await asyncio.gather(*(asyncio.wrap_future(future) for future in futures), return_exceptions=True)

    async def _register_session(self) -> bool:
        capacity = self._local_capacity_snapshot()
        async with self.session_factory() as session:
            repo = ManagerRepository(session, healthy_ttl=self.config.manager_healthy_ttl)
            registered = await repo.register_session(
                manager_id=self.manager_id,
                session_id=self.session_id,
                endpoint_url=self.endpoint_url,
                supported_modes=[ExecutionMode.JOB, ExecutionMode.LIVE],
                supported_spec_types=self.supported_spec_types,
                supported_instance_modes=[self.executor.instance_mode],
                capacity_total=capacity.capacity_total,
                force_replace=self.force_replace,
                environment_id=self.environment_id,
                pool_id=self.pool_id,
                workload=self.workload,
                labels=self.labels,
                metadata=self.metadata,
                draining=self.draining,
                bootstrap_version=self.bootstrap_version,
                artifact_ref_version=self.artifact_ref_version,
                supported_spec_versions=self.supported_spec_versions,
                owner_principal_id=self.owner_principal_id,
                allocation_id=self.allocation_id,
                allocation_spec_hash=self.allocation_spec_hash,
                request_policy_binding=self.request_policy.binding if self.request_policy else None,
                request_policy_scope=self.request_policy.scope if self.request_policy else None,
                capability_snapshot=self.capability_snapshot,
                now=utcnow(),
            )
            await session.commit()
            return registered

    async def set_draining(self, draining: bool) -> bool:
        async with self._heartbeat_lock:
            async with self.session_factory() as session:
                updated = await ManagerRepository(
                    session,
                    healthy_ttl=self.config.manager_healthy_ttl,
                ).set_draining(
                    manager_id=self.manager_id,
                    session_id=self.session_id,
                    draining=draining,
                    now=utcnow(),
                )
                await session.commit()
            if not updated:
                await self._self_fence('draining_fenced')
                return False
            self.draining = draining
            self.request_wakeup()
            return True

    async def heartbeat_once(self) -> bool:
        async with self._heartbeat_lock:
            return await self._heartbeat_once_unlocked()

    async def _heartbeat_once_unlocked(self) -> bool:
        capacity = self._local_capacity_snapshot()
        async with self.session_factory() as session:
            repo = ManagerRepository(session, healthy_ttl=self.config.manager_healthy_ttl)
            ok = await repo.heartbeat(
                manager_id=self.manager_id,
                session_id=self.session_id,
                capacity_total=capacity.capacity_total,
                observed_capacity_used=capacity.running_slot_cost,
                now=utcnow(),
            )
            await session.commit()
        if not ok:
            await self._self_fence('heartbeat_fenced')
            return False
        return True

    def _local_capacity_snapshot(self) -> ExecutorCapacitySnapshot:
        running_slot_cost = sum(item.slot_cost for item in self._running.values())
        effective_capacity = max(0, self._executor_health.effective_capacity)
        return ExecutorCapacitySnapshot(
            capacity_total=effective_capacity,
            running_slot_cost=running_slot_cost,
            available_slots=max(0, effective_capacity - running_slot_cost),
        )

    async def _probe_executor_once(self, *, now: float | None = None) -> ExecutorHealthSnapshot:
        observed_at = time.monotonic() if now is None else now
        previous_health = self._executor_health
        try:
            health = await self.executor.probe_health(
                timeout=_EXECUTOR_PROBE_TIMEOUT_SECONDS,
                hang_timeout=self.config.manager_executor_hang_timeout.total_seconds(),
                now=observed_at,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception('Executor health probe failed')
            health = ExecutorHealthSnapshot(responsive=False, effective_capacity=0, unresponsive_workers=1)

        self._executor_health = health
        for quarantined in health.quarantined_assignments:
            self._record_quarantined_worker(quarantined)
        restart_started = False
        if health.recycle_requested:
            restart_started = await self._maybe_recycle_degraded_manager(health)
        for quarantined in health.quarantined_assignments:
            await self._replay_quarantined_assignment(quarantined)
        if health.responsive:
            unresponsive_since = self._executor_unresponsive_since
            reconciliation_required = not previous_health.responsive
            if reconciliation_required:
                try:
                    await self._reconcile_executor_assignments_after_recovery()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception('Executor assignment reconciliation after recovery failed')
                    health = ExecutorHealthSnapshot(responsive=False, effective_capacity=0, unresponsive_workers=1)
            if health.responsive and unresponsive_since is not None:
                logger.info(
                    'Executor loop recovered: manager_id=%s unavailable_seconds=%.3f',
                    self.manager_id,
                    max(0.0, observed_at - unresponsive_since),
                )
            if health.responsive:
                self._executor_unresponsive_since = None
                if not restart_started:
                    await self._maybe_recycle_degraded_manager(health)
                return health

        self._executor_health = health
        if self._executor_unresponsive_since is None:
            self._executor_unresponsive_since = observed_at
            logger.warning('Executor loop became unresponsive: manager_id=%s', self.manager_id)
            return health
        unavailable_for = max(0.0, observed_at - self._executor_unresponsive_since)
        if unavailable_for >= self.config.manager_executor_hang_timeout.total_seconds():
            logger.error(
                'Executor loop remained unresponsive: manager_id=%s unavailable_seconds=%.3f',
                self.manager_id,
                unavailable_for,
            )
            await self._self_fence('executor_loop_hang')
        return health

    def _mark_executor_unavailable(self) -> None:
        self._executor_health = ExecutorHealthSnapshot(
            responsive=False,
            effective_capacity=0,
            unresponsive_workers=1,
        )

    async def _reconcile_executor_assignments_after_recovery(self) -> None:
        running_assignments = list(self._running.values())
        presence = await asyncio.gather(
            *(self.executor.has_assignment(running.assignment_id) for running in running_assignments)
        )
        missing = [
            running
            for running, assignment_present in zip(running_assignments, presence, strict=True)
            if not assignment_present
        ]
        if not missing:
            return

        error = {
            'type': 'ExecutorAssignmentMissingAfterRecovery',
            'message': 'assignment was not present after the executor event loop recovered',
        }
        marked_assignment_ids: set[str] = set()
        async with self.session_factory() as session:
            repository = ManagerRepository(session, healthy_ttl=self.config.manager_healthy_ttl)
            for running in missing:
                if await repository.mark_assignment_lost(
                    running.assignment_id,
                    self.manager_id,
                    self.session_id,
                    reason='executor_recovery_assignment_missing',
                    error=error,
                    now=utcnow(),
                ):
                    marked_assignment_ids.add(running.assignment_id)
            await session.commit()

        for running in missing:
            current = self._running.get(running.assignment_id)
            if running.assignment_id in marked_assignment_ids or current is running:
                await self._release_running_assignment(running.assignment_id)
        self.request_wakeup()

    def _record_quarantined_worker(self, event: QuarantinedAssignment) -> None:
        running = self._running.get(event.assignment_id) if event.assignment_id is not None else None
        if running is not None and event.execution_id is not None and running.execution_id != event.execution_id:
            raise RuntimeError('quarantined assignment identity mismatch')
        if event.worker_id not in self._quarantined_worker_ids:
            self._quarantined_worker_ids.add(event.worker_id)
            record('instance_exits', kind='worker', reason='unresponsive')
            logger.error(
                'Thread worker quarantined: manager_id=%s worker_id=%s assignment_id=%s slot_cost=%s',
                self.manager_id,
                event.worker_id,
                event.assignment_id,
                event.slot_cost,
            )

    async def _replay_quarantined_assignment(self, event: QuarantinedAssignment) -> None:
        if event.assignment_id is None:
            return
        running = self._running.get(event.assignment_id)
        error = {
            'type': 'ExecutorLoopUnresponsive',
            'message': 'thread worker event loop remained unresponsive',
            'worker_id': event.worker_id,
        }
        async with self.session_factory() as session:
            marked = await ManagerRepository(session, healthy_ttl=self.config.manager_healthy_ttl).mark_assignment_lost(
                event.assignment_id,
                self.manager_id,
                self.session_id,
                reason='executor_loop_unresponsive',
                error=error,
                now=utcnow(),
            )
            await session.commit()
        if marked or running is not None:
            await self._release_running_assignment(event.assignment_id)
            self.request_wakeup()

    async def _maybe_recycle_degraded_manager(self, health: ExecutorHealthSnapshot) -> bool:
        if not health.recycle_requested or self._stopping.is_set():
            return False
        effective_capacity = health.effective_capacity
        if effective_capacity / self.capacity_total < self.config.thread_executor_min_capacity_ratio:
            await self._self_fence('thread_capacity_below_threshold')
            return True
        if not self._running:
            await self._self_fence('thread_quarantine_drained')
            return True
        return False

    async def update_local_snapshot_once(self) -> bool:
        local_instances = {
            assignment_id: {'execution_id': item.execution_id, 'assignment_id': item.assignment_id}
            for assignment_id, item in self._running.items()
        }
        async with self.session_factory() as session:
            updated = await ManagerRepository(
                session,
                healthy_ttl=self.config.manager_healthy_ttl,
            ).update_local_snapshot(
                manager_id=self.manager_id,
                session_id=self.session_id,
                local_instances=local_instances,
                now=utcnow(),
            )
            await session.commit()
        if not updated:
            await self._self_fence('local_snapshot_fenced')
        return updated
