from __future__ import annotations

import asyncio
import contextlib
import logging
import random
import time

from typing import Protocol

from sqlalchemy.exc import DBAPIError
from sqlalchemy.exc import TimeoutError as SqlAlchemyTimeoutError

from nexus.distributed.config.views import InstanceManagerConfigView
from nexus.distributed.execution.executors.base import Executor
from nexus.distributed.instance_manager.reporting import RunningAssignment
from nexus.distributed.instance_manager.request_budget import RequestBudgetControl
from nexus.distributed.persistence.db import is_retryable_database_error


logger = logging.getLogger(__name__)
_EXECUTOR_PROBE_INTERVAL_SECONDS = 5.0


class HardExitGuardPort(Protocol):
    def arm(self, reason: str) -> None: ...

    def disarm(self) -> None: ...


class InstanceManagerLoopPort(Protocol):
    config: InstanceManagerConfigView
    executor: Executor
    manager_id: str
    _command_task: asyncio.Task | None
    _command_wakeup_event: asyncio.Event
    _heartbeat_backoff_random: random.Random
    _heartbeat_task: asyncio.Task | None
    _hard_exit_enabled: bool
    _hard_exit_guard: HardExitGuardPort
    _executor_watchdog_task: asyncio.Task | None
    _local_reconcile_task: asyncio.Task | None
    _poll_task: asyncio.Task | None
    _poll_wakeup_event: asyncio.Event
    _running: dict[str, RunningAssignment]
    _stopping: asyncio.Event
    request_budget_control: RequestBudgetControl | None

    async def heartbeat_once(self) -> bool: ...

    async def poll_assignments_once(self) -> int: ...

    async def claim_commands_once(self) -> int: ...

    async def update_local_snapshot_once(self) -> bool: ...

    async def reconcile_local_assignments_once(self) -> int: ...

    async def _probe_executor_once(self, *, now: float | None = None): ...

    async def _heartbeat_loop(self) -> None: ...

    async def _poll_loop(self) -> None: ...

    async def _command_loop(self) -> None: ...

    async def _local_reconcile_loop(self) -> None: ...

    async def _executor_watchdog_loop(self) -> None: ...

    def _heartbeat_task_done(self, task: asyncio.Task) -> None: ...

    async def _self_fence(self, reason: str) -> None: ...

    async def _release_running_assignment(self, assignment_id: str) -> RunningAssignment | None: ...

    async def _wait_for_stop(self, timeout: float) -> None: ...

    async def _wait_for_wakeup(self, event: asyncio.Event) -> None: ...


class InstanceManagerLoopsMixin:
    def start_background_tasks(self: InstanceManagerLoopPort) -> None:
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop(), name=f'dist-heartbeat:{self.manager_id}')
        self._heartbeat_task.add_done_callback(self._heartbeat_task_done)
        self._poll_task = asyncio.create_task(self._poll_loop(), name=f'dist-poll:{self.manager_id}')
        self._command_task = asyncio.create_task(self._command_loop(), name=f'dist-command:{self.manager_id}')
        self._local_reconcile_task = asyncio.create_task(
            self._local_reconcile_loop(), name=f'dist-local-reconcile:{self.manager_id}'
        )
        self._executor_watchdog_task = asyncio.create_task(
            self._executor_watchdog_loop(), name=f'dist-executor-watchdog:{self.manager_id}'
        )
        if self.request_budget_control is not None:
            self.request_budget_control.start()

    def request_wakeup(self: InstanceManagerLoopPort) -> None:
        self._poll_wakeup_event.set()
        self._command_wakeup_event.set()

    async def _heartbeat_loop(self: InstanceManagerLoopPort) -> None:
        last_confirmed = time.monotonic()
        self_fence_timeout = self.config.manager_self_fence_timeout.total_seconds()
        failure_count = 0
        while not self._stopping.is_set():
            deadline = last_confirmed + self_fence_timeout
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                await self._self_fence('heartbeat_timeout')
                return
            try:
                if not await asyncio.wait_for(self.heartbeat_once(), timeout=remaining):
                    return
            except asyncio.CancelledError:
                raise
            except (DBAPIError, SqlAlchemyTimeoutError, TimeoutError) as exc:
                if isinstance(exc, (DBAPIError, SqlAlchemyTimeoutError)) and not is_retryable_database_error(exc):
                    logger.exception('Instance manager heartbeat failed permanently')
                    await self._self_fence('heartbeat_fatal_error')
                    return
                failure_count += 1
                elapsed = time.monotonic() - last_confirmed
                if time.monotonic() >= deadline:
                    logger.error(
                        'Instance manager self-fencing after heartbeat timeout: manager_id=%s elapsed=%.3f',
                        self.manager_id,
                        elapsed,
                    )
                    await self._self_fence('heartbeat_timeout')
                    return
                base = min(0.25 * (2 ** min(failure_count - 1, 5)), self.config.manager_heartbeat_interval)
                delay = base * self._heartbeat_backoff_random.uniform(0.75, 1.25)
                delay = min(delay, max(0.0, deadline - time.monotonic()))
                logger.warning(
                    'Retrying Instance manager heartbeat: manager_id=%s failure_count=%s delay=%.3f',
                    self.manager_id,
                    failure_count,
                    delay,
                    exc_info=True,
                )
                await self._wait_for_stop(delay)
                continue
            except Exception:
                logger.exception('Instance manager heartbeat task failed')
                await self._self_fence('heartbeat_fatal_error')
                return
            failure_count = 0
            last_confirmed = time.monotonic()
            await self._wait_for_stop(self.config.manager_heartbeat_interval)

    async def _poll_loop(self: InstanceManagerLoopPort) -> None:
        while not self._stopping.is_set():
            try:
                await self.poll_assignments_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception('Instance manager poll loop failed')
            if self._stopping.is_set():
                break
            await self._wait_for_wakeup(self._poll_wakeup_event)

    async def _command_loop(self: InstanceManagerLoopPort) -> None:
        while not self._stopping.is_set():
            try:
                await self.claim_commands_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception('Instance manager command loop failed')
            if self._stopping.is_set():
                break
            await self._wait_for_wakeup(self._command_wakeup_event)

    async def _local_reconcile_loop(self: InstanceManagerLoopPort) -> None:
        while not self._stopping.is_set():
            try:
                if not await self.update_local_snapshot_once():
                    return
                await self.reconcile_local_assignments_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception('Instance manager local reconcile loop failed')
            if self._stopping.is_set():
                break
            await asyncio.sleep(self.config.manager_local_reconcile_interval)

    async def _executor_watchdog_loop(self: InstanceManagerLoopPort) -> None:
        while not self._stopping.is_set():
            try:
                await self._probe_executor_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception('Instance manager executor watchdog failed')
            if self._stopping.is_set():
                break
            await self._wait_for_stop(_EXECUTOR_PROBE_INTERVAL_SECONDS)

    def _heartbeat_task_done(self: InstanceManagerLoopPort, task: asyncio.Task) -> None:
        if task.cancelled() or self._stopping.is_set():
            return
        with contextlib.suppress(BaseException):
            error = task.exception()
            if error is not None:
                logger.error(
                    'Instance manager heartbeat task exited',
                    exc_info=(type(error), error, error.__traceback__),
                )
        self._stopping.set()

    async def _self_fence(self: InstanceManagerLoopPort, reason: str) -> None:
        if self._stopping.is_set() and not self._running:
            return
        if self._hard_exit_enabled:
            self._hard_exit_guard.arm(reason)
        logger.warning('Instance manager self-fenced: manager_id=%s reason=%s', self.manager_id, reason)
        self._stopping.set()
        if self.request_budget_control is not None:
            with contextlib.suppress(Exception):
                await self.request_budget_control.close()
        for assignment_id in list(self._running):
            with contextlib.suppress(Exception):
                await self.executor.stop_assignment(assignment_id, timeout=1.0)
            await self._release_running_assignment(assignment_id)

    async def _wait_for_stop(self: InstanceManagerLoopPort, timeout: float) -> None:
        if timeout <= 0:
            await asyncio.sleep(0)
            return
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._stopping.wait(), timeout=timeout)

    async def _wait_for_wakeup(self: InstanceManagerLoopPort, event: asyncio.Event) -> None:
        try:
            await asyncio.wait_for(event.wait(), timeout=self.config.manager_poll_interval)
        except asyncio.TimeoutError:
            return
        event.clear()


__all__ = ['InstanceManagerLoopPort', 'InstanceManagerLoopsMixin']
