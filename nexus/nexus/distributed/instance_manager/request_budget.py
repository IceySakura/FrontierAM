"""InstanceManager-local shared request budget coordinator and pacer.

Each InstanceManager bound to a ``request_policy`` owns a single ``RequestBudgetControl`` instance.
It coordinates at most one in-flight lease request in the background and paces local ``acquire()``
calls in memory across worker threads.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import logging
import random
import threading
import time

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import async_sessionmaker

from nexus.core.request_admission import (
    RequestAdmission,
    RequestAttempt,
    RequestPermit,
    SmoothRatePacer,
)
from nexus.distributed.persistence.orm import DistributedConfigOverrideRow
from nexus.distributed.policies.models import (
    PolicyExecutorConfig,
    RequestBudgetConfig,
    RequestPolicySelection,
)
from nexus.distributed.policies.request_rate import (
    RequestPolicyMode,
    configured_rate_per_minute,
    request_budget_key,
    validate_manager_request_policy,
)
from nexus.distributed.repositories.policy import PolicyRepository
from nexus.distributed.repositories.request_budget import (
    BudgetDemand,
    BudgetLease,
    BudgetLeaseResult,
    RequestBudgetRepository,
)


logger = logging.getLogger(__name__)

_POLICY_DEMAND_MIN_INTERVAL_SECONDS = 5.0
_FAILURE_BACKOFF_BASE_SECONDS = 0.5
_FAILURE_BACKOFF_MAX_SECONDS = 5.0


@dataclass(slots=True)
class _Waiter:
    loop: asyncio.AbstractEventLoop
    event: asyncio.Event
    execution_id: str
    attempt_id: str
    cancelled: bool = False

    def wake(self) -> None:
        with contextlib.suppress(RuntimeError):
            self.loop.call_soon_threadsafe(self.event.set)


@dataclass(frozen=True, slots=True)
class RequestBudgetStatusSnapshot:
    binding: str
    scope: str
    budget_key: str
    mode: RequestPolicyMode
    observed_config_revision: int
    observed_config_hash: str
    configured_quota_qpm: float | None
    lease_id: str | None
    lease_version: int | None
    granted_qpm: float
    lease_valid_until: datetime | None
    local_deadline_remaining_seconds: float | None
    snapshot_version: int | None
    reason_code: str
    waiting_requests: int
    permitted_requests_total: int
    enforced_permitted_requests_total: int
    leased_permit_capacity_total: float
    lease_utilization_ratio: float
    wait_seconds_total: float
    coordination_requests_total: int
    coordination_failures_total: int
    coordination_seconds_total: float

    def to_json(self) -> dict[str, Any]:
        return {
            'binding': self.binding,
            'scope': self.scope,
            'budget_key': self.budget_key,
            'mode': self.mode,
            'observed_config_revision': self.observed_config_revision,
            'observed_config_hash': self.observed_config_hash,
            'configured_quota_qpm': self.configured_quota_qpm,
            'lease_id': self.lease_id,
            'lease_version': self.lease_version,
            'granted_qpm': self.granted_qpm,
            'lease_valid_until': self.lease_valid_until.isoformat() if self.lease_valid_until else None,
            'local_deadline_remaining_seconds': self.local_deadline_remaining_seconds,
            'snapshot_version': self.snapshot_version,
            'reason_code': self.reason_code,
            'waiting_requests': self.waiting_requests,
            'permitted_requests_total': self.permitted_requests_total,
            'enforced_permitted_requests_total': self.enforced_permitted_requests_total,
            'leased_permit_capacity_total': round(self.leased_permit_capacity_total, 6),
            'lease_utilization_ratio': round(self.lease_utilization_ratio, 6),
            'wait_seconds_total': round(self.wait_seconds_total, 6),
            'coordination_requests_total': self.coordination_requests_total,
            'coordination_failures_total': self.coordination_failures_total,
            'coordination_seconds_total': round(self.coordination_seconds_total, 6),
        }


class RequestBudgetControl(RequestAdmission):
    """Single-binding request budget coordinator and local pacer for one InstanceManager."""

    def __init__(
        self,
        *,
        session_factory: async_sessionmaker,
        manager_id: str,
        manager_session_id: str,
        environment_id: str,
        owner_principal_id: str | None = None,
        selection: RequestPolicySelection,
        policy_config: PolicyExecutorConfig,
        budget_config: RequestBudgetConfig | None = None,
        shard_count: int = 64,
        monotonic_clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], datetime] | None = None,
        sleep_func: Callable[[float], Awaitable[None]] | None = None,
        config_loader: Callable[[], Awaitable[PolicyExecutorConfig | None]] | None = None,
    ) -> None:
        validated = validate_manager_request_policy(
            policy_config,
            selection,
            environment_id=environment_id,
            owner_principal_id=owner_principal_id,
        )
        assert validated is not None
        self.session_factory = session_factory
        self.manager_id = manager_id
        self.manager_session_id = manager_session_id
        self.environment_id = environment_id
        self.owner_principal_id = owner_principal_id
        self.selection = validated
        self.budget_key = request_budget_key(validated)
        self.budget_config = budget_config or RequestBudgetConfig()
        self.shard_count = shard_count
        self._monotonic = monotonic_clock
        self._wall_clock = wall_clock
        self._sleep = sleep_func
        self._config_loader = config_loader

        binding = policy_config.bindings[validated.binding]
        self._policy_config = policy_config
        self._mode: RequestPolicyMode = binding.mode
        self._observed_config_revision: int = binding.revision
        self._observed_config_hash: str = binding.fingerprint(policy_config.imports)
        self._config_invalid: bool = False

        self._lock = threading.Lock()
        self._pacer = SmoothRatePacer()
        self._active_lease: BudgetLease | None = None
        self._lease_deadline_mono: float | None = None
        self._last_reason_code: str = (
            f'policy_{binding.mode}' if binding.mode in {'shadow', 'disabled'} else 'policy_unavailable'
        )
        self._waiters: collections.deque[_Waiter] = collections.deque()
        self._stopped_executions: set[str] = set()
        self._stopped_attempts: set[tuple[str, str]] = set()
        self._last_activity_mono: float | None = None
        self._last_policy_demand_mono: float | None = None

        self._next_seq: int = 0
        self._pending_demand: BudgetDemand | None = None
        self._consecutive_failures: int = 0
        self._coordinator_has_active_demand: bool = False
        self._next_coordinate_at_mono: float = 0.0

        self._permitted_requests_total: int = 0
        self._enforced_permitted_requests_total: int = 0
        self._leased_permit_capacity_total: float = 0.0
        self._wait_seconds_total: float = 0.0
        self._coordination_requests_total: int = 0
        self._coordination_failures_total: int = 0
        self._coordination_seconds_total: float = 0.0

        self._closed: bool = False
        self._manager_loop: asyncio.AbstractEventLoop | None = None
        self._wakeup_event: asyncio.Event | None = None
        self._task: asyncio.Task[None] | None = None
        self._rng = random.Random(f'{manager_id}:{manager_session_id}:{self.budget_key}')

    @property
    def mode(self) -> RequestPolicyMode:
        with self._lock:
            return self._mode

    def _metrics_locked(self) -> dict[str, Any]:
        utilization = (
            min(1.0, self._enforced_permitted_requests_total / self._leased_permit_capacity_total)
            if self._leased_permit_capacity_total > 0
            else 0.0
        )
        return {
            'waiting_requests': len(self._waiters),
            'permitted_requests_total': self._permitted_requests_total,
            'enforced_permitted_requests_total': self._enforced_permitted_requests_total,
            'leased_permit_capacity_total': round(self._leased_permit_capacity_total, 6),
            'lease_utilization_ratio': round(utilization, 6),
            'wait_seconds_total': round(self._wait_seconds_total, 6),
            'coordination_requests_total': self._coordination_requests_total,
            'coordination_failures_total': self._coordination_failures_total,
            'coordination_seconds_total': round(self._coordination_seconds_total, 6),
        }

    def status_snapshot(self) -> RequestBudgetStatusSnapshot:
        with self._lock:
            now_mono = self._monotonic()
            lease = self._active_lease
            deadline = self._lease_deadline_mono
            valid = lease is not None and deadline is not None and now_mono < deadline
            binding = self._policy_config.bindings.get(self.selection.binding)
            quota_qpm = configured_rate_per_minute(binding, self.selection.scope)
            metrics = self._metrics_locked()
            return RequestBudgetStatusSnapshot(
                binding=self.selection.binding,
                scope=self.selection.scope,
                budget_key=self.budget_key,
                mode=self._mode,
                observed_config_revision=self._observed_config_revision,
                observed_config_hash=self._observed_config_hash,
                configured_quota_qpm=quota_qpm,
                lease_id=lease.lease_id if valid and lease is not None else None,
                lease_version=lease.lease_version if valid and lease is not None else None,
                granted_qpm=lease.granted_qpm if valid and lease is not None else 0.0,
                lease_valid_until=lease.valid_until if valid and lease is not None else None,
                local_deadline_remaining_seconds=(max(0.0, deadline - now_mono) if valid and deadline else None),
                snapshot_version=lease.snapshot_version if valid and lease is not None else None,
                reason_code=self._last_reason_code,
                **metrics,
            )

    def start(self) -> None:
        """Start the background coordination task without blocking on initial policy evaluation."""
        with self._lock:
            if self._closed:
                raise RuntimeError('request_budget_control_closed')
            if self._task is not None:
                return
            loop = asyncio.get_running_loop()
            self._manager_loop = loop
            self._wakeup_event = asyncio.Event()
            self._task = loop.create_task(
                self._coordination_loop(),
                name=f'dist-request-budget:{self.manager_id}',
            )

    async def close(self) -> None:
        """Stop lease renewal, revoke local lease usage, and cancel all waiting requests."""
        with self._lock:
            if self._closed:
                task = self._task
            else:
                self._closed = True
                self._active_lease = None
                self._lease_deadline_mono = None
                task = self._task
                self._task = None
                waiters = list(self._waiters)
                self._waiters.clear()
                self._notify_coordinator_locked()
        for waiter in waiters if 'waiters' in locals() else ():
            waiter.cancelled = True
            waiter.wake()
        if task is not None and task is not asyncio.current_task():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    def cancel_execution(self, execution_id: str, *, attempt_id: str | None = None) -> None:
        """Cancel pending acquire() waiters of a stopped or finished task without disturbing other tasks."""
        with self._lock:
            if attempt_id is None:
                self._stopped_executions.add(execution_id)
            else:
                self._stopped_attempts.add((execution_id, attempt_id))
            to_wake: list[_Waiter] = []
            for waiter in self._waiters:
                if waiter.execution_id == execution_id and (attempt_id is None or waiter.attempt_id == attempt_id):
                    waiter.cancelled = True
                    to_wake.append(waiter)
        for waiter in to_wake:
            waiter.wake()

    def clear_execution_cancellation(self, execution_id: str, *, attempt_id: str | None = None) -> None:
        with self._lock:
            if attempt_id is None:
                self._stopped_executions.discard(execution_id)
            else:
                self._stopped_attempts.discard((execution_id, attempt_id))

    def request_wakeup(self) -> None:
        with self._lock:
            self._next_coordinate_at_mono = 0.0
            self._notify_coordinator_locked()

    def _notify_coordinator_locked(self) -> None:
        if self._manager_loop is not None and self._wakeup_event is not None:
            with contextlib.suppress(RuntimeError):
                self._manager_loop.call_soon_threadsafe(self._wakeup_event.set)

    def _should_wake_coordinator_for_demand_locked(self, now_mono: float) -> bool:
        if not self._coordinator_has_active_demand:
            return True
        if self._mode in {'shadow', 'disabled'}:
            return False
        lease = self._active_lease
        deadline = self._lease_deadline_mono
        if lease is None or deadline is None or now_mono >= deadline or lease.granted_qpm <= 0:
            return now_mono >= self._next_coordinate_at_mono
        if deadline - now_mono <= self.budget_config.renew_before_seconds:
            return now_mono >= self._next_coordinate_at_mono
        return False

    async def acquire(self, request: RequestAttempt) -> RequestPermit:
        """Acquire a one-time send permit for ``request`` using local state only."""
        loop = asyncio.get_running_loop()
        waiter = _Waiter(
            loop=loop,
            event=asyncio.Event(),
            execution_id=request.execution_id,
            attempt_id=request.attempt_id,
        )
        started_wait_mono = self._monotonic()
        with self._lock:
            self._check_allowed_locked(request, waiter)
            now_mono = self._monotonic()
            self._last_activity_mono = now_mono
            if not self._waiters:
                permit, _ = self._try_head_permit_or_delay_locked(request, now_mono=now_mono)
                if permit is not None:
                    return permit
            self._waiters.append(waiter)
            if self._should_wake_coordinator_for_demand_locked(now_mono):
                self._notify_coordinator_locked()

        try:
            while True:
                waiter.event.clear()
                sleep_timeout: float | None = None
                with self._lock:
                    self._check_allowed_locked(request, waiter)
                    now_mono = self._monotonic()
                    self._last_activity_mono = now_mono
                    is_head = bool(self._waiters and self._waiters[0] is waiter)
                    if is_head:
                        permit, delay = self._try_head_permit_or_delay_locked(request, now_mono=now_mono)
                        if permit is not None:
                            self._waiters.popleft()
                            self._wait_seconds_total += max(0.0, now_mono - started_wait_mono)
                            if self._waiters:
                                self._waiters[0].wake()
                            return permit
                        sleep_timeout = delay
                        if self._should_wake_coordinator_for_demand_locked(now_mono):
                            self._notify_coordinator_locked()

                if sleep_timeout is None:
                    await waiter.event.wait()
                elif self._sleep is not None:
                    await self._wait_with_custom_sleep(waiter.event, max(0.001, sleep_timeout))
                else:
                    with contextlib.suppress(asyncio.TimeoutError):
                        await asyncio.wait_for(waiter.event.wait(), timeout=max(0.001, sleep_timeout))
        finally:
            with self._lock:
                if waiter in self._waiters:
                    self._waiters.remove(waiter)
                    if self._waiters:
                        self._waiters[0].wake()

    def _check_allowed_locked(self, request: RequestAttempt, waiter: _Waiter) -> None:
        if self._closed:
            raise RuntimeError('request_budget_control_closed')
        if (
            waiter.cancelled
            or request.execution_id in self._stopped_executions
            or (request.execution_id, request.attempt_id) in self._stopped_attempts
        ):
            raise asyncio.CancelledError('request_attempt_cancelled')
        if self._config_invalid:
            raise RuntimeError('request_policy_configuration_invalid')

    def _try_head_permit_or_delay_locked(
        self,
        request: RequestAttempt,
        *,
        now_mono: float,
    ) -> tuple[RequestPermit | None, float | None]:
        if self._mode in {'shadow', 'disabled'}:
            self._permitted_requests_total += 1
            if not self._coordinator_has_active_demand:
                self._notify_coordinator_locked()
            return (
                RequestPermit(
                    execution_id=request.execution_id,
                    attempt_id=request.attempt_id,
                    session_id=self.manager_session_id,
                    lease_id=None,
                    deadline_at=None,
                ),
                None,
            )

        # Enforce mode: verify local lease deadline before every single permit issuance.
        lease = self._active_lease
        deadline = self._lease_deadline_mono
        if lease is None or deadline is None or now_mono >= deadline or lease.granted_qpm <= 0:
            self._active_lease = None
            self._lease_deadline_mono = None
            return None, None

        wait_seconds = self._pacer.try_acquire(qpm=lease.granted_qpm, now=now_mono)
        if wait_seconds <= 0.0:
            self._permitted_requests_total += 1
            self._enforced_permitted_requests_total += 1
            if self._should_wake_coordinator_for_demand_locked(now_mono):
                self._notify_coordinator_locked()
            return (
                RequestPermit(
                    execution_id=request.execution_id,
                    attempt_id=request.attempt_id,
                    session_id=self.manager_session_id,
                    lease_id=lease.lease_id,
                    deadline_at=deadline,
                ),
                None,
            )

        return None, min(wait_seconds, max(0.001, deadline - now_mono))

    def _has_recent_demand_locked(self, now_mono: float) -> bool:
        if self._waiters:
            return True
        if self._last_activity_mono is None:
            return False
        return (now_mono - self._last_activity_mono) < self.budget_config.lease_seconds

    async def coordinate_once(self) -> BudgetLeaseResult | None:
        """Run one background coordination step (config check, policy demand, lease request/renewal)."""
        with self._lock:
            if self._closed:
                return None
            now_mono = self._monotonic()
            has_demand = self._has_recent_demand_locked(now_mono)
            had_lease = self._active_lease is not None

        await self._refresh_policy_config_if_available()

        with self._lock:
            if self._closed:
                return None
            if self._config_invalid:
                self._active_lease = None
                self._lease_deadline_mono = None
                self._wake_all_locked()
                return None
            now_mono = self._monotonic()
            has_demand = self._has_recent_demand_locked(now_mono)
            self._coordinator_has_active_demand = has_demand
            # When idle with no active lease in enforce mode, skip unnecessary DB lease writes.
            if not has_demand and not had_lease and self._pending_demand is None and self._mode == 'enforce':
                return None

            should_register_policy_demand = has_demand and (
                self._last_policy_demand_mono is None
                or (now_mono - self._last_policy_demand_mono) >= _POLICY_DEMAND_MIN_INTERVAL_SECONDS
            )
            requested_qpm = None if has_demand else 0.0
            local_metrics = self._metrics_locked()
            if self._pending_demand is not None:
                demand = self._pending_demand
            else:
                self._next_seq += 1
                demand = BudgetDemand(
                    budget_key=self.budget_key,
                    manager_id=self.manager_id,
                    manager_session_id=self.manager_session_id,
                    request_seq=self._next_seq,
                    requested_qpm=requested_qpm,
                    metrics=local_metrics,
                )
                self._pending_demand = demand
            policy_config = self._policy_config

        if should_register_policy_demand:
            await self._register_policy_demand()

        t0 = self._monotonic()
        try:
            async with self.session_factory() as session:
                repo = RequestBudgetRepository(session)
                result = await repo.coordinate_lease(
                    demand,
                    selection=self.selection,
                    config=policy_config,
                    budget_config=self.budget_config,
                    now=self._wall_clock() if self._wall_clock is not None else None,
                )
                await session.commit()
        except asyncio.CancelledError:
            raise
        except Exception:
            with self._lock:
                now_mono = self._monotonic()
                self._coordination_requests_total += 1
                self._coordination_failures_total += 1
                self._coordination_seconds_total += max(0.0, now_mono - t0)
                self._consecutive_failures += 1
                if self._lease_deadline_mono is not None and now_mono >= self._lease_deadline_mono:
                    self._active_lease = None
                    self._lease_deadline_mono = None
            logger.warning(
                'Request budget coordination failed: manager=%s binding=%s scope=%s',
                self.manager_id,
                self.selection.binding,
                self.selection.scope,
                exc_info=True,
            )
            return None

        with self._lock:
            now_mono = self._monotonic()
            self._coordination_requests_total += 1
            self._coordination_seconds_total += max(0.0, now_mono - t0)
            self._consecutive_failures = 0
            # Definitive response received: next coordination cycle uses a new request_seq.
            self._pending_demand = None
            self._apply_result_locked(result, t0=t0, now_mono=now_mono)
        return result

    def _apply_result_locked(self, result: BudgetLeaseResult, *, t0: float, now_mono: float) -> None:
        # Never allow an older configuration revision response to roll back a newer observed revision.
        if result.config_revision < self._observed_config_revision:
            return
        self._observed_config_revision = result.config_revision
        self._observed_config_hash = result.config_hash
        self._mode = result.mode
        self._last_reason_code = result.reason_code

        if result.mode in {'shadow', 'disabled'}:
            self._active_lease = None
            self._lease_deadline_mono = None
            self._wake_all_locked()
            return

        if result.status == 'granted' and result.lease is not None:
            lease = result.lease
            if lease.manager_session_id != self.manager_session_id or lease.budget_key != self.budget_key:
                return
            ttl_seconds = (lease.valid_until - result.server_now).total_seconds()
            local_deadline = t0 + ttl_seconds - self.budget_config.safety_margin_seconds
            if local_deadline <= now_mono:
                return

            prev = self._active_lease
            prev_deadline = self._lease_deadline_mono
            continuously_valid_same_lease = (
                prev is not None
                and prev_deadline is not None
                and now_mono < prev_deadline
                and prev.lease_id == lease.lease_id
            )
            if continuously_valid_same_lease:
                assert prev is not None
                assert prev_deadline is not None
                if lease.lease_version < prev.lease_version:
                    return
                added_seconds = max(0.0, local_deadline - prev_deadline)
                self._leased_permit_capacity_total += lease.granted_qpm * added_seconds / 60.0
                self._active_lease = lease
                self._lease_deadline_mono = max(prev_deadline, local_deadline)
            else:
                # Either a brand-new lease_id or a late renewal after local expiry: start pacing fresh
                # at now_mono with zero free permits and no back-calculated gap permits.
                self._pacer.reset(now_mono)
                added_seconds = max(0.0, local_deadline - now_mono)
                self._leased_permit_capacity_total += lease.granted_qpm * added_seconds / 60.0
                self._active_lease = lease
                self._lease_deadline_mono = local_deadline
            self._wake_head_locked()
            return

        # Wait / unavailable under enforce:
        # Stop local usage immediately when there is no local demand or when the policy snapshot
        # explicitly denied (allow=False, i.e. any wait reason other than lease contention/rebalance),
        # while the DB still accounts the lease until its original valid_until.
        if (result.status == 'wait' and result.reason_code not in {'budget_occupied', 'lease_rebalance_wait'}) or (
            self._lease_deadline_mono is not None and now_mono >= self._lease_deadline_mono
        ):
            self._active_lease = None
            self._lease_deadline_mono = None

    def _wake_head_locked(self) -> None:
        if self._waiters:
            self._waiters[0].wake()

    def _wake_all_locked(self) -> None:
        for waiter in self._waiters:
            waiter.wake()

    async def _wait_with_custom_sleep(self, event: asyncio.Event, timeout: float) -> None:
        sleep_fn = self._sleep
        assert sleep_fn is not None

        async def _run_sleep() -> None:
            await sleep_fn(timeout)

        event_task = asyncio.create_task(event.wait())
        sleep_task = asyncio.create_task(_run_sleep())
        try:
            await asyncio.wait({event_task, sleep_task}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in (event_task, sleep_task):
                if not task.done():
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task

    async def _register_policy_demand(self) -> None:
        try:
            async with self.session_factory() as session:
                repo = RequestBudgetRepository(session)
                db_now = await repo._read_server_now(self._wall_clock() if self._wall_clock is not None else None)
                await PolicyRepository(session, shard_count=self.shard_count).demand(
                    [self.budget_key],
                    now=db_now,
                )
                await session.commit()
            with self._lock:
                self._last_policy_demand_mono = self._monotonic()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning(
                'Request budget policy demand registration failed: manager=%s key=%s',
                self.manager_id,
                self.budget_key,
                exc_info=True,
            )

    async def _refresh_policy_config_if_available(self) -> None:
        try:
            loaded: PolicyExecutorConfig | None = None
            if self._config_loader is not None:
                loaded = await self._config_loader()
            else:
                loaded = await self._load_policy_config_from_db()
            if loaded is None:
                return
            with self._lock:
                binding = loaded.bindings.get(self.selection.binding)
                if binding is None or binding.kind != 'request_rate' or self.selection.scope not in binding.scopes:
                    # Deleting or invalidating a required binding is a configuration error, not disabled.
                    self._config_invalid = True
                    self._active_lease = None
                    self._lease_deadline_mono = None
                    self._last_reason_code = 'request_policy_binding_missing'
                    self._wake_all_locked()
                    return
                if binding.revision < self._observed_config_revision:
                    return
                self._config_invalid = False
                self._policy_config = loaded
                digest = binding.fingerprint(loaded.imports)
                mode_changed = binding.mode != self._mode or binding.revision > self._observed_config_revision
                self._observed_config_revision = binding.revision
                self._observed_config_hash = digest
                self._mode = binding.mode
                if binding.mode in {'shadow', 'disabled'}:
                    self._active_lease = None
                    self._lease_deadline_mono = None
                    self._last_reason_code = f'policy_{binding.mode}'
                if mode_changed:
                    self._wake_all_locked()
        except asyncio.CancelledError:
            raise
        except Exception:
            # Config check failure keeps the last known mode; never bypasses enforce or blocks shadow/disabled.
            logger.warning('Request budget config refresh failed: manager=%s', self.manager_id, exc_info=True)

    async def _load_policy_config_from_db(self) -> PolicyExecutorConfig | None:
        scope_keys = [
            ('*', '*'),
            ('instance-manager', '*'),
            ('instance-manager', self.manager_id),
        ]
        conditions = [
            and_(
                DistributedConfigOverrideRow.role == role,
                DistributedConfigOverrideRow.identity == identity,
            )
            for role, identity in scope_keys
        ]
        async with self.session_factory() as session:
            rows = (await session.scalars(select(DistributedConfigOverrideRow).where(or_(*conditions)))).all()
        rows_by_scope = {(row.role, row.identity): row for row in rows}
        raw: Any = None
        found = False
        for scope_key in scope_keys:
            row = rows_by_scope.get(scope_key)
            if row is None:
                continue
            if not isinstance(row.config_json, dict):
                raise ValueError(f'distributed config scope {row.role}/{row.identity} must contain an object')
            if 'policy_executor' in row.config_json:
                raw = row.config_json['policy_executor']
                found = True
        if not found or raw is None:
            return None
        return PolicyExecutorConfig.model_validate(raw)

    def _next_sleep_seconds_locked(self, last_result: BudgetLeaseResult | None) -> float:
        now_mono = self._monotonic()
        if self._consecutive_failures > 0:
            backoff = min(
                _FAILURE_BACKOFF_MAX_SECONDS,
                _FAILURE_BACKOFF_BASE_SECONDS * (2 ** min(4, self._consecutive_failures - 1)),
            )
            return backoff * self._rng.uniform(0.9, 1.1)

        if self._mode in {'shadow', 'disabled'}:
            return self.budget_config.lease_seconds * self._rng.uniform(0.9, 1.1)

        has_demand = self._has_recent_demand_locked(now_mono)
        if not has_demand and self._active_lease is None:
            return self.budget_config.lease_seconds

        if self._active_lease is not None and self._lease_deadline_mono is not None:
            remaining = self._lease_deadline_mono - now_mono
            renew_in = max(0.05, remaining - self.budget_config.renew_before_seconds)
            return renew_in * self._rng.uniform(0.85, 1.0)

        if last_result is not None:
            return max(0.05, last_result.retry_after) * self._rng.uniform(0.9, 1.1)
        return 0.5

    async def _coordination_loop(self) -> None:
        last_result: BudgetLeaseResult | None = None
        while True:
            with self._lock:
                if self._closed:
                    return
                wakeup = self._wakeup_event
            assert wakeup is not None
            wakeup.clear()
            last_result = await self.coordinate_once()
            with self._lock:
                if self._closed:
                    return
                sleep_seconds = self._next_sleep_seconds_locked(last_result)
                self._next_coordinate_at_mono = self._monotonic() + sleep_seconds
            while True:
                with self._lock:
                    if self._closed:
                        return
                    now_mono = self._monotonic()
                    remaining_sleep = self._next_coordinate_at_mono - now_mono
                    if remaining_sleep <= 0 or self._should_wake_coordinator_for_demand_locked(now_mono):
                        break
                try:
                    await asyncio.wait_for(wakeup.wait(), timeout=max(0.01, remaining_sleep))
                    wakeup.clear()
                except asyncio.TimeoutError:
                    break


__all__ = [
    'RequestBudgetControl',
    'RequestBudgetStatusSnapshot',
]
