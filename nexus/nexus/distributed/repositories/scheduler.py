from __future__ import annotations

import hashlib
import heapq
import math

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, cast

from sqlalchemy import and_, func, insert, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import load_only

from nexus.distributed.environment.models import (
    EnvironmentAllocationStatus,
    EnvironmentAllocationType,
)
from nexus.distributed.errors import DomainValidationError
from nexus.distributed.execution.models import (
    AssignmentRecord,
    AssignmentStatus,
    ExecutionStatus,
)
from nexus.distributed.instance_manager.liveness import effective_manager_liveness, manager_is_healthy_clause
from nexus.distributed.persistence.orm import (
    AssignmentRow,
    EnvironmentAllocationRow,
    EnvironmentRow,
    ExecutionEventRow,
    ExecutionRow,
    InstanceManagerLivenessRow,
    InstanceManagerRow,
)
from nexus.distributed.policies.models import PolicyExecutorConfig
from nexus.distributed.policies.scheduler import (
    SchedulerPolicyEffect,
    consume_snapshot,
    resolve_selection,
    selection_keys,
)
from nexus.distributed.repositories._records import assignment_record
from nexus.distributed.repositories.policy import PolicyRepository
from nexus.distributed.repositories.quota import AssignmentQuotaBatch, QuotaExceededError, QuotaRepository
from nexus.distributed.repositories.shard_lease import (
    ShardLeaseFenceStatus,
    ShardLeaseRepository,
)
from nexus.distributed.utils.clock import deadline_after
from nexus.distributed.utils.identity import new_id
from nexus.utils.otel.propagation import inject_carrier, use_remote_context
from nexus.utils.otel.transport import transport_span


@dataclass(frozen=True, slots=True)
class SchedulerClaimStats:
    scanned: int = 0
    claimed: int = 0
    assigned: int = 0
    lock_contention: int = 0
    no_capacity: int = 0
    candidate_queries: int = 0


@dataclass(frozen=True, slots=True)
class ManagerCandidate:
    row: InstanceManagerRow
    capacity_total: int
    observed_capacity_used: int
    last_heartbeat_at: datetime

    def __getattr__(self, name: str) -> Any:
        return getattr(self.row, name)


_SCHEDULER_EXECUTION_COLUMNS = (
    ExecutionRow.execution_id,
    ExecutionRow.owner_principal_id,
    ExecutionRow.environment_id,
    ExecutionRow.pool_id,
    ExecutionRow.allocation_id,
    ExecutionRow.workload,
    ExecutionRow.allocation_spec_hash,
    ExecutionRow.first_schedulable_at,
    ExecutionRow.version,
    ExecutionRow.shard_id,
    ExecutionRow.mode,
    ExecutionRow.spec_type,
    ExecutionRow.instance_mode,
    ExecutionRow.placement_json,
    ExecutionRow.trace_carrier_json,
    ExecutionRow.status,
    ExecutionRow.slot_cost,
    ExecutionRow.current_assignment_id,
    ExecutionRow.current_attempt_id,
    ExecutionRow.next_attempt_seq,
    ExecutionRow.attempt_count,
    ExecutionRow.next_event_seq,
    ExecutionRow.next_schedule_at,
    ExecutionRow.last_schedule_error,
    ExecutionRow.queue_ttl_seconds,
    ExecutionRow.queue_expires_at,
    ExecutionRow.updated_at,
)

_SCHEDULER_MANAGER_COLUMNS = (
    InstanceManagerRow.manager_id,
    InstanceManagerRow.session_id,
    InstanceManagerRow.labels_json,
    InstanceManagerRow.draining,
    InstanceManagerRow.owner_principal_id,
    InstanceManagerRow.environment_id,
    InstanceManagerRow.pool_id,
    InstanceManagerRow.allocation_id,
    InstanceManagerRow.workload,
    InstanceManagerRow.allocation_spec_hash,
    InstanceManagerRow.request_policy_binding,
    InstanceManagerRow.request_policy_scope,
    InstanceManagerRow.supported_spec_versions_json,
    InstanceManagerRow.supported_modes_json,
    InstanceManagerRow.supported_spec_types_json,
    InstanceManagerRow.supported_instance_modes_json,
    InstanceManagerRow.capacity_total,
    InstanceManagerRow.observed_capacity_used,
    InstanceManagerRow.last_heartbeat_at,
)


class SchedulerRepository:
    def __init__(
        self,
        session: AsyncSession,
        *,
        shard_count: int,
        manager_healthy_ttl: timedelta,
        no_capacity_backoff: timedelta = timedelta(seconds=3),
        default_user_quota: dict | None = None,
        scan_limit: int = 50,
        max_assignments: int = 50,
        manager_lock_limit: int = 8,
        manager_lock_contention_backoff: timedelta = timedelta(milliseconds=250),
        lease_owner_id: str | None = None,
        lease_shard_group: str = 'default',
        policy_config: PolicyExecutorConfig | None = None,
    ) -> None:
        self.policy_config = policy_config or PolicyExecutorConfig()
        self.policy_demands: set[str] = set()
        self._policy_effects: dict[str, SchedulerPolicyEffect] = {}
        self._policy_events: list[dict[str, Any]] = []
        self.session = session
        self.shard_count = shard_count
        self.manager_healthy_ttl = manager_healthy_ttl
        self.no_capacity_backoff = no_capacity_backoff
        self.scan_limit = scan_limit
        self.max_assignments = max_assignments
        self.manager_lock_limit = manager_lock_limit
        self.manager_lock_contention_backoff = manager_lock_contention_backoff
        self.lease_owner_id = lease_owner_id
        self.lease_shard_group = lease_shard_group
        self._lease_fenced_shard_ids: tuple[int, ...] = ()
        self._lease_fence_lost = False
        self._modified_shard_ids: set[int] = set()
        self._allocation_types: dict[str, str] = {}
        self._execution_spec_versions: dict[str, int | None] = {}
        self._claim_scanned = 0
        self._claim_claimed = 0
        self._candidate_queries = 0
        self._last_claim_stats = SchedulerClaimStats()
        self.quota_repo = (
            QuotaRepository(session, default_limits=default_user_quota) if default_user_quota is not None else None
        )

    async def renew_shard_leases(
        self,
        *,
        owner_id: str,
        shard_group: str,
        shard_ids: list[int],
        lease_ttl: timedelta,
        now: datetime,
    ) -> list[int]:
        return await ShardLeaseRepository(self.session, shard_count=self.shard_count).renew(
            owner_id=owner_id,
            shard_group=shard_group,
            shard_ids=shard_ids,
            lease_ttl=lease_ttl,
            now=now,
        )

    async def claim_and_assign(self, *, shard_ids: list[int], now: datetime):
        assignments = await self.claim_and_assign_batch(shard_ids=shard_ids, now=now, max_assignments=1)
        return assignments[0] if assignments else None

    async def claim_and_assign_batch(
        self,
        *,
        shard_ids: list[int],
        now: datetime,
        scan_limit: int | None = None,
        max_assignments: int | None = None,
    ) -> list[AssignmentRecord]:
        scan_limit = self.scan_limit if scan_limit is None else scan_limit
        max_assignments = self.max_assignments if max_assignments is None else max_assignments
        self._claim_scanned = 0
        self._claim_claimed = 0
        self._candidate_queries = 0
        self._last_claim_stats = SchedulerClaimStats()
        if not shard_ids or scan_limit <= 0 or max_assignments <= 0:
            return []

        self._lease_fence_lost = False
        self._lease_fenced_shard_ids = ()
        self._modified_shard_ids.clear()
        self.policy_demands.clear()
        self._policy_effects.clear()
        self._policy_events.clear()
        candidate_ids = await self._fair_execution_candidate_ids(
            shard_ids=shard_ids,
            now=now,
            scan_limit=scan_limit,
        )
        if not candidate_ids:
            self._publish_claim_stats(assigned=0, lock_contention=0, no_capacity=0)
            return []

        executions = await self._lock_fair_executions(
            candidate_ids=candidate_ids,
            shard_ids=shard_ids,
            now=now,
        )
        if not executions:
            self._publish_claim_stats(assigned=0, lock_contention=0, no_capacity=0)
            return []
        if self.lease_owner_id is not None:
            claimed_shards = sorted({execution.shard_id for execution in executions})
            with self.session.no_autoflush:
                owns_claimed_shards = await ShardLeaseRepository(
                    self.session,
                    shard_count=self.shard_count,
                ).owns(
                    owner_id=self.lease_owner_id,
                    shard_group=self.lease_shard_group,
                    shard_ids=claimed_shards,
                    now=now,
                )
            if not owns_claimed_shards:
                self._lease_fence_lost = True
                await self.session.rollback()
                self._publish_claim_stats(assigned=0, lock_contention=0, no_capacity=0)
                return []

        executions = await self._apply_policy_admission(executions, now=now)
        self._allocation_types = await self._ready_allocation_types(executions)
        quota_batch = (
            await self.quota_repo.prepare_assignment_batch(
                [execution for execution in executions if execution.allocation_id is None],
                now=now,
            )
            if self.quota_repo is not None
            else None
        )

        managers = await self._healthy_managers(now)
        reserved_slots = await self._reserved_slots_by_manager([manager.manager_id for manager in managers])
        owned_shard_ids = set(shard_ids)
        manager_home_shards = self._manager_home_shards(managers)
        manager_lock_ids = self._batch_manager_lock_candidates(
            executions,
            managers,
            reserved_slots,
            owned_shard_ids=owned_shard_ids,
            manager_home_shards=manager_home_shards,
        )
        locked_manager_rows = await self._lock_manager_candidates(
            manager_lock_ids,
            now,
            limit=len(manager_lock_ids),
        )
        locked_managers = {manager.manager_id: manager for manager in locked_manager_rows}
        locked_reserved_slots = await self._reserved_slots_for_locked_managers(list(locked_managers))
        reserved_slots.update({manager_id: locked_reserved_slots.get(manager_id, 0) for manager_id in locked_managers})
        assignments: list[AssignmentRecord] = []
        assignment_events: list[dict[str, Any]] = list(self._policy_events)
        lock_contention = 0
        no_capacity = 0
        for execution in executions:
            if len(assignments) >= max_assignments:
                break
            assignment, manager_lock_contended = await self._assign_to_best_available_manager(
                execution,
                managers,
                reserved_slots,
                locked_managers,
                quota_batch,
                assignment_events,
                now,
                owned_shard_ids=owned_shard_ids,
                manager_home_shards=manager_home_shards,
            )
            if assignment is None:
                if manager_lock_contended:
                    lock_contention += 1
                    execution.next_schedule_at = now + self._manager_lock_retry_delay(execution.execution_id)
                    execution.last_schedule_error = 'manager_lock_contended'
                else:
                    execution.next_schedule_at = now + self.no_capacity_backoff
                if not manager_lock_contended and not (execution.last_schedule_error or '').endswith('_quota_exceeded'):
                    no_capacity += 1
                    execution.last_schedule_error = 'no_capacity'
                execution.updated_at = now
                self._modified_shard_ids.add(execution.shard_id)
                continue

            effect = self._policy_effects.get(execution.execution_id)
            if effect and effect.evidence:
                assignment_events.append(
                    self._policy_event(execution, effect, now=now, event_type='scheduler.policy_applied')
                )
            assignments.append(assignment)
            self._modified_shard_ids.add(execution.shard_id)

        self._lease_fenced_shard_ids = tuple(sorted(self._modified_shard_ids))
        with self.session.no_autoflush:
            lease_is_valid = await self.revalidate_shard_lease_fence(now=now)
        if not lease_is_valid:
            await self.session.rollback()
            self._publish_claim_stats(
                assigned=0,
                lock_contention=lock_contention,
                no_capacity=no_capacity,
            )
            return []
        if assignment_events:
            await self.session.execute(insert(ExecutionEventRow), assignment_events)
        await self.session.flush()
        self._publish_claim_stats(
            assigned=len(assignments),
            lock_contention=lock_contention,
            no_capacity=no_capacity,
        )
        return assignments

    @property
    def lease_fence_lost(self) -> bool:
        return self._lease_fence_lost

    @property
    def last_claim_stats(self) -> SchedulerClaimStats:
        return self._last_claim_stats

    async def revalidate_shard_lease_fence(self, *, now: datetime) -> bool:
        if self.lease_owner_id is None or not self._lease_fenced_shard_ids:
            return not self._lease_fence_lost
        status = await ShardLeaseRepository(self.session, shard_count=self.shard_count).fence(
            owner_id=self.lease_owner_id,
            shard_group=self.lease_shard_group,
            shard_ids=list(self._lease_fenced_shard_ids),
            now=now,
        )
        if status is ShardLeaseFenceStatus.CONTENDED:
            self._lease_fenced_shard_ids = ()
            return False
        if status is ShardLeaseFenceStatus.LOST:
            self._lease_fence_lost = True
            return False
        return True

    async def _fair_execution_candidate_ids(
        self,
        *,
        shard_ids: list[int],
        now: datetime,
        scan_limit: int,
    ) -> list[str]:
        ranked = (
            select(
                ExecutionRow.execution_id.label('execution_id'),
                ExecutionRow.owner_principal_id.label('owner_principal_id'),
                func
                .row_number()
                .over(
                    partition_by=ExecutionRow.owner_principal_id,
                    order_by=(
                        ExecutionRow.next_schedule_at,
                        ExecutionRow.priority.desc(),
                        ExecutionRow.created_at,
                        ExecutionRow.execution_id,
                    ),
                )
                .label('owner_position'),
                func
                .min(ExecutionRow.next_schedule_at)
                .over(partition_by=ExecutionRow.owner_principal_id)
                .label('owner_oldest_schedule_at'),
            )
            .where(
                ExecutionRow.status == ExecutionStatus.QUEUED,
                ExecutionRow.shard_id.in_(shard_ids),
                ExecutionRow.next_schedule_at <= now,
                self._allocation_is_ready(),
            )
            .subquery()
        )
        self._candidate_queries += 1
        candidate_ids = list(
            await self.session.scalars(
                select(ranked.c.execution_id)
                .order_by(
                    ranked.c.owner_position,
                    ranked.c.owner_oldest_schedule_at,
                    func.coalesce(ranked.c.owner_principal_id, ''),
                    ranked.c.execution_id,
                )
                .limit(scan_limit)
            )
        )
        self._claim_scanned = len(candidate_ids)
        return candidate_ids

    async def _lock_fair_executions(
        self,
        *,
        candidate_ids: list[str],
        shard_ids: list[int],
        now: datetime,
    ) -> list[ExecutionRow]:
        candidate_order = {execution_id: position for position, execution_id in enumerate(candidate_ids)}
        self._candidate_queries += 1
        rows_with_versions = (
            await self.session.execute(
                select(
                    ExecutionRow,
                    ExecutionRow.spec_json['spec_version'].as_integer().label('spec_version'),
                )
                .options(load_only(*_SCHEDULER_EXECUTION_COLUMNS))
                .where(
                    ExecutionRow.execution_id.in_(candidate_ids),
                    ExecutionRow.status == ExecutionStatus.QUEUED,
                    ExecutionRow.shard_id.in_(shard_ids),
                    ExecutionRow.next_schedule_at <= now,
                    self._allocation_is_ready(),
                )
                .with_for_update(skip_locked=True)
                .execution_options(populate_existing=True)
            )
        ).all()
        rows = [row for row, _spec_version in rows_with_versions]
        rows.sort(key=lambda row: candidate_order[row.execution_id])
        self._execution_spec_versions = {row.execution_id: spec_version for row, spec_version in rows_with_versions}
        self._claim_claimed = len(rows)
        for row in rows:
            if row.allocation_id is not None and row.first_schedulable_at is None:
                row.first_schedulable_at = now
                row.queue_expires_at = row.queue_expires_at or deadline_after(now, row.queue_ttl_seconds)
                row.updated_at = now
                self._modified_shard_ids.add(row.shard_id)
        return rows

    def _publish_claim_stats(self, *, assigned: int, lock_contention: int, no_capacity: int) -> None:
        self._last_claim_stats = SchedulerClaimStats(
            scanned=self._claim_scanned,
            claimed=self._claim_claimed,
            assigned=assigned,
            lock_contention=lock_contention,
            no_capacity=no_capacity,
            candidate_queries=self._candidate_queries,
        )

    @staticmethod
    def _allocation_is_ready():
        ready_allocation = (
            select(EnvironmentAllocationRow.allocation_id)
            .where(
                EnvironmentAllocationRow.allocation_id == ExecutionRow.allocation_id,
                EnvironmentAllocationRow.status == EnvironmentAllocationStatus.READY,
            )
            .exists()
        )
        return or_(ExecutionRow.allocation_id.is_(None), ready_allocation)

    async def _ready_allocation_types(self, executions: list[ExecutionRow]) -> dict[str, str]:
        allocation_ids = list(
            dict.fromkeys(execution.allocation_id for execution in executions if execution.allocation_id is not None)
        )
        if not allocation_ids:
            return {}
        rows = (
            await self.session.execute(
                select(
                    EnvironmentAllocationRow.allocation_id,
                    EnvironmentAllocationRow.allocation_spec_json,
                    EnvironmentRow.allocation_type,
                )
                .join(EnvironmentRow, EnvironmentRow.environment_id == EnvironmentAllocationRow.environment_id)
                .where(
                    EnvironmentAllocationRow.allocation_id.in_(allocation_ids),
                    EnvironmentAllocationRow.status == EnvironmentAllocationStatus.READY,
                )
            )
        ).all()
        return {
            allocation_id: _allocation_type_from_spec(allocation_spec, fallback)
            for allocation_id, allocation_spec, fallback in rows
        }

    async def manager_endpoints_by_id(self, manager_ids: list[str]) -> dict[str, str]:
        if not manager_ids:
            return {}
        rows = (
            await self.session.execute(
                select(InstanceManagerRow.manager_id, InstanceManagerRow.endpoint_url).where(
                    InstanceManagerRow.manager_id.in_(manager_ids)
                )
            )
        ).all()
        return dict(cast(list[tuple[str, str]], rows))

    async def _healthy_managers(self, now: datetime) -> list[ManagerCandidate]:
        healthy_since = now - self.manager_healthy_ttl
        rows = (
            await self.session.execute(
                select(InstanceManagerRow, InstanceManagerLivenessRow)
                .options(load_only(*_SCHEDULER_MANAGER_COLUMNS))
                .outerjoin(
                    InstanceManagerLivenessRow,
                    and_(
                        InstanceManagerLivenessRow.manager_id == InstanceManagerRow.manager_id,
                        InstanceManagerLivenessRow.session_id == InstanceManagerRow.session_id,
                    ),
                )
                .where(manager_is_healthy_clause(healthy_since))
                .order_by(InstanceManagerRow.manager_id)
            )
        ).all()
        return [self._manager_candidate(manager, liveness) for manager, liveness in rows]

    async def _reserved_slots_by_manager(self, manager_ids: list[str]) -> dict[str, int]:
        return await self._read_reserved_slots_by_manager(manager_ids)

    async def _reserved_slots_for_locked_managers(self, manager_ids: list[str]) -> dict[str, int]:
        return await self._read_reserved_slots_by_manager(manager_ids)

    async def _read_reserved_slots_by_manager(self, manager_ids: list[str]) -> dict[str, int]:
        if not manager_ids:
            return {}
        rows = (
            await self.session.execute(
                select(AssignmentRow.manager_id, func.coalesce(func.sum(AssignmentRow.slot_cost), 0))
                .where(
                    AssignmentRow.manager_id.in_(manager_ids),
                    AssignmentRow.status.in_([AssignmentStatus.PENDING, AssignmentStatus.ACTIVE]),
                )
                .group_by(AssignmentRow.manager_id)
            )
        ).all()
        return {manager_id: int(slot_count or 0) for manager_id, slot_count in rows}

    async def _assign_to_best_available_manager(
        self,
        execution: ExecutionRow,
        managers: list[ManagerCandidate],
        reserved_slots: dict[str, int],
        locked_managers: dict[str, ManagerCandidate],
        quota_batch: AssignmentQuotaBatch | None,
        assignment_events: list[dict[str, Any]],
        now: datetime,
        *,
        owned_shard_ids: set[int],
        manager_home_shards: dict[str, int],
    ) -> tuple[AssignmentRecord | None, bool]:
        if execution.allocation_id is not None and execution.allocation_id not in self._allocation_types:
            return None, False
        locked_pool = self._manager_partition_pool(
            execution,
            list(locked_managers.values()),
            reserved_slots,
            owned_shard_ids=owned_shard_ids,
            manager_home_shards=manager_home_shards,
        )
        candidates = self._candidate_managers(execution, locked_pool, reserved_slots)
        for manager in candidates:
            locked_manager = locked_managers.get(manager.manager_id)
            if locked_manager is None:
                continue
            reserved = reserved_slots[locked_manager.manager_id]
            if not self._manager_can_run(
                execution,
                locked_manager,
                {locked_manager.manager_id: reserved},
                placement=execution.placement_json or {},
            ):
                continue
            if quota_batch is not None and execution.allocation_id is None:
                try:
                    quota_batch.reserve(execution, now=now)
                except QuotaExceededError as exc:
                    execution.last_schedule_error = str(exc)
                    execution.updated_at = now
                    return None, False
            assignment = self._assign_to_manager(execution, locked_manager, assignment_events, now)
            reserved_slots[locked_manager.manager_id] = reserved + execution.slot_cost
            return assignment, False
        candidate_pool = self._manager_partition_pool(
            execution,
            managers,
            reserved_slots,
            owned_shard_ids=owned_shard_ids,
            manager_home_shards=manager_home_shards,
        )
        return None, bool(self._candidate_managers(execution, candidate_pool, reserved_slots, limit=1))

    def _batch_manager_lock_candidates(
        self,
        executions: list[ExecutionRow],
        managers: list[ManagerCandidate],
        reserved_slots: dict[str, int],
        *,
        owned_shard_ids: set[int],
        manager_home_shards: dict[str, int],
    ) -> list[str]:
        if not executions or not managers:
            return []
        lock_budget = min(
            self.manager_lock_limit,
            max(1, math.ceil(len(managers) * len(owned_shard_ids) / self.shard_count)),
        )
        candidates_by_execution: list[list[ManagerCandidate] | None] = [None] * len(executions)
        selected: list[str] = []
        seen: set[str] = set()
        rank = 0
        while len(selected) < lock_budget:
            added = False
            for index, execution in enumerate(executions):
                candidates = candidates_by_execution[index]
                if candidates is None:
                    manager_pool = self._manager_partition_pool(
                        execution,
                        managers,
                        reserved_slots,
                        owned_shard_ids=owned_shard_ids,
                        manager_home_shards=manager_home_shards,
                    )
                    candidates = self._candidate_managers(
                        execution,
                        manager_pool,
                        reserved_slots,
                        limit=lock_budget,
                    )
                    candidates_by_execution[index] = candidates
                if rank >= len(candidates):
                    continue
                added = True
                manager_id = candidates[rank].manager_id
                if manager_id in seen:
                    continue
                seen.add(manager_id)
                selected.append(manager_id)
                if len(selected) >= lock_budget:
                    break
            if not added:
                break
            rank += 1
        return selected

    def _manager_home_shards(self, managers: list[ManagerCandidate]) -> dict[str, int]:
        managers_by_scope: dict[tuple[str, ...], list[ManagerCandidate]] = {}
        for manager in managers:
            managers_by_scope.setdefault(self._manager_partition_scope(manager), []).append(manager)

        home_shards: dict[str, int] = {}
        for scope, scoped_managers in managers_by_scope.items():
            ordered = sorted(
                scoped_managers,
                key=lambda manager: (
                    self._manager_partition_order(scope, manager.manager_id),
                    manager.manager_id,
                ),
            )
            for rank, manager in enumerate(ordered):
                home_shards[manager.manager_id] = rank % self.shard_count
        return home_shards

    @staticmethod
    def _manager_partition_scope(manager: ManagerCandidate) -> tuple[str, ...]:
        if manager.allocation_id:
            return ('allocation', manager.allocation_id)
        return (
            'pool',
            manager.environment_id,
            manager.pool_id,
            manager.workload,
            manager.request_policy_binding or '',
            manager.request_policy_scope or '',
        )

    @staticmethod
    def _manager_partition_order(scope: tuple[str, ...], manager_id: str) -> int:
        scope_key = '\0'.join(scope)
        digest = hashlib.blake2s(f'{scope_key}\0{manager_id}'.encode(), digest_size=8).digest()
        return int.from_bytes(digest, 'big')

    def _manager_partition_pool(
        self,
        execution: ExecutionRow,
        managers: list[ManagerCandidate],
        reserved_slots: dict[str, int],
        *,
        owned_shard_ids: set[int],
        manager_home_shards: dict[str, int],
    ) -> list[ManagerCandidate]:
        placement = execution.placement_json or {}
        if placement.get('target_manager_id') or placement.get('preferred_manager_id'):
            return managers
        partition = [manager for manager in managers if manager_home_shards.get(manager.manager_id) in owned_shard_ids]
        if any(self._manager_can_run(execution, manager, reserved_slots, placement=placement) for manager in partition):
            return partition
        return managers

    def _manager_lock_retry_delay(self, execution_id: str) -> timedelta:
        jitter = self._stable_manager_jitter(execution_id, 'manager-lock-contention') / ((1 << 64) - 1)
        return self.manager_lock_contention_backoff * (1 + 2 * jitter)

    async def _lock_manager_candidates(
        self,
        manager_ids: list[str],
        now: datetime,
        *,
        limit: int,
    ) -> list[ManagerCandidate]:
        if not manager_ids or limit <= 0:
            return []
        healthy_since = now - self.manager_healthy_ttl
        locked_ids = list(
            await self.session.scalars(
                select(InstanceManagerRow.manager_id)
                .where(InstanceManagerRow.manager_id.in_(manager_ids))
                .order_by(InstanceManagerRow.manager_id)
                .limit(limit)
                .with_for_update(skip_locked=True)
            )
        )
        if not locked_ids:
            return []
        rows = (
            await self.session.execute(
                select(InstanceManagerRow, InstanceManagerLivenessRow)
                .options(load_only(*_SCHEDULER_MANAGER_COLUMNS))
                .outerjoin(
                    InstanceManagerLivenessRow,
                    and_(
                        InstanceManagerLivenessRow.manager_id == InstanceManagerRow.manager_id,
                        InstanceManagerLivenessRow.session_id == InstanceManagerRow.session_id,
                    ),
                )
                .where(
                    InstanceManagerRow.manager_id.in_(locked_ids),
                    manager_is_healthy_clause(healthy_since),
                )
                .order_by(InstanceManagerRow.manager_id)
            )
        ).all()
        return [self._manager_candidate(manager, liveness) for manager, liveness in rows]

    @staticmethod
    def _manager_candidate(
        manager: InstanceManagerRow,
        liveness: InstanceManagerLivenessRow | None,
    ) -> ManagerCandidate:
        effective = effective_manager_liveness(manager, liveness)
        return ManagerCandidate(
            row=manager,
            capacity_total=effective.capacity_total,
            observed_capacity_used=effective.observed_capacity_used,
            last_heartbeat_at=effective.last_heartbeat_at,
        )

    def _candidate_managers(self, execution, managers, reserved_slots, *, limit=None):
        effect = self._policy_effects.get(execution.execution_id)
        if effect is None or not effect.preferred_managers:
            return self._default_candidate_managers(execution, managers, reserved_slots, limit=limit)
        candidates = self._default_candidate_managers(execution, managers, reserved_slots)
        if effect and effect.preferred_managers:
            ranks = {manager_id: rank for rank, manager_id in enumerate(effect.preferred_managers)}
            placement = execution.placement_json or {}
            preferred = placement.get('preferred_manager_id')
            candidates = sorted(
                candidates,
                key=lambda manager: (
                    manager.manager_id != preferred,
                    ranks.get(manager.manager_id, len(ranks)),
                ),
            )
        return candidates if limit is None else candidates[:limit]

    async def _apply_policy_admission(self, executions, *, now):
        selections = {}
        for execution in executions:
            requested = (execution.placement_json or {}).get('scheduler_policy')
            try:
                selection = resolve_selection(
                    self.policy_config,
                    requested,
                    environment_id=execution.environment_id or (execution.placement_json or {}).get('environment', ''),
                    owner_principal_id=execution.owner_principal_id,
                    spec_type=execution.spec_type,
                )
                if selection is not None:
                    selections[execution.execution_id] = selection
                    self.policy_demands.update(selection_keys(selection, self.policy_config))
            except (ValueError, DomainValidationError):
                self._policy_effects[execution.execution_id] = SchedulerPolicyEffect(
                    allow=False,
                    reason='policy_invalid',
                    evidence={'reason_code': 'policy_invalid'},
                )
        rows = await PolicyRepository(self.session, shard_count=self.shard_count).read_scopes(
            sorted(self.policy_demands)
        )
        fingerprints = {
            selection.binding: self.policy_config.bindings[selection.binding].fingerprint(self.policy_config.imports)
            for selection in {value.binding: value for value in selections.values()}.values()
        }
        retained = []
        for execution in executions:
            selection = selections.get(execution.execution_id)
            if selection is not None:
                self._policy_effects[execution.execution_id] = consume_snapshot(
                    selection, self.policy_config, rows, now=now, fingerprint=fingerprints[selection.binding]
                )
            effect = self._policy_effects.get(execution.execution_id)
            if effect is None or effect.allow:
                if (execution.last_schedule_error or '').startswith('policy_deferred:'):
                    execution.last_schedule_error = None
                    execution.updated_at = now
                    self._modified_shard_ids.add(execution.shard_id)
                retained.append(execution)
                continue
            reason = 'policy_deferred:' + effect.reason
            if execution.last_schedule_error != reason:
                self._policy_events.append(
                    self._policy_event(execution, effect, now=now, event_type='scheduler.policy_deferred')
                )
            execution.last_schedule_error = reason
            execution.next_schedule_at = now + timedelta(seconds=effect.delay_seconds)
            execution.updated_at = now
            self._modified_shard_ids.add(execution.shard_id)
        return retained

    @staticmethod
    def _policy_event(execution, effect, *, now, event_type):
        seq = execution.next_event_seq
        execution.next_event_seq += 1
        return {
            'execution_id': execution.execution_id,
            'seq': seq,
            'assignment_id': execution.current_assignment_id,
            'attempt_id': execution.current_attempt_id,
            'type': event_type,
            'source': 'scheduler',
            'visibility': 'public',
            'payload_json': effect.evidence,
            'event_key': None,
            'created_at': now,
        }

    def policy_snapshots_valid(self, *, now):
        return all(
            effect.valid_until is None or not effect.allow or effect.valid_until > now
            for effect in self._policy_effects.values()
        )

    def _default_candidate_managers(
        self,
        execution: ExecutionRow,
        managers: list[ManagerCandidate],
        reserved_slots: dict[str, int],
        *,
        limit: int | None = None,
    ) -> list[ManagerCandidate]:
        if limit is not None and limit <= 0:
            return []
        placement = execution.placement_json or {}
        managers_by_id = {manager.manager_id: manager for manager in managers}
        if target_manager_id := placement.get('target_manager_id'):
            manager = managers_by_id.get(target_manager_id)
            if manager is not None and self._manager_can_run(execution, manager, reserved_slots):
                return [manager]
            return []

        if preferred_manager_id := placement.get('preferred_manager_id'):
            manager = managers_by_id.get(preferred_manager_id)
            if manager is not None and self._manager_can_run(execution, manager, reserved_slots, placement=placement):
                remaining = self._managers_can_run(
                    execution,
                    [item for item in managers if item.manager_id != preferred_manager_id],
                    reserved_slots,
                    placement=placement,
                )
                remaining_limit = None if limit is None else limit - 1
                return [
                    manager,
                    *self._sort_managers_by_load(
                        execution,
                        remaining,
                        reserved_slots,
                        limit=remaining_limit,
                    ),
                ]

        return self._sort_managers_by_load(
            execution,
            self._managers_can_run(execution, managers, reserved_slots, placement=placement),
            reserved_slots,
            limit=limit,
        )

    def _sort_managers_by_load(
        self,
        execution: ExecutionRow,
        managers: list[ManagerCandidate],
        reserved_slots: dict[str, int],
        *,
        limit: int | None = None,
    ) -> list[ManagerCandidate]:
        def key(manager: ManagerCandidate) -> tuple[float, int, int, str]:
            return self._manager_load_score(execution, manager, reserved_slots)

        if limit is None or limit >= len(managers):
            return sorted(managers, key=key)
        return heapq.nsmallest(limit, managers, key=key)

    def _manager_load_score(
        self,
        execution: ExecutionRow,
        manager: ManagerCandidate,
        reserved_slots: dict[str, int],
    ) -> tuple[float, int, int, str]:
        post_assign_reserved = reserved_slots.get(manager.manager_id, 0) + execution.slot_cost
        post_assign_load_ratio = post_assign_reserved / manager.capacity_total
        return (
            post_assign_load_ratio,
            post_assign_reserved,
            self._stable_manager_jitter(execution.execution_id, manager.manager_id),
            manager.manager_id,
        )

    @staticmethod
    def _stable_manager_jitter(execution_id: str, manager_id: str) -> int:
        digest = hashlib.blake2s(f'{execution_id}\0{manager_id}'.encode(), digest_size=8).digest()
        return int.from_bytes(digest, 'big')

    def _managers_can_run(
        self,
        execution: ExecutionRow,
        managers: list[ManagerCandidate],
        reserved_slots: dict[str, int],
        *,
        placement: dict | None = None,
    ) -> list[ManagerCandidate]:
        return [
            manager
            for manager in managers
            if self._manager_can_run(execution, manager, reserved_slots, placement=placement)
        ]

    def _manager_can_run(
        self,
        execution: ExecutionRow,
        manager: ManagerCandidate,
        reserved_slots: dict[str, int],
        *,
        placement: dict | None = None,
    ) -> bool:
        if manager.draining:
            return False
        if not self._manager_matches_allocation(execution, manager):
            return False
        if not self._manager_matches_request_policy(execution, manager):
            return False
        if placement is not None and not self._manager_matches_placement(manager, placement):
            return False
        if execution.mode not in manager.supported_modes_json:
            return False
        if execution.spec_type not in manager.supported_spec_types_json:
            return False
        if execution.instance_mode not in manager.supported_instance_modes_json:
            return False
        supported_versions = manager.supported_spec_versions_json or {}
        if execution.execution_id in self._execution_spec_versions:
            spec_version = self._execution_spec_versions[execution.execution_id]
        else:
            spec_version = (execution.spec_json or {}).get('spec_version')
        if (
            spec_version is not None
            and supported_versions
            and spec_version not in supported_versions.get(execution.spec_type, [])
        ):
            return False
        return manager.capacity_total - reserved_slots.get(manager.manager_id, 0) >= execution.slot_cost

    def _manager_matches_allocation(self, execution: ExecutionRow, manager: ManagerCandidate) -> bool:
        execution_workload = execution.workload or (
            'orchestration' if execution.spec_type == 'nexus_batch' else 'execution'
        )
        if execution_workload != manager.workload:
            return False
        if execution.allocation_id is None:
            return True

        allocation_type = self._allocation_types.get(execution.allocation_id)
        if allocation_type == EnvironmentAllocationType.DYNAMIC:
            return (
                manager.allocation_id == execution.allocation_id
                and manager.environment_id == execution.environment_id
                and manager.pool_id == execution.pool_id
                and manager.allocation_spec_hash == execution.allocation_spec_hash
                and manager.owner_principal_id == execution.owner_principal_id
            )
        if allocation_type == EnvironmentAllocationType.PREDEFINED:
            return (
                manager.allocation_id is None
                and manager.environment_id == execution.environment_id
                and manager.pool_id == execution.pool_id
            )
        return False

    @staticmethod
    def _manager_matches_request_policy(execution: ExecutionRow, manager: ManagerCandidate) -> bool:
        if execution.spec_type == 'nexus_batch':
            return manager.request_policy_binding is None and manager.request_policy_scope is None
        placement = execution.placement_json or {}
        requested = placement.get('request_policy')
        if isinstance(requested, dict):
            expected_binding = requested.get('binding')
            expected_scope = requested.get('scope')
        else:
            expected_binding = None
            expected_scope = None
        return manager.request_policy_binding == expected_binding and manager.request_policy_scope == expected_scope

    @staticmethod
    def _manager_matches_placement(manager: ManagerCandidate, placement: dict) -> bool:
        if (environment_id := placement.get('environment')) and manager.environment_id != environment_id:
            return False
        if (pool_id := placement.get('pool')) and manager.pool_id != pool_id:
            return False
        manager_labels = manager.labels_json or {}
        return all(manager_labels.get(key) == value for key, value in (placement.get('labels') or {}).items())

    def _assign_to_manager(
        self,
        execution: ExecutionRow,
        manager: ManagerCandidate,
        assignment_events: list[dict[str, Any]],
        now: datetime,
    ) -> AssignmentRecord:
        attempt_seq = execution.next_attempt_seq
        assignment_id = new_id('asn')
        attempt_id = f'{execution.execution_id}-attempt-{attempt_seq}'
        with use_remote_context(execution.trace_carrier_json, fallback_only=False):
            with transport_span(
                'distributed.scheduler.assign',
                kind='producer',
                attributes={
                    'nexus.execution.id': execution.execution_id,
                    'nexus.assignment.id': assignment_id,
                    'nexus.attempt.id': attempt_id,
                    'nexus.manager.id': manager.manager_id,
                },
            ):
                assignment = AssignmentRow(
                    assignment_id=assignment_id,
                    execution_id=execution.execution_id,
                    owner_principal_id=execution.owner_principal_id,
                    environment_id=execution.environment_id,
                    pool_id=execution.pool_id,
                    allocation_id=execution.allocation_id,
                    workload=execution.workload,
                    allocation_spec_hash=execution.allocation_spec_hash,
                    attempt_id=attempt_id,
                    attempt_seq=attempt_seq,
                    manager_id=manager.manager_id,
                    manager_session_id=manager.session_id,
                    status=AssignmentStatus.PENDING,
                    slot_cost=execution.slot_cost,
                    endpoint_ready=False,
                    trace_carrier_json=inject_carrier() or None,
                    created_at=now,
                    updated_at=now,
                )
                self.session.add(assignment)
                execution.status = ExecutionStatus.ASSIGNED
                execution.version += 1
                execution.current_assignment_id = assignment.assignment_id
                execution.current_attempt_id = assignment.attempt_id
                execution.next_attempt_seq += 1
                execution.attempt_count += 1
                execution.last_schedule_error = None
                execution.updated_at = now
                self._append_assignment_events(execution, assignment, manager, assignment_events, now)
        record = assignment_record(assignment)
        assert record is not None
        return record

    def _append_assignment_events(
        self,
        execution: ExecutionRow,
        assignment: AssignmentRow,
        manager: ManagerCandidate,
        assignment_events: list[dict[str, Any]],
        now: datetime,
    ) -> None:
        first_seq = execution.next_event_seq
        execution.next_event_seq += 2
        assignment_events.extend([
            {
                'execution_id': execution.execution_id,
                'seq': first_seq,
                'assignment_id': assignment.assignment_id,
                'attempt_id': assignment.attempt_id,
                'type': 'assignment.status.changed',
                'source': 'scheduler',
                'visibility': 'public',
                'payload_json': {'from': None, 'to': AssignmentStatus.PENDING},
                'event_key': None,
                'created_at': now,
            },
            {
                'execution_id': execution.execution_id,
                'seq': first_seq + 1,
                'assignment_id': assignment.assignment_id,
                'attempt_id': assignment.attempt_id,
                'type': 'scheduler.assigned',
                'source': 'scheduler',
                'visibility': 'public',
                'payload_json': {
                    'manager_id': manager.manager_id,
                    'manager_session_id': manager.session_id,
                    'assignment_id': assignment.assignment_id,
                    'attempt_id': assignment.attempt_id,
                },
                'event_key': None,
                'created_at': now,
            },
        ])


def _allocation_type_from_spec(allocation_spec: object, fallback: str) -> str:
    if isinstance(allocation_spec, dict):
        value = cast(dict[str, object], allocation_spec).get('allocation_type')
        if isinstance(value, str) and value in {
            EnvironmentAllocationType.DYNAMIC,
            EnvironmentAllocationType.PREDEFINED,
        }:
            return value
    return fallback
