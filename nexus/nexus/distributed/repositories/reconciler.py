from __future__ import annotations

import logging

from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy import and_, delete, or_, select, true, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased, load_only

from nexus.distributed.environment.liveness import allocation_liveness
from nexus.distributed.execution.models import (
    AssignmentStatus,
    CommandStatus,
    ExecutionStatus,
    RouteStatus,
)
from nexus.distributed.instance_manager.liveness import (
    allocation_heartbeat_cutoff,
    effective_manager_liveness,
    manager_is_dead_clause,
)
from nexus.distributed.observability.database import queue_committed_metric
from nexus.distributed.persistence.orm import (
    AssignmentRow,
    BatchRunRow,
    EnvironmentAllocationRow,
    ExecutionCommandRow,
    ExecutionCredentialRow,
    ExecutionRow,
    InstanceManagerLivenessRow,
    InstanceManagerRow,
)
from nexus.distributed.repositories.events import append_control_event_row
from nexus.distributed.repositories.shard_lease import ShardLeaseFenceStatus, ShardLeaseRepository
from nexus.distributed.utils.clock import deadline_after, stop_deadline_after
from nexus.distributed.utils.identity import new_id


logger = logging.getLogger(__name__)


TERMINAL_STATUSES_FOR_CLEANUP = {
    ExecutionStatus.SUCCEEDED,
    ExecutionStatus.FAILED,
    ExecutionStatus.KILLED,
    ExecutionStatus.LOST,
    ExecutionStatus.EXPIRED,
}
ACTIVE_EXECUTION_STATUSES = {
    ExecutionStatus.ASSIGNED,
    ExecutionStatus.STARTING,
    ExecutionStatus.RUNNING,
    ExecutionStatus.WAITING_INPUT,
}

_RECONCILER_EXECUTION_COLUMNS = (
    ExecutionRow.execution_id,
    ExecutionRow.environment_id,
    ExecutionRow.parent_execution_id,
    ExecutionRow.first_schedulable_at,
    ExecutionRow.stop_reason,
    ExecutionRow.version,
    ExecutionRow.shard_id,
    ExecutionRow.mode,
    ExecutionRow.spec_type,
    ExecutionRow.status,
    ExecutionRow.route_status,
    ExecutionRow.current_assignment_id,
    ExecutionRow.current_attempt_id,
    ExecutionRow.attempt_count,
    ExecutionRow.max_attempts,
    ExecutionRow.next_event_seq,
    ExecutionRow.next_schedule_at,
    ExecutionRow.queue_ttl_seconds,
    ExecutionRow.execution_ttl_seconds,
    ExecutionRow.stop_timeout_seconds,
    ExecutionRow.queue_expires_at,
    ExecutionRow.execution_expires_at,
    ExecutionRow.stop_deadline_at,
    ExecutionRow.created_at,
    ExecutionRow.updated_at,
    ExecutionRow.started_at,
    ExecutionRow.finished_at,
    ExecutionRow.cleanup_done_at,
)

_RECONCILER_ASSIGNMENT_COLUMNS = (
    AssignmentRow.assignment_id,
    AssignmentRow.execution_id,
    AssignmentRow.attempt_id,
    AssignmentRow.manager_id,
    AssignmentRow.manager_session_id,
    AssignmentRow.status,
    AssignmentRow.endpoint_ready,
    AssignmentRow.accepted_at,
    AssignmentRow.started_at,
    AssignmentRow.finished_at,
    AssignmentRow.created_at,
    AssignmentRow.updated_at,
)

_RECONCILER_MANAGER_COLUMNS = (
    InstanceManagerRow.allocation_id,
    InstanceManagerRow.manager_id,
    InstanceManagerRow.environment_id,
    InstanceManagerRow.session_id,
    InstanceManagerRow.capacity_total,
    InstanceManagerRow.observed_capacity_used,
    InstanceManagerRow.last_heartbeat_at,
)


def _can_retry_assignment_loss(execution: ExecutionRow, *, started: bool) -> bool:
    return execution.attempt_count < execution.max_attempts and (not started or execution.spec_type == 'nexus_batch')


@dataclass(frozen=True, slots=True)
class _LockedExecutionAssignment:
    execution: ExecutionRow
    assignment: AssignmentRow | None


@dataclass(frozen=True, slots=True)
class ReconcilerStats:
    scanned: int = 0
    claimed: int = 0
    changed: int = 0
    overdue_lag_seconds: float = 0


class ReconcilerRepository:
    def __init__(
        self,
        session: AsyncSession,
        *,
        manager_dead_ttl: timedelta = timedelta(seconds=45),
        assign_ack_timeout: timedelta = timedelta(seconds=30),
        start_timeout: timedelta = timedelta(seconds=120),
        batch_size: int = 500,
        shard_ids: list[int] | None = None,
        shard_count: int = 64,
        lease_owner_id: str | None = None,
        lease_shard_group: str = 'execution-reconciler',
    ) -> None:
        self.session = session
        self.manager_dead_ttl = manager_dead_ttl
        self.assign_ack_timeout = assign_ack_timeout
        self.start_timeout = start_timeout
        self.batch_size = batch_size
        self.shard_ids = tuple(sorted(set(shard_ids))) if shard_ids is not None else None
        self.shard_count = shard_count
        self.lease_owner_id = lease_owner_id
        self.lease_shard_group = lease_shard_group
        self._claimed_shard_ids: set[int] = set()
        self._lease_fence_lost = False
        self._last_stats = ReconcilerStats()

    @property
    def last_stats(self) -> ReconcilerStats:
        return self._last_stats

    @property
    def lease_fence_lost(self) -> bool:
        return self._lease_fence_lost

    async def finalize_shard_fence(self, *, now: datetime) -> bool:
        if self.lease_owner_id is None or not self._claimed_shard_ids:
            return True
        status = await ShardLeaseRepository(self.session, shard_count=self.shard_count).fence(
            owner_id=self.lease_owner_id,
            shard_group=self.lease_shard_group,
            shard_ids=sorted(self._claimed_shard_ids),
            now=now,
        )
        if status is not ShardLeaseFenceStatus.VALID:
            self._lease_fence_lost = status is ShardLeaseFenceStatus.LOST
            await self.session.rollback()
            return False
        return True

    def _owned_shard_clause(self):
        if self.shard_ids is None:
            return true()
        return ExecutionRow.shard_id.in_(self.shard_ids)

    def _track_execution_rows(self, rows) -> None:
        self._claimed_shard_ids.update(row.shard_id for row in rows)

    async def reconcile_manager_dead(self, *, now: datetime) -> int:
        cutoff = allocation_heartbeat_cutoff(
            now=now,
            default_ttl=self.manager_dead_ttl,
            dialect=self.session.get_bind().dialect.name,
        )
        candidates = list(
            (
                await self.session.execute(
                    select(AssignmentRow.execution_id, AssignmentRow.assignment_id)
                    .join(ExecutionRow, ExecutionRow.execution_id == AssignmentRow.execution_id)
                    .outerjoin(InstanceManagerRow, AssignmentRow.manager_id == InstanceManagerRow.manager_id)
                    .outerjoin(
                        InstanceManagerLivenessRow,
                        and_(
                            InstanceManagerLivenessRow.manager_id == InstanceManagerRow.manager_id,
                            InstanceManagerLivenessRow.session_id == InstanceManagerRow.session_id,
                        ),
                    )
                    .outerjoin(
                        EnvironmentAllocationRow,
                        EnvironmentAllocationRow.allocation_id == InstanceManagerRow.allocation_id,
                    )
                    .where(
                        AssignmentRow.status.in_([AssignmentStatus.PENDING, AssignmentStatus.ACTIVE]),
                        self._owned_shard_clause(),
                        manager_is_dead_clause(cutoff),
                    )
                    .order_by(AssignmentRow.updated_at, AssignmentRow.execution_id, AssignmentRow.assignment_id)
                    .limit(self.batch_size)
                )
            ).all()
        )
        pairs = await self._lock_assignment_candidates(candidates)
        managers = await self._lock_managers([
            pair.assignment.manager_id for pair in pairs if pair.assignment is not None
        ])
        liveness_by_manager = await self._manager_liveness_by_id(list(managers))
        allocation_ids = {manager.allocation_id for manager in managers.values() if manager.allocation_id is not None}
        specs = {}
        if allocation_ids:
            rows = await self.session.execute(
                select(EnvironmentAllocationRow.allocation_id, EnvironmentAllocationRow.allocation_spec_json).where(
                    EnvironmentAllocationRow.allocation_id.in_(allocation_ids)
                )
            )
            specs = dict(rows.tuples().all())
        changed = 0
        for pair in pairs:
            assignment = pair.assignment
            if not self._is_current_active_pair(pair):
                continue
            assert assignment is not None
            manager = managers.get(assignment.manager_id)
            if manager is not None:
                liveness = liveness_by_manager.get(manager.manager_id)
                effective = effective_manager_liveness(manager, liveness)
                threshold = now - self.manager_dead_ttl
                if effective.active:
                    if manager.allocation_id is not None:
                        spec = specs.get(manager.allocation_id)
                        try:
                            if spec is None:
                                raise ValueError('missing_allocation')
                            policy = allocation_liveness(spec)
                        except ValueError:
                            logger.error('Invalid Manager liveness policy: allocation_id=%s', manager.allocation_id)
                            continue
                        if policy is not None:
                            threshold = now - timedelta(seconds=policy.dead_timeout_seconds)
                    if effective.last_heartbeat_at > threshold:
                        continue
                if effective.from_liveness_table and effective.active:
                    result = await self.session.execute(
                        update(InstanceManagerLivenessRow)
                        .where(
                            InstanceManagerLivenessRow.manager_id == manager.manager_id,
                            InstanceManagerLivenessRow.session_id == manager.session_id,
                            InstanceManagerLivenessRow.fenced_at.is_(None),
                            InstanceManagerLivenessRow.last_heartbeat_at <= threshold,
                        )
                        .values(fenced_at=now, updated_at=now)
                    )
                    if int(getattr(result, 'rowcount', 0) or 0) != 1:
                        continue
                    queue_committed_metric(
                        self.session,
                        'instance_exits',
                        kind='manager',
                        reason='heartbeat_expired',
                        environment=manager.environment_id or 'all',
                    )
            changed += int(
                await self._mark_lost_or_retry(
                    assignment,
                    execution=pair.execution,
                    now=now,
                )
            )
        await self.session.flush()
        self._publish_stats(scanned=len(candidates), claimed=len(pairs), changed=changed)
        return changed

    async def reconcile_session_mismatch(self, *, now: datetime) -> int:
        candidates = list(
            (
                await self.session.execute(
                    select(AssignmentRow.execution_id, AssignmentRow.assignment_id)
                    .join(ExecutionRow, ExecutionRow.execution_id == AssignmentRow.execution_id)
                    .join(InstanceManagerRow, AssignmentRow.manager_id == InstanceManagerRow.manager_id)
                    .where(
                        AssignmentRow.status.in_([AssignmentStatus.PENDING, AssignmentStatus.ACTIVE]),
                        self._owned_shard_clause(),
                        InstanceManagerRow.session_id != AssignmentRow.manager_session_id,
                    )
                    .order_by(AssignmentRow.updated_at, AssignmentRow.execution_id, AssignmentRow.assignment_id)
                    .limit(self.batch_size)
                )
            ).all()
        )
        pairs = await self._lock_assignment_candidates(candidates)
        managers = await self._lock_managers([
            pair.assignment.manager_id for pair in pairs if pair.assignment is not None
        ])
        changed = 0
        for pair in pairs:
            assignment = pair.assignment
            if not self._is_current_active_pair(pair):
                continue
            assert assignment is not None
            manager = managers.get(assignment.manager_id)
            if manager is None or manager.session_id == assignment.manager_session_id:
                continue
            changed += int(
                await self._mark_lost_or_retry(
                    assignment,
                    execution=pair.execution,
                    now=now,
                )
            )
        await self.session.flush()
        self._publish_stats(scanned=len(candidates), claimed=len(pairs), changed=changed)
        return changed

    async def reconcile_pending_timeouts(self, *, now: datetime) -> int:
        threshold = now - self.assign_ack_timeout
        candidates = list(
            (
                await self.session.execute(
                    select(AssignmentRow.execution_id, AssignmentRow.assignment_id)
                    .join(ExecutionRow, ExecutionRow.execution_id == AssignmentRow.execution_id)
                    .where(
                        AssignmentRow.status == AssignmentStatus.PENDING,
                        self._owned_shard_clause(),
                        AssignmentRow.created_at <= threshold,
                    )
                    .order_by(AssignmentRow.created_at, AssignmentRow.execution_id, AssignmentRow.assignment_id)
                    .limit(self.batch_size)
                )
            ).all()
        )
        pairs = await self._lock_assignment_candidates(candidates)
        changed = 0
        for pair in pairs:
            assignment = pair.assignment
            if (
                not self._is_current_active_pair(pair)
                or assignment is None
                or assignment.status != AssignmentStatus.PENDING
                or assignment.created_at > threshold
            ):
                continue
            changed += int(
                await self._mark_lost_or_retry(
                    assignment,
                    execution=pair.execution,
                    now=now,
                    assignment_status=AssignmentStatus.REVOKED,
                    reason='assignment_not_acknowledged',
                )
            )
        await self.session.flush()
        self._publish_stats(
            scanned=len(candidates),
            claimed=len(pairs),
            changed=changed,
            now=now,
            deadlines=(
                pair.assignment.created_at + self.assign_ack_timeout for pair in pairs if pair.assignment is not None
            ),
        )
        return changed

    async def reconcile_starting_timeouts(self, *, now: datetime) -> int:
        threshold = now - self.start_timeout
        candidates = list(
            (
                await self.session.execute(
                    select(AssignmentRow.execution_id, AssignmentRow.assignment_id)
                    .join(ExecutionRow, ExecutionRow.execution_id == AssignmentRow.execution_id)
                    .where(
                        AssignmentRow.status == AssignmentStatus.ACTIVE,
                        self._owned_shard_clause(),
                        AssignmentRow.accepted_at <= threshold,
                        AssignmentRow.started_at.is_(None),
                    )
                    .order_by(AssignmentRow.accepted_at, AssignmentRow.execution_id, AssignmentRow.assignment_id)
                    .limit(self.batch_size)
                )
            ).all()
        )
        pairs = await self._lock_assignment_candidates(candidates)
        changed = 0
        for pair in pairs:
            assignment = pair.assignment
            if (
                not self._is_current_active_pair(pair)
                or assignment is None
                or assignment.status != AssignmentStatus.ACTIVE
                or assignment.accepted_at is None
                or assignment.accepted_at > threshold
                or assignment.started_at is not None
            ):
                continue
            changed += int(
                await self._mark_lost_or_retry(
                    assignment,
                    execution=pair.execution,
                    now=now,
                )
            )
        await self.session.flush()
        self._publish_stats(
            scanned=len(candidates),
            claimed=len(pairs),
            changed=changed,
            now=now,
            deadlines=(
                pair.assignment.accepted_at + self.start_timeout
                for pair in pairs
                if pair.assignment is not None and pair.assignment.accepted_at is not None
            ),
        )
        return changed

    async def repair_active_assignments(self, *, now: datetime) -> int:
        candidates = list(
            await self.session.scalars(
                select(ExecutionRow.execution_id)
                .outerjoin(AssignmentRow, ExecutionRow.current_assignment_id == AssignmentRow.assignment_id)
                .where(
                    ExecutionRow.status.in_(ACTIVE_EXECUTION_STATUSES),
                    self._owned_shard_clause(),
                    or_(
                        ExecutionRow.current_assignment_id.is_(None),
                        AssignmentRow.assignment_id.is_(None),
                        AssignmentRow.status.notin_([AssignmentStatus.PENDING, AssignmentStatus.ACTIVE]),
                        ExecutionRow.current_attempt_id.is_(None),
                        ExecutionRow.current_attempt_id != AssignmentRow.attempt_id,
                    ),
                )
                .order_by(ExecutionRow.updated_at, ExecutionRow.execution_id)
                .limit(self.batch_size)
            )
        )
        pairs = await self._lock_execution_candidates(candidates)
        changed = 0
        for pair in pairs:
            if pair.execution.status not in ACTIVE_EXECUTION_STATUSES or self._is_current_active_pair(pair):
                continue
            changed += int(await self._repair_active_pair(pair, now=now))
        await self.session.flush()
        self._publish_stats(scanned=len(candidates), claimed=len(pairs), changed=changed)
        return changed

    async def expire_queued(self, *, now: datetime) -> int:
        await self._backfill_queue_deadlines()
        candidates = list(
            await self.session.scalars(
                select(ExecutionRow.execution_id)
                .where(
                    ExecutionRow.status == ExecutionStatus.QUEUED,
                    self._owned_shard_clause(),
                    ExecutionRow.queue_expires_at <= now,
                )
                .order_by(ExecutionRow.queue_expires_at, ExecutionRow.execution_id)
                .limit(self.batch_size)
            )
        )
        pairs = await self._lock_execution_candidates(candidates)
        changed = 0
        for pair in pairs:
            execution = pair.execution
            if execution.status != ExecutionStatus.QUEUED:
                continue
            if execution.queue_expires_at is None or execution.queue_expires_at > now:
                continue
            old_status = execution.status
            old_route_status = execution.route_status
            execution.status = ExecutionStatus.EXPIRED
            execution.version += 1
            execution.finished_at = now
            execution.updated_at = now
            if execution.mode == 'live':
                execution.route_status = RouteStatus.CLOSED
            await append_control_event_row(
                self.session,
                execution,
                event_type='execution.status.changed',
                source='reconciler',
                payload={'from': old_status, 'to': execution.status},
                now=now,
            )
            if old_route_status != execution.route_status:
                await append_control_event_row(
                    self.session,
                    execution,
                    event_type='route.status.changed',
                    source='reconciler',
                    payload={'from': old_route_status, 'to': execution.route_status},
                    now=now,
                )
            changed += 1
        await self.session.flush()
        self._publish_stats(
            scanned=len(candidates),
            claimed=len(pairs),
            changed=changed,
            now=now,
            deadlines=(
                pair.execution.queue_expires_at for pair in pairs if pair.execution.queue_expires_at is not None
            ),
        )
        return changed

    async def request_ttl_stops(self, *, now: datetime) -> int:
        await self._backfill_execution_deadlines()
        candidates = list(
            await self.session.scalars(
                select(ExecutionRow.execution_id)
                .where(
                    ExecutionRow.status.in_([ExecutionStatus.RUNNING, ExecutionStatus.WAITING_INPUT]),
                    self._owned_shard_clause(),
                    ExecutionRow.execution_expires_at <= now,
                )
                .order_by(ExecutionRow.execution_expires_at, ExecutionRow.execution_id)
                .limit(self.batch_size)
            )
        )
        pairs = await self._lock_execution_candidates(candidates)
        existing_stop_assignment_ids = await self._pending_stop_assignment_ids([
            pair.execution.execution_id for pair in pairs
        ])
        changed = 0
        for pair in pairs:
            execution = pair.execution
            if execution.status not in {ExecutionStatus.RUNNING, ExecutionStatus.WAITING_INPUT}:
                continue
            if execution.execution_expires_at is None or execution.execution_expires_at > now:
                continue
            assignment = pair.assignment
            if (
                assignment is None
                or assignment.status != AssignmentStatus.ACTIVE
                or assignment.execution_id != execution.execution_id
                or execution.current_assignment_id != assignment.assignment_id
                or execution.current_attempt_id != assignment.attempt_id
            ):
                continue
            if assignment.assignment_id not in existing_stop_assignment_ids:
                self.session.add(
                    ExecutionCommandRow(
                        command_id=new_id('cmd'),
                        execution_id=execution.execution_id,
                        assignment_id=assignment.assignment_id,
                        attempt_id=assignment.attempt_id,
                        type='stop.request',
                        status=CommandStatus.PENDING,
                        payload_json={'reason': 'execution_ttl'},
                        created_at=now,
                        updated_at=now,
                    )
                )
                await append_control_event_row(
                    self.session,
                    execution,
                    event_type='command.status.changed',
                    source='reconciler',
                    payload={'from': None, 'to': CommandStatus.PENDING, 'type': 'stop.request'},
                    assignment_id=assignment.assignment_id,
                    attempt_id=assignment.attempt_id,
                    now=now,
                )
            old_status = execution.status
            old_route_status = execution.route_status
            execution.status = ExecutionStatus.STOPPING
            execution.stop_reason = 'execution_ttl'
            execution.stop_deadline_at = execution.stop_deadline_at or stop_deadline_after(
                now,
                execution.stop_timeout_seconds,
            )
            execution.version += 1
            execution.updated_at = now
            if execution.mode == 'live':
                execution.route_status = RouteStatus.CLOSED
            if old_status != execution.status:
                await append_control_event_row(
                    self.session,
                    execution,
                    event_type='execution.status.changed',
                    source='reconciler',
                    payload={'from': old_status, 'to': execution.status},
                    assignment_id=assignment.assignment_id,
                    attempt_id=assignment.attempt_id,
                    now=now,
                )
            if old_route_status != execution.route_status:
                await append_control_event_row(
                    self.session,
                    execution,
                    event_type='route.status.changed',
                    source='reconciler',
                    payload={'from': old_route_status, 'to': execution.route_status},
                    assignment_id=assignment.assignment_id,
                    attempt_id=assignment.attempt_id,
                    now=now,
                )
            changed += 1
        await self.session.flush()
        self._publish_stats(
            scanned=len(candidates),
            claimed=len(pairs),
            changed=changed,
            now=now,
            deadlines=(
                pair.execution.execution_expires_at for pair in pairs if pair.execution.execution_expires_at is not None
            ),
        )
        return changed

    async def reconcile_stopping_timeouts(self, *, now: datetime) -> int:
        await self._backfill_stop_deadlines(now=now)
        candidates = list(
            await self.session.scalars(
                select(ExecutionRow.execution_id)
                .where(
                    ExecutionRow.status == ExecutionStatus.STOPPING,
                    self._owned_shard_clause(),
                    ExecutionRow.spec_type != 'nexus_batch',
                    ExecutionRow.stop_deadline_at <= now,
                )
                .order_by(ExecutionRow.stop_deadline_at, ExecutionRow.execution_id)
                .limit(self.batch_size)
            )
        )
        pairs = await self._lock_execution_candidates(candidates)
        changed = 0
        for pair in pairs:
            execution = pair.execution
            if execution.status != ExecutionStatus.STOPPING or execution.spec_type == 'nexus_batch':
                continue
            if execution.stop_deadline_at is None or execution.stop_deadline_at > now:
                continue
            assignment = pair.assignment
            if assignment is not None and (
                assignment.execution_id != execution.execution_id
                or execution.current_assignment_id != assignment.assignment_id
                or execution.current_attempt_id != assignment.attempt_id
            ):
                assignment = None
            old_execution_status = execution.status
            old_route_status = execution.route_status
            if assignment is not None and assignment.status in (AssignmentStatus.PENDING, AssignmentStatus.ACTIVE):
                old_assignment_status = assignment.status
                assignment.status = AssignmentStatus.RELEASED
                assignment.endpoint_ready = False
                assignment.finished_at = assignment.finished_at or now
                assignment.updated_at = now
                await self._cancel_unfinished_commands_for_assignment(assignment, execution, now=now)
                await append_control_event_row(
                    self.session,
                    execution,
                    event_type='assignment.status.changed',
                    source='reconciler',
                    payload={'from': old_assignment_status, 'to': assignment.status},
                    assignment_id=assignment.assignment_id,
                    attempt_id=assignment.attempt_id,
                    now=now,
                )
            execution.status = (
                ExecutionStatus.EXPIRED
                if execution.stop_reason in {'execution_ttl', 'parent_ttl'}
                else ExecutionStatus.KILLED
            )
            execution.version += 1
            execution.finished_at = execution.finished_at or now
            execution.updated_at = now
            if execution.mode == 'live':
                execution.route_status = RouteStatus.CLOSED
            await append_control_event_row(
                self.session,
                execution,
                event_type='execution.status.changed',
                source='reconciler',
                payload={'from': old_execution_status, 'to': execution.status, 'reason': 'stop_timeout'},
                assignment_id=assignment.assignment_id if assignment is not None else None,
                attempt_id=assignment.attempt_id if assignment is not None else None,
                now=now,
            )
            if old_route_status != execution.route_status:
                await append_control_event_row(
                    self.session,
                    execution,
                    event_type='route.status.changed',
                    source='reconciler',
                    payload={'from': old_route_status, 'to': execution.route_status},
                    assignment_id=assignment.assignment_id if assignment is not None else None,
                    attempt_id=assignment.attempt_id if assignment is not None else None,
                    now=now,
                )
            await append_control_event_row(
                self.session,
                execution,
                event_type='reconciler.terminal',
                source='reconciler',
                payload={'reason': 'stop_timeout'},
                assignment_id=assignment.assignment_id if assignment is not None else None,
                attempt_id=assignment.attempt_id if assignment is not None else None,
                now=now,
            )
            changed += 1
        await self.session.flush()
        self._publish_stats(
            scanned=len(candidates),
            claimed=len(pairs),
            changed=changed,
            now=now,
            deadlines=(
                pair.execution.stop_deadline_at for pair in pairs if pair.execution.stop_deadline_at is not None
            ),
        )
        return changed

    async def _backfill_queue_deadlines(self) -> int:
        rows = list(
            await self.session.scalars(
                select(ExecutionRow)
                .options(
                    load_only(
                        ExecutionRow.execution_id,
                        ExecutionRow.first_schedulable_at,
                        ExecutionRow.shard_id,
                        ExecutionRow.queue_ttl_seconds,
                        ExecutionRow.queue_expires_at,
                        ExecutionRow.created_at,
                    )
                )
                .where(
                    ExecutionRow.status == ExecutionStatus.QUEUED,
                    self._owned_shard_clause(),
                    ExecutionRow.queue_expires_at.is_(None),
                    ExecutionRow.queue_ttl_seconds.is_not(None),
                    or_(
                        ExecutionRow.allocation_id.is_(None),
                        ExecutionRow.first_schedulable_at.is_not(None),
                    ),
                )
                .order_by(ExecutionRow.created_at, ExecutionRow.execution_id)
                .limit(self.batch_size)
                .with_for_update(skip_locked=True)
                .execution_options(populate_existing=True)
            )
        )
        for execution in rows:
            queue_started_at = execution.first_schedulable_at or execution.created_at
            execution.queue_expires_at = deadline_after(queue_started_at, execution.queue_ttl_seconds)
        self._track_execution_rows(rows)
        await self.session.flush()
        return len(rows)

    async def _backfill_execution_deadlines(self) -> int:
        rows = list(
            await self.session.scalars(
                select(ExecutionRow)
                .options(
                    load_only(
                        ExecutionRow.execution_id,
                        ExecutionRow.shard_id,
                        ExecutionRow.execution_ttl_seconds,
                        ExecutionRow.execution_expires_at,
                        ExecutionRow.started_at,
                    )
                )
                .where(
                    ExecutionRow.status.in_([ExecutionStatus.RUNNING, ExecutionStatus.WAITING_INPUT]),
                    self._owned_shard_clause(),
                    ExecutionRow.execution_expires_at.is_(None),
                    ExecutionRow.execution_ttl_seconds.is_not(None),
                    ExecutionRow.started_at.is_not(None),
                )
                .order_by(ExecutionRow.started_at, ExecutionRow.execution_id)
                .limit(self.batch_size)
                .with_for_update(skip_locked=True)
                .execution_options(populate_existing=True)
            )
        )
        for execution in rows:
            execution.execution_expires_at = deadline_after(
                execution.started_at,
                execution.execution_ttl_seconds,
            )
        self._track_execution_rows(rows)
        await self.session.flush()
        return len(rows)

    async def _backfill_stop_deadlines(self, *, now: datetime) -> int:
        rows = list(
            await self.session.scalars(
                select(ExecutionRow)
                .options(
                    load_only(
                        ExecutionRow.execution_id,
                        ExecutionRow.shard_id,
                        ExecutionRow.stop_timeout_seconds,
                        ExecutionRow.stop_deadline_at,
                    )
                )
                .where(
                    ExecutionRow.status == ExecutionStatus.STOPPING,
                    self._owned_shard_clause(),
                    ExecutionRow.spec_type != 'nexus_batch',
                    ExecutionRow.stop_deadline_at.is_(None),
                )
                .order_by(ExecutionRow.updated_at, ExecutionRow.execution_id)
                .limit(self.batch_size)
                .with_for_update(skip_locked=True)
                .execution_options(populate_existing=True)
            )
        )
        for execution in rows:
            execution.stop_deadline_at = stop_deadline_after(now, execution.stop_timeout_seconds)
        self._track_execution_rows(rows)
        await self.session.flush()
        return len(rows)

    async def cleanup_terminal(self, *, now: datetime) -> int:
        candidates = list(
            await self.session.scalars(
                select(ExecutionRow.execution_id)
                .where(
                    ExecutionRow.status.in_(list(TERMINAL_STATUSES_FOR_CLEANUP)),
                    self._owned_shard_clause(),
                    ExecutionRow.cleanup_done_at.is_(None),
                )
                .order_by(ExecutionRow.updated_at, ExecutionRow.execution_id)
                .limit(self.batch_size)
            )
        )
        pairs = await self._lock_execution_candidates(candidates)
        changed = 0
        # Fast-path rows only need cleanup_done_at/updated_at stamped; we collect their ids and issue a
        # single set-based UPDATE at the end instead of letting the ORM emit one single-row UPDATE per row.
        # Rows that also require assignment release or live-route closure keep the per-row logic because
        # append_control_event_row must advance next_event_seq/version deterministically per execution.
        fast_execution_ids: list[str] = []
        for pair in pairs:
            execution = pair.execution
            if execution.status not in TERMINAL_STATUSES_FOR_CLEANUP or execution.cleanup_done_at is not None:
                continue
            assignment = pair.assignment
            if assignment is not None and (
                assignment.execution_id != execution.execution_id
                or execution.current_assignment_id != assignment.assignment_id
            ):
                assignment = None
            needs_assignment_release = assignment is not None and assignment.status in (
                AssignmentStatus.PENDING,
                AssignmentStatus.ACTIVE,
            )
            needs_route_close = execution.mode == 'live' and execution.route_status != RouteStatus.CLOSED
            if needs_assignment_release:
                assert assignment is not None  # needs_assignment_release implies assignment is set
                old_assignment_status = assignment.status
                assignment.status = AssignmentStatus.RELEASED
                assignment.endpoint_ready = False
                assignment.finished_at = assignment.finished_at or now
                assignment.updated_at = now
                await self._cancel_unfinished_commands_for_assignment(assignment, execution, now=now)
                await append_control_event_row(
                    self.session,
                    execution,
                    event_type='assignment.status.changed',
                    source='reconciler',
                    payload={'from': old_assignment_status, 'to': assignment.status},
                    assignment_id=assignment.assignment_id,
                    attempt_id=assignment.attempt_id,
                    now=now,
                )
            if needs_route_close:
                old_route_status = execution.route_status
                execution.route_status = RouteStatus.CLOSED
                execution.version += 1
                await append_control_event_row(
                    self.session,
                    execution,
                    event_type='route.status.changed',
                    source='reconciler',
                    payload={'from': old_route_status, 'to': execution.route_status},
                    assignment_id=assignment.assignment_id if assignment is not None else None,
                    attempt_id=assignment.attempt_id if assignment is not None else None,
                    now=now,
                )
            if needs_assignment_release or needs_route_close:
                # Slow path: stamp on the ORM object so it flushes alongside the per-row side effects above.
                execution.cleanup_done_at = now
                execution.updated_at = now
            else:
                # Fast path: do NOT touch the ORM object (that would trigger a second per-row UPDATE on flush);
                # defer the stamp to the batched UPDATE below.
                fast_execution_ids.append(execution.execution_id)
            changed += 1
        # Flush the slow-path ORM mutations first so ordering relative to the set-based statements is explicit.
        await self.session.flush()
        if fast_execution_ids:
            await self.session.execute(
                update(ExecutionRow)
                .where(ExecutionRow.execution_id.in_(fast_execution_ids))
                .values(cleanup_done_at=now, updated_at=now)
                .execution_options(synchronize_session=False)
            )
        # Re-scan completed parents as children may finish after the parent cleanup.
        # This also repairs missed cleanup after a reconciler restart without relying on TTL.
        child = aliased(ExecutionRow)
        has_unfinished_children = (
            select(child.execution_id)
            .where(
                child.parent_execution_id == ExecutionRow.execution_id,
                or_(child.status.not_in(list(TERMINAL_STATUSES_FOR_CLEANUP)), child.cleanup_done_at.is_(None)),
            )
            .exists()
        )
        generation_active = (
            select(BatchRunRow.parent_execution_id)
            .where(
                BatchRunRow.parent_execution_id == ExecutionRow.execution_id,
                BatchRunRow.lifecycle_lease_expires_at > now,
            )
            .exists()
        )
        credential_execution_ids = list(
            await self.session.scalars(
                select(ExecutionRow.execution_id)
                .where(
                    self._owned_shard_clause(),
                    ExecutionRow.parent_execution_id.is_(None),
                    ExecutionRow.status.in_(list(TERMINAL_STATUSES_FOR_CLEANUP)),
                    ExecutionRow.cleanup_done_at.is_not(None),
                    ~has_unfinished_children,
                    ~generation_active,
                    select(ExecutionCredentialRow.execution_id)
                    .where(
                        ExecutionCredentialRow.execution_id == ExecutionRow.execution_id,
                    )
                    .exists(),
                )
                .order_by(ExecutionRow.updated_at, ExecutionRow.execution_id)
                .limit(self.batch_size)
            )
        )
        if credential_execution_ids:
            await self.session.execute(
                delete(ExecutionCredentialRow).where(ExecutionCredentialRow.execution_id.in_(credential_execution_ids))
            )
        await self.session.flush()
        self._publish_stats(scanned=len(candidates), claimed=len(pairs), changed=changed)
        return changed

    def _publish_stats(
        self,
        *,
        scanned: int,
        claimed: int,
        changed: int,
        now: datetime | None = None,
        deadlines=(),
    ) -> None:
        overdue_lag = 0.0
        if now is not None:
            overdue_lag = max(
                (max(0.0, (now - deadline).total_seconds()) for deadline in deadlines),
                default=0.0,
            )
        self._last_stats = ReconcilerStats(
            scanned=scanned,
            claimed=claimed,
            changed=changed,
            overdue_lag_seconds=overdue_lag,
        )

    async def _lock_assignment_candidates(
        self,
        candidates: list[tuple[str, str]],
    ) -> list[_LockedExecutionAssignment]:
        identities = [(str(execution_id), str(assignment_id)) for execution_id, assignment_id in candidates]
        if not identities:
            return []
        execution_ids = sorted({execution_id for execution_id, _ in identities})
        executions = list(
            await self.session.scalars(
                select(ExecutionRow)
                .options(load_only(*_RECONCILER_EXECUTION_COLUMNS))
                .where(
                    ExecutionRow.execution_id.in_(execution_ids),
                    self._owned_shard_clause(),
                )
                .order_by(ExecutionRow.execution_id)
                .with_for_update(skip_locked=True)
                .execution_options(populate_existing=True)
            )
        )
        executions_by_id = {execution.execution_id: execution for execution in executions}
        self._track_execution_rows(executions)
        assignment_ids = sorted({
            assignment_id for execution_id, assignment_id in identities if execution_id in executions_by_id
        })
        assignments = list(
            await self.session.scalars(
                select(AssignmentRow)
                .options(load_only(*_RECONCILER_ASSIGNMENT_COLUMNS))
                .where(AssignmentRow.assignment_id.in_(assignment_ids))
                .order_by(AssignmentRow.execution_id, AssignmentRow.assignment_id)
                .with_for_update(skip_locked=True)
                .execution_options(populate_existing=True)
            )
        )
        assignments_by_id = {assignment.assignment_id: assignment for assignment in assignments}
        return [
            _LockedExecutionAssignment(
                execution=executions_by_id[execution_id],
                assignment=assignments_by_id.get(assignment_id),
            )
            for execution_id, assignment_id in identities
            if execution_id in executions_by_id and assignment_id in assignments_by_id
        ]

    async def _lock_execution_candidates(
        self,
        execution_ids: list[str],
    ) -> list[_LockedExecutionAssignment]:
        ordered_ids = list(dict.fromkeys(str(execution_id) for execution_id in execution_ids if execution_id))
        if not ordered_ids:
            return []
        executions = list(
            await self.session.scalars(
                select(ExecutionRow)
                .options(load_only(*_RECONCILER_EXECUTION_COLUMNS))
                .where(
                    ExecutionRow.execution_id.in_(ordered_ids),
                    self._owned_shard_clause(),
                )
                .order_by(ExecutionRow.execution_id)
                .with_for_update(skip_locked=True)
                .execution_options(populate_existing=True)
            )
        )
        executions_by_id = {execution.execution_id: execution for execution in executions}
        self._track_execution_rows(executions)
        assignment_ids = sorted({
            execution.current_assignment_id for execution in executions if execution.current_assignment_id is not None
        })
        assignments = (
            list(
                await self.session.scalars(
                    select(AssignmentRow)
                    .options(load_only(*_RECONCILER_ASSIGNMENT_COLUMNS))
                    .where(AssignmentRow.assignment_id.in_(assignment_ids))
                    .order_by(AssignmentRow.execution_id, AssignmentRow.assignment_id)
                    .with_for_update(skip_locked=True)
                    .execution_options(populate_existing=True)
                )
            )
            if assignment_ids
            else []
        )
        assignments_by_id = {assignment.assignment_id: assignment for assignment in assignments}
        unlocked_assignment_ids = set(assignment_ids) - assignments_by_id.keys()
        existing_unlocked_assignment_ids = (
            set(
                await self.session.scalars(
                    select(AssignmentRow.assignment_id).where(
                        AssignmentRow.assignment_id.in_(sorted(unlocked_assignment_ids))
                    )
                )
            )
            if unlocked_assignment_ids
            else set()
        )
        return [
            _LockedExecutionAssignment(
                execution=executions_by_id[execution_id],
                assignment=(
                    assignments_by_id.get(executions_by_id[execution_id].current_assignment_id)
                    if executions_by_id[execution_id].current_assignment_id is not None
                    else None
                ),
            )
            for execution_id in ordered_ids
            if execution_id in executions_by_id
            and executions_by_id[execution_id].current_assignment_id not in existing_unlocked_assignment_ids
        ]

    async def _lock_managers(self, manager_ids: list[str]) -> dict[str, InstanceManagerRow]:
        ids = sorted(set(manager_ids))
        if not ids:
            return {}
        managers = list(
            await self.session.scalars(
                select(InstanceManagerRow)
                .options(load_only(*_RECONCILER_MANAGER_COLUMNS))
                .where(InstanceManagerRow.manager_id.in_(ids))
                .order_by(InstanceManagerRow.manager_id)
                .with_for_update()
                .execution_options(populate_existing=True)
            )
        )
        return {manager.manager_id: manager for manager in managers}

    async def _manager_liveness_by_id(
        self,
        manager_ids: list[str],
    ) -> dict[str, InstanceManagerLivenessRow]:
        ids = sorted(set(manager_ids))
        if not ids:
            return {}
        rows = list(
            await self.session.scalars(
                select(InstanceManagerLivenessRow).where(InstanceManagerLivenessRow.manager_id.in_(ids))
            )
        )
        return {row.manager_id: row for row in rows}

    @staticmethod
    def _is_current_active_pair(pair: _LockedExecutionAssignment) -> bool:
        execution = pair.execution
        assignment = pair.assignment
        return bool(
            assignment is not None
            and execution.status in ACTIVE_EXECUTION_STATUSES | {ExecutionStatus.STOPPING}
            and execution.current_assignment_id == assignment.assignment_id
            and execution.current_attempt_id == assignment.attempt_id
            and assignment.execution_id == execution.execution_id
            and assignment.status in {AssignmentStatus.PENDING, AssignmentStatus.ACTIVE}
        )

    async def _repair_active_pair(self, pair: _LockedExecutionAssignment, *, now: datetime) -> bool:
        execution = pair.execution
        assignment = pair.assignment
        if execution.status not in ACTIVE_EXECUTION_STATUSES or self._is_current_active_pair(pair):
            return False
        old_execution_status = execution.status
        old_route_status = execution.route_status
        assignment_id = assignment.assignment_id if assignment is not None else None
        attempt_id = assignment.attempt_id if assignment is not None else execution.current_attempt_id
        if assignment is not None:
            await self._cancel_unfinished_commands_for_assignment(assignment, execution, now=now)
        started = (
            execution.started_at is not None
            or execution.status in {ExecutionStatus.RUNNING, ExecutionStatus.WAITING_INPUT}
            or (assignment is not None and assignment.started_at is not None)
        )
        retrying = _can_retry_assignment_loss(execution, started=started)
        if retrying:
            execution.status = ExecutionStatus.QUEUED
            execution.queue_entered_at = now
            execution.next_schedule_at = now
        else:
            execution.status = ExecutionStatus.LOST
            execution.finished_at = execution.finished_at or now
            if execution.mode == 'live':
                execution.route_status = RouteStatus.CLOSED
        execution.version += 1
        execution.updated_at = now
        await append_control_event_row(
            self.session,
            execution,
            event_type='execution.status.changed',
            source='reconciler',
            payload={'from': old_execution_status, 'to': execution.status},
            assignment_id=assignment_id,
            attempt_id=attempt_id,
            now=now,
        )
        if old_route_status != execution.route_status:
            await append_control_event_row(
                self.session,
                execution,
                event_type='route.status.changed',
                source='reconciler',
                payload={'from': old_route_status, 'to': execution.route_status},
                assignment_id=assignment_id,
                attempt_id=attempt_id,
                now=now,
            )
        await append_control_event_row(
            self.session,
            execution,
            event_type='reconciler.retry' if retrying else 'reconciler.terminal',
            source='reconciler',
            payload={'reason': 'current_assignment_inactive', 'attempt_count': execution.attempt_count},
            assignment_id=assignment_id,
            attempt_id=attempt_id,
            now=now,
        )
        return True

    async def _pending_stop_assignment_ids(self, execution_ids: list[str]) -> set[str]:
        ids = list(dict.fromkeys(execution_id for execution_id in execution_ids if execution_id))
        if not ids:
            return set()
        rows = list(
            await self.session.scalars(
                select(ExecutionCommandRow)
                .where(
                    ExecutionCommandRow.execution_id.in_(ids),
                    ExecutionCommandRow.type == 'stop.request',
                    ExecutionCommandRow.status == CommandStatus.PENDING,
                    ExecutionCommandRow.assignment_id.is_not(None),
                )
                .order_by(ExecutionCommandRow.command_id)
                .with_for_update()
                .execution_options(populate_existing=True)
            )
        )
        return {row.assignment_id for row in rows if row.assignment_id is not None}

    async def _mark_lost_or_retry(
        self,
        assignment: AssignmentRow,
        *,
        now: datetime,
        execution: ExecutionRow,
        assignment_status: str = AssignmentStatus.LOST,
        reason: str = 'assignment_lost',
    ) -> bool:
        if (
            assignment.status not in {AssignmentStatus.PENDING, AssignmentStatus.ACTIVE}
            or execution.status not in ACTIVE_EXECUTION_STATUSES | {ExecutionStatus.STOPPING}
            or execution.current_assignment_id != assignment.assignment_id
            or execution.current_attempt_id != assignment.attempt_id
            or assignment.execution_id != execution.execution_id
        ):
            return False
        old_assignment_status = assignment.status
        old_execution_status = execution.status
        old_route_status = execution.route_status
        started = (
            execution.started_at is not None
            or assignment.started_at is not None
            or execution.status
            in {
                ExecutionStatus.RUNNING,
                ExecutionStatus.WAITING_INPUT,
                ExecutionStatus.STOPPING,
            }
        )
        assignment.status = assignment_status
        assignment.endpoint_ready = False
        assignment.finished_at = assignment.finished_at or now
        assignment.updated_at = now
        await self._cancel_unfinished_commands_for_assignment(assignment, execution, now=now)
        if _can_retry_assignment_loss(execution, started=started):
            execution.status = ExecutionStatus.QUEUED
            execution.queue_entered_at = now
            execution.next_schedule_at = now
            retrying = True
        else:
            execution.status = ExecutionStatus.LOST
            execution.finished_at = execution.finished_at or now
            if execution.mode == 'live':
                execution.route_status = RouteStatus.CLOSED
            retrying = False
        execution.version += 1
        execution.updated_at = now
        await append_control_event_row(
            self.session,
            execution,
            event_type='assignment.status.changed',
            source='reconciler',
            payload={'from': old_assignment_status, 'to': assignment.status},
            assignment_id=assignment.assignment_id,
            attempt_id=assignment.attempt_id,
            now=now,
        )
        if old_execution_status != execution.status:
            await append_control_event_row(
                self.session,
                execution,
                event_type='execution.status.changed',
                source='reconciler',
                payload={'from': old_execution_status, 'to': execution.status},
                assignment_id=assignment.assignment_id,
                attempt_id=assignment.attempt_id,
                now=now,
            )
        if old_route_status != execution.route_status:
            await append_control_event_row(
                self.session,
                execution,
                event_type='route.status.changed',
                source='reconciler',
                payload={'from': old_route_status, 'to': execution.route_status},
                assignment_id=assignment.assignment_id,
                attempt_id=assignment.attempt_id,
                now=now,
            )
        await append_control_event_row(
            self.session,
            execution,
            event_type='reconciler.retry' if retrying else 'reconciler.terminal',
            source='reconciler',
            payload={'reason': reason, 'attempt_count': execution.attempt_count},
            assignment_id=assignment.assignment_id,
            attempt_id=assignment.attempt_id,
            now=now,
        )
        return True

    async def _cancel_unfinished_commands_for_assignment(
        self,
        assignment: AssignmentRow,
        execution: ExecutionRow,
        *,
        now: datetime,
    ) -> int:
        rows = list(
            await self.session.scalars(
                select(ExecutionCommandRow)
                .where(
                    ExecutionCommandRow.assignment_id == assignment.assignment_id,
                    ExecutionCommandRow.status.in_([CommandStatus.PENDING, CommandStatus.CLAIMED]),
                )
                .order_by(ExecutionCommandRow.command_id)
                .with_for_update()
                .execution_options(populate_existing=True)
            )
        )
        return await self._cancel_unfinished_command_rows(rows, execution, now=now)

    async def _cancel_unfinished_command_rows(
        self,
        rows: list[ExecutionCommandRow],
        execution: ExecutionRow,
        *,
        now: datetime,
    ) -> int:
        changed = 0
        for row in rows:
            old_status = row.status
            row.status = CommandStatus.CANCELLED
            row.completed_at = now
            row.updated_at = now
            await append_control_event_row(
                self.session,
                execution,
                event_type='command.status.changed',
                source='reconciler',
                payload={'command_id': row.command_id, 'from': old_status, 'to': row.status},
                assignment_id=row.assignment_id,
                attempt_id=row.attempt_id,
                now=now,
            )
            changed += 1
        return changed
