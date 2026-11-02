"""Transactional shared request budget allocation and lease renewal repository."""

from __future__ import annotations

import math

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Literal

from sqlalchemy import and_, delete, not_, or_, select, text
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from nexus.distributed.errors import DomainValidationError
from nexus.distributed.execution.models import AssignmentStatus
from nexus.distributed.persistence.orm import (
    AssignmentRow,
    ExecutionRow,
    InstanceManagerLivenessRow,
    InstanceManagerRow,
    PolicyScopeRow,
    RequestBudgetLeaseRow,
    RequestBudgetRow,
)
from nexus.distributed.policies.models import (
    PolicyExecutorConfig,
    RequestBudgetConfig,
    RequestPolicySelection,
)
from nexus.distributed.policies.request_rate import (
    RequestPolicyMode,
    consume_request_rate_snapshot,
    request_budget_key,
    resolve_request_selection,
    validate_manager_request_policy,
)
from nexus.distributed.utils.clock import utcnow
from nexus.distributed.utils.identity import new_id


LeaseStatus = Literal['granted', 'wait', 'inactive', 'unavailable']
_SHARE_TOLERANCE = 1e-6
_ACTIVE_ASSIGNMENT_STATUSES = (AssignmentStatus.PENDING, AssignmentStatus.ACTIVE)
MAX_ACTIVE_LEASES_PER_BUDGET = 256
EXPIRED_LEASE_RETENTION_SECONDS = 3600.0


@dataclass(frozen=True, slots=True)
class BudgetDemand:
    budget_key: str
    manager_id: str
    manager_session_id: str
    request_seq: int
    # None: waiter wants an equal share of available total QPM; >0: bounded demand; 0: no recent demand.
    requested_qpm: float | None = None
    metrics: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.budget_key or not self.manager_id or not self.manager_session_id:
            raise ValueError('budget_key, manager_id and manager_session_id are required')
        if isinstance(self.request_seq, bool) or not isinstance(self.request_seq, int) or self.request_seq < 1:
            raise ValueError('request_seq must be a positive integer')
        if self.requested_qpm is not None and (
            isinstance(self.requested_qpm, bool)
            or not isinstance(self.requested_qpm, (int, float))
            or not math.isfinite(self.requested_qpm)
            or self.requested_qpm < 0
        ):
            raise ValueError('requested_qpm must be None or a finite non-negative number')


@dataclass(frozen=True, slots=True)
class BudgetLease:
    lease_id: str
    lease_version: int
    budget_key: str
    manager_id: str
    manager_session_id: str
    granted_qpm: float
    valid_from: datetime
    valid_until: datetime
    config_revision: int
    config_hash: str
    snapshot_version: int

    def to_json(self) -> dict[str, Any]:
        return {
            'lease_id': self.lease_id,
            'lease_version': self.lease_version,
            'budget_key': self.budget_key,
            'manager_id': self.manager_id,
            'manager_session_id': self.manager_session_id,
            'granted_qpm': self.granted_qpm,
            'valid_from': self.valid_from.isoformat(),
            'valid_until': self.valid_until.isoformat(),
            'config_revision': self.config_revision,
            'config_hash': self.config_hash,
            'snapshot_version': self.snapshot_version,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any] | None) -> BudgetLease | None:
        if not isinstance(data, Mapping):
            return None
        return cls(
            lease_id=str(data['lease_id']),
            lease_version=int(data['lease_version']),
            budget_key=str(data['budget_key']),
            manager_id=str(data['manager_id']),
            manager_session_id=str(data['manager_session_id']),
            granted_qpm=float(data['granted_qpm']),
            valid_from=datetime.fromisoformat(str(data['valid_from'])),
            valid_until=datetime.fromisoformat(str(data['valid_until'])),
            config_revision=int(data['config_revision']),
            config_hash=str(data['config_hash']),
            snapshot_version=int(data['snapshot_version']),
        )


@dataclass(frozen=True, slots=True)
class BudgetLeaseResult:
    status: LeaseStatus
    mode: RequestPolicyMode
    config_revision: int
    config_hash: str
    lease: BudgetLease | None
    server_now: datetime
    retry_after: float
    reason_code: str

    def to_idempotent_json(self) -> dict[str, Any]:
        """Serialize stable result fields; server_now is intentionally excluded and re-read on replay."""
        return {
            'status': self.status,
            'mode': self.mode,
            'config_revision': self.config_revision,
            'config_hash': self.config_hash,
            'lease': self.lease.to_json() if self.lease is not None else None,
            'retry_after': self.retry_after,
            'reason_code': self.reason_code,
        }

    @classmethod
    def from_idempotent_json(cls, data: Mapping[str, Any], *, server_now: datetime) -> BudgetLeaseResult:
        return cls(
            status=data['status'],
            mode=data['mode'],
            config_revision=int(data['config_revision']),
            config_hash=str(data['config_hash']),
            lease=BudgetLease.from_json(data.get('lease')),
            server_now=server_now,
            retry_after=float(data['retry_after']),
            reason_code=str(data['reason_code']),
        )


def compute_target_shares(
    total_budget: float,
    demands: Mapping[tuple[str, str], float | None],
) -> dict[tuple[str, str], float]:
    """Divide ``total_budget`` equally across active demanders, subject to per-manager ``requested_qpm`` caps."""
    if total_budget <= 0 or not demands:
        return dict.fromkeys(demands, 0.0)
    remaining_budget = float(total_budget)
    remaining_keys = set(demands.keys())
    allocations: dict[tuple[str, str], float] = {}

    while remaining_keys and remaining_budget > 0:
        equal_share = remaining_budget / len(remaining_keys)
        capped = [
            key
            for key in sorted(remaining_keys)
            if (requested := demands[key]) is not None and float(requested) <= equal_share + _SHARE_TOLERANCE
        ]
        if not capped:
            for key in remaining_keys:
                allocations[key] = equal_share
            break
        for key in capped:
            requested = demands[key]
            assert requested is not None
            cap = max(0.0, float(requested))
            allocations[key] = cap
            remaining_budget = max(0.0, remaining_budget - cap)
            remaining_keys.remove(key)

    for key in demands:
        allocations.setdefault(key, 0.0)
    return allocations


class RequestBudgetRepository:
    """Short-transaction coordinator for shared request_rate budgets."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def _read_server_now(self, override: datetime | None) -> datetime:
        if override is not None:
            return override
        dialect = self.session.get_bind().dialect.name
        if dialect == 'mysql':
            value = await self.session.scalar(text('SELECT UTC_TIMESTAMP(6)'))
            if isinstance(value, datetime):
                return value
        elif dialect == 'sqlite':
            raw = await self.session.scalar(text("SELECT STRFTIME('%Y-%m-%dT%H:%M:%f', 'now')"))
            if isinstance(raw, str):
                return datetime.fromisoformat(raw)
        return utcnow()

    async def coordinate_lease(
        self,
        demand: BudgetDemand,
        *,
        selection: RequestPolicySelection,
        config: PolicyExecutorConfig,
        budget_config: RequestBudgetConfig | None = None,
        now: datetime | None = None,
    ) -> BudgetLeaseResult:
        """Execute one atomic budget demand registration and lease allocation/renewal transaction."""
        effective_budget_config = budget_config or RequestBudgetConfig()
        expected_key = request_budget_key(selection)
        if demand.budget_key != expected_key:
            raise DomainValidationError('request_budget_key_mismatch')

        # 1. Verify Manager session, fixed binding capability, and active task authorization.
        manager = await self.session.get(InstanceManagerRow, demand.manager_id)
        if manager is None or manager.session_id != demand.manager_session_id:
            raise DomainValidationError('manager_session_invalid')
        liveness = await self.session.get(InstanceManagerLivenessRow, demand.manager_id)
        if liveness is not None and (
            liveness.fenced_at is not None or liveness.session_id != demand.manager_session_id
        ):
            raise DomainValidationError('manager_session_fenced')
        if manager.request_policy_binding != selection.binding or manager.request_policy_scope != selection.scope:
            raise DomainValidationError('manager_request_policy_mismatch')

        try:
            validate_manager_request_policy(
                config,
                selection,
                environment_id=manager.environment_id,
                owner_principal_id=manager.owner_principal_id,
            )
        except ValueError as exc:
            raise DomainValidationError('request_policy_not_permitted', detail=str(exc)) from exc

        await self._verify_assigned_executions_authorized(
            manager_id=demand.manager_id,
            manager_session_id=demand.manager_session_id,
            selection=selection,
            config=config,
        )

        binding = config.bindings[selection.binding]
        digest = binding.fingerprint(config.imports)

        # 2. Lock policy scope (FOR SHARE) then budget master row (FOR UPDATE) in fixed order.
        scope_row = await self.session.scalar(
            select(PolicyScopeRow)
            .where(PolicyScopeRow.scope_key == expected_key)
            .with_for_update(read=True)
            .execution_options(populate_existing=True)
        )
        server_now = await self._read_server_now(now)
        budget_row = await self._lock_or_create_budget_row(
            budget_key=expected_key,
            selection=selection,
            config_revision=binding.revision,
            config_hash=digest,
            now=server_now,
        )
        if binding.revision < budget_row.config_revision:
            raise DomainValidationError('request_policy_revision_stale')
        if binding.revision > budget_row.config_revision or budget_row.config_hash != digest:
            budget_row.config_revision = binding.revision
            budget_row.config_hash = digest

        # Prune expired and inactive lease rows past retention so locked scans stay bounded.
        retention_cutoff = server_now - timedelta(
            seconds=max(EXPIRED_LEASE_RETENTION_SECONDS, 4.0 * effective_budget_config.lease_seconds)
        )
        await self.session.execute(
            delete(RequestBudgetLeaseRow).where(
                RequestBudgetLeaseRow.budget_key == expected_key,
                RequestBudgetLeaseRow.demand_expires_at < retention_cutoff,
                or_(
                    RequestBudgetLeaseRow.valid_until.is_(None),
                    RequestBudgetLeaseRow.valid_until < retention_cutoff,
                ),
                not_(
                    and_(
                        RequestBudgetLeaseRow.manager_id == demand.manager_id,
                        RequestBudgetLeaseRow.manager_session_id == demand.manager_session_id,
                    )
                ),
            )
        )

        # 3. Handle request_seq idempotency using fresh database time.
        lease_row = await self.session.scalar(
            select(RequestBudgetLeaseRow)
            .where(
                RequestBudgetLeaseRow.manager_id == demand.manager_id,
                RequestBudgetLeaseRow.manager_session_id == demand.manager_session_id,
            )
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        if lease_row is not None:
            if (
                lease_row.budget_key != expected_key
                or lease_row.binding_id != selection.binding
                or lease_row.scope_id != selection.scope
            ):
                raise DomainValidationError('manager_request_policy_immutable')
            if demand.request_seq < lease_row.request_seq:
                raise DomainValidationError('request_budget_seq_stale')
            if demand.request_seq == lease_row.request_seq:
                if not _same_requested_qpm(demand.requested_qpm, lease_row.requested_qpm):
                    raise DomainValidationError('request_budget_seq_conflict')
                return BudgetLeaseResult.from_idempotent_json(lease_row.last_result_json, server_now=server_now)

        has_demand = demand.requested_qpm is None or demand.requested_qpm > 0
        demand_expires_at = (
            server_now + timedelta(seconds=2.0 * effective_budget_config.lease_seconds) if has_demand else server_now
        )
        if lease_row is None:
            lease_row = RequestBudgetLeaseRow(
                manager_id=demand.manager_id,
                manager_session_id=demand.manager_session_id,
                budget_key=expected_key,
                binding_id=selection.binding,
                scope_id=selection.scope,
                environment_id=manager.environment_id,
                owner_principal_id=manager.owner_principal_id,
                request_seq=demand.request_seq,
                requested_qpm=float(demand.requested_qpm) if demand.requested_qpm is not None else None,
                demand_expires_at=demand_expires_at,
                lease_id=None,
                lease_version=0,
                granted_qpm=0.0,
                valid_from=None,
                valid_until=None,
                mode=binding.mode,
                config_revision=binding.revision,
                config_hash=digest,
                snapshot_version=None,
                reason_code='policy_unavailable',
                last_result_json={},
                updated_at=server_now,
            )
            self.session.add(lease_row)
        else:
            lease_row.request_seq = demand.request_seq
            lease_row.requested_qpm = float(demand.requested_qpm) if demand.requested_qpm is not None else None
            lease_row.demand_expires_at = demand_expires_at
            lease_row.updated_at = server_now

        reading = consume_request_rate_snapshot(
            selection,
            config,
            scope_row,
            now=server_now,
            fingerprint=digest,
        )

        # 4. Handle shadow / disabled modes (do not require snapshot, do not clear unexpired lease accounting early).
        if reading.mode in {'shadow', 'disabled'}:
            result = BudgetLeaseResult(
                status='inactive',
                mode=reading.mode,
                config_revision=reading.config_revision,
                config_hash=reading.config_hash,
                lease=None,
                server_now=server_now,
                retry_after=effective_budget_config.lease_seconds,
                reason_code=reading.reason_code,
            )
            self._record_result(
                lease_row, budget_row, result, snapshot_version=reading.snapshot_version, metrics=demand.metrics
            )
            return result

        # Enforce mode with no active demand: stop issuing/renewing, keep existing unexpired lease accounted in DB.
        if not has_demand:
            result = BudgetLeaseResult(
                status='wait',
                mode='enforce',
                config_revision=reading.config_revision,
                config_hash=reading.config_hash,
                lease=None,
                server_now=server_now,
                retry_after=effective_budget_config.lease_seconds,
                reason_code='no_demand',
            )
            self._record_result(
                lease_row, budget_row, result, snapshot_version=reading.snapshot_version, metrics=demand.metrics
            )
            return result

        if not reading.ready:
            result = BudgetLeaseResult(
                status='unavailable',
                mode='enforce',
                config_revision=reading.config_revision,
                config_hash=reading.config_hash,
                lease=None,
                server_now=server_now,
                retry_after=min(1.0, effective_budget_config.lease_seconds / 2.0),
                reason_code=reading.reason_code,
            )
            self._record_result(lease_row, budget_row, result, snapshot_version=None, metrics=demand.metrics)
            return result

        if not reading.allow or reading.rate_per_minute <= 0:
            result = BudgetLeaseResult(
                status='wait',
                mode='enforce',
                config_revision=reading.config_revision,
                config_hash=reading.config_hash,
                lease=None,
                server_now=server_now,
                retry_after=min(1.0, effective_budget_config.lease_seconds / 2.0),
                reason_code=reading.reason_code,
            )
            self._record_result(
                lease_row, budget_row, result, snapshot_version=reading.snapshot_version, metrics=demand.metrics
            )
            return result

        # 5. Summarize all unexpired leases and active demands under the budget lock.
        await self.session.flush()
        peer_rows = (
            await self.session.scalars(
                select(RequestBudgetLeaseRow)
                .where(
                    RequestBudgetLeaseRow.budget_key == expected_key,
                    or_(
                        RequestBudgetLeaseRow.valid_until > server_now,
                        RequestBudgetLeaseRow.demand_expires_at > server_now,
                    ),
                )
                .with_for_update()
            )
        ).all()

        if len(peer_rows) > MAX_ACTIVE_LEASES_PER_BUDGET:
            lease_row.demand_expires_at = server_now
            result = BudgetLeaseResult(
                status='wait',
                mode='enforce',
                config_revision=reading.config_revision,
                config_hash=reading.config_hash,
                lease=None,
                server_now=server_now,
                retry_after=effective_budget_config.lease_seconds,
                reason_code='budget_active_leases_exceeded',
            )
            self._record_result(
                lease_row, budget_row, result, snapshot_version=reading.snapshot_version, metrics=demand.metrics
            )
            return result

        total_budget = reading.rate_per_minute
        unexpired_rows = [
            row
            for row in peer_rows
            if row.valid_until is not None and row.valid_until > server_now and row.granted_qpm > 0
        ]
        total_committed = sum(row.granted_qpm for row in unexpired_rows)
        available_qpm = max(0.0, total_budget - total_committed)

        # Superseded sessions of the same manager_id keep their unexpired lease counted in total_committed
        # until valid_until, but no longer compete as a second active demander against their own new session.
        for row in peer_rows:
            if (
                row.manager_id == demand.manager_id
                and row.manager_session_id != demand.manager_session_id
                and row.demand_expires_at > server_now
            ):
                row.demand_expires_at = server_now

        active_demands: dict[tuple[str, str], float | None] = {
            (row.manager_id, row.manager_session_id): row.requested_qpm
            for row in peer_rows
            if row.demand_expires_at > server_now and (row.requested_qpm is None or row.requested_qpm > 0)
        }
        shares = compute_target_shares(total_budget, active_demands)
        target_qpm = shares.get((demand.manager_id, demand.manager_session_id), 0.0)

        assert reading.valid_until is not None
        assert reading.snapshot_version is not None
        candidate_valid_until = min(
            server_now + timedelta(seconds=effective_budget_config.lease_seconds),
            reading.valid_until,
        )

        has_unexpired_lease = (
            lease_row.lease_id is not None
            and lease_row.valid_until is not None
            and lease_row.valid_until > server_now
            and lease_row.granted_qpm > 0
        )

        if has_unexpired_lease:
            existing_lease_id = lease_row.lease_id
            existing_valid_from = lease_row.valid_from
            existing_valid_until = lease_row.valid_until
            assert existing_lease_id is not None
            assert existing_valid_from is not None
            assert existing_valid_until is not None
            # Same-share renewal is allowed when total committed budget fits within current total_budget
            # and this Manager's existing grant is not above its target share (either equal to target_qpm,
            # or below target_qpm while no uncommitted budget is available to step it up yet).
            can_keep_same_share = math.isclose(
                lease_row.granted_qpm, target_qpm, rel_tol=1e-6, abs_tol=_SHARE_TOLERANCE
            ) or (lease_row.granted_qpm <= target_qpm + _SHARE_TOLERANCE and available_qpm <= _SHARE_TOLERANCE)
            if (
                total_committed <= total_budget + _SHARE_TOLERANCE
                and can_keep_same_share
                and candidate_valid_until >= existing_valid_until
            ):
                lease_row.lease_version += 1
                lease_row.valid_until = candidate_valid_until
                lease = BudgetLease(
                    lease_id=existing_lease_id,
                    lease_version=lease_row.lease_version,
                    budget_key=expected_key,
                    manager_id=demand.manager_id,
                    manager_session_id=demand.manager_session_id,
                    granted_qpm=lease_row.granted_qpm,
                    valid_from=existing_valid_from,
                    valid_until=candidate_valid_until,
                    config_revision=reading.config_revision,
                    config_hash=reading.config_hash,
                    snapshot_version=reading.snapshot_version,
                )
                result = BudgetLeaseResult(
                    status='granted',
                    mode='enforce',
                    config_revision=reading.config_revision,
                    config_hash=reading.config_hash,
                    lease=lease,
                    server_now=server_now,
                    retry_after=max(
                        0.1,
                        effective_budget_config.lease_seconds - effective_budget_config.renew_before_seconds,
                    ),
                    reason_code=reading.reason_code,
                )
                self._record_result(
                    lease_row, budget_row, result, snapshot_version=reading.snapshot_version, metrics=demand.metrics
                )
                return result

            # Share needs to change or total committed exceeds reduced budget: do NOT extend or overwrite
            # the still-valid lease; wait until it expires so no rate is double-allocated.
            wait_seconds = max(0.05, (existing_valid_until - server_now).total_seconds())
            result = BudgetLeaseResult(
                status='wait',
                mode='enforce',
                config_revision=reading.config_revision,
                config_hash=reading.config_hash,
                lease=None,
                server_now=server_now,
                retry_after=wait_seconds,
                reason_code='lease_rebalance_wait',
            )
            self._record_result(
                lease_row, budget_row, result, snapshot_version=reading.snapshot_version, metrics=demand.metrics
            )
            return result

        # No unexpired lease for this Manager: allocate from remaining uncommitted budget.
        grant_qpm = min(target_qpm, available_qpm)
        if grant_qpm > _SHARE_TOLERANCE and candidate_valid_until > server_now:
            new_lease_id = new_id('bdl')
            lease_row.lease_id = new_lease_id
            lease_row.lease_version = 1
            lease_row.granted_qpm = grant_qpm
            lease_row.valid_from = server_now
            lease_row.valid_until = candidate_valid_until
            lease = BudgetLease(
                lease_id=new_lease_id,
                lease_version=lease_row.lease_version,
                budget_key=expected_key,
                manager_id=demand.manager_id,
                manager_session_id=demand.manager_session_id,
                granted_qpm=grant_qpm,
                valid_from=server_now,
                valid_until=candidate_valid_until,
                config_revision=reading.config_revision,
                config_hash=reading.config_hash,
                snapshot_version=reading.snapshot_version,
            )
            result = BudgetLeaseResult(
                status='granted',
                mode='enforce',
                config_revision=reading.config_revision,
                config_hash=reading.config_hash,
                lease=lease,
                server_now=server_now,
                retry_after=max(
                    0.1,
                    effective_budget_config.lease_seconds - effective_budget_config.renew_before_seconds,
                ),
                reason_code=reading.reason_code,
            )
            self._record_result(
                lease_row, budget_row, result, snapshot_version=reading.snapshot_version, metrics=demand.metrics
            )
            return result

        earliest_expiry = min(
            (row.valid_until for row in unexpired_rows if row.valid_until is not None),
            default=server_now + timedelta(seconds=1.0),
        )
        wait_seconds = max(
            0.05,
            min(effective_budget_config.lease_seconds, (earliest_expiry - server_now).total_seconds()),
        )
        result = BudgetLeaseResult(
            status='wait',
            mode='enforce',
            config_revision=reading.config_revision,
            config_hash=reading.config_hash,
            lease=None,
            server_now=server_now,
            retry_after=wait_seconds,
            reason_code='budget_occupied',
        )
        self._record_result(
            lease_row, budget_row, result, snapshot_version=reading.snapshot_version, metrics=demand.metrics
        )
        return result

    async def _verify_assigned_executions_authorized(
        self,
        *,
        manager_id: str,
        manager_session_id: str,
        selection: RequestPolicySelection,
        config: PolicyExecutorConfig,
    ) -> None:
        rows = (
            await self.session.execute(
                select(
                    ExecutionRow.environment_id,
                    ExecutionRow.owner_principal_id,
                    ExecutionRow.spec_type,
                    ExecutionRow.placement_json,
                )
                .join(AssignmentRow, AssignmentRow.execution_id == ExecutionRow.execution_id)
                .where(
                    AssignmentRow.manager_id == manager_id,
                    AssignmentRow.manager_session_id == manager_session_id,
                    AssignmentRow.status.in_(_ACTIVE_ASSIGNMENT_STATUSES),
                )
            )
        ).all()
        for env_id, owner_id, spec_type, placement_json in rows:
            if spec_type == 'nexus_batch':
                continue
            placement = placement_json if isinstance(placement_json, Mapping) else {}
            requested = placement.get('request_policy')
            resolved = resolve_request_selection(
                config,
                requested,
                environment_id=env_id or '',
                owner_principal_id=owner_id,
                spec_type=spec_type,
            )
            if resolved != selection:
                raise DomainValidationError('execution_request_policy_mismatch')

    async def _lock_or_create_budget_row(
        self,
        *,
        budget_key: str,
        selection: RequestPolicySelection,
        config_revision: int,
        config_hash: str,
        now: datetime,
    ) -> RequestBudgetRow:
        values = {
            'budget_key': budget_key,
            'binding_id': selection.binding,
            'scope_id': selection.scope,
            'config_revision': config_revision,
            'config_hash': config_hash,
            'ledger_version': 1,
            'updated_at': now,
        }
        dialect = self.session.get_bind().dialect.name
        if dialect == 'mysql':
            stmt = mysql_insert(RequestBudgetRow).values(**values)
            await self.session.execute(stmt.on_duplicate_key_update(budget_key=budget_key))
        elif dialect == 'sqlite':
            stmt = sqlite_insert(RequestBudgetRow).values(**values)
            await self.session.execute(stmt.on_conflict_do_nothing())
        else:
            raise ValueError(f'unsupported dialect: {dialect}')

        row = await self.session.scalar(
            select(RequestBudgetRow)
            .where(RequestBudgetRow.budget_key == budget_key)
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        assert row is not None
        return row

    @staticmethod
    def _record_result(
        lease_row: RequestBudgetLeaseRow,
        budget_row: RequestBudgetRow,
        result: BudgetLeaseResult,
        *,
        snapshot_version: int | None,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        lease_row.mode = result.mode
        lease_row.config_revision = result.config_revision
        lease_row.config_hash = result.config_hash
        lease_row.snapshot_version = snapshot_version
        lease_row.reason_code = result.reason_code
        payload = result.to_idempotent_json()
        if isinstance(metrics, dict):
            payload['metrics'] = dict(metrics)
        elif isinstance(lease_row.last_result_json, dict) and 'metrics' in lease_row.last_result_json:
            payload['metrics'] = lease_row.last_result_json['metrics']
        lease_row.last_result_json = payload
        lease_row.updated_at = result.server_now
        budget_row.ledger_version += 1
        budget_row.updated_at = result.server_now


def _same_requested_qpm(a: float | None, b: float | None) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-9)


__all__ = [
    'BudgetDemand',
    'BudgetLease',
    'BudgetLeaseResult',
    'LeaseStatus',
    'RequestBudgetRepository',
    'compute_target_shares',
]
