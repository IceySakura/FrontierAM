from __future__ import annotations

import logging

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, cast

from sqlalchemy import and_, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from nexus.distributed.environment.models import (
    EnvironmentAllocationRecord,
    EnvironmentAllocationStatus,
    EnvironmentAllocationType,
    EnvironmentProviderType,
    EnvironmentStatus,
    ZhiyanLogBinding,
)
from nexus.distributed.errors import (
    DomainConflictError,
    DomainValidationError,
    FencingError,
    ResourceNotFoundError,
)
from nexus.distributed.observability.database import queue_stage_result
from nexus.distributed.persistence.orm import EnvironmentAllocationRow, EnvironmentGrantRow, EnvironmentRow
from nexus.distributed.repositories.pagination import decode_cursor, encode_cursor
from nexus.distributed.submission.resources import validate_resource_totals
from nexus.distributed.utils.identity import new_id, validate_environment_id, validate_pool_id


@dataclass(frozen=True, slots=True)
class AllocationResources:
    manager_count: int = 0
    cpu_millicores: int = 0
    memory_bytes: int = 0
    slots: int = 0
    active_batches: int = 0

    def __post_init__(self) -> None:
        validate_resource_totals(
            manager_count=self.manager_count,
            cpu_millicores=self.cpu_millicores,
            memory_bytes=self.memory_bytes,
            slots=self.slots,
            active_batches=self.active_batches,
            field='allocation_resources',
        )


def _decode_allocation_cursor(value: str) -> tuple[datetime, str]:
    payload = decode_cursor(value, fields=frozenset({'created_at', 'allocation_id'}))
    try:
        created_at = datetime.fromisoformat(payload['created_at'])
    except ValueError as exc:
        raise DomainValidationError('invalid_cursor') from exc
    if created_at.tzinfo is not None:
        raise DomainValidationError('invalid_cursor')
    return created_at, payload['allocation_id']


def _clean_zhiyan_binding(log_name: Any, *, provider_type: str) -> ZhiyanLogBinding | None:
    if not isinstance(log_name, str):
        return None
    cleaned = log_name.strip()
    if not cleaned or len(cleaned) > 128:
        return None
    return ZhiyanLogBinding(log_name=cleaned, provider_type=provider_type)


@dataclass(frozen=True, slots=True)
class CreateEnvironmentAllocation:
    allocation_id: str
    top_level_execution_id: str
    owner_principal_id: str
    environment_id: str
    pool_id: str
    workload: str
    desired_manager_count: int
    resources: AllocationResources
    allocation_spec: dict[str, Any]
    allocation_spec_hash: str
    dependency_build_id: str | None = None


@dataclass(frozen=True, slots=True)
class EnvironmentAllocationClaim:
    allocation_id: str
    operation_id: str
    operation_owner: str
    operation_lease_until: datetime
    status: str
    status_version: int


_TRANSITIONS = {
    EnvironmentAllocationStatus.REQUESTED: {
        EnvironmentAllocationStatus.PROVISIONING,
        EnvironmentAllocationStatus.READY,
        EnvironmentAllocationStatus.DRAINING,
        EnvironmentAllocationStatus.RELEASED,
    },
    EnvironmentAllocationStatus.PROVISIONING: {
        EnvironmentAllocationStatus.READY,
        EnvironmentAllocationStatus.DRAINING,
        EnvironmentAllocationStatus.DELETING,
        EnvironmentAllocationStatus.RELEASED,
    },
    EnvironmentAllocationStatus.READY: {EnvironmentAllocationStatus.DRAINING},
    EnvironmentAllocationStatus.DRAINING: {
        EnvironmentAllocationStatus.DELETING,
        EnvironmentAllocationStatus.RELEASED,
    },
    EnvironmentAllocationStatus.DELETING: {EnvironmentAllocationStatus.RELEASED},
    EnvironmentAllocationStatus.RELEASED: set(),
}


class AllocationRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create_allocation(
        self,
        request: CreateEnvironmentAllocation,
        *,
        now: datetime,
    ) -> EnvironmentAllocationRecord:
        _validate_create_request(request)
        environment = await self.session.get(EnvironmentRow, request.environment_id)
        if environment is None:
            raise ResourceNotFoundError('environment_not_found')
        grant = await self.session.get(
            EnvironmentGrantRow,
            (request.owner_principal_id, request.environment_id),
        )
        if environment.status != EnvironmentStatus.ENABLED:
            raise DomainConflictError('environment_disabled')
        if grant is None:
            raise ResourceNotFoundError('environment_grant_not_found')
        if not grant.enabled:
            raise DomainConflictError('environment_grant_disabled')

        existing = await self.session.get(EnvironmentAllocationRow, request.allocation_id)
        if existing is not None:
            expected = (
                request.top_level_execution_id,
                request.owner_principal_id,
                request.environment_id,
                request.pool_id,
                request.workload,
                request.allocation_spec_hash,
            )
            actual = (
                existing.top_level_execution_id,
                existing.owner_principal_id,
                existing.environment_id,
                existing.pool_id,
                existing.workload,
                existing.allocation_spec_hash,
            )
            if actual != expected:
                raise DomainConflictError('allocation_id_conflict')
            return _allocation_record(existing)

        timeout = request.allocation_spec.get('provisioning_timeout_seconds')
        deadline_at = (
            now + timedelta(seconds=timeout)
            if isinstance(timeout, int) and not isinstance(timeout, bool) and timeout > 0
            else None
        )
        row = EnvironmentAllocationRow(
            allocation_id=request.allocation_id,
            top_level_execution_id=request.top_level_execution_id,
            owner_principal_id=request.owner_principal_id,
            environment_id=request.environment_id,
            pool_id=request.pool_id,
            workload=request.workload,
            status=EnvironmentAllocationStatus.REQUESTED,
            desired_manager_count=request.desired_manager_count,
            ready_manager_count=0,
            manager_count=request.resources.manager_count,
            cpu_millicores=request.resources.cpu_millicores,
            memory_bytes=request.resources.memory_bytes,
            slots=request.resources.slots,
            active_batches=request.resources.active_batches,
            allocation_spec_json=dict(request.allocation_spec),
            allocation_spec_hash=request.allocation_spec_hash,
            dependency_build_id=request.dependency_build_id,
            config_version=environment.config_version,
            status_version=1,
            last_progress_at=now,
            provisioning_deadline_at=deadline_at,
            created_at=now,
            updated_at=now,
        )
        self.session.add(row)
        await self.session.flush()
        return _allocation_record(row)

    async def get_allocation(self, allocation_id: str) -> EnvironmentAllocationRecord:
        row = await self.session.get(EnvironmentAllocationRow, allocation_id)
        if row is None:
            raise ResourceNotFoundError('allocation_not_found')
        return _allocation_record(row)

    async def zhiyan_log_binding(
        self, allocation_id: str, *, predefined_log_name: str | None = None
    ) -> ZhiyanLogBinding | None:
        """Return the immutable, non-secret Zhiyan dataflow binding for an allocation.

        Dynamic TKEx allocations carry the binding inside their immutable provider config
        snapshot. Predefined-pool allocations have no provider config, so their instance
        managers share a single, deployment-wide dataflow supplied via ``predefined_log_name``.
        """
        snapshot = await self.session.scalar(
            select(EnvironmentAllocationRow.allocation_spec_json).where(
                EnvironmentAllocationRow.allocation_id == allocation_id,
            )
        )
        if not isinstance(snapshot, dict):
            return None
        provider_type = snapshot.get('provider_type')
        if provider_type == EnvironmentProviderType.PREDEFINED:
            return _clean_zhiyan_binding(predefined_log_name, provider_type=EnvironmentProviderType.PREDEFINED)
        provider_config = snapshot.get('provider_config')
        if provider_type != EnvironmentProviderType.TKEX or not isinstance(provider_config, dict):
            return None
        zhiyan_log = provider_config.get('zhiyan_log')
        if not isinstance(zhiyan_log, dict):
            return None
        return _clean_zhiyan_binding(zhiyan_log.get('log_name'), provider_type=EnvironmentProviderType.TKEX)

    async def list_allocations(
        self,
        *,
        owner_principal_id: str | None = None,
        environment_id: str | None = None,
        status: str | None = None,
        top_level_execution_id: str | None = None,
        allocation_type: str | None = None,
        limit: int = 100,
        offset: int = 0,
        cursor: str | None = None,
    ) -> list[EnvironmentAllocationRecord]:
        query = select(EnvironmentAllocationRow)
        if allocation_type is not None:
            query = query.join(
                EnvironmentRow,
                EnvironmentRow.environment_id == EnvironmentAllocationRow.environment_id,
            ).where(EnvironmentRow.allocation_type == allocation_type)
        if owner_principal_id is not None:
            query = query.where(EnvironmentAllocationRow.owner_principal_id == owner_principal_id)
        if environment_id is not None:
            query = query.where(EnvironmentAllocationRow.environment_id == environment_id)
        if status is not None:
            query = query.where(EnvironmentAllocationRow.status == status)
        if top_level_execution_id is not None:
            query = query.where(EnvironmentAllocationRow.top_level_execution_id == top_level_execution_id)
        if cursor is not None:
            created_at, allocation_id = _decode_allocation_cursor(cursor)
            query = query.where(
                or_(
                    EnvironmentAllocationRow.created_at < created_at,
                    and_(
                        EnvironmentAllocationRow.created_at == created_at,
                        EnvironmentAllocationRow.allocation_id < allocation_id,
                    ),
                )
            )
        rows = (
            await self.session.scalars(
                query
                .order_by(
                    EnvironmentAllocationRow.created_at.desc(),
                    EnvironmentAllocationRow.allocation_id.desc(),
                )
                .offset(max(0, offset))
                .limit(max(0, limit))
            )
        ).all()
        return [_allocation_record(row) for row in rows]

    @staticmethod
    def page_cursor(allocation: EnvironmentAllocationRecord) -> str:
        return encode_cursor({
            'created_at': allocation.created_at.isoformat(),
            'allocation_id': allocation.allocation_id,
        })

    async def count_allocations(
        self,
        *,
        owner_principal_id: str | None = None,
        environment_id: str | None = None,
        status: str | None = None,
        top_level_execution_id: str | None = None,
        allocation_type: str | None = None,
    ) -> int:
        query = select(func.count(EnvironmentAllocationRow.allocation_id))
        if allocation_type is not None:
            query = query.join(
                EnvironmentRow,
                EnvironmentRow.environment_id == EnvironmentAllocationRow.environment_id,
            ).where(EnvironmentRow.allocation_type == allocation_type)
        if owner_principal_id is not None:
            query = query.where(EnvironmentAllocationRow.owner_principal_id == owner_principal_id)
        if environment_id is not None:
            query = query.where(EnvironmentAllocationRow.environment_id == environment_id)
        if status is not None:
            query = query.where(EnvironmentAllocationRow.status == status)
        if top_level_execution_id is not None:
            query = query.where(EnvironmentAllocationRow.top_level_execution_id == top_level_execution_id)
        return int(await self.session.scalar(query) or 0)

    async def force_cleanup(
        self, allocation_id: str, *, now: datetime, expected_version: int | None = None
    ) -> EnvironmentAllocationRecord:
        row = await self.session.scalar(
            select(EnvironmentAllocationRow)
            .where(EnvironmentAllocationRow.allocation_id == allocation_id)
            .with_for_update()
        )
        if row is None:
            raise ResourceNotFoundError('allocation_not_found')
        if expected_version is not None and row.status_version != expected_version:
            raise DomainConflictError('allocation_changed')
        changed = False
        if row.status not in {
            EnvironmentAllocationStatus.DRAINING,
            EnvironmentAllocationStatus.DELETING,
            EnvironmentAllocationStatus.RELEASED,
        }:
            if row.status in {EnvironmentAllocationStatus.REQUESTED, EnvironmentAllocationStatus.PROVISIONING}:
                self._startup_result(row, now=now, outcome='cancelled')
            row.status = EnvironmentAllocationStatus.DRAINING
            row.release_requested_at = row.release_requested_at or now
            row.ready_manager_count = 0
            row.last_error_code = 'admin_force_cleanup'
            row.last_error_message = None
            row.last_progress_at = now
            changed = True
        if row.status != EnvironmentAllocationStatus.RELEASED and any(
            value is not None for value in (row.operation_id, row.operation_owner, row.operation_lease_until)
        ):
            row.operation_id = None
            row.operation_owner = None
            row.operation_lease_until = None
            changed = True
        if changed:
            row.status_version += 1
            row.updated_at = now
            await self.session.flush()
        return _allocation_record(row)

    async def claim_allocation(
        self,
        allocation_id: str,
        *,
        operation_owner: str,
        operation_ttl: timedelta,
        now: datetime,
    ) -> EnvironmentAllocationClaim:
        row = await self.session.scalar(
            select(EnvironmentAllocationRow)
            .where(EnvironmentAllocationRow.allocation_id == allocation_id)
            .with_for_update()
        )
        if row is None:
            raise ResourceNotFoundError('allocation_not_found')
        if row.status == EnvironmentAllocationStatus.RELEASED:
            raise DomainConflictError('allocation_released')
        if row.operation_id is not None and row.operation_lease_until is not None and row.operation_lease_until > now:
            raise FencingError('allocation_operation_leased')
        return self._claim_row(row, operation_owner=operation_owner, operation_ttl=operation_ttl, now=now)

    async def claim_allocations(
        self,
        *,
        operation_owner: str,
        operation_ttl: timedelta,
        statuses: set[str] | None,
        limit: int,
        now: datetime,
        exclude_allocation_ids: set[str] | None = None,
    ) -> list[EnvironmentAllocationClaim]:
        if limit <= 0:
            return []
        selected_statuses = statuses or (set(_TRANSITIONS) - {EnvironmentAllocationStatus.RELEASED})
        query = select(EnvironmentAllocationRow.allocation_id).where(
            EnvironmentAllocationRow.status.in_(selected_statuses),
            or_(
                EnvironmentAllocationRow.operation_id.is_(None),
                EnvironmentAllocationRow.operation_lease_until.is_(None),
                EnvironmentAllocationRow.operation_lease_until <= now,
            ),
        )
        if exclude_allocation_ids:
            query = query.where(EnvironmentAllocationRow.allocation_id.not_in(exclude_allocation_ids))
        candidate_ids = list(
            await self.session.scalars(
                query
                .order_by(EnvironmentAllocationRow.updated_at, EnvironmentAllocationRow.allocation_id)
                .limit(limit)
                .with_for_update(skip_locked=True)
            )
        )
        if not candidate_ids:
            return []
        rows_by_id = {
            row.allocation_id: row
            for row in await self.session.scalars(
                select(EnvironmentAllocationRow).where(EnvironmentAllocationRow.allocation_id.in_(candidate_ids))
            )
        }
        if len(rows_by_id) != len(candidate_ids):
            raise RuntimeError('allocation_claim_not_visible')
        rows = [rows_by_id[allocation_id] for allocation_id in candidate_ids]
        claims = [
            self._claim_row(row, operation_owner=operation_owner, operation_ttl=operation_ttl, now=now) for row in rows
        ]
        await self.session.flush()
        return claims

    async def renew_claim(
        self,
        claim: EnvironmentAllocationClaim,
        *,
        operation_ttl: timedelta,
        now: datetime,
    ) -> bool:
        if operation_ttl <= timedelta(0):
            raise DomainValidationError('operation_ttl_must_be_positive')
        row = await self.session.get(EnvironmentAllocationRow, claim.allocation_id)
        if (
            row is None
            or row.operation_id != claim.operation_id
            or row.operation_owner != claim.operation_owner
            or row.status_version != claim.status_version
            or row.operation_lease_until is None
            or row.operation_lease_until <= now
        ):
            return False
        row.operation_lease_until = now + operation_ttl
        await self.session.flush()
        return True

    def _claim_row(
        self,
        row: EnvironmentAllocationRow,
        *,
        operation_owner: str,
        operation_ttl: timedelta,
        now: datetime,
    ) -> EnvironmentAllocationClaim:
        if operation_ttl <= timedelta(0):
            raise DomainValidationError('operation_ttl_must_be_positive')
        operation_id = new_id('op')
        operation_lease_until = now + operation_ttl
        row.operation_id = operation_id
        row.operation_owner = operation_owner
        row.operation_lease_until = operation_lease_until
        row.status_version += 1
        row.updated_at = now
        return EnvironmentAllocationClaim(
            allocation_id=row.allocation_id,
            operation_id=operation_id,
            operation_owner=operation_owner,
            operation_lease_until=operation_lease_until,
            status=row.status,
            status_version=row.status_version,
        )

    async def transition(
        self,
        claim: EnvironmentAllocationClaim,
        *,
        status: str,
        now: datetime,
        ready_manager_count: int | None = None,
        provider_resource_ref: dict[str, Any] | None = None,
        last_error_code: str | None = None,
        last_error_message: str | None = None,
    ) -> bool:
        row = await self.session.get(EnvironmentAllocationRow, claim.allocation_id)
        if (
            row is None
            or row.operation_id != claim.operation_id
            or row.operation_owner != claim.operation_owner
            or row.status_version != claim.status_version
            or row.operation_lease_until is None
            or row.operation_lease_until <= now
        ):
            return False
        if status != row.status and status not in _TRANSITIONS.get(row.status, set()):
            raise FencingError(
                'invalid_allocation_transition',
                detail=f'{row.status}:{status}',
            )
        environment = await self.session.get(EnvironmentRow, row.environment_id)
        if environment is None:
            raise RuntimeError('allocation_environment_missing')
        allocation_type = _allocation_type_from_spec(row.allocation_spec_json, fallback='')
        if not allocation_type:
            if row.config_version != environment.config_version:
                raise DomainConflictError('allocation_type_snapshot_missing')
            allocation_type = environment.allocation_type
        if status == EnvironmentAllocationStatus.READY:
            if ready_manager_count is None or ready_manager_count < row.desired_manager_count:
                raise DomainConflictError('allocation_managers_not_ready')
            if allocation_type == EnvironmentAllocationType.DYNAMIC and not (
                provider_resource_ref or row.provider_resource_ref_json
            ):
                raise DomainConflictError('provider_resource_ref_required')
        if status == EnvironmentAllocationStatus.RELEASED:
            if (
                allocation_type == EnvironmentAllocationType.DYNAMIC
                and row.status != EnvironmentAllocationStatus.DELETING
                and not (
                    row.status
                    in {
                        EnvironmentAllocationStatus.REQUESTED,
                        EnvironmentAllocationStatus.PROVISIONING,
                        EnvironmentAllocationStatus.DRAINING,
                    }
                    and not row.provider_resource_ref_json
                )
            ):
                raise DomainConflictError('dynamic_allocation_must_be_deleted_before_release')
            if (
                allocation_type == EnvironmentAllocationType.PREDEFINED
                and row.status != EnvironmentAllocationStatus.DRAINING
                and row.status
                not in {
                    EnvironmentAllocationStatus.REQUESTED,
                    EnvironmentAllocationStatus.PROVISIONING,
                }
            ):
                raise DomainConflictError('predefined_allocation_must_be_drained_before_release')
        if ready_manager_count is not None:
            if ready_manager_count < 0 or ready_manager_count > row.desired_manager_count:
                raise DomainValidationError('invalid_ready_manager_count')
            row.ready_manager_count = ready_manager_count
        if provider_resource_ref is not None:
            row.provider_resource_ref_json = dict(provider_resource_ref)
        if row.status != status:
            if status == EnvironmentAllocationStatus.READY:
                self._startup_result(row, now=now, outcome='success')
            elif row.status in {
                EnvironmentAllocationStatus.REQUESTED,
                EnvironmentAllocationStatus.PROVISIONING,
            } and status in {EnvironmentAllocationStatus.DRAINING, EnvironmentAllocationStatus.RELEASED}:
                self._startup_result(row, now=now, outcome='cancelled')
            if status in {EnvironmentAllocationStatus.DRAINING, EnvironmentAllocationStatus.DELETING}:
                row.release_requested_at = row.release_requested_at or now
        row.status = status
        row.last_error_code = last_error_code[:128] if last_error_code else None
        row.last_error_message = last_error_message[:2048] if last_error_message else None
        row.last_progress_at = now
        row.updated_at = now
        if status == EnvironmentAllocationStatus.READY and row.ready_at is None:
            row.ready_at = now
        if status == EnvironmentAllocationStatus.RELEASED:
            row.released_at = now
        row.operation_id = None
        row.operation_owner = None
        row.operation_lease_until = None
        row.status_version += 1
        await self.session.flush()
        return True

    async def fail_provisioning(
        self,
        claim: EnvironmentAllocationClaim,
        *,
        last_error_code: str,
        last_error_message: str,
        now: datetime,
    ) -> str | None:
        row = await self.session.scalar(
            select(EnvironmentAllocationRow)
            .where(EnvironmentAllocationRow.allocation_id == claim.allocation_id)
            .with_for_update()
        )
        if (
            row is None
            or row.operation_id != claim.operation_id
            or row.operation_owner != claim.operation_owner
            or row.status_version != claim.status_version
            or row.operation_lease_until is None
            or row.operation_lease_until <= now
            or row.status
            not in {
                EnvironmentAllocationStatus.REQUESTED,
                EnvironmentAllocationStatus.PROVISIONING,
            }
        ):
            return None
        next_status = (
            EnvironmentAllocationStatus.DRAINING
            if row.provider_resource_ref_json
            else EnvironmentAllocationStatus.RELEASED
        )
        self._startup_result(
            row,
            now=now,
            outcome='timeout' if last_error_code == 'EnvironmentProvisioningTimeoutError' else 'platform_error',
        )
        row.release_requested_at = row.release_requested_at or now
        row.status = next_status
        row.ready_manager_count = 0
        row.last_error_code = last_error_code[:128]
        row.last_error_message = last_error_message[:2048]
        row.last_progress_at = now
        row.updated_at = now
        if next_status == EnvironmentAllocationStatus.RELEASED:
            row.released_at = now
        row.operation_id = None
        row.operation_owner = None
        row.operation_lease_until = None
        row.status_version += 1
        await self.session.flush()
        return next_status

    def _startup_result(self, row: EnvironmentAllocationRow, *, now: datetime, outcome: str) -> None:
        if row.provisioning_started_at is not None:
            queue_stage_result(
                self.session,
                stage='manager_start',
                started_at=row.provisioning_started_at,
                now=now,
                outcome=outcome,
                environment=row.environment_id,
            )

    async def record_provisioning_clock(
        self,
        claim: EnvironmentAllocationClaim,
        *,
        now: datetime,
        deadline_at: datetime | None,
        started: bool = False,
    ) -> bool:
        row = await self.session.scalar(
            select(EnvironmentAllocationRow)
            .where(
                EnvironmentAllocationRow.allocation_id == claim.allocation_id,
                EnvironmentAllocationRow.operation_id == claim.operation_id,
                EnvironmentAllocationRow.operation_owner == claim.operation_owner,
                EnvironmentAllocationRow.status_version == claim.status_version,
                EnvironmentAllocationRow.operation_lease_until > now,
            )
            .with_for_update()
        )
        if row is None:
            return False
        row.provisioning_deadline_at = deadline_at
        if started and row.provisioning_started_at is None:
            row.provisioning_started_at = now
        await self.session.flush()
        return True


def _validate_create_request(request: CreateEnvironmentAllocation) -> None:
    if not all((
        request.allocation_id,
        request.top_level_execution_id,
        request.owner_principal_id,
        request.environment_id,
        request.pool_id,
        request.workload,
        request.allocation_spec_hash,
    )):
        raise DomainValidationError('invalid_allocation_identity')
    if request.desired_manager_count < 0:
        raise DomainValidationError('desired_manager_count_must_be_non_negative')
    validate_environment_id(request.environment_id)
    validate_pool_id(request.pool_id)
    if request.resources.manager_count != request.desired_manager_count:
        raise DomainValidationError('manager_count_mismatch')
    if request.workload not in {'execution', 'orchestration'}:
        raise DomainValidationError('invalid_allocation_workload')


def _allocation_type_from_spec(allocation_spec: object, *, fallback: str) -> str:
    if isinstance(allocation_spec, dict):
        value = cast(dict[str, object], allocation_spec).get('allocation_type')
        if isinstance(value, str) and value in {
            EnvironmentAllocationType.DYNAMIC,
            EnvironmentAllocationType.PREDEFINED,
        }:
            return value
    return fallback


def _allocation_record(row: EnvironmentAllocationRow) -> EnvironmentAllocationRecord:
    from nexus.distributed.environment.liveness import allocation_liveness

    try:
        policy = allocation_liveness(row.allocation_spec_json)
        source = 'allocation' if policy else 'platform'
    except ValueError:
        logging.getLogger(__name__).error('Invalid Manager liveness policy: allocation_id=%s', row.allocation_id)
        policy = None
        source = 'invalid'
    return EnvironmentAllocationRecord(
        allocation_id=row.allocation_id,
        top_level_execution_id=row.top_level_execution_id,
        owner_principal_id=row.owner_principal_id,
        environment_id=row.environment_id,
        pool_id=row.pool_id,
        workload=row.workload,
        status=row.status,
        desired_manager_count=row.desired_manager_count,
        ready_manager_count=row.ready_manager_count,
        manager_count=row.manager_count,
        cpu_millicores=row.cpu_millicores,
        memory_bytes=row.memory_bytes,
        slots=row.slots,
        active_batches=row.active_batches,
        dependency_build_id=row.dependency_build_id,
        provider_resource_ref=(dict(row.provider_resource_ref_json) if row.provider_resource_ref_json else None),
        allocation_spec_hash=row.allocation_spec_hash,
        configured_manager_liveness=policy.snapshot() if policy else None,
        manager_liveness_source=source,
        config_version=row.config_version,
        status_version=row.status_version,
        last_error_code=row.last_error_code,
        last_error_message=row.last_error_message,
        last_progress_at=row.last_progress_at,
        created_at=row.created_at,
        ready_at=row.ready_at,
        released_at=row.released_at,
        updated_at=row.updated_at,
    )
