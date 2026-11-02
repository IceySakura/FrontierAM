from __future__ import annotations

from collections.abc import Collection
from datetime import datetime, timedelta
from typing import Any, cast

from sqlalchemy import and_, case, func, select, tuple_
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from nexus.distributed.access.models import EnvironmentGrantRecord
from nexus.distributed.access.permissions import normalize_environment_grant_scopes
from nexus.distributed.environment.limits import (
    effective_environment_resource_limits,
    normalize_environment_configuration,
    normalize_environment_default_quota,
    normalize_environment_resource_limits,
    validate_timeout_defaults,
)
from nexus.distributed.environment.models import (
    EnvironmentAllocationType,
    EnvironmentProviderType,
    EnvironmentRecord,
    EnvironmentStatsRecord,
    EnvironmentStatus,
)
from nexus.distributed.errors import DomainConflictError, DomainValidationError, ResourceNotFoundError
from nexus.distributed.execution.models import AssignmentStatus
from nexus.distributed.instance_manager.liveness import (
    effective_manager_liveness,
    manager_is_healthy_clause,
)
from nexus.distributed.instance_manager.models import EnvironmentManagerInventoryRecord
from nexus.distributed.persistence.orm import (
    AssignmentRow,
    EnvironmentAllocationRow,
    EnvironmentGrantRow,
    EnvironmentResourceUsageRow,
    EnvironmentRow,
    InstanceManagerLivenessRow,
    InstanceManagerRow,
    PrincipalRow,
    UserRoleRow,
)
from nexus.distributed.utils.identity import validate_environment_id


ENVIRONMENT_FIELD_UNSET = object()


class EnvironmentInventoryRepository:
    def __init__(self, session: AsyncSession, *, manager_healthy_ttl: timedelta) -> None:
        self.session = session
        self.manager_healthy_ttl = manager_healthy_ttl

    async def get_environment_stats(
        self,
        environment_id: str,
        *,
        pool_id: str | None = None,
        label_key: str | None = None,
        label_value: str | None = None,
        now: datetime,
    ) -> EnvironmentStatsRecord:
        managers = await self.list_environment_managers(
            environment_id,
            pool_id=pool_id,
            label_key=label_key,
            label_value=label_value,
            now=now,
        )
        return EnvironmentStatsRecord(
            environment_id=environment_id,
            pool_id=pool_id,
            manager_count=len(managers),
            healthy_manager_count=sum(1 for manager in managers if manager.healthy),
            draining_manager_count=sum(1 for manager in managers if manager.draining),
            capacity_total=sum(manager.capacity_total for manager in managers),
            reserved_slots=sum(manager.reserved_slots for manager in managers),
            available_slots=sum(manager.available_slots for manager in managers),
        )

    async def list_environments(self, *, now: datetime) -> list[dict[str, Any]]:
        managers = await self.list_managers(now=now)
        definitions = (
            await self.session.scalars(
                select(EnvironmentRow).order_by(EnvironmentRow.name, EnvironmentRow.environment_id)
            )
        ).all()
        allocations = (
            await self.session.scalars(
                select(EnvironmentAllocationRow).order_by(
                    EnvironmentAllocationRow.environment_id,
                    EnvironmentAllocationRow.pool_id,
                    EnvironmentAllocationRow.created_at,
                    EnvironmentAllocationRow.allocation_id,
                )
            )
        ).all()
        environments = {
            definition.environment_id: _empty_environment(definition.environment_id, definition=definition)
            for definition in definitions
        }
        pools_by_environment: dict[str, set[str]] = {}
        for allocation in allocations:
            record = environments.setdefault(
                allocation.environment_id,
                _empty_environment(allocation.environment_id),
            )
            record['allocations'].append(_allocation_summary(allocation))
            pools_by_environment.setdefault(allocation.environment_id, set()).add(allocation.pool_id)
        for manager in managers:
            record = environments.setdefault(
                manager.environment_id,
                _empty_environment(manager.environment_id),
            )
            record['manager_count'] = int(record['manager_count']) + 1
            record['healthy_manager_count'] = int(record['healthy_manager_count']) + int(manager.healthy)
            record['draining_manager_count'] = int(record['draining_manager_count']) + int(manager.draining)
            record['lost_manager_count'] = int(record['lost_manager_count']) + int(not manager.healthy)
            record['capacity_total'] = int(record['capacity_total']) + manager.capacity_total
            record['reserved_slots'] = int(record['reserved_slots']) + manager.reserved_slots
            record['available_slots'] = int(record['available_slots']) + manager.available_slots
            pools_by_environment.setdefault(manager.environment_id, set()).add(manager.pool_id)
        for environment_id, record in environments.items():
            record['pools'] = sorted(pools_by_environment.get(environment_id, set()))
        return sorted(
            environments.values(), key=lambda item: (item['name'] or item['environment_id'], item['environment_id'])
        )

    async def list_managers(
        self,
        *,
        environment_id: str | None = None,
        pool_id: str | None = None,
        label_key: str | None = None,
        label_value: str | None = None,
        now: datetime,
    ) -> list[EnvironmentManagerInventoryRecord]:
        if environment_id is not None:
            return await self.list_environment_managers(
                environment_id,
                pool_id=pool_id,
                label_key=label_key,
                label_value=label_value,
                now=now,
            )
        query = select(InstanceManagerRow)
        if pool_id:
            query = query.where(InstanceManagerRow.pool_id == pool_id)
        rows = list(
            (
                await self.session.scalars(
                    query.order_by(
                        InstanceManagerRow.environment_id,
                        InstanceManagerRow.pool_id,
                        InstanceManagerRow.manager_id,
                    )
                )
            ).all()
        )
        if label_key:
            rows = [row for row in rows if (row.labels_json or {}).get(label_key) == label_value]
        return await self._manager_records(rows, now=now)

    def _manager_scope(self, *, environment_id=None, pool_id=None, workload=None, allocation_id=None):
        query = select(InstanceManagerRow).outerjoin(
            InstanceManagerLivenessRow,
            and_(
                InstanceManagerLivenessRow.manager_id == InstanceManagerRow.manager_id,
                InstanceManagerLivenessRow.session_id == InstanceManagerRow.session_id,
            ),
        )
        for column, value in (
            (InstanceManagerRow.environment_id, environment_id),
            (InstanceManagerRow.pool_id, pool_id),
            (InstanceManagerRow.workload, workload),
            (InstanceManagerRow.allocation_id, allocation_id),
        ):
            if value is not None:
                query = query.where(column == value)
        return query

    async def list_managers_page(
        self,
        *,
        environment_id: str | None = None,
        pool_id: str | None = None,
        workload: str | None = None,
        allocation_id: str | None = None,
        health: str = 'all',
        draining: bool | None = None,
        after_manager_id: str | None = None,
        limit: int,
        now: datetime,
    ) -> list[EnvironmentManagerInventoryRecord]:
        if health not in {'all', 'healthy', 'lost'}:
            raise DomainValidationError('invalid_manager_health_filter')
        if after_manager_id is not None and (not after_manager_id or len(after_manager_id) > 128):
            raise DomainValidationError('invalid_cursor')
        query = self._manager_scope(
            environment_id=environment_id, pool_id=pool_id, workload=workload, allocation_id=allocation_id
        )
        healthy = manager_is_healthy_clause(now - self.manager_healthy_ttl)
        if health != 'all':
            query = query.where(healthy if health == 'healthy' else ~healthy)
        if draining is not None:
            query = query.where(InstanceManagerRow.draining.is_(draining))
        if after_manager_id is not None:
            query = query.where(InstanceManagerRow.manager_id > after_manager_id)
        rows = list(await self.session.scalars(query.order_by(InstanceManagerRow.manager_id).limit(max(0, limit))))
        return await self._manager_records(rows, now=now)

    async def manager_summary(
        self, *, now: datetime, environment_id=None, pool_id=None, workload=None, allocation_id=None
    ) -> dict[str, Any]:
        manager, liveness = InstanceManagerRow, InstanceManagerLivenessRow
        healthy = manager_is_healthy_clause(now - self.manager_healthy_ttl)
        capacity = case(
            (
                and_(liveness.manager_id.is_not(None), liveness.last_heartbeat_at >= manager.last_heartbeat_at),
                liveness.capacity_total,
            ),
            else_=manager.capacity_total,
        )
        scoped = (
            self
            ._manager_scope(
                environment_id=environment_id, pool_id=pool_id, workload=workload, allocation_id=allocation_id
            )
            .with_only_columns(
                manager.manager_id,
                manager.session_id,
                manager.draining,
                healthy.label('healthy'),
                capacity.label('capacity'),
            )
            .subquery()
        )
        reserved = (
            select(AssignmentRow.manager_id, func.sum(AssignmentRow.slot_cost).label('slots'))
            .join(
                scoped,
                and_(
                    scoped.c.manager_id == AssignmentRow.manager_id,
                    scoped.c.session_id == AssignmentRow.manager_session_id,
                ),
            )
            .where(AssignmentRow.status.in_([AssignmentStatus.PENDING, AssignmentStatus.ACTIVE]))
            .group_by(AssignmentRow.manager_id)
            .subquery()
        )
        used = func.coalesce(reserved.c.slots, 0)
        accepting = and_(scoped.c.healthy, scoped.c.draining.is_(False))

        def total(expression):
            return func.coalesce(func.sum(expression), 0)

        row = (
            await self.session.execute(
                select(
                    func.count(scoped.c.manager_id),
                    total(case((scoped.c.healthy, 1), else_=0)),
                    total(case((and_(scoped.c.healthy, scoped.c.draining), 1), else_=0)),
                    total(case((scoped.c.healthy, scoped.c.capacity), else_=0)),
                    total(case((scoped.c.healthy, used), else_=0)),
                    total(case((and_(accepting, scoped.c.capacity > used), scoped.c.capacity - used), else_=0)),
                )
                .select_from(scoped)
                .outerjoin(reserved, reserved.c.manager_id == scoped.c.manager_id)
            )
        ).one()
        count, live, draining, capacity, reserved_slots, available = map(int, row)
        return {
            'data_status': 'ready',
            'queried_at': now,
            'scope': {
                'environment_id': environment_id,
                'pool_id': pool_id,
                'workload': workload,
                'allocation_id': allocation_id,
            },
            'total': count,
            'healthy': live,
            'draining': draining,
            'accepting': live - draining,
            'lost': count - live,
            'capacity_total': capacity,
            'reserved_slots': reserved_slots,
            'available_slots': available,
        }

    async def manager_pools(
        self,
        *,
        environment_id=None,
        workload=None,
        allocation_id=None,
        after: str | None = None,
        prefix: str | None = None,
        limit: int = 51,
    ) -> list[str]:
        query = (
            self
            ._manager_scope(environment_id=environment_id, workload=workload, allocation_id=allocation_id)
            .with_only_columns(InstanceManagerRow.pool_id)
            .distinct()
        )
        if prefix:
            query = query.where(InstanceManagerRow.pool_id.startswith(prefix, autoescape=True))
        if after is not None:
            query = query.where(InstanceManagerRow.pool_id > after)
        return list(await self.session.scalars(query.order_by(InstanceManagerRow.pool_id).limit(limit)))

    async def list_environment_managers(
        self,
        environment_id: str,
        *,
        pool_id: str | None = None,
        label_key: str | None = None,
        label_value: str | None = None,
        now: datetime,
    ) -> list[EnvironmentManagerInventoryRecord]:
        query = select(InstanceManagerRow).where(InstanceManagerRow.environment_id == environment_id)
        if pool_id:
            query = query.where(InstanceManagerRow.pool_id == pool_id)
        rows = list(
            (
                await self.session.scalars(query.order_by(InstanceManagerRow.pool_id, InstanceManagerRow.manager_id))
            ).all()
        )
        if label_key:
            rows = [row for row in rows if (row.labels_json or {}).get(label_key) == label_value]
        return await self._manager_records(rows, now=now)

    async def _manager_records(
        self, rows: list[InstanceManagerRow], *, now: datetime
    ) -> list[EnvironmentManagerInventoryRecord]:
        reserved_slots = await self._reserved_slots_by_manager(rows)
        liveness_rows = (
            list(
                await self.session.scalars(
                    select(InstanceManagerLivenessRow).where(
                        InstanceManagerLivenessRow.manager_id.in_([row.manager_id for row in rows])
                    )
                )
            )
            if rows
            else []
        )
        liveness_by_id = {row.manager_id: row for row in liveness_rows}
        records: list[EnvironmentManagerInventoryRecord] = []
        for row in rows:
            liveness = effective_manager_liveness(row, liveness_by_id.get(row.manager_id))
            reserved = reserved_slots.get(row.manager_id, 0)
            healthy = liveness.active and now - liveness.last_heartbeat_at <= self.manager_healthy_ttl
            available = max(0, liveness.capacity_total - reserved) if healthy and not row.draining else 0
            records.append(
                EnvironmentManagerInventoryRecord(
                    manager_id=row.manager_id,
                    session_id=row.session_id,
                    endpoint_url=row.endpoint_url,
                    environment_id=row.environment_id,
                    pool_id=row.pool_id,
                    allocation_id=row.allocation_id,
                    workload=row.workload,
                    allocation_spec_hash=row.allocation_spec_hash,
                    owner_principal_id=row.owner_principal_id,
                    labels=dict(row.labels_json or {}),
                    metadata=dict(row.metadata_json or {}),
                    draining=row.draining,
                    bootstrap_version=row.bootstrap_version,
                    artifact_ref_version=row.artifact_ref_version,
                    supported_spec_versions=dict(row.supported_spec_versions_json or {}),
                    capability_snapshot=dict(row.capability_snapshot_json or {}),
                    supported_modes=list(row.supported_modes_json),
                    supported_spec_types=list(row.supported_spec_types_json),
                    supported_instance_modes=list(row.supported_instance_modes_json),
                    capacity_total=liveness.capacity_total,
                    reserved_slots=reserved,
                    available_slots=available,
                    healthy=healthy,
                    observed_capacity_used=liveness.observed_capacity_used,
                    last_heartbeat_at=liveness.last_heartbeat_at,
                    created_at=row.created_at,
                    updated_at=row.updated_at,
                )
            )
        return records

    async def _reserved_slots_by_manager(self, managers: list[InstanceManagerRow]) -> dict[str, int]:
        if not managers:
            return {}
        rows = (
            await self.session.execute(
                select(AssignmentRow.manager_id, func.coalesce(func.sum(AssignmentRow.slot_cost), 0))
                .where(
                    tuple_(AssignmentRow.manager_id, AssignmentRow.manager_session_id).in_([
                        (row.manager_id, row.session_id) for row in managers
                    ]),
                    AssignmentRow.status.in_([AssignmentStatus.PENDING, AssignmentStatus.ACTIVE]),
                )
                .group_by(AssignmentRow.manager_id)
            )
        ).all()
        return {manager_id: int(slot_count or 0) for manager_id, slot_count in rows}

    async def delete_stale_manager(
        self, manager_id: str, *, now: datetime, expected_session_id: str | None = None
    ) -> None:
        threshold = now - self.manager_healthy_ttl
        row = await self.session.scalar(
            select(InstanceManagerRow)
            .where(InstanceManagerRow.manager_id == manager_id)
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        if row is None:
            raise ResourceNotFoundError('manager_not_found')
        liveness = await self.session.scalar(
            select(InstanceManagerLivenessRow)
            .where(InstanceManagerLivenessRow.manager_id == manager_id)
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        if expected_session_id is not None and row.session_id != expected_session_id:
            raise DomainConflictError('manager_session_changed')
        effective = effective_manager_liveness(row, liveness)
        if effective.active and effective.last_heartbeat_at >= threshold:
            raise DomainConflictError(
                'manager_healthy',
                detail=f'manager {manager_id} is healthy',
            )
        active_assignment_count = await self.session.scalar(
            select(func.count(AssignmentRow.assignment_id)).where(
                AssignmentRow.manager_id == manager_id,
                AssignmentRow.status.in_([AssignmentStatus.PENDING, AssignmentStatus.ACTIVE]),
            )
        )
        if active_assignment_count:
            raise DomainConflictError(
                'manager_has_reserved_slots',
                detail=f'manager {manager_id} has reserved slots',
            )
        if liveness is not None:
            await self.session.delete(liveness)
        await self.session.delete(row)


class EnvironmentRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create_environment(
        self,
        *,
        environment_id: str,
        name: str,
        allocation_type: str,
        provider_type: str,
        provider_config_ref: str | None,
        resource_limits: dict[str, Any],
        default_timeouts: dict[str, Any],
        status: str,
        actor: str,
        now: datetime,
        default_quota: dict[str, Any] | None = None,
    ) -> EnvironmentRecord:
        resource_limits, default_timeouts = _validate_environment_configuration(
            allocation_type=allocation_type,
            provider_type=provider_type,
            provider_config_ref=provider_config_ref,
            resource_limits=resource_limits,
            default_timeouts=default_timeouts,
            status=status,
        )
        normalized_default_quota = (
            normalize_environment_default_quota(resource_limits, default_quota) if default_quota is not None else None
        )
        validate_environment_id(environment_id)
        if not name.strip() or len(name) > 128:
            raise DomainValidationError('invalid_environment_identity')
        existing = await self.session.get(EnvironmentRow, environment_id)
        name_owner = await self.session.scalar(select(EnvironmentRow).where(EnvironmentRow.name == name.strip()))
        if existing is not None or name_owner is not None:
            raise DomainConflictError('environment_already_exists')
        row = EnvironmentRow(
            environment_id=environment_id,
            name=name.strip(),
            allocation_type=allocation_type,
            provider_type=provider_type,
            provider_config_ref=provider_config_ref,
            resource_limits_json=dict(resource_limits),
            default_timeouts_json=dict(default_timeouts),
            default_quota_json=normalized_default_quota,
            config_version=1,
            status=status,
            created_by=actor,
            updated_by=actor,
            created_at=now,
            updated_at=now,
        )
        self.session.add(row)
        self.session.add(_new_resource_usage(environment_id=environment_id, principal_id=None, now=now))
        await self.session.flush()
        return _environment_record(row)

    async def get_environment(self, environment_id: str) -> EnvironmentRecord:
        row = await self.session.get(EnvironmentRow, environment_id)
        if row is None:
            raise ResourceNotFoundError('environment_not_found')
        return _environment_record(row)

    async def list_environment_definitions(self, *, include_disabled: bool = True) -> list[EnvironmentRecord]:
        query = select(EnvironmentRow)
        if not include_disabled:
            query = query.where(EnvironmentRow.status == EnvironmentStatus.ENABLED)
        rows = (await self.session.scalars(query.order_by(EnvironmentRow.name, EnvironmentRow.environment_id))).all()
        return [_environment_record(row) for row in rows]

    async def update_environment(
        self,
        environment_id: str,
        *,
        name: str | None = None,
        allocation_type: str | None = None,
        provider_type: str | None = None,
        provider_config_ref: str | None | object = ENVIRONMENT_FIELD_UNSET,
        resource_limits: dict[str, Any] | None = None,
        default_timeouts: dict[str, Any] | None = None,
        default_quota: dict[str, Any] | None | object = ENVIRONMENT_FIELD_UNSET,
        status: str | None = None,
        actor: str,
        now: datetime,
    ) -> EnvironmentRecord:
        if await self.session.get(EnvironmentRow, environment_id) is None:
            raise ResourceNotFoundError('environment_not_found')
        usage = await self._lock_resource_usage(
            environment_id=environment_id,
            principal_id=None,
        )
        row = await self.session.scalar(
            select(EnvironmentRow)
            .where(EnvironmentRow.environment_id == environment_id)
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        if row is None:
            raise ResourceNotFoundError('environment_not_found')
        next_name = row.name if name is None else name.strip()
        if not next_name or len(next_name) > 128:
            raise DomainValidationError('invalid_environment_identity')
        if next_name != row.name:
            name_owner = await self.session.scalar(
                select(EnvironmentRow).where(
                    EnvironmentRow.name == next_name,
                    EnvironmentRow.environment_id != environment_id,
                )
            )
            if name_owner is not None:
                raise DomainConflictError('environment_already_exists')
        next_allocation_type = allocation_type or row.allocation_type
        next_provider_type = provider_type or row.provider_type
        next_provider_config_ref = (
            row.provider_config_ref if provider_config_ref is ENVIRONMENT_FIELD_UNSET else provider_config_ref
        )
        if next_provider_config_ref is not None and not isinstance(next_provider_config_ref, str):
            raise TypeError('provider_config_ref must be a string or null')
        next_resource_limits = (
            dict(resource_limits) if resource_limits is not None else dict(row.resource_limits_json or {})
        )
        next_default_timeouts = (
            dict(default_timeouts) if default_timeouts is not None else dict(row.default_timeouts_json or {})
        )
        default_quota_value = row.default_quota_json if default_quota is ENVIRONMENT_FIELD_UNSET else default_quota
        if default_quota_value is not None and not isinstance(default_quota_value, dict):
            raise TypeError('default_quota must be an object or null')
        next_default_quota = cast(dict[str, Any] | None, default_quota_value)
        next_status = status or row.status
        next_resource_limits, next_default_timeouts = _validate_environment_configuration(
            allocation_type=next_allocation_type,
            provider_type=next_provider_type,
            provider_config_ref=next_provider_config_ref,
            resource_limits=next_resource_limits,
            default_timeouts=next_default_timeouts,
            status=next_status,
        )
        normalized_default_quota = (
            normalize_environment_default_quota(next_resource_limits, next_default_quota)
            if next_default_quota is not None
            else None
        )
        if resource_limits is not None:
            _validate_resource_limits_cover_usage(
                usage,
                next_resource_limits,
                scope='environment',
            )
        changed = (
            row.name,
            row.allocation_type,
            row.provider_type,
            row.provider_config_ref,
            row.resource_limits_json,
            row.default_timeouts_json,
            row.default_quota_json,
            row.status,
        ) != (
            next_name,
            next_allocation_type,
            next_provider_type,
            next_provider_config_ref,
            next_resource_limits,
            next_default_timeouts,
            normalized_default_quota,
            next_status,
        )
        if changed:
            row.name = next_name
            row.allocation_type = next_allocation_type
            row.provider_type = next_provider_type
            row.provider_config_ref = next_provider_config_ref
            row.resource_limits_json = next_resource_limits
            row.default_timeouts_json = next_default_timeouts
            row.default_quota_json = normalized_default_quota
            row.status = next_status
            row.config_version += 1
            row.updated_by = actor
            row.updated_at = now
            await self.session.flush()
        return _environment_record(row)

    async def set_environment_status(
        self,
        environment_id: str,
        *,
        status: str,
        actor: str,
        now: datetime,
    ) -> EnvironmentRecord:
        return await self.update_environment(environment_id, status=status, actor=actor, now=now)

    async def get_grant(self, principal_id: str, environment_id: str) -> EnvironmentGrantRecord:
        row = await self.session.get(EnvironmentGrantRow, (principal_id, environment_id))
        if row is None:
            raise ResourceNotFoundError('environment_grant_not_found')
        return _grant_record(row)

    async def upsert_grant(
        self,
        *,
        principal_id: str,
        environment_id: str,
        limits: dict[str, Any],
        enabled: bool,
        actor: str,
        now: datetime,
        scopes: set[str] | frozenset[str] | None = None,
    ) -> EnvironmentGrantRecord:
        limits = normalize_environment_resource_limits(limits)
        normalized_scopes = normalize_environment_grant_scopes(scopes) if scopes is not None else None
        principal = await self.session.get(PrincipalRow, principal_id)
        if principal is None and await self.session.get(UserRoleRow, principal_id) is None:
            raise ResourceNotFoundError('user_not_found')
        if await self.session.get(EnvironmentRow, environment_id) is None:
            raise ResourceNotFoundError('environment_not_found')
        await self._lock_resource_usage(
            environment_id=environment_id,
            principal_id=None,
        )
        environment = await self.session.scalar(
            select(EnvironmentRow)
            .where(EnvironmentRow.environment_id == environment_id)
            .execution_options(populate_existing=True)
        )
        if environment is None:
            raise ResourceNotFoundError('environment_not_found')
        effective_limits = effective_environment_resource_limits(
            environment.resource_limits_json or {},
            limits,
        )
        validate_timeout_defaults(effective_limits, environment.default_timeouts_json or {})
        row = await self.session.get(EnvironmentGrantRow, (principal_id, environment_id))
        if row is None:
            usage_scope_key = _resource_usage_scope_key(
                environment_id=environment_id,
                principal_id=principal_id,
            )
            usage = await self.session.scalar(
                select(EnvironmentResourceUsageRow)
                .where(EnvironmentResourceUsageRow.scope_key == usage_scope_key)
                .with_for_update()
            )
            if usage is not None:
                _validate_resource_limits_cover_usage(usage, limits, scope='grant')
            row = EnvironmentGrantRow(
                principal_id=principal_id,
                environment_id=environment_id,
                limits_json=dict(limits),
                scopes_json=sorted(
                    normalized_scopes if normalized_scopes is not None else normalize_environment_grant_scopes(None)
                ),
                enabled=enabled,
                version=1,
                created_by=actor,
                updated_by=actor,
                created_at=now,
                updated_at=now,
            )
            self.session.add(row)
            if usage is None:
                self.session.add(_new_resource_usage(environment_id=environment_id, principal_id=principal_id, now=now))
        else:
            usage = await self._lock_resource_usage(
                environment_id=environment_id,
                principal_id=principal_id,
            )
            row = await self.session.scalar(
                select(EnvironmentGrantRow)
                .where(
                    EnvironmentGrantRow.principal_id == principal_id,
                    EnvironmentGrantRow.environment_id == environment_id,
                )
                .with_for_update()
                .execution_options(populate_existing=True)
            )
            if row is None:
                raise ResourceNotFoundError('environment_grant_not_found')
            _validate_resource_limits_cover_usage(usage, limits, scope='grant')
            row.limits_json = dict(limits)
            if normalized_scopes is not None:
                row.scopes_json = sorted(normalized_scopes)
            row.enabled = enabled
            row.version += 1
            row.updated_by = actor
            row.updated_at = now
        await self.session.flush()
        return _grant_record(row)

    async def _lock_resource_usage(
        self,
        *,
        environment_id: str,
        principal_id: str | None,
    ) -> EnvironmentResourceUsageRow:
        scope_key = _resource_usage_scope_key(environment_id=environment_id, principal_id=principal_id)
        row = await self.session.scalar(
            select(EnvironmentResourceUsageRow)
            .where(EnvironmentResourceUsageRow.scope_key == scope_key)
            .with_for_update()
        )
        if row is None:
            raise RuntimeError('environment_resource_usage_unavailable')
        return row

    async def list_grants(
        self,
        *,
        principal_id: str | None = None,
        environment_id: str | None = None,
    ) -> list[EnvironmentGrantRecord]:
        query = select(EnvironmentGrantRow)
        if principal_id is not None:
            query = query.where(EnvironmentGrantRow.principal_id == principal_id)
        if environment_id is not None:
            query = query.where(EnvironmentGrantRow.environment_id == environment_id)
        rows = (
            await self.session.scalars(
                query.order_by(EnvironmentGrantRow.environment_id, EnvironmentGrantRow.principal_id)
            )
        ).all()
        return [_grant_record(row) for row in rows]

    async def list_granted_environments(
        self,
        principal_id: str,
        *,
        include_disabled: bool = True,
    ) -> list[tuple[EnvironmentRecord, EnvironmentGrantRecord]]:
        query = (
            select(EnvironmentRow, EnvironmentGrantRow)
            .join(EnvironmentGrantRow, EnvironmentGrantRow.environment_id == EnvironmentRow.environment_id)
            .where(EnvironmentGrantRow.principal_id == principal_id)
        )
        if not include_disabled:
            query = query.where(
                EnvironmentRow.status == EnvironmentStatus.ENABLED,
                EnvironmentGrantRow.enabled.is_(True),
            )
        rows = (await self.session.execute(query.order_by(EnvironmentRow.name))).all()
        return [(_environment_record(environment), _grant_record(grant)) for environment, grant in rows]

    async def list_environment_catalog(
        self,
        principal_id: str,
    ) -> list[tuple[EnvironmentRecord, EnvironmentGrantRecord | None]]:
        rows = (
            await self.session.execute(
                select(EnvironmentRow, EnvironmentGrantRow)
                .outerjoin(
                    EnvironmentGrantRow,
                    and_(
                        EnvironmentGrantRow.environment_id == EnvironmentRow.environment_id,
                        EnvironmentGrantRow.principal_id == principal_id,
                    ),
                )
                .where(EnvironmentRow.status == EnvironmentStatus.ENABLED)
                .order_by(EnvironmentRow.name, EnvironmentRow.environment_id)
            )
        ).all()
        return [
            (
                _environment_record(environment),
                _grant_record(grant) if grant is not None else None,
            )
            for environment, grant in rows
            if environment.default_quota_json is not None or grant is not None
        ]

    async def claim_default_quota(
        self,
        *,
        principal_id: str,
        environment_id: str,
        actor: str,
        now: datetime,
    ) -> tuple[EnvironmentGrantRecord, bool]:
        principal = await self.session.get(PrincipalRow, principal_id)
        if principal is None and await self.session.get(UserRoleRow, principal_id) is None:
            raise ResourceNotFoundError('user_not_found')
        environment = await self.session.scalar(
            select(EnvironmentRow)
            .where(EnvironmentRow.environment_id == environment_id)
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        if environment is None:
            raise ResourceNotFoundError('environment_not_found')
        if environment.status != EnvironmentStatus.ENABLED:
            raise DomainConflictError('environment_disabled')

        existing = await self.session.scalar(
            select(EnvironmentGrantRow)
            .where(
                EnvironmentGrantRow.principal_id == principal_id,
                EnvironmentGrantRow.environment_id == environment_id,
            )
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        if existing is not None:
            if not existing.enabled:
                raise DomainConflictError('environment_grant_disabled')
            return _grant_record(existing), False
        if environment.default_quota_json is None:
            raise DomainConflictError('environment_default_quota_unavailable')

        quota = normalize_environment_default_quota(
            environment.resource_limits_json or {},
            environment.default_quota_json,
        )
        grant_values = {
            'principal_id': principal_id,
            'environment_id': environment_id,
            'limits_json': dict(quota),
            'scopes_json': sorted(normalize_environment_grant_scopes(None)),
            'enabled': True,
            'version': 1,
            'created_by': actor,
            'updated_by': actor,
            'created_at': now,
            'updated_at': now,
        }
        usage = _new_resource_usage(environment_id=environment_id, principal_id=principal_id, now=now)
        usage_values = {
            column.name: getattr(usage, column.name) for column in EnvironmentResourceUsageRow.__table__.columns
        }
        dialect = self.session.get_bind().dialect.name
        created = True
        if dialect == 'mysql':
            result = await self.session.execute(
                mysql_insert(EnvironmentGrantRow)
                .values(grant_values)
                .on_duplicate_key_update(principal_id=EnvironmentGrantRow.principal_id)
            )
            created = bool(getattr(result, 'rowcount', 0) and getattr(result, 'rowcount', 0) > 0)
            await self.session.execute(
                mysql_insert(EnvironmentResourceUsageRow)
                .values(usage_values)
                .on_duplicate_key_update(scope_key=EnvironmentResourceUsageRow.scope_key)
            )
        elif dialect == 'sqlite':
            result = await self.session.execute(
                sqlite_insert(EnvironmentGrantRow)
                .values(grant_values)
                .on_conflict_do_nothing(index_elements=['principal_id', 'environment_id'])
            )
            created = bool(getattr(result, 'rowcount', 0) and getattr(result, 'rowcount', 0) > 0)
            await self.session.execute(
                sqlite_insert(EnvironmentResourceUsageRow)
                .values(usage_values)
                .on_conflict_do_nothing(index_elements=['scope_key'])
            )
        else:
            self.session.add(EnvironmentGrantRow(**grant_values))
            self.session.add(EnvironmentResourceUsageRow(**usage_values))
        await self.session.flush()
        row = await self.session.scalar(
            select(EnvironmentGrantRow)
            .where(
                EnvironmentGrantRow.principal_id == principal_id,
                EnvironmentGrantRow.environment_id == environment_id,
            )
            .execution_options(populate_existing=True)
        )
        if row is None:
            raise RuntimeError('environment_grant_unavailable')
        if not row.enabled:
            raise DomainConflictError('environment_grant_disabled')
        return _grant_record(row), created

    async def list_granted_environment_ids(
        self,
        principal_id: str,
        *,
        required_scope: str,
    ) -> frozenset[str]:
        rows = await self.session.scalars(
            select(EnvironmentGrantRow).where(
                EnvironmentGrantRow.principal_id == principal_id,
                EnvironmentGrantRow.enabled.is_(True),
            )
        )
        return frozenset(row.environment_id for row in rows if required_scope in set(row.scopes_json or []))

    async def list_granted_owner_environment_pairs(
        self,
        principal_ids: Collection[str],
        *,
        required_scope: str,
    ) -> frozenset[tuple[str, str]]:
        if not principal_ids:
            return frozenset()
        rows = await self.session.scalars(
            select(EnvironmentGrantRow).where(
                EnvironmentGrantRow.principal_id.in_(principal_ids),
                EnvironmentGrantRow.enabled.is_(True),
            )
        )
        return frozenset(
            (row.principal_id, row.environment_id) for row in rows if required_scope in set(row.scopes_json or [])
        )

    async def require_grant_scope(
        self,
        principal_id: str,
        environment_id: str,
        *,
        required_scope: str,
    ) -> EnvironmentGrantRecord:
        grant = await self.session.get(EnvironmentGrantRow, (principal_id, environment_id))
        if grant is None:
            raise ResourceNotFoundError('environment_grant_not_found')
        if not grant.enabled:
            raise DomainConflictError('environment_grant_disabled')
        if required_scope not in set(grant.scopes_json or []):
            raise DomainConflictError('environment_scope_required')
        return _grant_record(grant)

    async def require_submission_access(
        self,
        principal_id: str,
        environment_id: str,
        *,
        required_scope: str | None = None,
    ) -> tuple[EnvironmentRecord, EnvironmentGrantRecord]:
        environment = await self.session.get(EnvironmentRow, environment_id)
        if environment is None:
            raise ResourceNotFoundError('environment_not_found')
        if environment.status != EnvironmentStatus.ENABLED:
            raise DomainConflictError('environment_disabled')
        grant = await self.session.get(EnvironmentGrantRow, (principal_id, environment_id))
        if grant is None:
            raise ResourceNotFoundError('environment_grant_not_found')
        if not grant.enabled:
            raise DomainConflictError('environment_grant_disabled')
        if required_scope is not None and required_scope not in set(grant.scopes_json or []):
            raise DomainConflictError('environment_scope_required')
        return _environment_record(environment), _grant_record(grant)


def _resource_usage_scope_key(*, environment_id: str, principal_id: str | None) -> str:
    return f'environment:{environment_id}' if principal_id is None else f'grant:{environment_id}:{principal_id}'


def _new_resource_usage(
    *,
    environment_id: str,
    principal_id: str | None,
    now: datetime,
) -> EnvironmentResourceUsageRow:
    return EnvironmentResourceUsageRow(
        scope_key=_resource_usage_scope_key(environment_id=environment_id, principal_id=principal_id),
        scope_type='environment' if principal_id is None else 'grant',
        environment_id=environment_id,
        principal_id=principal_id,
        manager_count=0,
        cpu_millicores=0,
        memory_bytes=0,
        slots=0,
        active_batches=0,
        version=1,
        updated_at=now,
    )


def _validate_resource_limits_cover_usage(
    usage: EnvironmentResourceUsageRow,
    limits: dict[str, int],
    *,
    scope: str,
) -> None:
    for dimension in ('manager_count', 'cpu_millicores', 'memory_bytes', 'slots', 'active_batches'):
        limit = limits.get(dimension)
        if limit is not None and getattr(usage, dimension) > limit:
            raise DomainConflictError(f'{scope}_{dimension}_limit_below_usage')


def _validate_environment_configuration(
    *,
    allocation_type: str,
    provider_type: str,
    provider_config_ref: str | None,
    resource_limits: dict[str, Any],
    default_timeouts: dict[str, Any],
    status: str,
) -> tuple[dict[str, int], dict[str, int]]:
    valid_pairs = {
        (EnvironmentAllocationType.PREDEFINED, EnvironmentProviderType.PREDEFINED),
        (EnvironmentAllocationType.DYNAMIC, EnvironmentProviderType.KUBERNETES),
        (EnvironmentAllocationType.DYNAMIC, EnvironmentProviderType.TKEX),
    }
    if (allocation_type, provider_type) not in valid_pairs:
        raise DomainValidationError('invalid_environment_provider')
    if allocation_type == EnvironmentAllocationType.DYNAMIC and not provider_config_ref:
        raise DomainValidationError('provider_config_ref_required')
    if allocation_type == EnvironmentAllocationType.PREDEFINED and provider_config_ref is not None:
        raise DomainValidationError('predefined_provider_config_ref_forbidden')
    if status not in {EnvironmentStatus.ENABLED, EnvironmentStatus.DISABLED}:
        raise DomainValidationError('invalid_environment_status')
    return normalize_environment_configuration(resource_limits, default_timeouts)


def _environment_record(row: EnvironmentRow) -> EnvironmentRecord:
    resource_limits, default_timeouts = normalize_environment_configuration(
        row.resource_limits_json or {},
        row.default_timeouts_json or {},
    )
    return EnvironmentRecord(
        environment_id=row.environment_id,
        name=row.name,
        allocation_type=row.allocation_type,
        provider_type=row.provider_type,
        provider_config_ref=row.provider_config_ref,
        resource_limits=resource_limits,
        default_timeouts=default_timeouts,
        default_quota=(
            normalize_environment_default_quota(resource_limits, row.default_quota_json)
            if row.default_quota_json is not None
            else None
        ),
        config_version=row.config_version,
        status=row.status,
        created_by=row.created_by,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _grant_record(row: EnvironmentGrantRow) -> EnvironmentGrantRecord:
    return EnvironmentGrantRecord(
        principal_id=row.principal_id,
        environment_id=row.environment_id,
        limits=normalize_environment_resource_limits(row.limits_json or {}),
        scopes=frozenset(row.scopes_json or []),
        enabled=row.enabled,
        version=row.version,
        created_by=row.created_by,
        updated_by=row.updated_by,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _empty_environment(environment_id: str, *, definition: EnvironmentRow | None = None) -> dict[str, Any]:
    return {
        'environment_id': environment_id,
        'name': definition.name if definition is not None else None,
        'allocation_type': definition.allocation_type if definition is not None else None,
        'provider_type': definition.provider_type if definition is not None else None,
        'status': definition.status if definition is not None else 'unknown',
        'manager_count': 0,
        'healthy_manager_count': 0,
        'draining_manager_count': 0,
        'lost_manager_count': 0,
        'capacity_total': 0,
        'reserved_slots': 0,
        'available_slots': 0,
        'pools': [],
        'allocations': [],
    }


def _allocation_summary(row: EnvironmentAllocationRow) -> dict[str, Any]:
    return {
        'allocation_id': row.allocation_id,
        'pool_id': row.pool_id,
        'workload': row.workload,
        'status': row.status,
        'desired_manager_count': row.desired_manager_count,
        'ready_manager_count': row.ready_manager_count,
        'manager_count': row.manager_count,
        'provider_resource_ref': dict(row.provider_resource_ref_json or {}),
        'last_error_code': row.last_error_code,
        'last_error_message': row.last_error_message,
        'updated_at': row.updated_at,
    }
