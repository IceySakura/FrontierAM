from __future__ import annotations

import copy

from collections.abc import Collection
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import or_, select
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from nexus.distributed.artifacts.models import ArtifactRef
from nexus.distributed.dependency_build.models import DependencyBuildStatus
from nexus.distributed.dependency_build.wheelhouse import WheelhouseBuildSpec, WheelhouseMetadata, wheelhouse_build_key
from nexus.distributed.errors import DomainConflictError, DomainValidationError, FencingError, ResourceNotFoundError
from nexus.distributed.observability.database import queue_stage_result
from nexus.distributed.persistence.orm import ArtifactRefRow, DependencyBuildRow
from nexus.distributed.utils.identity import new_id


def _mysql_upsert_dependency_build(values: dict[str, Any]) -> Any:
    return mysql_insert(DependencyBuildRow).values(values).on_duplicate_key_update(build_key=values['build_key'])


def _sqlite_upsert_dependency_build(values: dict[str, Any]) -> Any:
    return (
        sqlite_insert(DependencyBuildRow)
        .values(values)
        .on_conflict_do_nothing(index_elements=['owner_principal_id', 'environment_id', 'build_key'])
    )


@dataclass(frozen=True, slots=True)
class DependencyBuildRecord:
    dependency_build_id: str
    build_key: str
    owner_principal_id: str
    environment_id: str
    provider_type: str | None
    provider_config_ref: str | None
    provider_config_snapshot: dict[str, Any] | None
    status: str
    build_spec: WheelhouseBuildSpec
    wheelhouse: ArtifactRef | None
    metadata: WheelhouseMetadata | None
    provider_job_name: str | None
    attempt: int
    operation_id: str | None
    operation_owner: str | None
    operation_lease_until: datetime | None
    next_retry_at: datetime | None
    error_code: str | None
    error_message: str | None
    started_at: datetime | None
    finished_at: datetime | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class DependencyBuildClaim:
    record: DependencyBuildRecord
    operation_id: str
    operation_owner: str
    operation_lease_until: datetime


class DependencyBuildRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_or_create(
        self,
        spec: WheelhouseBuildSpec,
        *,
        provider_type: str,
        provider_config_ref: str,
        provider_config_snapshot: dict[str, Any],
        now: datetime,
    ) -> tuple[DependencyBuildRecord, bool]:
        _validate_provider_target(
            provider_type=provider_type,
            provider_config_ref=provider_config_ref,
            provider_config_snapshot=provider_config_snapshot,
        )
        build_key = wheelhouse_build_key(spec)
        existing = await self._find(spec.owner_principal_id, spec.environment_id, build_key)
        if existing is not None:
            if existing.status != DependencyBuildStatus.READY and not _provider_target_matches(
                existing,
                provider_type=provider_type,
                provider_config_ref=provider_config_ref,
                provider_config_snapshot=provider_config_snapshot,
            ):
                raise DomainConflictError('dependency_build_provider_snapshot_conflict')
            return _record(existing), True

        dependency_build_id = new_id('bld')
        values = {
            'dependency_build_id': dependency_build_id,
            'build_key': build_key,
            'owner_principal_id': spec.owner_principal_id,
            'environment_id': spec.environment_id,
            'provider_type': provider_type,
            'provider_config_ref': provider_config_ref,
            'provider_config_snapshot_json': copy.deepcopy(provider_config_snapshot),
            'status': DependencyBuildStatus.REQUESTED,
            'build_spec_json': spec.model_dump(mode='json'),
            'attempt': 0,
            'created_at': now,
            'updated_at': now,
        }
        dialect = self.session.get_bind().dialect.name
        if dialect == 'mysql':
            await self.session.execute(_mysql_upsert_dependency_build(values))
        elif dialect == 'sqlite':
            await self.session.execute(_sqlite_upsert_dependency_build(values))
        else:
            self.session.add(DependencyBuildRow(**values))
            await self.session.flush()
        current = await self.session.scalar(
            select(DependencyBuildRow)
            .where(
                DependencyBuildRow.owner_principal_id == spec.owner_principal_id,
                DependencyBuildRow.environment_id == spec.environment_id,
                DependencyBuildRow.build_key == build_key,
            )
            .with_for_update()
        )
        if current is None:
            raise RuntimeError('dependency_build_upsert_not_visible')
        return _record(current), current.dependency_build_id != dependency_build_id

    async def get(self, dependency_build_id: str) -> DependencyBuildRecord:
        row = await self.session.get(DependencyBuildRow, dependency_build_id)
        if row is None:
            raise ResourceNotFoundError('dependency_build_not_found')
        return _record(row)

    async def claim_next(
        self,
        *,
        operation_owner: str,
        lease_ttl: timedelta,
        now: datetime,
        exclude_dependency_build_ids: Collection[str] = (),
    ) -> DependencyBuildClaim | None:
        if not operation_owner:
            raise DomainValidationError('dependency_build_operation_owner_required')
        if lease_ttl <= timedelta(0):
            raise DomainValidationError('dependency_build_lease_ttl_must_be_positive')
        query = select(DependencyBuildRow.dependency_build_id).where(
            or_(
                DependencyBuildRow.status.in_({
                    DependencyBuildStatus.REQUESTED,
                    DependencyBuildStatus.BUILDING,
                }),
                DependencyBuildRow.status.in_({DependencyBuildStatus.FAILED, DependencyBuildStatus.READY})
                & DependencyBuildRow.next_retry_at.is_not(None),
            ),
            or_(DependencyBuildRow.next_retry_at.is_(None), DependencyBuildRow.next_retry_at <= now),
            or_(
                DependencyBuildRow.operation_id.is_(None),
                DependencyBuildRow.operation_lease_until.is_(None),
                DependencyBuildRow.operation_lease_until <= now,
            ),
        )
        if exclude_dependency_build_ids:
            query = query.where(DependencyBuildRow.dependency_build_id.not_in(exclude_dependency_build_ids))
        dependency_build_id = await self.session.scalar(
            query
            .order_by(DependencyBuildRow.updated_at, DependencyBuildRow.dependency_build_id)
            .limit(1)
            .with_for_update(skip_locked=True)
        )
        if dependency_build_id is None:
            return None
        row = await self.session.scalar(
            select(DependencyBuildRow).where(DependencyBuildRow.dependency_build_id == dependency_build_id)
        )
        if row is None:
            raise RuntimeError('dependency_build_claim_not_visible')
        operation_id = new_id('op')
        operation_lease_until = now + lease_ttl
        row.operation_id = operation_id
        row.operation_owner = operation_owner
        row.operation_lease_until = operation_lease_until
        row.updated_at = now
        await self.session.flush()
        return DependencyBuildClaim(
            record=_record(row),
            operation_id=operation_id,
            operation_owner=operation_owner,
            operation_lease_until=operation_lease_until,
        )

    async def renew(
        self,
        dependency_build_id: str,
        *,
        operation_id: str,
        lease_ttl: timedelta,
        now: datetime,
    ) -> DependencyBuildRecord:
        if lease_ttl <= timedelta(0):
            raise DomainValidationError('dependency_build_lease_ttl_must_be_positive')
        row = await self._locked_claim(dependency_build_id, operation_id=operation_id, now=now)
        row.operation_lease_until = now + lease_ttl
        row.updated_at = now
        await self.session.flush()
        return _record(row)

    async def begin_attempt(
        self,
        dependency_build_id: str,
        *,
        operation_id: str,
        attempt: int,
        provider_job_name: str,
        now: datetime,
        deadline_seconds: int | None = None,
    ) -> DependencyBuildRecord:
        if not provider_job_name:
            raise DomainValidationError('dependency_build_provider_job_required')
        row = await self._locked_claim(dependency_build_id, operation_id=operation_id, now=now)
        if row.status not in {DependencyBuildStatus.REQUESTED, DependencyBuildStatus.FAILED}:
            raise DomainConflictError('dependency_build_attempt_already_started')
        if attempt != row.attempt + 1:
            raise FencingError('dependency_build_attempt_mismatch')
        row.status = DependencyBuildStatus.BUILDING
        row.attempt = attempt
        row.provider_job_name = provider_job_name
        row.wheelhouse_ref_json = None
        row.manifest_json = None
        row.next_retry_at = None
        row.error_code = None
        row.error_message = None
        row.started_at = now
        row.deadline_at = now + timedelta(seconds=deadline_seconds) if deadline_seconds is not None else None
        row.finished_at = None
        row.updated_at = now
        await self.session.flush()
        return _record(row)

    async def mark_ready(
        self,
        dependency_build_id: str,
        *,
        operation_id: str,
        wheelhouse: ArtifactRef,
        metadata: WheelhouseMetadata,
        now: datetime,
        cleanup_at: datetime | None = None,
    ) -> DependencyBuildRecord:
        row = await self._locked_claim(dependency_build_id, operation_id=operation_id, now=now)
        if wheelhouse.kind != 'wheelhouse':
            raise DomainValidationError('dependency_build_output_must_be_wheelhouse')
        if metadata.build_key != row.build_key:
            raise FencingError('dependency_build_metadata_key_mismatch')
        if row.status == DependencyBuildStatus.BUILDING:
            queue_stage_result(
                self.session,
                stage='dependency_build',
                started_at=row.started_at,
                now=now,
                outcome='success',
                environment=row.environment_id,
            )
        row.status = DependencyBuildStatus.READY
        row.wheelhouse_ref_json = wheelhouse.model_dump(mode='json')
        row.manifest_json = metadata.model_dump(mode='json')
        # READY records reuse the reconcile due time for background resource cleanup.
        row.next_retry_at = cleanup_at
        row.error_code = None
        row.error_message = None
        row.finished_at = now
        self._clear_operation(row)
        row.updated_at = now
        ref_key = {
            'artifact_id': wheelhouse.artifact_id,
            'resource_type': 'dependency_build',
            'resource_id': row.dependency_build_id,
        }
        if await self.session.get(ArtifactRefRow, ref_key) is None:
            self.session.add(ArtifactRefRow(**ref_key, created_at=now))
        await self.session.flush()
        return _record(row)

    async def stage_output(
        self,
        dependency_build_id: str,
        *,
        attempt: int,
        wheelhouse: ArtifactRef,
        metadata: WheelhouseMetadata,
        now: datetime,
    ) -> DependencyBuildRecord:
        row = await self.session.scalar(
            select(DependencyBuildRow)
            .where(DependencyBuildRow.dependency_build_id == dependency_build_id)
            .with_for_update()
        )
        if row is None:
            raise ResourceNotFoundError('dependency_build_not_found')
        if row.status != DependencyBuildStatus.BUILDING:
            raise DomainConflictError('dependency_build_not_building')
        if row.attempt != attempt:
            raise FencingError('dependency_build_attempt_mismatch')
        if wheelhouse.kind != 'wheelhouse' or metadata.build_key != row.build_key:
            raise FencingError('dependency_build_output_mismatch')
        next_ref = wheelhouse.model_dump(mode='json')
        next_metadata = metadata.model_dump(mode='json')
        if row.manifest_json is not None and row.manifest_json != next_metadata:
            raise DomainConflictError('dependency_build_output_conflict')
        if row.wheelhouse_ref_json is not None and row.wheelhouse_ref_json != next_ref:
            raise DomainConflictError('dependency_build_output_conflict')
        row.wheelhouse_ref_json = next_ref
        row.manifest_json = next_metadata
        row.updated_at = now
        await self.session.flush()
        return _record(row)

    async def stage_metadata(
        self,
        dependency_build_id: str,
        *,
        attempt: int,
        metadata: WheelhouseMetadata,
        now: datetime,
    ) -> DependencyBuildRecord:
        row = await self.session.scalar(
            select(DependencyBuildRow)
            .where(DependencyBuildRow.dependency_build_id == dependency_build_id)
            .with_for_update()
        )
        if row is None:
            raise ResourceNotFoundError('dependency_build_not_found')
        if row.attempt != attempt:
            raise FencingError('dependency_build_attempt_mismatch')
        next_metadata = metadata.model_dump(mode='json')
        if row.status == DependencyBuildStatus.READY and row.manifest_json == next_metadata:
            return _record(row)
        if row.status != DependencyBuildStatus.BUILDING:
            raise DomainConflictError('dependency_build_not_building')
        if metadata.build_key != row.build_key:
            raise FencingError('dependency_build_metadata_key_mismatch')
        if row.manifest_json is not None:
            if row.manifest_json != next_metadata:
                raise DomainConflictError('dependency_build_metadata_conflict')
            return _record(row)
        row.manifest_json = next_metadata
        row.updated_at = now
        await self.session.flush()
        return _record(row)

    async def stage_failure(
        self,
        dependency_build_id: str,
        *,
        attempt: int,
        error_code: str,
        error_message: str,
        now: datetime,
    ) -> DependencyBuildRecord:
        row = await self.session.scalar(
            select(DependencyBuildRow)
            .where(DependencyBuildRow.dependency_build_id == dependency_build_id)
            .with_for_update()
        )
        if row is None:
            raise ResourceNotFoundError('dependency_build_not_found')
        if row.status != DependencyBuildStatus.BUILDING:
            raise DomainConflictError('dependency_build_not_building')
        if row.attempt != attempt:
            raise FencingError('dependency_build_attempt_mismatch')
        if row.wheelhouse_ref_json is not None:
            raise DomainConflictError('dependency_build_output_already_staged')
        next_code = error_code[:128]
        next_message = error_message[:2048]
        if row.error_code is not None and (row.error_code != next_code or row.error_message != next_message):
            raise DomainConflictError('dependency_build_failure_conflict')
        row.error_code = next_code
        row.error_message = next_message
        row.updated_at = now
        await self.session.flush()
        return _record(row)

    async def stage_failure_diagnostics(
        self,
        dependency_build_id: str,
        *,
        operation_id: str,
        attempt: int,
        error_code: str,
        error_message: str,
        now: datetime,
    ) -> DependencyBuildRecord:
        row = await self._locked_claim(dependency_build_id, operation_id=operation_id, now=now)
        if row.status != DependencyBuildStatus.BUILDING:
            raise DomainConflictError('dependency_build_not_building')
        if row.attempt != attempt:
            raise FencingError('dependency_build_attempt_mismatch')
        if row.wheelhouse_ref_json is not None:
            raise DomainConflictError('dependency_build_output_already_staged')
        row.error_code = row.error_code or error_code[:128]
        row.error_message = error_message[:2048]
        row.updated_at = now
        await self.session.flush()
        return _record(row)

    async def release_claim(
        self,
        dependency_build_id: str,
        *,
        operation_id: str,
        now: datetime,
        next_retry_at: datetime | None = None,
    ) -> DependencyBuildRecord:
        row = await self._locked_claim(dependency_build_id, operation_id=operation_id, now=now)
        row.next_retry_at = next_retry_at
        self._clear_operation(row)
        row.updated_at = now
        await self.session.flush()
        return _record(row)

    async def mark_failed(
        self,
        dependency_build_id: str,
        *,
        operation_id: str,
        error_code: str,
        error_message: str,
        now: datetime,
        retry_at: datetime | None = None,
    ) -> DependencyBuildRecord:
        row = await self._locked_claim(dependency_build_id, operation_id=operation_id, now=now)
        if row.status == DependencyBuildStatus.BUILDING:
            queue_stage_result(
                self.session,
                stage='dependency_build',
                started_at=row.started_at,
                now=now,
                outcome='unknown_error',
                environment=row.environment_id,
            )
        row.status = DependencyBuildStatus.FAILED
        row.error_code = error_code[:128]
        row.error_message = error_message[:2048]
        row.next_retry_at = retry_at
        row.finished_at = now
        self._clear_operation(row)
        row.updated_at = now
        await self.session.flush()
        return _record(row)

    async def mark_cancelled(
        self,
        dependency_build_id: str,
        *,
        operation_id: str,
        now: datetime,
    ) -> DependencyBuildRecord:
        row = await self._locked_claim(dependency_build_id, operation_id=operation_id, now=now)
        if row.status == DependencyBuildStatus.BUILDING:
            queue_stage_result(
                self.session,
                stage='dependency_build',
                started_at=row.started_at,
                now=now,
                outcome='cancelled',
                environment=row.environment_id,
            )
        row.status = DependencyBuildStatus.CANCELLED
        row.finished_at = now
        self._clear_operation(row)
        row.updated_at = now
        await self.session.flush()
        return _record(row)

    async def _find(self, owner_principal_id: str, environment_id: str, build_key: str) -> DependencyBuildRow | None:
        return await self.session.scalar(
            select(DependencyBuildRow).where(
                DependencyBuildRow.owner_principal_id == owner_principal_id,
                DependencyBuildRow.environment_id == environment_id,
                DependencyBuildRow.build_key == build_key,
            )
        )

    async def _locked_claim(
        self,
        dependency_build_id: str,
        *,
        operation_id: str,
        now: datetime,
    ) -> DependencyBuildRow:
        row = await self.session.scalar(
            select(DependencyBuildRow)
            .where(DependencyBuildRow.dependency_build_id == dependency_build_id)
            .with_for_update()
        )
        if (
            row is None
            or row.operation_id != operation_id
            or row.operation_lease_until is None
            or row.operation_lease_until <= now
        ):
            raise FencingError('dependency_build_lease_lost')
        return row

    @staticmethod
    def _clear_operation(row: DependencyBuildRow) -> None:
        row.operation_id = None
        row.operation_owner = None
        row.operation_lease_until = None


def _record(row: DependencyBuildRow) -> DependencyBuildRecord:
    return DependencyBuildRecord(
        dependency_build_id=row.dependency_build_id,
        build_key=row.build_key,
        owner_principal_id=row.owner_principal_id,
        environment_id=row.environment_id,
        provider_type=row.provider_type,
        provider_config_ref=row.provider_config_ref,
        provider_config_snapshot=copy.deepcopy(row.provider_config_snapshot_json),
        status=row.status,
        build_spec=WheelhouseBuildSpec.model_validate(row.build_spec_json),
        wheelhouse=ArtifactRef.model_validate(row.wheelhouse_ref_json) if row.wheelhouse_ref_json else None,
        metadata=WheelhouseMetadata.model_validate(row.manifest_json) if row.manifest_json else None,
        provider_job_name=row.provider_job_name,
        attempt=row.attempt,
        operation_id=row.operation_id,
        operation_owner=row.operation_owner,
        operation_lease_until=row.operation_lease_until,
        next_retry_at=row.next_retry_at,
        error_code=row.error_code,
        error_message=row.error_message,
        started_at=row.started_at,
        finished_at=row.finished_at,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _validate_provider_target(
    *,
    provider_type: str,
    provider_config_ref: str,
    provider_config_snapshot: dict[str, Any],
) -> None:
    if provider_type not in {'kubernetes', 'tkex'}:
        raise DomainValidationError('dependency_build_provider_type_invalid')
    if not provider_config_ref:
        raise DomainValidationError('dependency_build_provider_config_ref_required')
    if not isinstance(provider_config_snapshot, dict) or not provider_config_snapshot:
        raise DomainValidationError('dependency_build_provider_config_snapshot_required')
    if provider_config_snapshot.get('provider_type') != provider_type:
        raise DomainValidationError('dependency_build_provider_config_snapshot_type_mismatch')


def _provider_target_matches(
    row: DependencyBuildRow,
    *,
    provider_type: str,
    provider_config_ref: str,
    provider_config_snapshot: dict[str, Any],
) -> bool:
    return (
        row.provider_type == provider_type
        and row.provider_config_ref == provider_config_ref
        and row.provider_config_snapshot_json == provider_config_snapshot
    )


__all__ = ['DependencyBuildClaim', 'DependencyBuildRecord', 'DependencyBuildRepository']
