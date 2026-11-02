from __future__ import annotations

import hashlib

from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from nexus.distributed.access.models import TraceDatabaseAccountRecord
from nexus.distributed.errors import DomainConflictError, ResourceNotFoundError
from nexus.distributed.persistence.orm import (
    EnvironmentGrantRow,
    EnvironmentRow,
    RunTraceBackendGrantRow,
    TraceDatabaseAccountRow,
)
from nexus.distributed.repositories.run_trace_backend import RunTraceBackendGrantRepository
from nexus.distributed.utils.identity import new_id


_USERNAME_DOMAIN = b'nexus-trace-database-user:v1\0'


@dataclass(frozen=True, slots=True)
class TraceDatabaseOperationClaim:
    owner_principal_id: str
    backend_name: str
    username: str
    operation_id: str
    operation: str
    previous_status: str


class TraceDatabaseAccountRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def require_environment_access(self, *, owner_principal_id: str, environment_id: str) -> None:
        granted_environment_id = await self.session.scalar(
            select(EnvironmentGrantRow.environment_id)
            .join(EnvironmentRow, EnvironmentRow.environment_id == EnvironmentGrantRow.environment_id)
            .where(
                EnvironmentGrantRow.principal_id == owner_principal_id,
                EnvironmentGrantRow.environment_id == environment_id,
                EnvironmentGrantRow.enabled.is_(True),
                EnvironmentRow.status == 'enabled',
            )
        )
        if granted_environment_id is None:
            raise ResourceNotFoundError('environment_not_found')

    async def require_backend_access(self, *, owner_principal_id: str, backend_name: str) -> None:
        await RunTraceBackendGrantRepository(self.session).lock_principal(owner_principal_id)
        granted_backend_name = await self.session.scalar(
            select(RunTraceBackendGrantRow.backend_name)
            .where(
                RunTraceBackendGrantRow.principal_id == owner_principal_id,
                RunTraceBackendGrantRow.backend_name == backend_name,
                RunTraceBackendGrantRow.enabled.is_(True),
            )
            .with_for_update()
        )
        if granted_backend_name is None:
            raise ResourceNotFoundError('trace_database_backend_not_found')

    async def list_enabled_backend_grants(self, owner_principal_id: str) -> list[str]:
        rows = await self.session.scalars(
            select(RunTraceBackendGrantRow.backend_name)
            .where(
                RunTraceBackendGrantRow.principal_id == owner_principal_id,
                RunTraceBackendGrantRow.enabled.is_(True),
            )
            .order_by(RunTraceBackendGrantRow.backend_name)
        )
        return list(rows)

    async def has_any_environment_access(
        self,
        owner_principal_id: str,
        *,
        exclude_environment_id: str | None = None,
    ) -> bool:
        filters = [
            EnvironmentGrantRow.principal_id == owner_principal_id,
            EnvironmentGrantRow.enabled.is_(True),
            EnvironmentRow.status == 'enabled',
        ]
        if exclude_environment_id is not None:
            filters.append(EnvironmentGrantRow.environment_id != exclude_environment_id)
        environment_id = await self.session.scalar(
            select(EnvironmentGrantRow.environment_id)
            .join(EnvironmentRow, EnvironmentRow.environment_id == EnvironmentGrantRow.environment_id)
            .where(*filters)
            .limit(1)
        )
        return environment_id is not None

    async def claim_operation(
        self,
        *,
        owner_principal_id: str,
        backend_name: str,
        operation: str,
        now: datetime,
        operation_ttl: timedelta,
        allow_disabled_reset: bool = False,
    ) -> TraceDatabaseOperationClaim:
        row = await self.session.scalar(
            select(TraceDatabaseAccountRow)
            .where(
                TraceDatabaseAccountRow.owner_principal_id == owner_principal_id,
                TraceDatabaseAccountRow.backend_name == backend_name,
            )
            .with_for_update()
        )
        if row is not None and row.operation_id and row.operation_expires_at and row.operation_expires_at > now:
            raise DomainConflictError('trace_database_account_operation_in_progress')
        previous_status = 'disabled' if row is not None and row.disabled_at is not None else 'ready'
        if operation == 'create':
            if row is not None and row.status in {'ready', 'disabled'}:
                raise DomainConflictError('trace_database_account_already_exists')
            if row is None:
                row = TraceDatabaseAccountRow(
                    owner_principal_id=owner_principal_id,
                    backend_name=backend_name,
                    db_username=trace_database_username(owner_principal_id, backend_name),
                    status='creating',
                    credential_version=0,
                    created_at=now,
                    updated_at=now,
                )
                self.session.add(row)
        elif operation == 'reset':
            if row is None:
                raise ResourceNotFoundError('trace_database_account_not_found')
            if previous_status == 'disabled' and not allow_disabled_reset:
                raise DomainConflictError('trace_database_account_disabled')
        elif operation in {'disable', 'restore', 'reconcile'}:
            if row is None:
                raise ResourceNotFoundError('trace_database_account_not_found')
            if operation == 'disable' and previous_status == 'disabled':
                raise DomainConflictError('trace_database_account_already_disabled')
            if operation == 'restore' and previous_status != 'disabled':
                raise DomainConflictError('trace_database_account_not_disabled')
        else:
            raise ValueError(f'unsupported trace database operation: {operation}')
        assert row is not None
        operation_id = new_id('tdop')
        row.status = {
            'create': 'creating',
            'reset': 'resetting',
            'disable': 'disabling',
            'restore': 'restoring',
            'reconcile': 'reconciling',
        }[operation]
        row.operation_id = operation_id
        row.operation_type = operation
        row.operation_expires_at = now + operation_ttl
        row.last_error_code = None
        row.updated_at = now
        await self.session.flush()
        return TraceDatabaseOperationClaim(
            owner_principal_id=owner_principal_id,
            backend_name=backend_name,
            username=row.db_username,
            operation_id=operation_id,
            operation=operation,
            previous_status=previous_status,
        )

    async def finish_operation(
        self,
        claim: TraceDatabaseOperationClaim,
        *,
        status: str,
        now: datetime,
    ) -> TraceDatabaseAccountRecord:
        row = await self._claimed_row(claim)
        row.status = status
        if claim.operation in {'create', 'reset'}:
            row.credential_version += 1
        row.disabled_at = row.disabled_at or now if status == 'disabled' else None
        row.operation_id = None
        row.operation_type = None
        row.operation_expires_at = None
        row.last_error_code = None
        row.updated_at = now
        await self.session.flush()
        return _account_record(row)

    async def renew_operation(
        self,
        claim: TraceDatabaseOperationClaim,
        *,
        now: datetime,
        operation_ttl: timedelta,
    ) -> None:
        row = await self._claimed_row(claim)
        row.operation_expires_at = now + operation_ttl
        row.updated_at = now
        await self.session.flush()

    async def fail_operation(
        self,
        claim: TraceDatabaseOperationClaim,
        *,
        error_code: str,
        now: datetime,
    ) -> None:
        try:
            row = await self._claimed_row(claim)
        except DomainConflictError:
            return
        row.status = 'error'
        row.operation_id = None
        row.operation_type = None
        row.operation_expires_at = None
        row.last_error_code = error_code[:128]
        row.updated_at = now
        await self.session.flush()

    async def get_account(
        self,
        *,
        owner_principal_id: str,
        backend_name: str,
    ) -> TraceDatabaseAccountRecord | None:
        row = await self.session.get(TraceDatabaseAccountRow, (owner_principal_id, backend_name))
        return _account_record(row) if row is not None else None

    async def list_accounts(self, *, owner_principal_id: str | None = None) -> list[TraceDatabaseAccountRecord]:
        query = select(TraceDatabaseAccountRow)
        if owner_principal_id is not None:
            query = query.where(TraceDatabaseAccountRow.owner_principal_id == owner_principal_id)
        rows = (
            await self.session.scalars(
                query.order_by(TraceDatabaseAccountRow.owner_principal_id, TraceDatabaseAccountRow.backend_name)
            )
        ).all()
        return [_account_record(row) for row in rows]

    async def list_account_owners_with_environment_access(self, environment_id: str) -> list[str]:
        rows = await self.session.scalars(
            select(TraceDatabaseAccountRow.owner_principal_id)
            .join(
                EnvironmentGrantRow,
                EnvironmentGrantRow.principal_id == TraceDatabaseAccountRow.owner_principal_id,
            )
            .where(
                EnvironmentGrantRow.environment_id == environment_id,
                EnvironmentGrantRow.enabled.is_(True),
            )
            .distinct()
            .order_by(TraceDatabaseAccountRow.owner_principal_id)
        )
        return list(rows)

    async def _claimed_row(self, claim: TraceDatabaseOperationClaim) -> TraceDatabaseAccountRow:
        row = await self.session.scalar(
            select(TraceDatabaseAccountRow)
            .where(
                TraceDatabaseAccountRow.owner_principal_id == claim.owner_principal_id,
                TraceDatabaseAccountRow.backend_name == claim.backend_name,
            )
            .with_for_update()
        )
        if row is None or row.operation_id != claim.operation_id:
            raise DomainConflictError('trace_database_account_operation_fenced')
        return row


def trace_database_username(owner_principal_id: str, backend_name: str) -> str:
    identity = f'{owner_principal_id}\0{backend_name}'.encode()
    return f'nxs_ro_{hashlib.sha256(_USERNAME_DOMAIN + identity).hexdigest()[:24]}'


def _account_record(row: TraceDatabaseAccountRow) -> TraceDatabaseAccountRecord:
    return TraceDatabaseAccountRecord(
        owner_principal_id=row.owner_principal_id,
        backend_name=row.backend_name,
        username=row.db_username,
        status=row.status,
        credential_version=row.credential_version,
        last_error_code=row.last_error_code,
        created_at=row.created_at,
        updated_at=row.updated_at,
        disabled_at=row.disabled_at,
    )


__all__ = [
    'TraceDatabaseAccountRepository',
    'TraceDatabaseOperationClaim',
    'trace_database_username',
]
