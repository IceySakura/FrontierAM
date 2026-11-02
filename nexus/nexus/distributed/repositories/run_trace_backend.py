"""Principal grants for configured Run Trace Backends."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from nexus.distributed.errors import DomainValidationError, ResourceNotFoundError
from nexus.distributed.persistence.orm import PrincipalRow, RunTraceBackendGrantRow
from nexus.distributed.utils.identity import validate_run_trace_backend_name


_MAX_DEFAULT_BACKENDS = 8


@dataclass(frozen=True, slots=True)
class RunTraceBackendGrantInput:
    backend_name: str
    enabled: bool = True
    default_order: int | None = None


@dataclass(frozen=True, slots=True)
class RunTraceBackendGrantRecord:
    principal_id: str
    backend_name: str
    enabled: bool
    default_order: int | None
    version: int
    created_by: str
    updated_by: str
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class RunTraceBackendGrantMutation:
    grant: RunTraceBackendGrantRecord
    changed: bool


class RunTraceBackendGrantRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def lock_principal(self, principal_id: str) -> None:
        principal = await self.session.scalar(
            select(PrincipalRow).where(PrincipalRow.principal_id == principal_id).with_for_update()
        )
        if principal is None:
            raise ResourceNotFoundError('user_not_found')

    async def list_for_principal(
        self,
        principal_id: str,
        *,
        for_update: bool = False,
    ) -> list[RunTraceBackendGrantRecord]:
        query = (
            select(RunTraceBackendGrantRow)
            .where(RunTraceBackendGrantRow.principal_id == principal_id)
            .order_by(RunTraceBackendGrantRow.backend_name)
        )
        if for_update:
            query = query.with_for_update()
        rows = (await self.session.scalars(query)).all()
        return [_record(row) for row in rows]

    async def replace_for_principal(
        self,
        principal_id: str,
        grants: Sequence[RunTraceBackendGrantInput],
        *,
        actor: str,
        now: datetime,
    ) -> list[RunTraceBackendGrantRecord]:
        _validate_grant_set(grants)
        await self.lock_principal(principal_id)

        existing_rows = list(
            await self.session.scalars(
                select(RunTraceBackendGrantRow)
                .where(RunTraceBackendGrantRow.principal_id == principal_id)
                .order_by(RunTraceBackendGrantRow.backend_name)
                .with_for_update()
            )
        )
        existing = {row.backend_name: row for row in existing_rows}
        original = {row.backend_name: (row.enabled, row.default_order) for row in existing_rows}

        # Clearing the old positions first makes swaps safe under the unique
        # (principal_id, default_order) constraint on both MySQL and SQLite.
        if any(row.default_order is not None for row in existing_rows):
            for row in existing_rows:
                row.default_order = None
            await self.session.flush()

        requested_names = {grant.backend_name for grant in grants}
        for backend_name, row in existing.items():
            if backend_name not in requested_names:
                await self.session.delete(row)

        for grant in grants:
            row = existing.get(grant.backend_name)
            if row is None:
                self.session.add(
                    RunTraceBackendGrantRow(
                        principal_id=principal_id,
                        backend_name=grant.backend_name,
                        enabled=grant.enabled,
                        default_order=grant.default_order,
                        version=1,
                        created_by=actor,
                        updated_by=actor,
                        created_at=now,
                        updated_at=now,
                    )
                )
                continue
            previous = original[grant.backend_name]
            row.enabled = grant.enabled
            row.default_order = grant.default_order
            if previous != (grant.enabled, grant.default_order):
                row.version += 1
                row.updated_by = actor
                row.updated_at = now

        await self.session.flush()
        return await self.list_for_principal(principal_id)

    async def enable_for_principal(
        self,
        principal_id: str,
        backend_name: str,
        *,
        actor: str,
        now: datetime,
    ) -> RunTraceBackendGrantMutation:
        _validate_grant_set([RunTraceBackendGrantInput(backend_name)])
        await self.lock_principal(principal_id)
        row = await self.session.scalar(
            select(RunTraceBackendGrantRow)
            .where(
                RunTraceBackendGrantRow.principal_id == principal_id,
                RunTraceBackendGrantRow.backend_name == backend_name,
            )
            .with_for_update()
        )
        if row is not None and row.enabled:
            return RunTraceBackendGrantMutation(grant=_record(row), changed=False)
        if row is None:
            row = RunTraceBackendGrantRow(
                principal_id=principal_id,
                backend_name=backend_name,
                enabled=True,
                default_order=None,
                version=1,
                created_by=actor,
                updated_by=actor,
                created_at=now,
                updated_at=now,
            )
            self.session.add(row)
        else:
            row.enabled = True
            row.default_order = None
            row.version += 1
            row.updated_by = actor
            row.updated_at = now
        await self.session.flush()
        return RunTraceBackendGrantMutation(grant=_record(row), changed=True)


def _validate_grant_set(grants: Sequence[RunTraceBackendGrantInput]) -> None:
    names = [grant.backend_name for grant in grants]
    try:
        for name in names:
            validate_run_trace_backend_name(name)
    except ValueError as exc:
        raise DomainValidationError('run_trace_backend_grants_invalid') from exc
    default_orders = [grant.default_order for grant in grants if grant.default_order is not None]
    invalid_default_order = any(
        isinstance(order, bool) or not isinstance(order, int) or order < 0 for order in default_orders
    )
    if (
        len(set(names)) != len(names)
        or invalid_default_order
        or len(set(default_orders)) != len(default_orders)
        or len(default_orders) > _MAX_DEFAULT_BACKENDS
        or sorted(default_orders) != list(range(len(default_orders)))
        or any(not grant.enabled and grant.default_order is not None for grant in grants)
    ):
        raise DomainValidationError('run_trace_backend_grants_invalid')


def _record(row: RunTraceBackendGrantRow) -> RunTraceBackendGrantRecord:
    return RunTraceBackendGrantRecord(
        principal_id=row.principal_id,
        backend_name=row.backend_name,
        enabled=row.enabled,
        default_order=row.default_order,
        version=row.version,
        created_by=row.created_by,
        updated_by=row.updated_by,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


__all__ = [
    'RunTraceBackendGrantInput',
    'RunTraceBackendGrantMutation',
    'RunTraceBackendGrantRecord',
    'RunTraceBackendGrantRepository',
]
