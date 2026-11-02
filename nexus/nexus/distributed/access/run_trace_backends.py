"""Authorization policy for selectable Run Trace Backends."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from nexus.distributed.config import DistributedConfig, RunTraceBackendMetadata, run_trace_backend_catalog
from nexus.distributed.errors import DomainConflictError, DomainValidationError
from nexus.distributed.execution.trace_backends import prepare_configured_run_trace_backend_scope
from nexus.distributed.persistence.orm import TraceDatabaseAccountRow
from nexus.distributed.repositories.run_trace_backend import (
    RunTraceBackendGrantInput,
    RunTraceBackendGrantMutation,
    RunTraceBackendGrantRecord,
    RunTraceBackendGrantRepository,
)
from nexus.runner.trace import principal_trace_scope_id


_MAX_SELECTED_BACKENDS = 8


@dataclass(frozen=True, slots=True)
class RunTraceBackendView:
    name: str
    kind: str
    display_name: str
    description: str | None
    default_order: int | None


@dataclass(frozen=True, slots=True)
class ResolvedRunTraceBackendSelection:
    sinks: tuple[str, ...] | None
    selection_source: Literal['requested', 'principal_default', 'legacy_default']


@dataclass(frozen=True, slots=True)
class RunTraceBackendGrantReplacement:
    before: tuple[RunTraceBackendGrantRecord, ...]
    after: tuple[RunTraceBackendGrantRecord, ...]


@dataclass(frozen=True, slots=True)
class RunTraceBackendAccessView:
    items: tuple[RunTraceBackendView, ...]
    claimable_sinks: tuple[RunTraceBackendMetadata, ...]


class RunTraceBackendAccessPolicy:
    def __init__(self, session: AsyncSession, *, config: DistributedConfig) -> None:
        self.session = session
        self.config = config
        self._backends = {backend.name: backend for backend in run_trace_backend_catalog(config)}

    async def list_for_principal(self, principal_id: str) -> list[RunTraceBackendView]:
        return list((await self.list_access_for_principal(principal_id)).items)

    async def list_access_for_principal(self, principal_id: str) -> RunTraceBackendAccessView:
        grants = await RunTraceBackendGrantRepository(self.session).list_for_principal(principal_id)
        views = [self._view(grant) for grant in grants if grant.enabled and self._is_selectable(grant.backend_name)]
        items = tuple(
            sorted(
                views,
                key=lambda item: (
                    item.default_order is None,
                    item.default_order if item.default_order is not None else 0,
                    item.name,
                ),
            )
        )
        enabled_names = {grant.backend_name for grant in grants if grant.enabled}
        claimable_sinks = tuple(
            sorted(
                (
                    backend
                    for backend in self._backends.values()
                    if backend.self_grantable
                    and self._is_selectable(backend.name)
                    and backend.name not in enabled_names
                ),
                key=lambda backend: backend.name,
            )
        )
        return RunTraceBackendAccessView(items=items, claimable_sinks=claimable_sinks)

    async def self_grant(
        self,
        principal_id: str,
        backend_name: str,
        *,
        actor: str,
        now: datetime,
    ) -> RunTraceBackendGrantMutation:
        backend = self._backends.get(backend_name)
        if backend is None or not backend.self_grantable or not self._is_selectable(backend_name):
            raise DomainValidationError('run_trace_backend_not_available')
        mutation = await RunTraceBackendGrantRepository(self.session).enable_for_principal(
            principal_id,
            backend_name,
            actor=actor,
            now=now,
        )
        if mutation.changed:
            await prepare_configured_run_trace_backend_scope(
                self.config,
                backend_name=backend_name,
                scope_id=principal_trace_scope_id(principal_id),
            )
        return mutation

    async def replace_grants(
        self,
        principal_id: str,
        grants: Sequence[RunTraceBackendGrantInput],
        *,
        actor: str,
        now: datetime,
    ) -> RunTraceBackendGrantReplacement:
        if any(not self._is_selectable(grant.backend_name) for grant in grants):
            raise DomainValidationError('run_trace_backend_not_available')
        repository = RunTraceBackendGrantRepository(self.session)
        await repository.lock_principal(principal_id)
        existing = await repository.list_for_principal(
            principal_id,
            for_update=True,
        )
        enabled_names = {grant.backend_name for grant in grants if grant.enabled}
        revoked_names = {
            grant.backend_name for grant in existing if grant.enabled and grant.backend_name not in enabled_names
        }
        if revoked_names:
            active_account = await self.session.scalar(
                select(TraceDatabaseAccountRow.backend_name)
                .where(
                    TraceDatabaseAccountRow.owner_principal_id == principal_id,
                    TraceDatabaseAccountRow.backend_name.in_(revoked_names),
                    or_(
                        TraceDatabaseAccountRow.status != 'disabled',
                        TraceDatabaseAccountRow.disabled_at.is_(None),
                        TraceDatabaseAccountRow.operation_id.is_not(None),
                    ),
                )
                .limit(1)
            )
            if active_account is not None:
                raise DomainConflictError('trace_database_account_must_be_disabled')
        after = await repository.replace_for_principal(
            principal_id,
            grants,
            actor=actor,
            now=now,
        )
        scope_id = principal_trace_scope_id(principal_id)
        for backend_name in sorted(grant.backend_name for grant in after if grant.enabled):
            await prepare_configured_run_trace_backend_scope(
                self.config,
                backend_name=backend_name,
                scope_id=scope_id,
            )
        return RunTraceBackendGrantReplacement(before=tuple(existing), after=tuple(after))

    async def resolve_for_submission(
        self,
        principal_id: str,
        requested_sinks: Sequence[str] | None,
    ) -> ResolvedRunTraceBackendSelection:
        grants = await RunTraceBackendGrantRepository(self.session).list_for_principal(principal_id)
        authorized = {
            grant.backend_name for grant in grants if grant.enabled and self._is_selectable(grant.backend_name)
        }
        if requested_sinks is not None:
            names = tuple(requested_sinks)
            if (
                not names
                or len(names) > _MAX_SELECTED_BACKENDS
                or len(set(names)) != len(names)
                or any(name not in authorized for name in names)
            ):
                raise DomainValidationError('run_trace_backend_not_available')
            return ResolvedRunTraceBackendSelection(sinks=names, selection_source='requested')

        defaults = sorted(
            (grant for grant in grants if grant.enabled and grant.default_order is not None),
            key=lambda grant: grant.default_order if grant.default_order is not None else 0,
        )
        if defaults:
            if any(not self._is_selectable(grant.backend_name) for grant in defaults):
                raise DomainValidationError('run_trace_backend_not_available')
            return ResolvedRunTraceBackendSelection(
                sinks=tuple(grant.backend_name for grant in defaults),
                selection_source='principal_default',
            )
        if self.config.require_principal_run_trace_backend_grant:
            raise DomainValidationError('run_trace_backend_selection_required')
        return ResolvedRunTraceBackendSelection(sinks=None, selection_source='legacy_default')

    def _is_selectable(self, backend_name: str) -> bool:
        backend = self._backends.get(backend_name)
        return backend is not None and not backend.system_managed and backend.user_selectable

    def _view(self, grant: RunTraceBackendGrantRecord) -> RunTraceBackendView:
        backend = self._backends[grant.backend_name]
        return RunTraceBackendView(
            name=grant.backend_name,
            kind=backend.kind,
            display_name=backend.display_name,
            description=backend.description,
            default_order=grant.default_order,
        )


__all__ = [
    'ResolvedRunTraceBackendSelection',
    'RunTraceBackendAccessView',
    'RunTraceBackendAccessPolicy',
    'RunTraceBackendGrantReplacement',
    'RunTraceBackendView',
]
