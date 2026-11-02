from __future__ import annotations

import builtins
import copy
import re

from collections.abc import Mapping
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from nexus.distributed.environment.models import (
    ProviderConfigRecord,
    ProviderConfigStatus,
)
from nexus.distributed.environment.provider_config import parse_provider_config
from nexus.distributed.errors import DomainConflictError, DomainValidationError, ResourceNotFoundError
from nexus.distributed.persistence.orm import ProviderConfigRow


_PROVIDER_CONFIG_REF_PATTERN = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._/-]{0,511}$')


class ProviderConfigRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(
        self,
        *,
        provider_config_ref: str,
        config: Mapping[str, Any],
        status: str,
        actor: str,
        now: datetime,
    ) -> ProviderConfigRecord:
        _validate_ref(provider_config_ref)
        _validate_status(status)
        parsed = parse_provider_config(config)
        if await self.session.get(ProviderConfigRow, provider_config_ref) is not None:
            raise DomainConflictError('provider_config_already_exists')
        row = ProviderConfigRow(
            provider_config_ref=provider_config_ref,
            provider_type=parsed.provider_type,
            config_json=parsed.model_dump(mode='json'),
            status=status,
            version=1,
            created_by=actor,
            updated_by=actor,
            created_at=now,
            updated_at=now,
        )
        self.session.add(row)
        await self.session.flush()
        return _record(row)

    async def get(self, provider_config_ref: str) -> ProviderConfigRecord:
        row = await self.session.get(ProviderConfigRow, provider_config_ref)
        if row is None:
            raise ResourceNotFoundError('provider_config_not_found')
        return _record(row)

    async def list(self, *, include_disabled: bool = True) -> builtins.list[ProviderConfigRecord]:
        query = select(ProviderConfigRow)
        if not include_disabled:
            query = query.where(ProviderConfigRow.status == ProviderConfigStatus.ENABLED)
        rows = (await self.session.scalars(query.order_by(ProviderConfigRow.provider_config_ref))).all()
        return [_record(row) for row in rows]

    async def update(
        self,
        provider_config_ref: str,
        *,
        config: Mapping[str, Any] | None = None,
        status: str | None = None,
        actor: str,
        now: datetime,
    ) -> ProviderConfigRecord:
        row = await self.session.scalar(
            select(ProviderConfigRow)
            .where(ProviderConfigRow.provider_config_ref == provider_config_ref)
            .with_for_update()
        )
        if row is None:
            raise ResourceNotFoundError('provider_config_not_found')
        next_config = parse_provider_config(config) if config is not None else parse_provider_config(row.config_json)
        if next_config.provider_type != row.provider_type:
            raise DomainConflictError('provider_config_type_immutable')
        next_status = status if status is not None else row.status
        _validate_status(next_status)
        config_json = next_config.model_dump(mode='json')
        if config_json != row.config_json or next_status != row.status:
            row.config_json = config_json
            row.status = next_status
            row.version += 1
            row.updated_by = actor
            row.updated_at = now
            await self.session.flush()
        return _record(row)

    async def require_enabled(
        self,
        provider_config_ref: str,
        *,
        provider_type: str,
    ) -> ProviderConfigRecord:
        record = await self.get(provider_config_ref)
        if record.status != ProviderConfigStatus.ENABLED:
            raise DomainConflictError('provider_config_disabled', detail=provider_config_ref)
        if record.provider_type != provider_type:
            raise DomainConflictError(
                'provider_config_type_mismatch',
                detail=f'{provider_config_ref}:{provider_type}:{record.provider_type}',
            )
        return record

    async def validate_environment(
        self,
        *,
        allocation_type: str,
        provider_type: str,
        provider_config_ref: str | None,
        workload_namespace: str,
    ) -> ProviderConfigRecord | None:
        if allocation_type == 'predefined':
            if provider_type != 'predefined':
                raise DomainValidationError('invalid_environment_provider')
            if provider_config_ref is not None:
                raise DomainValidationError('predefined_provider_config_ref_forbidden')
            return None
        if allocation_type != 'dynamic' or provider_type not in {'kubernetes', 'tkex'}:
            raise DomainValidationError('invalid_environment_provider')
        if not provider_config_ref:
            raise DomainValidationError('provider_config_ref_required')
        try:
            record = await self.require_enabled(provider_config_ref, provider_type=provider_type)
        except ResourceNotFoundError as exc:
            raise DomainConflictError(
                'environment_provider_config_not_found',
                detail=provider_config_ref,
            ) from exc
        except DomainConflictError as exc:
            if exc.code == 'provider_config_type_mismatch':
                raise DomainConflictError('environment_provider_config_type_mismatch', detail=exc.detail) from exc
            raise
        if record.config.get('namespace') != workload_namespace:
            raise DomainConflictError(
                'environment_provider_config_namespace_mismatch',
                detail=f'{provider_config_ref}:{record.config.get("namespace")}:{workload_namespace}',
            )
        return record


def _validate_ref(value: str) -> None:
    if _PROVIDER_CONFIG_REF_PATTERN.fullmatch(value) is None or '..' in value.split('/'):
        raise DomainValidationError('invalid_provider_config_ref')


def _validate_status(value: str) -> None:
    if value not in {ProviderConfigStatus.ENABLED, ProviderConfigStatus.DISABLED}:
        raise DomainValidationError('invalid_provider_config_status')


def _record(row: ProviderConfigRow) -> ProviderConfigRecord:
    return ProviderConfigRecord(
        provider_config_ref=row.provider_config_ref,
        provider_type=row.provider_type,
        config=copy.deepcopy(row.config_json),
        status=row.status,
        version=row.version,
        created_by=row.created_by,
        updated_by=row.updated_by,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


__all__ = ['ProviderConfigRepository']
