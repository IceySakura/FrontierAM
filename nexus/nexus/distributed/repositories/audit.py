from __future__ import annotations

import builtins

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from nexus.distributed.persistence.orm import AuditRow
from nexus.distributed.utils.identity import new_id


class AuditRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def append(
        self,
        *,
        actor_principal_id: str,
        operation: str,
        now: datetime,
        target_owner_principal_id: str | None = None,
        resource_type: str | None = None,
        resource_id: str | None = None,
        request_id: str | None = None,
        metadata: dict | None = None,
    ) -> AuditRow:
        row = AuditRow(
            audit_id=new_id('aud'),
            actor_principal_id=actor_principal_id,
            target_owner_principal_id=target_owner_principal_id,
            operation=operation,
            resource_type=resource_type,
            resource_id=resource_id,
            request_id=request_id,
            metadata_json=dict(metadata or {}),
            created_at=now,
        )
        self.session.add(row)
        await self.session.flush()
        return row

    async def list(
        self,
        *,
        actor: str | None = None,
        target: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> builtins.list[AuditRow]:
        query = select(AuditRow)
        if actor:
            query = query.where(AuditRow.actor_principal_id == actor)
        if target:
            query = query.where(AuditRow.target_owner_principal_id == target)
        return list(
            (await self.session.scalars(query.order_by(AuditRow.created_at.desc()).offset(offset).limit(limit))).all()
        )
