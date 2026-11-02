from __future__ import annotations

import hashlib

from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from nexus.distributed.persistence.orm import ShardLeaseRow


class ShardLeaseFenceStatus(str, Enum):
    VALID = 'valid'
    CONTENDED = 'contended'
    LOST = 'lost'


def _mysql_ensure_shard_leases(values: list[dict[str, Any]]) -> Any:
    if not values:
        raise ValueError('shard_lease_values_required')
    return mysql_insert(ShardLeaseRow).values(values).on_duplicate_key_update(shard_group=values[0]['shard_group'])


class ShardLeaseRepository:
    def __init__(self, session: AsyncSession, *, shard_count: int) -> None:
        self.session = session
        self.shard_count = shard_count

    async def validate_monotonic_shard_count(self, *, shard_group: str) -> None:
        persisted = await self.session.scalar(
            select(func.max(ShardLeaseRow.shard_count)).where(ShardLeaseRow.shard_group == shard_group)
        )
        if persisted is not None and self.shard_count < int(persisted):
            raise ValueError(
                f'shard_count cannot decrease for {shard_group}: configured={self.shard_count}, persisted={persisted}'
            )

    async def lock_discovery(self, *, shard_group: str) -> None:
        expired_at = datetime(1970, 1, 1)
        value = {
            'shard_group': _discovery_guard_group(shard_group),
            'shard_id': 0,
            'shard_count': self.shard_count,
            'owner_id': 'discovery-guard',
            'lease_expires_at': expired_at,
            'updated_at': expired_at,
        }
        dialect = self.session.get_bind().dialect.name
        if dialect == 'mysql':
            await self.session.execute(_mysql_ensure_shard_leases([value]))
            await self._ensure_topology(shard_group=shard_group)
            return
        if dialect == 'sqlite':
            await self.session.execute(
                sqlite_insert(ShardLeaseRow)
                .values(value)
                .on_conflict_do_update(
                    index_elements=['shard_group', 'shard_id'],
                    set_={'shard_count': self.shard_count},
                )
            )
            await self._ensure_topology(shard_group=shard_group)
            return
        row = await self.session.get(ShardLeaseRow, (value['shard_group'], 0), with_for_update=True)
        if row is None:
            self.session.add(ShardLeaseRow(**value))
            await self.session.flush()
        await self._ensure_topology(shard_group=shard_group)

    async def renew(
        self,
        *,
        owner_id: str,
        shard_group: str,
        shard_ids: list[int],
        lease_ttl: timedelta,
        now: datetime,
    ) -> list[int]:
        requested = list(dict.fromkeys(shard_ids))
        if not requested:
            return []
        await self._ensure_topology(
            shard_group=shard_group,
        )
        await self.session.execute(
            update(ShardLeaseRow)
            .where(
                ShardLeaseRow.shard_group == shard_group,
                ShardLeaseRow.shard_id.in_(requested),
                ShardLeaseRow.owner_id == owner_id,
            )
            .values(
                shard_count=self.shard_count,
                lease_expires_at=now + lease_ttl,
                updated_at=now,
            )
        )
        available = list(
            await self.session.scalars(
                select(ShardLeaseRow)
                .where(
                    ShardLeaseRow.shard_group == shard_group,
                    ShardLeaseRow.shard_id.in_(requested),
                    ShardLeaseRow.lease_expires_at <= now,
                )
                .order_by(ShardLeaseRow.shard_id)
                .with_for_update(skip_locked=True)
                .execution_options(populate_existing=True)
            )
        )
        for row in available:
            row.shard_count = self.shard_count
            row.owner_id = owner_id
            row.lease_expires_at = now + lease_ttl
            row.updated_at = now
        await self.session.flush()
        owned = set(
            await self.session.scalars(
                select(ShardLeaseRow.shard_id).where(
                    ShardLeaseRow.shard_group == shard_group,
                    ShardLeaseRow.shard_id.in_(requested),
                    ShardLeaseRow.owner_id == owner_id,
                    ShardLeaseRow.lease_expires_at > now,
                )
            )
        )
        return [shard_id for shard_id in requested if shard_id in owned]

    async def list_owned(
        self,
        *,
        owner_id: str,
        shard_group: str,
        now: datetime,
    ) -> list[int]:
        return list(
            await self.session.scalars(
                select(ShardLeaseRow.shard_id)
                .where(
                    ShardLeaseRow.shard_group == shard_group,
                    ShardLeaseRow.owner_id == owner_id,
                    ShardLeaseRow.lease_expires_at > now,
                )
                .order_by(ShardLeaseRow.shard_id)
            )
        )

    async def fence(
        self,
        *,
        owner_id: str,
        shard_group: str,
        shard_ids: list[int],
        now: datetime,
    ) -> ShardLeaseFenceStatus:
        requested = sorted(set(shard_ids))
        if not requested:
            return ShardLeaseFenceStatus.VALID
        rows = list(
            await self.session.scalars(
                select(ShardLeaseRow)
                .where(
                    ShardLeaseRow.shard_group == shard_group,
                    ShardLeaseRow.shard_id.in_(requested),
                )
                .order_by(ShardLeaseRow.shard_id)
                .with_for_update(skip_locked=True)
                .execution_options(populate_existing=True)
            )
        )
        requested_set = set(requested)
        locked_ids = {row.shard_id for row in rows}
        if locked_ids != requested_set:
            with self.session.no_autoflush:
                still_owned = await self.owns(
                    owner_id=owner_id,
                    shard_group=shard_group,
                    shard_ids=requested,
                    now=now,
                )
            return ShardLeaseFenceStatus.CONTENDED if still_owned else ShardLeaseFenceStatus.LOST
        valid = {row.shard_id for row in rows if row.owner_id == owner_id and row.lease_expires_at > now}
        if valid == requested_set:
            return ShardLeaseFenceStatus.VALID
        return ShardLeaseFenceStatus.LOST

    async def owns(
        self,
        *,
        owner_id: str,
        shard_group: str,
        shard_ids: list[int],
        now: datetime,
    ) -> bool:
        requested = sorted(set(shard_ids))
        if not requested:
            return True
        owned = set(
            await self.session.scalars(
                select(ShardLeaseRow.shard_id).where(
                    ShardLeaseRow.shard_group == shard_group,
                    ShardLeaseRow.shard_id.in_(requested),
                    ShardLeaseRow.owner_id == owner_id,
                    ShardLeaseRow.lease_expires_at > now,
                )
            )
        )
        return owned == set(requested)

    async def release(
        self,
        *,
        owner_id: str,
        shard_group: str,
        shard_ids: list[int],
        now: datetime,
    ) -> int:
        requested = sorted(set(shard_ids))
        if not requested:
            return 0
        result = await self.session.execute(
            update(ShardLeaseRow)
            .where(
                ShardLeaseRow.shard_group == shard_group,
                ShardLeaseRow.shard_id.in_(requested),
                ShardLeaseRow.owner_id == owner_id,
            )
            .values(lease_expires_at=now, updated_at=now)
        )
        return int(getattr(result, 'rowcount', 0) or 0)

    async def _ensure_topology(
        self,
        *,
        shard_group: str,
    ) -> None:
        persisted = int(
            await self.session.scalar(
                select(func.count()).select_from(ShardLeaseRow).where(ShardLeaseRow.shard_group == shard_group)
            )
            or 0
        )
        if persisted >= self.shard_count:
            return
        expired_at = datetime(1970, 1, 1)
        values = [
            {
                'shard_group': shard_group,
                'shard_id': shard_id,
                'shard_count': self.shard_count,
                'owner_id': 'topology-unowned',
                'lease_expires_at': expired_at,
                'updated_at': expired_at,
            }
            for shard_id in range(self.shard_count)
        ]
        dialect = self.session.get_bind().dialect.name
        if dialect == 'mysql':
            await self.session.execute(_mysql_ensure_shard_leases(values))
            return
        if dialect == 'sqlite':
            await self.session.execute(
                sqlite_insert(ShardLeaseRow)
                .values(values)
                .on_conflict_do_nothing(index_elements=['shard_group', 'shard_id'])
            )
            return
        for value in values:
            row = await self.session.get(ShardLeaseRow, (shard_group, value['shard_id']))
            if row is None:
                self.session.add(ShardLeaseRow(**value))
        await self.session.flush()


def _discovery_guard_group(shard_group: str) -> str:
    digest = hashlib.sha256(shard_group.encode()).hexdigest()[:32]
    return f'discovery-guard-{digest}'


__all__ = ['ShardLeaseFenceStatus', 'ShardLeaseRepository']
