from __future__ import annotations

import copy
import re

from collections.abc import Mapping
from datetime import datetime
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from nexus.distributed.persistence.orm import ExecutionCredentialRow, ExecutionRow


_CREDENTIAL_KIND_PATTERN = re.compile(r'^[a-z][a-z0-9_]{0,63}$')


class ExecutionCredentialRepository:
    """Store user credentials under a top-level execution without projecting them into specs."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create_many(
        self,
        execution_id: str,
        payloads: Mapping[str, Mapping[str, Any]],
        *,
        expires_at: datetime | None,
        now: datetime,
    ) -> tuple[str, ...]:
        execution = (
            await self.session.execute(
                select(ExecutionRow.execution_id, ExecutionRow.parent_execution_id).where(
                    ExecutionRow.execution_id == execution_id
                )
            )
        ).first()
        if execution is None:
            raise KeyError(execution_id)
        if execution.parent_execution_id is not None:
            raise ValueError('execution credentials must be attached to a top-level execution')
        kinds: list[str] = []
        for kind, payload in payloads.items():
            _validate_kind(kind)
            if not isinstance(payload, Mapping):
                raise TypeError(f'execution credential {kind} must be an object')
            self.session.add(
                ExecutionCredentialRow(
                    execution_id=execution_id,
                    credential_kind=kind,
                    payload_json=copy.deepcopy(dict(payload)),
                    version=1,
                    expires_at=expires_at,
                    created_at=now,
                    updated_at=now,
                )
            )
            kinds.append(kind)
        await self.session.flush()
        return tuple(sorted(kinds))

    async def get_payload(
        self,
        execution_id: str,
        kind: str,
        *,
        now: datetime,
    ) -> dict[str, Any] | None:
        _validate_kind(kind)
        top_level_execution_id = await self._top_level_execution_id(execution_id)
        row = await self.session.get(ExecutionCredentialRow, (top_level_execution_id, kind))
        if row is None or (row.expires_at is not None and row.expires_at <= now):
            return None
        return copy.deepcopy(dict(row.payload_json))

    async def list_kinds(self, execution_id: str, *, now: datetime) -> tuple[str, ...]:
        top_level_execution_id = await self._top_level_execution_id(execution_id)
        query = select(ExecutionCredentialRow.credential_kind).where(
            ExecutionCredentialRow.execution_id == top_level_execution_id,
            (ExecutionCredentialRow.expires_at.is_(None) | (ExecutionCredentialRow.expires_at > now)),
        )
        return tuple(sorted((await self.session.scalars(query)).all()))

    async def delete_expired(self, *, now: datetime, limit: int = 500) -> int:
        if limit < 1:
            raise ValueError('execution credential cleanup limit must be positive')
        keys = (
            await self.session.execute(
                select(ExecutionCredentialRow.execution_id, ExecutionCredentialRow.credential_kind)
                .where(
                    ExecutionCredentialRow.expires_at.is_not(None),
                    ExecutionCredentialRow.expires_at <= now,
                )
                .order_by(ExecutionCredentialRow.expires_at, ExecutionCredentialRow.execution_id)
                .limit(limit)
            )
        ).all()
        if not keys:
            return 0
        for execution_id, credential_kind in keys:
            await self.session.execute(
                delete(ExecutionCredentialRow).where(
                    ExecutionCredentialRow.execution_id == execution_id,
                    ExecutionCredentialRow.credential_kind == credential_kind,
                )
            )
        return len(keys)

    async def _top_level_execution_id(self, execution_id: str) -> str:
        execution = (
            await self.session.execute(
                select(ExecutionRow.execution_id, ExecutionRow.parent_execution_id).where(
                    ExecutionRow.execution_id == execution_id
                )
            )
        ).first()
        if execution is None:
            raise KeyError(execution_id)
        return execution.parent_execution_id or execution.execution_id


def _validate_kind(kind: str) -> None:
    if not isinstance(kind, str) or _CREDENTIAL_KIND_PATTERN.fullmatch(kind) is None:
        raise ValueError('invalid execution credential kind')


__all__ = ['ExecutionCredentialRepository']
