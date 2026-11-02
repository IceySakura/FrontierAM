from __future__ import annotations

import hashlib

from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import select, true, update
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import load_only

from nexus.distributed.persistence.orm import PolicyScopeRow
from nexus.distributed.policies.models import PolicyBinding, PolicyExecutorConfig, PolicyResult, SignalObservation
from nexus.distributed.repositories.shard_lease import ShardLeaseFenceStatus, ShardLeaseRepository


POLICY_SHARD_GROUP = 'policy-executor'


def policy_scope_key(kind: str, binding: str, scope: str) -> str:
    return hashlib.sha256(f'{kind}\0{binding}\0{scope}'.encode()).hexdigest()


class PolicyRepository:
    def __init__(self, session: AsyncSession, *, shard_count: int):
        self.session, self.shard_count = session, shard_count

    async def ensure_scopes(self, config: PolicyExecutorConfig, *, now: datetime) -> None:
        for name, binding in config.bindings.items():
            digest = binding.fingerprint(config.imports)
            for scope in binding.scopes:
                key = policy_scope_key(binding.kind, name, scope)
                values = {
                    'scope_key': key,
                    'kind': binding.kind,
                    'binding_id': name,
                    'scope_id': scope,
                    'shard_id': int(key[:16], 16) % self.shard_count,
                    'config_revision': binding.revision,
                    'config_hash': digest,
                    'state_json': {},
                    'snapshot_json': None,
                    'valid_until': None,
                    'next_evaluate_at': now,
                    'last_demand_at': datetime(1970, 1, 1),
                    'version': 1,
                    'failures': 0,
                    'last_error': None,
                    'updated_at': now,
                }
                dialect = self.session.get_bind().dialect.name
                if dialect == 'mysql':
                    statement = mysql_insert(PolicyScopeRow).values(**values)
                    await self.session.execute(statement.on_duplicate_key_update(scope_key=key))
                elif dialect == 'sqlite':
                    await self.session.execute(sqlite_insert(PolicyScopeRow).values(**values).on_conflict_do_nothing())
                else:
                    raise ValueError('unsupported policy storage dialect')
                # A rolling old replica must never overwrite a newer binding revision.
                await self.session.execute(
                    update(PolicyScopeRow)
                    .where(PolicyScopeRow.scope_key == key, PolicyScopeRow.config_revision < binding.revision)
                    .values(
                        config_revision=binding.revision,
                        config_hash=digest,
                        state_json={},
                        snapshot_json=None,
                        valid_until=None,
                        next_evaluate_at=now,
                        version=PolicyScopeRow.version + 1,
                        failures=0,
                        last_error=None,
                        updated_at=now,
                    )
                )

    async def demand(self, keys: list[str], *, now: datetime) -> None:
        if keys:
            # Demand does not change the evaluation version or defeat failure backoff.
            await self.session.execute(
                update(PolicyScopeRow).where(PolicyScopeRow.scope_key.in_(keys)).values(last_demand_at=now)
            )

    async def read_scopes(self, keys: list[str]) -> dict[str, PolicyScopeRow]:
        if not keys:
            return {}
        rows = await self.session.scalars(
            select(PolicyScopeRow)
            .options(
                load_only(
                    PolicyScopeRow.scope_key,
                    PolicyScopeRow.config_hash,
                    PolicyScopeRow.snapshot_json,
                    PolicyScopeRow.valid_until,
                    PolicyScopeRow.last_error,
                    PolicyScopeRow.next_evaluate_at,
                )
            )
            .where(PolicyScopeRow.scope_key.in_(keys))
            .order_by(PolicyScopeRow.scope_key)
            .with_for_update(read=True)
            .execution_options(populate_existing=True)
        )
        return {row.scope_key: row for row in rows}

    async def due(
        self,
        shards: list[int],
        *,
        now: datetime,
        idle_seconds: float,
        limit: int,
        configured_keys: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        if not shards:
            return []
        rows = await self.session.scalars(
            select(PolicyScopeRow)
            .where(
                PolicyScopeRow.shard_id.in_(shards),
                PolicyScopeRow.scope_key.in_(configured_keys) if configured_keys is not None else true(),
                PolicyScopeRow.next_evaluate_at <= now,
                PolicyScopeRow.last_demand_at > now - timedelta(seconds=idle_seconds),
            )
            .order_by(PolicyScopeRow.next_evaluate_at, PolicyScopeRow.scope_key)
            .limit(limit)
        )
        return [
            {
                'key': row.scope_key,
                'kind': row.kind,
                'binding_id': row.binding_id,
                'scope_id': row.scope_id,
                'shard_id': row.shard_id,
                'version': row.version,
                'config_hash': row.config_hash,
                'state': dict(row.state_json),
                'failures': row.failures,
            }
            for row in rows
        ]

    async def publish(
        self,
        claim: dict[str, Any],
        *,
        owner_id: str,
        binding: PolicyBinding,
        result: PolicyResult | None,
        observation: SignalObservation | None,
        error: str | None,
        evaluation_id: str,
        now: datetime,
        delay: float,
    ) -> bool:
        row = await self.session.scalar(
            select(PolicyScopeRow).where(PolicyScopeRow.scope_key == claim['key']).with_for_update()
        )
        if row is None or row.version != claim['version'] or row.config_hash != claim['config_hash']:
            return False
        lease = ShardLeaseRepository(self.session, shard_count=self.shard_count)
        if (
            await lease.fence(
                owner_id=owner_id,
                shard_group=POLICY_SHARD_GROUP,
                shard_ids=[claim['shard_id']],
                now=now,
            )
            != ShardLeaseFenceStatus.VALID
        ):
            return False
        if result is not None:
            if observation is None or not now < result.valid_until <= observation.valid_until:
                raise ValueError('policy_result_expired')
            row.state_json = result.next_state
            row.valid_until = result.valid_until
            row.snapshot_json = {
                'kind': binding.kind,
                'schema_version': 1,
                'binding_id': row.binding_id,
                'scope_id': row.scope_id,
                'config_revision': binding.revision,
                'config_hash': row.config_hash,
                'snapshot_version': row.version + 1,
                'evaluation_id': evaluation_id,
                'input_version': observation.version,
                'observed_at': observation.observed_at.isoformat(),
                'generated_at': now.isoformat(),
                'valid_until': result.valid_until.isoformat(),
                'reason_code': result.reason_code,
                'payload': result.decision,
            }
            row.failures = 0
        else:
            row.failures += 1
        row.last_error = error
        row.next_evaluate_at = now + timedelta(seconds=delay)
        row.updated_at = now
        row.version += 1
        await self.session.flush()
        return True
