from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

from sqlalchemy.dialects import mysql

from nexus.distributed.persistence.orm import EnvironmentResourceUsageRow, QuotaUsageRow
from nexus.distributed.repositories import access as access_repository
from nexus.distributed.repositories import dependency_build as dependency_build_repository
from nexus.distributed.repositories import quota as quota_repository
from nexus.distributed.repositories import shard_lease as shard_lease_repository
from nexus.distributed.utils.clock import utcnow


def test_mysql_owner_quota_usage_initializer_uses_no_warning_upsert() -> None:
    owner_statement = quota_repository._mysql_ensure_owner_usage(
        {
            'owner_principal_id': 'user-1',
            'environment': '*',
            'pool': '*',
            'quota_class': 'owner',
        }
    )

    sql = str(owner_statement.compile(dialect=mysql.dialect()))
    assert 'INSERT IGNORE' not in sql
    assert 'ON DUPLICATE KEY UPDATE' in sql


async def test_allocation_reservation_does_not_write_lock_environment_or_grant_config() -> None:
    now = utcnow()
    allocation = SimpleNamespace(
        allocation_id='allocation-1',
        environment_id='env-1',
        owner_principal_id='owner-1',
        status='requested',
        manager_count=1,
        cpu_millicores=1000,
        memory_bytes=1024,
        slots=1,
        active_batches=0,
        resources_reserved_at=None,
        resources_released_at=None,
        updated_at=now,
    )
    environment = SimpleNamespace(status='enabled', resource_limits_json={})
    grant = SimpleNamespace(enabled=True, limits_json={})
    environment_usage = EnvironmentResourceUsageRow(
        scope_key='environment:env-1',
        scope_type='environment',
        environment_id='env-1',
        principal_id=None,
        manager_count=0,
        cpu_millicores=0,
        memory_bytes=0,
        slots=0,
        active_batches=0,
        version=1,
        updated_at=now,
    )
    grant_usage = EnvironmentResourceUsageRow(
        scope_key='grant:env-1:owner-1',
        scope_type='grant',
        environment_id='env-1',
        principal_id='owner-1',
        manager_count=0,
        cpu_millicores=0,
        memory_bytes=0,
        slots=0,
        active_batches=0,
        version=1,
        updated_at=now,
    )
    session = SimpleNamespace(
        scalar=AsyncMock(side_effect=[allocation, environment, grant]),
        flush=AsyncMock(),
    )
    repository = quota_repository.QuotaRepository(cast(Any, session), default_limits={})
    repository._lock_compute_usage_pair = AsyncMock(return_value=(environment_usage, grant_usage))

    await repository.reserve_allocation(allocation.allocation_id, now=now)

    statements = [call.args[0] for call in session.scalar.await_args_list]
    compiled = [str(statement.compile(dialect=mysql.dialect())) for statement in statements]
    assert 'FOR UPDATE' in compiled[0]
    assert all('FOR UPDATE' not in statement for statement in compiled[1:])


def test_remaining_mysql_idempotent_initializers_use_no_warning_upserts() -> None:
    principal_statement = access_repository._mysql_ensure_principal(
        {
            'principal_id': 'user-1',
            'issuer': 'tai',
            'tenant': 'tenant-1',
            'subject': 'subject-1',
        }
    )
    lease_statement = shard_lease_repository._mysql_ensure_shard_leases(
        [
            {
                'shard_group': 'default',
                'shard_id': 0,
                'shard_count': 16,
                'owner_id': 'scheduler-1',
            }
        ]
    )

    for statement in (principal_statement, lease_statement):
        sql = str(statement.compile(dialect=mysql.dialect()))
        assert 'INSERT IGNORE' not in sql
        assert 'ON DUPLICATE KEY UPDATE' in sql

    principal_sql = str(principal_statement.compile(dialect=mysql.dialect()))
    assert 'principal_id = distributed_principals.principal_id' in principal_sql


def test_mysql_dependency_build_uses_current_read_upsert() -> None:
    statement = dependency_build_repository._mysql_upsert_dependency_build(
        {
            'dependency_build_id': 'build-1',
            'owner_principal_id': 'user-1',
            'environment_id': 'env-1',
            'build_key': 'a' * 64,
            'status': 'requested',
            'build_spec_json': {},
            'attempt': 0,
        }
    )

    sql = str(statement.compile(dialect=mysql.dialect()))
    assert 'INSERT IGNORE' not in sql
    assert 'ON DUPLICATE KEY UPDATE' in sql
    assert 'build_key = %s' in sql


async def test_mysql_bulk_quota_read_does_not_relock_rows_after_upsert() -> None:
    now = utcnow()
    rows = [
        QuotaUsageRow(
            owner_principal_id=owner,
            environment='*',
            pool='*',
            quota_class='owner',
            queued_executions=0,
            running_executions=0,
            reserved_slots=0,
            version=1,
            updated_at=now,
        )
        for owner in ('owner-a', 'owner-b')
    ]
    session = SimpleNamespace(
        get_bind=lambda: SimpleNamespace(dialect=SimpleNamespace(name='mysql')),
        execute=AsyncMock(),
        flush=AsyncMock(),
        scalars=AsyncMock(return_value=rows),
    )

    result = await quota_repository.QuotaRepository(cast(Any, session), default_limits={})._lock_owner_usage_many(
        ['owner-a', 'owner-b'],
        now=now,
    )

    assert set(result) == {'owner-a', 'owner-b'}
    read_statement = session.scalars.await_args.args[0]
    assert 'FOR UPDATE' not in str(read_statement.compile(dialect=mysql.dialect()))
