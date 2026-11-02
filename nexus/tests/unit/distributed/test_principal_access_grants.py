from __future__ import annotations

from datetime import datetime

import pytest

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from nexus.distributed.errors import DomainValidationError, ResourceNotFoundError
from nexus.distributed.persistence.orm import create_schema
from nexus.distributed.repositories.access import AccessRepository
from nexus.distributed.repositories.execution import CreateExecution, ExecutionRepository


@pytest.fixture
async def session_factory(tmp_path):
    engine = create_async_engine(f'sqlite+aiosqlite:///{tmp_path / "principal-access.db"}')
    await create_schema(engine)
    try:
        yield async_sessionmaker(engine, expire_on_commit=False)
    finally:
        await engine.dispose()


async def _create_user(repository: AccessRepository, principal_id: str, now: datetime) -> None:
    await repository.upsert_user(
        principal_id=principal_id,
        issuer='tai',
        tenant='default',
        subject=principal_id,
        roles={'user'},
        display_name=principal_id.upper(),
        actor='admin',
        now=now,
    )


async def test_multiple_direct_grants_expand_accessible_owners_without_transitivity(session_factory) -> None:
    now = datetime(2026, 8, 13, 12, 0, 0)
    async with session_factory() as session:
        repository = AccessRepository(session)
        for principal_id in ('a', 'b', 'c', 'd'):
            await _create_user(repository, principal_id, now)

        await repository.put_principal_access_grant(
            owner_principal_id='a',
            grantee_principal_id='c',
            actor='a',
            now=now,
        )
        await repository.put_principal_access_grant(
            owner_principal_id='b',
            grantee_principal_id='c',
            actor='admin',
            now=now,
        )
        await repository.put_principal_access_grant(
            owner_principal_id='c',
            grantee_principal_id='d',
            actor='c',
            now=now,
        )
        await session.commit()

        assert await repository.list_accessible_owner_principal_ids('c') == frozenset({'a', 'b', 'c'})
        assert await repository.list_accessible_owner_principal_ids('d') == frozenset({'c', 'd'})
        assert [
            (grant.owner_principal_id, grant.grantee_principal_id)
            for grant in await repository.list_principal_access_grants(grantee_principal_id='c')
        ] == [('a', 'c'), ('b', 'c')]


async def test_principal_grant_rejects_self_and_unknown_users_and_delete_is_idempotent(session_factory) -> None:
    now = datetime(2026, 8, 13, 12, 0, 0)
    async with session_factory() as session:
        repository = AccessRepository(session)
        await _create_user(repository, 'a', now)
        await _create_user(repository, 'c', now)

        with pytest.raises(DomainValidationError, match='principal_access_grant_self'):
            await repository.put_principal_access_grant(
                owner_principal_id='a',
                grantee_principal_id='a',
                actor='a',
                now=now,
            )
        with pytest.raises(ResourceNotFoundError, match='user_not_found'):
            await repository.put_principal_access_grant(
                owner_principal_id='missing',
                grantee_principal_id='c',
                actor='admin',
                now=now,
            )

        await repository.put_principal_access_grant(
            owner_principal_id='a',
            grantee_principal_id='c',
            actor='a',
            now=now,
        )
        assert await repository.delete_principal_access_grant(
            owner_principal_id='a',
            grantee_principal_id='c',
        )
        assert not await repository.delete_principal_access_grant(
            owner_principal_id='a',
            grantee_principal_id='c',
        )


async def test_execution_list_filters_owner_environment_pairs_without_cross_owner_leakage(session_factory) -> None:
    now = datetime(2026, 8, 13, 12, 0, 0)
    async with session_factory() as session:
        repository = ExecutionRepository(session, shard_count=8)
        for owner in ('a', 'b'):
            await repository.create_execution(
                CreateExecution(
                    execution_id=f'execution-{owner}',
                    mode='job',
                    spec_type='nexus_single',
                    instance_mode='asyncio',
                    spec={'task_name': 'identity'},
                    owner_principal_id=owner,
                    environment_id='shared-environment',
                ),
                now=now,
            )
        await session.commit()

        rows = await repository.list_executions(
            owner_principal_ids={'a', 'b'},
            owner_environment_pairs={('a', 'shared-environment')},
        )

        assert [row.execution_id for row in rows] == ['execution-a']
