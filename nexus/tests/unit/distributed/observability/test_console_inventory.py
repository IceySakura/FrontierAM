from datetime import timedelta

import pytest

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from tests.unit.distributed.observability.test_status_queries import NOW, assignment, manager

from nexus.distributed.errors import DomainConflictError
from nexus.distributed.persistence.orm import Base, InstanceManagerLivenessRow
from nexus.distributed.repositories.environment import EnvironmentInventoryRepository


@pytest.fixture
async def inventory(tmp_path):
    engine = create_async_engine(f'sqlite+aiosqlite:///{tmp_path}/inventory.db')
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    async with async_sessionmaker(engine, expire_on_commit=False)() as session:
        yield session, EnvironmentInventoryRepository(session, manager_healthy_ttl=timedelta(seconds=30))
    await engine.dispose()


async def test_summary_is_independent_of_page_and_draining_filters_run_before_limit(inventory):
    session, repo = inventory
    session.add_all([manager(f'm-{i:03}', draining=i < 100) for i in range(120)])
    await session.commit()
    first = await repo.list_managers_page(health='healthy', limit=100, now=NOW)
    second = await repo.list_managers_page(health='healthy', after_manager_id=first[-1].manager_id, limit=100, now=NOW)
    accepting = await repo.list_managers_page(health='healthy', draining=False, limit=100, now=NOW)
    summary = await repo.manager_summary(now=NOW)
    assert len(first) == 100
    assert len(second) == 20
    assert len(accepting) == 20
    assert summary['total'] == summary['healthy'] == 120
    assert summary['accepting'] == 20
    assert summary['draining'] == 100
    assert summary['reserved_slots'] == summary['lost'] == 0
    assert summary['capacity_total'] == 1200
    assert summary['available_slots'] == 200
    empty = await repo.manager_summary(environment_id='missing', now=NOW)
    assert empty['total'] == empty['healthy'] == empty['available_slots'] == 0


async def test_inventory_health_and_capacity_ignore_previous_manager_session(inventory):
    session, repo = inventory
    for name, fenced in [('fenced-old', True), ('reserved-old', False)]:
        session.add(manager(name, session='new'))
        session.add(
            InstanceManagerLivenessRow(
                manager_id=name,
                session_id='old',
                capacity_total=99,
                last_heartbeat_at=NOW,
                fenced_at=NOW if fenced else None,
                updated_at=NOW,
            )
        )
        session.add(assignment(name, name, session='old', cost=9))
    session.add(assignment('current', 'reserved-old', session='new', cost=2))
    await session.commit()
    healthy = await repo.list_managers_page(health='healthy', limit=100, now=NOW)
    lost = await repo.list_managers_page(health='lost', limit=100, now=NOW)
    summary = await repo.manager_summary(now=NOW)
    assert len(healthy) == summary['healthy'] == 2
    assert lost == []
    assert sum(row.reserved_slots for row in healthy) == summary['reserved_slots'] == 2
    assert sum(row.available_slots for row in healthy) == summary['available_slots'] == 18
    assert summary['capacity_total'] == 20


async def test_manager_health_boundary_and_scoped_pool_search(inventory):
    session, repo = inventory
    session.add(manager('boundary', age=30))
    session.add(manager('stale', age=31))
    other = manager('other', environment_id='two')
    other.pool_id = 'other-pool'
    session.add(other)
    await session.commit()
    healthy = await repo.list_managers_page(environment_id='one', health='healthy', limit=100, now=NOW)
    lost = await repo.list_managers_page(environment_id='one', health='lost', limit=100, now=NOW)
    assert [row.manager_id for row in healthy] == ['boundary']
    assert [row.manager_id for row in lost] == ['stale']
    assert await repo.manager_pools(environment_id='one', prefix='bat') == ['batch']
    assert await repo.manager_pools(environment_id='one', prefix='%') == []
    assert await repo.manager_pools(after='batch') == ['other-pool']
    with pytest.raises(DomainConflictError):
        await repo.delete_stale_manager('stale', now=NOW, expected_session_id='old')
