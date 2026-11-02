"""MySQL lock and publication contracts. Use a disposable nexus_policy* database."""

from __future__ import annotations

import asyncio
import os

from datetime import timedelta
from pathlib import Path

import pytest

from sqlalchemy import delete, select
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from tests.unit.distributed.test_policy_framework import policy_config, seed_policy

from nexus.distributed.persistence.orm import PolicyScopeRow, ShardLeaseRow, create_schema
from nexus.distributed.policies.models import PolicyResult, SignalObservation
from nexus.distributed.repositories.policy import PolicyRepository
from nexus.distributed.utils.clock import utcnow


@pytest.fixture
async def mysql_sessions():
    filename = os.getenv('NEXUS_POLICY_TEST_DATABASE_URL_FILE')
    if not filename:
        pytest.skip('NEXUS_POLICY_TEST_DATABASE_URL_FILE is required')
    url = make_url(Path(filename).read_text().strip())
    if not (url.database or '').startswith('nexus_policy'):
        raise ValueError('use a dedicated nexus_policy test database')
    engine = create_async_engine(url, pool_size=4)
    await create_schema(engine)
    async with engine.begin() as connection:
        await connection.execute(delete(PolicyScopeRow))
        await connection.execute(delete(ShardLeaseRow))
    try:
        yield async_sessionmaker(engine, expire_on_commit=False)
    finally:
        await engine.dispose()


async def ready_claim(sessions, key):
    async with sessions() as session:
        row = await session.get(PolicyScopeRow, key)
        row.next_evaluate_at = utcnow()
        await session.commit()
    async with sessions() as session:
        return (
            await PolicyRepository(session, shard_count=8).due(
                list(range(8)),
                now=utcnow(),
                idle_seconds=60,
                limit=10,
            )
        )[0]


async def publish(sessions, config, claim):
    now = utcnow()
    signal = SignalObservation(version='positive', observed_at=now, valid_until=now + timedelta(seconds=30), values={})
    result = PolicyResult(decision={'allow': True}, valid_until=signal.valid_until, reason_code='capacity_available')
    async with sessions() as session:
        applied = await PolicyRepository(session, shard_count=8).publish(
            claim,
            owner_id='publisher',
            binding=config.bindings['tide'],
            result=result,
            observation=signal,
            error=None,
            evaluation_id='concurrent-positive',
            now=now,
            delay=1,
        )
        await session.commit()
        return applied


async def test_snapshot_read_fences_publication_until_assignment_transaction_commits(mysql_sessions):
    config = policy_config()
    key, _ = await seed_policy(mysql_sessions, config, allow=False)
    claim = await ready_claim(mysql_sessions, key)
    async with mysql_sessions() as reader:
        rows = await PolicyRepository(reader, shard_count=8).read_scopes([key])
        assert rows[key].snapshot_json['payload']['allow'] is False
        writer = asyncio.create_task(publish(mysql_sessions, config, claim))
        try:
            await asyncio.sleep(0.15)
            assert not writer.done()
            await reader.commit()
            assert await asyncio.wait_for(writer, timeout=5)
        finally:
            if not writer.done():
                writer.cancel()
                await asyncio.gather(writer, return_exceptions=True)
    async with mysql_sessions() as session:
        row = await session.get(PolicyScopeRow, key)
        assert row.snapshot_json['payload']['allow'] is True


async def test_only_one_concurrent_result_advances_state(mysql_sessions):
    config = policy_config()
    key, _ = await seed_policy(mysql_sessions, config, allow=False)
    claim = await ready_claim(mysql_sessions, key)
    results = await asyncio.gather(publish(mysql_sessions, config, claim), publish(mysql_sessions, config, claim))
    assert sorted(results) == [False, True]
    async with mysql_sessions() as session:
        row = await session.scalar(select(PolicyScopeRow).where(PolicyScopeRow.scope_key == key))
        assert row.version == claim['version'] + 1
