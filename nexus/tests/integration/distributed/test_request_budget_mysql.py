"""MySQL concurrency and locking contracts for RequestBudgetRepository.

Uses a disposable ``nexus_policy*`` database configured via ``NEXUS_POLICY_TEST_DATABASE_URL_FILE``.
"""

from __future__ import annotations

import asyncio
import math
import os

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from sqlalchemy import delete, select
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from nexus.distributed.persistence.orm import (
    InstanceManagerLivenessRow,
    InstanceManagerRow,
    PolicyScopeRow,
    RequestBudgetLeaseRow,
    RequestBudgetRow,
    ShardLeaseRow,
    create_schema,
)
from nexus.distributed.policies.collections.infer_proxy_request_rate import (
    REQUEST_CAPACITY_SOURCE,
    REQUEST_RATE_POLICY,
)
from nexus.distributed.policies.models import (
    PolicyExecutorConfig,
    PolicyResult,
    RequestBudgetConfig,
    RequestPolicySelection,
    SignalObservation,
)
from nexus.distributed.policies.request_rate import request_budget_key
from nexus.distributed.repositories.manager import ManagerRepository
from nexus.distributed.repositories.policy import PolicyRepository
from nexus.distributed.repositories.request_budget import BudgetDemand, RequestBudgetRepository
from nexus.distributed.utils.clock import utcnow


@pytest.fixture
async def mysql_sessions():
    filename = os.getenv('NEXUS_POLICY_TEST_DATABASE_URL_FILE')
    if not filename:
        pytest.skip('NEXUS_POLICY_TEST_DATABASE_URL_FILE is required')
    url = make_url(Path(filename).read_text().strip())
    if not (url.database or '').startswith('nexus_policy'):
        raise ValueError('use a dedicated nexus_policy test database')
    engine = create_async_engine(url, pool_size=12, max_overflow=12)
    await create_schema(engine)
    async with engine.begin() as connection:
        await connection.execute(delete(RequestBudgetLeaseRow))
        await connection.execute(delete(RequestBudgetRow))
        await connection.execute(delete(InstanceManagerLivenessRow))
        await connection.execute(delete(InstanceManagerRow))
        await connection.execute(delete(PolicyScopeRow))
        await connection.execute(delete(ShardLeaseRow))
    try:
        yield async_sessionmaker(engine, expire_on_commit=False)
    finally:
        await engine.dispose()


def _config() -> PolicyExecutorConfig:
    return PolicyExecutorConfig.model_validate({
        'bindings': {
            'team-a-hy3-rate': {
                'kind': 'request_rate',
                'policy': REQUEST_RATE_POLICY,
                'source': REQUEST_CAPACITY_SOURCE,
                'revision': 1,
                'mode': 'enforce',
                'on_unavailable': 'defer',
                'environments': ['env-a'],
                'principals': ['alice'],
                'scopes': {
                    'capacity': {
                        'source_config': {
                            'model': 'hy3-offline',
                            'api_key': 'test-key',
                        }
                    }
                },
                'params': {'quota_qpm': 600.0},
            }
        }
    })


async def _seed_scope_and_managers(
    sessions, config: PolicyExecutorConfig, *, manager_count: int
) -> tuple[str, datetime]:
    selection = RequestPolicySelection(binding='team-a-hy3-rate', scope='capacity')
    key = request_budget_key(selection)
    now = utcnow()
    async with sessions() as session:
        await PolicyRepository(session, shard_count=8).ensure_scopes(config, now=now)
        row = await session.get(PolicyScopeRow, key)
        assert row is not None
        row.snapshot_json = {
            'snapshot_version': 1,
            'reason_code': 'adopted',
            'payload': {'allow': True, 'rate_per_minute': 600.0},
        }
        row.valid_until = now + timedelta(seconds=120)
        mgr_repo = ManagerRepository(session)
        for idx in range(manager_count):
            await mgr_repo.register_session(
                manager_id=f'mgr-{idx}',
                session_id=f'sess-{idx}',
                endpoint_url=f'http://127.0.0.1:{8100 + idx}',
                supported_modes=['job'],
                supported_spec_types=['nexus_single'],
                supported_instance_modes=['asyncio'],
                supported_spec_versions={'nexus_single': [1]},
                capacity_total=4,
                force_replace=True,
                environment_id='env-a',
                pool_id='default',
                workload='execution',
                owner_principal_id='alice',
                request_policy_binding='team-a-hy3-rate',
                request_policy_scope='capacity',
                now=now,
            )
        await session.commit()
    return key, now


async def test_mysql_concurrent_lease_claims_never_exceed_total_budget(mysql_sessions):
    config = _config()
    selection = RequestPolicySelection(binding='team-a-hy3-rate', scope='capacity')
    budget_cfg = RequestBudgetConfig(lease_seconds=10, renew_before_seconds=4)
    key, now = await _seed_scope_and_managers(mysql_sessions, config, manager_count=6)

    async def claim_one(idx: int, seq: int, at: datetime):
        async with mysql_sessions() as session:
            repo = RequestBudgetRepository(session)
            result = await repo.coordinate_lease(
                BudgetDemand(
                    budget_key=key,
                    manager_id=f'mgr-{idx}',
                    manager_session_id=f'sess-{idx}',
                    request_seq=seq,
                ),
                selection=selection,
                config=config,
                budget_config=budget_cfg,
                now=at,
            )
            await session.commit()
            return result

    # 1. All 6 managers race concurrently at t=0.
    results = await asyncio.gather(*(claim_one(idx, 1, now) for idx in range(6)))
    granted = [r.lease.granted_qpm for r in results if r.status == 'granted' and r.lease is not None]
    assert sum(granted) <= 600.0 + 1e-6

    async with mysql_sessions() as session:
        budget_row = await session.get(RequestBudgetRow, key)
        assert budget_row is not None
        assert budget_row.ledger_version == 7  # initial 1 + 6 serialized transactions
        lease_rows = (
            await session.scalars(select(RequestBudgetLeaseRow).where(RequestBudgetLeaseRow.budget_key == key))
        ).all()
        assert len(lease_rows) == 6
        unexpired_sum = sum(r.granted_qpm for r in lease_rows if r.valid_until is not None and r.valid_until > now)
        assert unexpired_sum <= 600.0 + 1e-6

    # 2. After the initial 10s leases expire (t=11s), all 6 active demanders race again and each gets 100 QPM.
    t11 = now + timedelta(seconds=11)
    round2 = await asyncio.gather(*(claim_one(idx, 2, t11) for idx in range(6)))
    assert all(r.status == 'granted' and r.lease is not None for r in round2)
    grants_r2 = [r.lease.granted_qpm for r in round2 if r.lease is not None]
    assert all(math.isclose(g, 100.0, rel_tol=1e-6) for g in grants_r2)
    assert math.isclose(sum(grants_r2), 600.0, rel_tol=1e-6)


async def test_mysql_coordinate_lease_shared_scope_lock_fences_concurrent_snapshot_publish(mysql_sessions):
    config = _config()
    selection = RequestPolicySelection(binding='team-a-hy3-rate', scope='capacity')
    key, now = await _seed_scope_and_managers(mysql_sessions, config, manager_count=1)

    async with mysql_sessions() as session:
        row = await session.get(PolicyScopeRow, key)
        assert row is not None
        row.next_evaluate_at = now - timedelta(seconds=1)
        row.last_demand_at = now
        await session.commit()

    async with mysql_sessions() as session:
        from nexus.distributed.repositories.policy import POLICY_SHARD_GROUP
        from nexus.distributed.repositories.shard_lease import ShardLeaseRepository

        await ShardLeaseRepository(session, shard_count=8).renew(
            owner_id='pub-1',
            shard_group=POLICY_SHARD_GROUP,
            shard_ids=list(range(8)),
            lease_ttl=timedelta(seconds=60),
            now=now,
        )
        claim = (
            await PolicyRepository(session, shard_count=8).due(
                list(range(8)),
                now=now + timedelta(seconds=1),
                idle_seconds=60,
                limit=10,
            )
        )[0]
        await session.commit()

    async def publish_pause():
        signal = SignalObservation(
            version='pause-v2',
            observed_at=now,
            valid_until=now + timedelta(seconds=60),
            values={'remaining': 0, 'factor': 1.0},
        )
        result = PolicyResult(
            decision={'allow': False, 'rate_per_minute': 0.0},
            valid_until=signal.valid_until,
            reason_code='quota_exhausted',
        )
        async with mysql_sessions() as session:
            applied = await PolicyRepository(session, shard_count=8).publish(
                claim,
                owner_id='pub-1',
                binding=config.bindings['team-a-hy3-rate'],
                result=result,
                observation=signal,
                error=None,
                evaluation_id='eval-pause',
                now=now,
                delay=1,
            )
            await session.commit()
            return applied

    # Hold coordinate_lease transaction open (holding FOR SHARE on PolicyScopeRow):
    # concurrent publish() must block until coordinate_lease commits.
    async with mysql_sessions() as coord_session:
        repo = RequestBudgetRepository(coord_session)
        lease_res = await repo.coordinate_lease(
            BudgetDemand(budget_key=key, manager_id='mgr-0', manager_session_id='sess-0', request_seq=1),
            selection=selection,
            config=config,
            now=now,
        )
        assert lease_res.status == 'granted'
        writer = asyncio.create_task(publish_pause())
        try:
            await asyncio.sleep(0.15)
            assert not writer.done()
            await coord_session.commit()
            assert await asyncio.wait_for(writer, timeout=5.0) is True
        finally:
            if not writer.done():
                writer.cancel()
                await asyncio.gather(writer, return_exceptions=True)
