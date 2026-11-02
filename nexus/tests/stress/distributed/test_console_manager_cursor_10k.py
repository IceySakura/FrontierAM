from __future__ import annotations

import json
import time

from datetime import timedelta

from sqlalchemy import insert
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from nexus.distributed.persistence.orm import InstanceManagerLivenessRow, InstanceManagerRow, create_schema
from nexus.distributed.repositories.environment import EnvironmentInventoryRepository
from nexus.distributed.utils.clock import utcnow


MANAGER_COUNT = 10_000
PAGE_SIZE = 100


async def test_console_manager_cursor_pages_10k_history_without_materializing_inventory(tmp_path):
    started = time.monotonic()
    engine = create_async_engine(f'sqlite+aiosqlite:///{tmp_path / "console-managers-10k.db"}')
    await create_schema(engine)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    now = utcnow()
    stale_at = now - timedelta(days=1)
    manager_rows = []
    liveness_rows = []
    for index in range(MANAGER_COUNT):
        manager_id = f'manager-{index:05d}'
        session_id = f'session-{index:05d}'
        heartbeat_at = now if index % 2 == 0 else stale_at
        manager_rows.append(
            {
                'manager_id': manager_id,
                'session_id': session_id,
                'endpoint_url': f'http://127.0.0.1:{10_000 + index}',
                'labels_json': {},
                'draining': False,
                'draining_at': None,
                'owner_principal_id': 'owner-console-stress',
                'environment_id': 'env-console-stress',
                'pool_id': 'pool-console-stress',
                'allocation_id': f'allocation-{index // 25:04d}',
                'workload': 'execution',
                'allocation_spec_hash': None,
                'capability_snapshot_json': {},
                'bootstrap_version': 1,
                'artifact_ref_version': 1,
                'supported_spec_versions_json': {'nexus_single': [1]},
                'supported_modes_json': ['job'],
                'supported_spec_types_json': ['nexus_single'],
                'supported_instance_modes_json': ['asyncio'],
                'capacity_total': 20,
                'observed_capacity_used': 0,
                'local_instances_json': {},
                'version': 'stress',
                'metadata_json': {},
                'last_heartbeat_at': heartbeat_at,
                'created_at': stale_at,
                'updated_at': heartbeat_at,
            }
        )
        liveness_rows.append(
            {
                'manager_id': manager_id,
                'session_id': session_id,
                'capacity_total': 20,
                'observed_capacity_used': 0,
                'last_heartbeat_at': heartbeat_at,
                'fenced_at': None,
                'updated_at': heartbeat_at,
            }
        )

    try:
        async with session_factory() as session:
            await session.execute(insert(InstanceManagerRow), manager_rows)
            await session.execute(insert(InstanceManagerLivenessRow), liveness_rows)
            await session.commit()

        cursor = None
        healthy_ids = []
        page_count = 0
        while True:
            async with session_factory() as session:
                page = await EnvironmentInventoryRepository(
                    session,
                    manager_healthy_ttl=timedelta(seconds=30),
                ).list_managers_page(
                    environment_id='env-console-stress',
                    pool_id='pool-console-stress',
                    health='healthy',
                    draining=False,
                    after_manager_id=cursor,
                    limit=PAGE_SIZE + 1,
                    now=now,
                )
            has_more = len(page) > PAGE_SIZE
            page = page[:PAGE_SIZE]
            assert 0 < len(page) <= PAGE_SIZE
            assert [manager.manager_id for manager in page] == sorted(manager.manager_id for manager in page)
            assert all(manager.healthy for manager in page)
            healthy_ids.extend(manager.manager_id for manager in page)
            page_count += 1
            if not has_more:
                break
            cursor = page[-1].manager_id

        assert page_count == MANAGER_COUNT // 2 // PAGE_SIZE
        assert len(healthy_ids) == MANAGER_COUNT // 2
        assert len(set(healthy_ids)) == len(healthy_ids)
        assert healthy_ids[0] == 'manager-00000'
        assert healthy_ids[-1] == 'manager-09998'
        print(
            json.dumps(
                {
                    'console_manager_cursor_10k': {
                        'managers': MANAGER_COUNT,
                        'healthy_managers': len(healthy_ids),
                        'page_size': PAGE_SIZE,
                        'pages': page_count,
                        'seconds': time.monotonic() - started,
                    }
                },
                sort_keys=True,
            )
        )
    finally:
        await engine.dispose()
