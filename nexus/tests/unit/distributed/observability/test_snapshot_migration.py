from __future__ import annotations

import importlib

from datetime import datetime

from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, inspect, select

from nexus.distributed.persistence.db import _pending_migrations
from nexus.distributed.persistence.orm import Base, ExecutionRow


def test_snapshot_migration_preserves_legacy_rows_and_is_restartable(tmp_path):
    migration = importlib.import_module(
        'nexus.distributed.persistence.migrations.versions.0032_add_observability_snapshots'
    )
    engine = create_engine(f'sqlite:///{tmp_path}/migration.db')
    with engine.begin() as connection:
        Base.metadata.create_all(connection)
        now = datetime(2026, 9, 14)
        connection.execute(
            ExecutionRow.__table__.insert().values(
                execution_id='legacy',
                idempotency_scope='test',
                idempotency_key='one',
                shard_id=0,
                mode='job',
                spec_type='nexus_single',
                instance_mode='asyncio',
                spec_json={'retained': 'x' * 1000},
                status='queued',
                route_status='none',
                next_schedule_at=now,
                created_at=now,
                updated_at=now,
            )
        )
        operation = Operations(MigrationContext.configure(connection))
        migration.downgrade(operation)
        assert 'queue_entered_at' not in {column['name'] for column in inspect(connection).get_columns('executions')}
        migration.upgrade(operation)
        migration.upgrade(operation)
        assert connection.scalar(select(ExecutionRow.spec_json)) == {'retained': 'x' * 1000}
        assert connection.scalar(select(ExecutionRow.queue_entered_at)) is None
        assert 'distributed_observability_snapshots' in inspect(connection).get_table_names()
        assert 'provisioning_deadline_at' in {
            column['name'] for column in inspect(connection).get_columns('distributed_environment_allocations')
        }
    engine.dispose()


def test_deployed_extension_revision_uses_only_additive_main_migrations():
    pending = _pending_migrations('0031_execution_analytics')
    assert [migration.revision for migration in pending] == [
        '0031_build_list_index',
        '0032_observability_snapshots',
        '0033_business_statistics',
        '0034_batch_finish_backfill',
        '0035_policy_scopes',
        '0036_request_budget_leases',
    ]
