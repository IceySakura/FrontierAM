from __future__ import annotations

import importlib

from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, inspect, select

from nexus.distributed.persistence.orm import Base, ExecutionRow
from tests.unit.distributed.analytics.test_statistics import NOW


def test_statistics_migration_is_additive_restartable_and_preserves_legacy_payload(tmp_path):
    migration = importlib.import_module('nexus.distributed.persistence.migrations.versions.0033_add_business_statistics')
    engine = create_engine(f'sqlite:///{tmp_path}/migration.db')
    with engine.begin() as connection:
        Base.metadata.create_all(connection)
        operations = Operations(MigrationContext.configure(connection))
        migration.downgrade(operations)
        connection.exec_driver_sql('CREATE TABLE unrelated_extension (id INTEGER PRIMARY KEY)')
        connection.exec_driver_sql("INSERT INTO executions (execution_id, created_by, version, shard_id, mode, spec_type, instance_mode, spec_json, driver_config_json, placement_json, status, route_status, slot_cost, next_attempt_seq, attempt_count, max_attempts, next_event_seq, priority, next_schedule_at, created_at, updated_at) VALUES ('old', 'test', 1, 0, 'job', 'nexus_single', 'asyncio', '{\"run_input\":{\"task_name\":\"keep_task\"}}', '{}', '{}', 'succeeded', 'none', 1, 1, 0, 1, 1, 0, ?, ?, ?)", (NOW.isoformat(' '),) * 3)
        migration.upgrade(operations)
        migration.upgrade(operations)
        assert 'unrelated_extension' in inspect(connection).get_table_names()
        assert connection.scalar(select(ExecutionRow.task_name)) is None
        assert connection.scalar(select(ExecutionRow.spec_json)) == {'run_input': {'task_name': 'keep_task'}}
        assert {'task_name', 'first_started_at', 'finished_at'} <= {item['name'] for item in inspect(connection).get_columns('distributed_batch_items')}
        assert {'idx_exec_stats_created', 'idx_exec_stats_started', 'idx_exec_stats_finished'} <= {item['name'] for item in inspect(connection).get_indexes('executions')}
    engine.dispose()
