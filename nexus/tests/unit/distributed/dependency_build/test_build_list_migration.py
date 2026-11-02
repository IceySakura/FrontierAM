from importlib import import_module

from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, inspect, text


def test_build_list_index_migration_preserves_records_and_is_reentrant():
    migration = import_module('nexus.distributed.persistence.migrations.versions.0031_add_build_list_index')
    engine = create_engine('sqlite://')
    with engine.begin() as connection:
        connection.execute(
            text(
                'CREATE TABLE distributed_dependency_builds '
                '(dependency_build_id VARCHAR(64) PRIMARY KEY, created_at DATETIME NOT NULL)'
            )
        )
        connection.execute(text("INSERT INTO distributed_dependency_builds VALUES ('build', '2026-01-01')"))
        op = Operations(MigrationContext.configure(connection))
        migration.upgrade(op)
        migration.upgrade(op)
        indexes = inspect(connection).get_indexes('distributed_dependency_builds')
        assert any(i['column_names'] == ['created_at', 'dependency_build_id'] and not i['unique'] for i in indexes)
        migration.downgrade(op)
        migration.downgrade(op)
        assert inspect(connection).get_indexes('distributed_dependency_builds') == []
        assert connection.scalar(text('SELECT COUNT(*) FROM distributed_dependency_builds')) == 1
