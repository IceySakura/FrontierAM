from __future__ import annotations

import argparse
import asyncio
import importlib
import inspect as pyinspect
import json
import logging

from datetime import timedelta

import pytest

from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine as create_sync_engine
from sqlalchemy import inspect, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import create_async_engine

from nexus.cli.commands import distributed as distributed_cli
from nexus.distributed.config import DistributedConfig
from nexus.distributed.persistence import db as distributed_db
from nexus.distributed.persistence.db import create_engine, init_database, upgrade_database
from nexus.distributed.persistence.migrations import (
    CLOUD_SUBMIT_REVISION,
    CONFIG_OVERRIDES_REVISION,
    HEAD_REVISION,
    LEGACY_CLOUD_SUBMIT_REVISION,
    LEGACY_ENVIRONMENT_CLEANUP_REVISION,
    LEGACY_QUOTA_CLEANUP_REVISION,
    MIGRATIONS,
    PRINCIPAL_ACCESS_GRANTS_REVISION,
    REQUEST_BUDGET_LEASES_REVISION,
    RUN_TRACE_BACKEND_GRANTS_REVISION,
)
from nexus.distributed.persistence.orm import create_schema


def test_distributed_migration_revision_ids_fit_alembic_version_column():
    for migration in MIGRATIONS:
        assert len(migration.revision) <= 32
        if migration.down_revision is not None:
            assert len(migration.down_revision) <= 32


def test_run_trace_backend_grant_migration_is_registered_before_head():
    migration = importlib.import_module(
        'nexus.distributed.persistence.migrations.versions.0029_add_run_trace_backend_grants'
    )

    assert migration.revision == RUN_TRACE_BACKEND_GRANTS_REVISION
    registered = next(item for item in MIGRATIONS if item.revision == migration.revision)
    assert registered.down_revision == migration.down_revision


def test_principal_access_grant_migration_is_idempotent_and_registered(tmp_path):
    migration = importlib.import_module(
        'nexus.distributed.persistence.migrations.versions.0030_add_principal_access_grants'
    )
    assert migration.revision == PRINCIPAL_ACCESS_GRANTS_REVISION
    registered = next(item for item in MIGRATIONS if item.revision == migration.revision)
    assert registered.down_revision == RUN_TRACE_BACKEND_GRANTS_REVISION

    engine = create_sync_engine(f'sqlite:///{tmp_path / "principal-access-grants.db"}')
    try:
        with engine.begin() as connection:
            connection.exec_driver_sql(
                'CREATE TABLE distributed_principals (principal_id VARCHAR(64) NOT NULL PRIMARY KEY)'
            )
            operations = Operations(MigrationContext.configure(connection))
            migration.upgrade(operations)
            connection.exec_driver_sql('DROP INDEX idx_principal_access_grants_grantee_owner')
            migration.upgrade(operations)

            table = inspect(connection).get_pk_constraint('distributed_principal_access_grants')
            indexes = inspect(connection).get_indexes('distributed_principal_access_grants')
            foreign_keys = inspect(connection).get_foreign_keys('distributed_principal_access_grants')

            assert table['constrained_columns'] == ['owner_principal_id', 'grantee_principal_id']
            assert {index['name'] for index in indexes} == {'idx_principal_access_grants_grantee_owner'}
            assert {tuple(item['constrained_columns']) for item in foreign_keys} == {
                ('owner_principal_id',),
                ('grantee_principal_id',),
            }

            migration.downgrade(operations)
            migration.downgrade(operations)
            assert 'distributed_principal_access_grants' not in inspect(connection).get_table_names()
    finally:
        engine.dispose()


def test_run_trace_backend_grant_migration_backfills_active_or_uncertain_trace_database_accounts(tmp_path):
    migration = importlib.import_module(
        'nexus.distributed.persistence.migrations.versions.0029_add_run_trace_backend_grants'
    )
    engine = create_sync_engine(f'sqlite:///{tmp_path / "run-trace-backend-grants.db"}')
    try:
        with engine.begin() as connection:
            connection.exec_driver_sql(
                'CREATE TABLE distributed_principals (principal_id VARCHAR(64) NOT NULL PRIMARY KEY)'
            )
            connection.exec_driver_sql(
                'CREATE TABLE distributed_trace_database_accounts ('
                'owner_principal_id VARCHAR(64) NOT NULL, '
                'backend_name VARCHAR(64) NOT NULL, '
                'status VARCHAR(32) NOT NULL, '
                'operation_id VARCHAR(64), '
                'created_at DATETIME NOT NULL, '
                'updated_at DATETIME NOT NULL, '
                'disabled_at DATETIME, '
                'PRIMARY KEY (owner_principal_id, backend_name))'
            )
            connection.exec_driver_sql(
                "INSERT INTO distributed_principals VALUES ('active'), ('disabled'), ('error'), ('uncertain')"
            )
            connection.exec_driver_sql(
                'INSERT INTO distributed_trace_database_accounts VALUES '
                "('active', 'tdsql', 'ready', NULL, '2026-08-01 00:00:00', '2026-08-02 00:00:00', NULL), "
                "('disabled', 'tdsql', 'disabled', NULL, '2026-08-01 00:00:00', '2026-08-02 00:00:00', "
                "'2026-08-02 00:00:00'), "
                "('error', 'tdsql', 'error', NULL, '2026-08-01 00:00:00', '2026-08-02 00:00:00', "
                "'2026-08-02 00:00:00'), "
                "('uncertain', 'tdsql', 'restoring', 'tdop-1', '2026-08-01 00:00:00', '2026-08-02 00:00:00', "
                "'2026-08-02 00:00:00')"
            )
            operations = Operations(MigrationContext.configure(connection))

            migration.upgrade(operations)
            migration.upgrade(operations)

            rows = connection.execute(
                text(
                    'SELECT principal_id, backend_name, enabled, default_order, created_by '
                    'FROM distributed_run_trace_backend_grants ORDER BY principal_id'
                )
            ).all()
    finally:
        engine.dispose()

    assert rows == [
        ('active', 'tdsql', 1, None, 'system:migration:0029'),
        ('error', 'tdsql', 1, None, 'system:migration:0029'),
        ('uncertain', 'tdsql', 1, None, 'system:migration:0029'),
    ]


def test_batch_run_shard_migration_backfills_from_execution_table(tmp_path):
    migration = importlib.import_module('nexus.distributed.persistence.migrations.versions.0016_add_batch_run_shard')
    engine = create_sync_engine(f'sqlite:///{tmp_path / "batch-run-shard.db"}')
    try:
        with engine.begin() as conn:
            conn.exec_driver_sql(
                'CREATE TABLE executions (execution_id VARCHAR(64) PRIMARY KEY, shard_id INTEGER NOT NULL)'
            )
            conn.exec_driver_sql(
                'CREATE TABLE distributed_batch_runs ('
                'parent_execution_id VARCHAR(64) PRIMARY KEY, '
                'control_state VARCHAR(32) NOT NULL, '
                'last_reconciled_at DATETIME, '
                'lifecycle_lease_expires_at DATETIME)'
            )
            conn.exec_driver_sql(
                'CREATE INDEX idx_batch_runs_lifecycle_claim ON distributed_batch_runs '
                '(control_state, last_reconciled_at, lifecycle_lease_expires_at, parent_execution_id)'
            )
            conn.exec_driver_sql("INSERT INTO executions VALUES ('batch-parent', 37)")
            conn.exec_driver_sql(
                'INSERT INTO distributed_batch_runs (parent_execution_id, control_state) '
                "VALUES ('batch-parent', 'running')"
            )

            migration.upgrade(Operations(MigrationContext.configure(conn)))

            assert (
                conn.scalar(
                    text("SELECT shard_id FROM distributed_batch_runs WHERE parent_execution_id = 'batch-parent'")
                )
                == 37
            )
            columns = {column['name']: column for column in inspect(conn).get_columns('distributed_batch_runs')}
            indexes = {index['name'] for index in inspect(conn).get_indexes('distributed_batch_runs')}
    finally:
        engine.dispose()

    assert columns['shard_id']['nullable'] is False
    assert 'idx_batch_runs_shard_lifecycle_claim' in indexes
    assert 'idx_batch_runs_lifecycle_claim' not in indexes


def test_execution_parent_cursor_migration_backfills_batch_children(tmp_path):
    migration = importlib.import_module(
        'nexus.distributed.persistence.migrations.versions.0019_add_execution_parent_cursor'
    )
    engine = create_sync_engine(f'sqlite:///{tmp_path / "execution-parent-cursor.db"}')
    try:
        with engine.begin() as conn:
            conn.exec_driver_sql(
                'CREATE TABLE executions ('
                'execution_id VARCHAR(64) PRIMARY KEY, '
                'owner_principal_id VARCHAR(64), '
                'created_by VARCHAR(128) NOT NULL, '
                'spec_type VARCHAR(64) NOT NULL, '
                'created_at DATETIME NOT NULL, '
                'updated_at DATETIME NOT NULL)'
            )
            conn.exec_driver_sql(
                'CREATE TABLE distributed_batch_item_attempts ('
                'parent_execution_id VARCHAR(64) NOT NULL, '
                'item_id VARCHAR(512) NOT NULL, '
                'child_execution_id VARCHAR(64) NOT NULL, '
                'error_code VARCHAR(128))'
            )
            conn.exec_driver_sql(
                'INSERT INTO executions VALUES '
                "('batch-parent', 'alice', 'alice', 'nexus_batch', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP), "
                "('batch-child', 'alice', 'system:batch-parent', 'nexus_single', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP), "
                "('ordinary-run', 'alice', 'alice', 'nexus_single', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
            )
            conn.exec_driver_sql(
                "INSERT INTO distributed_batch_item_attempts VALUES ('batch-parent', 'item-1', 'batch-child', NULL)"
            )

            migration.upgrade(Operations(MigrationContext.configure(conn)))

            rows = dict(
                conn.execute(
                    text('SELECT execution_id, parent_execution_id FROM executions ORDER BY execution_id')
                ).all()
            )
            indexes = {index['name'] for index in inspect(conn).get_indexes('executions')}
    finally:
        engine.dispose()

    assert rows == {
        'batch-child': 'batch-parent',
        'batch-parent': None,
        'ordinary-run': None,
    }
    assert {
        'idx_executions_parent_updated',
        'idx_executions_owner_parent_updated',
        'idx_executions_spec_parent_updated',
    } <= indexes


def test_resource_usage_migration_backfills_environment_and_grant_without_overwriting(tmp_path):
    migration = importlib.import_module(
        'nexus.distributed.persistence.migrations.versions.0018_precreate_resource_usage'
    )
    engine = create_sync_engine(f'sqlite:///{tmp_path / "resource-usage.db"}')
    try:
        with engine.begin() as conn:
            conn.exec_driver_sql('CREATE TABLE distributed_environments (environment_id VARCHAR(64) PRIMARY KEY)')
            conn.exec_driver_sql(
                'CREATE TABLE distributed_environment_grants ('
                'principal_id VARCHAR(64) NOT NULL, environment_id VARCHAR(64) NOT NULL, '
                'PRIMARY KEY (principal_id, environment_id))'
            )
            conn.exec_driver_sql(
                'CREATE TABLE distributed_environment_resource_usage ('
                'scope_key VARCHAR(160) PRIMARY KEY, scope_type VARCHAR(16) NOT NULL, '
                'environment_id VARCHAR(64) NOT NULL, principal_id VARCHAR(64), '
                'manager_count INTEGER NOT NULL, cpu_millicores BIGINT NOT NULL, '
                'memory_bytes BIGINT NOT NULL, slots INTEGER NOT NULL, active_batches INTEGER NOT NULL, '
                'version BIGINT NOT NULL, updated_at DATETIME NOT NULL)'
            )
            conn.exec_driver_sql("INSERT INTO distributed_environments VALUES ('env-existing'), ('env-new')")
            conn.exec_driver_sql(
                "INSERT INTO distributed_environment_grants VALUES ('alice', 'env-existing'), ('bob', 'env-new')"
            )
            conn.exec_driver_sql(
                'INSERT INTO distributed_environment_resource_usage VALUES '
                "('environment:env-existing', 'environment', 'env-existing', NULL, 7, 7000, 70000, 14, 1, 3, "
                'CURRENT_TIMESTAMP)'
            )

            operations = Operations(MigrationContext.configure(conn))
            migration.upgrade(operations)
            migration.upgrade(operations)
            rows = {
                row.scope_key: row
                for row in conn.execute(
                    text(
                        'SELECT scope_key, manager_count, cpu_millicores, memory_bytes, slots, active_batches '
                        'FROM distributed_environment_resource_usage'
                    )
                )
            }
    finally:
        engine.dispose()

    assert set(rows) == {
        'environment:env-existing',
        'environment:env-new',
        'grant:env-existing:alice',
        'grant:env-new:bob',
    }
    assert tuple(rows['environment:env-existing'][1:]) == (7, 7000, 70000, 14, 1)
    for scope_key in {'environment:env-new', 'grant:env-existing:alice', 'grant:env-new:bob'}:
        assert tuple(rows[scope_key][1:]) == (0, 0, 0, 0, 0)


def test_environment_grant_scope_migration_backfills_existing_grants(tmp_path):
    migration = importlib.import_module('nexus.distributed.persistence.migrations.versions.0022_add_grant_scopes')
    assert 'nexus.distributed.access.permissions' not in pyinspect.getsource(migration)
    engine = create_sync_engine(f'sqlite:///{tmp_path / "grant-scopes.db"}')
    try:
        with engine.begin() as connection:
            connection.execute(
                text(
                    'CREATE TABLE distributed_environment_grants ('
                    'principal_id VARCHAR(64) NOT NULL, '
                    'environment_id VARCHAR(64) NOT NULL, '
                    'PRIMARY KEY (principal_id, environment_id))'
                )
            )
            connection.execute(
                text(
                    'INSERT INTO distributed_environment_grants (principal_id, environment_id) '
                    "VALUES ('principal-1', 'environment-1')"
                )
            )
            connection.execute(
                text(
                    'CREATE TABLE distributed_principals ('
                    'principal_id VARCHAR(64) NOT NULL PRIMARY KEY, roles_json JSON NOT NULL)'
                )
            )
            connection.execute(
                text(
                    'INSERT INTO distributed_principals (principal_id, roles_json) '
                    'VALUES (\'principal-1\', \'["user", "submit_all"]\')'
                )
            )
            operations = Operations(MigrationContext.configure(connection))
            migration.upgrade(operations)
            migration.upgrade(operations)

            columns = {
                column['name']: column for column in inspect(connection).get_columns('distributed_environment_grants')
            }
            assert columns['scopes_json']['nullable'] is False
            raw_scopes = connection.execute(
                text(
                    'SELECT scopes_json FROM distributed_environment_grants '
                    "WHERE principal_id = 'principal-1' AND environment_id = 'environment-1'"
                )
            ).scalar_one()
            scopes = json.loads(raw_scopes) if isinstance(raw_scopes, str) else raw_scopes
            assert scopes == [
                'artifact:write',
                'execution:command',
                'execution:read',
                'execution:stop',
                'submission:batch',
                'submission:live',
                'submission:single',
            ]
            raw_roles = connection.execute(
                text("SELECT roles_json FROM distributed_principals WHERE principal_id = 'principal-1'")
            ).scalar_one()
            roles = json.loads(raw_roles) if isinstance(raw_roles, str) else raw_roles
            assert roles == ['user']
    finally:
        engine.dispose()


def test_default_system_token_quota_migration_preserves_api_tokens(tmp_path):
    migration = importlib.import_module(
        'nexus.distributed.persistence.migrations.versions.0025_add_default_system_token_quota'
    )
    engine = create_sync_engine(f'sqlite:///{tmp_path / "default-system-token.db"}')
    try:
        with engine.begin() as connection:
            connection.execute(
                text(
                    'CREATE TABLE distributed_environments ('
                    'environment_id VARCHAR(64) PRIMARY KEY, '
                    'name VARCHAR(128) NOT NULL)'
                )
            )
            connection.execute(
                text(
                    'CREATE TABLE distributed_submit_tokens ('
                    'token_id VARCHAR(64) PRIMARY KEY, '
                    'token_prefix VARCHAR(16) NOT NULL, '
                    'token_hash VARCHAR(128) NOT NULL, '
                    'token_hash_key_id VARCHAR(32) NOT NULL, '
                    'owner_principal_id VARCHAR(64) NOT NULL, '
                    'environment_id VARCHAR(64) NOT NULL, '
                    'scopes_json JSON NOT NULL, '
                    'created_by VARCHAR(128) NOT NULL, '
                    'created_at DATETIME NOT NULL, '
                    'expires_at DATETIME, revoked_at DATETIME, last_used_at DATETIME)'
                )
            )
            connection.execute(
                text(
                    'INSERT INTO distributed_submit_tokens ('
                    'token_id, token_prefix, token_hash, token_hash_key_id, owner_principal_id, '
                    'environment_id, scopes_json, created_by, created_at) VALUES ('
                    "'tok-existing', 'nxs_tok_exist', 'hash', 'v1', 'principal-1', "
                    "'environment-1', '[]', 'principal-1', CURRENT_TIMESTAMP)"
                )
            )

            operations = Operations(MigrationContext.configure(connection))
            migration.upgrade(operations)
            migration.upgrade(operations)

            connection.execute(
                text(
                    'INSERT INTO distributed_submit_tokens ('
                    'token_id, token_prefix, token_hash, token_hash_key_id, owner_principal_id, '
                    'environment_id, scopes_json, created_by, created_at) VALUES ('
                    "'tok-rolling-old-writer', 'nxs_tok_roll', 'hash', 'v1', 'principal-1', "
                    "'environment-1', '[]', 'principal-1', CURRENT_TIMESTAMP)"
                )
            )

            environment_columns = {
                column['name']: column for column in inspect(connection).get_columns('distributed_environments')
            }
            token_columns = {
                column['name']: column for column in inspect(connection).get_columns('distributed_submit_tokens')
            }
            token_kind = connection.scalar(
                text("SELECT token_kind FROM distributed_submit_tokens WHERE token_id = 'tok-existing'")
            )
            rolling_token_kind = connection.scalar(
                text("SELECT token_kind FROM distributed_submit_tokens WHERE token_id = 'tok-rolling-old-writer'")
            )
    finally:
        engine.dispose()

    assert environment_columns['default_quota_json']['nullable'] is True
    assert token_columns['token_kind']['nullable'] is False
    assert all(
        token_columns[name]['nullable'] is True
        for name in ('token_prefix', 'token_hash', 'token_hash_key_id', 'environment_id')
    )
    assert token_kind == 'api'
    assert rolling_token_kind == 'api'


def test_submit_token_app_binding_migration_adds_unique_nullable_app_id(tmp_path):
    migration = importlib.import_module(
        'nexus.distributed.persistence.migrations.versions.0026_add_submit_token_app_binding'
    )
    engine = create_sync_engine(f'sqlite:///{tmp_path / "submit-token-app-binding.db"}')
    try:
        with engine.begin() as connection:
            connection.execute(
                text(
                    'CREATE TABLE distributed_submit_tokens ('
                    'token_id VARCHAR(64) PRIMARY KEY, '
                    'owner_principal_id VARCHAR(64) NOT NULL)'
                )
            )
            operations = Operations(MigrationContext.configure(connection))
            migration.upgrade(operations)
            migration.upgrade(operations)

            columns = {
                column['name']: column for column in inspect(connection).get_columns('distributed_submit_tokens')
            }
            indexes = {index['name']: index for index in inspect(connection).get_indexes('distributed_submit_tokens')}
    finally:
        engine.dispose()

    assert columns['bound_app_id']['nullable'] is True
    assert bool(indexes['uq_submit_tokens_bound_app_id']['unique']) is True


def test_trace_database_account_migration_is_idempotent_and_stores_no_password(tmp_path):
    migration = importlib.import_module(
        'nexus.distributed.persistence.migrations.versions.0027_add_trace_database_accounts'
    )
    engine = create_sync_engine(f'sqlite:///{tmp_path / "trace-database-accounts.db"}')
    try:
        with engine.begin() as connection:
            connection.exec_driver_sql(
                'CREATE TABLE distributed_principals (principal_id VARCHAR(64) NOT NULL PRIMARY KEY)'
            )
            operations = Operations(MigrationContext.configure(connection))

            migration.upgrade(operations)
            migration.upgrade(operations)

            columns = {
                column['name'] for column in inspect(connection).get_columns('distributed_trace_database_accounts')
            }
            indexes = {
                index['name'] for index in inspect(connection).get_indexes('distributed_trace_database_accounts')
            }
            assert columns == {
                'owner_principal_id',
                'backend_name',
                'db_username',
                'status',
                'credential_version',
                'operation_id',
                'operation_type',
                'operation_expires_at',
                'last_error_code',
                'created_at',
                'updated_at',
                'disabled_at',
            }
            assert 'idx_trace_database_accounts_status_updated' in indexes
            assert all('password' not in column for column in columns)

            migration.downgrade(operations)
            assert 'distributed_trace_database_accounts' not in inspect(connection).get_table_names()
    finally:
        engine.dispose()


def test_batch_module_dependency_migration_backfills_legacy_runs_and_is_idempotent(tmp_path):
    migration = importlib.import_module(
        'nexus.distributed.persistence.migrations.versions.0028_add_batch_module_dependencies'
    )
    engine = create_sync_engine(f'sqlite:///{tmp_path / "batch-module-dependencies.db"}')
    try:
        with engine.begin() as connection:
            connection.exec_driver_sql(
                'CREATE TABLE distributed_batch_runs (parent_execution_id VARCHAR(64) NOT NULL PRIMARY KEY)'
            )
            connection.exec_driver_sql(
                "INSERT INTO distributed_batch_runs (parent_execution_id) VALUES ('legacy-batch')"
            )
            operations = Operations(MigrationContext.configure(connection))

            migration.upgrade(operations)
            migration.upgrade(operations)

            columns = {column['name']: column for column in inspect(connection).get_columns('distributed_batch_runs')}
            assert set(columns) == {
                'parent_execution_id',
                'dependency_status',
                'module_dependencies_json',
                'child_dependency_build_id',
                'child_template_json',
            }
            assert columns['dependency_status']['nullable'] is False
            assert (
                connection.scalar(
                    text(
                        'SELECT dependency_status FROM distributed_batch_runs '
                        "WHERE parent_execution_id = 'legacy-batch'"
                    )
                )
                == 'not_required'
            )

            migration.downgrade(operations)
            migration.downgrade(operations)
            assert {column['name'] for column in inspect(connection).get_columns('distributed_batch_runs')} == {
                'parent_execution_id'
            }
    finally:
        engine.dispose()


def test_distributed_db_upgrade_recognizes_preview_cloud_submit_revision():
    pending = distributed_db._pending_migrations(LEGACY_CLOUD_SUBMIT_REVISION)

    assert [migration.revision for migration in pending] == [migration.revision for migration in MIGRATIONS[2:]]


def test_distributed_config_env_and_cli_overrides(monkeypatch):
    monkeypatch.setenv('NEXUS_DIST_DATABASE_URL', 'mysql+aiomysql://env/db')
    monkeypatch.setenv('NEXUS_DIST_INTERNAL_TOKEN', 'env-token')
    monkeypatch.setenv('NEXUS_DIST_SHARD_COUNT', '17')
    monkeypatch.setenv('NEXUS_DIST_DB_POOL_SIZE', '11')
    monkeypatch.setenv('NEXUS_DIST_DB_MAX_OVERFLOW', '7')
    monkeypatch.setenv('NEXUS_DIST_DB_POOL_TIMEOUT', '3.5')
    monkeypatch.setenv('NEXUS_DIST_DB_POOL_RECYCLE', '1800')
    monkeypatch.setenv('NEXUS_DIST_DB_WAIT_TIMEOUT', '4.5')
    monkeypatch.setenv('NEXUS_DIST_SLOW_SQL_LOG_THRESHOLD_MS', '12.5')
    monkeypatch.setenv('NEXUS_DIST_START_TIMEOUT_SECONDS', '900')
    monkeypatch.setenv('NEXUS_DIST_MANAGER_HEARTBEAT_INTERVAL', '10')
    monkeypatch.setenv('NEXUS_DIST_MANAGER_HEALTHY_TTL', '60')
    monkeypatch.setenv('NEXUS_DIST_MANAGER_DEAD_TTL_SECONDS', '180')
    monkeypatch.setenv('NEXUS_DIST_MANAGER_LOCAL_RECONCILE_INTERVAL', '20')
    monkeypatch.setenv('NEXUS_DIST_MANAGER_START_BATCH_SIZE', '8')
    monkeypatch.setenv('NEXUS_DIST_AIOMONITOR_ENABLED', 'false')
    monkeypatch.setenv('NEXUS_DIST_SCHEDULER_SHARD_OWNERSHIP_LIMIT', '13')
    monkeypatch.setenv('NEXUS_DIST_SCHEDULER_SHARD_DISCOVER_JITTER', '0.3')

    config = DistributedConfig.from_env()
    assert config.database_url == 'mysql+aiomysql://env/db'
    assert config.read_internal_token() == ''
    assert config.shard_count == 17
    assert config.db_pool_size == 11
    assert config.db_max_overflow == 7
    assert config.db_pool_timeout == 3.5
    assert config.db_pool_recycle == 1800
    assert config.db_wait_timeout == 4.5
    assert config.slow_sql_log_threshold_ms == 12.5
    assert config.start_timeout == timedelta(seconds=900)
    assert config.manager_heartbeat_interval == 10
    assert config.manager_healthy_ttl == timedelta(seconds=60)
    assert config.manager_dead_ttl == timedelta(seconds=180)
    assert config.manager_local_reconcile_interval == 20
    assert config.manager_start_batch_size == 8
    assert config.aiomonitor_enabled is False
    assert config.scheduler_shard_ownership_limit == 13
    assert config.scheduler_shard_discover_jitter == 0.3

    args = argparse.Namespace(
        database_url='mysql+aiomysql://cli/db',
        internal_token='cli-token',
        shard_count=23,
    )
    config = distributed_cli._config_from_args(args)
    assert config.database_url == 'mysql+aiomysql://cli/db'
    assert config.read_internal_token() == 'cli-token'
    assert config.shard_count == 23


def test_config_cli_overrides_exclude_environment_values(monkeypatch):
    monkeypatch.setenv('NEXUS_DIST_SHARD_COUNT', '17')
    args = argparse.Namespace(database_url=None, internal_token=None, shard_count=None)

    assert distributed_cli._config_cli_overrides(args) == {}


async def test_service_config_separates_environment_and_cli_values(monkeypatch):
    captured = {}

    async def fake_initialize_distributed_config(**kwargs):
        captured.update(kwargs)
        return DistributedConfig(database_url='sqlite+aiosqlite:///bootstrap.db')

    monkeypatch.setenv('NEXUS_DIST_SHARD_COUNT', '17')
    monkeypatch.setattr(
        'nexus.distributed.config.store.initialize_distributed_config',
        fake_initialize_distributed_config,
    )
    args = argparse.Namespace(database_url=None, internal_token=None, shard_count=23)

    await distributed_cli._initialize_service_config(
        args,
        role='scheduler',
        identity='scheduler-1',
    )

    assert captured['environment_values']['shard_count'] == '17'
    assert captured['overrides'] == {'shard_count': 23}
    assert captured['require_stored_config'] is False


async def test_service_config_honors_required_store_bootstrap_flag(monkeypatch):
    captured = {}

    async def fake_initialize_distributed_config(**kwargs):
        captured.update(kwargs)
        return DistributedConfig(database_url='sqlite+aiosqlite:///bootstrap.db')

    monkeypatch.setenv('NEXUS_DIST_CONFIG_STORE_REQUIRED', '1')
    monkeypatch.setattr(
        'nexus.distributed.config.store.initialize_distributed_config',
        fake_initialize_distributed_config,
    )

    await distributed_cli._initialize_service_config(
        argparse.Namespace(database_url=None, internal_token=None, shard_count=None),
        role='control-gateway',
        identity=None,
    )

    assert captured['require_stored_config'] is True


async def test_instance_manager_allows_newer_database_schema(monkeypatch):
    captured = {}

    async def fake_initialize_distributed_config(**kwargs):
        captured.update(kwargs)
        return DistributedConfig(database_url='sqlite+aiosqlite:///bootstrap.db')

    monkeypatch.setattr(
        'nexus.distributed.config.store.initialize_distributed_config',
        fake_initialize_distributed_config,
    )

    await distributed_cli._initialize_service_config(
        argparse.Namespace(database_url=None, internal_token=None, shard_count=None),
        role='instance-manager',
        identity='manager-1',
    )

    assert captured['minimum_schema_revision'] == REQUEST_BUDGET_LEASES_REVISION


async def test_control_gateway_requires_exact_database_schema(monkeypatch):
    captured = {}

    async def fake_initialize_distributed_config(**kwargs):
        captured.update(kwargs)
        return DistributedConfig(database_url='sqlite+aiosqlite:///bootstrap.db')

    monkeypatch.setattr(
        'nexus.distributed.config.store.initialize_distributed_config',
        fake_initialize_distributed_config,
    )

    await distributed_cli._initialize_service_config(
        argparse.Namespace(database_url=None, internal_token=None, shard_count=None),
        role='control-gateway',
        identity=None,
    )

    assert captured['minimum_schema_revision'] is None


def test_service_config_rejects_invalid_required_store_bootstrap_flag(monkeypatch):
    monkeypatch.setenv('NEXUS_DIST_CONFIG_STORE_REQUIRED', 'sometimes')

    with pytest.raises(ValueError, match='NEXUS_DIST_CONFIG_STORE_REQUIRED must be a boolean'):
        distributed_cli._config_store_required_from_env()


def test_distributed_db_engine_requires_database_url():
    with pytest.raises(ValueError, match='NEXUS_DIST_DATABASE_URL or NEXUS_DIST_DATABASE_URL_FILE is required'):
        create_engine(DistributedConfig(database_url=''))


def test_distributed_db_engine_uses_pool_settings(monkeypatch):
    captured = {}

    def fake_create_async_engine(url, **kwargs):
        captured['url'] = url
        captured['kwargs'] = kwargs
        return object()

    monkeypatch.setattr('nexus.distributed.persistence.db.create_async_engine', fake_create_async_engine)

    engine = create_engine(
        DistributedConfig(
            database_url='mysql+aiomysql://user:pass@host/db',
            db_pool_size=12,
            db_max_overflow=4,
            db_pool_timeout=2.5,
            db_pool_recycle=900,
        )
    )

    assert engine is not None
    assert captured == {
        'url': 'mysql+aiomysql://user:pass@host/db',
        'kwargs': {
            'pool_pre_ping': True,
            'isolation_level': 'READ COMMITTED',
            'pool_size': 12,
            'max_overflow': 4,
            'pool_timeout': 2.5,
            'pool_recycle': 900,
        },
    }


async def test_distributed_db_engine_logs_slow_sql_without_parameters(tmp_path, caplog):
    db_path = tmp_path / 'slow-sql.db'
    engine = create_engine(
        DistributedConfig(
            database_url=f'sqlite+aiosqlite:///{db_path}',
            slow_sql_log_threshold_ms=0.000001,
        )
    )

    caplog.set_level(logging.WARNING, logger='nexus.distributed.persistence.db')
    try:
        async with engine.connect() as conn:
            await conn.execute(text('SELECT :secret_value'), {'secret_value': 'sensitive-token'})
    finally:
        await engine.dispose()

    assert 'Distributed slow SQL' in caplog.text
    assert 'SELECT' in caplog.text
    assert 'sensitive-token' not in caplog.text


async def test_distributed_db_engine_logs_failed_sql_without_parameters(tmp_path, caplog):
    db_path = tmp_path / 'failed-sql.db'
    engine = create_engine(
        DistributedConfig(
            database_url=f'sqlite+aiosqlite:///{db_path}',
            slow_sql_log_threshold_ms=0.000001,
        )
    )

    caplog.set_level(logging.WARNING, logger='nexus.distributed.persistence.db')
    try:
        async with engine.connect() as conn:
            with pytest.raises(OperationalError):
                await conn.execute(text('SELECT * FROM missing_table WHERE token = :token'), {'token': 'secret-token'})
            await conn.execute(text('SELECT 1'))
    finally:
        await engine.dispose()

    assert 'Distributed SQL error' in caplog.text
    assert 'missing_table' in caplog.text
    assert 'secret-token' not in caplog.text


async def test_distributed_db_init_and_upgrade_create_schema(tmp_path):
    db_path = tmp_path / 'distributed.db'
    url = f'sqlite+aiosqlite:///{db_path}'

    await init_database(DistributedConfig(database_url=url))
    await upgrade_database(DistributedConfig(database_url=url))

    engine = create_async_engine(url)
    try:
        async with engine.connect() as conn:
            tables = await conn.run_sync(lambda sync_conn: set(inspect(sync_conn).get_table_names()))
            dependency_build_columns = await conn.run_sync(
                lambda sync_conn: {
                    column['name'] for column in inspect(sync_conn).get_columns('distributed_dependency_builds')
                }
            )
            batch_run_columns = await conn.run_sync(
                lambda sync_conn: {
                    column['name'] for column in inspect(sync_conn).get_columns('distributed_batch_runs')
                }
            )
    finally:
        await engine.dispose()

    assert {
        'executions',
        'assignments',
        'instance_managers',
        'execution_commands',
        'execution_events',
        'shard_leases',
        'distributed_artifacts',
        'distributed_artifact_refs',
        'distributed_artifact_uploads',
        'distributed_artifact_quota_usage',
        'distributed_batch_runs',
        'distributed_batch_items',
        'distributed_batch_item_attempts',
        'distributed_user_roles',
        'distributed_submit_tokens',
        'distributed_user_quota_overrides',
        'distributed_quota_usage',
        'distributed_audit_events',
        'schema_version',
        'alembic_version',
    } <= tables
    assert {
        'provider_type',
        'provider_config_ref',
        'provider_config_snapshot_json',
    } <= dependency_build_columns
    assert not {
        'running_items',
        'tide_models_json',
        'dispatch_policy_json',
        'quota_ratio',
        'concurrency_limit',
        'qpm_coefficient',
    } & batch_run_columns


async def test_distributed_db_schema_contains_internal_trace_carriers(tmp_path):
    db_path = tmp_path / 'distributed-trace-carriers.db'
    url = f'sqlite+aiosqlite:///{db_path}'

    await init_database(DistributedConfig(database_url=url))

    engine = create_async_engine(url)
    try:
        async with engine.connect() as conn:
            execution_columns = await conn.run_sync(
                lambda sync_conn: {column['name'] for column in inspect(sync_conn).get_columns('executions')}
            )
            assignment_columns = await conn.run_sync(
                lambda sync_conn: {column['name'] for column in inspect(sync_conn).get_columns('assignments')}
            )
    finally:
        await engine.dispose()

    assert 'trace_carrier_json' in execution_columns
    assert 'trace_carrier_json' in assignment_columns


async def test_distributed_db_init_creates_mysql_database_before_running_migrations(monkeypatch):
    created_engines = []
    executed_statements = []
    run_sync_calls = []

    class FakeConnection:
        def __init__(self, database: str | None):
            self.database = database

        async def execute(self, statement, params=None):
            executed_statements.append((self.database, str(statement), params))

        async def run_sync(self, fn, **kwargs):
            run_sync_calls.append((self.database, fn.__name__, kwargs))

    class FakeConnectionContext:
        def __init__(self, database: str | None):
            self.database = database

        async def __aenter__(self):
            return FakeConnection(self.database)

        async def __aexit__(self, exc_type, exc, tb):
            return None

    class FakeEngine:
        def __init__(self, url, **kwargs):
            self.url = url
            self.database = make_url(url).database
            self.kwargs = kwargs

        def connect(self):
            return FakeConnectionContext(self.database)

        def begin(self):
            return FakeConnectionContext(self.database)

        async def dispose(self):
            pass

    def fake_create_async_engine(url, **kwargs):
        engine = FakeEngine(url, **kwargs)
        created_engines.append((engine.database, kwargs))
        return engine

    monkeypatch.setattr(distributed_db, 'create_async_engine', fake_create_async_engine)

    await distributed_db.init_database(
        DistributedConfig(
            database_url='mysql+aiomysql://user:p%40ss@db.example.com:3306/nexus_runner_test?charset=utf8mb4'
        )
    )

    assert [database for database, _ in created_engines] == [None, 'nexus_runner_test']
    assert (
        None,
        'CREATE DATABASE IF NOT EXISTS `nexus_runner_test` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci',
        None,
    ) in executed_statements
    assert run_sync_calls == [('nexus_runner_test', '_run_migrations', {'commit': True})]


async def test_distributed_db_init_rejects_unsafe_mysql_database_names(monkeypatch):
    create_engine_calls = []

    def fake_create_async_engine(url, **kwargs):
        create_engine_calls.append(url)
        return object()

    monkeypatch.setattr(distributed_db, 'create_async_engine', fake_create_async_engine)

    with pytest.raises(ValueError, match='Unsafe MySQL database name'):
        await distributed_db.init_database(DistributedConfig(database_url='mysql+aiomysql://user:pass@db/nexus-runner'))

    assert create_engine_calls == []


async def test_distributed_db_upgrade_stamps_legacy_schema_version_database(tmp_path):
    db_path = tmp_path / 'legacy-distributed.db'
    url = f'sqlite+aiosqlite:///{db_path}'
    engine = create_async_engine(url)
    try:
        await create_schema(engine)
    finally:
        await engine.dispose()

    await upgrade_database(DistributedConfig(database_url=url))

    engine = create_async_engine(url)
    try:
        async with engine.connect() as conn:
            alembic_revision = await conn.scalar(text('SELECT version_num FROM alembic_version'))
            schema_version = await conn.scalar(text('SELECT version FROM schema_version WHERE version = 1'))
    finally:
        await engine.dispose()

    assert alembic_revision == HEAD_REVISION
    assert schema_version == 1


async def test_distributed_db_upgrade_from_cloud_0019_restores_config_overrides(tmp_path):
    db_path = tmp_path / 'distributed-cloud-0019.db'
    url = f'sqlite+aiosqlite:///{db_path}'
    engine = create_async_engine(url)

    def migrate_old_cloud_to_0019(sync_conn):
        applied_parent_cursor = False
        for migration in MIGRATIONS:
            if migration.revision == CONFIG_OVERRIDES_REVISION:
                continue
            if migration.revision == '0020_ensure_config_overrides':
                break
            module = importlib.import_module(migration.module)
            module.upgrade(Operations(MigrationContext.configure(sync_conn)))
            if migration.revision == '0019_add_execution_parent_cursor':
                applied_parent_cursor = True
                break
        if not applied_parent_cursor:
            module = importlib.import_module(
                'nexus.distributed.persistence.migrations.versions.0019_add_execution_parent_cursor'
            )
            module.upgrade(Operations(MigrationContext.configure(sync_conn)))
        distributed_db._set_alembic_revision(sync_conn, '0019_add_execution_parent_cursor')

    try:
        async with engine.begin() as conn:
            await conn.run_sync(migrate_old_cloud_to_0019)
            tables = await conn.run_sync(lambda sync_conn: set(inspect(sync_conn).get_table_names()))
            assert 'distributed_config_overrides' not in tables
    finally:
        await engine.dispose()

    await upgrade_database(DistributedConfig(database_url=url))

    engine = create_async_engine(url)
    try:
        async with engine.connect() as conn:
            tables = await conn.run_sync(lambda sync_conn: set(inspect(sync_conn).get_table_names()))
            revision = await conn.scalar(text('SELECT version_num FROM alembic_version'))
    finally:
        await engine.dispose()

    assert 'distributed_config_overrides' in tables
    assert revision == HEAD_REVISION


async def test_distributed_db_offline_upgrade_from_0003_clears_legacy_managers(tmp_path):
    db_path = tmp_path / 'distributed-0003.db'
    url = f'sqlite+aiosqlite:///{db_path}'
    engine = create_async_engine(url)

    def migrate_to_0003(sync_conn):
        for migration in MIGRATIONS[:3]:
            module = importlib.import_module(migration.module)
            module.upgrade(Operations(MigrationContext.configure(sync_conn)))
            distributed_db._set_alembic_revision(sync_conn, migration.revision)

    try:
        async with engine.begin() as conn:
            await conn.run_sync(migrate_to_0003)
            await conn.execute(
                text(
                    """
                    INSERT INTO instance_managers (
                        manager_id, session_id, endpoint_url,
                        environment_type, environment_name, pool_name,
                        labels_json, draining,
                        supported_modes_json, supported_spec_types_json,
                        supported_instance_modes_json,
                        capacity_total, observed_capacity_used,
                        local_instances_json, version, metadata_json,
                        last_heartbeat_at, created_at, updated_at
                    ) VALUES (
                        'legacy-manager', 'session-1', 'http://127.0.0.1:9001',
                        'predefined', 'legacy-environment', 'legacy-pool',
                        '{}', 0, '["job"]', '["nexus_single"]', '["asyncio"]',
                        1, 0, NULL, 'legacy', '{}',
                        CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
                    )
                    """
                )
            )
    finally:
        await engine.dispose()

    await upgrade_database(DistributedConfig(database_url=url))

    engine = create_async_engine(url)
    try:
        async with engine.connect() as conn:
            manager_count = await conn.scalar(text('SELECT COUNT(*) FROM instance_managers'))
            columns = await conn.run_sync(
                lambda sync_conn: {
                    column['name']: column for column in inspect(sync_conn).get_columns('instance_managers')
                }
            )
            revision = await conn.scalar(text('SELECT version_num FROM alembic_version'))
    finally:
        await engine.dispose()

    assert manager_count == 0
    assert revision == HEAD_REVISION
    assert {'environment_id', 'pool_id', 'workload'} <= columns.keys()
    assert all(columns[name]['nullable'] is False for name in ('environment_id', 'pool_id', 'workload'))
    assert {
        'environment_type',
        'environment_name',
        'pool_name',
        'environment_key',
        'workload_class',
    }.isdisjoint(columns)


async def test_distributed_db_upgrade_from_applied_0004_repeats_legacy_cleanup(tmp_path):
    db_path = tmp_path / 'distributed-applied-0004.db'
    url = f'sqlite+aiosqlite:///{db_path}'
    engine = create_async_engine(url)

    def migrate_to_0004(sync_conn):
        for migration in MIGRATIONS:
            module = importlib.import_module(migration.module)
            module.upgrade(Operations(MigrationContext.configure(sync_conn)))
            distributed_db._set_alembic_revision(sync_conn, migration.revision)
            if migration.revision == CLOUD_SUBMIT_REVISION:
                break

    try:
        async with engine.begin() as conn:
            await conn.run_sync(migrate_to_0004)
            for statement in (
                'ALTER TABLE executions ADD COLUMN environment_key VARCHAR(64)',
                'ALTER TABLE instance_managers ADD COLUMN environment_type VARCHAR(32)',
                'ALTER TABLE instance_managers ADD COLUMN environment_name VARCHAR(64)',
                'ALTER TABLE instance_managers ADD COLUMN pool_name VARCHAR(64)',
                'ALTER TABLE instance_managers ADD COLUMN environment_key VARCHAR(64)',
                'ALTER TABLE instance_managers ADD COLUMN workload_class VARCHAR(32)',
                'CREATE TABLE distributed_environment_demands (id INTEGER PRIMARY KEY)',
                'CREATE TABLE distributed_environment_pools (id INTEGER PRIMARY KEY)',
            ):
                await conn.execute(text(statement))
    finally:
        await engine.dispose()

    await upgrade_database(DistributedConfig(database_url=url))

    engine = create_async_engine(url)
    try:
        async with engine.connect() as conn:
            tables = await conn.run_sync(lambda sync_conn: set(inspect(sync_conn).get_table_names()))
            execution_columns = await conn.run_sync(
                lambda sync_conn: {column['name'] for column in inspect(sync_conn).get_columns('executions')}
            )
            manager_columns = await conn.run_sync(
                lambda sync_conn: {
                    column['name']: column for column in inspect(sync_conn).get_columns('instance_managers')
                }
            )
            revision = await conn.scalar(text('SELECT version_num FROM alembic_version'))
    finally:
        await engine.dispose()

    assert revision == HEAD_REVISION
    assert 'environment_key' not in execution_columns
    assert {
        'environment_type',
        'environment_name',
        'pool_name',
        'environment_key',
        'workload_class',
    }.isdisjoint(manager_columns)
    assert all(manager_columns[name]['nullable'] is False for name in ('environment_id', 'pool_id', 'workload'))
    assert {'distributed_environment_demands', 'distributed_environment_pools'}.isdisjoint(tables)


async def test_distributed_db_upgrade_from_0012_drops_legacy_quota_columns(tmp_path):
    db_path = tmp_path / 'distributed-applied-0012.db'
    url = f'sqlite+aiosqlite:///{db_path}'
    engine = create_async_engine(url)

    def migrate_to_0012(sync_conn):
        for migration in MIGRATIONS:
            module = importlib.import_module(migration.module)
            module.upgrade(Operations(MigrationContext.configure(sync_conn)))
            distributed_db._set_alembic_revision(sync_conn, migration.revision)
            if migration.revision == LEGACY_ENVIRONMENT_CLEANUP_REVISION:
                break

    try:
        async with engine.begin() as conn:
            await conn.run_sync(migrate_to_0012)
            await conn.execute(
                text('ALTER TABLE distributed_quota_usage ADD COLUMN active_environment_pools INTEGER NOT NULL')
            )
            await conn.execute(
                text('ALTER TABLE distributed_quota_usage ADD COLUMN desired_manager_replicas INTEGER NOT NULL')
            )
            await conn.execute(
                text(
                    """
                    INSERT INTO distributed_quota_usage (
                        owner_principal_id, environment, pool, quota_class,
                        queued_executions, running_executions, reserved_slots,
                        version, updated_at,
                        active_environment_pools, desired_manager_replicas
                    ) VALUES (
                        'owner-1', '*', '*', 'owner',
                        0, 0, 0,
                        1, CURRENT_TIMESTAMP,
                        2, 3
                    )
                    """
                )
            )
    finally:
        await engine.dispose()

    await upgrade_database(DistributedConfig(database_url=url))

    engine = create_async_engine(url)
    try:
        async with engine.connect() as conn:
            columns = await conn.run_sync(
                lambda sync_conn: {
                    column['name'] for column in inspect(sync_conn).get_columns('distributed_quota_usage')
                }
            )
            preserved = (
                await conn.execute(
                    text(
                        """
                        SELECT owner_principal_id, queued_executions, running_executions, reserved_slots
                        FROM distributed_quota_usage
                        """
                    )
                )
            ).one()
            revision = await conn.scalar(text('SELECT version_num FROM alembic_version'))
    finally:
        await engine.dispose()

    assert revision == HEAD_REVISION
    assert {'active_environment_pools', 'desired_manager_replicas'}.isdisjoint(columns)
    assert tuple(preserved) == ('owner-1', 0, 0, 0)


async def test_distributed_db_upgrade_from_0013_preserves_shard_count_baseline(tmp_path):
    db_path = tmp_path / 'distributed-applied-0013.db'
    url = f'sqlite+aiosqlite:///{db_path}'
    engine = create_async_engine(url)

    def migrate_to_0013(sync_conn):
        for migration in MIGRATIONS:
            module = importlib.import_module(migration.module)
            module.upgrade(Operations(MigrationContext.configure(sync_conn)))
            distributed_db._set_alembic_revision(sync_conn, migration.revision)
            if migration.revision == LEGACY_QUOTA_CLEANUP_REVISION:
                break

    try:
        async with engine.begin() as conn:
            await conn.run_sync(migrate_to_0013)
            for shard_id in range(4):
                await conn.execute(
                    text(
                        """
                        INSERT INTO scheduler_leases (
                            shard_group, shard_id, shard_count, owner_id,
                            lease_expires_at, updated_at
                        ) VALUES (
                            'default', :shard_id, 4, 'old-scheduler',
                            '2099-01-01 00:00:00', CURRENT_TIMESTAMP
                        )
                        """
                    ),
                    {'shard_id': shard_id},
                )
    finally:
        await engine.dispose()

    await upgrade_database(DistributedConfig(database_url=url))

    engine = create_async_engine(url)
    try:
        async with engine.connect() as conn:
            tables = await conn.run_sync(lambda sync_conn: set(inspect(sync_conn).get_table_names()))
            rows = (
                await conn.execute(
                    text(
                        """
                        SELECT shard_group, COUNT(*), MAX(shard_count), COUNT(DISTINCT owner_id)
                        FROM shard_leases
                        GROUP BY shard_group
                        ORDER BY shard_group
                        """
                    )
                )
            ).all()
            owners = set(await conn.scalars(text('SELECT DISTINCT owner_id FROM shard_leases')))
            revision = await conn.scalar(text('SELECT version_num FROM alembic_version'))
    finally:
        await engine.dispose()

    assert 'scheduler_leases' not in tables
    assert rows == [
        ('execution-reconciler', 4, 4, 1),
        ('execution-scheduler', 4, 4, 1),
    ]
    assert owners == {'migration-expired'}
    assert revision == HEAD_REVISION


def test_distributed_db_subcommands_dispatch(monkeypatch):
    called = []

    async def fake_init(config):
        called.append(('init', config.database_url))

    async def fake_upgrade(config):
        called.append(('upgrade', config.database_url))

    monkeypatch.setattr(distributed_cli, 'init_database', fake_init)
    monkeypatch.setattr(distributed_cli, 'upgrade_database', fake_upgrade)

    assert (
        distributed_cli.distributed_command(
            argparse.Namespace(
                distributed_command='db',
                db_command='init',
                database_url='sqlite+aiosqlite:///init.db',
                internal_token=None,
                shard_count=None,
            )
        )
        == 0
    )
    assert (
        distributed_cli.distributed_command(
            argparse.Namespace(
                distributed_command='db',
                db_command='upgrade',
                database_url='sqlite+aiosqlite:///upgrade.db',
                internal_token=None,
                shard_count=None,
            )
        )
        == 0
    )
    assert called == [
        ('init', 'sqlite+aiosqlite:///init.db'),
        ('upgrade', 'sqlite+aiosqlite:///upgrade.db'),
    ]


def test_distributed_db_command_validates_db_init_profile_before_execution(monkeypatch):
    calls = []

    class Config:
        database_url = 'sqlite+aiosqlite:///db-init.db'

        def validate_for_service(self, service):
            calls.append(('validate', service))

    async def fake_init(config):
        calls.append(('init', config.database_url))

    monkeypatch.setattr(distributed_cli, '_config_from_args', lambda args: Config())
    monkeypatch.setattr(distributed_cli, 'init_database', fake_init)

    assert distributed_cli.distributed_command(argparse.Namespace(distributed_command='db', db_command='init')) == 0
    assert calls == [('validate', 'db-init'), ('init', 'sqlite+aiosqlite:///db-init.db')]


def test_distributed_db_shell_builds_mycli_invocation_without_password_in_argv():
    invocation = distributed_cli._mycli_invocation_from_sqlalchemy_url(
        'mysql+aiomysql://user:p%40ss@db.example.com:3307/nexus_dist?charset=utf8mb4',
        prompt_password=False,
        extra_args=['--table'],
    )

    assert invocation.argv == [
        'mycli',
        '--host',
        'db.example.com',
        '--port',
        '3307',
        '--user',
        'user',
        '--database',
        'nexus_dist',
        '--charset',
        'utf8mb4',
        '--table',
    ]
    assert 'p@ss' not in invocation.argv
    assert invocation.env == {'MYSQL_PWD': 'p@ss'}


def test_distributed_db_shell_can_prompt_for_password():
    invocation = distributed_cli._mycli_invocation_from_sqlalchemy_url(
        'mysql+aiomysql://user:p%40ss@db.example.com/nexus_dist',
        prompt_password=True,
        extra_args=[],
    )

    assert invocation.argv == [
        'mycli',
        '--host',
        'db.example.com',
        '--user',
        'user',
        '--database',
        'nexus_dist',
    ]
    assert invocation.env == {}


def test_distributed_db_shell_requires_mysql_url():
    with pytest.raises(ValueError, match='only supports MySQL URLs'):
        distributed_cli._mycli_invocation_from_sqlalchemy_url(
            'sqlite+aiosqlite:///distributed.db',
            prompt_password=False,
            extra_args=[],
        )


def test_distributed_db_shell_dispatch_execs_mycli(monkeypatch):
    called = {}

    monkeypatch.setattr(distributed_cli.shutil, 'which', lambda binary: '/usr/bin/mycli' if binary == 'mycli' else None)

    def fake_execvpe(binary, argv, env):
        called['binary'] = binary
        called['argv'] = argv
        called['env'] = env

    monkeypatch.setattr(distributed_cli.os, 'execvpe', fake_execvpe)

    assert (
        distributed_cli.distributed_command(
            argparse.Namespace(
                distributed_command='db',
                db_command='shell',
                database_url='mysql+aiomysql://user:pass@db/nexus_dist',
                internal_token=None,
                shard_count=None,
                prompt_password=False,
                mycli_args=['--', '--table'],
            )
        )
        == 0
    )
    assert called['binary'] == 'mycli'
    assert called['argv'] == [
        'mycli',
        '--host',
        'db',
        '--user',
        'user',
        '--database',
        'nexus_dist',
        '--table',
    ]
    assert called['env']['MYSQL_PWD'] == 'pass'


def test_distributed_db_shell_returns_127_when_mycli_is_missing(monkeypatch):
    monkeypatch.setattr(distributed_cli.shutil, 'which', lambda binary: None)

    assert (
        distributed_cli.run_database_shell(
            DistributedConfig(database_url='mysql+aiomysql://user:pass@db/nexus_dist'),
            prompt_password=False,
            mycli_args=[],
        )
        == 127
    )


def test_distributed_parser_accepts_db_shell_command():
    parser = argparse.ArgumentParser(prog='nexus')
    subparsers = parser.add_subparsers(dest='command')
    distributed_cli.register_distributed_parser(subparsers)

    args = parser.parse_args(['distributed', 'db', 'shell', '--prompt-password', '--', '--table'])

    assert args.distributed_command == 'db'
    assert args.db_command == 'shell'
    assert args.prompt_password is True
    assert args.mycli_args == ['--', '--table']


def test_application_gateway_parser_accepts_explicit_live_compatibility_config():
    parser = argparse.ArgumentParser(prog='nexus')
    subparsers = parser.add_subparsers(dest='command')
    distributed_cli.register_distributed_parser(subparsers)

    args = parser.parse_args(
        [
            'distributed',
            'application-gateway',
            '--enable-live-compat',
            '--control-base-url',
            'http://control:8081',
            '--application-base-url',
            'http://application:8082',
            '--placement-environment',
            'predefined-local',
            '--placement-pool',
            'interactive',
            '--placement-label',
            'region=local',
        ]
    )

    assert args.enable_live_compat is True
    assert args.control_base_url == 'http://control:8081'
    assert args.application_base_url == 'http://application:8082'
    assert distributed_cli._placement_from_args(args) == {
        'environment': 'predefined-local',
        'pool': 'interactive',
        'labels': {'region': 'local'},
    }


def test_distributed_config_commands_manage_plain_and_secret_scopes_without_leaking(tmp_path, capsys):
    database_url = f'sqlite+aiosqlite:///{tmp_path / "config-management.db"}'
    asyncio.run(init_database(DistributedConfig(database_url=database_url)))
    capsys.readouterr()
    plain_file = tmp_path / 'plain.json'
    plain_file.write_text('{"scheduler_poll_interval": 0.25}', encoding='utf-8')
    secret_file = tmp_path / 'secret.json'
    secret_file.write_text('{"internal_token": "never-print-this"}', encoding='utf-8')

    parser = argparse.ArgumentParser(prog='nexus')
    subparsers = parser.add_subparsers(dest='command')
    distributed_cli.register_distributed_parser(subparsers)

    args = parser.parse_args(
        [
            'distributed',
            'config',
            'set',
            '--database-url',
            database_url,
            '--role',
            'scheduler',
            '--identity',
            '*',
            '--file',
            str(plain_file),
            '--expected-version',
            '0',
        ]
    )
    assert distributed_cli.distributed_command(args) == 0
    created = json.loads(capsys.readouterr().out)
    assert created == {
        'fields': ['scheduler_poll_interval'],
        'identity': '*',
        'role': 'scheduler',
        'values': {'scheduler_poll_interval': 0.25},
        'version': 1,
    }

    args = parser.parse_args(
        [
            'distributed',
            'config',
            'secret-set',
            '--database-url',
            database_url,
            '--role',
            'scheduler',
            '--identity',
            '*',
            '--file',
            str(secret_file),
            '--expected-version',
            '0',
        ]
    )
    assert distributed_cli.distributed_command(args) == 0
    secret_created_output = capsys.readouterr().out
    assert 'never-print-this' not in secret_created_output
    assert json.loads(secret_created_output) == {
        'fields': ['internal_token'],
        'identity': '*',
        'role': 'scheduler',
        'version': 1,
    }

    args = parser.parse_args(
        [
            'distributed',
            'config',
            'secret-list',
            '--database-url',
            database_url,
            '--role',
            'scheduler',
        ]
    )
    assert distributed_cli.distributed_command(args) == 0
    secret_list_output = capsys.readouterr().out
    assert 'never-print-this' not in secret_list_output
    assert json.loads(secret_list_output) == [
        {'fields': ['internal_token'], 'identity': '*', 'role': 'scheduler', 'version': 1}
    ]

    args = parser.parse_args(
        [
            'distributed',
            'config',
            'show',
            '--database-url',
            database_url,
            '--role',
            'scheduler',
            '--identity',
            '*',
        ]
    )
    assert distributed_cli.distributed_command(args) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown['values'] == {'scheduler_poll_interval': 0.25}
    assert shown['effective_values']['scheduler_poll_interval'] == 0.25
    assert shown['effective_values']['application_public_base_url'] == 'http://127.0.0.1:8082'
    assert 'database_url' not in shown['effective_values']
    assert 'database_url_file' not in shown['effective_values']
    assert 'secrets' not in shown['effective_values']


def test_distributed_config_secret_set_has_no_plaintext_value_option():
    parser = argparse.ArgumentParser(prog='nexus')
    subparsers = parser.add_subparsers(dest='command')
    distributed_cli.register_distributed_parser(subparsers)

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                'distributed',
                'config',
                'secret-set',
                '--role',
                'scheduler',
                '--identity',
                '*',
                '--expected-version',
                '0',
                '--value',
                'internal_token=forged',
            ]
        )


def test_distributed_parser_uses_instance_manager_command():
    parser = argparse.ArgumentParser(prog='nexus')
    subparsers = parser.add_subparsers(dest='command')
    distributed_cli.register_distributed_parser(subparsers)

    args = parser.parse_args(
        [
            'distributed',
            'instance-manager',
            '--instance-manager-id',
            'im-1',
            '--host',
            '127.0.0.1',
            '--port',
            '9001',
            '--environment-id',
            'environment-a',
            '--pool-id',
            'execution',
            '--workload',
            'execution',
        ]
    )

    assert args.distributed_command == 'instance-manager'
    assert args.instance_manager_id == 'im-1'
    assert args.environment_id == 'environment-a'
    assert args.pool_id == 'execution'
    assert args.workload == 'execution'

    with pytest.raises(SystemExit):
        parser.parse_args(['distributed', 'manager'])
    with pytest.raises(SystemExit):
        parser.parse_args(['distributed', 'instance-manager', '--environment-name', 'legacy'])


@pytest.mark.parametrize(
    ('args', 'expected'),
    [
        (argparse.Namespace(distributed_command='control-gateway'), ('control-gateway', None)),
        (
            argparse.Namespace(distributed_command='scheduler', owner_id='scheduler-7'),
            ('scheduler', 'scheduler-7'),
        ),
        (
            argparse.Namespace(distributed_command='reconciler', owner_id='reconciler-7'),
            ('reconciler', 'reconciler-7'),
        ),
        (
            argparse.Namespace(distributed_command='instance-manager', instance_manager_id='manager-3'),
            ('instance-manager', 'manager-3'),
        ),
    ],
)
def test_distributed_config_scope_comes_from_command_role_and_identity(args, expected):
    assert distributed_cli._config_scope_from_args(args) == expected


def test_service_config_is_initialized_before_logging_and_dispatch(monkeypatch):
    events = []
    config = DistributedConfig(database_url='sqlite+aiosqlite:///bootstrap.db')

    async def fake_initialize_service_config(args, *, role, identity):
        events.append(('config', role, identity))
        return config

    async def fake_run_scheduler(args, loaded_config):
        assert loaded_config is config
        events.append(('run', args.owner_id))

    async def fake_monitor_event_loop(name, operation):
        events.append(('monitor', name))
        return await operation

    monkeypatch.setattr(distributed_cli, '_initialize_service_config', fake_initialize_service_config)
    monkeypatch.setattr(
        distributed_cli,
        '_configure_distributed_logging',
        lambda loaded_config, component, identity=None: events.append(('logging', component, identity)),
    )
    monkeypatch.setattr(distributed_cli, '_run_scheduler', fake_run_scheduler)
    monkeypatch.setattr(distributed_cli, 'monitor_event_loop', fake_monitor_event_loop, raising=False)

    assert (
        distributed_cli.distributed_command(
            argparse.Namespace(
                distributed_command='scheduler',
                owner_id='scheduler-7',
                database_url=None,
                internal_token=None,
                shard_count=None,
            )
        )
        == 0
    )
    assert events == [
        ('config', 'scheduler', 'scheduler-7'),
        ('logging', 'scheduler', 'scheduler-7'),
        ('monitor', 'scheduler:scheduler-7'),
        ('run', 'scheduler-7'),
    ]


def test_instance_manager_cli_escalates_stalled_main_loop_with_bounded_hard_exit(monkeypatch):
    events = []
    config = DistributedConfig(
        database_url='sqlite+aiosqlite:///bootstrap.db',
        manager_self_fence_timeout=timedelta(seconds=30),
        manager_executor_hang_timeout=timedelta(minutes=3),
        manager_hard_exit_timeout=timedelta(seconds=5),
    )

    class FakeHardExitGuard:
        def arm(self, reason):
            events.append(('arm', reason))

        def disarm(self):
            events.append(('disarm',))

    guard = FakeHardExitGuard()

    async def fake_initialize_service_config(args, *, role, identity):
        del args
        events.append(('config', role, identity))
        return config

    async def fake_run_instance_manager(args, loaded_config, *, log_path, hard_exit_guard):
        assert loaded_config is config
        events.append(('run', args.instance_manager_id, log_path, hard_exit_guard is guard))

    async def fake_monitor_event_loop(name, operation, *, stall_timeout=None, on_stall=None):
        events.append(('monitor', name, stall_timeout))
        assert on_stall is not None
        on_stall(30.0)
        return await operation

    monkeypatch.setattr(distributed_cli, '_initialize_service_config', fake_initialize_service_config)
    monkeypatch.setattr(
        distributed_cli,
        '_configure_distributed_logging',
        lambda loaded_config, component, identity=None: '/tmp/manager.log',
    )
    monkeypatch.setattr(distributed_cli, '_run_instance_manager', fake_run_instance_manager)
    monkeypatch.setattr(distributed_cli, 'monitor_event_loop', fake_monitor_event_loop, raising=False)
    monkeypatch.setattr(
        distributed_cli,
        '_create_hard_exit_guard',
        lambda *, timeout: events.append(('guard', timeout)) or guard,
        raising=False,
    )

    assert (
        distributed_cli.distributed_command(
            argparse.Namespace(
                distributed_command='instance-manager',
                instance_manager_id='manager-3',
                database_url=None,
                internal_token=None,
                shard_count=None,
            )
        )
        == 0
    )
    assert events == [
        ('config', 'instance-manager', 'manager-3'),
        ('guard', 5.0),
        ('monitor', 'instance-manager:manager-3', 30.0),
        ('arm', 'manager_loop_stall'),
        ('run', 'manager-3', '/tmp/manager.log', True),
        ('disarm',),
    ]
