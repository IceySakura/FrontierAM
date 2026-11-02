from __future__ import annotations

import importlib

import pytest

from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, inspect, select
from sqlalchemy.dialects import mysql, sqlite
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from nexus.distributed.persistence.orm import Base, ExecutionRow, SubmitTokenRow, create_schema
from nexus.distributed.repositories.access import AccessRepository
from nexus.distributed.repositories.environment import EnvironmentRepository
from nexus.distributed.utils.clock import utcnow


initial_migration = importlib.import_module('nexus.distributed.persistence.migrations.versions.0001_initial')
cleanup_migration = importlib.import_module(
    'nexus.distributed.persistence.migrations.versions.0002_add_execution_cleanup'
)
trace_migration = importlib.import_module('nexus.distributed.persistence.migrations.versions.0003_add_trace_carriers')
cloud_migration = importlib.import_module(
    'nexus.distributed.persistence.migrations.versions.0004_cloud_submit_foundation'
)
run_identity_migration = importlib.import_module(
    'nexus.distributed.persistence.migrations.versions.0005_unify_distributed_run_identity'
)
deadline_migration = importlib.import_module(
    'nexus.distributed.persistence.migrations.versions.0006_add_execution_deadlines'
)
batch_lifecycle_migration = importlib.import_module(
    'nexus.distributed.persistence.migrations.versions.0007_add_batch_lifecycle_claim'
)
batch_item_index_migration = importlib.import_module(
    'nexus.distributed.persistence.migrations.versions.0008_add_batch_item_hot_path_index'
)
batch_item_collation_migration = importlib.import_module(
    'nexus.distributed.persistence.migrations.versions.0009_add_batch_item_binary_collation'
)
batch_lifecycle_fairness_migration = importlib.import_module(
    'nexus.distributed.persistence.migrations.versions.0010_add_batch_lifecycle_fairness'
)


@pytest.fixture
async def domain_session_factory(tmp_path):
    engine = create_async_engine(f'sqlite+aiosqlite:///{tmp_path / "domain.db"}')
    await create_schema(engine)
    yield async_sessionmaker(engine, expire_on_commit=False)
    await engine.dispose()


def test_cloud_foundation_migration_converges_0003_schema_and_is_idempotent(tmp_path):
    engine = create_engine(f'sqlite:///{tmp_path / "migration.db"}')

    with engine.begin() as connection:
        operations = Operations(MigrationContext.configure(connection))
        initial_migration.upgrade(operations)
        cleanup_migration.upgrade(operations)
        trace_migration.upgrade(operations)
        cloud_migration.upgrade(operations)
        cloud_migration.upgrade(operations)

        inspector = inspect(connection)
        tables = set(inspector.get_table_names())
        assert {
            'distributed_principals',
            'distributed_environments',
            'distributed_environment_grants',
            'distributed_environment_allocations',
            'distributed_environment_resource_usage',
            'distributed_submit_tokens',
        } <= tables
        assert 'distributed_environment_demands' not in tables
        assert 'distributed_environment_pools' not in tables

        execution_columns = {column['name']: column for column in inspector.get_columns('executions')}
        assert execution_columns['owner_principal_id']['nullable'] is True
        assert 'environment_key' not in execution_columns
        assert {
            'environment_id',
            'pool_id',
            'allocation_id',
            'workload',
            'submission_sha256',
            'allocation_spec_hash',
            'result_artifact_id',
            'first_schedulable_at',
            'stop_reason',
        } <= execution_columns.keys()

        assignment_columns = {column['name'] for column in inspector.get_columns('assignments')}
        assert {
            'owner_principal_id',
            'environment_id',
            'pool_id',
            'allocation_id',
            'workload',
            'allocation_spec_hash',
        } <= assignment_columns

        manager_columns = {column['name'] for column in inspector.get_columns('instance_managers')}
        assert {
            'environment_type',
            'environment_name',
            'pool_name',
            'environment_key',
            'workload_class',
        }.isdisjoint(manager_columns)
        assert {
            'owner_principal_id',
            'environment_id',
            'pool_id',
            'allocation_id',
            'workload',
            'allocation_spec_hash',
            'capability_snapshot_json',
        } <= manager_columns

        execution_indexes = {index['name'] for index in inspector.get_indexes('executions')}
        assert {'idx_executions_owner_cursor', 'idx_executions_allocation_status'} <= execution_indexes

        allocation_indexes = {index['name'] for index in inspector.get_indexes('distributed_environment_allocations')}
        assert {'idx_allocations_status_updated', 'idx_allocations_environment_status'} <= allocation_indexes

        upload_columns = {column['name'] for column in inspector.get_columns('distributed_artifact_uploads')}
        assert {'failure_count', 'last_failure_code'} <= upload_columns

    engine.dispose()

    assert ExecutionRow.__table__.c.idempotency_key.type.length == 256


def test_run_identity_migration_removes_legacy_batch_item_run_id(tmp_path):
    engine = create_engine(f'sqlite:///{tmp_path / "run-identity-migration.db"}')

    with engine.begin() as connection:
        connection.exec_driver_sql(
            """
            CREATE TABLE distributed_batch_items (
                parent_execution_id VARCHAR(64) NOT NULL,
                item_id VARCHAR(64) NOT NULL,
                run_id VARCHAR(256) NOT NULL,
                PRIMARY KEY (parent_execution_id, item_id),
                CONSTRAINT uq_batch_items_run_id UNIQUE (parent_execution_id, run_id)
            )
            """
        )
        operations = Operations(MigrationContext.configure(connection))
        run_identity_migration.upgrade(operations)
        run_identity_migration.upgrade(operations)

        inspector = inspect(connection)
        assert 'run_id' not in {column['name'] for column in inspector.get_columns('distributed_batch_items')}
        assert 'uq_batch_items_run_id' not in {
            constraint['name'] for constraint in inspector.get_unique_constraints('distributed_batch_items')
        }

    engine.dispose()


def test_execution_deadline_migration_adds_nullable_columns_and_scan_indexes(tmp_path):
    engine = create_engine(f'sqlite:///{tmp_path / "execution-deadline-migration.db"}')

    with engine.begin() as connection:
        operations = Operations(MigrationContext.configure(connection))
        initial_migration.upgrade(operations)
        cleanup_migration.upgrade(operations)
        trace_migration.upgrade(operations)
        cloud_migration.upgrade(operations)
        run_identity_migration.upgrade(operations)
        deadline_migration.upgrade(operations)
        deadline_migration.upgrade(operations)

        inspector = inspect(connection)
        execution_columns = {column['name']: column for column in inspector.get_columns('executions')}
        for column_name in ('queue_expires_at', 'execution_expires_at', 'stop_deadline_at'):
            assert execution_columns[column_name]['nullable'] is True
        execution_indexes = {index['name']: index['column_names'] for index in inspector.get_indexes('executions')}
        assert execution_indexes['idx_executions_queue_deadline'] == [
            'status',
            'queue_expires_at',
            'execution_id',
        ]
        assert execution_indexes['idx_executions_execution_deadline'] == [
            'status',
            'execution_expires_at',
            'execution_id',
        ]
        assert execution_indexes['idx_executions_stop_deadline'] == [
            'status',
            'stop_deadline_at',
            'execution_id',
        ]

    engine.dispose()


def test_batch_lifecycle_migration_adds_claim_and_cursor_contract(tmp_path):
    engine = create_engine(f'sqlite:///{tmp_path / "batch-lifecycle-migration.db"}')

    with engine.begin() as connection:
        operations = Operations(MigrationContext.configure(connection))
        initial_migration.upgrade(operations)
        cleanup_migration.upgrade(operations)
        trace_migration.upgrade(operations)
        cloud_migration.upgrade(operations)
        run_identity_migration.upgrade(operations)
        deadline_migration.upgrade(operations)
        batch_lifecycle_migration.upgrade(operations)
        batch_lifecycle_migration.upgrade(operations)

        inspector = inspect(connection)
        columns = {column['name'] for column in inspector.get_columns('distributed_batch_runs')}
        assert {
            'child_sync_cursor',
            'lifecycle_owner_id',
            'lifecycle_operation_id',
            'lifecycle_lease_expires_at',
            'last_reconciled_at',
        } <= columns
        indexes = {index['name']: index['column_names'] for index in inspector.get_indexes('distributed_batch_runs')}
        assert indexes['idx_batch_runs_lifecycle_claim'] == [
            'control_state',
            'lifecycle_lease_expires_at',
            'parent_execution_id',
        ]

    engine.dispose()


def test_batch_lifecycle_fairness_migration_replaces_legacy_claim_index(tmp_path):
    engine = create_engine(f'sqlite:///{tmp_path / "batch-lifecycle-fairness-migration.db"}')

    with engine.begin() as connection:
        operations = Operations(MigrationContext.configure(connection))
        initial_migration.upgrade(operations)
        cleanup_migration.upgrade(operations)
        trace_migration.upgrade(operations)
        cloud_migration.upgrade(operations)
        existing_indexes = {index['name'] for index in inspect(connection).get_indexes('distributed_batch_runs')}
        if 'idx_batch_runs_lifecycle_claim' in existing_indexes:
            operations.drop_index('idx_batch_runs_lifecycle_claim', table_name='distributed_batch_runs')
        operations.create_index(
            'idx_batch_runs_lifecycle_claim',
            'distributed_batch_runs',
            ['control_state', 'lifecycle_lease_expires_at', 'parent_execution_id'],
        )

        batch_lifecycle_fairness_migration.upgrade(operations)
        batch_lifecycle_fairness_migration.upgrade(operations)

        inspector = inspect(connection)
        columns = {column['name'] for column in inspector.get_columns('distributed_batch_runs')}
        assert 'last_reconciled_at' in columns
        indexes = {index['name']: index['column_names'] for index in inspector.get_indexes('distributed_batch_runs')}
        assert indexes['idx_batch_runs_lifecycle_claim'] == [
            'control_state',
            'last_reconciled_at',
            'lifecycle_lease_expires_at',
            'parent_execution_id',
        ]

    engine.dispose()


def test_batch_item_hot_path_migration_adds_pending_ordinal_index(tmp_path):
    engine = create_engine(f'sqlite:///{tmp_path / "batch-item-index-migration.db"}')

    with engine.begin() as connection:
        operations = Operations(MigrationContext.configure(connection))
        initial_migration.upgrade(operations)
        cleanup_migration.upgrade(operations)
        trace_migration.upgrade(operations)
        cloud_migration.upgrade(operations)
        run_identity_migration.upgrade(operations)
        deadline_migration.upgrade(operations)
        batch_lifecycle_migration.upgrade(operations)
        batch_item_index_migration.upgrade(operations)
        batch_item_index_migration.upgrade(operations)

        indexes = {
            index['name']: index['column_names'] for index in inspect(connection).get_indexes('distributed_batch_items')
        }
        assert indexes['idx_batch_items_parent_status_ordinal'] == [
            'parent_execution_id',
            'status',
            'task_ordinal',
        ]

    engine.dispose()


def test_batch_item_binary_collation_migration_is_idempotent_on_sqlite(tmp_path):
    engine = create_engine(f'sqlite:///{tmp_path / "batch-item-collation-migration.db"}')

    with engine.begin() as connection:
        operations = Operations(MigrationContext.configure(connection))
        initial_migration.upgrade(operations)
        cleanup_migration.upgrade(operations)
        trace_migration.upgrade(operations)
        cloud_migration.upgrade(operations)
        run_identity_migration.upgrade(operations)
        deadline_migration.upgrade(operations)
        batch_lifecycle_migration.upgrade(operations)
        batch_item_index_migration.upgrade(operations)
        batch_item_collation_migration.upgrade(operations)
        batch_item_collation_migration.upgrade(operations)

        assert {
            'item_id',
            'task_ordinal',
            'status',
        } <= {column['name'] for column in inspect(connection).get_columns('distributed_batch_items')}

    engine.dispose()


def test_fresh_schema_omits_legacy_environment_tables_and_columns():
    tables = Base.metadata.tables
    assert 'distributed_environment_demands' not in tables
    assert 'distributed_environment_pools' not in tables
    assert 'environment_key' not in tables['executions'].c
    assert {
        'environment_type',
        'environment_name',
        'pool_name',
        'environment_key',
        'workload_class',
    }.isdisjoint(tables['instance_managers'].c.keys())
    for table_name, column_names in {
        'executions': ('environment_id', 'pool_id'),
        'assignments': ('environment_id', 'pool_id'),
        'instance_managers': ('environment_id', 'pool_id'),
        'distributed_environment_allocations': ('environment_id', 'pool_id'),
        'distributed_quota_usage': ('environment', 'pool'),
    }.items():
        for column_name in column_names:
            assert tables[table_name].c[column_name].type.length == 64


def test_batch_item_id_columns_use_binary_database_collation():
    tables = Base.metadata.tables
    for table_name, column_name in (
        ('distributed_batch_items', 'item_id'),
        ('distributed_batch_item_attempts', 'item_id'),
        ('distributed_batch_runs', 'stop_cursor'),
        ('distributed_batch_runs', 'child_sync_cursor'),
    ):
        column_type = tables[table_name].c[column_name].type
        assert column_type.dialect_impl(mysql.dialect()).collation == 'utf8mb4_bin'
        assert column_type.dialect_impl(sqlite.dialect()).collation == 'BINARY'


async def test_environment_repository_rejects_identity_longer_than_database_contract(domain_session_factory):
    async with domain_session_factory() as session:
        with pytest.raises(ValueError, match='invalid_environment_id'):
            await EnvironmentRepository(session).create_environment(
                environment_id='e' * 65,
                name='oversized-environment',
                allocation_type='predefined',
                provider_type='predefined',
                provider_config_ref=None,
                resource_limits={},
                default_timeouts={},
                status='disabled',
                actor='admin',
                now=utcnow(),
            )


async def test_environment_configuration_is_validated_and_versioned(domain_session_factory):
    now = utcnow()
    async with domain_session_factory() as session:
        repository = EnvironmentRepository(session)
        with pytest.raises(ValueError, match='invalid_environment_provider'):
            await repository.create_environment(
                environment_id='env-invalid',
                name='invalid',
                allocation_type='predefined',
                provider_type='kubernetes',
                provider_config_ref='k8s/default',
                resource_limits={},
                default_timeouts={},
                status='disabled',
                actor='admin',
                now=now,
            )
        with pytest.raises(ValueError, match='predefined_provider_config_ref_forbidden'):
            await repository.create_environment(
                environment_id='env-predefined-ref',
                name='predefined-ref',
                allocation_type='predefined',
                provider_type='predefined',
                provider_config_ref='predefined/default',
                resource_limits={},
                default_timeouts={},
                status='disabled',
                actor='admin',
                now=now,
            )

        environment = await repository.create_environment(
            environment_id='env-kind',
            name='kind',
            allocation_type='dynamic',
            provider_type='kubernetes',
            provider_config_ref='k8s/default',
            resource_limits={'manager_count': 100, 'cpu_millicores': 200_000, 'memory_bytes': 1_000_000},
            default_timeouts={'queue_ttl_seconds': 300},
            status='disabled',
            actor='admin',
            now=now,
        )
        assert environment.config_version == 1
        assert environment.status == 'disabled'

        enabled = await repository.update_environment(
            'env-kind',
            status='enabled',
            default_timeouts={'queue_ttl_seconds': 600},
            actor='admin',
            now=utcnow(),
        )
        assert enabled.config_version == 2
        assert enabled.status == 'enabled'
        assert enabled.default_timeouts == {'queue_ttl_seconds': 600}

        predefined = await repository.update_environment(
            'env-kind',
            allocation_type='predefined',
            provider_type='predefined',
            provider_config_ref=None,
            actor='admin',
            now=utcnow(),
        )
        assert predefined.provider_config_ref is None


async def test_environment_grant_controls_submission_access(domain_session_factory):
    now = utcnow()
    async with domain_session_factory() as session:
        access = AccessRepository(session)
        await access.upsert_user(
            principal_id='principal-1',
            issuer='tai',
            tenant='tenant',
            subject='user',
            roles={'user'},
            display_name='User',
            actor='admin',
            now=now,
        )
        repository = EnvironmentRepository(session)
        await repository.create_environment(
            environment_id='env-predefined',
            name='predefined',
            allocation_type='predefined',
            provider_type='predefined',
            provider_config_ref=None,
            resource_limits={'slots': 100},
            default_timeouts={},
            status='enabled',
            actor='admin',
            now=now,
        )

        with pytest.raises(KeyError, match='environment_grant_not_found'):
            await repository.require_submission_access('principal-1', 'env-predefined')

        grant = await repository.upsert_grant(
            principal_id='principal-1',
            environment_id='env-predefined',
            limits={'slots': 10, 'active_batches': 1},
            scopes={'submission:single', 'artifact:write'},
            enabled=True,
            actor='admin',
            now=now,
        )
        assert grant.version == 1
        assert grant.scopes == frozenset({'submission:single', 'artifact:write'})
        authorized_environment, authorized_grant = await repository.require_submission_access(
            'principal-1',
            'env-predefined',
            required_scope='submission:single',
        )
        assert authorized_environment.environment_id == 'env-predefined'
        assert authorized_grant.limits == {'slots': 10, 'active_batches': 1}
        with pytest.raises(ValueError, match='environment_scope_required'):
            await repository.require_submission_access(
                'principal-1',
                'env-predefined',
                required_scope='submission:batch',
            )

        disabled_grant = await repository.upsert_grant(
            principal_id='principal-1',
            environment_id='env-predefined',
            limits={'slots': 10, 'active_batches': 1},
            enabled=False,
            actor='admin',
            now=utcnow(),
        )
        assert disabled_grant.version == 2
        assert disabled_grant.scopes == frozenset({'submission:single', 'artifact:write'})
        with pytest.raises(ValueError, match='environment_grant_disabled'):
            await repository.require_submission_access('principal-1', 'env-predefined')

        empty_grant = await repository.upsert_grant(
            principal_id='principal-1',
            environment_id='env-predefined',
            limits={'slots': 10, 'active_batches': 1},
            scopes=set(),
            enabled=True,
            actor='admin',
            now=utcnow(),
        )
        assert empty_grant.scopes == frozenset()
        with pytest.raises(ValueError, match='environment_scope_required'):
            await repository.require_submission_access(
                'principal-1',
                'env-predefined',
                required_scope='submission:single',
            )


async def test_submit_token_is_bound_to_one_granted_environment(domain_session_factory):
    now = utcnow()
    async with domain_session_factory() as session:
        access = AccessRepository(session)
        await access.upsert_user(
            principal_id='principal-token',
            issuer='tai',
            tenant='tenant',
            subject='token-user',
            roles={'user'},
            display_name=None,
            actor='admin',
            now=now,
        )
        environments = EnvironmentRepository(session)
        await environments.create_environment(
            environment_id='env-token',
            name='token-env',
            allocation_type='dynamic',
            provider_type='tkex',
            provider_config_ref='tkex/default',
            resource_limits={},
            default_timeouts={},
            status='enabled',
            actor='admin',
            now=now,
        )
        await environments.upsert_grant(
            principal_id='principal-token',
            environment_id='env-token',
            limits={},
            enabled=True,
            actor='admin',
            now=now,
        )

        with pytest.raises(ValueError, match='environment_id_required'):
            await access.create_token(
                owner_principal_id='principal-token',
                scopes={'submission:single'},
                created_by='principal-token',
                key_id='v1',
                key='secret',
                now=now,
            )

        token_row, token = await access.create_token(
            owner_principal_id='principal-token',
            environment_id='env-token',
            scopes={'submit:single'},
            created_by='principal-token',
            key_id='v1',
            key='secret',
            now=now,
        )
        assert token_row.environment_id == 'env-token'
        assert token_row.scopes_json == ['submission:single']

        authenticated = await access.authenticate_token(token, keys={'v1': 'secret'}, now=utcnow())
        assert authenticated is not None
        assert authenticated.environment_id == 'env-token'
        assert 'submission:single' in authenticated.scopes
        assert 'submit:single' in authenticated.scopes
        assert (await session.get(SubmitTokenRow, token_row.token_id)).environment_id == 'env-token'


async def test_system_default_token_is_idempotent_and_has_no_bearer_material(domain_session_factory):
    now = utcnow()
    async with domain_session_factory() as session:
        access = AccessRepository(session)
        await access.upsert_user(
            principal_id='principal-default-token',
            issuer='tai',
            tenant='default',
            subject='79471',
            roles={'user'},
            display_name='lawrentwang',
            actor='system:tai-identity',
            now=now,
        )

        first = await access.ensure_system_token(
            owner_principal_id='principal-default-token',
            created_by='system:tai-identity',
            now=now,
        )
        second = await access.ensure_system_token(
            owner_principal_id='principal-default-token',
            created_by='system:tai-identity',
            now=now,
        )
        await session.commit()

        assert first.token_id == second.token_id
        assert first.token_kind == 'system_default'
        assert first.owner_principal_id == 'principal-default-token'
        assert first.environment_id is None
        assert first.token_prefix is None
        assert first.token_hash is None
        assert first.token_hash_key_id is None
        assert first.scopes_json == []
        rows = list(await session.scalars(select(SubmitTokenRow)))
        assert [row.token_id for row in rows] == [first.token_id]
        assert await access.list_tokens(owner_principal_id='principal-default-token') == []


async def test_environment_persists_default_quota_snapshot(domain_session_factory):
    quota = {
        'manager_count': 2,
        'cpu_millicores': 10_000,
        'memory_bytes': 20 * 1024**3,
        'slots': 40,
        'active_batches': 1,
    }
    async with domain_session_factory() as session:
        environment = await EnvironmentRepository(session).create_environment(
            environment_id='env-default-quota',
            name='Default quota',
            allocation_type='predefined',
            provider_type='predefined',
            provider_config_ref=None,
            resource_limits={
                'manager_count': 5,
                'cpu_millicores': 20_000,
                'memory_bytes': 40 * 1024**3,
                'slots': 100,
                'active_batches': 2,
            },
            default_timeouts={},
            default_quota=quota,
            status='enabled',
            actor='admin',
            now=utcnow(),
        )
        await session.commit()

    assert environment.default_quota == quota
