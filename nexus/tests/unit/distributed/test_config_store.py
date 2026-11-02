from __future__ import annotations

import pytest

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from nexus.distributed.config import DistributedConfig
from nexus.distributed.config.store import (
    ConfigVersionConflictError,
    delete_config_override,
    get_config_override,
    initialize_distributed_config,
    list_secret_override_summaries,
    load_distributed_config,
    schema_revision_is_compatible,
    set_config_override,
)
from nexus.distributed.persistence.db import init_database
from nexus.distributed.persistence.orm import (
    AuditRow,
    Base,
    DistributedConfigOverrideRow,
    DistributedConfigSecretOverrideRow,
)


@pytest.fixture
async def config_session_factory():
    engine = create_async_engine('sqlite+aiosqlite:///:memory:')
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    try:
        yield async_sessionmaker(engine, expire_on_commit=False)
    finally:
        await engine.dispose()


async def _store_overrides(session_factory, *rows: tuple[str, str, dict]) -> None:
    async with session_factory() as session:
        session.add_all(
            DistributedConfigOverrideRow(role=role, identity=identity, config_json=config)
            for role, identity, config in rows
        )
        await session.commit()


async def _store_secret_overrides(session_factory, *rows: tuple[str, str, dict]) -> None:
    async with session_factory() as session:
        session.add_all(
            DistributedConfigSecretOverrideRow(role=role, identity=identity, secrets_json=secrets)
            for role, identity, secrets in rows
        )
        await session.commit()


async def test_db_secrets_merge_global_role_and_identity_in_order(config_session_factory) -> None:
    await _store_secret_overrides(
        config_session_factory,
        ('*', '*', {'internal_token': 'global-token', 'artifact_writer': {'username': 'global', 'token': 'g'}}),
        ('application-gateway', '*', {'artifact_writer': {'username': 'role', 'token': 'r'}}),
        ('application-gateway', 'app-1', {'internal_token': 'instance-token'}),
    )

    config = await load_distributed_config(
        config_session_factory,
        database_url='sqlite+aiosqlite:///bootstrap.db',
        role='application-gateway',
        identity='app-1',
    )

    assert config.secrets.read_internal_token() == 'instance-token'
    assert config.secrets.artifact_writer is not None
    assert config.secrets.artifact_writer.username == 'role'
    assert config.secrets.artifact_writer.token.get_secret_value() == 'r'


async def test_trace_database_maintenance_secret_is_scoped_to_control_gateway(config_session_factory) -> None:
    await _store_secret_overrides(
        config_session_factory,
        (
            'control-gateway',
            '*',
            {
                'trace_database_credentials': {
                    'tdsql-maintenance': {'username': 'account-admin', 'password': 'maintenance-secret'}
                }
            },
        ),
    )

    control = await load_distributed_config(
        config_session_factory,
        database_url='sqlite+aiosqlite:///bootstrap.db',
        role='control-gateway',
    )
    application = await load_distributed_config(
        config_session_factory,
        database_url='sqlite+aiosqlite:///bootstrap.db',
        role='application-gateway',
    )

    credential = control.resolve_trace_database_credential('tdsql-maintenance')
    assert credential is not None
    assert credential.read() == ('account-admin', 'maintenance-secret')
    assert application.resolve_trace_database_credential('tdsql-maintenance') is None


async def test_db_config_rejects_secret_fields(config_session_factory) -> None:
    await _store_overrides(config_session_factory, ('*', '*', {'secrets': {'internal_token': 'forged'}}))

    with pytest.raises(ValueError, match='secret field.*secrets'):
        await load_distributed_config(
            config_session_factory,
            database_url='sqlite+aiosqlite:///bootstrap.db',
            role='scheduler',
        )


async def test_db_secrets_reject_ordinary_fields(config_session_factory) -> None:
    await _store_secret_overrides(config_session_factory, ('*', '*', {'scheduler_poll_interval': 0.1}))

    with pytest.raises(ValueError, match='ordinary field.*secret scope.*scheduler_poll_interval'):
        await load_distributed_config(
            config_session_factory,
            database_url='sqlite+aiosqlite:///bootstrap.db',
            role='scheduler',
        )


async def test_db_config_merges_global_role_and_identity_in_order(config_session_factory) -> None:
    await _store_overrides(
        config_session_factory,
        ('*', '*', {'scheduler_poll_interval': 4.0, 'manager_poll_interval': 8.0}),
        ('scheduler', '*', {'scheduler_poll_interval': 2.0}),
        ('scheduler', 'scheduler-1', {'scheduler_poll_interval': 1.0}),
    )

    config = await load_distributed_config(
        config_session_factory,
        database_url='sqlite+aiosqlite:///bootstrap.db',
        role='scheduler',
        identity='scheduler-1',
    )

    assert config.scheduler_poll_interval == 1.0
    assert config.manager_poll_interval == 8.0


async def test_db_config_enables_trusted_x_rio_app_id_mode(config_session_factory) -> None:
    await _store_overrides(config_session_factory, ('*', '*', {'tai_trust_x_rio_app_id': True}))

    config = await load_distributed_config(
        config_session_factory,
        database_url='sqlite+aiosqlite:///bootstrap.db',
        role='control-gateway',
    )

    assert config.tai_trust_x_rio_app_id is True


async def test_required_stored_config_fails_closed_without_global_baseline(config_session_factory) -> None:
    with pytest.raises(ValueError, match='stored distributed config baseline is required'):
        await load_distributed_config(
            config_session_factory,
            database_url='sqlite+aiosqlite:///bootstrap.db',
            role='control-gateway',
            require_stored_config=True,
        )


async def test_required_stored_config_requires_authenticated_baseline(config_session_factory) -> None:
    await _store_overrides(config_session_factory, ('*', '*', {'scheduler_poll_interval': 0.5}))
    await _store_secret_overrides(config_session_factory, ('*', '*', {'internal_token': 'stored-token'}))

    with pytest.raises(ValueError, match='auth_mode=identity or auth_mode=required'):
        await load_distributed_config(
            config_session_factory,
            database_url='sqlite+aiosqlite:///bootstrap.db',
            role='control-gateway',
            require_stored_config=True,
        )


async def test_required_stored_config_accepts_complete_global_baseline(config_session_factory) -> None:
    await _store_overrides(
        config_session_factory,
        ('*', '*', {'auth_mode': 'required'}),
    )
    await _store_secret_overrides(config_session_factory, ('*', '*', {'internal_token': 'stored-token'}))

    config = await load_distributed_config(
        config_session_factory,
        database_url='sqlite+aiosqlite:///bootstrap.db',
        role='control-gateway',
        require_stored_config=True,
    )

    assert config.production_mode is False
    assert config.auth_mode == 'required'
    assert config.config_store_required is True


async def test_db_config_overrides_environment_values(config_session_factory) -> None:
    await _store_overrides(
        config_session_factory,
        ('scheduler', '*', {'scheduler_poll_interval': 2.0}),
    )

    config = await load_distributed_config(
        config_session_factory,
        database_url='sqlite+aiosqlite:///bootstrap.db',
        role='scheduler',
        environment_values={'scheduler_poll_interval': 9.0},
    )

    assert config.scheduler_poll_interval == 2.0


async def test_explicit_overrides_win_and_database_url_stays_bootstrap_only(config_session_factory) -> None:
    await _store_overrides(
        config_session_factory,
        ('scheduler', '*', {'scheduler_poll_interval': 2.0, 'db_pool_size': 12}),
    )

    config = await load_distributed_config(
        config_session_factory,
        database_url='sqlite+aiosqlite:///bootstrap.db',
        role='scheduler',
        identity='scheduler-1',
        overrides={'scheduler_poll_interval': 0.25},
    )

    assert config.database_url == 'sqlite+aiosqlite:///bootstrap.db'
    assert config.scheduler_poll_interval == 0.25
    assert config.db_pool_size == 12


async def test_db_config_rejects_database_url_override(config_session_factory) -> None:
    await _store_overrides(
        config_session_factory,
        ('*', '*', {'database_url': 'mysql+aiomysql://forged/db'}),
    )

    with pytest.raises(ValueError, match='database_url.*bootstrap-only'):
        await load_distributed_config(
            config_session_factory,
            database_url='sqlite+aiosqlite:///bootstrap.db',
            role='scheduler',
        )


async def test_db_config_rejects_database_url_file_override(config_session_factory) -> None:
    await _store_overrides(
        config_session_factory,
        ('*', '*', {'database_url_file': '/tmp/forged-database-url'}),
    )

    with pytest.raises(ValueError, match='database_url_file.*bootstrap-only'):
        await load_distributed_config(
            config_session_factory,
            database_url='sqlite+aiosqlite:///bootstrap.db',
            role='scheduler',
        )


async def test_db_config_rejects_config_store_requirement_override(config_session_factory) -> None:
    await _store_overrides(
        config_session_factory,
        ('*', '*', {'auth_mode': 'required', 'config_store_required': False}),
    )
    await _store_secret_overrides(config_session_factory, ('*', '*', {'internal_token': 'stored-token'}))

    with pytest.raises(ValueError, match='config_store_required.*bootstrap-only'):
        await load_distributed_config(
            config_session_factory,
            database_url='sqlite+aiosqlite:///bootstrap.db',
            role='scheduler',
            require_stored_config=True,
        )


async def test_db_config_rejects_internal_token_override(config_session_factory) -> None:
    await _store_overrides(
        config_session_factory,
        ('*', '*', {'internal_token': 'forged'}),
    )

    with pytest.raises(ValueError, match='secret field.*internal_token'):
        await load_distributed_config(
            config_session_factory,
            database_url='sqlite+aiosqlite:///bootstrap.db',
            role='scheduler',
        )


async def test_db_config_ignores_unknown_fields_for_forward_compatibility(
    config_session_factory,
    caplog,
) -> None:
    await _store_overrides(
        config_session_factory,
        ('scheduler', '*', {'future_scheduler_setting': True, 'scheduler_poll_interval': 2.0}),
    )

    with caplog.at_level('WARNING', logger='nexus.distributed.config.store'):
        config = await load_distributed_config(
            config_session_factory,
            database_url='sqlite+aiosqlite:///bootstrap.db',
            role='scheduler',
        )

    assert config.scheduler_poll_interval == 2.0
    assert 'future_scheduler_setting' in caplog.text


async def test_db_config_ignores_all_unknown_nested_fields(
    config_session_factory,
    caplog,
) -> None:
    await _store_overrides(
        config_session_factory,
        (
            'control-gateway',
            '*',
            {
                'run_trace_sinks': {
                    'tdsql': {
                        'kind': 'tdsql',
                        'sourceClass': 'nexus.runner.trace.tdsql::TDSQLTraceSource',
                        'config': {'db': 'trace_db'},
                        'futureMetadata': {'enabled': True},
                    }
                },
                'trace_database_access': {
                    'tdsql': {
                        'read_host': 'trace-db.internal',
                        'database': 'trace_db',
                        'maintenance_credential_ref': 'trace-maintenance',
                        'account_provider': {
                            'provider_type': 'mysql',
                            'future_provider_setting': 'NESTED_CONFIG_VALUE_SENTINEL',
                        },
                        'future_access_setting': 'NESTED_ACCESS_VALUE_SENTINEL',
                    }
                },
            },
        ),
    )

    with caplog.at_level('WARNING', logger='nexus.distributed.config.store'):
        config = await load_distributed_config(
            config_session_factory,
            database_url='sqlite+aiosqlite:///bootstrap.db',
            role='control-gateway',
        )

    access = config.trace_database_access['tdsql']
    assert access.read_host == 'trace-db.internal'
    assert access.account_provider.provider_type == 'mysql'
    assert config.run_trace_sinks['tdsql'].model_extra is None
    assert 'trace_database_access.tdsql.future_access_setting' in caplog.text
    assert 'trace_database_access.tdsql.account_provider.mysql.future_provider_setting' in caplog.text
    assert 'run_trace_sinks.tdsql.futureMetadata' in caplog.text
    assert 'NESTED_CONFIG_VALUE_SENTINEL' not in caplog.text
    assert 'NESTED_ACCESS_VALUE_SENTINEL' not in caplog.text


async def test_db_secrets_ignore_unknown_nested_fields_without_logging_values(
    config_session_factory,
    caplog,
) -> None:
    await _store_secret_overrides(
        config_session_factory,
        (
            'control-gateway',
            '*',
            {
                'future_secret_setting': 'TOP_LEVEL_SECRET_VALUE_SENTINEL',
                'trace_database_credentials': {
                    'trace-maintenance': {
                        'provider_type': 'mysql',
                        'username': 'trace-admin',
                        'password': 'trace-password',
                        'future_credential_setting': 'NESTED_SECRET_VALUE_SENTINEL',
                    }
                },
            },
        ),
    )

    with caplog.at_level('WARNING', logger='nexus.distributed.config.store'):
        config = await load_distributed_config(
            config_session_factory,
            database_url='sqlite+aiosqlite:///bootstrap.db',
            role='control-gateway',
        )

    credential = config.resolve_trace_database_credential('trace-maintenance')
    assert credential is not None
    assert credential.read() == ('trace-admin', 'trace-password')
    assert config.secrets.model_extra is None
    assert 'secrets.future_secret_setting' in caplog.text
    assert 'secrets.trace_database_credentials.trace-maintenance.mysql.future_credential_setting' in caplog.text
    assert 'TOP_LEVEL_SECRET_VALUE_SENTINEL' not in caplog.text
    assert 'NESTED_SECRET_VALUE_SENTINEL' not in caplog.text


async def test_db_config_does_not_ignore_invalid_known_nested_fields(config_session_factory, caplog) -> None:
    await _store_overrides(
        config_session_factory,
        (
            'control-gateway',
            '*',
            {
                'run_trace_sinks': {
                    'tdsql': {
                        'kind': 'tdsql',
                        'sourceClass': 'nexus.runner.trace.tdsql::TDSQLTraceSource',
                        'config': {'db': 'trace_db'},
                    }
                },
                'trace_database_access': {
                    'tdsql': {
                        'read_host': 'trace-db.internal',
                        'read_port': 0,
                        'database': 'trace_db',
                        'maintenance_credential_ref': 'trace-maintenance',
                        'future_access_setting': 'INVALID_CONFIG_VALUE_SENTINEL',
                    }
                },
            },
        ),
    )

    with caplog.at_level('WARNING', logger='nexus.distributed.config.store'):
        with pytest.raises(ValueError, match='trace_database_access.*read_port'):
            await load_distributed_config(
                config_session_factory,
                database_url='sqlite+aiosqlite:///bootstrap.db',
                role='control-gateway',
            )

    assert 'future_access_setting' not in caplog.text
    assert 'INVALID_CONFIG_VALUE_SENTINEL' not in caplog.text


async def test_startup_overrides_remain_strict_when_db_field_has_same_root(config_session_factory) -> None:
    await _store_overrides(
        config_session_factory,
        ('control-gateway', '*', {'trace_database_access': {}}),
    )

    with pytest.raises(ValueError, match='future_access_setting'):
        await load_distributed_config(
            config_session_factory,
            database_url='sqlite+aiosqlite:///bootstrap.db',
            role='control-gateway',
            overrides={
                'trace_database_access': {
                    'tdsql': {
                        'read_host': 'trace-db.internal',
                        'database': 'trace_db',
                        'maintenance_credential_ref': 'trace-maintenance',
                        'future_access_setting': True,
                    }
                }
            },
        )


async def test_initialize_config_uses_bootstrap_database_and_ready_schema(tmp_path) -> None:
    database_url = f'sqlite+aiosqlite:///{tmp_path / "distributed.db"}'
    await init_database(DistributedConfig(database_url=database_url))
    engine = create_async_engine(database_url)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    await _store_overrides(
        session_factory,
        ('scheduler', 'scheduler-1', {'scheduler_poll_interval': 0.5}),
    )
    await engine.dispose()

    config = await initialize_distributed_config(
        role='scheduler',
        identity='scheduler-1',
        overrides={'database_url': database_url},
    )

    assert config.database_url == database_url
    assert config.scheduler_poll_interval == 0.5


async def test_initialize_config_resolves_bootstrap_database_url_file(tmp_path) -> None:
    database_url = f'sqlite+aiosqlite:///{tmp_path / "distributed-from-file.db"}'
    database_url_file = tmp_path / 'database-url'
    database_url_file.write_text(database_url)
    await init_database(DistributedConfig(database_url=database_url))

    config = await initialize_distributed_config(
        role='scheduler',
        environment_values={'database_url_file': str(database_url_file)},
    )

    assert config.database_url == database_url
    assert config.database_url_file == str(database_url_file)


async def test_initialize_config_accepts_instance_manager_schema_floor(tmp_path) -> None:
    database_url = f'sqlite+aiosqlite:///{tmp_path / "manager-schema-floor.db"}'
    await init_database(DistributedConfig(database_url=database_url))
    engine = create_async_engine(database_url)
    try:
        async with engine.begin() as connection:
            await connection.exec_driver_sql("UPDATE alembic_version SET version_num = '0029_run_trace_backend_grants'")
    finally:
        await engine.dispose()

    config = await initialize_distributed_config(
        role='instance-manager',
        identity='manager-1',
        overrides={'database_url': database_url},
        minimum_schema_revision='0029_run_trace_backend_grants',
    )

    assert config.database_url == database_url


@pytest.mark.parametrize(
    ('actual', 'allow_newer', 'expected'),
    [
        ('0029_run_trace_backend_grants', False, True),
        ('0030_principal_access_grants', False, False),
        ('0028_batch_module_dependencies', True, False),
        ('0029_run_trace_backend_grants', True, True),
        ('0030_principal_access_grants', True, True),
        ('main', True, False),
    ],
)
def test_schema_revision_compatibility_uses_numeric_floor_for_newer_schema(
    actual: str,
    allow_newer: bool,
    expected: bool,
) -> None:
    assert (
        schema_revision_is_compatible(
            actual,
            expected='0029_run_trace_backend_grants',
            allow_newer=allow_newer,
        )
        is expected
    )


async def test_config_management_uses_cas_and_records_field_only_audit(config_session_factory) -> None:
    async with config_session_factory() as session:
        created = await set_config_override(
            session,
            role='scheduler',
            identity='*',
            values={'scheduler_poll_interval': 0.25},
            expected_version=0,
            actor='admin-1',
        )
        await session.commit()

    assert created.version == 1
    assert created.values == {'scheduler_poll_interval': 0.25}

    async with config_session_factory() as session:
        with pytest.raises(ConfigVersionConflictError, match='expected 0, found 1'):
            await set_config_override(
                session,
                role='scheduler',
                identity='*',
                values={'scheduler_poll_interval': 0.5},
                expected_version=0,
                actor='admin-1',
            )

        updated = await set_config_override(
            session,
            role='scheduler',
            identity='*',
            values={'scheduler_poll_interval': 0.5},
            expected_version=1,
            actor='admin-1',
        )
        await session.commit()

    assert updated.version == 2
    async with config_session_factory() as session:
        audit_rows = list((await session.scalars(select(AuditRow).order_by(AuditRow.created_at))).all())

    assert len(audit_rows) == 2
    assert audit_rows[-1].operation == 'distributed_config.set'
    assert audit_rows[-1].metadata_json == {'fields': ['scheduler_poll_interval'], 'version': 2}
    assert '0.5' not in str(audit_rows[-1].metadata_json)


async def test_secret_management_validates_payload_and_never_lists_values(config_session_factory) -> None:
    async with config_session_factory() as session:
        created = await set_config_override(
            session,
            role='reconciler',
            identity='reconciler-1',
            values={
                'internal_token': 'top-secret',
                'provider_credentials': {
                    'tkex/dev': {'provider_type': 'tkex', 'token': 'provider-secret'},
                },
                'run_trace_credentials': {'trajectory-db': 'trajectory-secret'},
                'zhiyan_log_credentials': {'nexus-zhiyan-openapi': 'zhiyan-secret'},
            },
            expected_version=0,
            actor='admin-1',
            secret=True,
        )
        await session.commit()

    assert created.values == {}
    assert created.fields == (
        'internal_token',
        'provider_credentials',
        'run_trace_credentials',
        'zhiyan_log_credentials',
    )

    async with config_session_factory() as session:
        summaries = await list_secret_override_summaries(session, role='reconciler')
        stored = await get_config_override(
            session,
            role='reconciler',
            identity='reconciler-1',
            secret=True,
            include_secret_values=True,
        )

    assert summaries == [created]
    assert stored is not None
    assert stored.values['internal_token'] == 'top-secret'
    assert 'top-secret' not in repr(created)
    assert 'provider-secret' not in repr(created)
    assert 'trajectory-secret' not in repr(created)
    assert 'zhiyan-secret' not in repr(created)


@pytest.mark.parametrize(
    'values',
    [
        {
            'provider_credentials': {
                'tkex/dev': {'provider_type': 'unsupported', 'token': 'SECRET_ERROR_SENTINEL'},
            }
        },
        {
            'provider_credentials': {
                'tkex/dev': {
                    'provider_type': 'tkex',
                    'token': 'valid-token',
                    'unexpected': 'SECRET_ERROR_SENTINEL',
                },
            }
        },
    ],
)
async def test_secret_validation_errors_hide_input_values(config_session_factory, values) -> None:
    async with config_session_factory() as session:
        with pytest.raises(ValueError, match='provider_credentials') as caught:
            await set_config_override(
                session,
                role='reconciler',
                identity='*',
                values=values,
                expected_version=0,
                actor='admin-1',
                secret=True,
            )

    assert 'SECRET_ERROR_SENTINEL' not in str(caught.value)
    assert 'SECRET_ERROR_SENTINEL' not in repr(caught.value)


async def test_startup_secret_validation_errors_hide_stored_values(config_session_factory) -> None:
    await _store_secret_overrides(
        config_session_factory,
        (
            '*',
            '*',
            {
                'provider_credentials': {
                    'tkex/dev': {'provider_type': 'unsupported', 'token': 'LEAKME'},
                }
            },
        ),
    )

    with pytest.raises(ValueError, match='provider_credentials') as caught:
        await load_distributed_config(
            config_session_factory,
            database_url='sqlite+aiosqlite:///bootstrap.db',
            role='reconciler',
        )

    assert 'LEAKME' not in str(caught.value)
    assert 'LEAKME' not in repr(caught.value)


async def test_config_management_rejects_wrong_table_fields(config_session_factory) -> None:
    async with config_session_factory() as session:
        with pytest.raises(ValueError, match='secret field.*internal_token'):
            await set_config_override(
                session,
                role='scheduler',
                identity='*',
                values={'internal_token': 'forged'},
                expected_version=0,
                actor='admin-1',
            )
        with pytest.raises(ValueError, match='unknown distributed secret field.*scheduler_poll_interval'):
            await set_config_override(
                session,
                role='scheduler',
                identity='*',
                values={'scheduler_poll_interval': 1.0},
                expected_version=0,
                actor='admin-1',
                secret=True,
            )
        with pytest.raises(ValueError, match='unknown distributed config field.*future_scheduler_setting'):
            await set_config_override(
                session,
                role='scheduler',
                identity='*',
                values={'future_scheduler_setting': True},
                expected_version=0,
                actor='admin-1',
            )


async def test_config_management_rejects_unknown_nested_config_fields(config_session_factory) -> None:
    async with config_session_factory() as session:
        with pytest.raises(ValueError, match='future_access_setting'):
            await set_config_override(
                session,
                role='control-gateway',
                identity='*',
                values={
                    'trace_database_access': {
                        'tdsql': {
                            'read_host': 'trace-db.internal',
                            'database': 'trace_db',
                            'maintenance_credential_ref': 'trace-maintenance',
                            'future_access_setting': True,
                        }
                    }
                },
                expected_version=0,
                actor='admin-1',
            )


async def test_config_management_persists_trace_backend_extensions_unmerged(config_session_factory) -> None:
    """A save must not promote extend_* entries into the fields older builds must read."""
    values = {
        'run_trace_sinks': {
            'tdsql': {
                'kind': 'tdsql',
                'sourceClass': 'nexus.runner.trace.tdsql::TDSQLTraceSource',
                'config': {'db': 'trace_db'},
            }
        },
        'extend_run_trace_sinks': {
            'tdsql-boundless': {
                'kind': 'tdsql',
                'sinkClass': 'nexus.runner.trace.tdsql::TDSQLBoundlessTraceSink',
                'sourceClass': 'nexus.runner.trace.tdsql::TDSQLBoundlessTraceSource',
                'config': {'db': 'trace_db'},
            }
        },
        'trace_database_access': {
            'tdsql': {
                'read_host': 'trace-db.internal',
                'database': 'trace_db',
                'maintenance_credential_ref': 'trace-maintenance',
            }
        },
        'extend_trace_database_access': {
            'tdsql-boundless': {
                'read_host': 'boundless-db.internal',
                'database': 'trace_db',
                'maintenance_credential_ref': 'boundless-maintenance',
                'account_provider': {
                    'provider_type': 'tencent_cloud_tdmysql',
                    'region': 'ap-nanjing',
                    'instance_id': 'tdsql-49qrr9xo6y',
                },
            }
        },
    }

    async with config_session_factory() as session:
        record = await set_config_override(
            session,
            role='*',
            identity='*',
            values=values,
            expected_version=0,
            actor='admin-1',
        )
        await session.commit()

    assert sorted(record.values['run_trace_sinks']) == ['tdsql']
    assert sorted(record.values['trace_database_access']) == ['tdsql']

    async with config_session_factory() as session:
        stored = await session.get(DistributedConfigOverrideRow, {'role': '*', 'identity': '*'})

    assert stored is not None
    assert sorted(stored.config_json['run_trace_sinks']) == ['tdsql']
    assert sorted(stored.config_json['extend_run_trace_sinks']) == ['tdsql-boundless']
    assert sorted(stored.config_json['trace_database_access']) == ['tdsql']
    assert sorted(stored.config_json['extend_trace_database_access']) == ['tdsql-boundless']

    # The merge is still what running services see.
    config = await load_distributed_config(
        config_session_factory,
        database_url='sqlite+aiosqlite:///bootstrap.db',
        role='control-gateway',
    )
    assert sorted(config.run_trace_sinks) == ['tdsql', 'tdsql-boundless']
    assert sorted(config.trace_database_access) == ['tdsql', 'tdsql-boundless']


async def test_config_management_rejects_unknown_nested_secret_fields(config_session_factory) -> None:
    async with config_session_factory() as session:
        with pytest.raises(ValueError, match='future_credential_setting'):
            await set_config_override(
                session,
                role='control-gateway',
                identity='*',
                values={
                    'trace_database_credentials': {
                        'trace-maintenance': {
                            'provider_type': 'mysql',
                            'username': 'trace-admin',
                            'password': 'trace-password',
                            'future_credential_setting': True,
                        }
                    }
                },
                expected_version=0,
                actor='admin-1',
                secret=True,
            )


async def test_config_management_delete_uses_expected_version(config_session_factory) -> None:
    async with config_session_factory() as session:
        await set_config_override(
            session,
            role='scheduler',
            identity='scheduler-1',
            values={'scheduler_poll_interval': 1.0},
            expected_version=0,
            actor='admin-1',
        )
        await session.commit()

    async with config_session_factory() as session:
        with pytest.raises(ConfigVersionConflictError, match='expected 2, found 1'):
            await delete_config_override(
                session,
                role='scheduler',
                identity='scheduler-1',
                expected_version=2,
                actor='admin-1',
            )

        assert await delete_config_override(
            session,
            role='scheduler',
            identity='scheduler-1',
            expected_version=1,
            actor='admin-1',
        )
        await session.commit()

    async with config_session_factory() as session:
        assert await get_config_override(session, role='scheduler', identity='scheduler-1') is None
