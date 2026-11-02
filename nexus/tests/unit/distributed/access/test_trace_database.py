from __future__ import annotations

import asyncio

from datetime import timedelta
from typing import Any

import pytest

from sqlalchemy.dialects import mysql
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from nexus.distributed.access.trace_database import (
    TraceDatabaseAccountManager,
    TraceDatabaseAccountProvider,
)
from nexus.distributed.access.trace_database_providers.mysql import MySQLTraceDatabaseAccountProvider
from nexus.distributed.config import DistributedConfig
from nexus.distributed.errors import DomainConflictError, ResourceNotFoundError
from nexus.distributed.persistence.orm import TraceDatabaseAccountRow, create_schema
from nexus.distributed.repositories.access import AccessRepository
from nexus.distributed.repositories.environment import EnvironmentRepository
from nexus.distributed.repositories.run_trace_backend import (
    RunTraceBackendGrantInput,
    RunTraceBackendGrantRepository,
)
from nexus.distributed.repositories.trace_database import TraceDatabaseAccountRepository
from nexus.distributed.utils.clock import utcnow


@pytest.fixture
async def session_factory():
    engine = create_async_engine('sqlite+aiosqlite:///:memory:')
    await create_schema(engine)
    try:
        yield async_sessionmaker(engine, expire_on_commit=False)
    finally:
        await engine.dispose()


class FakeProvider(TraceDatabaseAccountProvider):
    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self.block = asyncio.Event()
        self.block.set()
        self.started = asyncio.Event()

    async def set_readonly_account(self, *, username: str, password: str, database: str, table_name: str) -> None:
        self.started.set()
        await self.block.wait()
        self.calls.append(('set', username, password, database, table_name))

    async def disable_account(self, *, username: str) -> None:
        self.calls.append(('disable', username))

    async def restore_account(self, *, username: str, database: str, table_name: str) -> None:
        self.calls.append(('restore', username))

    async def reconcile_readonly_account(self, *, username: str, database: str, table_name: str) -> None:
        self.calls.append(('reconcile', username, database, table_name))


def _config() -> DistributedConfig:
    return DistributedConfig(
        database_url='sqlite+aiosqlite:///:memory:',
        run_trace_sinks={
            'tdsql': {
                'kind': 'tdsql',
                'userSelectable': True,
                'sinkClass': 'nexus.runner.trace.tdsql::TDSQLTraceSink',
                'sourceClass': 'nexus.runner.trace.tdsql::TDSQLTraceSource',
                'config': {'host': 'write.example', 'port': 3306, 'db': 'nexus_traces', 'user': 'writer'},
            }
        },
        trace_database_access={
            'tdsql': {
                'read_host': 'read.example',
                'read_port': 3307,
                'database': 'nexus_traces',
                'maintenance_credential_ref': 'tdsql-maintenance',
            }
        },
        secrets={
            'trace_database_credentials': {
                'tdsql-maintenance': {'username': 'account-admin', 'password': 'maintenance-secret'}
            }
        },
    )


async def _grant_environment(
    session_factory: async_sessionmaker,
    principal_id: str,
    *,
    grant_trace_backend: bool = True,
) -> None:
    now = utcnow()
    async with session_factory() as session:
        access = AccessRepository(session)
        await access.ensure_user(
            principal_id=principal_id,
            issuer='tai',
            tenant='default',
            subject=principal_id,
            roles={'user'},
            display_name=principal_id,
            actor='test',
            now=now,
        )
        environments = EnvironmentRepository(session)
        await environments.create_environment(
            environment_id='env-a',
            name='Environment A',
            allocation_type='predefined',
            provider_type='predefined',
            provider_config_ref=None,
            resource_limits={},
            default_timeouts={},
            status='enabled',
            actor='test',
            now=now,
        )
        await environments.upsert_grant(
            principal_id=principal_id,
            environment_id='env-a',
            limits={},
            scopes={'submission:batch'},
            enabled=True,
            actor='test',
            now=now,
        )
        if grant_trace_backend:
            await RunTraceBackendGrantRepository(session).replace_for_principal(
                principal_id,
                [RunTraceBackendGrantInput('tdsql')],
                actor='test',
                now=now,
            )
        await session.commit()


async def test_trace_database_access_requires_backend_grant(session_factory) -> None:
    await _grant_environment(session_factory, 'principal-owner-a', grant_trace_backend=False)
    manager = TraceDatabaseAccountManager(session_factory=session_factory, config=_config())

    assert await manager.list_for_principal('principal-owner-a') == []
    with pytest.raises(ResourceNotFoundError, match='trace_database_backend_not_found'):
        await manager.create_or_reset(
            owner_principal_id='principal-owner-a',
            environment_id='env-a',
            backend_name='tdsql',
            operation='create',
            actor_principal_id='principal-owner-a',
            now=utcnow(),
        )


async def test_trace_database_backend_access_locks_principal_before_grant() -> None:
    statements = []

    class RecordingSession:
        async def scalar(self, statement):
            statements.append(statement)
            return 'found'

    await TraceDatabaseAccountRepository(RecordingSession()).require_backend_access(  # type: ignore[arg-type]
        owner_principal_id='principal-owner-a',
        backend_name='tdsql',
    )

    sql = [str(statement.compile(dialect=mysql.dialect())) for statement in statements]
    assert len(sql) == 2
    assert 'FROM distributed_principals' in sql[0]
    assert sql[0].rstrip().endswith('FOR UPDATE')
    assert 'FROM distributed_run_trace_backend_grants' in sql[1]
    assert sql[1].rstrip().endswith('FOR UPDATE')


async def test_trace_database_access_checks_environment_before_resolving_backend(session_factory) -> None:
    await _grant_environment(session_factory, 'principal-owner-a')
    manager = TraceDatabaseAccountManager(session_factory=session_factory, config=_config())
    errors = []

    for backend_name in ('tdsql', 'missing'):
        with pytest.raises(ResourceNotFoundError) as captured:
            await manager.create_or_reset(
                owner_principal_id='principal-owner-a',
                environment_id='missing-environment',
                backend_name=backend_name,
                operation='create',
                actor_principal_id='principal-owner-a',
                now=utcnow(),
            )
        errors.append(captured.value.code)

    assert errors == ['environment_not_found', 'environment_not_found']


async def test_account_create_and_reset_returns_password_once_without_persisting_it(session_factory) -> None:
    await _grant_environment(session_factory, 'principal-owner-a')
    provider = FakeProvider()
    manager = TraceDatabaseAccountManager(
        session_factory=session_factory,
        config=_config(),
        provider_factory=lambda _target, _credential: provider,
    )

    created = await manager.create_or_reset(
        owner_principal_id='principal-owner-a',
        environment_id='env-a',
        backend_name='tdsql',
        operation='create',
        actor_principal_id='principal-owner-a',
        now=utcnow(),
    )

    assert created.account.backend_name == 'tdsql'
    assert created.account.read_host == 'read.example'
    assert created.account.read_port == 3307
    assert created.account.database == 'nexus_traces'
    assert created.account.access_scope == 'database'
    assert created.account.status == 'ready'
    assert created.account.trace_table_name == 'trajectory_span_events_pr_3190e1c7731fff7995861c3221cec9a6'
    assert created.account.batch_table_name == created.account.trace_table_name
    assert created.password
    assert provider.calls[0] == (
        'set',
        created.account.username,
        created.password,
        'nexus_traces',
        created.account.trace_table_name,
    )
    assert len(created.password) == 32
    assert any(character.islower() for character in created.password)
    assert any(character.isupper() for character in created.password)
    assert any(character.isdigit() for character in created.password)
    assert any(not character.isalnum() for character in created.password)

    async with session_factory() as session:
        row = await session.get(TraceDatabaseAccountRow, ('principal-owner-a', 'tdsql'))
        assert row is not None
        assert row.credential_version == 1
        assert not hasattr(row, 'password')
        assert created.password not in repr(row)

    with pytest.raises(DomainConflictError, match='trace_database_account_already_exists'):
        await manager.create_or_reset(
            owner_principal_id='principal-owner-a',
            environment_id='env-a',
            backend_name='tdsql',
            operation='create',
            actor_principal_id='principal-owner-a',
            now=utcnow(),
        )

    reset = await manager.create_or_reset(
        owner_principal_id='principal-owner-a',
        environment_id='env-a',
        backend_name='tdsql',
        operation='reset',
        actor_principal_id='principal-owner-a',
        now=utcnow(),
    )

    assert reset.password != created.password
    assert reset.account.username == created.account.username
    async with session_factory() as session:
        row = await session.get(TraceDatabaseAccountRow, ('principal-owner-a', 'tdsql'))
        assert row is not None
        assert row.credential_version == 2


async def test_account_operations_are_serialized_while_external_ddl_is_running(session_factory) -> None:
    await _grant_environment(session_factory, 'principal-owner-a')
    provider = FakeProvider()
    provider.block.clear()
    manager = TraceDatabaseAccountManager(
        session_factory=session_factory,
        config=_config(),
        provider_factory=lambda _target, _credential: provider,
        operation_ttl=timedelta(milliseconds=150),
    )

    creating = asyncio.create_task(
        manager.create_or_reset(
            owner_principal_id='principal-owner-a',
            environment_id='env-a',
            backend_name='tdsql',
            operation='create',
            actor_principal_id='principal-owner-a',
            now=utcnow(),
        )
    )
    await provider.started.wait()

    async with session_factory() as session:
        row = await session.get(TraceDatabaseAccountRow, ('principal-owner-a', 'tdsql'))
        assert row is not None
        initial_expiry = row.operation_expires_at

    await asyncio.sleep(0.3)

    async with session_factory() as session:
        row = await session.get(TraceDatabaseAccountRow, ('principal-owner-a', 'tdsql'))
        assert row is not None
        assert row.operation_expires_at is not None
        assert initial_expiry is not None
        assert row.operation_expires_at > initial_expiry

    with pytest.raises(DomainConflictError, match='trace_database_account_operation_in_progress'):
        await manager.create_or_reset(
            owner_principal_id='principal-owner-a',
            environment_id='env-a',
            backend_name='tdsql',
            operation='reset',
            actor_principal_id='principal-owner-a',
            now=utcnow(),
        )

    provider.block.set()
    await creating


async def test_admin_can_disable_restore_and_reconcile_without_returning_password(session_factory) -> None:
    await _grant_environment(session_factory, 'principal-owner-a')
    provider = FakeProvider()
    manager = TraceDatabaseAccountManager(
        session_factory=session_factory,
        config=_config(),
        provider_factory=lambda _target, _credential: provider,
    )
    issued = await manager.create_or_reset(
        owner_principal_id='principal-owner-a',
        environment_id='env-a',
        backend_name='tdsql',
        operation='create',
        actor_principal_id='principal-owner-a',
        now=utcnow(),
    )

    disabled = await manager.admin_operation(
        owner_principal_id='principal-owner-a',
        backend_name='tdsql',
        operation='disable',
        actor_principal_id='admin-a',
        now=utcnow(),
    )
    reconciled_disabled = await manager.admin_operation(
        owner_principal_id='principal-owner-a',
        backend_name='tdsql',
        operation='reconcile',
        actor_principal_id='admin-a',
        now=utcnow(),
    )
    restored = await manager.admin_operation(
        owner_principal_id='principal-owner-a',
        backend_name='tdsql',
        operation='restore',
        actor_principal_id='admin-a',
        now=utcnow(),
    )

    assert disabled.status == 'disabled'
    assert reconciled_disabled.status == 'disabled'
    assert restored.status == 'ready'
    admin_calls = provider.calls[1:]
    assert admin_calls == [
        ('disable', issued.account.username),
        ('reconcile', issued.account.username, 'nexus_traces', issued.account.trace_table_name),
        ('disable', issued.account.username),
        ('restore', issued.account.username),
    ]

    with pytest.raises(DomainConflictError, match='trace_database_account_not_disabled'):
        await manager.admin_operation(
            owner_principal_id='principal-owner-a',
            backend_name='tdsql',
            operation='restore',
            actor_principal_id='admin-a',
            now=utcnow(),
        )

    await manager.admin_operation(
        owner_principal_id='principal-owner-a',
        backend_name='tdsql',
        operation='disable',
        actor_principal_id='admin-a',
        now=utcnow(),
    )
    with pytest.raises(DomainConflictError, match='trace_database_account_disabled'):
        await manager.create_or_reset(
            owner_principal_id='principal-owner-a',
            environment_id='env-a',
            backend_name='tdsql',
            operation='reset',
            actor_principal_id='principal-owner-a',
            now=utcnow(),
        )
    admin_reset = await manager.admin_reset(
        owner_principal_id='principal-owner-a',
        backend_name='tdsql',
        actor_principal_id='admin-a',
        now=utcnow(),
    )
    assert admin_reset.account.status == 'ready'
    assert admin_reset.password


async def test_admin_cannot_reconcile_account_after_backend_grant_is_revoked(session_factory) -> None:
    await _grant_environment(session_factory, 'principal-owner-a')
    provider = FakeProvider()
    manager = TraceDatabaseAccountManager(
        session_factory=session_factory,
        config=_config(),
        provider_factory=lambda _target, _credential: provider,
    )
    await manager.create_or_reset(
        owner_principal_id='principal-owner-a',
        environment_id='env-a',
        backend_name='tdsql',
        operation='create',
        actor_principal_id='principal-owner-a',
        now=utcnow(),
    )
    await manager.admin_operation(
        owner_principal_id='principal-owner-a',
        backend_name='tdsql',
        operation='disable',
        actor_principal_id='admin-a',
        now=utcnow(),
    )
    async with session_factory() as session:
        await RunTraceBackendGrantRepository(session).replace_for_principal(
            'principal-owner-a',
            [],
            actor='admin-a',
            now=utcnow(),
        )
        await session.commit()

    with pytest.raises(ResourceNotFoundError, match='trace_database_backend_not_found'):
        await manager.admin_operation(
            owner_principal_id='principal-owner-a',
            backend_name='tdsql',
            operation='reconcile',
            actor_principal_id='admin-a',
            now=utcnow(),
        )


class _Cursor:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...] | None]] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def execute(self, sql: str, params: tuple[Any, ...] | None = None) -> None:
        self.calls.append((sql, params))


class _Connection:
    def __init__(self) -> None:
        self.cursor_instance = _Cursor()
        self.commits = 0
        self.rollbacks = 0
        self.closed = False

    def cursor(self) -> _Cursor:
        return self.cursor_instance

    async def commit(self) -> None:
        self.commits += 1

    async def rollback(self) -> None:
        self.rollbacks += 1

    def close(self) -> None:
        self.closed = True


async def test_mysql_provider_reconciles_table_select_only(monkeypatch) -> None:
    connection = _Connection()

    async def connect(**_kwargs):
        return connection

    monkeypatch.setattr('nexus.runner.trace.tdsql._driver.connect', connect)
    provider = MySQLTraceDatabaseAccountProvider(
        host='write.example',
        port=3306,
        database='nexus_traces',
        username='account-admin',
        password='maintenance-secret',
    )

    await provider.set_readonly_account(
        username='nxs_ro_0123456789abcdef01234567',
        password='generated-secret',
        database='nexus_traces',
        table_name='trajectory_span_events_pr_0123456789abcdef0123456789abcdef',
    )

    statements = [sql for sql, _params in connection.cursor_instance.calls]
    assert any(sql.startswith('CREATE USER IF NOT EXISTS') for sql in statements)
    assert any(sql.startswith('ALTER USER') and 'IDENTIFIED BY' in sql for sql in statements)
    assert "REVOKE ALL PRIVILEGES, GRANT OPTION FROM 'nxs_ro_0123456789abcdef01234567'@'%'" in statements
    assert (
        "GRANT SELECT ON `nexus_traces`.`trajectory_span_events_pr_0123456789abcdef0123456789abcdef` TO 'nxs_ro_0123456789abcdef01234567'@'%'"
        in statements
    )
    assert all('nexus_control' not in statement for statement in statements)
    assert connection.commits == 1
    assert connection.closed is True
