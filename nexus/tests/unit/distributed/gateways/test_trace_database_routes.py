from __future__ import annotations

from typing import Any

import pytest

from aiohttp.test_utils import TestClient, TestServer
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from nexus.distributed.access import trace_database as trace_database_module
from nexus.distributed.access.trace_database import TraceDatabaseAccountProvider
from nexus.distributed.config import DistributedConfig
from nexus.distributed.errors import ResourceNotFoundError
from nexus.distributed.gateways.control.app import create_control_app
from nexus.distributed.persistence.orm import create_schema
from nexus.distributed.repositories.access import AccessRepository
from nexus.distributed.repositories.environment import EnvironmentRepository
from nexus.distributed.repositories.run_trace_backend import (
    RunTraceBackendGrantInput,
    RunTraceBackendGrantRepository,
)
from nexus.distributed.utils.clock import utcnow


@pytest.fixture
async def session_factory():
    engine = create_async_engine('sqlite+aiosqlite:///:memory:')
    await create_schema(engine)
    try:
        yield async_sessionmaker(engine, expire_on_commit=False)
    finally:
        await engine.dispose()


class FakeTraceDatabaseAccountProvider(TraceDatabaseAccountProvider):
    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self.fail_disable = False

    async def set_readonly_account(self, *, username: str, password: str, database: str, table_name: str) -> None:
        self.calls.append(('set', username, password, database))

    async def disable_account(self, *, username: str) -> None:
        self.calls.append(('disable', username))
        if self.fail_disable:
            raise RuntimeError('injected trace database disable failure')

    async def restore_account(self, *, username: str, database: str, table_name: str) -> None:
        self.calls.append(('restore', username))

    async def reconcile_readonly_account(self, *, username: str, database: str, table_name: str) -> None:
        self.calls.append(('reconcile', username, database))


def _config() -> DistributedConfig:
    return DistributedConfig(
        auth_mode='required',
        secrets={
            'internal_token': 'test-internal-token',
            'trace_database_credentials': {
                'tdsql-maintenance': {'username': 'account-admin', 'password': 'maintenance-secret'}
            },
        },
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
    )


async def _seed_access(
    session_factory: async_sessionmaker,
    *,
    principal_id: str,
    roles: set[str],
    grant_environment: bool,
) -> None:
    now = utcnow()
    async with session_factory() as session:
        access = AccessRepository(session)
        await access.ensure_user(
            principal_id=principal_id,
            issuer='tai',
            tenant='default',
            subject=principal_id,
            roles=roles,
            display_name=principal_id,
            actor='test',
            now=now,
        )
        await access.ensure_system_token(
            owner_principal_id=principal_id,
            created_by='test',
            now=now,
        )
        if grant_environment:
            environments = EnvironmentRepository(session)
            try:
                await environments.get_environment('env-a')
            except ResourceNotFoundError:
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
            await RunTraceBackendGrantRepository(session).replace_for_principal(
                principal_id,
                [RunTraceBackendGrantInput('tdsql')],
                actor='test',
                now=now,
            )
        await session.commit()


def _identity_headers(principal_id: str) -> dict[str, str]:
    return {
        'X-Nexus-Internal-Token': 'test-internal-token',
        'X-Nexus-Delegated-Principal-Id': principal_id,
    }


async def test_user_creates_and_resets_own_trace_database_credentials_once(
    session_factory,
    monkeypatch,
) -> None:
    await _seed_access(session_factory, principal_id='principal-owner-a', roles={'user'}, grant_environment=True)
    provider = FakeTraceDatabaseAccountProvider()
    monkeypatch.setattr(
        trace_database_module,
        '_create_provider',
        lambda _target, _credential: provider,
    )
    app = create_control_app(session_factory=session_factory, config=_config())
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        headers = _identity_headers('principal-owner-a')
        response = await client.get('/v1/my-environments', headers=headers)
        assert response.status == 200
        payload = await response.json()
        assert payload['trace_database_accounts'] == [
            {
                'owner_principal_id': 'principal-owner-a',
                'backend_name': 'tdsql',
                'read_host': 'read.example',
                'read_port': 3307,
                'database': 'nexus_traces',
                'username': None,
                'status': 'not_created',
                'access_scope': 'database',
                'trace_table_name': 'trajectory_span_events_pr_3190e1c7731fff7995861c3221cec9a6',
                'batch_table_name': 'trajectory_span_events_pr_3190e1c7731fff7995861c3221cec9a6',
                'credential_version': 0,
                'last_error_code': None,
                'updated_at': None,
            }
        ]

        response = await client.post(
            '/v1/my-environments/env-a/trace-database-credentials',
            headers=headers,
            json={'backend_name': 'tdsql', 'operation': 'create'},
        )
        assert response.status == 200
        assert response.headers['Cache-Control'] == 'no-store'
        created = await response.json()
        first_password = created.pop('password')
        assert first_password
        assert created['status'] == 'ready'

        response = await client.get('/v1/my-environments', headers=headers)
        account = (await response.json())['trace_database_accounts'][0]
        assert account['username'] == created['username']
        assert account['credential_version'] == 1
        assert 'password' not in account

        response = await client.post(
            '/v1/my-environments/env-a/trace-database-credentials',
            headers=headers,
            json={'backend_name': 'tdsql', 'operation': 'reset'},
        )
        assert response.status == 200
        reset = await response.json()
        assert response.headers['Cache-Control'] == 'no-store'
        assert reset['password'] != first_password
        assert reset['username'] == created['username']
        assert reset['credential_version'] == 2

        response = await client.get('/v1/access/trace-database-accounts', headers=headers)
        assert response.status == 403
    finally:
        await client.close()


async def test_admin_manages_trace_database_accounts_and_only_reset_returns_password(
    session_factory,
    monkeypatch,
) -> None:
    await _seed_access(session_factory, principal_id='principal-owner-a', roles={'user'}, grant_environment=True)
    await _seed_access(
        session_factory, principal_id='principal-admin', roles={'user', 'admin'}, grant_environment=False
    )
    provider = FakeTraceDatabaseAccountProvider()
    monkeypatch.setattr(
        trace_database_module,
        '_create_provider',
        lambda _target, _credential: provider,
    )
    app = create_control_app(session_factory=session_factory, config=_config())
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        owner_headers = _identity_headers('principal-owner-a')
        admin_headers = _identity_headers('principal-admin')
        response = await client.post(
            '/v1/my-environments/env-a/trace-database-credentials',
            headers=owner_headers,
            json={'backend_name': 'tdsql', 'operation': 'create'},
        )
        assert response.status == 200

        response = await client.get('/v1/access/trace-database-accounts', headers=admin_headers)
        assert response.status == 200
        items = (await response.json())['items']
        assert len(items) == 1
        assert items[0]['owner_principal_id'] == 'principal-owner-a'
        assert 'password' not in items[0]

        for operation, expected_status in (
            ('disable', 'disabled'),
            ('restore', 'ready'),
            ('reconcile', 'ready'),
        ):
            response = await client.post(
                f'/v1/access/trace-database-accounts/principal-owner-a/tdsql/{operation}',
                headers=admin_headers,
            )
            assert response.status == 200
            payload = await response.json()
            assert payload['status'] == expected_status
            assert 'password' not in payload

        response = await client.post(
            '/v1/access/trace-database-accounts/principal-owner-a/tdsql/reset',
            headers=admin_headers,
        )
        assert response.status == 200
        payload = await response.json()
        assert response.headers['Cache-Control'] == 'no-store'
        assert payload['password']
        assert payload['credential_version'] == 2

        response = await client.put(
            '/v1/environments/env-a/grants/principal-owner-a',
            headers=admin_headers,
            json={'enabled': False},
        )
        assert response.status == 200
        assert (await response.json())['enabled'] is False

        response = await client.get('/v1/access/trace-database-accounts', headers=admin_headers)
        account = (await response.json())['items'][0]
        assert account['status'] == 'disabled'
        assert provider.calls[-1][0] == 'disable'

        response = await client.get('/v1/my-environments', headers=owner_headers)
        assert (await response.json())['trace_database_accounts'] == []
    finally:
        await client.close()


@pytest.mark.parametrize(
    ('method', 'path', 'json'),
    [
        ('patch', '/v1/environments/env-a', {'status': 'disabled'}),
        ('post', '/v1/environments/env-a/disable', None),
    ],
)
async def test_disabling_environment_disables_accounts_before_access_is_revoked(
    session_factory,
    monkeypatch,
    method: str,
    path: str,
    json: dict[str, str] | None,
) -> None:
    await _seed_access(session_factory, principal_id='principal-owner-a', roles={'user'}, grant_environment=True)
    await _seed_access(
        session_factory, principal_id='principal-admin', roles={'user', 'admin'}, grant_environment=False
    )
    provider = FakeTraceDatabaseAccountProvider()
    monkeypatch.setattr(trace_database_module, '_create_provider', lambda _target, _credential: provider)
    app = create_control_app(session_factory=session_factory, config=_config())
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        owner_headers = _identity_headers('principal-owner-a')
        admin_headers = _identity_headers('principal-admin')
        response = await client.post(
            '/v1/my-environments/env-a/trace-database-credentials',
            headers=owner_headers,
            json={'backend_name': 'tdsql', 'operation': 'create'},
        )
        assert response.status == 200

        response = await getattr(client, method)(path, headers=admin_headers, json=json)
        assert response.status == 200
        assert (await response.json())['status'] == 'disabled'
        assert provider.calls[-1][0] == 'disable'

        response = await client.get('/v1/access/trace-database-accounts', headers=admin_headers)
        assert (await response.json())['items'][0]['status'] == 'disabled'
    finally:
        await client.close()


async def test_environment_disable_fails_closed_when_account_cannot_be_disabled(
    session_factory,
    monkeypatch,
) -> None:
    await _seed_access(session_factory, principal_id='principal-owner-a', roles={'user'}, grant_environment=True)
    await _seed_access(
        session_factory, principal_id='principal-admin', roles={'user', 'admin'}, grant_environment=False
    )
    provider = FakeTraceDatabaseAccountProvider()
    monkeypatch.setattr(trace_database_module, '_create_provider', lambda _target, _credential: provider)
    app = create_control_app(session_factory=session_factory, config=_config())
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        owner_headers = _identity_headers('principal-owner-a')
        admin_headers = _identity_headers('principal-admin')
        response = await client.post(
            '/v1/my-environments/env-a/trace-database-credentials',
            headers=owner_headers,
            json={'backend_name': 'tdsql', 'operation': 'create'},
        )
        assert response.status == 200

        provider.fail_disable = True
        response = await client.post('/v1/environments/env-a/disable', headers=admin_headers)
        assert response.status == 503

        response = await client.get('/v1/environments/env-a', headers=admin_headers)
        assert response.status == 200
        assert (await response.json())['status'] == 'enabled'
    finally:
        await client.close()
