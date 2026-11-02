from __future__ import annotations

import importlib
import importlib.util

from unittest.mock import AsyncMock

import pytest

from aiohttp.test_utils import TestClient, TestServer
from sqlalchemy import inspect, select
from sqlalchemy.dialects import mysql
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from nexus.distributed.access.run_trace_backends import RunTraceBackendAccessPolicy
from nexus.distributed.config import DistributedConfig, run_trace_backend_catalog
from nexus.distributed.errors import DomainConflictError, DomainValidationError
from nexus.distributed.gateways.control.app import create_control_app
from nexus.distributed.persistence.orm import AuditRow, TraceDatabaseAccountRow, create_schema
from nexus.distributed.repositories.access import AccessRepository
from nexus.distributed.repositories.run_trace_backend import (
    RunTraceBackendGrantInput,
    RunTraceBackendGrantRepository,
)
from nexus.distributed.utils.clock import utcnow
from nexus.runner.trace import principal_trace_scope_id


@pytest.fixture
async def session_factory() -> async_sessionmaker:
    engine = create_async_engine('sqlite+aiosqlite:///:memory:')
    await create_schema(engine)
    try:
        yield async_sessionmaker(engine, expire_on_commit=False)
    finally:
        await engine.dispose()


async def _seed_principal(session_factory: async_sessionmaker, principal_id: str = 'principal-a') -> None:
    async with session_factory() as session:
        repository = AccessRepository(session)
        await repository.ensure_user(
            principal_id=principal_id,
            issuer='test',
            tenant='default',
            subject=principal_id,
            roles={'user'},
            display_name=principal_id,
            actor='test',
            now=utcnow(),
        )
        await repository.ensure_system_token(
            owner_principal_id=principal_id,
            created_by='test',
            now=utcnow(),
        )
        await session.commit()


async def _seed_api_principal(
    session_factory: async_sessionmaker,
    principal_id: str,
    *,
    roles: set[str],
) -> None:
    async with session_factory() as session:
        repository = AccessRepository(session)
        await repository.ensure_user(
            principal_id=principal_id,
            issuer='test',
            tenant='default',
            subject=principal_id,
            roles=roles,
            display_name=principal_id,
            actor='test',
            now=utcnow(),
        )
        await repository.ensure_system_token(
            owner_principal_id=principal_id,
            created_by='test',
            now=utcnow(),
        )
        await session.commit()


def _identity_headers(principal_id: str) -> dict[str, str]:
    return {
        'X-Nexus-Internal-Token': 'test-internal-token',
        'X-Nexus-Delegated-Principal-Id': principal_id,
    }


def test_run_trace_backend_catalog_metadata_is_explicit_and_safe_by_default() -> None:
    config = DistributedConfig(
        run_trace_sinks={
            'private': {
                'sinkClass': 'nexus.runner.trace.local::RunTraceLocalSink',
                'sourceClass': 'nexus.runner.trace.local::RunTraceLocalSource',
            },
            'team-a': {
                'kind': 'tdsql',
                'displayName': 'Team A Trace Database',
                'description': 'Team A production traces',
                'userSelectable': True,
                'selfGrantable': True,
                'sinkClass': 'nexus.runner.trace.local::RunTraceLocalSink',
                'sourceClass': 'nexus.runner.trace.local::RunTraceLocalSource',
            },
            'compliance': {
                'kind': 'archive',
                'displayName': 'Compliance Archive',
                'systemManaged': True,
                'required': True,
                'sinkClass': 'nexus.runner.trace.local::RunTraceLocalSink',
                'sourceClass': 'nexus.runner.trace.local::RunTraceLocalSource',
            },
        }
    )

    assert config.run_trace_sinks['private'].kind == 'custom'
    assert config.run_trace_sinks['private'].user_selectable is False
    assert config.run_trace_sinks['private'].self_grantable is False
    assert config.run_trace_sinks['team-a'].kind == 'tdsql'
    assert config.run_trace_sinks['team-a'].display_name == 'Team A Trace Database'
    assert config.run_trace_sinks['team-a'].description == 'Team A production traces'
    assert config.run_trace_sinks['team-a'].user_selectable is True
    assert config.run_trace_sinks['team-a'].self_grantable is True
    assert config.run_trace_sinks['compliance'].system_managed is True
    assert config.run_trace_sinks['compliance'].required is True

    catalog = {backend.name: backend for backend in run_trace_backend_catalog(config)}
    assert (catalog['local'].system_managed, catalog['local'].required) == (True, False)
    assert (catalog['artifact'].system_managed, catalog['artifact'].required) == (True, True)
    assert (catalog['compliance'].system_managed, catalog['compliance'].required) == (True, True)
    assert catalog['team-a'].self_grantable is True


@pytest.mark.parametrize(
    ('backend', 'message'),
    [
        (
            {'systemManaged': True, 'userSelectable': True},
            'system-managed run trace sinks cannot be user selectable',
        ),
        ({'required': True}, 'required run trace sinks must be system managed'),
        ({'selfGrantable': True}, 'self-grantable run trace sinks must be user selectable'),
    ],
)
def test_run_trace_backend_catalog_rejects_conflicting_access_metadata(
    backend: dict[str, bool],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        DistributedConfig(run_trace_sinks={'invalid': backend})


@pytest.mark.asyncio
async def test_run_trace_backend_grants_are_part_of_the_control_schema() -> None:
    engine = create_async_engine('sqlite+aiosqlite:///:memory:')
    try:
        await create_schema(engine)

        async with engine.connect() as connection:
            columns, primary_key = await connection.run_sync(
                lambda sync_connection: (
                    {
                        column['name']
                        for column in inspect(sync_connection).get_columns('distributed_run_trace_backend_grants')
                    },
                    inspect(sync_connection).get_pk_constraint('distributed_run_trace_backend_grants')[
                        'constrained_columns'
                    ],
                )
            )

        assert columns == {
            'principal_id',
            'backend_name',
            'enabled',
            'default_order',
            'version',
            'created_by',
            'updated_by',
            'created_at',
            'updated_at',
        }
        assert primary_key == ['principal_id', 'backend_name']
    finally:
        await engine.dispose()


def test_run_trace_backend_grant_repository_has_a_dedicated_module() -> None:
    assert importlib.util.find_spec('nexus.distributed.repositories.run_trace_backend') is not None


def test_run_trace_backend_grant_repository_exports_typed_boundary() -> None:
    module = importlib.import_module('nexus.distributed.repositories.run_trace_backend')

    assert hasattr(module, 'RunTraceBackendGrantInput')
    assert hasattr(module, 'RunTraceBackendGrantRecord')
    assert hasattr(module, 'RunTraceBackendGrantRepository')


@pytest.mark.asyncio
async def test_grant_repository_uses_the_principal_row_as_a_replacement_lock_anchor() -> None:
    statements = []

    class RecordingSession:
        async def scalar(self, statement):
            statements.append(statement)
            return object()

    repository = RunTraceBackendGrantRepository(RecordingSession())  # type: ignore[arg-type]
    lock_principal = getattr(repository, 'lock_principal', None)

    assert lock_principal is not None
    await lock_principal('principal-a')
    assert str(statements[0].compile(dialect=mysql.dialect())).rstrip().endswith('FOR UPDATE')


def test_run_trace_backend_access_policy_has_a_dedicated_module() -> None:
    assert importlib.util.find_spec('nexus.distributed.access.run_trace_backends') is not None


def test_run_trace_backend_access_policy_exports_typed_boundary() -> None:
    module = importlib.import_module('nexus.distributed.access.run_trace_backends')

    assert hasattr(module, 'RunTraceBackendAccessPolicy')
    assert hasattr(module, 'RunTraceBackendView')
    assert hasattr(module, 'ResolvedRunTraceBackendSelection')


@pytest.mark.asyncio
async def test_grant_repository_replaces_a_principal_grant_set_atomically(session_factory: async_sessionmaker) -> None:
    await _seed_principal(session_factory)
    now = utcnow()
    async with session_factory() as session:
        repository = RunTraceBackendGrantRepository(session)
        created = await repository.replace_for_principal(
            'principal-a',
            [
                RunTraceBackendGrantInput('archive', default_order=1),
                RunTraceBackendGrantInput('primary', default_order=0),
                RunTraceBackendGrantInput('optional', enabled=False),
            ],
            actor='admin-a',
            now=now,
        )
        await session.commit()

    assert [(item.backend_name, item.enabled, item.default_order, item.version) for item in created] == [
        ('archive', True, 1, 1),
        ('optional', False, None, 1),
        ('primary', True, 0, 1),
    ]

    later = utcnow()
    async with session_factory() as session:
        repository = RunTraceBackendGrantRepository(session)
        replaced = await repository.replace_for_principal(
            'principal-a',
            [
                RunTraceBackendGrantInput('archive', enabled=False),
                RunTraceBackendGrantInput('primary', default_order=0),
            ],
            actor='admin-b',
            now=later,
        )
        await session.commit()

    assert [(item.backend_name, item.enabled, item.default_order, item.version) for item in replaced] == [
        ('archive', False, None, 2),
        ('primary', True, 0, 1),
    ]
    assert replaced[0].updated_by == 'admin-b'
    assert replaced[1].updated_by == 'admin-a'

    async with session_factory() as session:
        persisted = await RunTraceBackendGrantRepository(session).list_for_principal('principal-a')
    assert [item.backend_name for item in persisted] == ['archive', 'primary']


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'grants',
    [
        [RunTraceBackendGrantInput('primary'), RunTraceBackendGrantInput('primary')],
        [RunTraceBackendGrantInput('primary', default_order=1)],
        [
            RunTraceBackendGrantInput('primary', default_order=0),
            RunTraceBackendGrantInput('archive', default_order=0),
        ],
        [RunTraceBackendGrantInput('primary', enabled=False, default_order=0)],
        [RunTraceBackendGrantInput(f'backend-{index}', default_order=index) for index in range(9)],
        [RunTraceBackendGrantInput('primary', default_order=True)],
        [RunTraceBackendGrantInput('primary', default_order='0')],  # type: ignore[arg-type]
        [
            RunTraceBackendGrantInput('primary', default_order=0),
            RunTraceBackendGrantInput('archive', default_order='1'),  # type: ignore[arg-type]
        ],
    ],
)
async def test_grant_repository_rejects_invalid_grant_sets(
    session_factory: async_sessionmaker,
    grants: list[RunTraceBackendGrantInput],
) -> None:
    await _seed_principal(session_factory)

    async with session_factory() as session:
        with pytest.raises(DomainValidationError, match='run_trace_backend_grants_invalid'):
            await RunTraceBackendGrantRepository(session).replace_for_principal(
                'principal-a',
                grants,
                actor='admin-a',
                now=utcnow(),
            )


def _backend_access_config(*, grants_required: bool = False) -> DistributedConfig:
    return DistributedConfig(
        require_principal_run_trace_backend_grant=grants_required,
        run_trace_sinks={
            'archive': {
                'kind': 'cos',
                'displayName': 'Archive',
                'description': 'Long-term archive',
                'userSelectable': True,
                'selfGrantable': True,
                'sinkClass': 'nexus.runner.trace.local::RunTraceLocalSink',
                'sourceClass': 'nexus.runner.trace.local::RunTraceLocalSource',
            },
            'primary': {
                'kind': 'tdsql',
                'displayName': 'Primary',
                'userSelectable': True,
                'sinkClass': 'nexus.runner.trace.local::RunTraceLocalSink',
                'sourceClass': 'nexus.runner.trace.local::RunTraceLocalSource',
            },
            'hidden': {
                'kind': 'internal',
                'displayName': 'Hidden',
                'sinkClass': 'nexus.runner.trace.local::RunTraceLocalSink',
                'sourceClass': 'nexus.runner.trace.local::RunTraceLocalSource',
            },
            'system-required': {
                'kind': 'internal',
                'displayName': 'System Required',
                'systemManaged': True,
                'required': True,
                'sinkClass': 'nexus.runner.trace.local::RunTraceLocalSink',
                'sourceClass': 'nexus.runner.trace.local::RunTraceLocalSource',
            },
        },
    )


@pytest.mark.asyncio
async def test_access_policy_lists_only_enabled_selectable_grants(session_factory: async_sessionmaker) -> None:
    await _seed_principal(session_factory)
    async with session_factory() as session:
        await RunTraceBackendGrantRepository(session).replace_for_principal(
            'principal-a',
            [
                RunTraceBackendGrantInput('archive'),
                RunTraceBackendGrantInput('primary', default_order=0),
                RunTraceBackendGrantInput('hidden'),
                RunTraceBackendGrantInput('missing'),
            ],
            actor='admin-a',
            now=utcnow(),
        )
        await session.commit()

    async with session_factory() as session:
        views = await RunTraceBackendAccessPolicy(session, config=_backend_access_config()).list_for_principal(
            'principal-a'
        )

    assert [(view.name, view.kind, view.display_name, view.default_order) for view in views] == [
        ('primary', 'tdsql', 'Primary', 0),
        ('archive', 'cos', 'Archive', None),
    ]
    assert views[1].description == 'Long-term archive'


@pytest.mark.asyncio
async def test_access_policy_lists_only_ungranted_self_grantable_backends(
    session_factory: async_sessionmaker,
) -> None:
    await _seed_principal(session_factory)
    async with session_factory() as session:
        await RunTraceBackendGrantRepository(session).replace_for_principal(
            'principal-a',
            [
                RunTraceBackendGrantInput('archive', enabled=False),
                RunTraceBackendGrantInput('primary'),
            ],
            actor='admin-a',
            now=utcnow(),
        )
        await session.commit()

    async with session_factory() as session:
        access = await RunTraceBackendAccessPolicy(
            session,
            config=_backend_access_config(),
        ).list_access_for_principal('principal-a')

    assert [item.name for item in access.items] == ['primary']
    assert [item.name for item in access.claimable_sinks] == ['archive']


@pytest.mark.asyncio
async def test_access_policy_self_grant_preserves_existing_grants_and_is_idempotent(
    session_factory: async_sessionmaker,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _seed_principal(session_factory)
    prepare_scope = AsyncMock()
    monkeypatch.setattr(
        'nexus.distributed.access.run_trace_backends.prepare_configured_run_trace_backend_scope',
        prepare_scope,
    )
    async with session_factory() as session:
        existing = await RunTraceBackendGrantRepository(session).replace_for_principal(
            'principal-a',
            [
                RunTraceBackendGrantInput('archive', enabled=False),
                RunTraceBackendGrantInput('primary', default_order=0),
            ],
            actor='admin-a',
            now=utcnow(),
        )
        await session.commit()
    original_primary = next(item for item in existing if item.backend_name == 'primary')

    async with session_factory() as session:
        first = await RunTraceBackendAccessPolicy(
            session,
            config=_backend_access_config(),
        ).self_grant(
            'principal-a',
            'archive',
            actor='principal-a',
            now=utcnow(),
        )
        await session.commit()

    assert first.changed is True
    assert (first.grant.backend_name, first.grant.enabled, first.grant.default_order, first.grant.version) == (
        'archive',
        True,
        None,
        2,
    )
    prepare_scope.assert_awaited_once_with(
        _backend_access_config(),
        backend_name='archive',
        scope_id=principal_trace_scope_id('principal-a'),
    )

    async with session_factory() as session:
        repeated = await RunTraceBackendAccessPolicy(
            session,
            config=_backend_access_config(),
        ).self_grant(
            'principal-a',
            'archive',
            actor='principal-a',
            now=utcnow(),
        )
        await session.commit()
        persisted = await RunTraceBackendGrantRepository(session).list_for_principal('principal-a')

    assert repeated.changed is False
    assert repeated.grant.version == first.grant.version
    prepare_scope.assert_awaited_once()
    persisted_primary = next(item for item in persisted if item.backend_name == 'primary')
    assert persisted_primary == original_primary


@pytest.mark.asyncio
@pytest.mark.parametrize('backend_name', ['primary', 'hidden', 'artifact', 'missing'])
async def test_access_policy_rejects_self_grant_for_unavailable_backends(
    session_factory: async_sessionmaker,
    backend_name: str,
) -> None:
    await _seed_principal(session_factory)

    async with session_factory() as session:
        with pytest.raises(DomainValidationError, match='run_trace_backend_not_available'):
            await RunTraceBackendAccessPolicy(
                session,
                config=_backend_access_config(),
            ).self_grant(
                'principal-a',
                backend_name,
                actor='principal-a',
                now=utcnow(),
            )


@pytest.mark.asyncio
async def test_access_policy_does_not_persist_self_grant_when_scope_preparation_fails(
    session_factory: async_sessionmaker,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _seed_principal(session_factory)
    monkeypatch.setattr(
        'nexus.distributed.access.run_trace_backends.prepare_configured_run_trace_backend_scope',
        AsyncMock(side_effect=RuntimeError('TDSQL DDL unavailable')),
    )

    async with session_factory() as session:
        with pytest.raises(RuntimeError, match='TDSQL DDL unavailable'):
            await RunTraceBackendAccessPolicy(
                session,
                config=_backend_access_config(),
            ).self_grant(
                'principal-a',
                'archive',
                actor='principal-a',
                now=utcnow(),
            )

    async with session_factory() as session:
        grants = await RunTraceBackendGrantRepository(session).list_for_principal('principal-a')
    assert grants == []


@pytest.mark.asyncio
async def test_access_policy_resolves_requested_and_principal_default_sinks(
    session_factory: async_sessionmaker,
) -> None:
    await _seed_principal(session_factory)
    async with session_factory() as session:
        policy = RunTraceBackendAccessPolicy(session, config=_backend_access_config())
        await policy.replace_grants(
            'principal-a',
            [
                RunTraceBackendGrantInput('archive', default_order=1),
                RunTraceBackendGrantInput('primary', default_order=0),
            ],
            actor='admin-a',
            now=utcnow(),
        )
        await session.commit()

    async with session_factory() as session:
        policy = RunTraceBackendAccessPolicy(session, config=_backend_access_config())
        requested = await policy.resolve_for_submission('principal-a', ['archive', 'primary'])
        defaulted = await policy.resolve_for_submission('principal-a', None)

    assert requested.sinks == ('archive', 'primary')
    assert requested.selection_source == 'requested'
    assert defaulted.sinks == ('primary', 'archive')
    assert defaulted.selection_source == 'principal_default'


@pytest.mark.asyncio
async def test_access_policy_returns_the_locked_before_and_after_grant_snapshots(
    session_factory: async_sessionmaker,
) -> None:
    await _seed_principal(session_factory)
    config = _backend_access_config()
    async with session_factory() as session:
        await RunTraceBackendAccessPolicy(session, config=config).replace_grants(
            'principal-a',
            [RunTraceBackendGrantInput('primary', default_order=0)],
            actor='admin-a',
            now=utcnow(),
        )
        await session.commit()

    async with session_factory() as session:
        replacement = await RunTraceBackendAccessPolicy(session, config=config).replace_grants(
            'principal-a',
            [RunTraceBackendGrantInput('archive', default_order=0)],
            actor='admin-b',
            now=utcnow(),
        )

    assert [item.backend_name for item in getattr(replacement, 'before', ())] == ['primary']
    assert [item.backend_name for item in getattr(replacement, 'after', ())] == ['archive']


@pytest.mark.asyncio
async def test_access_policy_prepares_enabled_backend_scope_before_persisting_grant(
    session_factory: async_sessionmaker,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _seed_principal(session_factory)
    prepare_scope = AsyncMock()
    monkeypatch.setattr(
        'nexus.distributed.access.run_trace_backends.prepare_configured_run_trace_backend_scope',
        prepare_scope,
    )
    config = _backend_access_config()

    async with session_factory() as session:
        await RunTraceBackendAccessPolicy(session, config=config).replace_grants(
            'principal-a',
            [
                RunTraceBackendGrantInput('archive', enabled=False),
                RunTraceBackendGrantInput('primary'),
            ],
            actor='admin-a',
            now=utcnow(),
        )

    prepare_scope.assert_awaited_once_with(
        config,
        backend_name='primary',
        scope_id=principal_trace_scope_id('principal-a'),
    )


@pytest.mark.asyncio
async def test_access_policy_does_not_persist_grant_when_scope_preparation_fails(
    session_factory: async_sessionmaker,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _seed_principal(session_factory)
    prepare_scope = AsyncMock(side_effect=RuntimeError('TDSQL DDL unavailable'))
    monkeypatch.setattr(
        'nexus.distributed.access.run_trace_backends.prepare_configured_run_trace_backend_scope',
        prepare_scope,
    )

    async with session_factory() as session:
        with pytest.raises(RuntimeError, match='TDSQL DDL unavailable'):
            await RunTraceBackendAccessPolicy(session, config=_backend_access_config()).replace_grants(
                'principal-a',
                [RunTraceBackendGrantInput('primary')],
                actor='admin-a',
                now=utcnow(),
            )

    async with session_factory() as session:
        grants = await RunTraceBackendGrantRepository(session).list_for_principal('principal-a')
    assert grants == []


@pytest.mark.asyncio
@pytest.mark.parametrize('requested', [['missing'], ['hidden'], ['primary', 'primary'], []])
async def test_access_policy_rejects_unavailable_or_invalid_requested_sinks(
    session_factory: async_sessionmaker,
    requested: list[str],
) -> None:
    await _seed_principal(session_factory)
    async with session_factory() as session:
        await RunTraceBackendAccessPolicy(session, config=_backend_access_config()).replace_grants(
            'principal-a',
            [RunTraceBackendGrantInput('primary')],
            actor='admin-a',
            now=utcnow(),
        )
        await session.commit()

    async with session_factory() as session:
        with pytest.raises(DomainValidationError, match='run_trace_backend_not_available'):
            await RunTraceBackendAccessPolicy(session, config=_backend_access_config()).resolve_for_submission(
                'principal-a', requested
            )


@pytest.mark.asyncio
async def test_access_policy_preserves_legacy_default_until_grants_are_required(
    session_factory: async_sessionmaker,
) -> None:
    await _seed_principal(session_factory)

    async with session_factory() as session:
        resolved = await RunTraceBackendAccessPolicy(
            session,
            config=_backend_access_config(grants_required=False),
        ).resolve_for_submission('principal-a', None)
    assert resolved.sinks is None
    assert resolved.selection_source == 'legacy_default'

    async with session_factory() as session:
        with pytest.raises(DomainValidationError, match='run_trace_backend_selection_required'):
            await RunTraceBackendAccessPolicy(
                session,
                config=_backend_access_config(grants_required=True),
            ).resolve_for_submission('principal-a', None)


@pytest.mark.asyncio
async def test_access_policy_rejects_grants_for_hidden_or_missing_backends(
    session_factory: async_sessionmaker,
) -> None:
    await _seed_principal(session_factory)

    async with session_factory() as session:
        policy = RunTraceBackendAccessPolicy(session, config=_backend_access_config())
        for backend_name in ('hidden', 'missing'):
            with pytest.raises(DomainValidationError, match='run_trace_backend_not_available'):
                await policy.replace_grants(
                    'principal-a',
                    [RunTraceBackendGrantInput(backend_name)],
                    actor='admin-a',
                    now=utcnow(),
                )


@pytest.mark.asyncio
async def test_access_policy_requires_active_trace_database_account_to_be_disabled_before_revoke(
    session_factory: async_sessionmaker,
) -> None:
    await _seed_principal(session_factory)
    now = utcnow()
    async with session_factory() as session:
        policy = RunTraceBackendAccessPolicy(session, config=_backend_access_config())
        await policy.replace_grants(
            'principal-a',
            [RunTraceBackendGrantInput('primary')],
            actor='admin-a',
            now=now,
        )
        session.add(
            TraceDatabaseAccountRow(
                owner_principal_id='principal-a',
                backend_name='primary',
                db_username='nxs_ro_test',
                status='ready',
                credential_version=1,
                created_at=now,
                updated_at=now,
            )
        )
        await session.commit()

    async with session_factory() as session:
        policy = RunTraceBackendAccessPolicy(session, config=_backend_access_config())
        with pytest.raises(DomainConflictError, match='trace_database_account_must_be_disabled'):
            await policy.replace_grants('principal-a', [], actor='admin-a', now=utcnow())

        account = await session.get(TraceDatabaseAccountRow, ('principal-a', 'primary'))
        assert account is not None
        account.status = 'disabled'
        account.disabled_at = utcnow()
        account.operation_id = 'restore-in-progress'
        await session.commit()

    async with session_factory() as session:
        with pytest.raises(DomainConflictError, match='trace_database_account_must_be_disabled'):
            await RunTraceBackendAccessPolicy(session, config=_backend_access_config()).replace_grants(
                'principal-a',
                [],
                actor='admin-a',
                now=utcnow(),
            )

        account = await session.get(TraceDatabaseAccountRow, ('principal-a', 'primary'))
        assert account is not None
        account.operation_id = None
        account.status = 'error'
        await session.commit()

    async with session_factory() as session:
        with pytest.raises(DomainConflictError, match='trace_database_account_must_be_disabled'):
            await RunTraceBackendAccessPolicy(session, config=_backend_access_config()).replace_grants(
                'principal-a',
                [],
                actor='admin-a',
                now=utcnow(),
            )

        account = await session.get(TraceDatabaseAccountRow, ('principal-a', 'primary'))
        assert account is not None
        account.status = 'disabled'
        await session.commit()

    async with session_factory() as session:
        await RunTraceBackendAccessPolicy(session, config=_backend_access_config()).replace_grants(
            'principal-a',
            [],
            actor='admin-a',
            now=utcnow(),
        )
        await session.commit()


@pytest.mark.asyncio
async def test_run_trace_backend_access_routes_enforce_roles_and_hide_backend_config(
    session_factory: async_sessionmaker,
) -> None:
    await _seed_api_principal(session_factory, 'principal-admin', roles={'user', 'admin'})
    await _seed_api_principal(session_factory, 'principal-user', roles={'user'})
    backend_config = _backend_access_config()
    config = DistributedConfig(
        auth_mode='required',
        secrets={'internal_token': 'test-internal-token'},
        run_trace_sinks=backend_config.run_trace_sinks,
    )
    app = create_control_app(session_factory=session_factory, config=config)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        admin_headers = _identity_headers('principal-admin')
        user_headers = _identity_headers('principal-user')

        response = await client.get('/v1/admin/run-trace-backends', headers=admin_headers)
        assert response.status == 200
        catalog = await response.json()
        assert catalog == {
            'items': [
                {
                    'name': 'archive',
                    'kind': 'cos',
                    'display_name': 'Archive',
                    'description': 'Long-term archive',
                    'user_selectable': True,
                    'self_grantable': True,
                    'system_managed': False,
                    'required': False,
                },
                {
                    'name': 'artifact',
                    'kind': 'artifact',
                    'display_name': 'Artifact Storage',
                    'description': None,
                    'user_selectable': False,
                    'self_grantable': False,
                    'system_managed': True,
                    'required': True,
                },
                {
                    'name': 'hidden',
                    'kind': 'internal',
                    'display_name': 'Hidden',
                    'description': None,
                    'user_selectable': False,
                    'self_grantable': False,
                    'system_managed': False,
                    'required': False,
                },
                {
                    'name': 'local',
                    'kind': 'local',
                    'display_name': 'Local Storage',
                    'description': None,
                    'user_selectable': False,
                    'self_grantable': False,
                    'system_managed': True,
                    'required': False,
                },
                {
                    'name': 'primary',
                    'kind': 'tdsql',
                    'display_name': 'Primary',
                    'description': None,
                    'user_selectable': True,
                    'self_grantable': False,
                    'system_managed': False,
                    'required': False,
                },
                {
                    'name': 'system-required',
                    'kind': 'internal',
                    'display_name': 'System Required',
                    'description': None,
                    'user_selectable': False,
                    'self_grantable': False,
                    'system_managed': True,
                    'required': True,
                },
            ]
        }
        serialized_catalog = str(catalog)
        assert 'sinkClass' not in serialized_catalog
        assert 'sourceClass' not in serialized_catalog
        assert 'config' not in serialized_catalog
        assert 'secret' not in serialized_catalog.lower()

        response = await client.put(
            '/v1/admin/principals/principal-user/run-trace-backends',
            headers=user_headers,
            json={'grants': [{'backend_name': 'primary', 'enabled': True, 'default_order': 0}]},
        )
        assert response.status == 403

        response = await client.put(
            '/v1/admin/principals/principal-user/run-trace-backends',
            headers=admin_headers,
            json={
                'grants': [
                    {'backend_name': 'archive', 'enabled': True, 'default_order': 1},
                    {'backend_name': 'primary', 'enabled': True, 'default_order': 0},
                ]
            },
        )
        assert response.status == 200
        updated = await response.json()
        assert [(item['backend_name'], item['default_order']) for item in updated['items']] == [
            ('archive', 1),
            ('primary', 0),
        ]

        response = await client.get('/v1/access/run-trace-backends', headers=user_headers)
        assert response.status == 200
        visible = await response.json()
        assert [(item['name'], item['default_order']) for item in visible['items']] == [
            ('primary', 0),
            ('archive', 1),
        ]
        assert all('system_managed' not in item and 'required' not in item for item in visible['items'])
        assert visible['required_sinks'] == [
            {
                'name': 'artifact',
                'kind': 'artifact',
                'display_name': 'Artifact Storage',
                'description': None,
                'system_managed': True,
                'required': True,
            },
            {
                'name': 'system-required',
                'kind': 'internal',
                'display_name': 'System Required',
                'description': None,
                'system_managed': True,
                'required': True,
            },
        ]
        assert visible['claimable_sinks'] == []

        response = await client.get(
            '/v1/admin/principals/principal-user/run-trace-backends',
            headers=admin_headers,
        )
        assert response.status == 200
        assert (await response.json())['items'] == updated['items']

        response = await client.put(
            '/v1/admin/principals/principal-user/run-trace-backends',
            headers=admin_headers,
            json={'grants': [{'backend_name': 'hidden', 'enabled': True, 'default_order': 0}]},
        )
        assert response.status == 422
        assert (await response.json())['error'] == 'run_trace_backend_not_available'
    finally:
        await client.close()

    async with session_factory() as session:
        audit = await session.scalar(select(AuditRow).where(AuditRow.operation == 'access.run_trace_backends.replace'))
    assert audit is not None
    assert audit.actor_principal_id == 'principal-admin'
    assert audit.target_owner_principal_id == 'principal-user'
    assert audit.metadata_json == {
        'added': ['archive', 'primary'],
        'removed': [],
        'updated': [],
    }


@pytest.mark.asyncio
async def test_user_self_grants_only_claimable_run_trace_backend_and_route_is_idempotent(
    session_factory: async_sessionmaker,
) -> None:
    await _seed_api_principal(session_factory, 'principal-user', roles={'user'})
    backend_config = _backend_access_config()
    config = DistributedConfig(
        auth_mode='required',
        secrets={'internal_token': 'test-internal-token'},
        run_trace_sinks=backend_config.run_trace_sinks,
    )
    app = create_control_app(session_factory=session_factory, config=config)
    client = TestClient(TestServer(app))
    await client.start_server()
    headers = _identity_headers('principal-user')
    try:
        response = await client.get('/v1/access/run-trace-backends', headers=headers)
        assert response.status == 200
        visible = await response.json()
        assert visible['items'] == []
        assert visible['claimable_sinks'] == [
            {
                'name': 'archive',
                'kind': 'cos',
                'display_name': 'Archive',
                'description': 'Long-term archive',
            }
        ]

        response = await client.put('/v1/access/run-trace-backends/primary', headers=headers)
        assert response.status == 422
        assert (await response.json())['error'] == 'run_trace_backend_not_available'

        response = await client.put('/v1/access/run-trace-backends/archive', headers=headers)
        assert response.status == 200, await response.text()
        granted = (await response.json())['item']
        assert (granted['backend_name'], granted['enabled'], granted['default_order']) == ('archive', True, None)

        response = await client.put('/v1/access/run-trace-backends/archive', headers=headers)
        assert response.status == 200, await response.text()
        assert (await response.json())['item']['version'] == granted['version']

        response = await client.get('/v1/access/run-trace-backends', headers=headers)
        assert response.status == 200
        visible = await response.json()
        assert [item['name'] for item in visible['items']] == ['archive']
        assert visible['claimable_sinks'] == []
    finally:
        await client.close()

    async with session_factory() as session:
        audits = list(
            await session.scalars(select(AuditRow).where(AuditRow.operation == 'access.run_trace_backends.self_grant'))
        )
    assert len(audits) == 1
    assert audits[0].actor_principal_id == 'principal-user'
    assert audits[0].target_owner_principal_id == 'principal-user'
    assert audits[0].resource_type == 'run_trace_backend_grant'
    assert audits[0].resource_id == 'archive'
    assert audits[0].metadata_json == {'backend_name': 'archive'}


@pytest.mark.asyncio
async def test_unicode_run_trace_backend_can_be_granted_and_selected(
    session_factory: async_sessionmaker,
) -> None:
    await _seed_api_principal(session_factory, 'principal-admin', roles={'user', 'admin'})
    await _seed_api_principal(session_factory, 'principal-user', roles={'user'})
    backend_name = '团队归档'
    config = DistributedConfig(
        auth_mode='required',
        secrets={'internal_token': 'test-internal-token'},
        run_trace_sinks={
            backend_name: {
                'kind': 'custom',
                'displayName': 'Team Archive',
                'userSelectable': True,
                'sinkClass': 'nexus.runner.trace.local::RunTraceLocalSink',
                'sourceClass': 'nexus.runner.trace.local::RunTraceLocalSource',
            }
        },
    )
    app = create_control_app(session_factory=session_factory, config=config)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.put(
            '/v1/admin/principals/principal-user/run-trace-backends',
            headers=_identity_headers('principal-admin'),
            json={'grants': [{'backend_name': backend_name, 'enabled': True, 'default_order': 0}]},
        )
        assert response.status == 200
    finally:
        await client.close()

    async with session_factory() as session:
        resolved = await RunTraceBackendAccessPolicy(session, config=config).resolve_for_submission(
            'principal-user',
            [backend_name],
        )

    assert resolved.sinks == (backend_name,)
