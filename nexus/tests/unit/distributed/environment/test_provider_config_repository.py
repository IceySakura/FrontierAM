from __future__ import annotations

import importlib
import inspect as pyinspect

import pytest

from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, inspect
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from nexus.distributed.errors import DomainConflictError, ResourceNotFoundError
from nexus.distributed.persistence.orm import create_schema
from nexus.distributed.repositories.provider_config import ProviderConfigRepository
from nexus.distributed.utils.clock import utcnow


@pytest.fixture
async def session_factory(tmp_path):
    engine = create_async_engine(f'sqlite+aiosqlite:///{tmp_path / "provider-config.db"}')
    await create_schema(engine)
    yield async_sessionmaker(engine, expire_on_commit=False)
    await engine.dispose()


def _tkex_config(*, credential_ref: str = 'tkex-shanghai') -> dict[str, object]:
    return {
        'provider_type': 'tkex',
        'namespace': 'nexus-workloads',
        'base_url': 'https://tkex.example',
        'api_account': 'nexus-runner',
        'credential_ref': credential_ref,
        'cluster_name': 'cls-shanghai',
        'project_name': 'nexus',
    }


async def test_provider_config_repository_creates_and_resolves_enabled_config(session_factory) -> None:
    now = utcnow()
    async with session_factory() as session:
        repository = ProviderConfigRepository(session)
        created = await repository.create(
            provider_config_ref='tkex/shanghai-load',
            config=_tkex_config(),
            status='enabled',
            actor='admin',
            now=now,
        )
        await session.commit()

    async with session_factory() as session:
        resolved = await ProviderConfigRepository(session).require_enabled(
            'tkex/shanghai-load',
            provider_type='tkex',
        )

    assert created.provider_config_ref == 'tkex/shanghai-load'
    assert created.provider_type == 'tkex'
    assert created.config['credential_ref'] == 'tkex-shanghai'
    assert created.version == 1
    assert resolved == created


async def test_provider_config_repository_rejects_duplicate_ref(session_factory) -> None:
    async with session_factory() as session:
        repository = ProviderConfigRepository(session)
        await repository.create(
            provider_config_ref='tkex/shanghai-load',
            config=_tkex_config(),
            status='enabled',
            actor='admin',
            now=utcnow(),
        )
        with pytest.raises(DomainConflictError, match='provider_config_already_exists'):
            await repository.create(
                provider_config_ref='tkex/shanghai-load',
                config=_tkex_config(credential_ref='another-token'),
                status='enabled',
                actor='admin',
                now=utcnow(),
            )


async def test_provider_config_repository_disables_config_and_fails_closed(session_factory) -> None:
    async with session_factory() as session:
        repository = ProviderConfigRepository(session)
        await repository.create(
            provider_config_ref='tkex/shanghai-load',
            config=_tkex_config(),
            status='enabled',
            actor='admin',
            now=utcnow(),
        )
        disabled = await repository.update(
            'tkex/shanghai-load',
            status='disabled',
            actor='admin',
            now=utcnow(),
        )
        await session.commit()

        with pytest.raises(DomainConflictError, match='provider_config_disabled'):
            await repository.require_enabled('tkex/shanghai-load', provider_type='tkex')

    assert disabled.status == 'disabled'
    assert disabled.version == 2


async def test_provider_config_repository_rejects_provider_type_change(session_factory) -> None:
    async with session_factory() as session:
        repository = ProviderConfigRepository(session)
        await repository.create(
            provider_config_ref='tkex/shanghai-load',
            config=_tkex_config(),
            status='disabled',
            actor='admin',
            now=utcnow(),
        )

        with pytest.raises(DomainConflictError, match='provider_config_type_immutable'):
            await repository.update(
                'tkex/shanghai-load',
                config={
                    'provider_type': 'kubernetes',
                    'namespace': 'nexus-workloads',
                    'auth_type': 'in_cluster',
                },
                actor='admin',
                now=utcnow(),
            )


async def test_provider_config_repository_rejects_unknown_ref(session_factory) -> None:
    async with session_factory() as session:
        with pytest.raises(ResourceNotFoundError, match='provider_config_not_found'):
            await ProviderConfigRepository(session).get('missing')


def test_provider_config_migration_creates_registry_table_idempotently(tmp_path) -> None:
    migration = importlib.import_module('nexus.distributed.persistence.migrations.versions.0021_add_provider_configs')
    assert 'nexus.distributed.persistence.orm' not in pyinspect.getsource(migration)
    engine = create_engine(f'sqlite:///{tmp_path / "provider-config-migration.db"}')

    with engine.begin() as connection:
        operations = Operations(MigrationContext.configure(connection))
        migration.upgrade(operations)
        migration.upgrade(operations)

        inspector = inspect(connection)
        assert 'distributed_provider_configs' in inspector.get_table_names()
        assert {
            'provider_config_ref',
            'provider_type',
            'config_json',
            'status',
            'version',
            'created_by',
            'updated_by',
            'created_at',
            'updated_at',
        } == {column['name'] for column in inspector.get_columns('distributed_provider_configs')}

    engine.dispose()
