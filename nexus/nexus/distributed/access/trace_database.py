from __future__ import annotations

import asyncio
import logging
import secrets
import string

from collections.abc import Callable, Coroutine
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TypeVar

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker

from nexus.distributed.access.models import TraceDatabaseAccountRecord
from nexus.distributed.access.trace_database_providers import (
    MySQLTraceDatabaseAccountProvider,
    TencentCloudDCDBTraceDatabaseAccountProvider,
    TencentCloudTDMySQLTraceDatabaseAccountProvider,
    TraceDatabaseAccountProvider,
)
from nexus.distributed.config import (
    DistributedConfig,
    MySQLTraceDatabaseAccountProviderConfig,
    MySQLTraceDatabaseCredential,
    TencentCloudDCDBTraceDatabaseAccountProviderConfig,
    TencentCloudTDMySQLTraceDatabaseAccountProviderConfig,
    TencentCloudTraceDatabaseCredential,
    TraceDatabaseAccountProviderConfig,
    TraceDatabaseCredential,
)
from nexus.distributed.errors import DomainConflictError, ResourceNotFoundError, TemporaryUnavailableError
from nexus.distributed.repositories.audit import AuditRepository
from nexus.distributed.repositories.trace_database import (
    TraceDatabaseAccountRepository,
    TraceDatabaseOperationClaim,
)
from nexus.distributed.utils.clock import utcnow
from nexus.runner.trace import principal_trace_scope_id
from nexus.runner.trace.tdsql.schema import principal_trace_table_name


logger = logging.getLogger(__name__)

_T = TypeVar('_T')

_PASSWORD_LOWER = string.ascii_lowercase
_PASSWORD_UPPER = string.ascii_uppercase
_PASSWORD_DIGITS = string.digits
_PASSWORD_SYMBOLS = '!@#$%^&*-+=_'
_PASSWORD_ALPHABET = _PASSWORD_LOWER + _PASSWORD_UPPER + _PASSWORD_DIGITS + _PASSWORD_SYMBOLS
_PASSWORD_LENGTH = 32
_SYSTEM_RANDOM = secrets.SystemRandom()


@dataclass(frozen=True, slots=True)
class TraceDatabaseTarget:
    backend_name: str
    write_host: str
    write_port: int
    read_host: str
    read_port: int
    database: str
    account_provider: TraceDatabaseAccountProviderConfig


@dataclass(frozen=True, slots=True)
class TraceDatabaseAccountView:
    owner_principal_id: str
    backend_name: str
    read_host: str
    read_port: int
    database: str
    username: str | None
    status: str
    access_scope: str
    trace_table_name: str
    batch_table_name: str
    credential_version: int
    last_error_code: str | None
    updated_at: datetime | None


@dataclass(frozen=True, slots=True)
class IssuedTraceDatabaseCredential:
    account: TraceDatabaseAccountView
    password: str


TraceDatabaseAccountProviderFactory = Callable[
    [TraceDatabaseTarget, TraceDatabaseCredential],
    TraceDatabaseAccountProvider,
]


class TraceDatabaseAccountManager:
    def __init__(
        self,
        *,
        session_factory: async_sessionmaker,
        config: DistributedConfig,
        provider_factory: TraceDatabaseAccountProviderFactory | None = None,
        operation_ttl: timedelta = timedelta(minutes=2),
    ) -> None:
        if operation_ttl <= timedelta(0):
            raise ValueError('trace database operation TTL must be positive')
        self.session_factory = session_factory
        self.config = config
        self.provider_factory = provider_factory or _create_provider
        self.operation_ttl = operation_ttl
        self.operation_heartbeat_interval = max(0.01, min(operation_ttl.total_seconds() / 3, 30.0))

    async def create_or_reset(
        self,
        *,
        owner_principal_id: str,
        environment_id: str,
        backend_name: str,
        operation: str,
        actor_principal_id: str,
        now: datetime,
    ) -> IssuedTraceDatabaseCredential:
        if operation not in {'create', 'reset'}:
            raise ValueError('trace database credential operation must be create or reset')
        try:
            async with self.session_factory() as session:
                repository = TraceDatabaseAccountRepository(session)
                await repository.require_environment_access(
                    owner_principal_id=owner_principal_id,
                    environment_id=environment_id,
                )
                await repository.require_backend_access(
                    owner_principal_id=owner_principal_id,
                    backend_name=backend_name,
                )
                target, credential = self._target(backend_name)
                claim = await repository.claim_operation(
                    owner_principal_id=owner_principal_id,
                    backend_name=backend_name,
                    operation=operation,
                    now=now,
                    operation_ttl=self.operation_ttl,
                )
                await session.commit()
        except IntegrityError as exc:
            raise DomainConflictError('trace_database_account_operation_in_progress') from exc
        return await self._issue_credential(
            claim=claim,
            target=target,
            credential=credential,
            actor_principal_id=actor_principal_id,
        )

    async def admin_reset(
        self,
        *,
        owner_principal_id: str,
        backend_name: str,
        actor_principal_id: str,
        now: datetime,
    ) -> IssuedTraceDatabaseCredential:
        target, credential = self._target(backend_name)
        async with self.session_factory() as session:
            repository = TraceDatabaseAccountRepository(session)
            await repository.require_backend_access(
                owner_principal_id=owner_principal_id,
                backend_name=backend_name,
            )
            claim = await repository.claim_operation(
                owner_principal_id=owner_principal_id,
                backend_name=backend_name,
                operation='reset',
                now=now,
                operation_ttl=self.operation_ttl,
                allow_disabled_reset=True,
            )
            await session.commit()
        return await self._issue_credential(
            claim=claim,
            target=target,
            credential=credential,
            actor_principal_id=actor_principal_id,
        )

    async def admin_operation(
        self,
        *,
        owner_principal_id: str,
        backend_name: str,
        operation: str,
        actor_principal_id: str,
        now: datetime,
    ) -> TraceDatabaseAccountView:
        if operation not in {'disable', 'restore', 'reconcile'}:
            raise ValueError('unsupported trace database admin operation')
        target, credential = self._target(backend_name)
        async with self.session_factory() as session:
            repository = TraceDatabaseAccountRepository(session)
            if operation in {'restore', 'reconcile'}:
                await repository.require_backend_access(
                    owner_principal_id=owner_principal_id,
                    backend_name=backend_name,
                )
            claim = await repository.claim_operation(
                owner_principal_id=owner_principal_id,
                backend_name=backend_name,
                operation=operation,
                now=now,
                operation_ttl=self.operation_ttl,
            )
            await session.commit()
        provider = self.provider_factory(target, credential)

        async def apply_operation() -> str:
            if operation == 'disable':
                await provider.disable_account(username=claim.username)
                return 'disabled'
            if operation == 'restore':
                await provider.restore_account(
                    username=claim.username,
                    database=target.database,
                    table_name=principal_trace_table_name(principal_trace_scope_id(claim.owner_principal_id)),
                )
                return 'ready'
            await provider.reconcile_readonly_account(
                username=claim.username,
                database=target.database,
                table_name=principal_trace_table_name(principal_trace_scope_id(claim.owner_principal_id)),
            )
            if claim.previous_status == 'disabled':
                await provider.disable_account(username=claim.username)
            return claim.previous_status

        try:
            status = await self._run_with_operation_lease(claim, apply_operation)
        except Exception as exc:
            await self._fail(claim, error_code='trace_database_maintenance_failed')
            logger.exception(
                'Trace database admin operation failed',
                extra={'backend_name': backend_name, 'operation': operation},
            )
            raise TemporaryUnavailableError('trace_database_maintenance_failed') from exc
        record = await self._finish(
            claim,
            status=status,
            actor_principal_id=actor_principal_id,
            now=utcnow(),
        )
        return self._view(record, target)

    async def list_for_principal(self, owner_principal_id: str) -> list[TraceDatabaseAccountView]:
        async with self.session_factory() as session:
            repository = TraceDatabaseAccountRepository(session)
            if not await repository.has_any_environment_access(owner_principal_id):
                return []
            granted_backend_names = set(await repository.list_enabled_backend_grants(owner_principal_id))
            records = {
                record.backend_name: record
                for record in await repository.list_accounts(owner_principal_id=owner_principal_id)
            }
        views: list[TraceDatabaseAccountView] = []
        for backend_name, access in sorted(self.config.trace_database_access.items()):
            if not access.enabled or backend_name not in granted_backend_names:
                continue
            target, _credential = self._target(backend_name)
            record = records.get(backend_name)
            views.append(self._view(record, target, owner_principal_id=owner_principal_id))
        return views

    async def list_all(self) -> list[TraceDatabaseAccountView]:
        async with self.session_factory() as session:
            records = await TraceDatabaseAccountRepository(session).list_accounts()
        result: list[TraceDatabaseAccountView] = []
        for record in records:
            access = self.config.trace_database_access.get(record.backend_name)
            if access is None:
                continue
            target, _credential = self._target(record.backend_name)
            result.append(self._view(record, target))
        return result

    async def disable_for_access_revocation(
        self,
        *,
        owner_principal_id: str,
        revoked_environment_id: str,
        actor_principal_id: str,
        now: datetime,
    ) -> None:
        async with self.session_factory() as session:
            repository = TraceDatabaseAccountRepository(session)
            if await repository.has_any_environment_access(
                owner_principal_id,
                exclude_environment_id=revoked_environment_id,
            ):
                return
            records = await repository.list_accounts(owner_principal_id=owner_principal_id)
        for record in records:
            if record.disabled_at is not None:
                continue
            await self.admin_operation(
                owner_principal_id=owner_principal_id,
                backend_name=record.backend_name,
                operation='disable',
                actor_principal_id=actor_principal_id,
                now=now,
            )

    async def disable_for_environment_revocation(
        self,
        *,
        environment_id: str,
        actor_principal_id: str,
        now: datetime,
    ) -> None:
        async with self.session_factory() as session:
            owners = await TraceDatabaseAccountRepository(session).list_account_owners_with_environment_access(
                environment_id
            )
        for owner_principal_id in owners:
            await self.disable_for_access_revocation(
                owner_principal_id=owner_principal_id,
                revoked_environment_id=environment_id,
                actor_principal_id=actor_principal_id,
                now=now,
            )

    async def _finish(
        self,
        claim: TraceDatabaseOperationClaim,
        *,
        status: str,
        actor_principal_id: str,
        now: datetime,
    ) -> TraceDatabaseAccountRecord:
        async with self.session_factory() as session:
            record = await TraceDatabaseAccountRepository(session).finish_operation(claim, status=status, now=now)
            await AuditRepository(session).append(
                actor_principal_id=actor_principal_id,
                target_owner_principal_id=claim.owner_principal_id,
                operation=f'trace_database_account.{claim.operation}',
                resource_type='trace_database_account',
                resource_id=f'{claim.owner_principal_id}:{claim.backend_name}',
                metadata={
                    'backend_name': claim.backend_name,
                    'status': status,
                    'credential_version': record.credential_version,
                },
                now=now,
            )
            await session.commit()
            return record

    async def _fail(self, claim: TraceDatabaseOperationClaim, *, error_code: str) -> None:
        async with self.session_factory() as session:
            await TraceDatabaseAccountRepository(session).fail_operation(
                claim,
                error_code=error_code,
                now=utcnow(),
            )
            await session.commit()

    async def _issue_credential(
        self,
        *,
        claim: TraceDatabaseOperationClaim,
        target: TraceDatabaseTarget,
        credential: TraceDatabaseCredential,
        actor_principal_id: str,
    ) -> IssuedTraceDatabaseCredential:
        password = _generate_account_password()
        provider = self.provider_factory(target, credential)
        try:
            await self._run_with_operation_lease(
                claim,
                lambda: provider.set_readonly_account(
                    username=claim.username,
                    password=password,
                    database=target.database,
                    table_name=principal_trace_table_name(principal_trace_scope_id(claim.owner_principal_id)),
                ),
            )
        except Exception as exc:
            await self._fail(claim, error_code='trace_database_maintenance_failed')
            logger.exception(
                'Trace database credential operation failed',
                extra={'backend_name': claim.backend_name, 'operation': claim.operation},
            )
            raise TemporaryUnavailableError('trace_database_maintenance_failed') from exc
        record = await self._finish(
            claim,
            status='ready',
            actor_principal_id=actor_principal_id,
            now=utcnow(),
        )
        return IssuedTraceDatabaseCredential(account=self._view(record, target), password=password)

    async def _run_with_operation_lease(
        self,
        claim: TraceDatabaseOperationClaim,
        operation: Callable[[], Coroutine[object, object, _T]],
    ) -> _T:
        operation_task = asyncio.create_task(
            operation(),
            name=f'trace-database-operation:{claim.operation_id}',
        )
        heartbeat_task = asyncio.create_task(
            self._heartbeat_operation(claim),
            name=f'trace-database-operation-heartbeat:{claim.operation_id}',
        )
        try:
            done, _pending = await asyncio.wait(
                (operation_task, heartbeat_task),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if heartbeat_task in done:
                await heartbeat_task
                raise RuntimeError('trace database operation heartbeat stopped unexpectedly')
            return await operation_task
        finally:
            if not heartbeat_task.done():
                heartbeat_task.cancel()
                with suppress(asyncio.CancelledError):
                    await heartbeat_task
            if not operation_task.done():
                operation_task.cancel()
                with suppress(asyncio.CancelledError):
                    await operation_task

    async def _heartbeat_operation(self, claim: TraceDatabaseOperationClaim) -> None:
        while True:
            await asyncio.sleep(self.operation_heartbeat_interval)
            now = utcnow()
            async with self.session_factory() as session:
                await TraceDatabaseAccountRepository(session).renew_operation(
                    claim,
                    now=now,
                    operation_ttl=self.operation_ttl,
                )
                await session.commit()

    def _target(self, backend_name: str) -> tuple[TraceDatabaseTarget, TraceDatabaseCredential]:
        access = self.config.trace_database_access.get(backend_name)
        backend = self.config.run_trace_sinks.get(backend_name)
        if access is None or not access.enabled or backend is None:
            raise ResourceNotFoundError('trace_database_backend_not_found')
        credential = self.config.resolve_trace_database_credential(access.maintenance_credential_ref)
        if credential is None:
            raise RuntimeError('trace database maintenance credential is unavailable')
        write_host = backend.config.get('host')
        write_port = backend.config.get('port', 3306)
        database = backend.config.get('db')
        if not isinstance(write_host, str) or not isinstance(write_port, int) or database != access.database:
            raise RuntimeError('trace database backend target is invalid')
        return (
            TraceDatabaseTarget(
                backend_name=backend_name,
                write_host=write_host,
                write_port=write_port,
                read_host=access.read_host,
                read_port=access.read_port,
                database=access.database,
                account_provider=access.account_provider,
            ),
            credential,
        )

    @staticmethod
    def _view(
        record: TraceDatabaseAccountRecord | None,
        target: TraceDatabaseTarget,
        *,
        owner_principal_id: str | None = None,
    ) -> TraceDatabaseAccountView:
        owner_principal_id = record.owner_principal_id if record is not None else owner_principal_id
        if owner_principal_id is None:
            raise ValueError('trace database account owner is required')
        trace_table_name = principal_trace_table_name(principal_trace_scope_id(owner_principal_id))
        return TraceDatabaseAccountView(
            owner_principal_id=owner_principal_id,
            backend_name=target.backend_name,
            read_host=target.read_host,
            read_port=target.read_port,
            database=target.database,
            username=record.username if record is not None else None,
            status=record.status if record is not None else 'not_created',
            access_scope='database',
            trace_table_name=trace_table_name,
            batch_table_name=trace_table_name,
            credential_version=record.credential_version if record is not None else 0,
            last_error_code=record.last_error_code if record is not None else None,
            updated_at=record.updated_at if record is not None else None,
        )


def _create_provider(
    target: TraceDatabaseTarget,
    credential: TraceDatabaseCredential,
) -> TraceDatabaseAccountProvider:
    provider_config = target.account_provider
    if isinstance(provider_config, MySQLTraceDatabaseAccountProviderConfig):
        if not isinstance(credential, MySQLTraceDatabaseCredential):
            raise RuntimeError('trace database provider credential type mismatch')
        username, password = credential.read()
        return MySQLTraceDatabaseAccountProvider(
            host=target.write_host,
            port=target.write_port,
            database=target.database,
            username=username,
            password=password,
        )
    # Both cloud branches accept the same credential type: one secret id/key pair serves
    # every Tencent Cloud product it is authorised for, so which product it was registered
    # under says nothing about where it can be used.
    if isinstance(provider_config, TencentCloudDCDBTraceDatabaseAccountProviderConfig):
        if not isinstance(credential, TencentCloudTraceDatabaseCredential):
            raise RuntimeError('trace database provider credential type mismatch')
        secret_id, secret_key = credential.read()
        return TencentCloudDCDBTraceDatabaseAccountProvider(
            region=provider_config.region,
            instance_id=provider_config.instance_id,
            api_endpoint=provider_config.api_endpoint,
            database=target.database,
            account_host=provider_config.account_host,
            secret_id=secret_id,
            secret_key=secret_key,
        )
    if isinstance(provider_config, TencentCloudTDMySQLTraceDatabaseAccountProviderConfig):
        if not isinstance(credential, TencentCloudTraceDatabaseCredential):
            raise RuntimeError('trace database provider credential type mismatch')
        secret_id, secret_key = credential.read()
        return TencentCloudTDMySQLTraceDatabaseAccountProvider(
            region=provider_config.region,
            instance_id=provider_config.instance_id,
            api_endpoint=provider_config.api_endpoint,
            database=target.database,
            account_host=provider_config.account_host,
            secret_id=secret_id,
            secret_key=secret_key,
            flow_timeout_seconds=provider_config.flow_timeout_seconds,
            verify_host=target.read_host,
            verify_port=target.read_port,
        )
    raise RuntimeError(f'unsupported trace database account provider: {provider_config.provider_type}')


def _generate_account_password() -> str:
    characters = [
        secrets.choice(_PASSWORD_LOWER),
        secrets.choice(_PASSWORD_UPPER),
        secrets.choice(_PASSWORD_DIGITS),
        secrets.choice(_PASSWORD_SYMBOLS),
    ]
    characters.extend(secrets.choice(_PASSWORD_ALPHABET) for _ in range(_PASSWORD_LENGTH - len(characters)))
    _SYSTEM_RANDOM.shuffle(characters)
    return ''.join(characters)


__all__ = [
    'IssuedTraceDatabaseCredential',
    'TraceDatabaseAccountManager',
    'TraceDatabaseAccountProvider',
    'TraceDatabaseAccountView',
    'TraceDatabaseTarget',
]
