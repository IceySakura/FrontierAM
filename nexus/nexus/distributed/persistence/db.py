from __future__ import annotations

import asyncio
import importlib
import logging
import re
import time

from collections.abc import Awaitable, Callable
from enum import Enum
from typing import Any, TypeVar

from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import event, inspect, text
from sqlalchemy.engine import URL, make_url
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.exc import TimeoutError as SqlAlchemyTimeoutError
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

from nexus.distributed.config import DistributedConfig
from nexus.distributed.persistence.migrations import (
    CLEANUP_REVISION,
    HEAD_REVISION,
    INITIAL_REVISION,
    LEGACY_CLOUD_SUBMIT_REVISION,
    MIGRATIONS,
)


_MYSQL_DATABASE_RE = re.compile(r'^[A-Za-z0-9_]+$')
_MAX_LOGGED_SQL_LENGTH = 2000
logger = logging.getLogger(__name__)
ResultT = TypeVar('ResultT')


class TransactionRetryError(RuntimeError):
    """Retry an idempotent aggregate in a fresh database transaction."""


class DatabaseFailureKind(str, Enum):
    TRANSACTION_CONFLICT = 'transaction_conflict'
    CONNECTION_LOST = 'connection_lost'
    PERMANENT = 'permanent'


_MYSQL_TRANSACTION_CONFLICT_CODES = {1205, 1213}
# Server shutdown, network I/O failures, and failed or lost connections.
_MYSQL_CONNECTION_LOST_CODES = {1053, 1158, 1159, 1160, 1161, 2003, 2006, 2013, 2055}
_SQLSTATE_TRANSACTION_CONFLICTS = {'40001', '40P01', '55P03'}


def classify_database_error(error: BaseException) -> DatabaseFailureKind:
    if isinstance(error, SqlAlchemyTimeoutError):
        return DatabaseFailureKind.CONNECTION_LOST
    if isinstance(error, DBAPIError) and error.connection_invalidated:
        return DatabaseFailureKind.CONNECTION_LOST

    original = error.orig if isinstance(error, DBAPIError) and error.orig is not None else error
    sqlstate = _database_sqlstate(original)
    if sqlstate:
        if sqlstate.startswith('08'):
            return DatabaseFailureKind.CONNECTION_LOST
        if sqlstate in _SQLSTATE_TRANSACTION_CONFLICTS:
            return DatabaseFailureKind.TRANSACTION_CONFLICT

    error_code = _database_error_code(original)
    if error_code in _MYSQL_TRANSACTION_CONFLICT_CODES:
        return DatabaseFailureKind.TRANSACTION_CONFLICT
    if error_code in _MYSQL_CONNECTION_LOST_CODES:
        return DatabaseFailureKind.CONNECTION_LOST

    detail = str(original).lower()
    if 'database is locked' in detail or 'database table is locked' in detail:
        return DatabaseFailureKind.TRANSACTION_CONFLICT
    return DatabaseFailureKind.PERMANENT


def is_retryable_database_error(error: BaseException) -> bool:
    return classify_database_error(error) is not DatabaseFailureKind.PERMANENT


def is_database_transaction_conflict(error: BaseException) -> bool:
    return classify_database_error(error) is DatabaseFailureKind.TRANSACTION_CONFLICT


def _database_sqlstate(error: BaseException) -> str | None:
    for attribute in ('sqlstate', 'pgcode'):
        value = getattr(error, attribute, None)
        if isinstance(value, str) and value:
            return value.upper()
    return None


def _database_error_code(error: BaseException) -> int | None:
    value = getattr(error, 'errno', None)
    if isinstance(value, int):
        return value
    for item in getattr(error, 'args', ()):
        if isinstance(item, int):
            return item
        if isinstance(item, str) and item.isdigit():
            return int(item)
    return None


def create_engine(config: DistributedConfig) -> AsyncEngine:
    database_url = config.read_database_url()
    if not database_url:
        raise ValueError('NEXUS_DIST_DATABASE_URL or NEXUS_DIST_DATABASE_URL_FILE is required')
    engine = create_async_engine(database_url, **_engine_kwargs(config, database_url=database_url))
    if config.slow_sql_log_threshold_ms > 0:
        _install_slow_sql_logging(engine, threshold_ms=config.slow_sql_log_threshold_ms)
    return engine


def _engine_kwargs(config: DistributedConfig, *, database_url: str | URL | None = None) -> dict:
    kwargs = {'pool_pre_ping': True}
    if database_url is not None and make_url(database_url).get_backend_name() == 'mysql':
        kwargs['isolation_level'] = 'READ COMMITTED'
    if config.db_pool_size is not None:
        kwargs['pool_size'] = config.db_pool_size
    if config.db_max_overflow is not None:
        kwargs['max_overflow'] = config.db_max_overflow
    if config.db_pool_timeout is not None:
        kwargs['pool_timeout'] = config.db_pool_timeout
    if config.db_pool_recycle is not None:
        kwargs['pool_recycle'] = config.db_pool_recycle
    return kwargs


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker:
    from nexus.distributed.observability.database import ObservedAsyncSession

    return async_sessionmaker(engine, expire_on_commit=False, class_=ObservedAsyncSession)


async def run_idempotent_transaction(
    session_factory: async_sessionmaker,
    operation: Callable[[Any], Awaitable[ResultT]],
    *,
    conflict_markers: tuple[str, ...],
) -> ResultT:
    for attempt in range(2):
        async with session_factory() as session:
            try:
                result = await operation(session)
                await session.commit()
                return result
            except IntegrityError as exc:
                await session.rollback()
                detail = str(exc.orig or exc).lower()
                if attempt or not any(marker.lower() in detail for marker in conflict_markers):
                    raise
            except TransactionRetryError:
                await session.rollback()
                if attempt:
                    raise
    raise AssertionError('unreachable')


def _install_slow_sql_logging(engine: AsyncEngine, *, threshold_ms: float) -> None:
    @event.listens_for(engine.sync_engine, 'before_cursor_execute')
    def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):  # noqa: ARG001
        conn.info.setdefault('nexus_dist_sql_start_times', []).append(time.perf_counter())

    @event.listens_for(engine.sync_engine, 'after_cursor_execute')
    def after_cursor_execute(conn, cursor, statement, parameters, context, executemany):  # noqa: ARG001
        elapsed_ms = _pop_sql_elapsed_ms(conn)
        if elapsed_ms is None:
            return
        if elapsed_ms < threshold_ms:
            return
        logger.warning(
            'Distributed slow SQL elapsed_ms=%.3f rowcount=%s statement=%r',
            elapsed_ms,
            getattr(cursor, 'rowcount', None),
            _logged_sql_statement(statement),
        )

    @event.listens_for(engine.sync_engine, 'handle_error')
    def handle_error(exception_context):
        conn = exception_context.connection
        if conn is None:
            return
        elapsed_ms = _pop_sql_elapsed_ms(conn)
        if elapsed_ms is None or elapsed_ms < threshold_ms:
            return
        error = exception_context.original_exception
        logger.warning(
            'Distributed SQL error elapsed_ms=%.3f error_type=%s error_code=%s statement=%r',
            elapsed_ms,
            type(error).__name__,
            _db_error_code(error),
            _logged_sql_statement(exception_context.statement or ''),
        )


def _pop_sql_elapsed_ms(conn) -> float | None:
    start_times = conn.info.get('nexus_dist_sql_start_times')
    if not start_times:
        return None
    return (time.perf_counter() - start_times.pop()) * 1000


def _db_error_code(error: BaseException) -> object:
    code = getattr(error, 'errno', None)
    if code is not None:
        return code
    args = getattr(error, 'args', ())
    return args[0] if args else None


def _logged_sql_statement(statement: str) -> str:
    normalized = ' '.join(statement.split())
    if len(normalized) <= _MAX_LOGGED_SQL_LENGTH:
        return normalized
    return f'{normalized[:_MAX_LOGGED_SQL_LENGTH]}...'


async def init_database(config: DistributedConfig) -> None:
    await _ensure_mysql_database(config)
    await upgrade_database(config)


async def upgrade_database(config: DistributedConfig) -> None:
    engine = create_engine(config)
    try:
        async with engine.connect() as conn:
            await conn.run_sync(_run_migrations, commit=True)
    finally:
        await engine.dispose()


async def _ensure_mysql_database(config: DistributedConfig) -> None:
    database_url = config.read_database_url()
    if not database_url:
        raise ValueError('NEXUS_DIST_DATABASE_URL or NEXUS_DIST_DATABASE_URL_FILE is required')

    url = make_url(database_url)
    if url.get_backend_name() != 'mysql':
        return

    database = _mysql_database_name(url)
    server_url = URL.create(
        drivername=url.drivername,
        username=url.username,
        password=url.password,
        host=url.host,
        port=url.port,
        database=None,
        query=url.query,
    )
    deadline = time.monotonic() + config.db_wait_timeout
    last_error: BaseException | None = None

    while True:
        engine = create_async_engine(server_url, **_engine_kwargs(config, database_url=server_url))
        try:
            async with engine.connect() as conn:
                await conn.execute(text('SELECT 1'))
                await conn.execute(
                    text(f'CREATE DATABASE IF NOT EXISTS `{database}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci')
                )
            return
        except Exception as exc:
            last_error = exc
            now = time.monotonic()
            if now >= deadline:
                raise TimeoutError(f'database did not become ready: {last_error}') from exc
            await asyncio.sleep(min(1.0, max(0.05, deadline - now)))
        finally:
            await engine.dispose()


def _mysql_database_name(url: URL) -> str:
    if not url.database:
        raise ValueError('NEXUS_DIST_DATABASE_URL must include a MySQL database name')
    if not _MYSQL_DATABASE_RE.fullmatch(url.database):
        raise ValueError(f'Unsafe MySQL database name: {url.database!r}')
    return url.database


def _run_migrations(connection, *, commit: bool = False) -> None:
    current_revision = _current_alembic_revision(connection)
    if current_revision is None and _has_legacy_schema_version_database(connection):
        _set_alembic_revision(connection, INITIAL_REVISION)
        current_revision = INITIAL_REVISION

    for migration in _pending_migrations(current_revision):
        context = MigrationContext.configure(connection)
        op = Operations(context)
        module = importlib.import_module(migration.module)
        if hasattr(module, 'upgrade_batches'):
            for _ in module.upgrade_batches(op):
                if commit:
                    connection.commit()
        else:
            module.upgrade(op)
        _set_alembic_revision(connection, migration.revision)
        if commit:
            connection.commit()


def _pending_migrations(current_revision: str | None):
    if current_revision == HEAD_REVISION:
        return []
    # Preview deployments stamped cloud submit as 0003 before trace carriers
    # became a separate revision. Both following migrations are idempotent.
    if current_revision == LEGACY_CLOUD_SUBMIT_REVISION:
        current_revision = CLEANUP_REVISION
    # The dev deployment has the 0030 schema plus unrelated extension tables.
    # Preserve those tables and apply the idempotent mainline 0031 before 0032.
    if current_revision == '0031_execution_analytics':
        current_revision = '0030_principal_access_grants'
    revisions = [migration.revision for migration in MIGRATIONS]
    if current_revision is None:
        return list(MIGRATIONS)
    if current_revision not in revisions:
        raise ValueError(f'Unsupported distributed database revision: {current_revision}')
    return list(MIGRATIONS[revisions.index(current_revision) + 1 :])


def _has_legacy_schema_version_database(connection) -> bool:
    tables = set(inspect(connection).get_table_names())
    if not {'schema_version', 'executions'} <= tables:
        return False
    version = connection.scalar(text('SELECT version FROM schema_version WHERE version = 1'))
    return version == 1


def _current_alembic_revision(connection) -> str | None:
    if 'alembic_version' not in inspect(connection).get_table_names():
        return None
    return connection.scalar(text('SELECT version_num FROM alembic_version'))


def _set_alembic_revision(connection, revision: str) -> None:
    connection.execute(
        text('CREATE TABLE IF NOT EXISTS alembic_version (version_num VARCHAR(32) NOT NULL PRIMARY KEY)')
    )
    current = connection.scalar(text('SELECT version_num FROM alembic_version'))
    if current is None:
        connection.execute(text('INSERT INTO alembic_version (version_num) VALUES (:revision)'), {'revision': revision})
    else:
        connection.execute(text('UPDATE alembic_version SET version_num = :revision'), {'revision': revision})
