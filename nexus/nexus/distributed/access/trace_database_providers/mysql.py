from __future__ import annotations

import asyncio
import contextlib

from nexus.distributed.access.trace_database_providers.base import (
    TraceDatabaseAccountProvider,
    TraceDatabaseTarget,
    validate_account_name,
    validate_database_name,
    validate_table_name,
)
from nexus.runner.trace.tdsql.config import build_db_config


class MySQLTraceDatabaseAccountProvider(TraceDatabaseTarget, TraceDatabaseAccountProvider):
    def __init__(
        self,
        *,
        host: str,
        database: str,
        username: str,
        password: str,
        port: int = 3306,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.db_config = build_db_config(
            host=host,
            port=port,
            user=username,
            password=password,
            db=database,
        )
        self.database = validate_database_name(database)
        self.timeout_seconds = timeout_seconds

    async def set_readonly_account(self, *, username: str, password: str, database: str, table_name: str) -> None:
        account = _account_sql(username)
        self._require_database(database)
        table_name = validate_table_name(table_name)
        await self._execute([
            (f'CREATE USER IF NOT EXISTS {account} IDENTIFIED BY %s', (password,)),
            (f'ALTER USER {account} IDENTIFIED BY %s ACCOUNT UNLOCK', (password,)),
            (f'REVOKE ALL PRIVILEGES, GRANT OPTION FROM {account}', None),
            (f'GRANT SELECT ON `{self.database}`.`{table_name}` TO {account}', None),
        ])

    async def disable_account(self, *, username: str) -> None:
        await self._execute([(f'ALTER USER {_account_sql(username)} ACCOUNT LOCK', None)])

    async def restore_account(self, *, username: str, database: str, table_name: str) -> None:
        await self.reconcile_readonly_account(username=username, database=database, table_name=table_name)
        await self._execute([(f'ALTER USER {_account_sql(username)} ACCOUNT UNLOCK', None)])

    async def reconcile_readonly_account(self, *, username: str, database: str, table_name: str) -> None:
        account = _account_sql(username)
        self._require_database(database)
        table_name = validate_table_name(table_name)
        await self._execute([
            (f'REVOKE ALL PRIVILEGES, GRANT OPTION FROM {account}', None),
            (f'GRANT SELECT ON `{self.database}`.`{table_name}` TO {account}', None),
        ])

    async def _execute(self, statements: list[tuple[str, tuple[str, ...] | None]]) -> None:
        from nexus.runner.trace.tdsql._driver import connect as mysql_connect

        connection = None

        async def run() -> None:
            nonlocal connection
            connection = await mysql_connect(**self.db_config)
            async with connection.cursor() as cursor:
                for sql, params in statements:
                    if params is None:
                        await cursor.execute(sql)
                    else:
                        await cursor.execute(sql, params)
            await connection.commit()

        try:
            await asyncio.wait_for(run(), timeout=self.timeout_seconds)
        except Exception:
            if connection is not None:
                with contextlib.suppress(Exception):
                    await connection.rollback()
            raise
        finally:
            if connection is not None:
                with contextlib.suppress(Exception):
                    connection.close()


def _account_sql(username: str) -> str:
    return f"'{validate_account_name(username)}'@'%'"


__all__ = ['MySQLTraceDatabaseAccountProvider']
