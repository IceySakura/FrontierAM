from __future__ import annotations

import re

from typing import Protocol


_SQL_IDENTIFIER_PATTERN = re.compile(r'[A-Za-z0-9_]+')
_ACCOUNT_NAME_PATTERN = re.compile(r'nxs_ro_[0-9a-f]{24}')


class TraceDatabaseAccountProvider(Protocol):
    async def set_readonly_account(self, *, username: str, password: str, database: str, table_name: str) -> None: ...

    async def disable_account(self, *, username: str) -> None: ...

    async def restore_account(self, *, username: str, database: str, table_name: str) -> None: ...

    async def reconcile_readonly_account(self, *, username: str, database: str, table_name: str) -> None: ...


def validate_account_name(username: str) -> str:
    if not isinstance(username, str) or _ACCOUNT_NAME_PATTERN.fullmatch(username) is None:
        raise ValueError('invalid trace database username')
    return username


def validate_database_name(database: str) -> str:
    if not isinstance(database, str) or _SQL_IDENTIFIER_PATTERN.fullmatch(database) is None:
        raise ValueError('invalid trace database name')
    return database


def validate_table_name(table_name: str) -> str:
    if not isinstance(table_name, str) or _SQL_IDENTIFIER_PATTERN.fullmatch(table_name) is None:
        raise ValueError('invalid trace database table name')
    return table_name


class TraceDatabaseTarget:
    """Guard for providers pinned to the single database they were constructed for.

    Every provider takes the database name from its caller, so a name other than its own
    means the request was routed to the wrong backend and must not be served. The check
    lives here rather than in each provider because ``TraceDatabaseAccountProvider`` is a
    ``Protocol`` and carries no implementation: three copies of the same three lines were
    free to drift apart on the message an operator reads when routing breaks.
    """

    database: str

    def _require_database(self, database: str) -> None:
        if validate_database_name(database) != self.database:
            raise ValueError('trace database account provider target mismatch')


__all__ = [
    'TraceDatabaseAccountProvider',
    'TraceDatabaseTarget',
    'validate_account_name',
    'validate_database_name',
    'validate_table_name',
]
