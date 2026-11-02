from __future__ import annotations

import pytest

from sqlalchemy.exc import OperationalError

from nexus.distributed.persistence.db import DatabaseFailureKind, classify_database_error


class _DatabaseError(Exception):
    pass


@pytest.mark.parametrize(
    ('code', 'kind'),
    [
        (1205, DatabaseFailureKind.TRANSACTION_CONFLICT),
        (1213, DatabaseFailureKind.TRANSACTION_CONFLICT),
        (1053, DatabaseFailureKind.CONNECTION_LOST),
        (1158, DatabaseFailureKind.CONNECTION_LOST),
        (1159, DatabaseFailureKind.CONNECTION_LOST),
        (1160, DatabaseFailureKind.CONNECTION_LOST),
        (1161, DatabaseFailureKind.CONNECTION_LOST),
        (2003, DatabaseFailureKind.CONNECTION_LOST),
        (2006, DatabaseFailureKind.CONNECTION_LOST),
        (2013, DatabaseFailureKind.CONNECTION_LOST),
        (2055, DatabaseFailureKind.CONNECTION_LOST),
        (1040, DatabaseFailureKind.PERMANENT),
        (1045, DatabaseFailureKind.PERMANENT),
        (1049, DatabaseFailureKind.PERMANENT),
        (1142, DatabaseFailureKind.PERMANENT),
        (1203, DatabaseFailureKind.PERMANENT),
        (2002, DatabaseFailureKind.PERMANENT),
        (2005, DatabaseFailureKind.PERMANENT),
        (2012, DatabaseFailureKind.PERMANENT),
        (2020, DatabaseFailureKind.PERMANENT),
        (2026, DatabaseFailureKind.PERMANENT),
    ],
)
def test_classify_mysql_database_errors(code: int, kind: DatabaseFailureKind) -> None:
    error = OperationalError('statement', {}, _DatabaseError(code, 'database failure'))

    assert classify_database_error(error) is kind


@pytest.mark.parametrize(
    ('sqlstate', 'kind'),
    [
        ('40001', DatabaseFailureKind.TRANSACTION_CONFLICT),
        ('40P01', DatabaseFailureKind.TRANSACTION_CONFLICT),
        ('08006', DatabaseFailureKind.CONNECTION_LOST),
        ('42P01', DatabaseFailureKind.PERMANENT),
    ],
)
def test_classify_sqlstate_database_errors(sqlstate: str, kind: DatabaseFailureKind) -> None:
    original = _DatabaseError('database failure')
    original.sqlstate = sqlstate
    error = OperationalError('statement', {}, original)

    assert classify_database_error(error) is kind


def test_classify_invalidated_connection_as_connection_lost() -> None:
    error = OperationalError(
        'statement',
        {},
        _DatabaseError('connection invalidated'),
        connection_invalidated=True,
    )

    assert classify_database_error(error) is DatabaseFailureKind.CONNECTION_LOST


@pytest.mark.parametrize('message', ['database is locked', 'database table is locked'])
def test_classify_sqlite_lock_as_transaction_conflict(message: str) -> None:
    error = OperationalError('statement', {}, _DatabaseError(message))

    assert classify_database_error(error) is DatabaseFailureKind.TRANSACTION_CONFLICT
