#!/usr/bin/env python3
from __future__ import annotations

import csv
import re
import sqlite3

from pathlib import Path
from typing import Any


_LOAD_DATA_RE = re.compile(
    r'LOAD\s+DATA\s+LOCAL\s+INFILE\s+"(?P<path>[^"]+)"\s+'
    r'INTO\s+TABLE\s+(?P<table>[`"\w]+).*?IGNORE\s+\d+\s+LINES\s*;',
    re.IGNORECASE | re.DOTALL,
)
_DROP_DATABASE_RE = re.compile(r'^\s*CREATE\s+DATABASE\b.*?;\s*', re.IGNORECASE | re.MULTILINE | re.DOTALL)
_USE_DATABASE_RE = re.compile(r'^\s*USE\b.*?;\s*', re.IGNORECASE | re.MULTILINE | re.DOTALL)
_SET_RE = re.compile(r'^\s*SET\b.*?;\s*', re.IGNORECASE | re.MULTILINE | re.DOTALL)
_CREATE_USER_RE = re.compile(r'^\s*CREATE\s+USER\b.*?;\s*', re.IGNORECASE | re.MULTILINE | re.DOTALL)
_GRANT_RE = re.compile(r'^\s*GRANT\b.*?;\s*', re.IGNORECASE | re.MULTILINE | re.DOTALL)
_FLUSH_RE = re.compile(r'^\s*FLUSH\s+PRIVILEGES\b.*?;\s*', re.IGNORECASE | re.MULTILINE | re.DOTALL)
_AUTO_INCREMENT_RE = re.compile(r'\bAUTO_INCREMENT\b', re.IGNORECASE)
_ENGINE_RE = re.compile(r'\)\s*ENGINE\s*=\s*[^;]+;', re.IGNORECASE)
_INT_WIDTH_RE = re.compile(r'\b(?:TINYINT|SMALLINT|MEDIUMINT|INT|INTEGER|BIGINT)\s*\(\s*\d+\s*\)', re.IGNORECASE)
_FLOAT_WIDTH_RE = re.compile(r'\b(?:FLOAT|DOUBLE|DECIMAL)\s*\(\s*\d+(?:\s*,\s*\d+)?\s*\)', re.IGNORECASE)
_TRUNCATE_RE = re.compile(r'\bTRUNCATE\s+TABLE\s+([`"\w]+)\s*;', re.IGNORECASE)
_COMMENT_RE = re.compile(r'^\s*--.*$', re.MULTILINE)


def _connect(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute('PRAGMA journal_mode=WAL')
    connection.execute('PRAGMA synchronous=NORMAL')
    connection.create_function('CONCAT', -1, _concat)
    connection.create_function('CONCAT_WS', -1, _concat_ws)
    connection.create_function('IF', 3, _if_fn)
    connection.create_function('YEAR', 1, _year_fn)
    connection.create_function('RAND', 0, _rand_fn)
    return connection


def _concat(*values: Any) -> str:
    return ''.join('' if value is None else str(value) for value in values)


def _concat_ws(separator: Any, *values: Any) -> str:
    glue = '' if separator is None else str(separator)
    rendered = [str(value) for value in values if value not in {None, ''}]
    return glue.join(rendered)


def _if_fn(condition: Any, true_value: Any, false_value: Any) -> Any:
    return true_value if condition else false_value


def _year_fn(value: Any) -> int | None:
    if value in {None, ''}:
        return None
    text = str(value)
    if len(text) >= 4 and text[:4].isdigit():
        return int(text[:4])
    return None


def _rand_fn() -> float:
    return 0.5


def _normalized_sql(script: str) -> str:
    sql = script.replace('\r\n', '\n')
    sql = _COMMENT_RE.sub('', sql)
    sql = _DROP_DATABASE_RE.sub('', sql)
    sql = _USE_DATABASE_RE.sub('', sql)
    sql = _SET_RE.sub('', sql)
    sql = _CREATE_USER_RE.sub('', sql)
    sql = _GRANT_RE.sub('', sql)
    sql = _FLUSH_RE.sub('', sql)
    sql = _AUTO_INCREMENT_RE.sub('', sql)
    sql = _INT_WIDTH_RE.sub(lambda match: match.group(0).split('(', 1)[0], sql)
    sql = _FLOAT_WIDTH_RE.sub(lambda match: match.group(0).split('(', 1)[0], sql)
    sql = _TRUNCATE_RE.sub(lambda match: f'DELETE FROM {match.group(1)};', sql)
    return _ENGINE_RE.sub(');', sql)


def apply_schema_sql(db_path: str | Path, sql_text: str) -> None:
    script = _normalized_sql(sql_text)
    with _connect(db_path) as connection:
        connection.executescript(script)


def _table_types(connection: sqlite3.Connection, table_name: str) -> dict[str, str]:
    rows = connection.execute(f'PRAGMA table_info(`{table_name}`)').fetchall()
    return {str(row['name']): str(row['type'] or '').upper() for row in rows}


def _cast_value(raw: str, declared_type: str) -> Any:
    if raw == '':
        return None
    dtype = declared_type.upper()
    if 'INT' in dtype:
        try:
            return int(raw)
        except ValueError:
            return raw
    if any(token in dtype for token in ('REAL', 'FLOA', 'DOUB', 'NUMERIC', 'DECIMAL')):
        try:
            return float(raw)
        except ValueError:
            return raw
    return raw


def _normalize_identifier(value: str) -> str:
    text = re.sub(r'[^0-9A-Za-z]+', '_', str(value).strip())
    text = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', text)
    text = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', text)
    text = re.sub(r'_+', '_', text).strip('_')
    return text.lower()


def _resolved_csv_columns(csv_columns: list[str], table_types: dict[str, str]) -> list[tuple[str, str]]:
    normalized_lookup = {_normalize_identifier(column): column for column in table_types}
    resolved: list[tuple[str, str]] = []
    for csv_column in csv_columns:
        if csv_column in table_types:
            resolved.append((csv_column, csv_column))
            continue
        normalized = _normalize_identifier(csv_column)
        target = normalized_lookup.get(normalized)
        if target is None:
            raise sqlite3.OperationalError(
                f'table column for CSV header {csv_column!r} is missing; available columns: {sorted(table_types)}'
            )
        resolved.append((csv_column, target))
    return resolved


def load_csv_file(db_path: str | Path, table_name: str, csv_path: str | Path) -> None:
    with _connect(db_path) as connection:
        types = _table_types(connection, table_name)
        with Path(csv_path).open('r', encoding='utf-8', newline='') as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
        connection.execute(f'DELETE FROM `{table_name}`')
        if not rows:
            return
        resolved_columns = _resolved_csv_columns(list(rows[0].keys()), types)
        target_columns = [target for _, target in resolved_columns]
        placeholders = ', '.join('?' for _ in target_columns)
        quoted_columns = ', '.join(f'`{column}`' for column in target_columns)
        payload = [
            tuple(_cast_value(row.get(source, ''), types.get(target, 'TEXT')) for source, target in resolved_columns)
            for row in rows
        ]
        connection.executemany(
            f'INSERT INTO `{table_name}` ({quoted_columns}) VALUES ({placeholders})',
            payload,
        )


def load_csv_directory(db_path: str | Path, csv_dir: str | Path) -> None:
    directory = Path(csv_dir)
    for path in sorted(directory.glob('*.csv')):
        load_csv_file(db_path, path.stem, path)


def apply_mysql_script(db_path: str | Path, sql_text: str, *, csv_root: str | Path | None = None) -> None:
    load_ops: list[tuple[str, str]] = []

    def _extract(match: re.Match[str]) -> str:
        load_ops.append((match.group('table').strip('`"'), match.group('path')))
        return ''

    script = _LOAD_DATA_RE.sub(_extract, sql_text.replace('\r\n', '\n'))
    script = _normalized_sql(script)
    with _connect(db_path) as connection:
        if script.strip():
            connection.executescript(script)
        if csv_root is not None:
            csv_base = Path(csv_root)
            for table_name, load_path in load_ops:
                relative = load_path.removeprefix('/csv/').removeprefix('./')
                load_csv_file(db_path, table_name, csv_base / relative)


def execute_query(db_path: str | Path, sql: str) -> tuple[list[str], list[tuple[Any, ...]]]:
    with _connect(db_path) as connection:
        cursor = connection.execute(sql)
        if cursor.description is None:
            connection.commit()
            return [], []
        column_names = [str(item[0]) for item in cursor.description]
        rows = [tuple(row) for row in cursor.fetchall()]
        return column_names, rows


def schema_mapping(db_path: str | Path) -> dict[str, dict[str, str]]:
    with _connect(db_path) as connection:
        tables = [
            str(row['name'])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
        ]
        return {table_name: _table_types(connection, table_name) for table_name in tables}


__all__ = [
    'apply_mysql_script',
    'apply_schema_sql',
    'execute_query',
    'load_csv_directory',
    'load_csv_file',
    'schema_mapping',
]
