from __future__ import annotations

import re

from datetime import date, timedelta
from typing import Literal


SHARED_TRACE_TABLE = 'trajectory_span_events'
PRINCIPAL_TRACE_TABLE_PREFIX = f'{SHARED_TRACE_TABLE}_pr_'
_PRINCIPAL_SCOPE_ID = r'[0-9a-f]{32}'
_PRINCIPAL_SCOPE_ID_PATTERN = re.compile(_PRINCIPAL_SCOPE_ID)
# Exported because the name is derived, not literal: every reader that recognises a
# principal trace table has to agree with the writer that produced it, and a spelling of
# its own would keep matching a table this module had stopped creating.
PRINCIPAL_TRACE_TABLE_PATTERN = re.compile(f'{re.escape(PRINCIPAL_TRACE_TABLE_PREFIX)}{_PRINCIPAL_SCOPE_ID}')

# ``tdsql`` targets TDSQL MySQL (DCDB): manual sharding declared with the proprietary
# ``shardkey=`` table option. ``boundless`` targets TDSQL Boundless (formerly TDStore):
# transparent distribution using standard MySQL partitioning, so the table carries a
# ``parent_created_at`` RANGE partition key plus a KEY subpartition over ``run_id``.
TDSQL_DIALECT = 'tdsql'
BOUNDLESS_DIALECT = 'boundless'
TraceSchemaDialect = Literal['tdsql', 'boundless']
_DIALECTS = frozenset({TDSQL_DIALECT, BOUNDLESS_DIALECT})

# A multiple of the instance's 6 nodes, so subpartitions spread evenly if the engine maps
# them to nodes by index; 12 rather than 6 leaves room for one node-count doubling. Only
# applied at CREATE TABLE time, so changing it later does not reshape existing tables.
DEFAULT_BOUNDLESS_SUBPARTITIONS = 12
DEFAULT_BOUNDLESS_PARTITION_INTERVAL = 'DAY, 1'
_PARTITION_INTERVAL_PATTERN = re.compile(r'(?i)^(YEAR|QUARTER|MONTH|WEEK|DAY|HOUR)\s*,\s*[1-9][0-9]{0,3}$')


def validate_dialect(dialect: str) -> str:
    if dialect not in _DIALECTS:
        raise ValueError(f'unsupported TDSQL trace schema dialect: {dialect!r}')
    return dialect


def validate_subpartitions(subpartitions: int) -> int:
    if isinstance(subpartitions, bool) or not isinstance(subpartitions, int) or not 1 <= subpartitions <= 4096:
        raise ValueError('TDSQL trace subpartitions must be an integer between 1 and 4096')
    return subpartitions


def validate_partition_interval(partition_interval: str) -> str:
    if not isinstance(partition_interval, str) or _PARTITION_INTERVAL_PATTERN.fullmatch(partition_interval) is None:
        raise ValueError('TDSQL trace partition interval must look like "DAY, 1"')
    return partition_interval


SHARED_TRACE_TABLE_STATEMENT = f"""
CREATE TABLE IF NOT EXISTS {SHARED_TRACE_TABLE} (
    id BIGINT NOT NULL AUTO_INCREMENT,
    run_id VARCHAR(255) NOT NULL DEFAULT '',
    run_span_id VARCHAR(255) NOT NULL DEFAULT '',
    trajectory_group_id VARCHAR(255) NOT NULL,
    record_id BIGINT NOT NULL DEFAULT 0,
    event_type VARCHAR(16) NOT NULL CHECK (event_type IN ('start', 'update', 'end')),
    time_unix_nano BIGINT NOT NULL DEFAULT 0,
    trace_id VARCHAR(255) NOT NULL DEFAULT '',
    span_id VARCHAR(255) NOT NULL DEFAULT '',
    parent_span_id VARCHAR(255),
    event_data LONGTEXT NOT NULL,
    created_at DATETIME(6) DEFAULT CURRENT_TIMESTAMP(6),
    PRIMARY KEY (id, trajectory_group_id),
    UNIQUE KEY uq_traj_span_events_record (trajectory_group_id, run_span_id, record_id),
    INDEX idx_traj_span_events_run_span_record (run_id, run_span_id, record_id),
    INDEX idx_traj_span_events_group_id (trajectory_group_id),
    INDEX idx_traj_span_events_run_span_id (run_span_id),
    INDEX idx_traj_span_events_record_id (record_id),
    INDEX idx_traj_span_events_event_type (event_type),
    INDEX idx_traj_span_events_trace_id (trace_id),
    INDEX idx_traj_span_events_span_id (span_id),
    INDEX idx_traj_span_events_parent_span_id (parent_span_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 shardkey=trajectory_group_id
"""


def principal_trace_table_name(principal_scope_id: str) -> str:
    if not isinstance(principal_scope_id, str) or _PRINCIPAL_SCOPE_ID_PATTERN.fullmatch(principal_scope_id) is None:
        raise ValueError('invalid TDSQL principal_scope_id')
    return f'{PRINCIPAL_TRACE_TABLE_PREFIX}{principal_scope_id}'


def principal_scope_id_from_table(table: str) -> str:
    """The owner principal that a principal trace table belongs to.

    The inverse of :func:`principal_trace_table_name`, and here for the same reason: a
    caller stripping the prefix itself has no way to reject a name this module never
    produced, and ``removeprefix`` on an unrelated table quietly returns the whole name as
    though it were a scope id.
    """
    if not isinstance(table, str) or PRINCIPAL_TRACE_TABLE_PATTERN.fullmatch(table) is None:
        raise ValueError('invalid TDSQL principal trace table')
    return table[len(PRINCIPAL_TRACE_TABLE_PREFIX) :]


def principal_trace_check_constraint_name(principal_scope_id: str) -> str:
    principal_trace_table_name(principal_scope_id)
    return f'chk_tse_pr_{principal_scope_id}'


def _boundless_partition_clause(*, subpartitions: int, initial_partition_date: date) -> str:
    """Seed one history catch-all partition plus the day that owns ``initial_partition_date``.

    ``SET INTERVAL`` only extends an existing RANGE layout, so the table must ship with at
    least one bound before out-of-range inserts can auto-create the following days.
    """
    boundary = initial_partition_date.strftime('%Y-%m-%d')
    next_boundary = (initial_partition_date + timedelta(days=1)).strftime('%Y-%m-%d')
    suffix = initial_partition_date.strftime('%Y%m%d')
    return (
        'PARTITION BY RANGE COLUMNS (parent_created_at)\n'
        f'SUBPARTITION BY KEY (run_id) SUBPARTITIONS {subpartitions} (\n'
        f"    PARTITION p_before_{suffix} VALUES LESS THAN ('{boundary} 00:00:00'),\n"
        f"    PARTITION p{suffix} VALUES LESS THAN ('{next_boundary} 00:00:00')\n"
        ')'
    )


def _tdsql_principal_trace_table_statement(principal_scope_id: str) -> str:
    table_name = principal_trace_table_name(principal_scope_id)
    check_constraint_name = principal_trace_check_constraint_name(principal_scope_id)
    return f"""
CREATE TABLE IF NOT EXISTS {table_name} (
    id BIGINT NOT NULL AUTO_INCREMENT,
    run_id VARCHAR(64) NOT NULL DEFAULT '',
    run_span_id VARCHAR(64) NOT NULL DEFAULT '',
    parent_execution_id VARCHAR(64) NULL,
    record_id BIGINT NOT NULL DEFAULT 0,
    event_type VARCHAR(16) NOT NULL,
    time_unix_nano BIGINT NOT NULL DEFAULT 0,
    trace_id VARCHAR(64) NOT NULL DEFAULT '',
    span_id VARCHAR(64) NOT NULL DEFAULT '',
    parent_span_id VARCHAR(64),
    event_data LONGTEXT NOT NULL,
    created_at DATETIME(6) DEFAULT CURRENT_TIMESTAMP(6),
    CONSTRAINT {check_constraint_name} CHECK (event_type IN ('start', 'update', 'end')),
    PRIMARY KEY (id, run_id),
    UNIQUE KEY uq_traj_span_events_record (run_id, run_span_id, record_id),
    INDEX idx_traj_span_events_run_span_record (run_id, run_span_id, record_id),
    INDEX idx_traj_span_events_parent_cursor
        (parent_execution_id, time_unix_nano, run_id, run_span_id, record_id),
    INDEX idx_traj_span_events_run_span_id (run_span_id),
    INDEX idx_traj_span_events_record_id (record_id),
    INDEX idx_traj_span_events_event_type (event_type),
    INDEX idx_traj_span_events_trace_id (trace_id),
    INDEX idx_traj_span_events_span_id (span_id),
    INDEX idx_traj_span_events_parent_span_id (parent_span_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 shardkey=run_id
"""


def _boundless_principal_trace_table_statement(
    principal_scope_id: str,
    *,
    subpartitions: int,
    initial_partition_date: date,
) -> str:
    """Boundless keeps standard MySQL DDL: no ``shardkey=``, no explicit ENGINE.

    ``parent_created_at`` joins the primary and unique keys because MySQL requires every
    partitioning column to be part of them, and the storage engine is left to the instance
    default (verified to resolve to ROCKSDB on ``tdsql3-49qrr9xo6y``).
    """
    table_name = principal_trace_table_name(principal_scope_id)
    check_constraint_name = principal_trace_check_constraint_name(principal_scope_id)
    partition_clause = _boundless_partition_clause(
        subpartitions=subpartitions,
        initial_partition_date=initial_partition_date,
    )
    return f"""
CREATE TABLE IF NOT EXISTS {table_name} (
    id BIGINT NOT NULL AUTO_INCREMENT,
    run_id VARCHAR(64) NOT NULL DEFAULT '',
    run_span_id VARCHAR(64) NOT NULL DEFAULT '',
    parent_execution_id VARCHAR(64) NULL,
    parent_created_at DATETIME(6) NOT NULL,
    record_id BIGINT NOT NULL DEFAULT 0,
    event_type VARCHAR(16) NOT NULL,
    time_unix_nano BIGINT NOT NULL DEFAULT 0,
    trace_id VARCHAR(64) NOT NULL DEFAULT '',
    span_id VARCHAR(64) NOT NULL DEFAULT '',
    parent_span_id VARCHAR(64),
    event_data LONGTEXT NOT NULL,
    created_at DATETIME(6) DEFAULT CURRENT_TIMESTAMP(6),
    CONSTRAINT {check_constraint_name} CHECK (event_type IN ('start', 'update', 'end')),
    PRIMARY KEY (id, run_id, parent_created_at),
    UNIQUE KEY uq_traj_span_events_record (run_id, run_span_id, record_id, parent_created_at),
    INDEX idx_traj_span_events_parent_cursor
        (parent_execution_id, time_unix_nano, run_id, run_span_id, record_id),
    INDEX idx_traj_span_events_run_span_id (run_span_id),
    INDEX idx_traj_span_events_record_id (record_id),
    INDEX idx_traj_span_events_event_type (event_type),
    INDEX idx_traj_span_events_trace_id (trace_id),
    INDEX idx_traj_span_events_span_id (span_id),
    INDEX idx_traj_span_events_parent_span_id (parent_span_id),
    INDEX idx_traj_span_events_time_unix_nano (time_unix_nano)
) DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci
{partition_clause}
"""


def principal_trace_table_statement(
    principal_scope_id: str,
    *,
    dialect: str = TDSQL_DIALECT,
    subpartitions: int = DEFAULT_BOUNDLESS_SUBPARTITIONS,
    initial_partition_date: date | None = None,
) -> str:
    if validate_dialect(dialect) == TDSQL_DIALECT:
        return _tdsql_principal_trace_table_statement(principal_scope_id)
    return _boundless_principal_trace_table_statement(
        principal_scope_id,
        subpartitions=validate_subpartitions(subpartitions),
        initial_partition_date=initial_partition_date or date.today(),
    )


def principal_trace_interval_statement(
    principal_scope_id: str,
    *,
    partition_interval: str = DEFAULT_BOUNDLESS_PARTITION_INTERVAL,
) -> str:
    """Enable Boundless auto-extension so out-of-range inserts create the next partition.

    Must run right after the CREATE TABLE; Boundless echoes it back inside the versioned
    ``/*B!210604 INTERVAL(DAY, 1) */`` comment.
    """
    table_name = principal_trace_table_name(principal_scope_id)
    return f'ALTER TABLE {table_name} SET INTERVAL({validate_partition_interval(partition_interval)})'


def principal_trace_table_upgrade_statement(principal_scope_id: str) -> str:
    table_name = principal_trace_table_name(principal_scope_id)
    return f'ALTER TABLE {table_name} MODIFY COLUMN parent_execution_id VARCHAR(64) NULL'


# Persisted metadata and external imports may still use the original Batch-specific name.
principal_batch_trace_table_name = principal_trace_table_name


def span_event_insert_sql(principal_scope_id: str | None = None, *, dialect: str = TDSQL_DIALECT) -> str:
    validate_dialect(dialect)
    table_name = SHARED_TRACE_TABLE if principal_scope_id is None else principal_trace_table_name(principal_scope_id)
    if principal_scope_id is None:
        if dialect == BOUNDLESS_DIALECT:
            raise ValueError('boundless trace dialect requires principal storage scope')
        columns = (
            'run_id, run_span_id, trajectory_group_id, record_id, event_type, time_unix_nano, '
            'trace_id, span_id, parent_span_id, event_data'
        )
    elif dialect == BOUNDLESS_DIALECT:
        columns = (
            'run_id, run_span_id, parent_execution_id, parent_created_at, record_id, event_type, '
            'time_unix_nano, trace_id, span_id, parent_span_id, event_data'
        )
    else:
        columns = (
            'run_id, run_span_id, parent_execution_id, record_id, event_type, time_unix_nano, '
            'trace_id, span_id, parent_span_id, event_data'
        )
    placeholders = ', '.join(['%s'] * len(columns.split(',')))
    return f'INSERT INTO {table_name}\n    ({columns})\nVALUES ({placeholders})\nON DUPLICATE KEY UPDATE id = id'


SCHEMA_STATEMENTS = [SHARED_TRACE_TABLE_STATEMENT]


def schema_statements(dialect: str = TDSQL_DIALECT) -> list[str]:
    """Bootstrap statements a writer runs once per database.

    Boundless backends only serve the principal scope, and those tables are created lazily by
    ``prepare_scope``; the shared table would need a partition key it has no business value for,
    so it is intentionally not created there.
    """
    if validate_dialect(dialect) == BOUNDLESS_DIALECT:
        return []
    return list(SCHEMA_STATEMENTS)


__all__ = [
    'BOUNDLESS_DIALECT',
    'DEFAULT_BOUNDLESS_PARTITION_INTERVAL',
    'DEFAULT_BOUNDLESS_SUBPARTITIONS',
    'PRINCIPAL_TRACE_TABLE_PATTERN',
    'PRINCIPAL_TRACE_TABLE_PREFIX',
    'SCHEMA_STATEMENTS',
    'SHARED_TRACE_TABLE',
    'SHARED_TRACE_TABLE_STATEMENT',
    'TDSQL_DIALECT',
    'TraceSchemaDialect',
    'principal_batch_trace_table_name',
    'principal_scope_id_from_table',
    'principal_trace_check_constraint_name',
    'principal_trace_interval_statement',
    'principal_trace_table_name',
    'principal_trace_table_statement',
    'principal_trace_table_upgrade_statement',
    'schema_statements',
    'span_event_insert_sql',
    'validate_dialect',
    'validate_partition_interval',
    'validate_subpartitions',
]
