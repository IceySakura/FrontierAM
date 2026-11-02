from __future__ import annotations

import sqlalchemy as sa

from alembic.operations import Operations
from sqlalchemy import inspect


revision = '0003_add_trace_carriers'
down_revision = '0002_add_execution_cleanup'


def _add_json_column_if_missing(op: Operations, table_name: str, column_name: str) -> None:
    bind = op.get_bind()
    columns = {column['name'] for column in inspect(bind).get_columns(table_name)}
    if column_name not in columns:
        op.add_column(table_name, sa.Column(column_name, sa.JSON(), nullable=True))


def upgrade(op: Operations) -> None:
    _add_json_column_if_missing(op, 'executions', 'trace_carrier_json')
    _add_json_column_if_missing(op, 'assignments', 'trace_carrier_json')
