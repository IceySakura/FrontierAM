from __future__ import annotations

import sqlalchemy as sa

from alembic.operations import Operations
from sqlalchemy import inspect


revision = '0002_add_execution_cleanup'
down_revision = '0001_initial'


def upgrade(op: Operations) -> None:
    bind = op.get_bind()
    columns = {column['name'] for column in inspect(bind).get_columns('executions')}
    if 'cleanup_done_at' not in columns:
        op.add_column('executions', sa.Column('cleanup_done_at', sa.DateTime(), nullable=True))

    indexes = {index['name'] for index in inspect(bind).get_indexes('executions')}
    if 'idx_executions_cleanup' not in indexes:
        op.create_index('idx_executions_cleanup', 'executions', ['status', 'cleanup_done_at'])
