"""Add shard ownership to batch lifecycle rows.

Revision ID: 0016_add_batch_run_shard
Revises: 0015_add_manager_liveness
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic.operations import Operations
from sqlalchemy import inspect


revision = '0016_add_batch_run_shard'
down_revision = '0015_add_manager_liveness'
branch_labels = None
depends_on = None


def upgrade(op: Operations) -> None:
    bind = op.get_bind()
    columns = {column['name'] for column in inspect(bind).get_columns('distributed_batch_runs')}
    if 'shard_id' not in columns:
        op.add_column(
            'distributed_batch_runs',
            sa.Column('shard_id', sa.Integer(), nullable=True),
        )
        op.execute(
            sa.text(
                'UPDATE distributed_batch_runs '
                'SET shard_id = ('
                'SELECT executions.shard_id '
                'FROM executions '
                'WHERE executions.execution_id = distributed_batch_runs.parent_execution_id'
                ')'
            )
        )
        missing = int(bind.scalar(sa.text('SELECT COUNT(*) FROM distributed_batch_runs WHERE shard_id IS NULL')) or 0)
        if missing:
            raise RuntimeError('batch run shard backfill found missing parent executions')
        with op.batch_alter_table('distributed_batch_runs') as batch:
            batch.alter_column('shard_id', existing_type=sa.Integer(), nullable=False)

    indexes = {index['name'] for index in inspect(bind).get_indexes('distributed_batch_runs')}
    if 'idx_batch_runs_lifecycle_claim' in indexes:
        op.drop_index('idx_batch_runs_lifecycle_claim', table_name='distributed_batch_runs')
    if 'idx_batch_runs_shard_lifecycle_claim' not in indexes:
        op.create_index(
            'idx_batch_runs_shard_lifecycle_claim',
            'distributed_batch_runs',
            [
                'shard_id',
                'control_state',
                'last_reconciled_at',
                'lifecycle_lease_expires_at',
                'parent_execution_id',
            ],
            unique=False,
        )


def downgrade(op: Operations) -> None:
    bind = op.get_bind()
    columns = {column['name'] for column in inspect(bind).get_columns('distributed_batch_runs')}
    if 'shard_id' not in columns:
        return
    indexes = {index['name'] for index in inspect(bind).get_indexes('distributed_batch_runs')}
    if 'idx_batch_runs_shard_lifecycle_claim' in indexes:
        op.drop_index('idx_batch_runs_shard_lifecycle_claim', table_name='distributed_batch_runs')
    if 'idx_batch_runs_lifecycle_claim' not in indexes:
        op.create_index(
            'idx_batch_runs_lifecycle_claim',
            'distributed_batch_runs',
            [
                'control_state',
                'last_reconciled_at',
                'lifecycle_lease_expires_at',
                'parent_execution_id',
            ],
            unique=False,
        )
    with op.batch_alter_table('distributed_batch_runs') as batch:
        batch.drop_column('shard_id')
