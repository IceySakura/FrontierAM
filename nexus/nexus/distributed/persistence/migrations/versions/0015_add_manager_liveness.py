from __future__ import annotations

import sqlalchemy as sa

from alembic.operations import Operations
from sqlalchemy import inspect


revision = '0015_add_manager_liveness'
down_revision = '0014_rename_scheduler_leases'


def upgrade(op: Operations) -> None:
    bind = op.get_bind()
    tables = set(inspect(bind).get_table_names())
    if 'instance_managers' not in tables:
        return
    if 'instance_manager_liveness' not in tables:
        op.create_table(
            'instance_manager_liveness',
            sa.Column('manager_id', sa.String(length=128), nullable=False),
            sa.Column('session_id', sa.String(length=64), nullable=False),
            sa.Column('capacity_total', sa.Integer(), nullable=False),
            sa.Column('observed_capacity_used', sa.Integer(), nullable=False, server_default='0'),
            sa.Column('last_heartbeat_at', sa.DateTime(), nullable=False),
            sa.Column('fenced_at', sa.DateTime(), nullable=True),
            sa.Column('updated_at', sa.DateTime(), nullable=False),
            sa.ForeignKeyConstraint(
                ['manager_id'],
                ['instance_managers.manager_id'],
                ondelete='CASCADE',
            ),
            sa.PrimaryKeyConstraint('manager_id'),
        )
        op.create_index(
            'idx_instance_manager_liveness_heartbeat',
            'instance_manager_liveness',
            ['last_heartbeat_at', 'manager_id'],
            unique=False,
        )
        op.create_index(
            'idx_instance_manager_liveness_fenced',
            'instance_manager_liveness',
            ['fenced_at', 'last_heartbeat_at'],
            unique=False,
        )

    op.execute(
        sa.text(
            'INSERT INTO instance_manager_liveness '
            '(manager_id, session_id, capacity_total, observed_capacity_used, '
            'last_heartbeat_at, fenced_at, updated_at) '
            'SELECT manager_id, session_id, capacity_total, observed_capacity_used, '
            'last_heartbeat_at, NULL, updated_at '
            'FROM instance_managers '
            'WHERE manager_id NOT IN (SELECT manager_id FROM instance_manager_liveness)'
        )
    )


def downgrade(op: Operations) -> None:
    if 'instance_manager_liveness' in set(inspect(op.get_bind()).get_table_names()):
        op.drop_table('instance_manager_liveness')
