"""Add metadata for self-service Trace database accounts."""

from __future__ import annotations

import sqlalchemy as sa

from alembic.operations import Operations


revision = '0027_trace_database_accounts'
down_revision = '0026_submit_token_app_binding'
branch_labels = None
depends_on = None


def upgrade(op: Operations) -> None:
    if 'distributed_trace_database_accounts' in sa.inspect(op.get_bind()).get_table_names():
        return
    op.create_table(
        'distributed_trace_database_accounts',
        sa.Column('owner_principal_id', sa.String(length=64), nullable=False),
        sa.Column('backend_name', sa.String(length=64), nullable=False),
        sa.Column('db_username', sa.String(length=64), nullable=False),
        sa.Column('status', sa.String(length=32), nullable=False),
        sa.Column('credential_version', sa.BigInteger(), nullable=False, server_default='0'),
        sa.Column('operation_id', sa.String(length=64), nullable=True),
        sa.Column('operation_type', sa.String(length=32), nullable=True),
        sa.Column('operation_expires_at', sa.DateTime(), nullable=True),
        sa.Column('last_error_code', sa.String(length=128), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('disabled_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(
            ['owner_principal_id'],
            ['distributed_principals.principal_id'],
            ondelete='CASCADE',
        ),
        sa.PrimaryKeyConstraint('owner_principal_id', 'backend_name'),
        sa.UniqueConstraint(
            'backend_name',
            'db_username',
            name='uq_trace_database_accounts_backend_username',
        ),
    )
    op.create_index(
        'idx_trace_database_accounts_status_updated',
        'distributed_trace_database_accounts',
        ['status', 'updated_at'],
        unique=False,
    )


def downgrade(op: Operations) -> None:
    if 'distributed_trace_database_accounts' in sa.inspect(op.get_bind()).get_table_names():
        op.drop_table('distributed_trace_database_accounts')
