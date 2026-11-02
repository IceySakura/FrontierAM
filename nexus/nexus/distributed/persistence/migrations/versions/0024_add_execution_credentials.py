"""Store execution-scoped user data credentials outside public specs."""

from __future__ import annotations

import sqlalchemy as sa

from alembic.operations import Operations


revision = '0024_add_execution_credentials'
down_revision = '0023_add_config_secret_overrides'
branch_labels = None
depends_on = None


def upgrade(op: Operations) -> None:
    if 'distributed_execution_credentials' in sa.inspect(op.get_bind()).get_table_names():
        return
    op.create_table(
        'distributed_execution_credentials',
        sa.Column('execution_id', sa.String(length=64), nullable=False),
        sa.Column('credential_kind', sa.String(length=64), nullable=False),
        sa.Column('payload_json', sa.JSON(), nullable=False),
        sa.Column('version', sa.BigInteger(), nullable=False, server_default=sa.text('1')),
        sa.Column('expires_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['execution_id'], ['executions.execution_id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('execution_id', 'credential_kind'),
    )
    op.create_index(
        'idx_execution_credentials_expiry',
        'distributed_execution_credentials',
        ['expires_at', 'execution_id'],
        unique=False,
    )


def downgrade(op: Operations) -> None:
    if 'distributed_execution_credentials' in sa.inspect(op.get_bind()).get_table_names():
        op.drop_table('distributed_execution_credentials')
