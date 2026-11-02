"""Add principal grants for selectable Run Trace Backends."""

from __future__ import annotations

import sqlalchemy as sa

from alembic.operations import Operations


revision = '0029_run_trace_backend_grants'
down_revision = '0028_batch_module_dependencies'
branch_labels = None
depends_on = None


_TABLE = 'distributed_run_trace_backend_grants'
_ACCOUNT_TABLE = 'distributed_trace_database_accounts'
_MIGRATION_ACTOR = 'system:migration:0029'


def upgrade(op: Operations) -> None:
    bind = op.get_bind()
    table_names = set(sa.inspect(bind).get_table_names())
    if _TABLE not in table_names:
        op.create_table(
            _TABLE,
            sa.Column('principal_id', sa.String(length=64), nullable=False),
            sa.Column('backend_name', sa.String(length=64), nullable=False),
            sa.Column('enabled', sa.Boolean(), nullable=False, server_default=sa.true()),
            sa.Column('default_order', sa.Integer(), nullable=True),
            sa.Column('version', sa.BigInteger(), nullable=False, server_default='1'),
            sa.Column('created_by', sa.String(length=128), nullable=False),
            sa.Column('updated_by', sa.String(length=128), nullable=False),
            sa.Column('created_at', sa.DateTime(), nullable=False),
            sa.Column('updated_at', sa.DateTime(), nullable=False),
            sa.ForeignKeyConstraint(
                ['principal_id'],
                ['distributed_principals.principal_id'],
                ondelete='CASCADE',
            ),
            sa.PrimaryKeyConstraint('principal_id', 'backend_name'),
            sa.UniqueConstraint(
                'principal_id',
                'default_order',
                name='uq_run_trace_backend_grants_principal_default_order',
            ),
        )
        op.create_index(
            'idx_run_trace_backend_grants_backend_enabled',
            _TABLE,
            ['backend_name', 'enabled', 'principal_id'],
            unique=False,
        )
    if _ACCOUNT_TABLE in table_names:
        _backfill_active_trace_database_accounts(op)


def _backfill_active_trace_database_accounts(op: Operations) -> None:
    accounts = sa.table(
        _ACCOUNT_TABLE,
        sa.column('owner_principal_id', sa.String(length=64)),
        sa.column('backend_name', sa.String(length=64)),
        sa.column('status', sa.String(length=32)),
        sa.column('operation_id', sa.String(length=64)),
        sa.column('created_at', sa.DateTime()),
        sa.column('updated_at', sa.DateTime()),
        sa.column('disabled_at', sa.DateTime()),
    )
    grants = sa.table(
        _TABLE,
        sa.column('principal_id', sa.String(length=64)),
        sa.column('backend_name', sa.String(length=64)),
        sa.column('enabled', sa.Boolean()),
        sa.column('default_order', sa.Integer()),
        sa.column('version', sa.BigInteger()),
        sa.column('created_by', sa.String(length=128)),
        sa.column('updated_by', sa.String(length=128)),
        sa.column('created_at', sa.DateTime()),
        sa.column('updated_at', sa.DateTime()),
    )
    missing_active_accounts = (
        sa
        .select(
            accounts.c.owner_principal_id,
            accounts.c.backend_name,
            sa.literal(True),
            sa.literal(None, type_=sa.Integer()),
            sa.literal(1),
            sa.literal(_MIGRATION_ACTOR),
            sa.literal(_MIGRATION_ACTOR),
            accounts.c.created_at,
            accounts.c.updated_at,
        )
        .select_from(
            accounts.outerjoin(
                grants,
                (grants.c.principal_id == accounts.c.owner_principal_id)
                & (grants.c.backend_name == accounts.c.backend_name),
            )
        )
        .where(
            grants.c.principal_id.is_(None),
            sa.or_(
                accounts.c.status != 'disabled',
                accounts.c.disabled_at.is_(None),
                accounts.c.operation_id.is_not(None),
            ),
        )
    )
    op.execute(
        sa.insert(grants).from_select(
            [
                'principal_id',
                'backend_name',
                'enabled',
                'default_order',
                'version',
                'created_by',
                'updated_by',
                'created_at',
                'updated_at',
            ],
            missing_active_accounts,
        )
    )


def downgrade(op: Operations) -> None:
    if _TABLE in sa.inspect(op.get_bind()).get_table_names():
        op.drop_table(_TABLE)
