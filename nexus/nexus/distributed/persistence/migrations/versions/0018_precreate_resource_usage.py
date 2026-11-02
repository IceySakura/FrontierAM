"""Precreate Environment and Grant resource usage rows."""

from __future__ import annotations

import sqlalchemy as sa

from alembic.operations import Operations


revision = '0018_precreate_resource_usage'
down_revision = '0017_reconciler_shard_indexes'
branch_labels = None
depends_on = None


def upgrade(op: Operations) -> None:
    environments = sa.table(
        'distributed_environments',
        sa.column('environment_id', sa.String()),
    )
    grants = sa.table(
        'distributed_environment_grants',
        sa.column('principal_id', sa.String()),
        sa.column('environment_id', sa.String()),
    )
    usage = sa.table(
        'distributed_environment_resource_usage',
        sa.column('scope_key', sa.String()),
        sa.column('scope_type', sa.String()),
        sa.column('environment_id', sa.String()),
        sa.column('principal_id', sa.String()),
        sa.column('manager_count', sa.Integer()),
        sa.column('cpu_millicores', sa.BigInteger()),
        sa.column('memory_bytes', sa.BigInteger()),
        sa.column('slots', sa.Integer()),
        sa.column('active_batches', sa.Integer()),
        sa.column('version', sa.BigInteger()),
        sa.column('updated_at', sa.DateTime()),
    )
    columns = [
        'scope_key',
        'scope_type',
        'environment_id',
        'principal_id',
        'manager_count',
        'cpu_millicores',
        'memory_bytes',
        'slots',
        'active_batches',
        'version',
        'updated_at',
    ]

    environment_scope_key = sa.literal('environment:') + environments.c.environment_id
    environment_rows = sa.select(
        environment_scope_key,
        sa.literal('environment'),
        environments.c.environment_id,
        sa.null(),
        sa.literal(0),
        sa.literal(0),
        sa.literal(0),
        sa.literal(0),
        sa.literal(0),
        sa.literal(1),
        sa.func.current_timestamp(),
    ).where(~sa.exists(sa.select(1).where(usage.c.scope_key == environment_scope_key)))
    op.execute(sa.insert(usage).from_select(columns, environment_rows))

    grant_scope_key = sa.literal('grant:') + grants.c.environment_id + sa.literal(':') + grants.c.principal_id
    grant_rows = sa.select(
        grant_scope_key,
        sa.literal('grant'),
        grants.c.environment_id,
        grants.c.principal_id,
        sa.literal(0),
        sa.literal(0),
        sa.literal(0),
        sa.literal(0),
        sa.literal(0),
        sa.literal(1),
        sa.func.current_timestamp(),
    ).where(~sa.exists(sa.select(1).where(usage.c.scope_key == grant_scope_key)))
    op.execute(sa.insert(usage).from_select(columns, grant_rows))


def downgrade(op: Operations) -> None:
    del op
