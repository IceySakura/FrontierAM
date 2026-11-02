"""Add canonical permission scopes to EnvironmentGrant."""

from __future__ import annotations

import json

import sqlalchemy as sa

from alembic.operations import Operations


revision = '0022_add_grant_scopes'
down_revision = '0021_add_provider_configs'
branch_labels = None
depends_on = None
_MIGRATED_USER_SCOPES = (
    'artifact:write',
    'execution:command',
    'execution:read',
    'execution:stop',
    'submission:batch',
    'submission:live',
    'submission:single',
)


def upgrade(op: Operations) -> None:
    bind = op.get_bind()
    columns = {column['name']: column for column in sa.inspect(bind).get_columns('distributed_environment_grants')}
    if 'scopes_json' not in columns:
        with op.batch_alter_table('distributed_environment_grants') as batch:
            batch.add_column(sa.Column('scopes_json', sa.JSON(), nullable=True))

    grants = sa.table(
        'distributed_environment_grants',
        sa.column('scopes_json', sa.JSON()),
    )
    op.execute(sa.update(grants).where(grants.c.scopes_json.is_(None)).values(scopes_json=list(_MIGRATED_USER_SCOPES)))

    columns = {column['name']: column for column in sa.inspect(bind).get_columns('distributed_environment_grants')}
    if columns['scopes_json']['nullable']:
        with op.batch_alter_table('distributed_environment_grants') as batch:
            batch.alter_column('scopes_json', existing_type=sa.JSON(), nullable=False)
    _remove_legacy_submit_all_role(bind)


def downgrade(op: Operations) -> None:
    columns = {column['name'] for column in sa.inspect(op.get_bind()).get_columns('distributed_environment_grants')}
    if 'scopes_json' in columns:
        with op.batch_alter_table('distributed_environment_grants') as batch:
            batch.drop_column('scopes_json')


def _remove_legacy_submit_all_role(bind: sa.Connection) -> None:
    tables = set(sa.inspect(bind).get_table_names())
    for table_name in ('distributed_principals', 'distributed_user_roles'):
        if table_name not in tables:
            continue
        table = sa.table(
            table_name,
            sa.column('principal_id', sa.String()),
            sa.column('roles_json', sa.JSON()),
        )
        for principal_id, raw_roles in bind.execute(sa.select(table.c.principal_id, table.c.roles_json)):
            roles = json.loads(raw_roles) if isinstance(raw_roles, str) else raw_roles
            if not isinstance(roles, list) or 'submit_all' not in roles:
                continue
            next_roles = sorted(set(roles) - {'submit_all'} or {'user'})
            bind.execute(sa.update(table).where(table.c.principal_id == principal_id).values(roles_json=next_roles))
