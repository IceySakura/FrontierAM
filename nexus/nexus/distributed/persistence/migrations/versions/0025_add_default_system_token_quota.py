"""Add persistent default system tokens and Environment self-service quota."""

from __future__ import annotations

import sqlalchemy as sa

from alembic.operations import Operations


revision = '0025_default_token_quota'
down_revision = '0024_add_execution_credentials'
branch_labels = None
depends_on = None


def upgrade(op: Operations) -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    environment_columns = {column['name'] for column in inspector.get_columns('distributed_environments')}
    if 'default_quota_json' not in environment_columns:
        op.add_column('distributed_environments', sa.Column('default_quota_json', sa.JSON(), nullable=True))

    token_columns = {column['name']: column for column in sa.inspect(bind).get_columns('distributed_submit_tokens')}
    if 'token_kind' not in token_columns:
        op.add_column(
            'distributed_submit_tokens',
            sa.Column('token_kind', sa.String(length=32), nullable=True, server_default='api'),
        )
        op.execute(sa.text("UPDATE distributed_submit_tokens SET token_kind = 'api' WHERE token_kind IS NULL"))
        token_columns = {column['name']: column for column in sa.inspect(bind).get_columns('distributed_submit_tokens')}

    nullable_fields = {
        'token_prefix': sa.String(length=16),
        'token_hash': sa.String(length=128),
        'token_hash_key_id': sa.String(length=32),
        'environment_id': sa.String(length=64),
    }
    token_kind_has_default = _is_api_default(token_columns['token_kind'].get('default'))
    needs_token_rebuild = (
        token_columns['token_kind']['nullable']
        or not token_kind_has_default
        or any(not token_columns[field]['nullable'] for field in nullable_fields)
    )
    if needs_token_rebuild:
        with op.batch_alter_table('distributed_submit_tokens') as batch:
            if token_columns['token_kind']['nullable'] or not token_kind_has_default:
                batch.alter_column(
                    'token_kind',
                    existing_type=sa.String(length=32),
                    nullable=False,
                    server_default='api',
                )
            for field, field_type in nullable_fields.items():
                if not token_columns[field]['nullable']:
                    batch.alter_column(field, existing_type=field_type, nullable=True)


def downgrade(op: Operations) -> None:
    bind = op.get_bind()
    token_columns = {column['name']: column for column in sa.inspect(bind).get_columns('distributed_submit_tokens')}
    if 'token_kind' in token_columns:
        op.execute(sa.text("DELETE FROM distributed_submit_tokens WHERE token_kind = 'system_default'"))
        with op.batch_alter_table('distributed_submit_tokens') as batch:
            for field, field_type in {
                'token_prefix': sa.String(length=16),
                'token_hash': sa.String(length=128),
                'token_hash_key_id': sa.String(length=32),
                'environment_id': sa.String(length=64),
            }.items():
                batch.alter_column(field, existing_type=field_type, nullable=False)
            batch.drop_column('token_kind')
    environment_columns = {column['name'] for column in sa.inspect(bind).get_columns('distributed_environments')}
    if 'default_quota_json' in environment_columns:
        op.drop_column('distributed_environments', 'default_quota_json')


def _is_api_default(value: object) -> bool:
    if value is None:
        return False
    normalized = str(value).strip()
    while normalized.startswith('(') and normalized.endswith(')'):
        normalized = normalized[1:-1].strip()
    return normalized.strip('\'"') == 'api'
