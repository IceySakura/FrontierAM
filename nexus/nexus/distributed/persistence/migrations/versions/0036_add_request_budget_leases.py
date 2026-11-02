"""Add shared request budget tables and manager request policy capability columns."""

from __future__ import annotations

import sqlalchemy as sa

from alembic.operations import Operations


revision = '0036_request_budget_leases'
down_revision = '0035_policy_scopes'
branch_labels = None
depends_on = None

_BUDGETS_TABLE = 'distributed_request_budgets'
_LEASES_TABLE = 'distributed_request_budget_leases'
_MANAGERS_TABLE = 'instance_managers'
_MANAGERS_INDEX = 'idx_instance_managers_request_policy'


def upgrade(op: Operations) -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = set(inspector.get_table_names())

    if _BUDGETS_TABLE not in tables:
        op.create_table(
            _BUDGETS_TABLE,
            sa.Column('budget_key', sa.String(64), primary_key=True),
            sa.Column('binding_id', sa.String(64), nullable=False),
            sa.Column('scope_id', sa.String(128), nullable=False),
            sa.Column('config_revision', sa.BigInteger(), nullable=False),
            sa.Column('config_hash', sa.String(64), nullable=False),
            sa.Column('ledger_version', sa.BigInteger(), nullable=False),
            sa.Column('updated_at', sa.DateTime(), nullable=False),
        )

    if _LEASES_TABLE not in tables:
        op.create_table(
            _LEASES_TABLE,
            sa.Column('manager_id', sa.String(128), primary_key=True),
            sa.Column('manager_session_id', sa.String(64), primary_key=True),
            sa.Column('budget_key', sa.String(64), nullable=False),
            sa.Column('binding_id', sa.String(64), nullable=False),
            sa.Column('scope_id', sa.String(128), nullable=False),
            sa.Column('environment_id', sa.String(64), nullable=False),
            sa.Column('owner_principal_id', sa.String(64), nullable=True),
            sa.Column('request_seq', sa.BigInteger(), nullable=False),
            sa.Column('requested_qpm', sa.Double(), nullable=True),
            sa.Column('demand_expires_at', sa.DateTime(), nullable=False),
            sa.Column('lease_id', sa.String(64), nullable=True),
            sa.Column('lease_version', sa.BigInteger(), nullable=False),
            sa.Column('granted_qpm', sa.Double(), nullable=False),
            sa.Column('valid_from', sa.DateTime(), nullable=True),
            sa.Column('valid_until', sa.DateTime(), nullable=True),
            sa.Column('mode', sa.String(16), nullable=False),
            sa.Column('config_revision', sa.BigInteger(), nullable=False),
            sa.Column('config_hash', sa.String(64), nullable=False),
            sa.Column('snapshot_version', sa.BigInteger(), nullable=True),
            sa.Column('reason_code', sa.String(64), nullable=False),
            sa.Column('last_result_json', sa.JSON(), nullable=False),
            sa.Column('updated_at', sa.DateTime(), nullable=False),
        )
    lease_indexes = {idx['name'] for idx in sa.inspect(bind).get_indexes(_LEASES_TABLE)}
    if 'idx_request_budget_leases_key_valid' not in lease_indexes:
        op.create_index('idx_request_budget_leases_key_valid', _LEASES_TABLE, ['budget_key', 'valid_until'])
    if 'idx_request_budget_leases_key_demand' not in lease_indexes:
        op.create_index('idx_request_budget_leases_key_demand', _LEASES_TABLE, ['budget_key', 'demand_expires_at'])

    if _MANAGERS_TABLE in tables:
        columns = {column['name'] for column in sa.inspect(bind).get_columns(_MANAGERS_TABLE)}
        if 'request_policy_binding' not in columns:
            op.add_column(_MANAGERS_TABLE, sa.Column('request_policy_binding', sa.String(64), nullable=True))
        if 'request_policy_scope' not in columns:
            op.add_column(_MANAGERS_TABLE, sa.Column('request_policy_scope', sa.String(128), nullable=True))
        manager_indexes = {idx['name'] for idx in sa.inspect(bind).get_indexes(_MANAGERS_TABLE)}
        if _MANAGERS_INDEX not in manager_indexes:
            op.create_index(
                _MANAGERS_INDEX,
                _MANAGERS_TABLE,
                ['environment_id', 'pool_id', 'request_policy_binding', 'request_policy_scope'],
            )


def downgrade(op: Operations) -> None:
    bind = op.get_bind()
    tables = set(sa.inspect(bind).get_table_names())
    if _MANAGERS_TABLE in tables:
        manager_indexes = {idx['name'] for idx in sa.inspect(bind).get_indexes(_MANAGERS_TABLE)}
        if _MANAGERS_INDEX in manager_indexes:
            op.drop_index(_MANAGERS_INDEX, table_name=_MANAGERS_TABLE)
        columns = {column['name'] for column in sa.inspect(bind).get_columns(_MANAGERS_TABLE)}
        for column_name in ('request_policy_scope', 'request_policy_binding'):
            if column_name in columns:
                op.drop_column(_MANAGERS_TABLE, column_name)
    if _LEASES_TABLE in tables:
        op.drop_table(_LEASES_TABLE)
    if _BUDGETS_TABLE in tables:
        op.drop_table(_BUDGETS_TABLE)
