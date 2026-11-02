"""Store asynchronous policy decisions independently of task lifecycle state."""

from __future__ import annotations

from alembic.operations import Operations
from sqlalchemy import JSON, BigInteger, Column, DateTime, Integer, String, inspect


revision = '0035_policy_scopes'
down_revision = '0034_batch_finish_backfill'
branch_labels = None
depends_on = None
_TABLE = 'distributed_policy_scopes'


def upgrade(op: Operations) -> None:
    if _TABLE in inspect(op.get_bind()).get_table_names():
        return
    op.create_table(
        _TABLE,
        Column('scope_key', String(64), primary_key=True),
        Column('kind', String(32), nullable=False),
        Column('binding_id', String(64), nullable=False),
        Column('scope_id', String(128), nullable=False),
        Column('shard_id', Integer, nullable=False),
        Column('config_revision', BigInteger, nullable=False),
        Column('config_hash', String(64), nullable=False),
        Column('state_json', JSON, nullable=False),
        Column('snapshot_json', JSON, nullable=True),
        Column('valid_until', DateTime, nullable=True),
        Column('next_evaluate_at', DateTime, nullable=False),
        Column('last_demand_at', DateTime, nullable=False),
        Column('version', BigInteger, nullable=False),
        Column('failures', Integer, nullable=False),
        Column('last_error', String(64), nullable=True),
        Column('updated_at', DateTime, nullable=False),
    )
    op.create_index('idx_policy_scope_due', _TABLE, ['shard_id', 'next_evaluate_at'])


def downgrade(op: Operations) -> None:
    if _TABLE in inspect(op.get_bind()).get_table_names():
        op.drop_table(_TABLE)
