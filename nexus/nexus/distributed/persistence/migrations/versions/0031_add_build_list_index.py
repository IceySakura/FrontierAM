"""Index dependency build ordering and cursor pagination."""

from __future__ import annotations

from alembic.operations import Operations
from sqlalchemy import inspect


revision = '0031_build_list_index'
down_revision = '0030_principal_access_grants'
branch_labels = None
depends_on = None

_TABLE = 'distributed_dependency_builds'
_INDEX = 'idx_dependency_builds_created_id'


def upgrade(op: Operations) -> None:
    indexes = {index['name'] for index in inspect(op.get_bind()).get_indexes(_TABLE)}
    if _INDEX not in indexes:
        op.create_index(_INDEX, _TABLE, ['created_at', 'dependency_build_id'], unique=False)


def downgrade(op: Operations) -> None:
    indexes = {index['name'] for index in inspect(op.get_bind()).get_indexes(_TABLE)}
    if _INDEX in indexes:
        op.drop_index(_INDEX, table_name=_TABLE)
