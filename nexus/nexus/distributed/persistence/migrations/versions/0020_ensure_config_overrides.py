"""Ensure DB-backed distributed configuration exists after branch convergence."""

from __future__ import annotations

from alembic.operations import Operations

from nexus.distributed.persistence.orm import Base


revision = '0020_ensure_config_overrides'
down_revision = '0019_add_execution_parent_cursor'
branch_labels = None
depends_on = None


def upgrade(op: Operations) -> None:
    Base.metadata.tables['distributed_config_overrides'].create(bind=op.get_bind(), checkfirst=True)
