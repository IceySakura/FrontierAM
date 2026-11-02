from __future__ import annotations

from alembic.operations import Operations

from nexus.distributed.persistence.orm import Base


revision = '0004_add_config_overrides'
down_revision = '0003_add_trace_carriers'


def upgrade(op: Operations) -> None:
    bind = op.get_bind()
    Base.metadata.tables['distributed_config_overrides'].create(bind=bind, checkfirst=True)
