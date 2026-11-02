from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy import DateTime, Integer, String, and_, case, cast, func, literal, or_, select, text, type_coerce
from sqlalchemy.dialects.mysql import DATETIME
from sqlalchemy.sql.elements import ColumnElement

from nexus.distributed.environment.liveness import MAX_MANAGER_LIVENESS_SECONDS
from nexus.distributed.persistence.orm import EnvironmentAllocationRow, InstanceManagerLivenessRow, InstanceManagerRow


@dataclass(frozen=True, slots=True)
class ManagerLiveness:
    session_id: str
    capacity_total: int
    observed_capacity_used: int
    last_heartbeat_at: datetime
    fenced_at: datetime | None
    from_liveness_table: bool

    @property
    def active(self) -> bool:
        return self.fenced_at is None


def effective_manager_liveness(
    manager: InstanceManagerRow,
    liveness: InstanceManagerLivenessRow | None,
) -> ManagerLiveness:
    if liveness is not None and liveness.session_id == manager.session_id:
        if liveness.fenced_at is None and manager.last_heartbeat_at > liveness.last_heartbeat_at:
            return ManagerLiveness(
                session_id=manager.session_id,
                capacity_total=manager.capacity_total,
                observed_capacity_used=manager.observed_capacity_used,
                last_heartbeat_at=manager.last_heartbeat_at,
                fenced_at=None,
                from_liveness_table=False,
            )
        return ManagerLiveness(
            session_id=liveness.session_id,
            capacity_total=liveness.capacity_total,
            observed_capacity_used=liveness.observed_capacity_used,
            last_heartbeat_at=liveness.last_heartbeat_at,
            fenced_at=liveness.fenced_at,
            from_liveness_table=True,
        )
    return ManagerLiveness(
        session_id=manager.session_id,
        capacity_total=manager.capacity_total,
        observed_capacity_used=manager.observed_capacity_used,
        last_heartbeat_at=manager.last_heartbeat_at,
        fenced_at=None,
        from_liveness_table=False,
    )


def manager_is_healthy_clause(healthy_since: datetime) -> ColumnElement[bool]:
    """Select a healthy current session while old Managers still write the registration row."""

    return or_(
        and_(
            InstanceManagerLivenessRow.manager_id.is_not(None),
            InstanceManagerLivenessRow.fenced_at.is_(None),
            or_(
                InstanceManagerLivenessRow.last_heartbeat_at >= healthy_since,
                InstanceManagerRow.last_heartbeat_at >= healthy_since,
            ),
        ),
        and_(
            InstanceManagerLivenessRow.manager_id.is_(None),
            InstanceManagerRow.last_heartbeat_at >= healthy_since,
        ),
    )


def manager_is_dead_clause(threshold: datetime | ColumnElement[datetime]) -> ColumnElement[bool]:
    """Select a missing, fenced, or stale Manager without breaking a rolling workload upgrade."""

    return or_(
        InstanceManagerRow.manager_id.is_(None),
        and_(
            InstanceManagerLivenessRow.manager_id.is_not(None),
            or_(
                InstanceManagerLivenessRow.fenced_at.is_not(None),
                and_(
                    InstanceManagerLivenessRow.last_heartbeat_at <= threshold,
                    InstanceManagerRow.last_heartbeat_at <= threshold,
                ),
            ),
        ),
        and_(
            InstanceManagerLivenessRow.manager_id.is_(None),
            InstanceManagerRow.last_heartbeat_at <= threshold,
        ),
    )


def allocation_heartbeat_cutoff(
    *,
    now: datetime,
    default_ttl: timedelta,
    dialect: str,
) -> ColumnElement[datetime]:
    """SQL equivalent of allocation_liveness; invalid policies never use a shorter fallback.

    The caller must outer-join the Manager's Allocation. Filtering must happen
    before LIMIT so long-lived, not-yet-expired Managers cannot starve the scan.
    """
    spec = EnvironmentAllocationRow.allocation_spec_json
    path = '$.manager_liveness'

    def json_type(key: str) -> ColumnElement[str]:
        return func.json_type(spec, key) if dialect == 'sqlite' else func.json_type(func.json_extract(spec, key))

    def number(key: str) -> ColumnElement[int]:
        value = func.json_extract(spec, f'{path}.{key}')
        if dialect != 'sqlite':
            value = func.json_unquote(value)
        return cast(value, Integer)

    integer_type, object_type = ('integer', 'object') if dialect == 'sqlite' else ('INTEGER', 'OBJECT')
    version = number('version')
    fence = number('self_fence_timeout_seconds')
    hang = number('executor_hang_timeout_seconds')
    dead = number('dead_timeout_seconds')
    if dialect == 'sqlite':
        members = func.json_each(spec, path).table_valued('key', 'value')
        key_count = select(func.count()).select_from(members).correlate(EnvironmentAllocationRow).scalar_subquery()
    else:
        key_count = func.json_length(func.json_extract(spec, path))
    valid = and_(
        json_type(path) == object_type,
        key_count == 4,
        json_type(f'{path}.version') == integer_type,
        version == 1,
        *(
            and_(json_type(f'{path}.{key}') == integer_type, value.between(1, MAX_MANAGER_LIVENESS_SECONDS))
            for key, value in (
                ('self_fence_timeout_seconds', fence),
                ('executor_hang_timeout_seconds', hang),
                ('dead_timeout_seconds', dead),
            )
        ),
        fence < dead,
        fence <= hang,
    )
    absent = or_(
        InstanceManagerRow.allocation_id.is_(None),
        and_(
            EnvironmentAllocationRow.allocation_id.is_not(None),
            json_type('$') == object_type,
            json_type(path).is_(None),
        ),
    )
    seconds = dead
    current = literal(now, type_=DateTime())
    if dialect == 'sqlite':
        # datetime() truncates fractional seconds; preserve them from the bound
        # timestamp so exact-deadline checks match MySQL TIMESTAMPADD.
        cutoff = cast(func.datetime(current, func.printf('-%d seconds', seconds)), String) + func.substr(
            cast(current, String), 20
        )
    else:
        cutoff = func.timestampadd(text('SECOND'), -seconds, current)
    expression = case((absent, literal(now - default_ttl, type_=DateTime())), (valid, cutoff), else_=None)
    if dialect != 'sqlite':
        return cast(expression, DATETIME(fsp=6))
    return type_coerce(expression, DateTime())


__all__ = [
    'ManagerLiveness',
    'allocation_heartbeat_cutoff',
    'effective_manager_liveness',
    'manager_is_dead_clause',
    'manager_is_healthy_clause',
]
