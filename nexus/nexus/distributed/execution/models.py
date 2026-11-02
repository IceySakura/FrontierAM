from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


class ExecutionMode:
    JOB = 'job'
    LIVE = 'live'


class ExecutionStatus:
    QUEUED = 'queued'
    ASSIGNED = 'assigned'
    STARTING = 'starting'
    RUNNING = 'running'
    WAITING_INPUT = 'waiting_input'
    STOPPING = 'stopping'
    SUCCEEDED = 'succeeded'
    FAILED = 'failed'
    KILLED = 'killed'
    LOST = 'lost'
    EXPIRED = 'expired'


class AssignmentStatus:
    PENDING = 'pending'
    ACTIVE = 'active'
    RELEASED = 'released'
    REVOKED = 'revoked'
    LOST = 'lost'


class RouteStatus:
    NONE = 'none'
    CACHED = 'cached'
    ACTIVE = 'active'
    CLOSED = 'closed'


class CommandStatus:
    PENDING = 'pending'
    CLAIMED = 'claimed'
    DONE = 'done'
    FAILED = 'failed'
    CANCELLED = 'cancelled'


TERMINAL_EXECUTION_STATUSES = {
    ExecutionStatus.SUCCEEDED,
    ExecutionStatus.FAILED,
    ExecutionStatus.KILLED,
    ExecutionStatus.LOST,
    ExecutionStatus.EXPIRED,
}


@dataclass(frozen=True, slots=True)
class AssignmentRecord:
    assignment_id: str
    execution_id: str
    attempt_id: str
    attempt_seq: int
    manager_id: str
    manager_session_id: str
    status: str
    slot_cost: int
    created_at: datetime
    updated_at: datetime
    endpoint_url: str | None = None
    endpoint_ready: bool = False
    runtime_info: dict[str, Any] | None = None
    owner_principal_id: str | None = None
    environment_id: str | None = None
    pool_id: str | None = None
    allocation_id: str | None = None
    workload: str | None = None
    allocation_spec_hash: str | None = None


@dataclass(frozen=True, slots=True)
class ExecutionRecord:
    execution_id: str
    shard_id: int
    mode: str
    spec_type: str
    instance_mode: str
    spec: dict[str, Any]
    driver_config: dict[str, Any]
    placement: dict[str, Any]
    status: str
    route_status: str
    slot_cost: int
    current_assignment_id: str | None
    current_attempt_id: str | None
    attempt_count: int
    max_attempts: int
    next_schedule_at: datetime
    created_at: datetime
    updated_at: datetime
    stop_timeout_seconds: int | None = None
    cleanup_done_at: datetime | None = None
    result_json: dict[str, Any] | None = None
    error_json: dict[str, Any] | None = None
    current_assignment: AssignmentRecord | None = None
    owner_principal_id: str | None = None
    owner_tenant: str | None = None
    created_by: str = 'system:legacy'
    parent_execution_id: str | None = None
    spec_sha256: str | None = None
    environment_id: str | None = None
    pool_id: str | None = None
    allocation_id: str | None = None
    workload: str | None = None
    submission_sha256: str | None = None
    allocation_spec_hash: str | None = None
    result_artifact_id: str | None = None
    first_schedulable_at: datetime | None = None
    stop_reason: str | None = None
    queue_expires_at: datetime | None = None
    execution_expires_at: datetime | None = None
    stop_deadline_at: datetime | None = None
    version: int = 1


@dataclass(frozen=True, slots=True)
class CommandRecord:
    command_id: str
    execution_id: str
    assignment_id: str | None
    attempt_id: str | None
    type: str
    status: str
    payload: dict[str, Any]
    created_at: datetime
    updated_at: datetime
    idempotency_key: str | None = None


@dataclass(frozen=True, slots=True)
class EventRecord:
    event_id: int | None
    execution_id: str
    seq: int
    assignment_id: str | None
    attempt_id: str | None
    type: str
    source: str
    visibility: str
    payload: dict[str, Any]
    event_key: str | None
    created_at: datetime


__all__ = [
    'AssignmentRecord',
    'AssignmentStatus',
    'CommandRecord',
    'CommandStatus',
    'EventRecord',
    'ExecutionMode',
    'ExecutionRecord',
    'ExecutionStatus',
    'RouteStatus',
    'TERMINAL_EXECUTION_STATUSES',
]
