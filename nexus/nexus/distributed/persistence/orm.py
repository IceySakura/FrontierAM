from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Double,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    insert,
    inspect,
    select,
    text,
)
from sqlalchemy.dialects import mysql
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import JSON

from nexus.distributed.access.models import SubmitTokenKind
from nexus.distributed.access.permissions import CANONICAL_USER_SCOPES
from nexus.distributed.utils.clock import utcnow


class Base(DeclarativeBase):
    pass


JsonDict = dict[str, Any]
_TASK_NAME_TYPE = Text().with_variant(mysql.LONGTEXT(collation='utf8mb4_bin'), 'mysql')
_BATCH_ITEM_ID_TYPE = (
    String(64)
    .with_variant(String(64, collation='BINARY'), 'sqlite')
    .with_variant(mysql.VARCHAR(64, charset='utf8mb4', collation='utf8mb4_bin'), 'mysql')
)


class SchemaVersionRow(Base):
    __tablename__ = 'schema_version'

    version: Mapped[int] = mapped_column(Integer, primary_key=True)


class DistributedConfigOverrideRow(Base):
    __tablename__ = 'distributed_config_overrides'

    role: Mapped[str] = mapped_column(String(64), primary_key=True)
    identity: Mapped[str] = mapped_column(String(128), primary_key=True)
    config_json: Mapped[JsonDict] = mapped_column(JSON, nullable=False, default=dict)
    version: Mapped[int] = mapped_column(BigInteger, nullable=False, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utcnow)


class DistributedConfigSecretOverrideRow(Base):
    __tablename__ = 'distributed_config_secret_overrides'

    role: Mapped[str] = mapped_column(String(64), primary_key=True)
    identity: Mapped[str] = mapped_column(String(128), primary_key=True)
    secrets_json: Mapped[JsonDict] = mapped_column(JSON, nullable=False, default=dict)
    version: Mapped[int] = mapped_column(BigInteger, nullable=False, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utcnow)


class ObservabilitySnapshotRow(Base):
    __tablename__ = 'distributed_observability_snapshots'

    scope_type: Mapped[str] = mapped_column(String(16), primary_key=True)
    scope_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    metric_group: Mapped[str] = mapped_column(String(32), primary_key=True)
    values_json: Mapped[JsonDict | None] = mapped_column(JSON, nullable=True)
    sampled_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_attempt_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    collection_status: Mapped[str] = mapped_column(String(16), nullable=False)
    last_error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    version: Mapped[int] = mapped_column(BigInteger, nullable=False, default=1)
    publisher_id: Mapped[str] = mapped_column(String(128), nullable=False)


class PolicyScopeRow(Base):
    __tablename__ = 'distributed_policy_scopes'
    __table_args__ = (Index('idx_policy_scope_due', 'shard_id', 'next_evaluate_at'),)

    scope_key: Mapped[str] = mapped_column(String(64), primary_key=True)
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    binding_id: Mapped[str] = mapped_column(String(64), nullable=False)
    scope_id: Mapped[str] = mapped_column(String(128), nullable=False)
    shard_id: Mapped[int] = mapped_column(Integer, nullable=False)
    config_revision: Mapped[int] = mapped_column(BigInteger, nullable=False)
    config_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    state_json: Mapped[JsonDict] = mapped_column(JSON, nullable=False, default=dict)
    snapshot_json: Mapped[JsonDict | None] = mapped_column(JSON, nullable=True)
    valid_until: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    next_evaluate_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    last_demand_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    version: Mapped[int] = mapped_column(BigInteger, nullable=False, default=1)
    failures: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_error: Mapped[str | None] = mapped_column(String(64), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class RequestBudgetRow(Base):
    __tablename__ = 'distributed_request_budgets'

    budget_key: Mapped[str] = mapped_column(String(64), primary_key=True)
    binding_id: Mapped[str] = mapped_column(String(64), nullable=False)
    scope_id: Mapped[str] = mapped_column(String(128), nullable=False)
    config_revision: Mapped[int] = mapped_column(BigInteger, nullable=False, default=1)
    config_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    ledger_version: Mapped[int] = mapped_column(BigInteger, nullable=False, default=1)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class RequestBudgetLeaseRow(Base):
    __tablename__ = 'distributed_request_budget_leases'
    __table_args__ = (
        Index('idx_request_budget_leases_key_valid', 'budget_key', 'valid_until'),
        Index('idx_request_budget_leases_key_demand', 'budget_key', 'demand_expires_at'),
    )

    manager_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    manager_session_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    budget_key: Mapped[str] = mapped_column(String(64), nullable=False)
    binding_id: Mapped[str] = mapped_column(String(64), nullable=False)
    scope_id: Mapped[str] = mapped_column(String(128), nullable=False)
    environment_id: Mapped[str] = mapped_column(String(64), nullable=False)
    owner_principal_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    request_seq: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    requested_qpm: Mapped[float | None] = mapped_column(Double, nullable=True)
    demand_expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    lease_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    lease_version: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    granted_qpm: Mapped[float] = mapped_column(Double, nullable=False, default=0.0)
    valid_from: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    valid_until: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    mode: Mapped[str] = mapped_column(String(16), nullable=False, default='shadow')
    config_revision: Mapped[int] = mapped_column(BigInteger, nullable=False, default=1)
    config_hash: Mapped[str] = mapped_column(String(64), nullable=False, default='')
    snapshot_version: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    reason_code: Mapped[str] = mapped_column(String(64), nullable=False, default='policy_unavailable')
    last_result_json: Mapped[JsonDict] = mapped_column(JSON, nullable=False, default=dict)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class ExecutionRow(Base):
    __tablename__ = 'executions'
    __table_args__ = (
        UniqueConstraint('idempotency_scope', 'idempotency_key', name='uq_executions_idempotency'),
        Index('idx_executions_sched', 'status', 'shard_id', 'next_schedule_at', 'priority', 'created_at'),
        Index('idx_executions_status_updated', 'status', 'shard_id', 'updated_at', 'execution_id'),
        Index('idx_executions_route_status', 'route_status', 'status', 'shard_id', 'updated_at', 'execution_id'),
        Index('idx_executions_current_assignment', 'current_assignment_id'),
        Index('idx_executions_env_status_schedule', 'environment_id', 'status', 'next_schedule_at'),
        Index('idx_executions_cleanup', 'status', 'shard_id', 'cleanup_done_at', 'execution_id'),
        Index('idx_executions_owner_status_updated', 'owner_principal_id', 'status', 'updated_at'),
        Index('idx_executions_owner_cursor', 'owner_principal_id', 'created_at', 'execution_id'),
        Index(
            'idx_executions_parent_updated',
            'parent_execution_id',
            'updated_at',
            'created_at',
            'execution_id',
        ),
        Index(
            'idx_executions_owner_parent_updated',
            'owner_principal_id',
            'parent_execution_id',
            'updated_at',
            'created_at',
            'execution_id',
        ),
        Index(
            'idx_executions_spec_parent_updated',
            'spec_type',
            'parent_execution_id',
            'updated_at',
            'created_at',
            'execution_id',
        ),
        Index('idx_executions_allocation_status', 'allocation_id', 'status'),
        Index('idx_executions_queue_deadline', 'status', 'shard_id', 'queue_expires_at', 'execution_id'),
        Index('idx_executions_execution_deadline', 'status', 'shard_id', 'execution_expires_at', 'execution_id'),
        Index('idx_executions_stop_deadline', 'status', 'shard_id', 'stop_deadline_at', 'execution_id'),
        Index('idx_exec_stats_created', 'parent_execution_id', 'created_at', 'execution_id'),
        Index('idx_exec_stats_started', 'parent_execution_id', 'started_at', 'execution_id'),
        Index('idx_exec_stats_finished', 'parent_execution_id', 'finished_at', 'execution_id'),
    )

    execution_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    idempotency_scope: Mapped[str | None] = mapped_column(String(128), nullable=True)
    idempotency_key: Mapped[str | None] = mapped_column(String(256), nullable=True)
    owner_principal_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    owner_tenant: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_by: Mapped[str] = mapped_column(String(128), nullable=False, default='system:legacy')
    parent_execution_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    spec_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    environment_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    pool_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    allocation_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    workload: Mapped[str | None] = mapped_column(String(32), nullable=True)
    submission_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    allocation_spec_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    result_artifact_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    first_schedulable_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    stop_reason: Mapped[str | None] = mapped_column(String(32), nullable=True)
    version: Mapped[int] = mapped_column(BigInteger, nullable=False, default=1)
    shard_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    mode: Mapped[str] = mapped_column(String(16), nullable=False)
    spec_type: Mapped[str] = mapped_column(String(64), nullable=False)
    instance_mode: Mapped[str] = mapped_column(String(32), nullable=False)
    spec_json: Mapped[JsonDict] = mapped_column(JSON, nullable=False)
    task_name: Mapped[str | None] = mapped_column(_TASK_NAME_TYPE, nullable=True)
    driver_config_json: Mapped[JsonDict] = mapped_column(JSON, nullable=False, default=dict)
    placement_json: Mapped[JsonDict] = mapped_column(JSON, nullable=False, default=dict)
    trace_carrier_json: Mapped[JsonDict | None] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    route_status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    slot_cost: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    current_assignment_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    current_attempt_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    next_attempt_seq: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    next_event_seq: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    priority: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    next_schedule_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    queue_entered_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_schedule_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    queue_ttl_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)
    execution_ttl_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)
    stop_timeout_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)
    queue_expires_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    execution_expires_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    stop_deadline_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    result_json: Mapped[JsonDict | None] = mapped_column(JSON, nullable=True)
    error_json: Mapped[JsonDict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    cleanup_done_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class ExecutionCredentialRow(Base):
    __tablename__ = 'distributed_execution_credentials'
    __table_args__ = (Index('idx_execution_credentials_expiry', 'expires_at', 'execution_id'),)

    execution_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey('executions.execution_id', ondelete='CASCADE'),
        primary_key=True,
    )
    credential_kind: Mapped[str] = mapped_column(String(64), primary_key=True)
    payload_json: Mapped[JsonDict] = mapped_column(JSON, nullable=False)
    version: Mapped[int] = mapped_column(BigInteger, nullable=False, default=1)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class AssignmentRow(Base):
    __tablename__ = 'assignments'
    __table_args__ = (
        Index('idx_assignments_manager_status', 'manager_id', 'status', 'updated_at'),
        Index('idx_assignments_manager_session_status', 'manager_id', 'manager_session_id', 'status', 'slot_cost'),
        Index('idx_assignments_execution_status', 'execution_id', 'status', 'updated_at'),
        Index('idx_assignments_status_updated', 'status', 'updated_at'),
        Index('idx_assignments_allocation_status', 'allocation_id', 'status', 'updated_at'),
    )

    assignment_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    execution_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    owner_principal_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    environment_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    pool_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    allocation_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    workload: Mapped[str | None] = mapped_column(String(32), nullable=True)
    allocation_spec_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    attempt_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    attempt_seq: Mapped[int] = mapped_column(Integer, nullable=False)
    manager_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    manager_session_id: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    slot_cost: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    endpoint_url: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    endpoint_ready: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    endpoint_updated_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    endpoint_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    runtime_info_json: Mapped[JsonDict | None] = mapped_column(JSON, nullable=True)
    trace_carrier_json: Mapped[JsonDict | None] = mapped_column(JSON, nullable=True)
    last_instance_update_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    accepted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class InstanceManagerRow(Base):
    __tablename__ = 'instance_managers'
    __table_args__ = (
        Index('idx_instance_managers_heartbeat', 'last_heartbeat_at'),
        Index('idx_managers_env_heartbeat', 'environment_id', 'last_heartbeat_at', 'manager_id'),
        Index('idx_instance_managers_capacity', 'capacity_total'),
        Index('idx_instance_managers_draining', 'draining', 'last_heartbeat_at'),
        Index('idx_instance_managers_allocation', 'allocation_id', 'last_heartbeat_at'),
        Index(
            'idx_instance_managers_environment_pool_workload',
            'environment_id',
            'pool_id',
            'workload',
            'last_heartbeat_at',
        ),
        Index(
            'idx_instance_managers_request_policy',
            'environment_id',
            'pool_id',
            'request_policy_binding',
            'request_policy_scope',
        ),
    )

    manager_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    session_id: Mapped[str] = mapped_column(String(64), nullable=False)
    endpoint_url: Mapped[str] = mapped_column(String(2048), nullable=False)
    labels_json: Mapped[dict[str, str]] = mapped_column(JSON, nullable=False, default=dict)
    draining: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    draining_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    owner_principal_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    environment_id: Mapped[str] = mapped_column(String(64), nullable=False)
    pool_id: Mapped[str] = mapped_column(String(64), nullable=False)
    allocation_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    workload: Mapped[str] = mapped_column(String(32), nullable=False)
    allocation_spec_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    request_policy_binding: Mapped[str | None] = mapped_column(String(64), nullable=True)
    request_policy_scope: Mapped[str | None] = mapped_column(String(128), nullable=True)
    capability_snapshot_json: Mapped[JsonDict | None] = mapped_column(JSON, nullable=True)
    bootstrap_version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    artifact_ref_version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    supported_spec_versions_json: Mapped[JsonDict] = mapped_column(JSON, nullable=False, default=dict)
    supported_modes_json: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    supported_spec_types_json: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    supported_instance_modes_json: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    capacity_total: Mapped[int] = mapped_column(Integer, nullable=False)
    observed_capacity_used: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    local_instances_json: Mapped[JsonDict | None] = mapped_column(JSON, nullable=True)
    version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    metadata_json: Mapped[JsonDict | None] = mapped_column(JSON, nullable=True)
    last_heartbeat_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class InstanceManagerLivenessRow(Base):
    __tablename__ = 'instance_manager_liveness'
    __table_args__ = (
        Index('idx_instance_manager_liveness_heartbeat', 'last_heartbeat_at', 'manager_id'),
        Index('idx_instance_manager_liveness_fenced', 'fenced_at', 'last_heartbeat_at'),
    )

    manager_id: Mapped[str] = mapped_column(
        String(128),
        ForeignKey('instance_managers.manager_id', ondelete='CASCADE'),
        primary_key=True,
    )
    session_id: Mapped[str] = mapped_column(String(64), nullable=False)
    capacity_total: Mapped[int] = mapped_column(Integer, nullable=False)
    observed_capacity_used: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_heartbeat_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    fenced_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class ExecutionCommandRow(Base):
    __tablename__ = 'execution_commands'
    __table_args__ = (
        UniqueConstraint('execution_id', 'idempotency_key', name='uq_execution_commands_idempotency'),
        Index('idx_execution_commands_assignment_pending', 'assignment_id', 'status', 'created_at'),
        Index('idx_execution_commands_execution', 'execution_id', 'created_at'),
        Index('idx_execution_commands_status_updated', 'status', 'updated_at'),
        Index('idx_commands_status_created', 'status', 'created_at'),
    )

    command_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    execution_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    assignment_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    attempt_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    type: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    payload_json: Mapped[JsonDict] = mapped_column(JSON, nullable=False, default=dict)
    idempotency_key: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
    claimed_by_manager_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    claimed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    error_json: Mapped[JsonDict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class ExecutionEventRow(Base):
    __tablename__ = 'execution_events'
    __table_args__ = (
        UniqueConstraint('execution_id', 'seq', name='uq_execution_events_execution_seq'),
        UniqueConstraint('execution_id', 'event_key', name='uq_execution_events_event_key'),
        Index('idx_execution_events_created', 'created_at'),
        Index('idx_execution_events_attempt', 'attempt_id', 'seq'),
    )

    event_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    execution_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    assignment_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    attempt_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    type: Mapped[str] = mapped_column(String(64), nullable=False)
    source: Mapped[str] = mapped_column(String(64), nullable=False)
    visibility: Mapped[str] = mapped_column(String(16), nullable=False)
    payload_json: Mapped[JsonDict] = mapped_column(JSON, nullable=False, default=dict)
    event_key: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class ShardLeaseRow(Base):
    __tablename__ = 'shard_leases'
    __table_args__ = (Index('idx_shard_leases_expires', 'lease_expires_at'),)

    shard_group: Mapped[str] = mapped_column(String(64), primary_key=True)
    shard_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    shard_count: Mapped[int] = mapped_column(Integer, nullable=False)
    owner_id: Mapped[str] = mapped_column(String(128), nullable=False)
    lease_expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class ArtifactRow(Base):
    __tablename__ = 'distributed_artifacts'
    __table_args__ = (
        UniqueConstraint('owner_principal_id', 'kind', 'producer_key', name='uq_artifacts_owner_producer'),
        Index('idx_artifacts_owner_status_created', 'owner_principal_id', 'status', 'created_at'),
        Index(
            'idx_artifacts_owner_environment_status_created',
            'owner_principal_id',
            'environment_id',
            'status',
            'created_at',
        ),
        Index('idx_artifacts_status_expires', 'status', 'expires_at'),
        Index('idx_artifacts_status_operation', 'status', 'operation_expires_at'),
    )

    artifact_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    owner_principal_id: Mapped[str] = mapped_column(String(64), nullable=False)
    environment_id: Mapped[str] = mapped_column(String(64), nullable=False)
    owner_tenant: Mapped[str | None] = mapped_column(String(128), nullable=True)
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    producer_key: Mapped[str | None] = mapped_column(String(256), nullable=True)
    uri: Mapped[str] = mapped_column(String(2048), nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    provider_etag: Mapped[str | None] = mapped_column(String(512), nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    operation_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    operation_owner: Mapped[str | None] = mapped_column(String(128), nullable=True)
    operation_expires_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    failure_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_by: Mapped[str] = mapped_column(String(128), nullable=False)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class ArtifactRefRow(Base):
    __tablename__ = 'distributed_artifact_refs'

    artifact_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    resource_type: Mapped[str] = mapped_column(String(32), primary_key=True)
    resource_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class DependencyBuildRow(Base):
    __tablename__ = 'distributed_dependency_builds'
    __table_args__ = (
        UniqueConstraint(
            'owner_principal_id',
            'environment_id',
            'build_key',
            name='uq_dependency_builds_owner_environment_build_key',
        ),
        Index('idx_dependency_builds_created_id', 'created_at', 'dependency_build_id'),
        Index('idx_dependency_builds_status_operation', 'status', 'operation_lease_until'),
        Index(
            'idx_dependency_builds_owner_environment_status',
            'owner_principal_id',
            'environment_id',
            'status',
        ),
    )

    dependency_build_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    build_key: Mapped[str] = mapped_column(String(64), nullable=False)
    owner_principal_id: Mapped[str] = mapped_column(String(64), nullable=False)
    environment_id: Mapped[str] = mapped_column(String(64), nullable=False)
    provider_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    provider_config_ref: Mapped[str | None] = mapped_column(String(512), nullable=True)
    provider_config_snapshot_json: Mapped[JsonDict | None] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    build_spec_json: Mapped[JsonDict] = mapped_column(JSON, nullable=False)
    wheelhouse_ref_json: Mapped[JsonDict | None] = mapped_column(JSON, nullable=True)
    manifest_json: Mapped[JsonDict | None] = mapped_column(JSON, nullable=True)
    provider_job_name: Mapped[str | None] = mapped_column(String(253), nullable=True)
    attempt: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    operation_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    operation_owner: Mapped[str | None] = mapped_column(String(128), nullable=True)
    operation_lease_until: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    next_retry_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(128), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    deadline_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class ArtifactUploadRow(Base):
    __tablename__ = 'distributed_artifact_uploads'
    __table_args__ = (
        UniqueConstraint('idempotency_scope', 'idempotency_key', name='uq_artifact_uploads_idempotency'),
        Index('idx_artifact_uploads_status_operation', 'status', 'operation_expires_at'),
        Index('idx_artifact_uploads_owner_status_created', 'owner_principal_id', 'status', 'created_at'),
        Index(
            'idx_artifact_uploads_owner_environment_status_created',
            'owner_principal_id',
            'environment_id',
            'status',
            'created_at',
        ),
        Index('idx_artifact_uploads_artifact_status', 'artifact_id', 'status'),
    )

    upload_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    artifact_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    owner_principal_id: Mapped[str] = mapped_column(String(64), nullable=False)
    environment_id: Mapped[str] = mapped_column(String(64), nullable=False)
    idempotency_scope: Mapped[str] = mapped_column(String(128), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(128), nullable=False)
    expected_kind: Mapped[str] = mapped_column(String(32), nullable=False)
    expected_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    expected_size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    provider_etag: Mapped[str | None] = mapped_column(String(512), nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    operation_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    operation_owner: Mapped[str | None] = mapped_column(String(128), nullable=True)
    operation_expires_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    failure_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_failure_code: Mapped[str | None] = mapped_column(String(128), nullable=True)
    version: Mapped[int] = mapped_column(BigInteger, nullable=False, default=1)
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class ArtifactQuotaUsageRow(Base):
    """Compatibility-only preview schema; new admission does not read or write it."""

    __tablename__ = 'distributed_artifact_quota_usage'

    owner_principal_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    active_upload_sessions: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    accounted_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    version: Mapped[int] = mapped_column(BigInteger, nullable=False, default=1)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class PrincipalRow(Base):
    __tablename__ = 'distributed_principals'
    __table_args__ = (UniqueConstraint('issuer', 'tenant', 'subject', name='uq_principals_identity'),)

    principal_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    issuer: Mapped[str] = mapped_column(String(64), nullable=False)
    tenant: Mapped[str] = mapped_column(String(128), nullable=False)
    subject: Mapped[str] = mapped_column(String(256), nullable=False)
    roles_json: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    display_name: Mapped[str | None] = mapped_column(String(256), nullable=True)
    created_by: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class PrincipalAccessGrantRow(Base):
    __tablename__ = 'distributed_principal_access_grants'
    __table_args__ = (Index('idx_principal_access_grants_grantee_owner', 'grantee_principal_id', 'owner_principal_id'),)

    owner_principal_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey('distributed_principals.principal_id', ondelete='CASCADE'),
        primary_key=True,
    )
    grantee_principal_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey('distributed_principals.principal_id', ondelete='CASCADE'),
        primary_key=True,
    )
    created_by: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class TraceDatabaseAccountRow(Base):
    __tablename__ = 'distributed_trace_database_accounts'
    __table_args__ = (
        UniqueConstraint('backend_name', 'db_username', name='uq_trace_database_accounts_backend_username'),
        Index('idx_trace_database_accounts_status_updated', 'status', 'updated_at'),
    )

    owner_principal_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey('distributed_principals.principal_id', ondelete='CASCADE'),
        primary_key=True,
    )
    backend_name: Mapped[str] = mapped_column(String(64), primary_key=True)
    db_username: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    credential_version: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    operation_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    operation_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    operation_expires_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_error_code: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    disabled_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class RunTraceBackendGrantRow(Base):
    __tablename__ = 'distributed_run_trace_backend_grants'
    __table_args__ = (
        UniqueConstraint(
            'principal_id',
            'default_order',
            name='uq_run_trace_backend_grants_principal_default_order',
        ),
        Index('idx_run_trace_backend_grants_backend_enabled', 'backend_name', 'enabled', 'principal_id'),
    )

    principal_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey('distributed_principals.principal_id', ondelete='CASCADE'),
        primary_key=True,
    )
    backend_name: Mapped[str] = mapped_column(String(64), primary_key=True)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    default_order: Mapped[int | None] = mapped_column(Integer, nullable=True)
    version: Mapped[int] = mapped_column(BigInteger, nullable=False, default=1)
    created_by: Mapped[str] = mapped_column(String(128), nullable=False)
    updated_by: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class EnvironmentRow(Base):
    __tablename__ = 'distributed_environments'
    __table_args__ = (
        UniqueConstraint('name', name='uq_environments_name'),
        Index('idx_environments_status_name', 'status', 'name'),
    )

    environment_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    allocation_type: Mapped[str] = mapped_column(String(16), nullable=False)
    provider_type: Mapped[str] = mapped_column(String(32), nullable=False)
    provider_config_ref: Mapped[str | None] = mapped_column(String(512), nullable=True)
    resource_limits_json: Mapped[JsonDict] = mapped_column(JSON, nullable=False, default=dict)
    default_timeouts_json: Mapped[JsonDict] = mapped_column(JSON, nullable=False, default=dict)
    default_quota_json: Mapped[JsonDict | None] = mapped_column(JSON, nullable=True)
    config_version: Mapped[int] = mapped_column(BigInteger, nullable=False, default=1)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    created_by: Mapped[str] = mapped_column(String(128), nullable=False)
    updated_by: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class ProviderConfigRow(Base):
    __tablename__ = 'distributed_provider_configs'
    __table_args__ = (Index('idx_provider_configs_status_ref', 'status', 'provider_config_ref'),)

    provider_config_ref: Mapped[str] = mapped_column(String(512), primary_key=True)
    provider_type: Mapped[str] = mapped_column(String(32), nullable=False)
    config_json: Mapped[JsonDict] = mapped_column(JSON, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    version: Mapped[int] = mapped_column(BigInteger, nullable=False, default=1)
    created_by: Mapped[str] = mapped_column(String(128), nullable=False)
    updated_by: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class EnvironmentGrantRow(Base):
    __tablename__ = 'distributed_environment_grants'
    __table_args__ = (
        Index('idx_environment_grants_principal_enabled', 'principal_id', 'enabled', 'environment_id'),
        Index('idx_environment_grants_environment_enabled', 'environment_id', 'enabled', 'principal_id'),
    )

    principal_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    environment_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    limits_json: Mapped[JsonDict] = mapped_column(JSON, nullable=False, default=dict)
    scopes_json: Mapped[list[str]] = mapped_column(
        JSON,
        nullable=False,
        default=lambda: sorted(CANONICAL_USER_SCOPES),
    )
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    version: Mapped[int] = mapped_column(BigInteger, nullable=False, default=1)
    created_by: Mapped[str] = mapped_column(String(128), nullable=False)
    updated_by: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class EnvironmentAllocationRow(Base):
    __tablename__ = 'distributed_environment_allocations'
    __table_args__ = (
        Index('idx_allocations_status_updated', 'status', 'updated_at'),
        Index('idx_allocations_environment_status', 'environment_id', 'status'),
        Index('idx_allocations_owner_environment_status', 'owner_principal_id', 'environment_id', 'status'),
        Index('idx_allocations_execution', 'top_level_execution_id', 'workload'),
        Index('idx_allocations_dependency_build', 'dependency_build_id'),
        UniqueConstraint(
            'top_level_execution_id',
            'pool_id',
            'workload',
            name='uq_allocations_execution_pool_workload',
        ),
    )

    allocation_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    top_level_execution_id: Mapped[str] = mapped_column(String(64), nullable=False)
    owner_principal_id: Mapped[str] = mapped_column(String(64), nullable=False)
    environment_id: Mapped[str] = mapped_column(String(64), nullable=False)
    pool_id: Mapped[str] = mapped_column(String(64), nullable=False)
    workload: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    desired_manager_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    ready_manager_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    manager_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    cpu_millicores: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    memory_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    slots: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    active_batches: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    dependency_build_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    provider_resource_ref_json: Mapped[JsonDict | None] = mapped_column(JSON, nullable=True)
    allocation_spec_json: Mapped[JsonDict] = mapped_column(JSON, nullable=False, default=dict)
    allocation_spec_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    config_version: Mapped[int] = mapped_column(BigInteger, nullable=False)
    operation_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    operation_owner: Mapped[str | None] = mapped_column(String(128), nullable=True)
    operation_lease_until: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    status_version: Mapped[int] = mapped_column(BigInteger, nullable=False, default=1)
    last_error_code: Mapped[str | None] = mapped_column(String(128), nullable=True)
    last_error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    last_progress_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utcnow)
    resources_reserved_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    resources_released_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    provisioning_started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    provisioning_deadline_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    release_requested_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    ready_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    released_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class EnvironmentResourceUsageRow(Base):
    __tablename__ = 'distributed_environment_resource_usage'
    __table_args__ = (
        Index('idx_environment_resource_usage_environment', 'environment_id', 'scope_type'),
        Index('idx_resource_usage_scope_environment', 'scope_type', 'environment_id'),
        UniqueConstraint(
            'scope_type',
            'environment_id',
            'principal_id',
            name='uq_environment_resource_usage_scope',
        ),
    )

    scope_key: Mapped[str] = mapped_column(String(160), primary_key=True)
    scope_type: Mapped[str] = mapped_column(String(16), nullable=False)
    environment_id: Mapped[str] = mapped_column(String(64), nullable=False)
    principal_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    manager_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    cpu_millicores: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    memory_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    slots: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    active_batches: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    version: Mapped[int] = mapped_column(BigInteger, nullable=False, default=1)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class BatchRunRow(Base):
    __tablename__ = 'distributed_batch_runs'
    __table_args__ = (
        Index(
            'idx_batch_runs_shard_lifecycle_claim',
            'shard_id',
            'control_state',
            'last_reconciled_at',
            'lifecycle_lease_expires_at',
            'parent_execution_id',
        ),
        Index('idx_batch_runs_control_progress', 'control_state', 'last_progress_at'),
        Index('idx_batch_runs_generation_progress', 'generation_status', 'last_progress_at'),
    )

    parent_execution_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    shard_id: Mapped[int] = mapped_column(Integer, nullable=False)
    generation_mode: Mapped[str] = mapped_column(String(32), nullable=False)
    source_parent_execution_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    source_summary_version: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    generation_status: Mapped[str] = mapped_column(String(32), nullable=False)
    generation_epoch: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    preparation_started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    preparation_deadline_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    materialized_cursor: Mapped[str | None] = mapped_column(String(256), nullable=True)
    materialize_cursor_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    materialized_count: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    task_list_ref_json: Mapped[JsonDict | None] = mapped_column(JSON, nullable=True)
    task_list_manifest_json: Mapped[JsonDict | None] = mapped_column(JSON, nullable=True)
    task_list_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    dependency_status: Mapped[str] = mapped_column(String(32), nullable=False, default='not_required')
    module_dependencies_json: Mapped[JsonDict | None] = mapped_column(JSON, nullable=True)
    child_dependency_build_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    child_template_json: Mapped[JsonDict | None] = mapped_column(JSON, nullable=True)
    owner_principal_id: Mapped[str] = mapped_column(String(64), nullable=False)
    owner_tenant: Mapped[str | None] = mapped_column(String(128), nullable=True)
    total_items: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    submitted_items: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    active_items: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    terminal_items: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    submission_closed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    control_state: Mapped[str] = mapped_column(String(32), nullable=False, default='active')
    stop_reason: Mapped[str | None] = mapped_column(String(32), nullable=True)
    stop_requested_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    stop_cursor: Mapped[str | None] = mapped_column(_BATCH_ITEM_ID_TYPE, nullable=True)
    child_sync_cursor: Mapped[str | None] = mapped_column(_BATCH_ITEM_ID_TYPE, nullable=True)
    lifecycle_owner_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    lifecycle_operation_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    lifecycle_lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_reconciled_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    active_parent_attempt_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    summary_version: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    summary_json: Mapped[JsonDict | None] = mapped_column(JSON, nullable=True)
    last_progress_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    finalized_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class BatchItemRow(Base):
    __tablename__ = 'distributed_batch_items'
    __table_args__ = (
        UniqueConstraint('parent_execution_id', 'task_ordinal', name='uq_batch_items_ordinal'),
        UniqueConstraint('current_child_execution_id', name='uq_batch_items_current_child'),
        Index('idx_batch_items_parent_status', 'parent_execution_id', 'status', 'item_id'),
        Index('idx_batch_items_parent_status_ordinal', 'parent_execution_id', 'status', 'task_ordinal'),
        Index('idx_batch_items_parent_updated', 'parent_execution_id', 'updated_at'),
        Index('idx_batch_stats_created', 'created_at', 'parent_execution_id', 'item_id'),
        Index('idx_batch_stats_started', 'first_started_at', 'parent_execution_id', 'item_id'),
        Index('idx_batch_stats_finished', 'finished_at', 'parent_execution_id', 'item_id'),
    )

    parent_execution_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    item_id: Mapped[str] = mapped_column(_BATCH_ITEM_ID_TYPE, primary_key=True)
    task_ordinal: Mapped[int] = mapped_column(BigInteger, nullable=False)
    task_offset: Mapped[int] = mapped_column(BigInteger, nullable=False)
    task_length: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    current_attempt_no: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    current_child_execution_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    spec_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    status_version: Mapped[int] = mapped_column(BigInteger, nullable=False, default=1)
    child_execution_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    child_execution_version: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    task_name: Mapped[str | None] = mapped_column(_TASK_NAME_TYPE, nullable=True)
    first_started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at_estimated: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=text('0')
    )
    metadata_json: Mapped[JsonDict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class BatchItemAttemptRow(Base):
    __tablename__ = 'distributed_batch_item_attempts'
    __table_args__ = (
        UniqueConstraint('child_execution_id', name='uq_batch_item_attempts_child'),
        Index('idx_batch_item_attempts_parent_status', 'parent_execution_id', 'status'),
        Index(
            'idx_batch_item_attempts_parent_error_item',
            'parent_execution_id',
            'error_code',
            'item_id',
        ),
        Index(
            'idx_batch_item_attempts_parent_item_status',
            'parent_execution_id',
            'status',
            'item_id',
            'attempt_no',
        ),
    )

    parent_execution_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    item_id: Mapped[str] = mapped_column(_BATCH_ITEM_ID_TYPE, primary_key=True)
    attempt_no: Mapped[int] = mapped_column(Integer, primary_key=True)
    child_execution_id: Mapped[str] = mapped_column(String(64), nullable=False)
    spec_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    status_version: Mapped[int] = mapped_column(BigInteger, nullable=False, default=1)
    result_artifact_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(128), nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class UserRoleRow(Base):
    __tablename__ = 'distributed_user_roles'
    __table_args__ = (UniqueConstraint('issuer', 'tenant', 'subject', name='uq_user_roles_identity'),)

    principal_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    issuer: Mapped[str] = mapped_column(String(64), nullable=False)
    tenant: Mapped[str] = mapped_column(String(128), nullable=False)
    subject: Mapped[str] = mapped_column(String(256), nullable=False)
    roles_json: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    display_name: Mapped[str | None] = mapped_column(String(256), nullable=True)
    created_by: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class SubmitTokenRow(Base):
    __tablename__ = 'distributed_submit_tokens'
    __table_args__ = (
        Index('uq_submit_tokens_bound_app_id', 'bound_app_id', unique=True),
        Index('idx_submit_tokens_owner_created', 'owner_principal_id', 'created_at'),
        Index('idx_submit_tokens_environment_owner', 'environment_id', 'owner_principal_id', 'created_at'),
        Index('idx_submit_tokens_prefix', 'token_prefix'),
    )

    token_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    token_kind: Mapped[str] = mapped_column(
        String(32), nullable=False, default=SubmitTokenKind.API, server_default=SubmitTokenKind.API
    )
    token_prefix: Mapped[str | None] = mapped_column(String(16), nullable=True)
    token_hash: Mapped[str | None] = mapped_column(String(128), nullable=True)
    token_hash_key_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    owner_principal_id: Mapped[str] = mapped_column(String(64), nullable=False)
    environment_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    bound_app_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    scopes_json: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    created_by: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class UserQuotaOverrideRow(Base):
    __tablename__ = 'distributed_user_quota_overrides'

    owner_principal_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    limits_json: Mapped[JsonDict] = mapped_column(JSON, nullable=False)
    version: Mapped[int] = mapped_column(BigInteger, nullable=False, default=1)
    updated_by: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class QuotaUsageRow(Base):
    __tablename__ = 'distributed_quota_usage'

    owner_principal_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    environment: Mapped[str] = mapped_column(String(64), primary_key=True)
    pool: Mapped[str] = mapped_column(String(64), primary_key=True)
    quota_class: Mapped[str] = mapped_column(String(32), primary_key=True)
    queued_executions: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    running_executions: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    reserved_slots: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    version: Mapped[int] = mapped_column(BigInteger, nullable=False, default=1)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class AuditRow(Base):
    __tablename__ = 'distributed_audit_events'
    __table_args__ = (
        Index('idx_audit_actor_created', 'actor_principal_id', 'created_at'),
        Index('idx_audit_target_created', 'target_owner_principal_id', 'created_at'),
    )

    audit_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    actor_principal_id: Mapped[str] = mapped_column(String(64), nullable=False)
    target_owner_principal_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    operation: Mapped[str] = mapped_column(String(128), nullable=False)
    resource_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    resource_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    request_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    metadata_json: Mapped[JsonDict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


async def create_schema(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await _ensure_execution_cleanup_schema(conn)
        current = await conn.scalar(select(SchemaVersionRow).where(SchemaVersionRow.version == 1))
        if current is None:
            await conn.execute(insert(SchemaVersionRow).values(version=1))


async def _ensure_execution_cleanup_schema(conn) -> None:
    columns = await conn.run_sync(
        lambda sync_conn: {column['name'] for column in inspect(sync_conn).get_columns('executions')}
    )
    if 'cleanup_done_at' not in columns:
        await conn.execute(text('ALTER TABLE executions ADD COLUMN cleanup_done_at DATETIME NULL'))

    indexes = await conn.run_sync(
        lambda sync_conn: {index['name'] for index in inspect(sync_conn).get_indexes('executions')}
    )
    if 'idx_executions_cleanup' not in indexes:
        await conn.execute(
            text('CREATE INDEX idx_executions_cleanup ON executions (status, shard_id, cleanup_done_at, execution_id)')
        )
