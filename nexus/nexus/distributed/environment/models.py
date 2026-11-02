from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


class EnvironmentAllocationType:
    PREDEFINED = 'predefined'
    DYNAMIC = 'dynamic'


class EnvironmentProviderType:
    PREDEFINED = 'predefined'
    KUBERNETES = 'kubernetes'
    TKEX = 'tkex'


class EnvironmentStatus:
    ENABLED = 'enabled'
    DISABLED = 'disabled'


class EnvironmentAllocationStatus:
    REQUESTED = 'requested'
    PROVISIONING = 'provisioning'
    READY = 'ready'
    DRAINING = 'draining'
    DELETING = 'deleting'
    RELEASED = 'released'


class ProviderConfigStatus:
    ENABLED = 'enabled'
    DISABLED = 'disabled'


DEFAULT_ENVIRONMENT_ID = 'predefined-default'
DEFAULT_POOL_ID = 'default'


@dataclass(slots=True)
class EnvironmentStatsRecord:
    environment_id: str
    pool_id: str | None
    manager_count: int
    healthy_manager_count: int
    draining_manager_count: int
    capacity_total: int
    reserved_slots: int
    available_slots: int


@dataclass(frozen=True, slots=True)
class ProviderConfigRecord:
    provider_config_ref: str
    provider_type: str
    config: dict[str, Any]
    status: str
    version: int
    created_by: str
    updated_by: str
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class EnvironmentRecord:
    environment_id: str
    name: str
    allocation_type: str
    provider_type: str
    provider_config_ref: str | None
    resource_limits: dict[str, Any]
    default_timeouts: dict[str, Any]
    default_quota: dict[str, int] | None
    config_version: int
    status: str
    created_by: str
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class EnvironmentAllocationRecord:
    allocation_id: str
    top_level_execution_id: str
    owner_principal_id: str
    environment_id: str
    pool_id: str
    workload: str
    status: str
    desired_manager_count: int
    ready_manager_count: int
    manager_count: int
    cpu_millicores: int
    memory_bytes: int
    slots: int
    active_batches: int
    dependency_build_id: str | None
    provider_resource_ref: dict[str, Any] | None
    allocation_spec_hash: str
    config_version: int
    status_version: int
    last_error_code: str | None
    last_error_message: str | None
    last_progress_at: datetime
    created_at: datetime
    ready_at: datetime | None
    released_at: datetime | None
    updated_at: datetime

    configured_manager_liveness: dict[str, Any] | None = None
    manager_liveness_source: str = 'platform'


@dataclass(frozen=True, slots=True)
class ZhiyanLogBinding:
    log_name: str
    provider_type: str


__all__ = [
    'DEFAULT_ENVIRONMENT_ID',
    'DEFAULT_POOL_ID',
    'EnvironmentAllocationRecord',
    'EnvironmentAllocationStatus',
    'EnvironmentAllocationType',
    'EnvironmentProviderType',
    'EnvironmentRecord',
    'EnvironmentStatsRecord',
    'EnvironmentStatus',
    'ProviderConfigRecord',
    'ProviderConfigStatus',
    'ZhiyanLogBinding',
]
