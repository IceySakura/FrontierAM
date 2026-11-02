from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True, slots=True)
class ManagerRecord:
    manager_id: str
    session_id: str
    endpoint_url: str
    environment_id: str
    pool_id: str
    workload: str
    labels: dict[str, str]
    metadata: dict[str, Any]
    draining: bool
    supported_modes: list[str]
    supported_spec_types: list[str]
    supported_instance_modes: list[str]
    capacity_total: int
    observed_capacity_used: int
    last_heartbeat_at: datetime
    created_at: datetime
    updated_at: datetime
    bootstrap_version: int | None = None
    artifact_ref_version: int | None = None
    supported_spec_versions: dict[str, list[int]] | None = None
    owner_principal_id: str | None = None
    allocation_id: str | None = None
    allocation_spec_hash: str | None = None
    capability_snapshot: dict[str, Any] | None = None


@dataclass(slots=True)
class EnvironmentManagerInventoryRecord:
    manager_id: str
    session_id: str
    endpoint_url: str
    environment_id: str
    pool_id: str
    allocation_id: str | None
    workload: str
    allocation_spec_hash: str | None
    owner_principal_id: str | None
    labels: dict[str, str]
    metadata: dict[str, Any]
    draining: bool
    bootstrap_version: int | None
    artifact_ref_version: int | None
    supported_spec_versions: dict[str, Any]
    capability_snapshot: dict[str, Any]
    supported_modes: list[str]
    supported_spec_types: list[str]
    supported_instance_modes: list[str]
    capacity_total: int
    reserved_slots: int
    available_slots: int
    healthy: bool
    observed_capacity_used: int
    last_heartbeat_at: datetime
    created_at: datetime
    updated_at: datetime


__all__ = ['EnvironmentManagerInventoryRecord', 'ManagerRecord']
