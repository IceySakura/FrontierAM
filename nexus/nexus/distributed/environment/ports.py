from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from nexus.distributed.utils.identity import validate_environment_id, validate_pool_id


@dataclass(frozen=True, slots=True)
class AllocationSnapshot:
    allocation_id: str
    top_level_execution_id: str
    owner_principal_id: str
    environment_id: str
    allocation_type: str
    provider_type: str
    pool_id: str
    workload: str
    status: str
    desired_manager_count: int
    ready_manager_count: int
    cpu_millicores: int
    memory_bytes: int
    slots: int
    allocation_spec: dict[str, Any]
    allocation_spec_hash: str
    provider_resource_ref: dict[str, Any] | None
    provider_config_ref: str | None = None
    provider_config_snapshot: dict[str, Any] | None = None
    execution_status: str | None = None
    batch_control_state: str | None = None
    batch_dependency_status: str | None = None
    created_at: datetime | None = None
    provisioning_timeout_seconds: int | None = None
    artifact_credential_secret_name: str | None = None
    artifact_credential_token: str | None = None
    dependency_build_id: str | None = None
    dependency_build_status: str | None = None
    dependency_build_next_retry_at: datetime | None = None

    def __post_init__(self) -> None:
        validate_environment_id(self.environment_id)
        validate_pool_id(self.pool_id)


@dataclass(frozen=True, slots=True)
class AllocatorObservation:
    exists: bool
    provider_ready_replicas: int
    provider_resource_ref: dict[str, Any] | None


class EnvironmentAllocator(Protocol):
    async def observe(self, allocation: AllocationSnapshot) -> AllocatorObservation: ...

    async def observe_deletion(self, allocation: AllocationSnapshot) -> AllocatorObservation: ...

    async def create(self, allocation: AllocationSnapshot) -> AllocatorObservation: ...

    async def delete(self, allocation: AllocationSnapshot) -> None: ...

    async def close(self) -> None: ...


class AllocationWorkloadClient(Protocol):
    async def get_service(self, name: str) -> dict[str, Any]: ...

    async def create_service(self, body: dict[str, Any]) -> dict[str, Any]: ...

    async def delete_service(self, name: str) -> dict[str, Any]: ...

    async def get_secret(self, name: str) -> dict[str, Any]: ...

    async def create_secret(self, body: dict[str, Any]) -> dict[str, Any]: ...

    async def patch_secret(self, name: str, body: dict[str, Any]) -> dict[str, Any]: ...

    async def delete_secret(self, name: str) -> dict[str, Any]: ...

    async def get_stateful_workload(self, name: str) -> dict[str, Any]: ...

    async def create_stateful_workload(self, body: dict[str, Any]) -> dict[str, Any]: ...

    async def delete_stateful_workload(self, name: str) -> dict[str, Any]: ...


class DependencyBuildWorkloadClient(Protocol):
    async def get_secret(self, name: str) -> dict[str, Any]: ...

    async def create_secret(self, body: dict[str, Any]) -> dict[str, Any]: ...

    async def patch_secret(self, name: str, body: dict[str, Any]) -> dict[str, Any]: ...

    async def delete_secret(self, name: str) -> dict[str, Any]: ...

    async def get_job(self, name: str) -> dict[str, Any]: ...

    async def create_job(self, body: dict[str, Any]) -> dict[str, Any]: ...

    async def delete_job(self, name: str) -> dict[str, Any]: ...

    async def list_job_pods(self, job_name: str) -> list[dict[str, Any]]: ...

    async def list_resource_events(self, resource_name: str) -> list[dict[str, Any]]: ...

    async def read_pod_log(
        self,
        pod_name: str,
        *,
        container: str,
        tail_lines: int,
        limit_bytes: int,
    ) -> str: ...


class WorkloadClient(AllocationWorkloadClient, DependencyBuildWorkloadClient, Protocol):
    async def close(self) -> None: ...


class ResolvedAllocationWorkloadClient(Protocol):
    provider_config: Any
    client: AllocationWorkloadClient


class AllocationWorkloadClientResolver(Protocol):
    def resolve(
        self,
        *,
        provider_type: str,
        provider_config_ref: str | None,
        snapshot: Mapping[str, Any] | None = None,
    ) -> ResolvedAllocationWorkloadClient: ...


class ResolvedWorkloadClientPort(ResolvedAllocationWorkloadClient, Protocol):
    client: WorkloadClient


class WorkloadClientResolverPort(AllocationWorkloadClientResolver, Protocol):
    def resolve(
        self,
        *,
        provider_type: str,
        provider_config_ref: str | None,
        snapshot: Mapping[str, Any] | None = None,
    ) -> ResolvedWorkloadClientPort: ...

    async def close(self) -> None: ...


__all__ = [
    'AllocationSnapshot',
    'AllocationWorkloadClient',
    'AllocationWorkloadClientResolver',
    'AllocatorObservation',
    'DependencyBuildWorkloadClient',
    'EnvironmentAllocator',
    'ResolvedAllocationWorkloadClient',
    'ResolvedWorkloadClientPort',
    'WorkloadClient',
    'WorkloadClientResolverPort',
]
