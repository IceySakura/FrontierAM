from __future__ import annotations

import base64

from contextlib import suppress
from typing import Any

from nexus.distributed.environment.database_secret import EnvironmentDatabaseSecretSynchronizer
from nexus.distributed.environment.kubernetes import WorkloadConflictError, WorkloadNotFoundError
from nexus.distributed.environment.naming import allocation_resource_names
from nexus.distributed.environment.ports import (
    AllocationSnapshot,
    AllocationWorkloadClient,
    AllocatorObservation,
    EnvironmentAllocator,
)
from nexus.distributed.environment.rendering import AllocationWorkloadRenderer
from nexus.distributed.utils.identity import canonical_sha256


class AllocationSpecConflictError(ValueError):
    pass


class EnvironmentProvisioningTimeoutError(AllocationSpecConflictError):
    pass


class PredefinedEnvironmentAllocator(EnvironmentAllocator):
    async def observe(self, allocation: AllocationSnapshot) -> AllocatorObservation:
        return AllocatorObservation(
            exists=True,
            provider_ready_replicas=allocation.desired_manager_count,
            provider_resource_ref={'type': 'predefined', 'pool_id': allocation.pool_id},
        )

    async def create(self, allocation: AllocationSnapshot) -> AllocatorObservation:
        return await self.observe(allocation)

    async def observe_deletion(self, allocation: AllocationSnapshot) -> AllocatorObservation:
        return await self.observe(allocation)

    async def delete(self, allocation: AllocationSnapshot) -> None:
        del allocation

    async def close(self) -> None:
        pass


class StatefulEnvironmentAllocator(EnvironmentAllocator):
    def __init__(
        self,
        *,
        client: AllocationWorkloadClient,
        renderer: AllocationWorkloadRenderer,
        resource_kind: str,
        database_secret_synchronizer: EnvironmentDatabaseSecretSynchronizer,
    ) -> None:
        self.client = client
        self.renderer = renderer
        self.resource_kind = resource_kind
        self.database_secret_synchronizer = database_secret_synchronizer

    async def observe(self, allocation: AllocationSnapshot) -> AllocatorObservation:
        await self.database_secret_synchronizer.ensure(self.client)
        await self._ensure_artifact_credential(allocation)
        return await self.observe_deletion(allocation)

    async def observe_deletion(self, allocation: AllocationSnapshot) -> AllocatorObservation:
        workload_name, service_name = allocation_resource_names(allocation.allocation_id)
        service = await self._get_optional_service(service_name)
        workload = await self._get_optional_workload(workload_name)
        if service is None and workload is None:
            return AllocatorObservation(False, 0, None)
        if service is not None:
            _validate_owned_resource(service, allocation)
        if workload is not None:
            _validate_owned_resource(workload, allocation)
        workload_status = (workload or {}).get('status') or {}
        ready_replicas = int(
            workload_status.get('readyReplicas')
            or workload_status.get('availableReplicas')
            or workload_status.get('currentReadyReplicas')
            or 0
        )
        provider_resource_ref: dict[str, Any] = {'kind': self.resource_kind}
        if service is not None:
            provider_resource_ref['service'] = _resource_ref(service)
        if workload is not None:
            provider_resource_ref['workload'] = _resource_ref(workload)
        return AllocatorObservation(
            exists=True,
            provider_ready_replicas=ready_replicas if workload is not None else 0,
            provider_resource_ref=provider_resource_ref,
        )

    async def create(self, allocation: AllocationSnapshot) -> AllocatorObservation:
        await self.database_secret_synchronizer.ensure(self.client)
        await self._ensure_artifact_credential(allocation)
        rendered = self.renderer.render(allocation, resource_kind=self.resource_kind)
        await self._ensure_workload(rendered.workload, allocation)
        return await self.observe_deletion(allocation)

    async def delete(self, allocation: AllocationSnapshot) -> None:
        workload_name, service_name = allocation_resource_names(allocation.allocation_id)
        workload = await self._get_optional_workload(workload_name)
        service = await self._get_optional_service(service_name)
        if workload is not None:
            _validate_owned_resource(workload, allocation)
            await self.client.delete_stateful_workload(workload_name)
        if service is not None:
            _validate_owned_resource(service, allocation)
            await self.client.delete_service(service_name)
        if allocation.artifact_credential_secret_name:
            with suppress(WorkloadNotFoundError):
                await self.client.delete_secret(allocation.artifact_credential_secret_name)

    async def close(self) -> None:
        pass

    async def _get_optional_service(self, name: str) -> dict[str, Any] | None:
        try:
            return await self.client.get_service(name)
        except WorkloadNotFoundError:
            return None

    async def _get_optional_workload(self, name: str) -> dict[str, Any] | None:
        try:
            return await self.client.get_stateful_workload(name)
        except WorkloadNotFoundError:
            return None

    async def _ensure_workload(self, body: dict[str, Any], allocation: AllocationSnapshot) -> None:
        existing = await self._get_optional_workload(body['metadata']['name'])
        if existing is None:
            try:
                existing = await self.client.create_stateful_workload(body)
            except WorkloadConflictError:
                existing = await self.client.get_stateful_workload(body['metadata']['name'])
        _validate_owned_resource(existing, allocation)

    async def _ensure_artifact_credential(self, allocation: AllocationSnapshot) -> None:
        name = allocation.artifact_credential_secret_name
        token = allocation.artifact_credential_token
        if not name or not token:
            raise AllocationSpecConflictError('allocation_artifact_credential_missing')
        desired = _artifact_credential_secret(allocation, namespace=self.renderer.workload_namespace)
        try:
            existing = await self.client.get_secret(name)
        except WorkloadNotFoundError:
            try:
                existing = await self.client.create_secret(desired)
            except WorkloadConflictError:
                existing = await self.client.get_secret(name)
        _validate_owned_resource(existing, allocation)
        current_hash = ((existing.get('metadata') or {}).get('annotations') or {}).get('nexus.io/credential-sha256')
        desired_hash = desired['metadata']['annotations']['nexus.io/credential-sha256']
        if current_hash != desired_hash:
            existing = await self.client.patch_secret(name, desired)
            _validate_owned_resource(existing, allocation)


class KubernetesEnvironmentAllocator(StatefulEnvironmentAllocator):
    def __init__(
        self,
        *,
        client: AllocationWorkloadClient,
        renderer: AllocationWorkloadRenderer,
        database_secret_synchronizer: EnvironmentDatabaseSecretSynchronizer,
    ) -> None:
        super().__init__(
            client=client,
            renderer=renderer,
            resource_kind='StatefulSet',
            database_secret_synchronizer=database_secret_synchronizer,
        )


class TkexEnvironmentAllocator(StatefulEnvironmentAllocator):
    def __init__(
        self,
        *,
        client: AllocationWorkloadClient,
        renderer: AllocationWorkloadRenderer,
        database_secret_synchronizer: EnvironmentDatabaseSecretSynchronizer,
    ) -> None:
        super().__init__(
            client=client,
            renderer=renderer,
            resource_kind='StatefulSetPlus',
            database_secret_synchronizer=database_secret_synchronizer,
        )


def _artifact_credential_secret(allocation: AllocationSnapshot, *, namespace: str) -> dict[str, Any]:
    name = allocation.artifact_credential_secret_name
    token = allocation.artifact_credential_token
    if not name or not token:
        raise AllocationSpecConflictError('allocation_artifact_credential_missing')
    return {
        'apiVersion': 'v1',
        'kind': 'Secret',
        'metadata': {
            'name': name,
            'namespace': namespace,
            'labels': {
                'app.kubernetes.io/managed-by': 'nexus',
                'nexus.io/workload': allocation.workload,
            },
            'annotations': {
                'nexus.io/managed-by': 'nexus',
                'nexus.io/allocation-id': allocation.allocation_id,
                'nexus.io/environment-id': allocation.environment_id,
                'nexus.io/allocation-spec-hash': allocation.allocation_spec_hash,
                'nexus.io/credential-sha256': canonical_sha256(token),
            },
        },
        'type': 'Opaque',
        'data': {'token': base64.b64encode(token.encode()).decode()},
    }


def _resource_ref(resource: dict[str, Any] | None) -> dict[str, Any] | None:
    if resource is None:
        return None
    metadata = resource.get('metadata') or {}
    return {'name': metadata.get('name'), 'uid': metadata.get('uid')}


def _validate_owned_resource(resource: dict[str, Any], allocation: AllocationSnapshot) -> None:
    annotations = (resource.get('metadata') or {}).get('annotations') or {}
    if annotations.get('nexus.io/managed-by') != 'nexus':
        raise AllocationSpecConflictError('provider_resource_not_owned')
    if annotations.get('nexus.io/allocation-id') != allocation.allocation_id:
        raise AllocationSpecConflictError('provider_resource_allocation_conflict')
    if annotations.get('nexus.io/allocation-spec-hash') != allocation.allocation_spec_hash:
        raise AllocationSpecConflictError('provider_resource_spec_hash_conflict')


__all__ = [
    'AllocationSpecConflictError',
    'EnvironmentProvisioningTimeoutError',
    'KubernetesEnvironmentAllocator',
    'PredefinedEnvironmentAllocator',
    'StatefulEnvironmentAllocator',
    'TkexEnvironmentAllocator',
]
