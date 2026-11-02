from __future__ import annotations

import base64
import hashlib

from typing import Any

from nexus.distributed.environment.kubernetes import (
    WorkloadConflictError,
    WorkloadNotFoundError,
    WorkloadPermanentError,
)
from nexus.distributed.environment.ports import AllocationWorkloadClient


class EnvironmentDatabaseSecretSynchronizer:
    """Ensure one shared database URL Secret exists in a workload namespace."""

    __slots__ = (
        '_encoded_database_url',
        '_identity',
        'app_name',
        'namespace',
        'release_name',
        'secret_key',
        'secret_name',
    )

    def __init__(
        self,
        *,
        namespace: str,
        app_name: str,
        release_name: str,
        secret_name: str,
        secret_key: str,
        database_url: str,
    ) -> None:
        if not namespace or not app_name or not release_name or not secret_name or not secret_key or not database_url:
            raise ValueError('environment_database_secret_configuration_incomplete')
        self.namespace = namespace
        self.app_name = app_name
        self.release_name = release_name
        self.secret_name = secret_name
        self.secret_key = secret_key
        self._encoded_database_url = base64.b64encode(database_url.encode()).decode()
        self._identity = hashlib.sha256(database_url.encode()).hexdigest()

    def __repr__(self) -> str:
        return (
            f'{type(self).__name__}(namespace={self.namespace!r}, '
            f'app_name={self.app_name!r}, release_name={self.release_name!r}, '
            f'secret_name={self.secret_name!r}, secret_key={self.secret_key!r})'
        )

    async def ensure(self, client: AllocationWorkloadClient) -> None:
        desired = self._desired_secret()
        try:
            existing = await client.get_secret(self.secret_name)
        except WorkloadNotFoundError:
            try:
                existing = await client.create_secret(desired)
            except WorkloadConflictError:
                existing = await client.get_secret(self.secret_name)
        if not self._matches(existing):
            existing = await client.patch_secret(self.secret_name, desired)
        if not self._matches(existing):
            raise WorkloadPermanentError('environment_database_secret_identity_mismatch')

    def _desired_secret(self) -> dict[str, Any]:
        return {
            'apiVersion': 'v1',
            'kind': 'Secret',
            'metadata': {
                'name': self.secret_name,
                'namespace': self.namespace,
                'labels': {
                    'app.kubernetes.io/name': self.app_name,
                    'app.kubernetes.io/instance': self.release_name,
                    'app.kubernetes.io/component': 'database-bootstrap',
                    'app.kubernetes.io/managed-by': 'nexus',
                },
                'annotations': {
                    'nexus.io/managed-by': 'nexus',
                    'nexus.io/shared-secret': 'database-url',
                    'nexus.io/credential-sha256': self._identity,
                },
            },
            'type': 'Opaque',
            'data': {self.secret_key: self._encoded_database_url},
        }

    def _matches(self, secret: dict[str, Any]) -> bool:
        metadata = secret.get('metadata') or {}
        annotations = metadata.get('annotations') or {}
        data = secret.get('data') or {}
        return (
            metadata.get('name') == self.secret_name
            and metadata.get('namespace', self.namespace) == self.namespace
            and annotations.get('nexus.io/credential-sha256') == self._identity
            and data.get(self.secret_key) == self._encoded_database_url
        )


__all__ = ['EnvironmentDatabaseSecretSynchronizer']
