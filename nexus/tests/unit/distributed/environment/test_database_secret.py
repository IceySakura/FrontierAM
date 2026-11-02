from __future__ import annotations

import base64
import copy

from nexus.distributed.environment.database_secret import EnvironmentDatabaseSecretSynchronizer
from nexus.distributed.environment.kubernetes import WorkloadConflictError, WorkloadNotFoundError


class FakeSecretClient:
    def __init__(self) -> None:
        self.secrets: dict[str, dict] = {}
        self.create_calls = 0
        self.patch_calls = 0

    async def get_secret(self, name: str) -> dict:
        try:
            return copy.deepcopy(self.secrets[name])
        except KeyError as exc:
            raise WorkloadNotFoundError(name) from exc

    async def create_secret(self, body: dict) -> dict:
        self.create_calls += 1
        name = body['metadata']['name']
        if name in self.secrets:
            raise WorkloadConflictError(name)
        self.secrets[name] = copy.deepcopy(body)
        return copy.deepcopy(body)

    async def patch_secret(self, name: str, body: dict) -> dict:
        self.patch_calls += 1
        existing = self.secrets[name]
        existing.setdefault('metadata', {}).setdefault('labels', {}).update(body['metadata'].get('labels', {}))
        existing['metadata'].setdefault('annotations', {}).update(body['metadata'].get('annotations', {}))
        existing.setdefault('data', {}).update(body.get('data', {}))
        return copy.deepcopy(existing)


async def test_database_secret_synchronizer_creates_and_reuses_shared_secret() -> None:
    client = FakeSecretClient()
    synchronizer = EnvironmentDatabaseSecretSynchronizer(
        namespace='nexus-workloads',
        app_name='nexus-distributed',
        release_name='runner-dev',
        secret_name='nexus-database',
        secret_key='database-url',
        database_url='mysql+asyncmy://nexus:secret@database/nexus',
    )

    await synchronizer.ensure(client)
    await synchronizer.ensure(client)

    secret = client.secrets['nexus-database']
    assert secret['metadata']['namespace'] == 'nexus-workloads'
    assert secret['metadata']['labels']['app.kubernetes.io/name'] == 'nexus-distributed'
    assert secret['metadata']['labels']['app.kubernetes.io/instance'] == 'runner-dev'
    assert secret['metadata']['annotations']['nexus.io/shared-secret'] == 'database-url'
    assert base64.b64decode(secret['data']['database-url']).decode() == 'mysql+asyncmy://nexus:secret@database/nexus'
    assert client.create_calls == 1
    assert client.patch_calls == 0
    assert 'mysql+asyncmy' not in repr(synchronizer)


async def test_database_secret_synchronizer_repairs_key_without_removing_cluster_data() -> None:
    client = FakeSecretClient()
    client.secrets['nexus-database'] = {
        'apiVersion': 'v1',
        'kind': 'Secret',
        'metadata': {
            'name': 'nexus-database',
            'namespace': 'nexus-workloads',
            'labels': {'cluster-owned': 'true'},
        },
        'type': 'Opaque',
        'data': {
            'database-url': base64.b64encode(b'old-url').decode(),
            'cluster-key': base64.b64encode(b'keep-me').decode(),
        },
    }
    synchronizer = EnvironmentDatabaseSecretSynchronizer(
        namespace='nexus-workloads',
        app_name='nexus-distributed',
        release_name='runner-dev',
        secret_name='nexus-database',
        secret_key='database-url',
        database_url='mysql+asyncmy://nexus:new-secret@database/nexus',
    )

    await synchronizer.ensure(client)
    await synchronizer.ensure(client)

    secret = client.secrets['nexus-database']
    assert base64.b64decode(secret['data']['database-url']).decode() == (
        'mysql+asyncmy://nexus:new-secret@database/nexus'
    )
    assert base64.b64decode(secret['data']['cluster-key']).decode() == 'keep-me'
    assert secret['metadata']['labels']['cluster-owned'] == 'true'
    assert client.create_calls == 0
    assert client.patch_calls == 1
