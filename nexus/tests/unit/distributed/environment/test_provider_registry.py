from __future__ import annotations

import pytest

from nexus.distributed.config import DistributedConfig
from nexus.distributed.dependency_build.controller import DependencyBuildController
from nexus.distributed.environment.controller import EnvironmentController
from nexus.distributed.environment.providers import WorkloadClientResolver


class _Client:
    def __init__(self) -> None:
        self.close_calls = 0

    async def close(self) -> None:
        self.close_calls += 1


def _config(tmp_path) -> DistributedConfig:
    del tmp_path
    return DistributedConfig(
        database_url='sqlite+aiosqlite:///nexus.db',
        environment_namespace='workloads',
        environment_database_url_secret_name='nexus-database',
        secrets={
            'provider_credentials': {
                'kube-main': {
                    'provider_type': 'kubernetes',
                    'token': 'token',
                    'ca_certificate': 'ca',
                }
            }
        },
    )


def _snapshot() -> dict[str, object]:
    return {
        'provider_type': 'kubernetes',
        'namespace': 'workloads',
        'auth_type': 'secret',
        'api_url': 'https://kubernetes.example',
        'credential_ref': 'kube-main',
        'manager_resource_request_ratio': 0.25,
    }


def test_workload_client_resolver_reuses_equivalent_snapshot(tmp_path) -> None:
    config = _config(tmp_path)
    client = _Client()
    resolver = WorkloadClientResolver(
        config=config,
        clients={('kubernetes', 'kubernetes/main'): client},
    )

    first = resolver.resolve(
        provider_type='kubernetes',
        provider_config_ref='kubernetes/main',
        snapshot=_snapshot(),
    )
    second = resolver.resolve(
        provider_type='kubernetes',
        provider_config_ref='kubernetes/main',
        snapshot=_snapshot(),
    )

    assert first.client is client
    assert first.client is second.client
    assert first.provider_config == second.provider_config


def test_workload_client_resolver_requires_snapshot(tmp_path) -> None:
    resolver = WorkloadClientResolver(config=_config(tmp_path))

    with pytest.raises(ValueError, match='allocation_provider_config_snapshot_required'):
        resolver.resolve(
            provider_type='kubernetes',
            provider_config_ref='kubernetes/main',
        )


def test_workload_client_resolver_requires_ref_even_with_snapshot(tmp_path) -> None:
    resolver = WorkloadClientResolver(config=_config(tmp_path))

    with pytest.raises(ValueError, match='provider_config_ref_required'):
        resolver.resolve(
            provider_type='kubernetes',
            provider_config_ref=None,
            snapshot=_snapshot(),
        )


def test_workload_client_resolver_builds_client_from_startup_credentials(tmp_path) -> None:
    resolver = WorkloadClientResolver(config=_config(tmp_path))

    resolved = resolver.resolve(
        provider_type='kubernetes',
        provider_config_ref='kubernetes/main',
        snapshot=_snapshot(),
    )

    assert resolved.client.token == 'token'
    assert resolved.client.ca_certificate == 'ca'


async def test_workload_client_resolver_closes_each_client_once(tmp_path) -> None:
    config = _config(tmp_path)
    client = _Client()
    resolver = WorkloadClientResolver(
        config=config,
        clients={
            ('kubernetes', 'kubernetes/main'): client,
            ('kubernetes', 'alias'): client,
        },
    )
    resolver.resolve(
        provider_type='kubernetes',
        provider_config_ref='kubernetes/main',
        snapshot=_snapshot(),
    )

    await resolver.close()
    await resolver.close()

    assert client.close_calls == 1


async def test_workload_client_resolver_rejects_resolve_after_close(tmp_path) -> None:
    resolver = WorkloadClientResolver(config=_config(tmp_path))

    await resolver.close()

    with pytest.raises(RuntimeError, match='workload client resolver is closed'):
        resolver.resolve(
            provider_type='kubernetes',
            provider_config_ref='kubernetes/main',
            snapshot=_snapshot(),
        )


async def test_environment_and_dependency_build_controllers_share_but_do_not_own_resolver(tmp_path) -> None:
    config = _config(tmp_path)
    client = _Client()
    resolver = WorkloadClientResolver(
        config=config,
        clients={('kubernetes', 'kubernetes/main'): client},
    )
    environment = EnvironmentController(
        session_factory=None,  # type: ignore[arg-type]
        config=config,
        client_resolver=resolver,
    )
    dependency_build = DependencyBuildController(
        session_factory=None,  # type: ignore[arg-type]
        config=config,
        client_resolver=resolver,
    )

    allocator = environment._allocator_for(
        'kubernetes',
        'kubernetes/main',
        provider_config_snapshot=_snapshot(),
    )

    assert allocator.client is client
    assert allocator.renderer.resource_request_ratio == 0.25
    assert environment.client_resolver is resolver
    assert dependency_build.client_resolver is resolver
    assert not hasattr(environment, '_dependency_build_controller')

    await environment.close()
    await dependency_build.close()
    assert client.close_calls == 0

    await resolver.close()
    assert client.close_calls == 1
