from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from nexus.distributed.config.views import EnvironmentConfigView
from nexus.distributed.environment.credentials import InClusterKubernetesCredential, ProviderCredentialResolver
from nexus.distributed.environment.kubernetes import KubernetesWorkloadClient
from nexus.distributed.environment.ports import (
    ResolvedWorkloadClientPort,
    WorkloadClient,
    WorkloadClientResolverPort,
)
from nexus.distributed.environment.provider_config import (
    KubernetesProviderConfig,
    ProviderConfig,
    TkexProviderConfig,
    parse_provider_snapshot,
)
from nexus.distributed.environment.tkex import TkexWorkloadClient
from nexus.distributed.errors import DomainConflictError, DomainValidationError
from nexus.distributed.utils.identity import canonical_sha256


@dataclass(frozen=True, slots=True)
class ResolvedWorkloadClient(ResolvedWorkloadClientPort):
    provider_config: ProviderConfig
    client: WorkloadClient


class WorkloadClientResolver(WorkloadClientResolverPort):
    """Resolve immutable allocation snapshots to shared transport clients."""

    def __init__(
        self,
        *,
        config: EnvironmentConfigView,
        credential_resolver: ProviderCredentialResolver | None = None,
        clients: Mapping[tuple[str, str], WorkloadClient] | None = None,
    ) -> None:
        self.credential_resolver = credential_resolver or ProviderCredentialResolver(
            credential_lookup=config.resolve_provider_credential,
            service_account_token_file=config.kubernetes_token_file,
            service_account_ca_file=config.kubernetes_ca_file,
        )
        self._registered_clients = dict(clients or {})
        self._resolved_clients: dict[str, WorkloadClient] = {}
        self._closed = False

    def resolve(
        self,
        *,
        provider_type: str,
        provider_config_ref: str | None,
        snapshot: Mapping[str, Any] | None = None,
    ) -> ResolvedWorkloadClient:
        if self._closed:
            raise RuntimeError('workload client resolver is closed')
        if not provider_config_ref:
            raise DomainValidationError('provider_config_ref_required')
        if snapshot is None:
            raise DomainValidationError('allocation_provider_config_snapshot_required')
        try:
            provider_config = parse_provider_snapshot(snapshot, provider_config_ref=provider_config_ref)
        except ValidationError as exc:
            raise DomainValidationError('allocation_provider_config_snapshot_invalid') from exc
        if provider_config.provider_type != provider_type:
            raise DomainConflictError(
                'environment_provider_config_type_mismatch',
                detail=f'{provider_config_ref}:{provider_type}:{provider_config.provider_type}',
            )
        cache_key = _client_cache_key(
            provider_type=provider_type,
            provider_config_ref=provider_config_ref,
            provider_config=provider_config,
        )
        client = self._resolved_clients.get(cache_key)
        if client is None:
            client = self._registered_clients.get((provider_type, provider_config_ref))
            if client is None:
                client = _create_workload_client(provider_config, credential_resolver=self.credential_resolver)
            self._resolved_clients[cache_key] = client
        return ResolvedWorkloadClient(provider_config=provider_config, client=client)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        clients = [*self._registered_clients.values(), *self._resolved_clients.values()]
        seen: set[int] = set()
        for client in clients:
            if id(client) in seen:
                continue
            seen.add(id(client))
            await client.close()


def _client_cache_key(
    *,
    provider_type: str,
    provider_config_ref: str,
    provider_config: ProviderConfig,
) -> str:
    config_hash = canonical_sha256(provider_config.model_dump(mode='json'))
    return f'{provider_type}:{provider_config_ref}:{config_hash}'


def _create_workload_client(
    provider_config: ProviderConfig,
    *,
    credential_resolver: ProviderCredentialResolver,
) -> WorkloadClient:
    if isinstance(provider_config, KubernetesProviderConfig):
        credential = credential_resolver.resolve_kubernetes(provider_config)
        if isinstance(credential, InClusterKubernetesCredential):
            return KubernetesWorkloadClient(
                namespace=provider_config.namespace,
                api_url=provider_config.api_url,
                token_file=str(credential.token_file),
                ca_file=str(credential.ca_file),
                tls_strict=provider_config.tls_strict,
            )
        return KubernetesWorkloadClient(
            namespace=provider_config.namespace,
            api_url=provider_config.api_url,
            token=credential.token.get_secret_value(),
            ca_certificate=credential.ca_certificate.get_secret_value(),
            tls_strict=provider_config.tls_strict,
        )
    if isinstance(provider_config, TkexProviderConfig):
        credential = credential_resolver.resolve_tkex(provider_config)
        return TkexWorkloadClient(
            base_url=provider_config.base_url,
            namespace=provider_config.namespace,
            api_account=provider_config.api_account,
            api_token=credential.token.get_secret_value(),
            cluster_name=provider_config.cluster_name,
            project_name=provider_config.project_name,
        )
    raise AssertionError('unsupported provider config')


__all__ = [
    'ProviderConfig',
    'ResolvedWorkloadClient',
    'WorkloadClient',
    'WorkloadClientResolver',
]
