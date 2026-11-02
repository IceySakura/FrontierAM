from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from nexus.distributed.config import KubernetesProviderCredential, ProviderCredential, TkexProviderCredential
from nexus.distributed.environment.provider_config import KubernetesProviderConfig, TkexProviderConfig
from nexus.distributed.errors import DomainConflictError


@dataclass(frozen=True, slots=True)
class InClusterKubernetesCredential:
    token_file: Path
    ca_file: Path


class ProviderCredentialResolver:
    def __init__(
        self,
        *,
        credential_lookup: Callable[[str], ProviderCredential | None],
        service_account_token_file: str | Path = '/var/run/secrets/kubernetes.io/serviceaccount/token',
        service_account_ca_file: str | Path = '/var/run/secrets/kubernetes.io/serviceaccount/ca.crt',
    ) -> None:
        self.credential_lookup = credential_lookup
        self.service_account_token_file = Path(service_account_token_file)
        self.service_account_ca_file = Path(service_account_ca_file)

    def resolve_kubernetes(
        self,
        config: KubernetesProviderConfig,
    ) -> InClusterKubernetesCredential | KubernetesProviderCredential:
        if config.auth_type == 'in_cluster':
            token_file = self.service_account_token_file
            ca_file = self.service_account_ca_file
            self._require_file(token_file, credential_ref='in_cluster')
            self._require_file(ca_file, credential_ref='in_cluster')
            return InClusterKubernetesCredential(token_file=token_file, ca_file=ca_file)
        assert config.credential_ref is not None
        credential = self.credential_lookup(config.credential_ref)
        if not isinstance(credential, KubernetesProviderCredential):
            raise DomainConflictError('provider_credential_unavailable', detail=config.credential_ref)
        return credential

    def resolve_tkex(self, config: TkexProviderConfig) -> TkexProviderCredential:
        credential = self.credential_lookup(config.credential_ref)
        if not isinstance(credential, TkexProviderCredential):
            raise DomainConflictError('provider_credential_unavailable', detail=config.credential_ref)
        return credential

    @staticmethod
    def _require_file(path: Path, *, credential_ref: str) -> None:
        if not path.is_file():
            raise DomainConflictError('provider_credential_unavailable', detail=credential_ref)


__all__ = ['InClusterKubernetesCredential', 'ProviderCredentialResolver']
