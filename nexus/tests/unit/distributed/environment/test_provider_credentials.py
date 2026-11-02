from __future__ import annotations

import pytest

from nexus.distributed.config import KubernetesProviderCredential, TkexProviderCredential
from nexus.distributed.environment.credentials import ProviderCredentialResolver
from nexus.distributed.environment.provider_config import KubernetesProviderConfig, TkexProviderConfig
from nexus.distributed.errors import DomainConflictError


def test_provider_credential_resolver_uses_deployment_secret_snapshot() -> None:
    config = KubernetesProviderConfig(
        provider_type='kubernetes',
        namespace='nexus-workloads',
        auth_type='secret',
        api_url='https://kubernetes.example',
        credential_ref='kube-shanghai',
    )
    stored = KubernetesProviderCredential(
        provider_type='kubernetes',
        token='token',
        ca_certificate='ca',
    )

    credential = ProviderCredentialResolver(credential_lookup=lambda ref: stored if ref == 'kube-shanghai' else None)
    resolved = credential.resolve_kubernetes(config)

    assert resolved is stored


def test_provider_credential_resolver_uses_service_account_for_in_cluster(tmp_path) -> None:
    token = tmp_path / 'service-account-token'
    ca = tmp_path / 'service-account-ca'
    token.write_text('token')
    ca.write_text('ca')
    config = KubernetesProviderConfig(
        provider_type='kubernetes',
        namespace='nexus-workloads',
        auth_type='in_cluster',
    )

    credential = ProviderCredentialResolver(
        credential_lookup=lambda _ref: None,
        service_account_token_file=token,
        service_account_ca_file=ca,
    ).resolve_kubernetes(config)

    assert credential.token_file == token
    assert credential.ca_file == ca


def test_provider_credential_resolver_fails_closed_when_secret_is_missing() -> None:
    config = TkexProviderConfig(
        provider_type='tkex',
        namespace='nexus-workloads',
        base_url='https://tkex.example',
        api_account='nexus-runner',
        credential_ref='tkex-shanghai',
        cluster_name='cls-shanghai',
        project_name='nexus',
    )

    with pytest.raises(DomainConflictError, match='provider_credential_unavailable'):
        ProviderCredentialResolver(credential_lookup=lambda _ref: None).resolve_tkex(config)


def test_provider_credential_resolver_rejects_wrong_provider_type() -> None:
    config = KubernetesProviderConfig(
        provider_type='kubernetes',
        namespace='nexus-workloads',
        auth_type='secret',
        api_url='https://kubernetes.example',
        credential_ref='shared-credential',
    )
    stored = TkexProviderCredential(provider_type='tkex', token='token')

    with pytest.raises(DomainConflictError, match='provider_credential_unavailable'):
        ProviderCredentialResolver(credential_lookup=lambda _ref: stored).resolve_kubernetes(config)
