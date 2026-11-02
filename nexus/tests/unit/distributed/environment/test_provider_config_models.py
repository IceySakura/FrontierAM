from __future__ import annotations

import pytest

from pydantic import ValidationError

from nexus.distributed.environment.provider_config import (
    KubernetesProviderConfig,
    TkexProviderConfig,
    TkexZhiyanLogConfig,
    parse_provider_config,
    parse_provider_snapshot,
)


def test_kubernetes_in_cluster_rejects_external_api_and_credential_ref() -> None:
    with pytest.raises(ValidationError, match='in_cluster_kubernetes_credentials_must_be_implicit'):
        KubernetesProviderConfig.model_validate(
            {
                'provider_type': 'kubernetes',
                'namespace': 'nexus-workloads',
                'auth_type': 'in_cluster',
                'api_url': 'https://kubernetes.example',
                'credential_ref': 'kube-shanghai',
            }
        )


def test_kubernetes_secret_auth_requires_api_url_and_credential_ref() -> None:
    with pytest.raises(ValidationError, match='kubernetes_secret_credentials_required'):
        KubernetesProviderConfig.model_validate(
            {
                'provider_type': 'kubernetes',
                'namespace': 'nexus-workloads',
                'auth_type': 'secret',
            }
        )


def test_provider_config_parser_returns_discriminated_model() -> None:
    config = parse_provider_config(
        {
            'provider_type': 'tkex',
            'namespace': 'nexus-workloads',
            'base_url': 'https://tkex.example',
            'api_account': 'nexus-runner',
            'credential_ref': 'tkex-shanghai',
            'cluster_name': 'cls-shanghai',
            'project_name': 'nexus',
        }
    )

    assert isinstance(config, TkexProviderConfig)
    assert config.credential_ref == 'tkex-shanghai'


def test_tkex_provider_config_accepts_only_official_zhiyan_log_sidecar() -> None:
    config = parse_provider_config(
        {
            'provider_type': 'tkex',
            'namespace': 'nexus-workloads',
            'base_url': 'https://tkex.example',
            'api_account': 'nexus-runner',
            'credential_ref': 'tkex-shanghai',
            'cluster_name': 'cls-shanghai',
            'project_name': 'nexus',
            'zhiyan_log': {
                'log_name': 'dev_nexus_dynamic_manager_log',
                'agent_image': 'mirrors.tencent.com/zhiyan-agent/agent-integration:0.0.12',
                'volume_size_bytes': 4 * 1024**3,
            },
        }
    )

    assert isinstance(config, TkexProviderConfig)
    assert config.zhiyan_log == TkexZhiyanLogConfig(
        log_name='dev_nexus_dynamic_manager_log',
        agent_image='mirrors.tencent.com/zhiyan-agent/agent-integration:0.0.12',
        volume_size_bytes=4 * 1024**3,
    )

    with pytest.raises(ValueError, match='String should match pattern'):
        parse_provider_config(
            {
                **config.model_dump(mode='json', exclude={'zhiyan_log'}),
                'zhiyan_log': {
                    'log_name': 'dev_nexus_dynamic_manager_log',
                    'agent_image': 'registry.example/untrusted-agent:latest',
                    'volume_size_bytes': 4 * 1024**3,
                },
            }
        )
    for invalid_size in (True, '4294967296'):
        with pytest.raises(ValidationError, match='volume_size_bytes'):
            parse_provider_config(
                {
                    **config.model_dump(mode='json', exclude={'zhiyan_log'}),
                    'zhiyan_log': {
                        'log_name': 'dev_nexus_dynamic_manager_log',
                        'agent_image': 'mirrors.tencent.com/zhiyan-agent/agent-integration:0.0.12',
                        'volume_size_bytes': invalid_size,
                    },
                }
            )


def test_remote_kubernetes_endpoint_requires_https() -> None:
    with pytest.raises(ValidationError, match='provider_endpoint_must_use_https'):
        parse_provider_config(
            {
                'provider_type': 'kubernetes',
                'namespace': 'nexus-workloads',
                'auth_type': 'secret',
                'api_url': 'http://kubernetes.example',
                'credential_ref': 'kube-shanghai',
            }
        )


def test_tkex_endpoint_accepts_official_http_transport() -> None:
    config = parse_provider_config(
        {
            'provider_type': 'tkex',
            'namespace': 'nexus-workloads',
            'base_url': 'http://tke.kubernetes.woa.com',
            'api_account': 'nexus-runner',
            'credential_ref': 'tkex-shanghai',
            'cluster_name': 'cls-shanghai',
            'project_name': 'nexus',
        }
    )

    assert isinstance(config, TkexProviderConfig)
    assert config.base_url == 'http://tke.kubernetes.woa.com'


def test_tkex_endpoint_rejects_non_http_transport() -> None:
    with pytest.raises(ValidationError, match='provider_endpoint_must_use_http_transport'):
        parse_provider_config(
            {
                'provider_type': 'tkex',
                'namespace': 'nexus-workloads',
                'base_url': 'ftp://tkex.example',
                'api_account': 'nexus-runner',
                'credential_ref': 'tkex-shanghai',
                'cluster_name': 'cls-shanghai',
                'project_name': 'nexus',
            }
        )


@pytest.mark.parametrize(
    'legacy_fields',
    [
        {'api_token_file': '/var/run/secrets/nexus-tkex/api-token'},
        {
            'token_file': '/var/run/secrets/kubernetes.io/serviceaccount/token',
            'ca_file': '/var/run/secrets/kubernetes.io/serviceaccount/ca.crt',
        },
    ],
)
def test_legacy_provider_snapshots_require_offline_drain(legacy_fields: dict[str, str]) -> None:
    with pytest.raises(ValueError, match='legacy provider snapshot.*drain allocations before upgrade'):
        parse_provider_snapshot(
            {'provider_type': 'legacy', **legacy_fields},
            provider_config_ref='legacy/provider',
        )


def test_builder_collection_configuration_does_not_enable_manager_collection():
    config = parse_provider_config({
        'provider_type': 'tkex',
        'namespace': 'test',
        'base_url': 'https://tkex.example.com',
        'api_account': 'test',
        'credential_ref': 'tkex-test',
        'cluster_name': 'test',
        'project_name': 'test',
        'builder_zhiyan_log': {
            'log_name': 'build-flow',
            'agent_image': 'mirrors.tencent.com/zhiyan-agent/agent-integration:0.0.12',
            'volume_size_bytes': 16777216,
        },
    })
    assert isinstance(config, TkexProviderConfig)
    assert config.zhiyan_log is None
    assert config.builder_zhiyan_log is not None
    restored = parse_provider_snapshot(config.model_dump(mode='json'), provider_config_ref='tkex/test')
    assert restored == config
