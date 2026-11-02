"""label 配置测试：provider 绑定七彩石数据，启动期参数取自环境变量。"""

from __future__ import annotations

import pytest

from infra.config import bind
from label.config import AgsConfig, DevCloudConfig, LabelConfig, ProviderConfig


# 七彩石 `provider` 键的值（YAML 解析后的嵌套结构）。
PROVIDER_DATA = {
    'ags': {'secretId': 'rb-ags-id', 'toolId': 'rb-tool'},
    'devcloud': {'appId': 'rb-app', 'project': 'rb-project'},
    'smartrun': {'token': 'rb-smartrun'},
}


def test_provider_binds_camel_case_keys() -> None:
    provider = bind(ProviderConfig, PROVIDER_DATA)
    assert provider.ags.secret_id == 'rb-ags-id'
    assert provider.ags.tool_id == 'rb-tool'
    assert provider.devcloud.app_id == 'rb-app'
    assert provider.devcloud.project == 'rb-project'
    assert provider.smartrun.token == 'rb-smartrun'


def test_provider_defaults_when_data_empty() -> None:
    provider = bind(ProviderConfig, {})
    assert provider.ags.secret_id is None
    assert provider.ags.region == 'ap-guangzhou'
    assert provider.devcloud.pipeline_id == 'llm_t3'
    assert provider.devcloud.control_api_base_url == 'http://ratel.devcloud.woa.com'
    assert provider.smartrun.api_base_url == 'https://smartrun.woa.com'
    assert provider.smartrun.duration == 1800


def test_partial_data_keeps_other_defaults() -> None:
    provider = bind(ProviderConfig, {'ags': {'toolId': 'only-tool'}})
    assert provider.ags.tool_id == 'only-tool'
    assert provider.ags.secret_id is None
    assert provider.ags.region == 'ap-guangzhou'


def test_nested_models_are_typed() -> None:
    provider = bind(ProviderConfig, PROVIDER_DATA)
    assert isinstance(provider.ags, AgsConfig)
    assert isinstance(provider.devcloud, DevCloudConfig)


def test_rainbow_key_declared_on_provider() -> None:
    assert ProviderConfig.rainbow_key.key == 'provider'
    assert ProviderConfig.rainbow_key.group is None


def test_from_env_reads_bootstrap_params(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('NEXUS_LABEL_HOST', '127.0.0.1')
    monkeypatch.setenv('NEXUS_LABEL_PORT', '9000')
    monkeypatch.setenv('NEXUS_LABEL_LOG_LEVEL', 'DEBUG')

    config = LabelConfig.from_env()
    assert config.host == '127.0.0.1'
    assert config.port == 9000
    assert config.log_level == 'DEBUG'


def test_from_env_defaults_without_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ('NEXUS_LABEL_HOST', 'NEXUS_LABEL_PORT', 'NEXUS_LABEL_LOG_LEVEL'):
        monkeypatch.delenv(name, raising=False)

    config = LabelConfig.from_env()
    assert config.host == '0.0.0.0'
    assert config.port == 8080
    assert config.log_level == 'INFO'


def test_from_env_leaves_provider_at_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """from_env 只读启动期参数，provider 由 lifespan 从七彩石加载。"""
    monkeypatch.setenv('NEXUS_LABEL_PORT', '9000')

    config = LabelConfig.from_env()
    assert config.provider.ags.secret_id is None
    assert config.provider.devcloud.project == ''
    assert config.provider.ags.region == 'ap-guangzhou'
