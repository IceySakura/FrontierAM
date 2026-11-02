"""label 热更新接线测试：holder 与 SandboxService 的协作。

热更新机制本身（字段替换、hook 容错、watch 开关）在
`services/packages/infra/tests/test_holder.py` 覆盖，此处只验证接线。
"""

from __future__ import annotations

import pytest

from infra.config import ConfigHolder, bind
from label.config import LabelConfig, ProviderConfig, load_provider
from label.schemas.sandbox import Provider
from label.services.sandbox import SandboxService


def _holder(config: LabelConfig | None = None) -> ConfigHolder[LabelConfig, ProviderConfig]:
    return ConfigHolder(config or LabelConfig(), field='provider', loader=load_provider)


def test_holder_watches_provider_field() -> None:
    holder = _holder(LabelConfig(port=9999))
    assert holder.current.port == 9999
    assert holder.current.provider.ags.tool_id is None


def test_sandbox_service_reads_config_lazily() -> None:
    """服务持有 holder 而非配置快照，替换后立即看到新值。"""
    holder = _holder()
    service = SandboxService(holder)
    assert service._config.provider.ags.tool_id is None

    holder._config = holder._config.model_copy(
        update={'provider': bind(ProviderConfig, {'ags': {'toolId': 'new-tool'}})}
    )
    assert service._config.provider.ags.tool_id == 'new-tool'


def test_invalidate_clears_provider_cache() -> None:
    service = SandboxService(_holder())
    service._providers[Provider.SMARTRUN] = object()

    service.invalidate()
    assert service._providers == {}


@pytest.mark.asyncio
async def test_reload_hook_invalidates_sandbox_provider_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """重载后按旧凭据构造的 provider 实例必须被丢弃。

    hook 由 SandboxService 在构造时自行注册，调用方无需手动挂。
    """
    holder = _holder(LabelConfig(port=9999))
    service = SandboxService(holder)
    service._providers[Provider.SMARTRUN] = 'stale'

    async def fake_load_provider() -> ProviderConfig:
        return bind(ProviderConfig, {'ags': {'toolId': 'hot-tool'}})

    monkeypatch.setattr(holder, '_loader', fake_load_provider)
    await holder.reload()

    assert holder.current.provider.ags.tool_id == 'hot-tool'
    assert service._providers == {}
    # 启动期参数不参与热更新
    assert holder.current.port == 9999


def test_service_registers_reload_hook_on_init() -> None:
    """构造即注册，避免调用方遗漏导致热更新后仍用旧凭据。"""
    holder = _holder()
    assert holder._on_reload == []

    service = SandboxService(holder)
    assert holder._on_reload == [service.invalidate]
