"""lifespan 测试：七彩石为硬依赖，未就绪时启动必须失败。"""

from __future__ import annotations

import pytest

from fastapi.testclient import TestClient

from infra.config import RainbowError, bind, reset_rainbow
from label import LabelConfig, create_app
from label.config import ProviderConfig


_RAINBOW_ENV = ('RAINBOW_APP_ID', 'RAINBOW_GROUP', 'RAINBOW_ENV_NAME', 'RAINBOW_WATCH')


@pytest.fixture(autouse=True)
def _clean_global_client() -> object:
    reset_rainbow()
    yield
    reset_rainbow()


def test_startup_fails_when_rainbow_unconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    """七彩石未配置时启动直接失败，避免带空配置接流量。"""
    for name in _RAINBOW_ENV:
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(RainbowError, match='settings incomplete'), TestClient(create_app(LabelConfig())):
        pass  # pragma: no cover - 启动即失败，不会执行到这里


def test_startup_loads_provider_before_serving(monkeypatch: pytest.MonkeyPatch) -> None:
    """就绪后 provider 必须已加载完毕，否则请求会打到空凭据的服务上。"""
    import label.lifespan as lifespan_module

    loaded = bind(ProviderConfig, {'ags': {'toolId': 'from-rainbow', 'secretId': 'sid'}})

    async def fake_init() -> None:
        return None

    async def fake_load(*, strict: bool = False) -> ProviderConfig:
        assert strict is True, 'startup must load strictly'
        return loaded

    monkeypatch.setattr(lifespan_module, 'init_rainbow', fake_init)
    app = create_app(LabelConfig())
    monkeypatch.setattr(app.state.config_holder, '_loader', fake_load)

    with TestClient(app) as client:
        # ready 返回 200 时配置已就绪
        assert client.get('/health/ready').status_code == 200
        body = client.get('/health/config').json()

    assert body['ags']['tool_id'] == 'from-rainbow'
    assert body['watching'] is False


def test_startup_fails_when_provider_fetch_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """已配置但拉取失败同样让启动失败，由 K8s 重启重试。"""
    import label.lifespan as lifespan_module

    async def fake_init() -> None:
        return None

    async def failing_load(*, strict: bool = False) -> ProviderConfig:
        raise RainbowError('fetch failed')

    monkeypatch.setattr(lifespan_module, 'init_rainbow', fake_init)
    app = create_app(LabelConfig())
    monkeypatch.setattr(app.state.config_holder, '_loader', failing_load)

    with pytest.raises(RainbowError, match='fetch failed'), TestClient(app):
        pass  # pragma: no cover - 启动即失败
