"""七彩石环境变量读取与全局客户端初始化测试。"""

from __future__ import annotations

import pytest

from pydantic import BaseModel

from infra.config import (
    RainbowError,
    get_rainbow_client,
    init_rainbow,
    is_rainbow_ready,
    load_from_rainbow,
    rainbow_settings_from_env,
    reset_rainbow,
)


class Sample(BaseModel):
    token: str | None = None
    region: str = 'ap-guangzhou'


_RAINBOW_ENV = (
    'RAINBOW_APP_ID',
    'RAINBOW_USER_ID',
    'RAINBOW_SECRET_KEY',
    'RAINBOW_GROUP',
    'RAINBOW_ENV_NAME',
    'RAINBOW_CONNECT_STR',
    'RAINBOW_CACHE_PATH',
    'RAINBOW_LOG_PATH',
)


@pytest.fixture(autouse=True)
def _clean_global_client() -> object:
    """每个用例前后都重置全局客户端，避免相互影响。"""
    reset_rainbow()
    yield
    reset_rainbow()


def _clear_rainbow_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _RAINBOW_ENV:
        monkeypatch.delenv(name, raising=False)


def test_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_rainbow_env(monkeypatch)
    monkeypatch.setenv('RAINBOW_APP_ID', 'app-1')
    monkeypatch.setenv('RAINBOW_GROUP', 'g-1')
    monkeypatch.setenv('RAINBOW_ENV_NAME', 'Dev')
    monkeypatch.setenv('RAINBOW_USER_ID', 'u-1')
    monkeypatch.setenv('RAINBOW_SECRET_KEY', 's-1')

    settings = rainbow_settings_from_env()
    assert settings.app_id == 'app-1'
    assert settings.group == 'g-1'
    assert settings.env_name == 'Dev'
    assert settings.enabled is True


def test_settings_disabled_when_required_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_rainbow_env(monkeypatch)
    monkeypatch.setenv('RAINBOW_APP_ID', 'app-1')
    # 缺 group / env_name
    assert rainbow_settings_from_env().enabled is False


def test_connect_str_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_rainbow_env(monkeypatch)
    assert rainbow_settings_from_env().connect_str.startswith('http')


def test_custom_cache_and_log_path(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_rainbow_env(monkeypatch)
    monkeypatch.setenv('RAINBOW_CACHE_PATH', '/var/cache/rb/')
    monkeypatch.setenv('RAINBOW_LOG_PATH', '/var/log/rb/')

    settings = rainbow_settings_from_env()
    assert settings.cache_path == '/var/cache/rb/'
    assert settings.log_path == '/var/log/rb/'


@pytest.mark.asyncio
async def test_init_raises_when_unconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    """七彩石是硬依赖：未配置即抛异常，让启动失败而非带空配置运行。"""
    _clear_rainbow_env(monkeypatch)
    with pytest.raises(RainbowError, match='settings incomplete'):
        await init_rainbow()
    assert is_rainbow_ready() is False


def test_get_client_raises_before_init() -> None:
    with pytest.raises(RainbowError, match='not initialized'):
        get_rainbow_client()


@pytest.mark.asyncio
async def test_load_raises_before_init(monkeypatch: pytest.MonkeyPatch) -> None:
    """未初始化就加载应显式失败，而非静默返回默认值。"""
    _clear_rainbow_env(monkeypatch)
    with pytest.raises(RainbowError, match='not initialized'):
        await load_from_rainbow(Sample)


@pytest.mark.asyncio
async def test_load_uses_global_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """初始化后 load 走全局客户端，不再重复建连。"""
    import infra.config.env as env_module

    seen: list[bool] = []

    class _FakeClient:
        async def load(
            self,
            model_cls: type[Sample],
            *,
            group: str | None = None,
            key: str | None = None,
            strict: bool = False,
        ) -> Sample:
            seen.append(strict)
            return model_cls(token='from-global')

    monkeypatch.setattr(env_module, '_client', _FakeClient())

    config = await load_from_rainbow(Sample)
    assert config.token == 'from-global'
    assert seen == [False]

    # strict 需透传给底层 client
    await load_from_rainbow(Sample, strict=True)
    assert seen == [False, True]
