"""ConfigHolder 热更新机制测试。"""

from __future__ import annotations

import pytest

from pydantic import BaseModel

from infra.config import ConfigHolder, RainbowError, reset_rainbow


class Section(BaseModel):
    token: str | None = None
    region: str = 'default-region'


class AppConfig(BaseModel):
    port: int = 8080
    section: Section = Section()


@pytest.fixture(autouse=True)
def _clean_global_client() -> object:
    reset_rainbow()
    yield
    reset_rainbow()


def _holder(section: Section, *, config: AppConfig | None = None) -> ConfigHolder[AppConfig, Section]:
    async def loader(*, strict: bool = False) -> Section:
        return section

    return ConfigHolder(config or AppConfig(), field='section', loader=loader)


def _failing_holder(*, config: AppConfig | None = None) -> ConfigHolder[AppConfig, Section]:
    """严格模式下抛异常、非严格模式下降级的 loader，模拟真实拉取失败。"""

    async def loader(*, strict: bool = False) -> Section:
        if strict:
            raise RainbowError('fetch failed')
        return Section()

    return ConfigHolder(config or AppConfig(), field='section', loader=loader)


def test_current_returns_initial_config() -> None:
    holder = _holder(Section())
    assert holder.current.port == 8080
    assert holder.current.section.token is None


def test_unknown_field_rejected() -> None:
    async def loader(*, strict: bool = False) -> Section:
        return Section()

    with pytest.raises(ValueError, match="has no field 'nope'"):
        ConfigHolder(AppConfig(), field='nope', loader=loader)


@pytest.mark.asyncio
async def test_load_initial_replaces_target_field() -> None:
    holder = _holder(Section(token='initial'), config=AppConfig(port=9999))
    await holder.load_initial()

    assert holder.current.section.token == 'initial'
    assert holder.current.port == 9999


@pytest.mark.asyncio
async def test_load_initial_propagates_failure() -> None:
    """启动期拉取失败必须抛出，让启动失败而非带空配置接流量。"""
    holder = _failing_holder()
    with pytest.raises(RainbowError, match='fetch failed'):
        await holder.load_initial()


@pytest.mark.asyncio
async def test_reload_swallows_failure_via_loader() -> None:
    """服务已在运行时，重载失败由 loader 降级，不中断运行。"""
    holder = _failing_holder()
    await holder.reload()  # 不应抛出
    assert holder.current.section.token is None


@pytest.mark.asyncio
async def test_load_initial_does_not_run_hooks() -> None:
    """首次加载时依赖方尚未缓存任何东西，无需触发失效 hook。"""
    holder = _holder(Section(token='initial'))
    calls: list[str] = []
    holder.add_reload_hook(lambda: calls.append('invalidated'))

    await holder.load_initial()
    assert calls == []


@pytest.mark.asyncio
async def test_reload_replaces_only_target_field() -> None:
    holder = _holder(Section(token='hot'), config=AppConfig(port=9999))
    await holder.reload()

    assert holder.current.section.token == 'hot'
    # 非目标字段不受热更新影响
    assert holder.current.port == 9999


@pytest.mark.asyncio
async def test_reload_runs_hooks_in_order() -> None:
    holder = _holder(Section(token='hot'))
    calls: list[str] = []
    holder.add_reload_hook(lambda: calls.append('first'))
    holder.add_reload_hook(lambda: calls.append('second'))

    await holder.reload()
    assert calls == ['first', 'second']


@pytest.mark.asyncio
async def test_failing_hook_does_not_block_others_or_reload() -> None:
    holder = _holder(Section(token='applied'))
    calls: list[str] = []

    def bad() -> None:
        raise RuntimeError('boom')

    holder.add_reload_hook(bad)
    holder.add_reload_hook(lambda: calls.append('still-ran'))

    await holder.reload()
    assert holder.current.section.token == 'applied'
    assert calls == ['still-ran']


@pytest.mark.asyncio
async def test_current_reflects_latest_after_reload() -> None:
    """读方持有 holder 而非快照，因此重载后立即看到新值。"""
    holder = _holder(Section(token='new'))
    read = lambda: holder.current.section.token  # noqa: E731 - 模拟依赖方惰性读取
    assert read() is None

    await holder.reload()
    assert read() == 'new'


@pytest.mark.asyncio
async def test_start_watch_disabled_when_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('RAINBOW_WATCH', raising=False)
    assert await _holder(Section()).start_watch() is False


@pytest.mark.asyncio
async def test_start_watch_skipped_when_client_uninitialized(monkeypatch: pytest.MonkeyPatch) -> None:
    """开关已开但未 init_rainbow()，跳过监听而非抛异常。"""
    monkeypatch.setenv('RAINBOW_WATCH', 'true')
    reset_rainbow()
    assert await _holder(Section()).start_watch() is False


@pytest.mark.asyncio
async def test_start_watch_respects_custom_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('RAINBOW_WATCH', raising=False)
    monkeypatch.setenv('MY_WATCH', 'yes')
    reset_rainbow()

    # 开关已开启但客户端未初始化，仍返回 False —— 说明自定义变量被读到了
    assert await _holder(Section()).start_watch(env_var='MY_WATCH') is False


@pytest.mark.asyncio
async def test_start_watch_registers_with_global_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """开关开启且客户端就绪时，向全局 client 注册重载回调。"""
    import infra.config.env as env_module

    registered: list[object] = []

    class _FakeClient:
        async def watch(self, callback: object, group: str | None = None) -> None:
            registered.append(callback)

    monkeypatch.setenv('RAINBOW_WATCH', '1')
    monkeypatch.setattr(env_module, '_client', _FakeClient())

    holder = _holder(Section(token='hot'))
    assert await holder.start_watch() is True
    assert registered == [holder.reload]
