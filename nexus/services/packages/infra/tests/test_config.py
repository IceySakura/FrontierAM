"""infra.config 绑定逻辑测试：Key 覆盖、嵌套下钻、大小写、空值语义。"""

from __future__ import annotations

import asyncio

from collections.abc import Callable
from typing import Annotated, ClassVar

import pytest

from pydantic import BaseModel

from infra.config import Key, RainbowError, RainbowKey, bind
from infra.config.rainbow import RainbowClient, RainbowSettings, _resolve_target


class DatabaseConfig(BaseModel):
    host: str = 'localhost'
    port: int = 5432
    pool_size: Annotated[int, Key('poolSize')] = 10


class AppConfig(BaseModel):
    rainbow_key: ClassVar[RainbowKey] = RainbowKey(key='app')

    log_level: str = 'INFO'
    database: DatabaseConfig = DatabaseConfig()


def test_field_name_as_key() -> None:
    config = bind(AppConfig, {'log_level': 'DEBUG'})
    assert config.log_level == 'DEBUG'


def test_key_marker_overrides_single_level() -> None:
    config = bind(AppConfig, {'database': {'poolSize': 20}})
    assert config.database.pool_size == 20


def test_nested_model_drills_down_by_field_name() -> None:
    config = bind(AppConfig, {'database': {'host': '10.0.0.1', 'port': 6543}})
    assert config.database.host == '10.0.0.1'
    assert config.database.port == 6543


def test_missing_keys_fall_back_to_defaults() -> None:
    config = bind(AppConfig, {})
    assert config.log_level == 'INFO'
    assert config.database.host == 'localhost'
    assert config.database.pool_size == 10


def test_lookup_is_case_insensitive() -> None:
    config = bind(AppConfig, {'LOG_LEVEL': 'WARNING', 'Database': {'POOLSIZE': 30}})
    assert config.log_level == 'WARNING'
    assert config.database.pool_size == 30


def test_empty_string_counts_as_hit() -> None:
    """空串是合法值，覆盖默认值；仅键不存在或 None 才回落。"""
    config = bind(AppConfig, {'database': {'host': ''}})
    assert config.database.host == ''


def test_none_falls_back_to_default() -> None:
    config = bind(AppConfig, {'database': {'host': None}})
    assert config.database.host == 'localhost'


def test_unknown_keys_are_ignored() -> None:
    config = bind(AppConfig, {'log_level': 'DEBUG', 'not_a_field': 'x'})
    assert config.log_level == 'DEBUG'


def test_nested_key_marker_on_model_field() -> None:
    class Outer(BaseModel):
        database: Annotated[DatabaseConfig, Key('db')] = DatabaseConfig()

    config = bind(Outer, {'db': {'host': 'renamed'}})
    assert config.database.host == 'renamed'


def test_non_mapping_for_nested_model_uses_defaults() -> None:
    config = bind(AppConfig, {'database': 'oops'})
    assert config.database.host == 'localhost'


def test_multiple_key_markers_rejected() -> None:
    class Bad(BaseModel):
        value: Annotated[str, Key('a'), Key('b')] = ''

    with pytest.raises(ValueError, match='multiple Key markers'):
        bind(Bad, {})


def test_type_coercion_from_string() -> None:
    config = bind(AppConfig, {'database': {'port': '6543'}})
    assert config.database.port == 6543


def test_resolve_target_prefers_explicit_args() -> None:
    group, key = _resolve_target(AppConfig, 'g-explicit', 'k-explicit', 'g-default')
    assert (group, key) == ('g-explicit', 'k-explicit')


def test_resolve_target_uses_classvar_then_default_group() -> None:
    group, key = _resolve_target(AppConfig, None, None, 'g-default')
    assert (group, key) == ('g-default', 'app')


def test_resolve_target_without_declaration_raises() -> None:
    class Undeclared(BaseModel):
        value: str = ''

    with pytest.raises(RainbowError, match='no rainbow key'):
        _resolve_target(Undeclared, None, None, 'g-default')


class _FakeSdk:
    """模拟 SDK 的整组拉取与监听接口。"""

    def __init__(self, payload: dict[str, str], code: int = 0) -> None:
        self._payload = payload
        self._code = code
        self.listener: Callable[[], None] | None = None

    def get_configs(self, _group: str, _env_name: str) -> dict[str, object]:
        return {
            'code': self._code,
            'message': 'ok' if self._code == 0 else 'boom',
            'data': [{'key_values': [{'key': k, 'value': v} for k, v in self._payload.items()]}],
        }

    def add_listener(self, _group: str, _env_name: str, callback: Callable[[], None] | None = None) -> None:
        self.listener = callback

    def trigger(self) -> None:
        """模拟 SDK 在自有线程中触发变更回调。"""
        assert self.listener is not None
        self.listener()


def _client(sdk: _FakeSdk) -> RainbowClient:
    return RainbowClient(sdk, RainbowSettings(group='g-default', env_name='test'))


@pytest.mark.asyncio
async def test_load_parses_yaml_value() -> None:
    sdk = _FakeSdk({'app': 'log_level: DEBUG\ndatabase:\n  poolSize: 42\n'})
    config = await _client(sdk).load(AppConfig)
    assert config.log_level == 'DEBUG'
    assert config.database.pool_size == 42


@pytest.mark.asyncio
async def test_load_parses_json_value() -> None:
    sdk = _FakeSdk({'app': '{"database": {"host": "json-host"}}'})
    config = await _client(sdk).load(AppConfig)
    assert config.database.host == 'json-host'


@pytest.mark.asyncio
async def test_load_falls_back_to_defaults_on_fetch_failure() -> None:
    config = await _client(_FakeSdk({}, code=-1)).load(AppConfig)
    assert config.log_level == 'INFO'
    assert config.database.host == 'localhost'


@pytest.mark.asyncio
async def test_load_falls_back_when_key_absent() -> None:
    config = await _client(_FakeSdk({'other': 'x: 1'})).load(AppConfig)
    assert config.log_level == 'INFO'


@pytest.mark.asyncio
async def test_load_explicit_key_overrides_classvar() -> None:
    sdk = _FakeSdk({'app_v2': 'log_level: TRACE'})
    config = await _client(sdk).load(AppConfig, key='app_v2')
    assert config.log_level == 'TRACE'


@pytest.mark.asyncio
async def test_watch_registers_listener() -> None:
    sdk = _FakeSdk({'app': 'log_level: DEBUG'})
    await _client(sdk).watch(lambda: None)
    assert callable(sdk.listener)


@pytest.mark.asyncio
async def test_watch_runs_coroutine_callback_from_sdk_thread() -> None:
    """SDK 回调在独立线程触发，协程需投递回注册时的事件循环执行。"""

    sdk = _FakeSdk({'app': 'log_level: DEBUG'})
    seen: list[str] = []

    async def on_change() -> None:
        seen.append('reloaded')

    await _client(sdk).watch(on_change)
    await asyncio.to_thread(sdk.trigger)
    assert seen == ['reloaded']


@pytest.mark.asyncio
async def test_watch_callback_exception_is_swallowed() -> None:

    sdk = _FakeSdk({'app': 'log_level: DEBUG'})

    async def bad() -> None:
        raise RuntimeError('boom')

    await _client(sdk).watch(bad)
    await asyncio.to_thread(sdk.trigger)  # 不应向外抛出
