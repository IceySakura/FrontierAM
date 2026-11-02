"""七彩石（Rainbow）配置中心接入。

用法：

    class ProviderConfig(BaseModel):
        rainbow_key: ClassVar[RainbowKey] = RainbowKey(key='provider')
        ags: AgsConfig = AgsConfig()

    client = await create_client(settings)
    config = await client.load(ProviderConfig)                    # 用 ClassVar 声明
    config = await client.load(ProviderConfig, key='provider_v2')  # 显式覆盖，最高优

设计要点：
- SDK（rainbow-cpp-sdk）为 C++ 封装，底层 client 必须为全局单例且不可在 fork 后的
  子进程复用，否则监听线程失效。本模块通过模块级 `_sdk_client` 持有引用避免被 GC 回收；
  fork 后的子进程（如 uvicorn 多 worker）需各自调用 `create_client()`。
- SDK 接口是阻塞的，统一用 `asyncio.to_thread` 包装，避免堵塞事件循环。
- 不做结果缓存：每次 `load()` 都实拉一次，保证读到最新配置。
"""

from __future__ import annotations

import asyncio
import logging
import threading

from collections.abc import Callable
from typing import Any, TypeVar

import yaml

from pydantic import BaseModel

from infra.config.keys import RAINBOW_KEY_ATTR, RainbowKey
from infra.config.resolver import bind


logger = logging.getLogger(__name__)

# 本地/无北极星环境使用 http 域名直连；北极星寻址串形如 65026305:65536。
DEFAULT_CONNECT_STR = 'http://api.rainbow.woa.com:8080'

ModelT = TypeVar('ModelT', bound=BaseModel)

# 配置变更回调若为协程，等待其在主事件循环中执行完成的上限（秒）。
_WATCH_CALLBACK_TIMEOUT = 30.0

_sdk_client: Any = None
_lock = threading.Lock()


class RainbowError(RuntimeError):
    """七彩石不可用（未配置、SDK 缺失或拉取失败）。"""


class RainbowSettings(BaseModel):
    """七彩石接入的引导配置，只能来自环境变量（无法自举）。"""

    app_id: str | None = None
    user_id: str | None = None
    secret_key: str | None = None
    group: str | None = None
    env_name: str | None = None
    connect_str: str = DEFAULT_CONNECT_STR
    cache_path: str = '/tmp/rainbow_cache/'
    log_path: str = '/tmp/rainbow_log/'

    @property
    def enabled(self) -> bool:
        return bool(self.app_id and self.group and self.env_name)


def _build_sdk_client(settings: RainbowSettings) -> Any:
    try:
        from rainbow_cpplib.rainbow_client import RainbowClient as _SdkClient
    except ImportError as exc:  # pragma: no cover - 依赖缺失路径
        raise RainbowError('rainbow-cpp-sdk not installed') from exc

    init_config: dict[str, Any] = {
        'connectStr': settings.connect_str,
        'isUsingPolaris': not settings.connect_str.startswith('http'),
        'fileCachePath': settings.cache_path,
        'log_path': settings.log_path,
        'tokenConfig': {
            'isOpenSign': bool(settings.user_id and settings.secret_key),
            'app_id': settings.app_id,
            'user_id': settings.user_id or '',
            'secret_key': settings.secret_key or '',
        },
    }
    return _SdkClient(init_config)


def _get_sdk_client(settings: RainbowSettings) -> Any:
    """获取全局单例底层 client。fork 后的进程需各自首次调用。"""
    global _sdk_client
    with _lock:
        if _sdk_client is None:
            _sdk_client = _build_sdk_client(settings)
        return _sdk_client


def _resolve_target(
    model_cls: type[BaseModel],
    group: str | None,
    key: str | None,
    default_group: str,
) -> tuple[str, str]:
    """确定要拉取的 (group, key)：显式传参 > model 上的 ClassVar 声明。"""
    declared: RainbowKey | None = getattr(model_cls, RAINBOW_KEY_ATTR, None)
    if declared is not None and not isinstance(declared, RainbowKey):
        declared = None

    resolved_key = key or (declared.key if declared else None)
    if not resolved_key:
        raise RainbowError(
            f'{model_cls.__name__} has no rainbow key: '
            f'pass key= explicitly or declare `{RAINBOW_KEY_ATTR}: ClassVar[RainbowKey]`'
        )

    resolved_group = group or (declared.group if declared else None) or default_group
    if not resolved_group:
        raise RainbowError(f'no rainbow group for {model_cls.__name__}: pass group= or configure RAINBOW_GROUP')

    return resolved_group, resolved_key


class RainbowClient:
    """七彩石客户端。按 model 声明的 group/key 拉取配置并绑定。"""

    def __init__(self, sdk_client: Any, settings: RainbowSettings) -> None:
        self._sdk = sdk_client
        self._default_group = settings.group or ''
        self._env_name = settings.env_name or ''

    async def load(
        self,
        model_cls: type[ModelT],
        *,
        group: str | None = None,
        key: str | None = None,
        strict: bool = False,
    ) -> ModelT:
        """拉取配置并绑定到 model。

        group/key 显式传参优先，否则取 model 上的 `rainbow_key` ClassVar 声明。

        `strict=False`（默认）时拉取失败记录告警并返回全默认值的 model，
        适合热更新等"失败也要继续跑"的场景；`strict=True` 时直接抛
        `RainbowError`，供启动期据此终止启动，避免带空配置接流量。
        """
        resolved_group, resolved_key = _resolve_target(model_cls, group, key, self._default_group)

        try:
            data = await self.fetch(resolved_group, resolved_key)
        except RainbowError:
            if strict:
                raise
            logger.warning(
                'rainbow config %s/%s unavailable, %s uses defaults',
                resolved_group,
                resolved_key,
                model_cls.__name__,
                exc_info=True,
            )
            data = {}

        return bind(model_cls, data)

    async def fetch(self, group: str, key: str) -> dict[str, Any]:
        """拉取指定 key 的值并解析为嵌套 dict，失败抛 RainbowError。"""
        raw = await asyncio.to_thread(self._get_raw_value, group, key)
        return self._parse(key, raw)

    async def watch(self, callback: Callable[[], Any], group: str | None = None) -> None:
        """注册配置变更回调，实现热更新。回调内自行重新 `load()` 所需 model。

        SDK 的回调在其自有线程中触发，因此协程回调需投递回注册时所在的事件循环
        （用 `run_coroutine_threadsafe`），不能直接 `asyncio.run`。
        """
        resolved_group = group or self._default_group
        loop = asyncio.get_running_loop()

        def _on_change(*_args: Any, **_kwargs: Any) -> None:
            try:
                result = callback()
                if asyncio.iscoroutine(result):
                    future = asyncio.run_coroutine_threadsafe(result, loop)
                    future.result(timeout=_WATCH_CALLBACK_TIMEOUT)
                logger.info('rainbow config changed, reload done')
            except Exception:
                logger.exception('failed to handle rainbow config change')

        await asyncio.to_thread(self._sdk.add_listener, resolved_group, self._env_name, _on_change)
        logger.info('watching rainbow group %s for config changes', resolved_group)

    def _get_raw_value(self, group: str, key: str) -> str:
        """同步拉取 group 下全部配置并挑出目标 key，在 to_thread 中执行。

        SDK 只提供 `get_configs(group, env_name)`（整组拉取），无单键接口。
        """
        result = self._sdk.get_configs(group, self._env_name)
        if not isinstance(result, dict) or result.get('code') != 0:
            message = result.get('message') if isinstance(result, dict) else 'unknown'
            raise RainbowError(f'fetch rainbow configs {group} failed: {message}')

        for item in result.get('data') or []:
            for kv in item.get('key_values', []):
                if kv.get('key') == key:
                    return kv.get('value', '')

        raise RainbowError(f'rainbow key {key!r} not found in group {group}')

    def _parse(self, key: str, raw: str) -> dict[str, Any]:
        """七彩石的值为文本，按 YAML 解析（兼容 JSON）为嵌套 dict。"""
        if not raw:
            return {}
        try:
            parsed = yaml.safe_load(raw)
        except yaml.YAMLError as exc:
            raise RainbowError(f'failed to parse rainbow config {key} as YAML/JSON') from exc

        if not isinstance(parsed, dict):
            raise RainbowError(f'rainbow config {key} is not a mapping')
        return parsed


async def create_client(settings: RainbowSettings) -> RainbowClient:
    """初始化七彩石客户端。settings 只在此处使用一次。"""
    if not settings.enabled:
        raise RainbowError('rainbow settings incomplete')
    sdk_client = await asyncio.to_thread(_get_sdk_client, settings)
    return RainbowClient(sdk_client, settings)
