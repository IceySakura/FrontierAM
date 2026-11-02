"""七彩石引导配置的环境变量读取与全局客户端管理。

七彩石接入参数无法自举（要连配置中心才能读配置），因此固定来自环境变量。
本模块统一约定这些变量名，避免每个服务各写一份。

生命周期：

    await init_rainbow()                         # 启动时一次，失败抛 RainbowError
    provider = await load_from_rainbow(Model)     # 之后任意处，走全局 client

底层 SDK client 为全局单例且不可跨 fork 复用，因此多 worker 部署时
每个子进程都要各自调用 `init_rainbow()`（放在 FastAPI lifespan 内即可）。
"""

from __future__ import annotations

import logging
import os
import threading

from typing import TypeVar

from pydantic import BaseModel

from infra.config.rainbow import (
    DEFAULT_CONNECT_STR,
    RainbowClient,
    RainbowError,
    RainbowSettings,
    create_client,
)


logger = logging.getLogger(__name__)

ModelT = TypeVar('ModelT', bound=BaseModel)

_client: RainbowClient | None = None
_client_lock = threading.Lock()


def rainbow_settings_from_env() -> RainbowSettings:
    """从标准 `RAINBOW_*` 环境变量构建接入配置。

    | 变量 | 说明 |
    |------|------|
    | `RAINBOW_APP_ID` | 应用 ID，必填 |
    | `RAINBOW_GROUP` | 配置分组，必填 |
    | `RAINBOW_ENV_NAME` | 环境名，必填 |
    | `RAINBOW_USER_ID` / `RAINBOW_SECRET_KEY` | 签名凭据，开启鉴权时必填 |
    | `RAINBOW_CONNECT_STR` | 接入地址，缺省用 http 直连 |
    | `RAINBOW_CACHE_PATH` / `RAINBOW_LOG_PATH` | SDK 本地缓存与日志目录 |

    三个必填项缺失时 `enabled` 为 False，调用方应据此跳过七彩石。
    """
    return RainbowSettings(
        app_id=os.environ.get('RAINBOW_APP_ID'),
        user_id=os.environ.get('RAINBOW_USER_ID'),
        secret_key=os.environ.get('RAINBOW_SECRET_KEY'),
        group=os.environ.get('RAINBOW_GROUP'),
        env_name=os.environ.get('RAINBOW_ENV_NAME'),
        connect_str=os.environ.get('RAINBOW_CONNECT_STR') or DEFAULT_CONNECT_STR,
        cache_path=os.environ.get('RAINBOW_CACHE_PATH', '/tmp/rainbow_cache/'),
        log_path=os.environ.get('RAINBOW_LOG_PATH', '/tmp/rainbow_log/'),
    )


async def init_rainbow() -> None:
    """初始化全局七彩石客户端。

    未配置（缺 `RAINBOW_*` 必填项）或初始化失败一律抛 `RainbowError`
    —— 配置中心是硬依赖，起不来比带空配置接流量更安全。
    """
    global _client

    settings = rainbow_settings_from_env()
    if not settings.enabled:
        raise RainbowError('rainbow settings incomplete: RAINBOW_APP_ID/GROUP/ENV_NAME are required')

    if _client is not None:
        logger.debug('rainbow client already initialized')
        return

    client = await create_client(settings)
    with _client_lock:
        _client = client

    logger.info('rainbow client initialized (group=%s env=%s)', settings.group, settings.env_name)


def get_rainbow_client() -> RainbowClient:
    """取全局客户端。未初始化时抛 `RainbowError`。"""
    if _client is None:
        raise RainbowError('rainbow client not initialized; call init_rainbow() during startup')
    return _client


def is_rainbow_ready() -> bool:
    """全局客户端是否已初始化。"""
    return _client is not None


def reset_rainbow() -> None:
    """丢弃全局客户端引用，仅供测试使用。"""
    global _client
    with _client_lock:
        _client = None


async def load_from_rainbow(
    model_cls: type[ModelT],
    *,
    group: str | None = None,
    key: str | None = None,
    strict: bool = False,
) -> ModelT:
    """用全局客户端加载配置。

    需先 `init_rainbow()`；未初始化则抛 `RainbowError`。

    `strict=False`（默认）时拉取失败降级为全默认值，适合热更新场景；
    `strict=True` 时拉取失败抛 `RainbowError`，供启动期终止启动。
    """
    return await get_rainbow_client().load(model_cls, group=group, key=key, strict=strict)
