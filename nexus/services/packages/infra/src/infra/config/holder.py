"""配置热更新：承载当前生效配置并在七彩石变更后原子替换。

用法：

    holder = ConfigHolder(config, field='provider', loader=load_provider)
    holder.add_reload_hook(service.invalidate)
    await holder.start_watch()

设计要点：
- 读方通过 `current` 惰性取值，因此热更新后无需重建依赖方；
- 依赖方若按配置缓存了对象（如用旧凭据构造的客户端），
  需注册 reload hook 在重载后失效缓存，否则配置变了但请求仍走旧值；
- 重载只替换 `loader` 负责的那部分，进程启动期参数（端口、日志级别）
  已作用于运行时，不参与热更新。
"""

from __future__ import annotations

import logging
import os

from collections.abc import Callable
from typing import Generic, Protocol, TypeVar

from pydantic import BaseModel

from infra.config.env import get_rainbow_client, is_rainbow_ready
from infra.config.rainbow import RainbowError


logger = logging.getLogger(__name__)

ConfigT = TypeVar('ConfigT', bound=BaseModel)
SectionT = TypeVar('SectionT', bound=BaseModel)

_TRUTHY = ('1', 'true', 'yes', 'on')


class SectionLoader(Protocol[SectionT]):
    """子配置加载器：`strict=True` 时拉取失败应抛异常而非降级。"""

    async def __call__(self, *, strict: bool = False) -> SectionT: ...


class ConfigHolder(Generic[ConfigT, SectionT]):
    """持有当前生效配置，支持七彩石变更后原子替换其中一个子配置。

    `field` 指定被热更新的字段名，`loader` 负责重新拉取该字段的值。
    其余字段（如进程启动期参数）保持不变。
    """

    def __init__(
        self,
        config: ConfigT,
        *,
        field: str,
        loader: SectionLoader[SectionT],
    ) -> None:
        if field not in type(config).model_fields:
            raise ValueError(f'{type(config).__name__} has no field {field!r}')

        self._config = config
        self._field = field
        self._loader = loader
        self._on_reload: list[Callable[[], None]] = []

    @property
    def current(self) -> ConfigT:
        """当前生效配置。读方应每次调用而非缓存返回值。"""
        return self._config

    def add_reload_hook(self, hook: Callable[[], None]) -> None:
        """注册重载后的回调，用于失效依赖方按旧配置构造的缓存。"""
        self._on_reload.append(hook)

    async def load_initial(self) -> None:
        """启动期首次加载被监听的子配置。

        与 `reload()` 的区别：这里**不吞异常**——启动阶段拉取失败应让
        启动失败，避免服务带空配置开始接流量（K8s 会重启 Pod 并重试）。
        """
        section = await self._loader(strict=True)
        self._config = self._config.model_copy(update={self._field: section})
        logger.info('config field %r loaded', self._field)

    async def reload(self) -> None:
        """重新拉取并替换被监听的子配置，随后触发所有 hook。

        与 `load_initial()` 的区别：此时服务已在接流量，拉取失败由
        `loader` 内部降级处理，不中断运行。单个 hook 失败不影响其余
        hook 与本次重载结果。
        """
        section = await self._loader()
        self._config = self._config.model_copy(update={self._field: section})
        logger.info('config field %r reloaded', self._field)

        for hook in self._on_reload:
            try:
                hook()
            except Exception:
                logger.exception('config reload hook failed')

    async def start_watch(self, *, env_var: str = 'RAINBOW_WATCH') -> bool:
        """按需注册七彩石变更监听。返回是否成功启用热更新。

        需先 `init_rainbow()`。未开启开关、客户端未初始化或注册失败时返回 False，
        配置保持为启动时加载的静态值，不影响服务可用性。
        """
        if os.environ.get(env_var, '').lower() not in _TRUTHY:
            logger.info('rainbow watch disabled (%s not set)', env_var)
            return False

        if not is_rainbow_ready():
            logger.info('rainbow client not initialized, watch skipped')
            return False

        try:
            await get_rainbow_client().watch(self.reload)
        except RainbowError:
            logger.warning('failed to start rainbow watch, config stays static', exc_info=True)
            return False
        return True
