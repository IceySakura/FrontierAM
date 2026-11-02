"""七彩石配置绑定与热更新。

model 上用 ClassVar 声明对应的配置 key，字段键名默认取字段名、
需要时用 `Key` 覆盖当前这一级；嵌套 model 按字段名逐级下钻。

    class AgsConfig(BaseModel):
        secret_id: Annotated[str | None, Key('secretId')] = None
        region: str = 'ap-guangzhou'

    class ProviderConfig(BaseModel):
        rainbow_key: ClassVar[RainbowKey] = RainbowKey(key='provider')
        ags: AgsConfig = AgsConfig()

启动时初始化一次全局客户端（读 `RAINBOW_*` 环境变量），之后任意处加载：

    await init_rainbow()
    provider = await load_from_rainbow(ProviderConfig)

热更新（配置变更后原子替换，并失效依赖方缓存）：

    holder = ConfigHolder(config, field='provider', loader=partial(load_from_rainbow, ProviderConfig))
    holder.add_reload_hook(service.invalidate)
    await holder.start_watch()
"""

from infra.config.env import (
    get_rainbow_client,
    init_rainbow,
    is_rainbow_ready,
    load_from_rainbow,
    rainbow_settings_from_env,
    reset_rainbow,
)
from infra.config.holder import ConfigHolder
from infra.config.keys import Key, RainbowKey
from infra.config.rainbow import (
    DEFAULT_CONNECT_STR,
    RainbowClient,
    RainbowError,
    RainbowSettings,
    create_client,
)
from infra.config.resolver import bind


__all__ = [
    'DEFAULT_CONNECT_STR',
    'ConfigHolder',
    'Key',
    'RainbowClient',
    'RainbowError',
    'RainbowKey',
    'RainbowSettings',
    'bind',
    'create_client',
    'get_rainbow_client',
    'init_rainbow',
    'is_rainbow_ready',
    'load_from_rainbow',
    'rainbow_settings_from_env',
    'reset_rainbow',
]
