"""配置绑定标记：字段键名覆盖与 model 级 group/key 声明。

设计要点：
- 字段的键名默认取字段名；`Key` 用于覆盖**当前这一级**的名字，不支持跨级路径。
- 嵌套 pydantic model 按字段名（或 `Key` 覆盖后的名字）逐级下钻，
  因此标记里写的是相对当前层级的单级键名，model 挪动层级时无需改标记。
- `RainbowKey` 以 ClassVar 挂在 model 上，声明该 model 对应七彩石的哪份配置。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Key:
    """覆盖字段在配置源中的键名（仅当前这一级）。

        class AgsConfig(BaseModel):
            secret_id: Annotated[str | None, Key('secretId')] = None
            region: str = 'ap-guangzhou'   # 无标记，键名即字段名

    对嵌套 model 字段同样有效，用于源中子树名与字段名不一致的情况：

        class AppConfig(BaseModel):
            database: Annotated[DatabaseConfig, Key('db')] = DatabaseConfig()
    """

    name: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError('Key name must not be empty')


@dataclass(frozen=True)
class RainbowKey:
    """声明 model 对应的七彩石配置位置。

    以 ClassVar 挂在 model 类上（ClassVar 不会被 pydantic 收集为字段）：

        class ProviderConfig(BaseModel):
            rainbow_key: ClassVar[RainbowKey] = RainbowKey(key='provider')

    `group` 省略时使用 client 绑定的默认 group（来自 RainbowSettings）。
    `load()` 的显式传参优先级高于此声明。
    """

    key: str
    group: str | None = None

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError('RainbowKey key must not be empty')


# model 上承载 RainbowKey 的 ClassVar 属性名。
RAINBOW_KEY_ATTR = 'rainbow_key'
