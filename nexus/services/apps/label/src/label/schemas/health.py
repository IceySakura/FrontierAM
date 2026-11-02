from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """存活/就绪探针响应。"""

    status: str
    service: str


class ConfigSnapshotResponse(BaseModel):
    """当前生效配置摘要，用于确认七彩石热更新是否生效。

    敏感值只回显前缀与长度，不暴露完整凭据。
    """

    watching: bool
    ags: dict[str, str | int | None]
    devcloud: dict[str, str | int | None]
    smartrun: dict[str, str | int | None]
