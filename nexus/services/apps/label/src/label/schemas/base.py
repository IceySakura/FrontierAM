from __future__ import annotations

from pydantic import BaseModel, Field

from infra.middleware import current_trace_id


class BaseResponse(BaseModel):
    """统一响应体：业务码 + 描述 + 链路 ID。

    `trace_id` 默认从当前 OTel span 自动填充，因此业务代码构造响应时
    无需显式传入；无有效 span（如未启用 OTel）时为 None。

    注意：响应头 `X-Trace-Id` 由 `TraceHeaderMiddleware` 统一注入，
    覆盖校验失败与未捕获异常等不经过本类的路径。此处的字段是为了
    让前端能直接从 body 取到，两者是互补而非重复。
    """

    code: int = 0
    message: str = 'success'
    trace_id: str | None = Field(default_factory=current_trace_id)
