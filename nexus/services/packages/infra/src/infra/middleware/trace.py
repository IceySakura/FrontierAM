"""把当前请求的 trace_id 回写到响应头，便于按 ID 关联日志与链路。

放在 middleware 层而非路由层，是为了覆盖**所有**响应——包括
参数校验失败（422）、未捕获异常（500）以及非 JSON 响应，
这些路径都不会经过业务 handler。
"""

from __future__ import annotations

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from infra.logging import trace_context


TRACE_HEADER = 'X-Trace-Id'

_HEADER_KEY = TRACE_HEADER.lower().encode('latin-1')


def current_trace_id() -> str | None:
    """当前 OTel span 的 trace_id（32 位十六进制），无有效 span 时为 None。"""
    return trace_context().get('trace_id')


class TraceHeaderMiddleware:
    """向所有 HTTP 响应注入 `X-Trace-Id` 响应头。

    实现为纯 ASGI middleware 而非 `BaseHTTPMiddleware`：后者会把响应
    包装成新的 StreamingResponse，破坏流式响应并额外占用一层任务栈。
    这里只改写 `http.response.start` 的 headers，不触碰 body。
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope['type'] != 'http':
            await self.app(scope, receive, send)
            return

        async def send_with_trace(message: Message) -> None:
            if message['type'] == 'http.response.start':
                trace_id = current_trace_id()
                if trace_id:
                    # headers 是 list[tuple[bytes, bytes]]，就地追加即可
                    headers = message.setdefault('headers', [])
                    headers.append((_HEADER_KEY, trace_id.encode('latin-1')))
            await send(message)

        await self.app(scope, receive, send_with_trace)
