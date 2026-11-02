"""通用 ASGI middleware：跨服务复用的横切关注点。

from infra.middleware import TraceHeaderMiddleware

app.add_middleware(TraceHeaderMiddleware)
"""

from infra.middleware.trace import TRACE_HEADER, TraceHeaderMiddleware, current_trace_id


__all__ = ['TRACE_HEADER', 'TraceHeaderMiddleware', 'current_trace_id']
