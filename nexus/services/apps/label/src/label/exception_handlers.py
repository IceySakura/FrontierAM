"""全局异常处理器：把领域异常映射为统一的错误响应体。"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from infra.middleware import current_trace_id
from label.services.sandbox import SandboxProviderError


async def sandbox_provider_error_handler(_request: Request, exc: Exception) -> JSONResponse:
    """沙盒 provider 异常按其错误码映射为对应 HTTP 状态。

    响应体同样带 trace_id：出错时最需要按 ID 去查日志，
    此路径不经过 BaseResponse，需在此显式补上。
    """
    assert isinstance(exc, SandboxProviderError)
    return JSONResponse(
        status_code=exc.code.http_status,
        content={'code': exc.code, 'message': exc.message, 'trace_id': current_trace_id()},
    )


def register_exception_handlers(app: FastAPI) -> None:
    """把所有领域异常处理器注册到应用。"""
    app.add_exception_handler(SandboxProviderError, sandbox_provider_error_handler)
