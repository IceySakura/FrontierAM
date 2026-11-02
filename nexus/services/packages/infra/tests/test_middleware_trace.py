"""TraceHeaderMiddleware 测试：所有响应路径都带 X-Trace-Id。"""

from __future__ import annotations

import pytest

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel

from infra.middleware import TRACE_HEADER, TraceHeaderMiddleware, current_trace_id


class _Body(BaseModel):
    value: int


def _app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(TraceHeaderMiddleware)  # ty: ignore[invalid-argument-type]

    @app.get('/ok')
    async def ok() -> dict[str, str]:
        return {'status': 'ok'}

    @app.post('/echo')
    async def echo(body: _Body) -> dict[str, int]:
        return {'value': body.value}

    @app.get('/boom')
    async def boom() -> dict[str, str]:
        raise HTTPException(status_code=418, detail='teapot')

    @app.get('/text')
    async def text() -> str:
        return 'plain'

    return app


@pytest.fixture
def client() -> TestClient:
    return TestClient(_app())


def test_no_header_without_active_span(client: TestClient) -> None:
    """未启用 OTel 时无有效 span，不注入空头。"""
    response = client.get('/ok')
    assert response.status_code == 200
    assert TRACE_HEADER not in response.headers


def test_current_trace_id_none_without_span() -> None:
    assert current_trace_id() is None


def test_header_injected_for_all_paths(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    """成功、校验失败、异常、非 JSON 响应都应带上 trace 头。"""
    import infra.middleware.trace as trace_module

    fake_id = 'a' * 32
    monkeypatch.setattr(trace_module, 'current_trace_id', lambda: fake_id)

    assert client.get('/ok').headers[TRACE_HEADER] == fake_id
    # 422 参数校验失败不经过业务 handler
    assert client.post('/echo', json={'value': 'bad'}).headers[TRACE_HEADER] == fake_id
    # HTTP 异常路径
    assert client.get('/boom').headers[TRACE_HEADER] == fake_id
    # 非 JSON 响应
    assert client.get('/text').headers[TRACE_HEADER] == fake_id


def test_existing_headers_preserved(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    """注入不应破坏原有响应头。"""
    import infra.middleware.trace as trace_module

    monkeypatch.setattr(trace_module, 'current_trace_id', lambda: 'b' * 32)
    response = client.get('/ok')

    assert response.headers['content-type'].startswith('application/json')
    assert response.json() == {'status': 'ok'}


def test_non_http_scope_passes_through() -> None:
    """非 HTTP scope（如 lifespan/websocket）直接透传，不做处理。"""
    seen: list[str] = []

    async def inner_app(scope: dict[str, object], receive: object, send: object) -> None:
        seen.append(str(scope['type']))

    middleware = TraceHeaderMiddleware(inner_app)  # type: ignore[arg-type]

    import asyncio

    async def noop() -> None:
        return None

    asyncio.run(middleware({'type': 'lifespan'}, noop, noop))  # type: ignore[arg-type]
    assert seen == ['lifespan']
