"""trace_id 注入测试：响应体与响应头两条路径。

响应头由 infra 的 TraceHeaderMiddleware 统一注入（覆盖校验失败、异常等
不经过业务 handler 的路径），响应体由 BaseResponse 自动填充。
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from fastapi.testclient import TestClient

from infra.middleware import TRACE_HEADER
from label import LabelConfig, create_app
from label.schemas.sandbox import Provider
from label.services.errors import SandboxErrorCode
from label.services.sandbox import SandboxProviderError


_FAKE_TRACE_ID = 'c' * 32


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app(LabelConfig()))


def test_response_body_carries_trace_id_field(client: TestClient) -> None:
    """成功响应体含 trace_id 字段；未启用 OTel 时为 None。"""
    with patch('label.services.sandbox.SandboxService._create_provider') as create:
        create.return_value = _FakeProvider()
        body = client.post(
            '/api/v1/label/sandbox/release',
            json={'platform': 'windows', 'provider': 'ags', 'instance_id': 'i-1'},
        ).json()

    assert 'trace_id' in body
    assert body['trace_id'] is None


def test_body_trace_id_filled_from_active_span(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    """有活跃 span 时，响应体 trace_id 自动填充。

    `default_factory` 在类定义时已捕获函数引用，替换 schema 模块属性无效，
    因此 patch 上游的 `trace_context`。
    """
    import infra.middleware.trace as trace_module

    monkeypatch.setattr(trace_module, 'trace_context', lambda: {'trace_id': _FAKE_TRACE_ID})

    with patch('label.services.sandbox.SandboxService._create_provider') as create:
        create.return_value = _FakeProvider()
        body = client.post(
            '/api/v1/label/sandbox/release',
            json={'platform': 'windows', 'provider': 'ags', 'instance_id': 'i-1'},
        ).json()

    assert body['trace_id'] == _FAKE_TRACE_ID


def test_header_injected_on_success(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    import infra.middleware.trace as trace_module

    monkeypatch.setattr(trace_module, 'current_trace_id', lambda: _FAKE_TRACE_ID)
    assert client.get('/health/live').headers[TRACE_HEADER] == _FAKE_TRACE_ID


def test_header_injected_on_validation_error(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    """422 不经过业务 handler，仍需可追溯。"""
    import infra.middleware.trace as trace_module

    monkeypatch.setattr(trace_module, 'current_trace_id', lambda: _FAKE_TRACE_ID)
    response = client.post('/api/v1/label/sandbox/acquire', json={'platform': 'android', 'provider': 'ags'})

    assert response.status_code == 422
    assert response.headers[TRACE_HEADER] == _FAKE_TRACE_ID


def test_provider_error_body_and_header_carry_trace_id(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    """出错时最需要按 ID 查日志：body 与 header 都要有。"""
    import infra.middleware.trace as trace_module
    import label.exception_handlers as handlers_module

    monkeypatch.setattr(trace_module, 'current_trace_id', lambda: _FAKE_TRACE_ID)
    monkeypatch.setattr(handlers_module, 'current_trace_id', lambda: _FAKE_TRACE_ID)

    def boom(_self: object, _provider: Provider) -> None:
        raise SandboxProviderError(SandboxErrorCode.PROVIDER_NOT_CONFIGURED, 'not configured')

    with patch('label.services.sandbox.SandboxService._create_provider', new=boom):
        response = client.post(
            '/api/v1/label/sandbox/release',
            json={'platform': 'windows', 'provider': 'ags', 'instance_id': 'i-1'},
        )

    assert response.status_code == SandboxErrorCode.PROVIDER_NOT_CONFIGURED.http_status
    assert response.json()['trace_id'] == _FAKE_TRACE_ID
    assert response.headers[TRACE_HEADER] == _FAKE_TRACE_ID


class _FakeProvider:
    """最小可用的 provider 替身，只支持 release 路径。"""

    async def _call_api(self, *_args: Any) -> dict[str, Any]:
        return {}
