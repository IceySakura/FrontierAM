from fastapi.testclient import TestClient

from infra.config import bind
from label import LabelConfig, create_app
from label.config import ProviderConfig


def test_health_endpoints() -> None:
    """不进入 lifespan（不触发七彩石初始化），只验证路由与响应体。"""
    client = TestClient(create_app(LabelConfig(service_name='label')))

    live_response = client.get('/health/live')
    ready_response = client.get('/health/ready')

    assert live_response.status_code == 200
    assert live_response.json() == {'status': 'ok', 'service': 'label'}
    assert ready_response.status_code == 200
    assert ready_response.json() == {'status': 'ready', 'service': 'label'}


def test_config_snapshot_masks_secrets() -> None:
    provider = bind(
        ProviderConfig,
        {'ags': {'secretId': 'AKID1234567890', 'toolId': 'sdt-1'}, 'smartrun': {'token': 'tok_abcdefgh'}},
    )
    client = TestClient(create_app(LabelConfig(provider=provider)))

    body = client.get('/health/config').json()

    # 敏感值只回显前 8 位与长度
    assert body['ags']['secret_id'] == 'AKID1234...(14)'
    assert body['smartrun']['token'] == 'tok_abcd...(12)'
    # 非敏感值原样返回
    assert body['ags']['tool_id'] == 'sdt-1'
    assert body['ags']['region'] == 'ap-guangzhou'


def test_config_snapshot_reports_watch_state() -> None:
    """未进入 lifespan 时 watching 为 False（app.state 未写入）。"""
    client = TestClient(create_app(LabelConfig()))
    assert client.get('/health/config').json()['watching'] is False


def test_config_snapshot_handles_empty_secrets() -> None:
    client = TestClient(create_app(LabelConfig()))
    body = client.get('/health/config').json()

    assert body['ags']['secret_id'] is None
    assert body['smartrun']['duration'] == 1800
