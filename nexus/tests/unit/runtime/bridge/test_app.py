from __future__ import annotations

import pytest

from fastapi.testclient import TestClient

from nexus.runtime.bridge.adapters import ags_osworld
from nexus.runtime.bridge.app import _ServerSio, create_bridge_fastapi_app


class FakeAsyncServer:
    def __init__(self) -> None:
        self.registered: dict[str, object] = {}
        self.emitted: list[tuple[str, object]] = []

    def on(self, event: str, handler: object) -> None:
        self.registered[event] = handler

    async def emit(self, event: str, data: object = None) -> None:
        self.emitted.append((event, data))


@pytest.mark.parametrize(
    'api_protocol', ['anthropic', 'responses', 'osworld-anthropic', 'osworld-responses', 'responses-chat']
)
def test_bridge_health_contains_connected_hosts_and_pending(api_protocol: str) -> None:
    factory = (
        create_bridge_fastapi_app
        if api_protocol in {'anthropic', 'responses'}
        else ags_osworld.create_bridge_fastapi_app
    )
    app, bridge = factory(
        api_protocol=api_protocol,
        request_timeout=12,
        launch_id='launch-123',
    )
    client = TestClient(app)

    response = client.get('/health')

    assert response.status_code == 200
    assert response.json() == {
        'status': 'ok',
        'bridge': 'nexus-llm-bridge',
        'launch_id': 'launch-123',
        'connected_hosts': 0,
        'pending_requests': 0,
    }
    assert bridge._pending == {}


@pytest.mark.parametrize('api_protocol', ['responses', 'responses-chat'])
def test_bridge_exposes_responses_routes(api_protocol: str) -> None:
    factory = create_bridge_fastapi_app if api_protocol == 'responses' else ags_osworld.create_bridge_fastapi_app
    app, _ = factory(
        api_protocol=api_protocol,
        request_timeout=12,
    )
    client = TestClient(app)
    paths = {route.path for route in app.routes}

    response = client.get('/v1/models')

    assert response.status_code == 200
    assert response.json() == {'data': [], 'object': 'list'}
    assert '/v1/responses' in paths
    assert '/v1/responses/compact' in paths
    assert '/v1/messages' not in paths


def test_bridge_uses_anthropic_proxy_by_default() -> None:
    app, _ = create_bridge_fastapi_app(request_timeout=12)
    paths = {route.path for route in app.routes}

    assert '/v1/messages' in paths
    assert '/v1/responses' not in paths


@pytest.mark.parametrize('api_protocol', ['osworld-anthropic', 'osworld-responses'])
def test_osworld_bridge_exposes_requested_routes(api_protocol: str) -> None:
    app, bridge = ags_osworld.create_bridge_fastapi_app(
        api_protocol=api_protocol,
    )
    paths = {route.path for route in app.routes}

    assert app.state.bridge is bridge
    assert '/v1/messages' in paths
    assert ('/v1/responses' in paths) == (api_protocol == 'osworld-responses')


def test_bridge_rejects_unknown_api_protocol() -> None:
    with pytest.raises(ValueError, match='Unsupported bridge API protocol'):
        create_bridge_fastapi_app(api_protocol='unknown')


@pytest.mark.asyncio
async def test_server_sio_shim_strips_sid_and_forwards_emit() -> None:
    raw = FakeAsyncServer()
    shim = _ServerSio(raw)
    received: list[tuple] = []

    @shim.on('llm_response')
    async def handler(data: dict) -> None:
        received.append((data,))

    # Server delivers (sid, data); the shim must drop sid before calling the handler.
    await raw.registered['llm_response']('host-sid', {'request_id': 'r1'})
    assert received == [({'request_id': 'r1'},)]

    await shim.emit('llm_request', {'request_id': 'r1'})
    assert raw.emitted == [('llm_request', {'request_id': 'r1'})]
