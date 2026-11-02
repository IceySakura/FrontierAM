from __future__ import annotations

import asyncio

import pytest

from nexus.runtime.bridge.host_client import BridgeHostClient, connect_bridge_llm_handler


class FakeLLMBridgeClient:
    def __init__(self) -> None:
        self.connected = False
        self.registered: list[tuple] = []

    async def ws_connect(self) -> None:
        self.connected = True

    def on(self, event, listener):
        self.registered.append((event, listener))
        return lambda: None

    def off(self, event, listener=None):
        pass

    async def emit(self, event, data):
        pass


class FakeOpenAIClient:
    class Config:
        request_args = {'model': 'deepseek-v4-pro'}
        client_args: dict = {}

    config = Config()


@pytest.mark.asyncio
async def test_connect_bridge_llm_handler_registers_llm_request() -> None:
    client = FakeLLMBridgeClient()

    handler = await connect_bridge_llm_handler(client, FakeOpenAIClient())

    assert client.connected is True
    assert client.registered[0][0] == 'llm_request'
    assert handler is not None


def test_host_client_event_registry_off_removes_listener() -> None:
    client = BridgeHostClient(base_url='http://127.0.0.1:1')

    async def handler(data: dict) -> None:
        pass

    client.on('llm_request', handler)
    assert 'llm_request' in client._event_listeners

    client.off('llm_request', handler)
    assert 'llm_request' not in client._event_listeners


@pytest.mark.asyncio
async def test_emit_requires_connection() -> None:
    client = BridgeHostClient(base_url='http://127.0.0.1:1')

    with pytest.raises(RuntimeError, match='Socket.IO is not connected'):
        await client.emit('llm_response', {'request_id': 'r'})


@pytest.mark.asyncio
@pytest.mark.parametrize('prefix', ['', '/port-proxy/test-token'])
async def test_host_client_sends_keepalive_until_disconnect(monkeypatch: pytest.MonkeyPatch, prefix: str) -> None:
    class FakeAsyncClient:
        instances: list[FakeAsyncClient] = []

        def __init__(self, **kwargs) -> None:
            self.connected = False
            self.handlers = {}
            self.emits = []
            FakeAsyncClient.instances.append(self)

        def event(self, handler):
            self.handlers[handler.__name__] = handler
            return handler

        def on(self, event, handler):
            self.handlers[event] = handler

        async def connect(self, *args, **kwargs) -> None:
            assert kwargs['socketio_path'] == prefix + '/socket.io'
            self.connected = True
            await self.handlers['connect']()

        async def disconnect(self) -> None:
            self.connected = False
            await self.handlers['disconnect']()

        async def emit(self, event, data=None, callback=None) -> None:
            self.emits.append((event, data))

    monkeypatch.setattr('nexus.runtime.bridge.host_client.socketio.AsyncClient', FakeAsyncClient)

    client = BridgeHostClient(base_url='http://127.0.0.1:1' + prefix, keepalive_interval=0.01)
    await client.ws_connect()
    await asyncio.sleep(0.035)

    sio = FakeAsyncClient.instances[0]
    keepalive_count = sum(1 for event, _data in sio.emits if event == 'bridge_keepalive')
    assert keepalive_count >= 1

    await client.ws_disconnect()
    await asyncio.sleep(0.03)

    assert sum(1 for event, _data in sio.emits if event == 'bridge_keepalive') == keepalive_count
