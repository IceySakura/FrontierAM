from __future__ import annotations

import asyncio
import contextvars

from types import SimpleNamespace
from typing import Any

import aiohttp
import pytest
import uvicorn

from nexus.extensions.agents.openclaw.web_search_bridge import SerperBridge, SerperBridgeConfig
from nexus.runtime.providers.gongfeng import GongfengRuntimeProvider
from nexus.runtime.server.services.anthropic_proxy.server import AnthropicProxy
from nexus.runtime.server.services.gemini_proxy.server import GeminiProxy
from nexus.runtime.server.services.openai_proxy import OpenAIProxy
from nexus.runtime.server.services.responses_proxy.server import ResponsesProxy
from nexus.runtime.server.worker.http import WorkerHTTPServer


@pytest.mark.asyncio
async def test_long_lived_uvicorn_tasks_do_not_inherit_request_context(monkeypatch: pytest.MonkeyPatch) -> None:
    request_context = contextvars.ContextVar('request_context', default='detached')
    observed: list[str] = []

    class FakeServer:
        def __init__(self, _config: Any) -> None:
            self.started = False
            self.should_exit = False
            self.servers = [SimpleNamespace(sockets=[SimpleNamespace(getsockname=lambda: ('127.0.0.1', 43210))])]

        async def serve(self) -> None:
            observed.append(request_context.get())
            self.started = True

    class FakeResponse:
        status = 200

        async def __aenter__(self) -> FakeResponse:
            await asyncio.sleep(0)
            return self

        async def __aexit__(self, *_args: Any) -> None:
            return None

    class FakeClientSession:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> FakeClientSession:
            return self

        async def __aexit__(self, *_args: Any) -> None:
            return None

        def get(self, _url: str) -> FakeResponse:
            return FakeResponse()

    class SIO:
        def on(self, _event: str):
            return lambda handler: handler

    original_sleep = asyncio.sleep

    async def fast_sleep(_delay: float) -> None:
        await original_sleep(0)

    monkeypatch.setattr(uvicorn, 'Server', FakeServer)
    monkeypatch.setattr(aiohttp, 'ClientSession', FakeClientSession)
    monkeypatch.setattr(asyncio, 'sleep', fast_sleep)
    bridge = SerperBridge(config=SerperBridgeConfig(api_key='unit-test-key'), listen_port=43210)

    token = request_context.set('runtime.worker.invoke')
    try:
        proxies = [
            AnthropicProxy(SIO()),  # type: ignore[arg-type]
            OpenAIProxy(SIO()),  # type: ignore[arg-type]
            ResponsesProxy(SIO()),  # type: ignore[arg-type]
            GeminiProxy(SIO()),  # type: ignore[arg-type]
        ]
        for proxy in proxies:
            await proxy.start()
        worker_http = WorkerHTTPServer()
        await worker_http.start()
        await bridge.start()
    finally:
        await bridge.stop()
        request_context.reset(token)

    assert observed == ['detached'] * 6


@pytest.mark.asyncio
async def test_gongfeng_watchdog_does_not_inherit_request_context() -> None:
    request_context = contextvars.ContextVar('request_context', default='detached')
    observed: list[str] = []
    provider = GongfengRuntimeProvider(
        api_base_url='http://gongfeng.example.com',
        token='unit-test-token',
        image='unit-test/image:latest',
        print_port_mappings=False,
    )

    async def fake_watchdog_loop(_name: str, _runtime: Any, _runtime_id: str) -> None:
        observed.append(request_context.get())

    provider._watchdog_loop = fake_watchdog_loop  # type: ignore[method-assign]
    runtime: Any = SimpleNamespace()
    token = request_context.set('agent.main_agent')
    try:
        provider._start_watchdog('runtime', runtime, 'runtime-id')
    finally:
        request_context.reset(token)

    await provider._watchdog_tasks['runtime']

    assert observed == ['detached']
