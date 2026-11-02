"""Unit coverage for runtime debug page helpers."""

import asyncio
import sys

from importlib import resources
from types import ModuleType, SimpleNamespace

import pytest


def _import_runtime_app_module():
    pytest.importorskip('fastapi')
    try:
        __import__('playwright.async_api')
    except ModuleNotFoundError:
        playwright_module = ModuleType('playwright')
        async_api_module = SimpleNamespace(
            Browser=object,
            BrowserContext=object,
            Page=object,
            Playwright=object,
            async_playwright=lambda: None,
        )
        sys.modules.setdefault('playwright', playwright_module)
        sys.modules['playwright.async_api'] = async_api_module
    from nexus.runtime.server import app as runtime_app

    return runtime_app


def test_build_debug_page_includes_novnc_entrypoint():
    from nexus.runtime.server.debug import build_debug_page

    html = build_debug_page(novnc_path='/debug/gui/vnc/vnc.html', websocket_path='/debug/gui/vnc/websockify')

    assert 'Sandbox Desktop' in html
    assert 'noVNC' in html
    assert '/debug/gui/vnc/vnc.html' in html
    assert '/debug/gui/vnc/websockify' in html
    assert '<iframe' in html


def test_runtime_debug_home_includes_debugger_workspace_tools():
    html = resources.files('nexus.runtime.server').joinpath('debug.html').read_text(encoding='utf-8')

    assert '<title>Nexus Debugger</title>' in html
    assert 'Web IDE' in html
    assert '沙盒桌面' in html
    assert '交互式代码控制台' in html
    assert '站点预览' in html
    assert 'MCP 调试台' in html
    assert '文件管理' in html
    assert '执行器调试台' in html
    assert 'api/v1/call_executor' in html
    assert 'debug/ports/${port}' in html
    assert '/runs\\/[^/]+\\/runtimes\\/[^/]+\\/proxy' in html


def test_build_debug_page_uses_absolute_websocket_path_query():
    from nexus.runtime.server.debug import build_debug_page

    html = build_debug_page(
        novnc_path='/debug/gui/vnc/vnc.html?autoconnect=1&resize=remote&path=/debug/gui/vnc/websockify',
        websocket_path='/debug/gui/vnc/websockify',
    )

    assert 'path=/debug/gui/vnc/websockify' in html
    assert 'path=debug/gui/vnc/websockify' not in html
    assert 'location.pathname' in html
    assert 'encodeURIComponent(websocketPath)' in html


def test_build_novnc_path_uses_absolute_websocket_path():
    from nexus.runtime.server.debug import build_novnc_path

    novnc_path = build_novnc_path('/debug/gui/vnc/websockify')

    assert novnc_path == '/debug/gui/vnc/vnc.html?autoconnect=1&resize=remote&path=/debug/gui/vnc/websockify'


def test_debug_port_proxy_allows_valid_tcp_ports():
    from nexus.runtime.server.debug import validate_debug_port

    validate_debug_port(1)
    validate_debug_port(8080)
    validate_debug_port(65535)


def test_debug_port_proxy_rejects_invalid_tcp_ports():
    from nexus.runtime.server.debug import validate_debug_port

    for port in (0, -1, 65536):
        with pytest.raises(ValueError, match='invalid debug proxy port'):
            validate_debug_port(port)


def test_runtime_app_registers_run_scoped_debug_port_proxy_routes():
    app = _import_runtime_app_module().app

    paths = {route.path for route in app.routes}

    assert '/debug/ports/{port}' in paths
    assert '/debug/ports/{port}/{asset_path:path}' in paths
    assert '/debug/runs/{run_id}/ports/{port}' in paths
    assert '/debug/runs/{run_id}/ports/{port}/{asset_path:path}' in paths


def test_runtime_app_registers_aiomonitor_debug_proxy_routes():
    app = _import_runtime_app_module().app

    paths = {route.path for route in app.routes}

    assert '/debug/aiomonitor/{loop_name}' in paths
    assert '/debug/aiomonitor/{loop_name}/{asset_path:path}' in paths


def test_debug_port_upstream_url_strips_public_prefix():
    _debug_port_upstream_url = _import_runtime_app_module()._debug_port_upstream_url

    assert _debug_port_upstream_url(port=8080, asset_path='', query='folder=%2Fworkspace') == (
        'http://127.0.0.1:8080/?folder=%2Fworkspace'
    )
    assert _debug_port_upstream_url(port=8080, asset_path='_static/app.js') == 'http://127.0.0.1:8080/_static/app.js'
    assert _debug_port_upstream_url(port=8080, asset_path='socket', websocket=True) == 'ws://127.0.0.1:8080/socket'


def test_debug_port_websocket_headers_use_runtime_local_origin():
    _proxy_websocket_headers = _import_runtime_app_module()._proxy_websocket_headers

    class FakeWebSocket:
        headers = {
            'Host': '192.168.2.141:18765',
            'Origin': 'http://192.168.2.141:18765',
            'User-Agent': 'test',
        }

    headers = _proxy_websocket_headers(FakeWebSocket(), host='127.0.0.1:8080')

    assert headers['Host'] == '127.0.0.1:8080'
    assert headers['Origin'] == 'http://127.0.0.1:8080'
    assert headers['User-Agent'] == 'test'


@pytest.mark.asyncio
async def test_debug_port_proxy_keeps_encoded_response_bytes(monkeypatch):
    runtime_app = _import_runtime_app_module()

    payload = b'gzip payload bytes'
    captured = {}

    class FakeURL:
        query = 'folder=%2Fworkspace'

    class FakeRequest:
        method = 'GET'
        headers = {'Accept-Encoding': 'gzip'}
        url = FakeURL()

        async def body(self) -> bytes:
            return b''

    class FakeUpstream:
        status = 200
        headers = {
            'Content-Encoding': 'gzip',
            'Content-Length': '999',
            'Transfer-Encoding': 'chunked',
        }

        async def read(self) -> bytes:
            return payload

    class FakeRequestContext:
        async def __aenter__(self) -> FakeUpstream:
            return FakeUpstream()

        async def __aexit__(self, exc_type, exc, traceback) -> None:
            return None

    class FakeSession:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def __aenter__(self) -> 'FakeSession':
            return self

        async def __aexit__(self, exc_type, exc, traceback) -> None:
            return None

        def request(self, *args, **kwargs) -> FakeRequestContext:
            captured['request_headers'] = kwargs.get('headers')
            return FakeRequestContext()

    monkeypatch.setattr(runtime_app.aiohttp, 'ClientSession', FakeSession)

    response = await runtime_app.debug_port_http_proxy(port=8080, asset_path='index.html', request=FakeRequest())

    assert captured['auto_decompress'] is False
    assert captured['request_headers']['Accept-Encoding'] == 'gzip'
    assert captured['request_headers']['Host'] == '127.0.0.1:8080'
    assert response.body == payload
    assert response.headers['content-encoding'] == 'gzip'
    assert response.headers['content-length'] == str(len(payload))
    assert 'transfer-encoding' not in response.headers


@pytest.mark.asyncio
async def test_aiomonitor_debug_proxy_rewrites_html_root_paths(monkeypatch):
    runtime_app = _import_runtime_app_module()

    captured = {}

    class FakeURL:
        query = ''

    class FakeRequest:
        method = 'GET'
        headers = {'X-Nexus-Debug-Base-Path': '/public/debug/aiomonitor/main'}
        url = FakeURL()

        async def body(self) -> bytes:
            return b''

    class FakeUpstream:
        status = 200
        headers = {'Content-Type': 'text/html; charset=utf-8'}

        async def read(self) -> bytes:
            return b'<script src="/static/app.js"></script><form hx-post="/api/console"></form>'

    class FakeRequestContext:
        async def __aenter__(self) -> FakeUpstream:
            return FakeUpstream()

        async def __aexit__(self, exc_type, exc, traceback) -> None:
            return None

    class FakeSession:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def __aenter__(self) -> 'FakeSession':
            return self

        async def __aexit__(self, exc_type, exc, traceback) -> None:
            return None

        def request(self, *args, **kwargs) -> FakeRequestContext:
            captured['request_headers'] = kwargs.get('headers')
            return FakeRequestContext()

    monkeypatch.setattr(runtime_app.aiohttp, 'ClientSession', FakeSession)

    response = await runtime_app.debug_aiomonitor_http_proxy(loop_name='main', asset_path='', request=FakeRequest())

    assert captured['auto_decompress'] is False
    assert response.status_code == 200
    html = response.body.decode()
    assert 'src="/public/debug/aiomonitor/main/static/app.js"' in html
    assert 'hx-post="/public/debug/aiomonitor/main/api/console"' in html


@pytest.mark.asyncio
async def test_debug_port_websocket_bridge_treats_closing_transport_as_normal():
    runtime_app = _import_runtime_app_module()

    cancelled = False

    async def closing_transport() -> None:
        raise runtime_app.aiohttp.ClientConnectionResetError('Cannot write to closing transport')

    async def pending_side() -> None:
        nonlocal cancelled
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled = True
            raise

    await runtime_app._run_websocket_bridge(closing_transport(), pending_side())

    assert cancelled
