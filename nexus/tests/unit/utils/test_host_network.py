"""Host egress is opt-in and separate from sandbox proxy configuration."""

import pytest

from nexus.utils.host_network import host_session_kwargs, host_trust_env, websocket_proxy


@pytest.fixture(autouse=True)
def clear_proxy_environment(monkeypatch):
    for key in ('http_proxy', 'https_proxy', 'ws_proxy', 'wss_proxy', 'all_proxy', 'no_proxy'):
        monkeypatch.delenv(key, raising=False)
        monkeypatch.delenv(key.upper(), raising=False)
    monkeypatch.delenv('NEXUS_HOST_TRUST_ENV', raising=False)


def test_disabled_by_default(monkeypatch):
    monkeypatch.setenv('https_proxy', 'http://proxy.example:3128')
    assert host_trust_env() is False
    assert host_session_kwargs() == {}
    assert websocket_proxy('wss://sandbox.example/ws') is None


def test_enabled_proxy_and_protocol_priority(monkeypatch):
    monkeypatch.setenv('NEXUS_HOST_TRUST_ENV', '1')
    monkeypatch.setenv('https_proxy', 'http://proxy.example:3128')
    assert host_session_kwargs() == {'trust_env': True}
    assert websocket_proxy('wss://sandbox.example/ws') == 'http://proxy.example:3128'
    monkeypatch.setenv('wss_proxy', 'http://websocket.example:3128')
    assert websocket_proxy('https://sandbox.example') == 'http://websocket.example:3128'


@pytest.mark.parametrize('target', [
    'http://127.0.0.1:5000', 'ws://localhost:9222',
    'https://service.polaris:8080', 'wss://service.polaris:9281',
])
def test_no_proxy_bypasses_local_and_selected_hosts(monkeypatch, target):
    monkeypatch.setenv('NEXUS_HOST_TRUST_ENV', 'true')
    monkeypatch.setenv('http_proxy', 'http://proxy.example:3128')
    monkeypatch.setenv('https_proxy', 'http://proxy.example:3128')
    monkeypatch.setenv('no_proxy', 'localhost,127.0.0.1,.polaris')
    assert websocket_proxy(target) is None


def test_enabled_without_proxy_is_direct(monkeypatch):
    monkeypatch.setenv('NEXUS_HOST_TRUST_ENV', 'yes')
    assert websocket_proxy('https://sandbox.example') is None
