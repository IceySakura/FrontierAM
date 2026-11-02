from __future__ import annotations

import subprocess
import sys

from contextlib import asynccontextmanager

import pytest

from nexus.core.tool.mcp.models import MCPClientConfig, MCPTransportKind
from nexus.core.tool.mcp.transport.factory import create_mcp_transport
from nexus.utils.otel import subprocess as otel_subprocess


@pytest.fixture
def capture_stdio(monkeypatch):
    from mcp.client import stdio as stdio_module

    captured = []

    @asynccontextmanager
    async def stdio_client(params):
        captured.append(params)
        yield object(), object()

    monkeypatch.setattr(stdio_module, 'stdio_client', stdio_client)
    return captured


@pytest.mark.asyncio
async def test_enabled_managed_stdio_uses_official_client(monkeypatch) -> None:
    from mcp.client import stdio as stdio_module

    selected = []

    @asynccontextmanager
    async def official_client(_params):
        selected.append('official')
        yield object(), object()

    monkeypatch.setenv('NEXUS_OTEL_ENABLED', 'true')
    monkeypatch.setattr(stdio_module, 'stdio_client', official_client)
    config = MCPClientConfig(transport=MCPTransportKind.STDIO, command='nexus-mcp')

    async with create_mcp_transport(config, server_name='managed', nexus_managed=True):
        pass

    assert selected == ['official']


def test_factory_import_does_not_require_opentelemetry_package() -> None:
    code = """
import builtins

original_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name == 'opentelemetry' or name.startswith('opentelemetry.'):
        raise ModuleNotFoundError(name)
    return original_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
import nexus.core.tool.mcp.transport.factory
"""

    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stderr


@pytest.mark.asyncio
async def test_enabled_managed_stdio_uses_clean_otel_env_without_mutating_config(monkeypatch, capture_stdio) -> None:
    monkeypatch.setenv('NEXUS_OTEL_ENABLED', 'true')
    monkeypatch.setenv('OTEL_EXPORTER_OTLP_ENDPOINT', 'http://parent-collector:4318')
    monkeypatch.setenv('OTEL_EXPORTER_OTLP_HEADERS', 'parent-auth')
    monkeypatch.setenv('OTEL_RESOURCE_ATTRIBUTES', 'deployment.environment=test')
    monkeypatch.setenv('OTEL_SERVICE_NAME', 'parent-service')
    monkeypatch.setenv('OTEL_PROPAGATORS', 'forged-parent')
    monkeypatch.setenv('OTEL_EXPORTER_OTLP_METRICS_HEADERS', 'authorization=parent-metrics-secret')
    monkeypatch.setenv('NEXUS_OTEL_PROCESS_ROLE', 'parent-role')
    monkeypatch.delenv('OTEL_TRACES_SAMPLER', raising=False)
    config = MCPClientConfig(
        transport=MCPTransportKind.STDIO,
        command='third-party-server',
        env={
            'CUSTOM': 'keep',
            'OTEL_EXPORTER_OTLP_HEADERS': 'forged-config-auth',
            'OTEL_SERVICE_NAME': 'forged-config-service',
            'OTEL_TRACES_SAMPLER': 'forged-config-sampler',
            'NEXUS_OTEL_PROCESS_ROLE': 'forged-config-role',
            'OTEL_EXPORTER_OTLP_LOGS_HEADERS': 'authorization=config-logs-secret',
        },
    )
    original = config.model_dump()

    async with create_mcp_transport(config, server_name='Catalog / US', nexus_managed=True):
        pass

    [params] = capture_stdio
    assert params.command == 'third-party-server'
    assert params.env['CUSTOM'] == 'keep'
    assert params.env['OTEL_EXPORTER_OTLP_HEADERS'] == 'parent-auth'
    assert 'OTEL_TRACES_SAMPLER' not in params.env
    assert params.env['NEXUS_OTEL_PROCESS_ROLE'] == 'mcp'
    assert params.env['OTEL_SERVICE_NAME'] == 'nexus-mcp-catalog-us'
    assert params.env['OTEL_PROPAGATORS'] == 'tracecontext'
    assert params.env['OTEL_EXPORTER_OTLP_METRICS_HEADERS'] == 'authorization=parent-metrics-secret'
    assert 'OTEL_EXPORTER_OTLP_LOGS_HEADERS' not in params.env
    assert params.env['OTEL_RESOURCE_ATTRIBUTES'] == ('deployment.environment=test,nexus.mcp.server=Catalog%20%2F%20US')
    assert config.model_dump() == original


@pytest.mark.asyncio
async def test_managed_stdio_preserves_exporter_env_only_while_spawning(monkeypatch) -> None:
    from mcp.client import stdio as stdio_module

    monkeypatch.setenv('NEXUS_OTEL_ENABLED', 'true')
    monkeypatch.setenv('OTEL_EXPORTER_OTLP_HEADERS', 'authorization=collector-secret')
    monkeypatch.setattr(otel_subprocess, 'inject_carrier', lambda: {})
    observed: dict[str, dict[str, str]] = {}

    @asynccontextmanager
    async def stdio_client(params):
        observed['spawn'] = otel_subprocess.inject_current_context_env(params.env)
        yield object(), object()

    monkeypatch.setattr(stdio_module, 'stdio_client', stdio_client)
    config = MCPClientConfig(transport=MCPTransportKind.STDIO, command='nexus-mcp')

    async with create_mcp_transport(config, server_name='managed', nexus_managed=True):
        observed['body'] = otel_subprocess.inject_current_context_env(dict(observed['spawn']))

    assert observed['spawn']['OTEL_EXPORTER_OTLP_HEADERS'] == 'authorization=collector-secret'
    assert 'OTEL_EXPORTER_OTLP_HEADERS' not in observed['body']


@pytest.mark.asyncio
async def test_enabled_third_party_stdio_does_not_receive_managed_otel_env(monkeypatch, capture_stdio) -> None:
    monkeypatch.setenv('NEXUS_OTEL_ENABLED', 'true')
    monkeypatch.setenv('OTEL_EXPORTER_OTLP_ENDPOINT', 'http://parent-collector:4318')
    monkeypatch.setenv('OTEL_EXPORTER_OTLP_HEADERS', 'parent-auth')
    config = MCPClientConfig(
        transport=MCPTransportKind.STDIO,
        command='third-party-server',
        env={'CUSTOM': 'keep', 'OTEL_SERVICE_NAME': 'forged'},
    )

    async with create_mcp_transport(config, server_name='third-party', nexus_managed=False):
        pass

    [params] = capture_stdio
    assert params.env['CUSTOM'] == 'keep'
    assert not any(key.startswith('OTEL_') or key.startswith('NEXUS_OTEL_') for key in params.env)


@pytest.mark.asyncio
async def test_requested_otel_without_local_dependencies_still_strips_third_party_env(
    monkeypatch,
    capture_stdio,
) -> None:
    from nexus.core.tool.mcp.transport import factory

    monkeypatch.setenv('NEXUS_OTEL_ENABLED', 'true')
    monkeypatch.setenv('OTEL_EXPORTER_OTLP_HEADERS', 'parent-auth')
    monkeypatch.setattr(factory, 'is_requested', lambda: True)
    monkeypatch.setattr(factory, 'is_enabled', lambda: False)
    config = MCPClientConfig(
        transport=MCPTransportKind.STDIO,
        command='third-party-server',
        env={'CUSTOM': 'keep'},
    )

    async with create_mcp_transport(config, nexus_managed=False):
        pass

    [params] = capture_stdio
    assert params.env['CUSTOM'] == 'keep'
    assert 'OTEL_EXPORTER_OTLP_HEADERS' not in params.env


@pytest.mark.asyncio
@pytest.mark.parametrize('config_env', [{}, {'CUSTOM': 'configured', 'OTEL_SERVICE_NAME': 'untrusted'}])
async def test_disabled_stdio_retains_legacy_env_shape(monkeypatch, capture_stdio, config_env) -> None:
    monkeypatch.setenv('NEXUS_OTEL_ENABLED', 'false')
    monkeypatch.setenv('PARENT_ONLY', 'parent')
    config = MCPClientConfig(
        transport=MCPTransportKind.STDIO,
        command='third-party-server',
        env=config_env,
    )

    async with create_mcp_transport(config, server_name='ignored', nexus_managed=True):
        pass

    [params] = capture_stdio
    if config_env:
        assert params.env['PARENT_ONLY'] == 'parent'
        assert params.env['CUSTOM'] == 'configured'
        assert params.env['OTEL_SERVICE_NAME'] == 'untrusted'
    else:
        assert params.env is None


@pytest.mark.asyncio
async def test_non_stdio_transport_does_not_consume_mcp_child_options(monkeypatch) -> None:
    from nexus.core.tool.mcp.transport import factory as factory_module

    captured: dict = {}

    @asynccontextmanager
    async def http_client(url, **kwargs):
        captured.update(url=url, kwargs=kwargs)
        yield object(), object(), lambda: None

    monkeypatch.setattr(factory_module, 'aiohttp_streamable_http_client', http_client)
    config = MCPClientConfig(
        transport=MCPTransportKind.AIOHTTP_STREAMABLE_HTTP,
        url='http://server.invalid/mcp',
    )

    async with create_mcp_transport(config, server_name='catalog', nexus_managed=True):
        pass

    assert captured['url'] == 'http://server.invalid/mcp'
    assert captured['kwargs'] == {'session': None, 'headers': {}, 'timeout_s': None}
