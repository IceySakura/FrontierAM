from contextlib import asynccontextmanager

import pytest

from nexus.core.tool.mcp.models import MCPClientConfig, MCPTransportKind
from nexus.core.tool.mcp.transport.factory import create_mcp_transport


@pytest.mark.asyncio
async def test_sse_transport_passes_headers_and_timeout(monkeypatch):
    captured = {}

    @asynccontextmanager
    async def fake_sse_client(url, **kwargs):
        captured['url'] = url
        captured.update(kwargs)
        yield ('read', 'write')

    monkeypatch.setattr('mcp.client.sse.sse_client', fake_sse_client)
    config = MCPClientConfig(
        transport=MCPTransportKind.SSE,
        url='http://localhost/events',
        headers={'Authorization': 'Bearer t'},
        timeout_s=12,
    )

    async with create_mcp_transport(config) as streams:
        assert streams == ('read', 'write')

    assert captured == {
        'url': 'http://localhost/events',
        'headers': {'Authorization': 'Bearer t'},
        'timeout': 12,
        'sse_read_timeout': 12,
    }
