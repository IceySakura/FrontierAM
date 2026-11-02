from __future__ import annotations

from types import SimpleNamespace

import pytest

from nexus.core.tool.mcp.manager import MCPToolManager
from nexus.core.tool.mcp.models import MCPServerSpec, MCPToolInfo


class _Connection:
    session = object()


@pytest.mark.asyncio
async def test_manager_threads_identity_to_runtime_proxy_and_proxy_connection(monkeypatch) -> None:
    from nexus.core.tool.mcp import manager as manager_module

    captured: dict = {}

    class Runtime:
        base_url = 'http://runtime.invalid'

        async def mcp_start_proxy(self, client, *, server_name='mcp', nexus_managed=False):
            captured.update(
                runtime_client=client,
                runtime_server_name=server_name,
                runtime_nexus_managed=nexus_managed,
            )
            return SimpleNamespace(
                proxy_id='proxy-1',
                endpoint_path='/mcp/proxy/proxy-1',
                tools=[MCPToolInfo(name='search')],
            )

    class Context:
        tool_state = {}

        async def get_runtime(self, name):
            assert name == 'main'
            return Runtime()

    async def connect(config, **kwargs):
        captured.update(connect_config=config, connect_kwargs=kwargs)
        return _Connection()

    monkeypatch.setattr(manager_module, 'connect_mcp_session', connect)
    manager = MCPToolManager(Context())
    spec = MCPServerSpec.model_validate({
        'name': 'catalog',
        'runtime_name': 'main',
        'nexus_managed': True,
        'client': {'transport': 'stdio', 'command': 'managed-server'},
    })

    attached = await manager._attach_server('server-key', spec)

    assert attached.runtime_proxy_id == 'proxy-1'
    assert captured['runtime_server_name'] == 'catalog'
    assert captured['runtime_nexus_managed'] is True
    assert captured['connect_kwargs'] == {'server_name': 'catalog', 'nexus_managed': True}


@pytest.mark.asyncio
async def test_manager_threads_identity_to_direct_connection(monkeypatch) -> None:
    from nexus.core.tool.mcp import manager as manager_module

    captured: dict = {}

    class Context:
        tool_state = {}

    async def connect(config, **kwargs):
        captured.update(config=config, kwargs=kwargs)
        return _Connection()

    async def list_tools(session, *, timeout_s=None):
        return [MCPToolInfo(name='search')]

    monkeypatch.setattr(manager_module, 'connect_mcp_session', connect)
    monkeypatch.setattr(manager_module, 'list_mcp_tools', list_tools)
    manager = MCPToolManager(Context())
    spec = MCPServerSpec.model_validate({
        'name': 'catalog',
        'nexus_managed': True,
        'client': {'transport': 'stdio', 'command': 'managed-server'},
    })

    attached = await manager._attach_server('server-key', spec)

    assert attached.tools[0].name == 'search'
    assert captured['kwargs'] == {'server_name': 'catalog', 'nexus_managed': True}
