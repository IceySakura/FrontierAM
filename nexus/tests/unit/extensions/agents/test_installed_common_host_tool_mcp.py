from __future__ import annotations

import json

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import TextContent

from nexus.extensions.agents.installed_common.host_rpc import HostToolError
from nexus.extensions.agents.installed_common.host_tool_mcp import HostToolMCPServer


pytestmark = pytest.mark.asyncio

_LOOKUP = {
    'name': 'kb_lookup',
    'title': 'Lookup',
    'description': 'Look up one entry.',
    'inputSchema': {
        'type': 'object',
        'properties': {'query': {'type': 'string'}},
        'required': ['query'],
    },
    'annotations': {'readOnlyHint': True},
}
_NOTE = {'name': 'kb_note', 'description': 'Write a note.', 'inputSchema': {'type': 'object', 'properties': {}}}


class FakeHost:
    """Answer Host tool calls from a script and record what reached the Host."""

    def __init__(self, result: Any = None, error: HostToolError | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.result = result
        self.error = error

    async def call(self, *, tool_name: str, arguments: dict[str, Any], timeout: float) -> Any:
        self.calls.append({'tool_name': tool_name, 'arguments': arguments, 'timeout': timeout})
        if self.error is not None:
            raise self.error
        return self.result


@asynccontextmanager
async def _served(host: FakeHost, **kwargs: Any) -> AsyncIterator[tuple[HostToolMCPServer, dict[str, Any]]]:
    server = HostToolMCPServer(
        server_name='kb', tools=kwargs.pop('tools', [_LOOKUP, _NOTE]), call_host=host.call, **kwargs
    )
    spec = await server.start()
    try:
        yield server, spec
    finally:
        await server.close()


@asynccontextmanager
async def _client(url: str) -> AsyncIterator[ClientSession]:
    async with streamablehttp_client(url) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            yield session


def _text(result: Any) -> str:
    return ''.join(item.text for item in result.content if isinstance(item, TextContent))


async def test_start_returns_acp_http_spec_and_lists_host_tools_verbatim() -> None:
    async with _served(FakeHost()) as (_, spec):
        assert spec['type'] == 'http'
        assert spec['name'] == 'kb'
        assert spec['headers'] == []
        assert spec['url'].startswith('http://127.0.0.1:')
        async with _client(spec['url']) as client:
            tools = (await client.list_tools()).tools
    assert [tool.name for tool in tools] == ['kb_lookup', 'kb_note']
    lookup = tools[0]
    assert lookup.title == 'Lookup'
    assert lookup.description == 'Look up one entry.'
    assert lookup.inputSchema == _LOOKUP['inputSchema'], 'The Host-owned schema must reach the model unchanged'
    assert lookup.annotations is not None
    assert lookup.annotations.readOnlyHint is True


@pytest.mark.parametrize(
    ('host_result', 'expected_text'),
    [
        ({'entry': '中文', 'count': 2}, json.dumps({'entry': '中文', 'count': 2}, ensure_ascii=False, indent=2)),
        ('plain', 'plain'),
    ],
)
async def test_tool_call_forwards_name_and_arguments_and_renders_host_result(
    host_result: Any, expected_text: str
) -> None:
    host = FakeHost(result=host_result)
    async with _served(host, request_timeout=30.0) as (_, spec):
        async with _client(spec['url']) as client:
            result = await client.call_tool('kb_lookup', {'query': 'x'})
    assert not result.isError
    assert _text(result) == expected_text
    assert host.calls == [{'tool_name': 'kb_lookup', 'arguments': {'query': 'x'}, 'timeout': 30.0}]


async def test_host_error_becomes_coded_mcp_tool_error() -> None:
    host = FakeHost(error=HostToolError('no such entry', code='NOT_FOUND'))
    async with _served(host) as (_, spec):
        async with _client(spec['url']) as client:
            result = await client.call_tool('kb_lookup', {'query': 'x'})
    assert result.isError
    assert _text(result) == 'NOT_FOUND: no such entry', 'The model must see the Host error code and message'


async def test_schema_violation_is_rejected_before_reaching_host() -> None:
    host = FakeHost(result='unused')
    async with _served(host) as (_, spec):
        async with _client(spec['url']) as client:
            result = await client.call_tool('kb_lookup', {'query': 3})
    assert result.isError
    assert host.calls == [], 'Invalid arguments must not reach Host services'


async def test_per_tool_timeout_overrides_default_budget() -> None:
    host = FakeHost(result='ok')
    async with _served(host, request_timeout=60.0, tool_timeouts={'kb_lookup': 10.0}) as (_, spec):
        async with _client(spec['url']) as client:
            await client.call_tool('kb_lookup', {'query': 'x'})
            await client.call_tool('kb_note', {})
    assert [call['timeout'] for call in host.calls] == [10.0, 60.0]


@pytest.mark.parametrize(
    ('kwargs', 'message'),
    [
        ({'server_name': ''}, 'server_name'),
        ({'tools': []}, 'at least one tool'),
        ({'tools': [_NOTE, _NOTE]}, 'Duplicate'),
        ({'tools': [{'description': 'no name'}]}, 'Invalid host tool'),
        ({'tools': [{**_NOTE, 'outputSchema': {'type': 'object'}}]}, 'outputSchema'),
        ({'tool_timeouts': {'missing': 1.0}}, 'unknown tools'),
        ({'tool_timeouts': {'kb_note': 0}}, 'positive'),
        ({'request_timeout': float('inf')}, 'positive'),
    ],
)
async def test_invalid_server_configuration_is_rejected(kwargs: dict[str, Any], message: str) -> None:
    arguments: dict[str, Any] = {'server_name': 'kb', 'tools': [_NOTE], 'call_host': FakeHost().call, **kwargs}
    with pytest.raises(ValueError, match=message):
        HostToolMCPServer(**arguments)


async def test_close_stops_serving() -> None:
    async with _served(FakeHost()) as (server, spec):
        pass
    with pytest.raises(httpx.ConnectError):
        async with httpx.AsyncClient(trust_env=False, timeout=2) as client:
            await client.post(spec['url'], json={})
    await server.close()
