from __future__ import annotations

import asyncio

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
import socketio

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import TextContent

from nexus.extensions.agents.codebuddy.host_relay import host_relay_extension
from nexus.extensions.agents.codebuddy.session_runtime import CodeBuddySessionExtension
from nexus.extensions.agents.installed_common.host_rpc import (
    ACP_ELICITATION_REQUEST_EVENT,
    ACP_ELICITATION_RESOLVE_EVENT,
    ACP_PERMISSION_REQUEST_EVENT,
    ACP_PERMISSION_RESOLVE_EVENT,
    ACP_SESSION_UPDATE_EVENT,
    HOST_TOOL_REQUEST_EVENT,
    HOST_TOOL_RESPONSE_EVENT,
    RESOLVE_INTERRUPTION_METHOD,
)
from nexus.runtime.context import SIOBridge


pytestmark = pytest.mark.asyncio

_ECHO = {
    'name': 'echo',
    'description': 'Echo text through the Host.',
    'inputSchema': {'type': 'object', 'properties': {'text': {'type': 'string'}}, 'required': ['text']},
}


class FakePermissionConnection:
    def __init__(self) -> None:
        self.resolved: list[tuple[int, str]] = []
        self.ext_methods: list[str] = []
        self.withdrawn: set[int] = set()

    def resolve_deferred_permission(self, request_id: int, *, option_id: str) -> bool:
        self.resolved.append((request_id, option_id))
        return True

    def cancel_deferred_permission(self, request_id: int) -> bool:
        return True

    def has_deferred_permission(self, request_id: int) -> bool:
        return request_id not in self.withdrawn

    async def extension_request(self, method: str, params: dict[str, Any], *, timeout: float) -> Any:
        self.ext_methods.append(method)
        return {'resolved': True}


class FakeHostLink:
    """A real python-socketio client whose outbound frames are answered by a scripted Host."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.raw = socketio.AsyncClient()
        self.sio = SIOBridge(self.raw, '/worker')
        self.emitted: list[tuple[str, dict[str, Any]]] = []
        self._replies: set[asyncio.Task[None]] = set()
        monkeypatch.setattr(self.raw, 'emit', AsyncMock(side_effect=self._emit))

    async def _emit(self, event: str, data: dict[str, Any], namespace: str | None = None) -> None:
        self.emitted.append((event, data))
        if event == HOST_TOOL_REQUEST_EVENT:
            reply = {
                'request_id': data['request_id'],
                'channel_id': data['channel_id'],
                'success': True,
                'result': f'{data["channel_id"]}:{data["arguments"]["text"]}',
            }
            task = asyncio.create_task(self.deliver(HOST_TOOL_RESPONSE_EVENT, reply))
            self._replies.add(task)
            task.add_done_callback(self._replies.discard)

    async def deliver(self, event: str, data: dict[str, Any]) -> None:
        await self.raw.handlers['/worker'][event](data)


@pytest.fixture
def host_link(monkeypatch: pytest.MonkeyPatch) -> FakeHostLink:
    return FakeHostLink(monkeypatch)


@asynccontextmanager
async def _extension(
    host_link: FakeHostLink, config: dict[str, Any], *, session_id: str = 'session-a', connection: Any = None
) -> AsyncIterator[CodeBuddySessionExtension]:
    async with host_relay_extension(
        session_id=session_id,
        sio=host_link.sio,
        connection=connection if connection is not None else FakePermissionConnection(),
        config=config,
    ) as extension:
        yield extension


async def _call_echo(url: str, text: str) -> str:
    async with streamablehttp_client(url) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as client:
            await client.initialize()
            result = await client.call_tool('echo', {'text': text})
    assert not result.isError, result
    return ''.join(item.text for item in result.content if isinstance(item, TextContent))


async def test_permission_relay_routes_deferred_requests_by_session_channel(host_link: FakeHostLink) -> None:
    connection = FakePermissionConnection()
    async with _extension(host_link, {'relay_permissions': True}, connection=connection) as extension:
        assert extension.mcp_servers == ()
        assert extension.permission_handler is not None
        params = {'options': [{'optionId': 'opaque-allow', 'kind': 'allow_once'}]}
        assert await extension.permission_handler(params, 3) is None
        [(event, request)] = host_link.emitted
        assert event == ACP_PERMISSION_REQUEST_EVENT
        assert (request['channel_id'], request['acp_request_id'], request['params']) == ('session-a', 3, params), (
            'The session id is the Host routing channel, matching the LLM proxy channel'
        )
        await host_link.deliver(
            ACP_PERMISSION_RESOLVE_EVENT,
            {
                'channel_id': 'session-a',
                'acp_request_id': 3,
                'permission_token': request['permission_token'],
                'option_id': 'opaque-allow',
            },
        )
    assert connection.resolved == [(3, 'opaque-allow')]
    assert extension.client_capabilities is None, 'Without client_questions the handshake must stay as it was'
    assert extension.elicitation_handler is None


async def test_client_questions_declare_the_demo_capabilities_and_relay_elicitations(host_link: FakeHostLink) -> None:
    config = {'relay_permissions': True, 'client_questions': True}
    async with _extension(host_link, config) as extension:
        assert extension.client_capabilities == {
            'elicitation': {'form': {}},
            '_meta': {'codebuddy.ai': {'question': True}},
        }, 'Exactly what hy-live-demo declares, so CodeBuddy exposes AskUserQuestion'
        assert extension.permission_handler is not None
        assert extension.elicitation_handler is not None
        params = {'sessionId': 'remote', 'mode': 'form', 'message': '主色调？', 'requestedSchema': {}}
        pending = asyncio.create_task(extension.elicitation_handler(params))
        while not host_link.emitted:
            await asyncio.sleep(0)
        event, request = host_link.emitted[-1]
        assert (event, request['channel_id'], request['params']) == (ACP_ELICITATION_REQUEST_EVENT, 'session-a', params)
        await host_link.deliver(
            ACP_ELICITATION_RESOLVE_EVENT,
            {
                'channel_id': 'session-a',
                'elicitation_token': request['elicitation_token'],
                'result': {'action': 'decline'},
            },
        )
        assert await pending == {'action': 'decline'}
        late = asyncio.create_task(extension.elicitation_handler(params))
        while len(host_link.emitted) < 2:
            await asyncio.sleep(0)
    with pytest.raises(asyncio.CancelledError):
        await late


@pytest.mark.parametrize(('client_questions', 'sent'), [(False, []), (True, [RESOLVE_INTERRUPTION_METHOD])])
async def test_permission_ext_requests_are_honoured_only_with_client_questions(
    host_link: FakeHostLink, client_questions: bool, sent: list[str]
) -> None:
    connection = FakePermissionConnection()
    config = {'relay_permissions': True, 'client_questions': client_questions}
    async with _extension(host_link, config, connection=connection) as extension:
        assert extension.permission_handler is not None
        await extension.permission_handler({'options': [{'optionId': 'allow', 'kind': 'allow_once'}]}, 3)
        request = host_link.emitted[-1][1]
        await host_link.deliver(
            ACP_PERMISSION_RESOLVE_EVENT,
            {
                'channel_id': 'session-a',
                'permission_token': request['permission_token'],
                'option_id': 'allow',
                'ext_requests': [{'method': RESOLVE_INTERRUPTION_METHOD, 'params': {'decision': 'allow'}}],
            },
        )
        for _ in range(100):
            if connection.resolved:
                break
            await asyncio.sleep(0)
    assert connection.ext_methods == sent
    assert connection.resolved == [(3, 'allow')]


async def test_a_question_closed_after_its_turn_ended_sends_no_resolve_interruption(host_link: FakeHostLink) -> None:
    connection = FakePermissionConnection()
    config = {'relay_permissions': True, 'client_questions': True}
    async with _extension(host_link, config, connection=connection) as extension:
        assert extension.permission_handler is not None
        assert extension.on_turn_end is not None, 'The session must tell the relay when each prompt ends'
        await extension.permission_handler({'options': [{'optionId': 'reject', 'kind': 'reject_once'}]}, 3)
        request = host_link.emitted[-1][1]
        connection.withdrawn.add(3)
        extension.on_turn_end()
        await host_link.deliver(
            ACP_PERMISSION_RESOLVE_EVENT,
            {
                'channel_id': 'session-a',
                'permission_token': request['permission_token'],
                'option_id': 'reject',
                'ext_requests': [{'method': RESOLVE_INTERRUPTION_METHOD, 'params': {'decision': 'deny'}}],
            },
        )
        for _ in range(5):
            await asyncio.sleep(0)
    assert connection.ext_methods == []
    assert connection.resolved == []


async def test_without_relayed_permissions_the_extension_has_no_turn_end_hook(host_link: FakeHostLink) -> None:
    async with _extension(host_link, {'relay_updates': True}, connection=FakeUpdateConnection()) as extension:
        assert extension.on_turn_end is None


async def test_host_tools_are_served_over_http_mcp_and_answered_by_host(host_link: FakeHostLink) -> None:
    config = {'host_tools': {'server_name': 'host', 'tools': [_ECHO], 'request_timeout': 5}}
    async with _extension(host_link, config) as extension:
        assert extension.permission_handler is None, 'Tools alone must not take over the permission policy'
        (spec,) = extension.mcp_servers
        assert spec['type'] == 'http'
        assert spec['name'] == 'host'
        assert await _call_echo(spec['url'], 'hi') == 'session-a:hi'
    (event, request) = host_link.emitted[0]
    assert event == HOST_TOOL_REQUEST_EVENT
    assert request['tool_name'] == 'echo'
    assert request['channel_id'] == 'session-a'


async def test_exit_stops_server_and_releases_session_channels(host_link: FakeHostLink) -> None:
    config = {'relay_permissions': True, 'host_tools': {'server_name': 'host', 'tools': [_ECHO]}}
    connection = FakePermissionConnection()
    async with _extension(host_link, config, connection=connection) as extension:
        url = extension.mcp_servers[0]['url']
        assert extension.permission_handler is not None
        await extension.permission_handler({'options': []}, 1)
        issued = host_link.emitted[-1][1]
    with pytest.raises(httpx.ConnectError):
        async with httpx.AsyncClient(trust_env=False, timeout=2) as client:
            await client.post(url, json={})
    await host_link.deliver(
        ACP_PERMISSION_RESOLVE_EVENT,
        {
            'channel_id': 'session-a',
            'acp_request_id': 1,
            'permission_token': issued['permission_token'],
            'option_id': 'late',
        },
    )
    assert connection.resolved == [], 'Late Host decisions must not reach a closed session'
    async with _extension(host_link, config) as replacement:
        assert await _call_echo(replacement.mcp_servers[0]['url'], 'again') == 'session-a:again'


class FakeUpdateConnection(FakePermissionConnection):
    """An ACP connection that lets a test play ``session/update`` notifications to its listeners."""

    def __init__(self) -> None:
        super().__init__()
        self.listeners: list[Any] = []

    def add_update_listener(self, listener: Any) -> Any:
        self.listeners.append(listener)
        return lambda: self.listeners.remove(listener)

    def play(self, update: dict[str, Any]) -> None:
        for listener in list(self.listeners):
            listener({'sessionId': 'remote', 'update': update})


async def test_update_relay_alone_pushes_session_updates_until_exit(host_link: FakeHostLink) -> None:
    connection = FakeUpdateConnection()
    async with _extension(host_link, {'relay_updates': True}, connection=connection) as extension:
        assert extension.permission_handler is None, 'Pushing updates must not take over the permission policy'
        assert extension.mcp_servers == ()
        assert extension.update_relay is not None, 'The session needs the relay to mark its prompts'
        extension.update_relay.begin_turn('turn-1')
        connection.play({'sessionUpdate': 'agent_message_chunk', 'content': {'type': 'text', 'text': 'hi'}})
    connection.play({'sessionUpdate': 'agent_message_chunk', 'content': {'type': 'text', 'text': 'late'}})
    await asyncio.sleep(0)

    pushed = [payload for event, payload in host_link.emitted if event == ACP_SESSION_UPDATE_EVENT]
    assert [
        (payload['channel_id'], payload['turn_id'], payload['seq'], payload['update']['content']['text'])
        for payload in pushed
    ] == [('session-a', 'turn-1', 1, 'hi')], 'Exit must flush what was queued and stop listening'
    assert connection.listeners == []


@pytest.mark.parametrize(
    ('config', 'message'),
    [
        ({}, 'relay_permissions, relay_updates or host_tools'),
        ({'relay_permissions': False}, 'relay_permissions, relay_updates or host_tools'),
        ({'relay_permissions': 'yes'}, 'relay_permissions must be a bool'),
        ({'relay_updates': 'yes'}, 'relay_updates must be a bool'),
        ({'relay_permissions': True, 'enable_session_mcp': True}, 'Unknown host_relay_extension config'),
        ({'host_tools': {'server_name': 'host', 'tools': [_ECHO], 'port': 1}}, 'Unknown host_tools config'),
        ({'host_tools': []}, 'host_tools must be a mapping'),
        ({'host_tools': {'tools': [_ECHO]}}, 'missing required keys'),
        ({'relay_permissions': True, 'host_tools': {'server_name': 'host', 'tools': [_ECHO, _ECHO]}}, 'Duplicate'),
        ({'relay_permissions': True, 'client_questions': 'yes'}, 'client_questions must be a bool'),
        ({'relay_updates': True, 'client_questions': True}, 'client_questions needs relay_permissions'),
    ],
)
async def test_invalid_config_fails_without_leaking_channel_bindings(
    host_link: FakeHostLink, config: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        async with _extension(host_link, config):
            pytest.fail('Invalid configuration must not yield an extension')
    async with _extension(host_link, {'relay_permissions': True, 'host_tools': {'server_name': 'h', 'tools': [_ECHO]}}):
        pass


async def test_a_fork_enters_its_own_extension_on_the_shared_connection(host_link: FakeHostLink) -> None:
    connection = FakeUpdateConnection()
    config = {'relay_permissions': True, 'relay_updates': True}
    async with (
        _extension(host_link, config, session_id='source', connection=connection) as source,
        _extension(host_link, config, session_id='fork', connection=connection) as fork,
    ):
        assert source.permission_handler is not fork.permission_handler, 'Each session owns its permission relay'
        assert source.update_relay is not None
        assert fork.update_relay is not None
        source.update_relay.begin_turn('t-source', session_id='remote-source')
        fork.update_relay.begin_turn('t-fork', session_id='remote-fork')
        for session_id, text in (('remote-source', 'main'), ('remote-fork', 'answer')):
            for listener in list(connection.listeners):
                listener({
                    'sessionId': session_id,
                    'update': {'sessionUpdate': 'agent_message_chunk', 'content': {'type': 'text', 'text': text}},
                })
        assert await fork.permission_handler({'options': []}, 9) is None
    pushed = [payload for event, payload in host_link.emitted if event == ACP_SESSION_UPDATE_EVENT]
    assert sorted((payload['channel_id'], payload['update']['content']['text']) for payload in pushed) == [
        ('fork', 'answer'),
        ('source', 'main'),
    ], 'Each session pushes only its own updates on its own channel'
    [request] = [payload for event, payload in host_link.emitted if event == ACP_PERMISSION_REQUEST_EVENT]
    assert request['channel_id'] == 'fork'
    assert connection.listeners == []
