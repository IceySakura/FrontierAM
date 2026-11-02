from __future__ import annotations

import asyncio
import json

from typing import Any
from unittest.mock import AsyncMock

import pytest
import socketio

from nexus.extensions.agents.installed_common.acp import AcpConnection
from nexus.extensions.agents.installed_common.host_rpc import (
    ACP_ELICITATION_REQUEST_EVENT,
    ACP_ELICITATION_RESOLVE_EVENT,
    ACP_PERMISSION_REQUEST_EVENT,
    ACP_PERMISSION_RESOLVE_EVENT,
    ACP_SESSION_UPDATE_EVENT,
    HOST_TOOL_REQUEST_EVENT,
    HOST_TOOL_RESPONSE_EVENT,
    _PUSHED_VISIBLE_CHARS,
    AcpElicitationRelay,
    AcpPermissionRelay,
    AcpUpdateRelay,
    HostToolClient,
    HostToolError,
    trim_acp_update,
)
from nexus.runtime.context import SIOBridge


pytestmark = pytest.mark.asyncio


class FakePermissionConnection:
    """Record which deferred ACP permission the Host resolved or cancelled."""

    def __init__(self, *, ext_error: Exception | None = None, ext_result: Any = None) -> None:
        self.resolved: list[tuple[int, str]] = []
        self.cancelled: list[int] = []
        self.calls: list[tuple[str, Any]] = []
        self.ext_error = ext_error
        self.ext_result = {'resolved': True, 'queued': False} if ext_result is None else ext_result
        self.withdrawn: set[int] = set()

    def resolve_deferred_permission(self, request_id: int, *, option_id: str) -> bool:
        self.resolved.append((request_id, option_id))
        self.calls.append(('resolve', (request_id, option_id)))
        return True

    def cancel_deferred_permission(self, request_id: int) -> bool:
        self.cancelled.append(request_id)
        self.calls.append(('cancel', request_id))
        return True

    def has_deferred_permission(self, request_id: int) -> bool:
        return request_id not in self.withdrawn

    async def extension_request(self, method: str, params: dict[str, Any], *, timeout: float) -> Any:
        self.calls.append(('ext', (method, params)))
        await asyncio.sleep(0)
        if self.ext_error is not None:
            raise self.ext_error
        return self.ext_result


@pytest.fixture
def raw_transport(monkeypatch: pytest.MonkeyPatch) -> tuple[socketio.AsyncClient, list[tuple[str, dict[str, Any]]]]:
    raw_sio = socketio.AsyncClient()
    emitted: list[tuple[str, dict[str, Any]]] = []

    async def emit(event: str, data: dict[str, Any], namespace: str | None = None) -> None:
        emitted.append((event, data))

    monkeypatch.setattr(raw_sio, 'emit', AsyncMock(side_effect=emit))
    return raw_sio, emitted


async def _deliver(raw_sio: socketio.AsyncClient, event: str, data: dict[str, Any]) -> None:
    await raw_sio.handlers['/worker'][event](data)


def _permission_request(emitted: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    event, payload = emitted[-1]
    assert event == ACP_PERMISSION_REQUEST_EVENT
    return payload


def _resolution(request: dict[str, Any], option_id: str | None) -> dict[str, Any]:
    """Answer exactly like a Host that echoes the request identity it received."""
    return {
        'channel_id': request['channel_id'],
        'acp_request_id': request['acp_request_id'],
        'permission_token': request['permission_token'],
        'option_id': option_id,
    }


async def test_permission_relay_publishes_channel_request_and_applies_host_option(raw_transport: Any) -> None:
    raw_sio, emitted = raw_transport
    connection = FakePermissionConnection()
    relay = AcpPermissionRelay(sio=SIOBridge(raw_sio, '/worker'), channel_id='chan-a', connection=connection)
    params = {'sessionId': 'remote', 'options': [{'optionId': 'opaque-allow', 'kind': 'allow_once'}]}

    assert await relay.request(params, 7) is None, 'The Host decides later; the ACP callback must keep waiting'
    request = _permission_request(emitted)
    assert request.keys() == {'channel_id', 'acp_request_id', 'permission_token', 'params'}
    assert (request['channel_id'], request['acp_request_id'], request['params']) == ('chan-a', 7, params)
    assert isinstance(request['permission_token'], str)
    assert request['permission_token'], 'Every request needs a non-empty token the Host must echo'

    await _deliver(raw_sio, ACP_PERMISSION_RESOLVE_EVENT, _resolution(request, 'opaque-allow'))
    assert connection.resolved == [(7, 'opaque-allow')], 'The exact Host-selected option must reach ACP'
    relay.close()


@pytest.mark.parametrize('option_id', [None, ''])
async def test_permission_resolution_without_option_cancels_request(raw_transport: Any, option_id: str | None) -> None:
    raw_sio, emitted = raw_transport
    connection = FakePermissionConnection()
    relay = AcpPermissionRelay(sio=SIOBridge(raw_sio, '/worker'), channel_id='chan-a', connection=connection)
    await relay.request({'options': []}, 9)
    await _deliver(raw_sio, ACP_PERMISSION_RESOLVE_EVENT, _resolution(_permission_request(emitted), option_id))
    assert connection.cancelled == [9], 'A missing Host option must cancel, never guess an approval'
    assert connection.resolved == []
    relay.close()


async def test_each_permission_request_gets_a_distinct_token(raw_transport: Any) -> None:
    raw_sio, emitted = raw_transport
    relay = AcpPermissionRelay(
        sio=SIOBridge(raw_sio, '/worker'), channel_id='chan-a', connection=FakePermissionConnection()
    )
    await relay.request({'options': []}, 1)
    await relay.request({'options': []}, 2)
    assert emitted[0][1]['permission_token'] != emitted[1][1]['permission_token']
    relay.close()


@pytest.mark.parametrize('token', [None, '', 'forged-token'])
async def test_resolution_without_the_issued_token_is_ignored(raw_transport: Any, token: str | None) -> None:
    raw_sio, emitted = raw_transport
    connection = FakePermissionConnection()
    relay = AcpPermissionRelay(sio=SIOBridge(raw_sio, '/worker'), channel_id='chan-a', connection=connection)
    await relay.request({'options': []}, 1)
    resolution = {**_resolution(_permission_request(emitted), 'allow'), 'permission_token': token}
    await _deliver(raw_sio, ACP_PERMISSION_RESOLVE_EVENT, resolution)
    assert connection.resolved == [], 'A request id alone must never authorize an ACP operation'
    assert connection.cancelled == []
    relay.close()


async def test_a_token_authorizes_only_one_resolution(raw_transport: Any) -> None:
    raw_sio, emitted = raw_transport
    connection = FakePermissionConnection()
    relay = AcpPermissionRelay(sio=SIOBridge(raw_sio, '/worker'), channel_id='chan-a', connection=connection)
    await relay.request({'options': []}, 1)
    resolution = _resolution(_permission_request(emitted), 'allow')
    await _deliver(raw_sio, ACP_PERMISSION_RESOLVE_EVENT, resolution)
    await _deliver(raw_sio, ACP_PERMISSION_RESOLVE_EVENT, resolution)
    assert connection.resolved == [(1, 'allow')], 'A replayed Host decision must not be applied twice'
    relay.close()


async def test_stale_approval_cannot_authorize_a_session_recreated_with_the_same_id(raw_transport: Any) -> None:
    """ACP request ids restart per CLI process and approval option ids are fixed, so neither identifies a request."""
    raw_sio, emitted = raw_transport
    old_connection, new_connection = FakePermissionConnection(), FakePermissionConnection()
    old_relay = AcpPermissionRelay(sio=SIOBridge(raw_sio, '/worker'), channel_id='session-1', connection=old_connection)
    await old_relay.request({'options': []}, 1)
    stale = _resolution(_permission_request(emitted), 'allow')
    old_relay.close()

    new_relay = AcpPermissionRelay(sio=SIOBridge(raw_sio, '/worker'), channel_id='session-1', connection=new_connection)
    await new_relay.request({'options': []}, 1)
    fresh = _resolution(_permission_request(emitted), 'allow')
    assert (fresh['channel_id'], fresh['acp_request_id']) == (stale['channel_id'], stale['acp_request_id'])

    await _deliver(raw_sio, ACP_PERMISSION_RESOLVE_EVENT, stale)
    assert new_connection.resolved == [], 'An approval issued to a closed session must not reach its successor'
    await _deliver(raw_sio, ACP_PERMISSION_RESOLVE_EVENT, fresh)
    assert new_connection.resolved == [(1, 'allow')]
    assert old_connection.resolved == []
    new_relay.close()


async def test_relays_on_wrappers_sharing_one_socket_route_by_channel(raw_transport: Any) -> None:
    raw_sio, emitted = raw_transport
    first, second = FakePermissionConnection(), FakePermissionConnection()
    relay_a = AcpPermissionRelay(sio=SIOBridge(raw_sio, '/worker'), channel_id='chan-a', connection=first)
    relay_b = AcpPermissionRelay(sio=SIOBridge(raw_sio, '/worker'), channel_id='chan-b', connection=second)
    assert list(raw_sio.handlers['/worker']) == [ACP_PERMISSION_RESOLVE_EVENT], (
        'python-socketio keeps one handler per event, so relays must share it'
    )
    await relay_a.request({'options': []}, 1)
    request_a = _permission_request(emitted)
    await relay_b.request({'options': []}, 2)
    request_b = _permission_request(emitted)
    await _deliver(raw_sio, ACP_PERMISSION_RESOLVE_EVENT, _resolution(request_a, 'allow'))
    await _deliver(raw_sio, ACP_PERMISSION_RESOLVE_EVENT, _resolution(request_b, 'deny'))
    assert first.resolved == [(1, 'allow')]
    assert second.resolved == [(2, 'deny')]
    relay_a.close()
    relay_b.close()


async def test_permission_relay_rejects_a_second_owner_for_one_channel(raw_transport: Any) -> None:
    raw_sio, _ = raw_transport
    relay = AcpPermissionRelay(
        sio=SIOBridge(raw_sio, '/worker'), channel_id='chan-a', connection=FakePermissionConnection()
    )
    with pytest.raises(ValueError, match='chan-a'):
        AcpPermissionRelay(
            sio=SIOBridge(raw_sio, '/worker'), channel_id='chan-a', connection=FakePermissionConnection()
        )
    relay.close()


async def test_closed_permission_relay_ignores_late_resolution_and_frees_channel(raw_transport: Any) -> None:
    raw_sio, emitted = raw_transport
    connection = FakePermissionConnection()
    relay = AcpPermissionRelay(sio=SIOBridge(raw_sio, '/worker'), channel_id='chan-a', connection=connection)
    await relay.request({'options': []}, 1)
    late = _resolution(_permission_request(emitted), 'allow')
    relay.close()
    await _deliver(raw_sio, ACP_PERMISSION_RESOLVE_EVENT, late)
    assert connection.resolved == [], 'A closed session must not receive stale approvals'
    replacement = AcpPermissionRelay(
        sio=SIOBridge(raw_sio, '/worker'), channel_id='chan-a', connection=FakePermissionConnection()
    )
    replacement.close()


_RESOLVE = {
    'method': '_codebuddy.ai/resolveInterruption',
    'params': {'sessionId': 'remote', 'toolCallId': 'call-1', 'decision': 'allow', 'answers': {'q_0': '蓝色'}},
}


async def _settle(connection: FakePermissionConnection, count: int) -> None:
    for _ in range(100):
        if len(connection.calls) >= count:
            return
        await asyncio.sleep(0)
    raise AssertionError(f'expected {count} calls, got {connection.calls!r}')


_ASK_OPTIONS = [{'optionId': 'allow', 'kind': 'allow_once'}, {'optionId': 'reject', 'kind': 'reject_once'}]


def _question_relay(raw_sio: Any, connection: FakePermissionConnection) -> AcpPermissionRelay:
    return AcpPermissionRelay(
        sio=SIOBridge(raw_sio, '/worker'), channel_id='chan-a', connection=connection, client_questions=True
    )


async def test_ext_requests_are_sent_before_the_permission_is_answered(raw_transport: Any) -> None:
    raw_sio, emitted = raw_transport
    connection = FakePermissionConnection()
    relay = _question_relay(raw_sio, connection)
    await relay.request({'options': _ASK_OPTIONS}, 4)
    resolution = {**_resolution(_permission_request(emitted), 'allow'), 'ext_requests': [_RESOLVE]}
    await _deliver(raw_sio, ACP_PERMISSION_RESOLVE_EVENT, resolution)
    await _settle(connection, 2)
    assert connection.calls == [('ext', (_RESOLVE['method'], _RESOLVE['params'])), ('resolve', (4, 'allow'))], (
        'CodeBuddy must learn the answer before its AskUserQuestion permission is released'
    )
    relay.close()


@pytest.mark.parametrize(
    'connection_kwargs',
    [
        {'ext_error': RuntimeError('peer refused 蓝色')},
        {'ext_error': asyncio.TimeoutError()},
        {'ext_result': {'resolved': False, 'echo': '蓝色'}},
        {'ext_result': '蓝色'},
    ],
)
async def test_an_unresolved_ext_request_is_logged_and_the_host_option_still_applies(
    raw_transport: Any,
    caplog: pytest.LogCaptureFixture,
    connection_kwargs: dict[str, Any],
) -> None:
    raw_sio, emitted = raw_transport
    connection = FakePermissionConnection(**connection_kwargs)
    relay = _question_relay(raw_sio, connection)
    await relay.request({'options': _ASK_OPTIONS}, 4)
    resolution = {**_resolution(_permission_request(emitted), 'allow'), 'ext_requests': [_RESOLVE]}
    with caplog.at_level('INFO', logger='nexus.extensions.agents.installed_common.host_rpc'):
        await _deliver(raw_sio, ACP_PERMISSION_RESOLVE_EVENT, resolution)
        await _settle(connection, 2)
    assert connection.calls == [('ext', (_RESOLVE['method'], _RESOLVE['params'])), ('resolve', (4, 'allow'))], (
        'CodeBuddy returns {} on an answer it accepted, so a failed extension request must not override it'
    )
    assert 'ACP extension request failed' in caplog.text
    assert '蓝色' not in caplog.text, 'The user answer must never reach the logs'
    relay.close()


@pytest.mark.parametrize(
    'ext_requests',
    [
        {'method': '_x', 'params': {}},
        [{'method': '', 'params': {}}],
        [{'method': '_x', 'params': 'not-a-mapping'}],
        [_RESOLVE, {'method': _RESOLVE['method'], 'params': ['蓝色']}],
        ['_x'],
        [{'method': 'session/prompt', 'params': {}}],
        [{'method': '_codebuddy.ai/other', 'params': {}}, _RESOLVE],
    ],
)
async def test_malformed_or_disallowed_ext_requests_cancel_the_permission(
    raw_transport: Any, caplog: pytest.LogCaptureFixture, ext_requests: Any
) -> None:
    raw_sio, emitted = raw_transport
    connection = FakePermissionConnection()
    relay = _question_relay(raw_sio, connection)
    await relay.request({'options': _ASK_OPTIONS}, 4)
    resolution = {**_resolution(_permission_request(emitted), 'allow'), 'ext_requests': ext_requests}
    await _deliver(raw_sio, ACP_PERMISSION_RESOLVE_EVENT, resolution)
    assert connection.calls == [('cancel', 4)], 'A malformed or disallowed reply is not sent and nothing is guessed'
    assert '蓝色' not in caplog.text
    relay.close()


@pytest.mark.parametrize(
    ('client_questions', 'ext_requests'),
    [
        (False, None),
        (True, None),
        (False, [_RESOLVE]),
        (False, [{'method': 'session/prompt', 'params': {}}]),
        (False, 'malformed'),
        (True, []),
    ],
)
async def test_replies_without_honoured_ext_requests_are_answered_like_before_the_key_existed(
    raw_transport: Any, client_questions: bool, ext_requests: Any
) -> None:
    raw_sio, emitted = raw_transport
    connection = FakePermissionConnection()
    relay = AcpPermissionRelay(
        sio=SIOBridge(raw_sio, '/worker'), channel_id='chan-a', connection=connection, client_questions=client_questions
    )
    await relay.request({'options': _ASK_OPTIONS}, 4)
    resolution = _resolution(_permission_request(emitted), 'allow')
    if ext_requests is not None:
        resolution['ext_requests'] = ext_requests
    await _deliver(raw_sio, ACP_PERMISSION_RESOLVE_EVENT, resolution)
    assert connection.calls == [('resolve', (4, 'allow'))], 'The Host option must be applied at once, nothing sent'
    relay.close()


async def test_closing_the_relay_drops_ext_requests_still_in_flight(raw_transport: Any) -> None:
    raw_sio, emitted = raw_transport
    gate = asyncio.Event()

    class SlowConnection(FakePermissionConnection):
        async def extension_request(self, method: str, params: dict[str, Any], *, timeout: float) -> Any:
            self.calls.append(('ext', (method, params)))
            await gate.wait()

    connection = SlowConnection()
    relay = _question_relay(raw_sio, connection)
    await relay.request({'options': []}, 4)
    resolution = {**_resolution(_permission_request(emitted), 'allow'), 'ext_requests': [_RESOLVE]}
    await _deliver(raw_sio, ACP_PERMISSION_RESOLVE_EVENT, resolution)
    await _settle(connection, 1)
    relay.close()
    gate.set()
    await asyncio.sleep(0)
    assert connection.resolved == [], 'A closed session must not be answered'


@pytest.mark.parametrize('ext_requests', [None, [_RESOLVE]])
async def test_a_reply_to_a_withdrawn_request_is_dropped_without_sending_anything(
    raw_transport: Any, caplog: pytest.LogCaptureFixture, ext_requests: list[dict[str, Any]] | None
) -> None:
    raw_sio, emitted = raw_transport
    connection = FakePermissionConnection()
    relay = _question_relay(raw_sio, connection)
    await relay.request({'options': _ASK_OPTIONS}, 4)
    connection.withdrawn.add(4)
    resolution = _resolution(_permission_request(emitted), '')
    if ext_requests is not None:
        resolution['ext_requests'] = ext_requests
    with caplog.at_level('INFO', logger='nexus.extensions.agents.installed_common.host_rpc'):
        await _deliver(raw_sio, ACP_PERMISSION_RESOLVE_EVENT, resolution)
        for _ in range(5):
            await asyncio.sleep(0)
    assert connection.calls == [], 'A finished tool call must not get resolveInterruption or any answer'
    assert 'already answered or withdrawn' in caplog.text
    relay.close()


async def test_ext_requests_are_not_sent_when_the_request_is_withdrawn_before_they_start(raw_transport: Any) -> None:
    raw_sio, emitted = raw_transport
    connection = FakePermissionConnection()
    relay = _question_relay(raw_sio, connection)
    await relay.request({'options': _ASK_OPTIONS}, 4)
    resolution = {**_resolution(_permission_request(emitted), 'allow'), 'ext_requests': [_RESOLVE]}
    await _deliver(raw_sio, ACP_PERMISSION_RESOLVE_EVENT, resolution)
    connection.withdrawn.add(4)
    for _ in range(5):
        await asyncio.sleep(0)
    assert connection.calls == [], 'The prompt ended between the reply and its first extension request'
    relay.close()


async def test_end_turn_forgets_tokens_of_settled_requests_and_keeps_waiting_ones(raw_transport: Any) -> None:
    raw_sio, emitted = raw_transport
    connection = FakePermissionConnection()
    relay = AcpPermissionRelay(sio=SIOBridge(raw_sio, '/worker'), channel_id='chan-a', connection=connection)
    await relay.request({'options': []}, 1)
    settled = _permission_request(emitted)
    await relay.request({'options': []}, 2)
    waiting = _permission_request(emitted)
    connection.withdrawn.add(1)

    relay.end_turn()
    # A later request may reuse the id; the old prompt's token must not answer it.
    connection.withdrawn.clear()
    await _deliver(raw_sio, ACP_PERMISSION_RESOLVE_EVENT, _resolution(settled, 'allow'))
    await _deliver(raw_sio, ACP_PERMISSION_RESOLVE_EVENT, _resolution(waiting, 'allow'))

    assert connection.resolved == [(2, 'allow')]
    relay.close()


@pytest.mark.parametrize('option_id', ['allow', 'reject'])
@pytest.mark.parametrize(
    'reply',
    [
        {'result': {'resolved': True, 'queued': False}},
        # The current CodeBuddy acknowledges an accepted answer with an empty result.
        {'result': {}},
        {'error': {'code': -32603, 'message': 'no pending interruption for 蓝色'}},
    ],
)
async def test_real_acp_connection_resolves_interruption_before_the_permission_reply(
    raw_transport: Any, caplog: pytest.LogCaptureFixture, reply: dict[str, Any], option_id: str
) -> None:
    raw_sio, emitted = raw_transport
    reader = asyncio.StreamReader()
    writer = _AcpWriter()
    connection = AcpConnection(reader, writer, permission_policy='defer')
    relay = AcpPermissionRelay(
        sio=SIOBridge(raw_sio, '/worker'), channel_id='chan-a', connection=connection, client_questions=True
    )
    connection.deferred_permission_handler = relay.request
    connection.start()
    _feed_acp(
        reader,
        {
            'jsonrpc': '2.0',
            'id': 9,
            'method': 'session/request_permission',
            'params': {
                'sessionId': 'remote',
                'toolCall': {'toolCallId': 'call-1', '_meta': {'codebuddy.ai/toolName': 'AskUserQuestion'}},
                'options': [
                    {'optionId': 'allow', 'kind': 'allow_once'},
                    {'optionId': 'reject', 'kind': 'reject_once'},
                ],
            },
        },
    )
    for _ in range(100):
        if emitted:
            break
        await asyncio.sleep(0)
    resolution = {**_resolution(_permission_request(emitted), option_id), 'ext_requests': [_RESOLVE]}
    await _deliver(raw_sio, ACP_PERMISSION_RESOLVE_EVENT, resolution)
    (ext_request,) = await writer.wait_for(1)
    assert (ext_request['method'], ext_request['params']) == (_RESOLVE['method'], _RESOLVE['params'])
    _feed_acp(reader, {'jsonrpc': '2.0', 'id': ext_request['id'], **reply})
    frames = await writer.wait_for(2)
    assert frames[1] == {
        'jsonrpc': '2.0',
        'id': 9,
        'result': {'outcome': {'outcome': 'selected', 'optionId': option_id}},
    }, "Whatever resolveInterruption returns, the permission gets the user's choice"
    assert '蓝色' not in caplog.text
    relay.close()


async def test_real_acp_connection_sends_nothing_for_a_reply_after_the_prompt_withdrew_the_request(
    raw_transport: Any,
) -> None:
    raw_sio, emitted = raw_transport
    reader = asyncio.StreamReader()
    writer = _AcpWriter()
    connection = AcpConnection(reader, writer, permission_policy='defer')
    relay = AcpPermissionRelay(
        sio=SIOBridge(raw_sio, '/worker'), channel_id='chan-a', connection=connection, client_questions=True
    )
    connection.deferred_permission_handler = relay.request
    connection.start()
    _feed_acp(
        reader,
        {
            'jsonrpc': '2.0',
            'id': 9,
            'method': 'session/request_permission',
            'params': {'sessionId': 'remote', 'toolCall': {'toolCallId': 'call-1'}, 'options': _ASK_OPTIONS},
        },
    )
    for _ in range(100):
        if emitted:
            break
        await asyncio.sleep(0)
    assert connection.has_deferred_permission(9)
    connection._cancel_session_permissions('remote')
    (withdrawn,) = await writer.wait_for(1)
    assert withdrawn == {'jsonrpc': '2.0', 'id': 9, 'result': {'outcome': {'outcome': 'cancelled'}}}
    assert not connection.has_deferred_permission(9)

    resolution = {**_resolution(_permission_request(emitted), ''), 'ext_requests': [_RESOLVE]}
    await _deliver(raw_sio, ACP_PERMISSION_RESOLVE_EVENT, resolution)
    for _ in range(20):
        await asyncio.sleep(0)
    assert len(bytes(writer.data).splitlines()) == 1, 'No resolveInterruption may reach the CLI after the prompt'
    relay.close()
    await connection.close()
    await connection.close()


class _AcpWriter:
    def __init__(self) -> None:
        self.data = bytearray()

    def write(self, data: bytes) -> None:
        self.data.extend(data)

    async def drain(self) -> None:
        return

    async def wait_for(self, count: int) -> list[dict[str, Any]]:
        for _ in range(200):
            frames = [json.loads(line) for line in bytes(self.data).splitlines()]
            if len(frames) >= count:
                return frames
            await asyncio.sleep(0)
        raise AssertionError(f'expected {count} ACP frames, got {bytes(self.data)!r}')


def _feed_acp(reader: asyncio.StreamReader, frame: dict[str, Any]) -> None:
    reader.feed_data((json.dumps(frame) + '\n').encode())


async def test_elicitation_relay_publishes_request_and_returns_host_result(raw_transport: Any) -> None:
    raw_sio, emitted = raw_transport
    relay = AcpElicitationRelay(sio=SIOBridge(raw_sio, '/worker'), channel_id='chan-a')
    params = {'sessionId': 'remote', 'mode': 'form', 'message': '主色调？', 'requestedSchema': {}}
    pending = asyncio.create_task(relay.request(params))
    for _ in range(100):
        if emitted:
            break
        await asyncio.sleep(0)
    event, request = emitted[-1]
    assert event == ACP_ELICITATION_REQUEST_EVENT
    assert request.keys() == {'channel_id', 'elicitation_token', 'params'}
    assert (request['channel_id'], request['params']) == ('chan-a', params)
    await _deliver(
        raw_sio,
        ACP_ELICITATION_RESOLVE_EVENT,
        {**request, 'elicitation_token': 'forged', 'result': {'action': 'accept', 'content': {'x': 'no'}}},
    )
    assert not pending.done(), 'Only the token this session issued may answer'
    result = {'action': 'accept', 'content': {'color': '蓝色'}}
    await _deliver(
        raw_sio,
        ACP_ELICITATION_RESOLVE_EVENT,
        {'channel_id': 'chan-a', 'elicitation_token': request['elicitation_token'], 'result': result},
    )
    assert await pending == result
    relay.close()


async def test_closing_the_elicitation_relay_cancels_pending_requests(raw_transport: Any) -> None:
    raw_sio, emitted = raw_transport
    relay = AcpElicitationRelay(sio=SIOBridge(raw_sio, '/worker'), channel_id='chan-a')
    pending = asyncio.create_task(relay.request({'mode': 'form', 'message': '?'}))
    for _ in range(100):
        if emitted:
            break
        await asyncio.sleep(0)
    relay.close()
    with pytest.raises(asyncio.CancelledError):
        await pending
    with pytest.raises(RuntimeError, match='closed'):
        await relay.request({'mode': 'form'})


async def _respond_to_tool_request(
    raw_sio: socketio.AsyncClient, emitted: list[tuple[str, dict[str, Any]]], response: dict[str, Any]
) -> dict[str, Any]:
    while not emitted:
        await asyncio.sleep(0)
    event, request = emitted[-1]
    assert event == HOST_TOOL_REQUEST_EVENT
    await _deliver(
        raw_sio,
        HOST_TOOL_RESPONSE_EVENT,
        {'request_id': request['request_id'], 'channel_id': request['channel_id'], **response},
    )
    return request


async def test_host_tool_call_returns_host_result(raw_transport: Any) -> None:
    raw_sio, emitted = raw_transport
    client = HostToolClient(sio=SIOBridge(raw_sio, '/worker'), channel_id='chan-a')
    call = asyncio.create_task(client.call(tool_name='lookup', arguments={'q': 'x'}, timeout=5))
    request = await _respond_to_tool_request(raw_sio, emitted, {'success': True, 'result': {'answer': 1}})
    assert await call == {'answer': 1}
    assert request['tool_name'] == 'lookup'
    assert request['arguments'] == {'q': 'x'}
    assert request['channel_id'] == 'chan-a'
    client.close()


async def test_host_tool_error_response_raises_coded_error(raw_transport: Any) -> None:
    raw_sio, emitted = raw_transport
    client = HostToolClient(sio=SIOBridge(raw_sio, '/worker'), channel_id='chan-a')
    call = asyncio.create_task(client.call(tool_name='lookup', arguments={}, timeout=5))
    await _respond_to_tool_request(
        raw_sio, emitted, {'success': False, 'error': {'code': 'NOT_FOUND', 'message': 'missing'}}
    )
    with pytest.raises(HostToolError, match='missing') as error:
        await call
    assert error.value.code == 'NOT_FOUND'
    client.close()


async def test_host_tool_response_for_another_channel_is_ignored(raw_transport: Any) -> None:
    raw_sio, emitted = raw_transport
    client_a = HostToolClient(sio=SIOBridge(raw_sio, '/worker'), channel_id='chan-a')
    client_b = HostToolClient(sio=SIOBridge(raw_sio, '/worker'), channel_id='chan-b')
    call = asyncio.create_task(client_a.call(tool_name='lookup', arguments={}, timeout=5))
    while not emitted:
        await asyncio.sleep(0)
    request_id = emitted[-1][1]['request_id']
    await _deliver(
        raw_sio,
        HOST_TOOL_RESPONSE_EVENT,
        {'request_id': request_id, 'channel_id': 'chan-b', 'success': True, 'result': 'wrong'},
    )
    await asyncio.sleep(0)
    assert not call.done(), 'A response on another channel must not settle this call'
    await _deliver(
        raw_sio,
        HOST_TOOL_RESPONSE_EVENT,
        {'request_id': request_id, 'channel_id': 'chan-a', 'success': True, 'result': 'right'},
    )
    assert await call == 'right'
    client_a.close()
    client_b.close()


async def test_host_tool_timeout_raises_timeout_code(raw_transport: Any) -> None:
    raw_sio, _ = raw_transport
    client = HostToolClient(sio=SIOBridge(raw_sio, '/worker'), channel_id='chan-a')
    with pytest.raises(HostToolError) as error:
        await client.call(tool_name='lookup', arguments={}, timeout=0.01)
    assert error.value.code == 'HOST_TOOL_TIMEOUT'
    client.close()


async def test_host_tool_close_fails_pending_and_rejects_new_calls(raw_transport: Any) -> None:
    raw_sio, emitted = raw_transport
    client = HostToolClient(sio=SIOBridge(raw_sio, '/worker'), channel_id='chan-a')
    call = asyncio.create_task(client.call(tool_name='lookup', arguments={}, timeout=5))
    while not emitted:
        await asyncio.sleep(0)
    client.close()
    with pytest.raises(HostToolError) as pending_error:
        await call
    assert pending_error.value.code == 'HOST_TOOL_CLIENT_CLOSED'
    with pytest.raises(HostToolError) as closed_error:
        await client.call(tool_name='lookup', arguments={}, timeout=5)
    assert closed_error.value.code == 'HOST_TOOL_CLIENT_CLOSED'


def _tool_call(status: str) -> dict[str, Any]:
    return {
        'sessionId': 'remote',
        'update': {
            'sessionUpdate': 'tool_call',
            'toolCallId': 'call_1',
            'status': status,
            'title': '`ls`',
            'kind': 'execute',
            'rawInput': {'command': 'ls', 'description': 'List'},
            'rawOutput': {'type': 'text', 'text': 'a\nb'},
            'content': [{'type': 'content', 'content': {'type': 'text', 'text': 'List'}}],
            '_meta': {'codebuddy.ai/toolName': 'Bash'},
        },
    }


def _session_updates(emitted: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    return [payload for event, payload in emitted if event == ACP_SESSION_UPDATE_EVENT]


async def test_update_relay_stamps_a_turn_and_restarts_seq_for_the_next(raw_transport: Any) -> None:
    raw_sio, emitted = raw_transport
    relay = AcpUpdateRelay(sio=SIOBridge(raw_sio, '/worker'), channel_id='chan-a')
    relay.begin_turn('turn-1')
    for status in ('pending', 'completed'):
        relay.publish(_tool_call(status))
    assert relay.end_turn() == 2
    relay.begin_turn('turn-2')
    relay.publish(_tool_call('pending'))
    assert relay.end_turn() == 1
    await relay.aclose()

    payloads = _session_updates(emitted)
    assert [(payload['turn_id'], payload['seq']) for payload in payloads] == [
        ('turn-1', 1),
        ('turn-1', 2),
        ('turn-2', 1),
    ]
    assert {payload['channel_id'] for payload in payloads} == {'chan-a'}
    assert [payload['update']['status'] for payload in payloads] == ['pending', 'completed', 'pending']


async def test_update_relay_sends_nothing_outside_a_turn(raw_transport: Any) -> None:
    # A CLI still settling an abandoned prompt between turns belongs to no task.
    raw_sio, emitted = raw_transport
    relay = AcpUpdateRelay(sio=SIOBridge(raw_sio, '/worker'), channel_id='chan-a')
    relay.publish(_tool_call('pending'))
    relay.begin_turn('turn-1')
    assert relay.end_turn() == 0
    relay.publish(_tool_call('completed'))
    await relay.aclose()

    assert _session_updates(emitted) == []


async def test_update_relay_flush_waits_for_queued_frames(raw_transport: Any) -> None:
    raw_sio, emitted = raw_transport
    relay = AcpUpdateRelay(sio=SIOBridge(raw_sio, '/worker'), channel_id='chan-a')
    relay.begin_turn('turn-1')
    relay.publish(_tool_call('pending'))
    relay.end_turn()

    await relay.flush()

    assert len(_session_updates(emitted)) == 1
    await relay.aclose()


@pytest.mark.parametrize(
    ('update', 'expected'),
    [
        (_tool_call('pending')['update'], _tool_call('pending')['update']),
        (_tool_call('completed')['update'], _tool_call('completed')['update']),
        (
            {'sessionUpdate': 'agent_thought_chunk', 'content': {'type': 'text', 'text': 'secret plan'}},
            {'sessionUpdate': 'agent_thought_chunk'},
        ),
        (
            {
                'sessionUpdate': 'agent_message_chunk',
                'content': {'type': 'text', 'text': 'Done.'},
                'messageId': 'm1',
                '_meta': {'codebuddy.ai/requestId': 'r'},
            },
            {'sessionUpdate': 'agent_message_chunk', 'content': {'type': 'text', 'text': 'Done.'}, 'messageId': 'm1'},
        ),
        (
            {'sessionUpdate': 'session_info_update', 'title': 'Zoo', '_meta': {'codebuddy.ai/agentPhase': {}}},
            {'sessionUpdate': 'session_info_update', 'title': 'Zoo'},
        ),
        ({'sessionUpdate': 'available_commands_update', 'availableCommands': [{'name': 'clear'}]}, None),
    ],
)
async def test_trim_acp_update_keeps_only_what_the_host_reads(
    update: dict[str, Any], expected: dict[str, Any] | None
) -> None:
    assert trim_acp_update(update) == expected


_LONG_DIRECTORY_PATH = '/workspace/' + 'd' * 2000 + '/snake.html'


def _text_block(text: str) -> dict[str, Any]:
    return {'type': 'content', 'content': {'type': 'text', 'text': text}}


def _string_leaves(value: Any, key: str = '') -> list[tuple[str, str]]:
    if isinstance(value, str):
        return [(key, value)]
    if isinstance(value, dict):
        return [leaf for name, item in value.items() for leaf in _string_leaves(item, name)]
    if isinstance(value, list):
        return [leaf for item in value for leaf in _string_leaves(item, key)]
    return []


def _shape(value: Any) -> Any:
    if isinstance(value, dict):
        return {name: _shape(item) for name, item in value.items()}
    if isinstance(value, list):
        return [_shape(item) for item in value]
    return type(value).__name__


async def test_trim_acp_update_bounds_every_string_and_keeps_the_frame_structure() -> None:
    update = {
        'sessionUpdate': 'tool_call_update',
        'toolCallId': 't1',
        'status': 'completed',
        'rawInput': {'file_path': _LONG_DIRECTORY_PATH, 'content': 'w' * 200_000},
        'rawOutput': {'stdout': 'o' * 200_000, 'blank': ' ' * 200_000, 'lines': ['l' * 200_000, 3]},
        'content': [
            {'type': 'diff', 'path': _LONG_DIRECTORY_PATH, 'oldText': 'a' * 200_000, 'newText': 'b' * 200_000},
            _text_block('t' * 200_000),
        ],
        '_meta': {'codebuddy.ai/toolName': 'Write'},
    }

    trimmed = trim_acp_update(update)

    assert trimmed is not None
    assert _shape(trimmed) == _shape(update), 'No field, list item or nesting may be dropped'
    leaves = list(zip(_string_leaves(update), _string_leaves(trimmed), strict=True))
    for (key, original), (_, cut) in leaves:
        assert original.startswith(cut), 'A string is only ever cut to a prefix'
        if key == 'path' or key == 'file_path':
            assert cut == original
        else:
            assert len(cut) <= 2 * _PUSHED_VISIBLE_CHARS
            assert len(cut) == min(len(original), _PUSHED_VISIBLE_CHARS) or not original[:_PUSHED_VISIBLE_CHARS].strip()
    assert len(json.dumps(trimmed)) < 250_000, 'Seven 200k-character strings must fit in a bounded frame'


async def test_trim_acp_update_pushes_assistant_text_whole() -> None:
    text = 'The snake game is done. ' * 10_000
    update = {'sessionUpdate': 'agent_message_chunk', 'content': {'type': 'text', 'text': text}}

    assert trim_acp_update(update) == update


async def test_update_relay_skips_dropped_updates_without_counting_them(raw_transport: Any) -> None:
    raw_sio, emitted = raw_transport
    relay = AcpUpdateRelay(sio=SIOBridge(raw_sio, '/worker'), channel_id='chan-a')
    relay.begin_turn('turn-1')
    relay.publish({'sessionId': 'remote', 'update': {'sessionUpdate': 'available_commands_update'}})
    relay.publish(_tool_call('pending'))

    assert relay.end_turn() == 1
    await relay.flush()
    assert [payload['seq'] for payload in _session_updates(emitted)] == [1]
    await relay.aclose()


async def test_update_relay_keeps_publishing_after_an_emit_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_sio = socketio.AsyncClient()
    emitted: list[dict[str, Any]] = []
    calls = 0

    async def flaky_emit(event: str, data: dict[str, Any], namespace: str | None = None) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ConnectionError('socket down')
        emitted.append(data)

    monkeypatch.setattr(raw_sio, 'emit', AsyncMock(side_effect=flaky_emit))
    relay = AcpUpdateRelay(sio=SIOBridge(raw_sio, '/worker'), channel_id='chan-a')
    relay.begin_turn('turn-1')
    relay.publish(_tool_call('pending'))
    relay.publish(_tool_call('completed'))

    # The lost frame still counts, so the Host receives one frame fewer than the turn reports.
    assert relay.end_turn() == 2
    await relay.flush()
    assert [payload['seq'] for payload in emitted] == [2]
    await relay.aclose()


async def test_update_relay_ignores_updates_after_close(raw_transport: Any) -> None:
    raw_sio, emitted = raw_transport
    relay = AcpUpdateRelay(sio=SIOBridge(raw_sio, '/worker'), channel_id='chan-a')
    relay.begin_turn('turn-1')
    await relay.aclose()
    relay.publish(_tool_call('pending'))
    await asyncio.sleep(0)

    assert _session_updates(emitted) == []


async def test_update_relay_keeps_only_the_session_its_turn_names(raw_transport: Any) -> None:
    # A WorkBuddy fork shares its source's process, so both sessions' updates reach every listener.
    raw_sio, emitted = raw_transport
    relay = AcpUpdateRelay(sio=SIOBridge(raw_sio, '/worker'), channel_id='chan-fork')
    relay.begin_turn('turn-1', session_id='remote-fork')
    relay.publish({**_tool_call('pending'), 'sessionId': 'remote-source'})
    relay.publish({**_tool_call('completed'), 'sessionId': 'remote-fork'})
    assert relay.end_turn() == 1
    relay.begin_turn('turn-2')
    relay.publish({**_tool_call('pending'), 'sessionId': 'remote-source'})
    assert relay.end_turn() == 1, 'A turn that names no session keeps every update, as for CodeBuddy'
    await relay.aclose()

    assert [payload['update']['status'] for payload in _session_updates(emitted)] == ['completed', 'pending']


@pytest.mark.parametrize('kind', ['usage_update', 'config_option_update'])
async def test_trim_acp_update_drops_version_2_bookkeeping(kind: str) -> None:
    assert trim_acp_update({'sessionUpdate': kind, 'used': 1, 'configOptions': []}) is None


async def test_trim_acp_update_keeps_command_paths_and_urls_whole() -> None:
    update = {
        'sessionUpdate': '_codebuddy.ai/command',
        'action': 'presentFiles',
        'params': {'files': [_LONG_DIRECTORY_PATH], 'focusFile': _LONG_DIRECTORY_PATH},
    }
    trimmed = trim_acp_update(update)
    assert trimmed is not None
    assert trimmed['params']['focusFile'] == _LONG_DIRECTORY_PATH, 'The Host registers the focus file by its path'
    url = 'https://example.com/' + 'q' * 2000
    browser = trim_acp_update({
        'sessionUpdate': '_codebuddy.ai/command',
        'action': 'openBrowser',
        'params': {'url': url},
    })
    assert browser is not None
    assert browser['params']['url'] == url
    result = trim_acp_update({
        'sessionUpdate': '_codebuddy.ai/command',
        'action': 'openResultView',
        'params': {'targetFile': _LONG_DIRECTORY_PATH},
    })
    assert result is not None
    assert result['params']['targetFile'] == _LONG_DIRECTORY_PATH
