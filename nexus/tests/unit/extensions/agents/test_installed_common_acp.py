from __future__ import annotations

import asyncio
import json

from typing import Any

import pytest

from nexus.extensions.agents.installed_common.acp import (
    ACP_COMMAND_METHOD,
    ACP_PROMPT_META,
    AcpConnection,
    AcpProtocolError,
    AcpRemoteError,
    PermissionPolicy,
    accepted_request_id,
    assistant_text_from_updates,
    side_question_turn_limit,
    supports_prompt_steer,
    supports_session_fork,
    supports_side_question,
)


class _Writer:
    def __init__(self) -> None:
        self.data = bytearray()

    def write(self, data: bytes) -> None:
        self.data.extend(data)

    async def drain(self) -> None:
        return

    def frames(self) -> list[dict[str, Any]]:
        return [json.loads(line) for line in bytes(self.data).splitlines()]


async def _wait_for_frames(writer: _Writer, count: int) -> list[dict[str, Any]]:
    for _ in range(100):
        frames = writer.frames()
        if len(frames) >= count:
            return frames
        await asyncio.sleep(0)
    raise AssertionError(f'expected {count} frames, got {writer.frames()!r}')


def _feed(reader: asyncio.StreamReader, frame: dict[str, Any]) -> None:
    reader.feed_data((json.dumps(frame) + '\n').encode())


@pytest.mark.asyncio
async def test_acp_initialize_and_prompt_collect_committed_update() -> None:
    reader = asyncio.StreamReader()
    writer = _Writer()
    connection = AcpConnection(reader, writer)

    init_task = asyncio.create_task(connection.initialize())
    init_request = (await _wait_for_frames(writer, 1))[0]
    _feed(reader, {'jsonrpc': '2.0', 'id': init_request['id'], 'result': {'protocolVersion': 1}})
    assert (await init_task)['protocolVersion'] == 1

    prompt_task = asyncio.create_task(connection.prompt(session_id='s1', text='hello', timeout=1))
    prompt_request = (await _wait_for_frames(writer, 2))[1]
    _feed(
        reader,
        {
            'jsonrpc': '2.0',
            'method': 'session/update',
            'params': {
                'sessionId': 's1',
                'update': {
                    'sessionUpdate': 'agent_message_chunk',
                    'content': {'type': 'text', 'text': 'answer'},
                },
            },
        },
    )
    _feed(reader, {'jsonrpc': '2.0', 'id': prompt_request['id'], 'result': {'stopReason': 'end_turn'}})

    assert await prompt_task == {'stopReason': 'end_turn'}
    assert assistant_text_from_updates(connection.drain_updates(session_id='s1')) == 'answer'
    await connection.close()


@pytest.mark.asyncio
async def test_acp_permission_policy_selects_allow_once() -> None:
    reader = asyncio.StreamReader()
    writer = _Writer()
    connection = AcpConnection(reader, writer, permission_policy='allow')
    prompt_task = asyncio.create_task(connection.prompt(session_id='s1', text='run', timeout=1))
    prompt_request = (await _wait_for_frames(writer, 1))[0]

    _feed(
        reader,
        {
            'jsonrpc': '2.0',
            'id': 99,
            'method': 'session/request_permission',
            'params': {
                'sessionId': 's1',
                'options': [
                    {'optionId': 'reject-once', 'kind': 'reject_once'},
                    {'optionId': 'allow-once', 'kind': 'allow_once'},
                ],
                'toolCall': {'toolCallId': 'tool-1'},
            },
        },
    )
    permission_response = (await _wait_for_frames(writer, 2))[1]
    assert permission_response == {
        'jsonrpc': '2.0',
        'id': 99,
        'result': {'outcome': {'outcome': 'selected', 'optionId': 'allow-once'}},
    }

    _feed(reader, {'jsonrpc': '2.0', 'id': prompt_request['id'], 'result': {'stopReason': 'end_turn'}})
    await prompt_task
    await connection.close()


@pytest.mark.asyncio
async def test_acp_invalid_json_fails_pending_request() -> None:
    reader = asyncio.StreamReader()
    writer = _Writer()
    connection = AcpConnection(reader, writer)
    task = asyncio.create_task(connection.request('initialize', {'protocolVersion': 1}, timeout=1))
    await _wait_for_frames(writer, 1)
    reader.feed_data(b'not-json\n')

    with pytest.raises(AcpProtocolError, match='invalid JSON'):
        await task
    await connection.close()


@pytest.mark.asyncio
async def test_acp_reject_policy_selects_reject_once() -> None:
    connection = AcpConnection(asyncio.StreamReader(), _Writer(), permission_policy='reject')

    assert connection._permission_response({
        'options': [
            {'optionId': 'allow-once', 'kind': 'allow_once'},
            {'optionId': 'reject-once', 'kind': 'reject_once'},
        ]
    }) == {'outcome': {'outcome': 'selected', 'optionId': 'reject-once'}}


@pytest.mark.asyncio
async def test_drain_updates_preserves_other_sessions() -> None:
    connection = AcpConnection(asyncio.StreamReader(), _Writer())
    connection.updates.put_nowait({'sessionId': 's2', 'update': {'value': 2}})
    connection.updates.put_nowait({'sessionId': 's1', 'update': {'value': 1}})

    assert connection.drain_updates(session_id='s1') == [{'sessionId': 's1', 'update': {'value': 1}}]
    assert connection.drain_updates(session_id='s2') == [{'sessionId': 's2', 'update': {'value': 2}}]


def _update_frame(value: int) -> dict[str, Any]:
    return {'jsonrpc': '2.0', 'method': 'session/update', 'params': {'sessionId': 's1', 'update': {'value': value}}}


async def _wait_for_queue(connection: AcpConnection, count: int) -> None:
    for _ in range(100):
        if connection.updates.qsize() >= count:
            return
        await asyncio.sleep(0)
    raise AssertionError(f'expected {count} queued updates, got {connection.updates.qsize()}')


@pytest.mark.asyncio
async def test_update_listeners_see_every_update_in_order_and_the_queue_still_drains() -> None:
    reader = asyncio.StreamReader()
    connection = AcpConnection(reader, _Writer())
    seen: list[int] = []
    connection.add_update_listener(lambda params: seen.append(params['update']['value']))
    connection.start()

    for value in (1, 2, 3):
        _feed(reader, _update_frame(value))
    await _wait_for_queue(connection, 3)

    assert seen == [1, 2, 3]
    assert [item['update']['value'] for item in connection.drain_updates(session_id='s1')] == [1, 2, 3]
    await connection.close()


@pytest.mark.asyncio
async def test_failing_update_listener_does_not_block_other_listeners_or_the_queue() -> None:
    reader = asyncio.StreamReader()
    connection = AcpConnection(reader, _Writer())
    seen: list[int] = []

    def explode(params: dict[str, Any]) -> None:
        raise RuntimeError('listener bug')

    connection.add_update_listener(explode)
    connection.add_update_listener(lambda params: seen.append(params['update']['value']))
    connection.start()

    _feed(reader, _update_frame(1))
    await _wait_for_queue(connection, 1)

    assert seen == [1]
    assert len(connection.drain_updates(session_id='s1')) == 1
    await connection.close()


@pytest.mark.asyncio
async def test_removed_update_listener_stops_receiving_updates() -> None:
    reader = asyncio.StreamReader()
    connection = AcpConnection(reader, _Writer())
    seen: list[int] = []
    remove = connection.add_update_listener(lambda params: seen.append(params['update']['value']))
    connection.start()

    _feed(reader, _update_frame(1))
    await _wait_for_queue(connection, 1)
    remove()
    remove()
    _feed(reader, _update_frame(2))
    await _wait_for_queue(connection, 2)

    assert seen == [1]
    await connection.close()


def _permission(session_id: str, request_id: int) -> dict[str, Any]:
    return {
        'jsonrpc': '2.0',
        'id': request_id,
        'method': 'session/request_permission',
        'params': {
            'sessionId': session_id,
            'options': [
                {'optionId': 'approve-opaque', 'kind': 'allow_once'},
                {'optionId': 'deny-opaque', 'kind': 'reject_once'},
            ],
        },
    }


@pytest.mark.asyncio
async def test_deferred_permission_does_not_block_updates_or_other_responses() -> None:
    entered = asyncio.Event()

    async def handler(params: dict[str, Any], request_id: int) -> None:
        entered.set()
        await asyncio.Event().wait()

    reader, writer = asyncio.StreamReader(), _Writer()
    connection = AcpConnection(reader, writer, permission_policy='defer', deferred_permission_handler=handler)
    prompt = asyncio.create_task(connection.prompt(session_id='s1', text='run', timeout=1))
    prompt_id = (await _wait_for_frames(writer, 1))[0]['id']
    _feed(reader, _permission('s1', 99))
    await asyncio.wait_for(entered.wait(), 1)
    other = asyncio.create_task(connection.request('other', {}, timeout=1))
    other_id = (await _wait_for_frames(writer, 2))[1]['id']
    _feed(reader, {'jsonrpc': '2.0', 'method': 'session/update', 'params': {'sessionId': 's1'}})
    _feed(reader, {'jsonrpc': '2.0', 'id': other_id, 'result': {}})
    assert await asyncio.wait_for(other, 0.2) == {}, 'approval must not block unrelated responses'
    assert connection.drain_updates() == [{'sessionId': 's1'}], 'approval must not block updates'
    _feed(reader, {'jsonrpc': '2.0', 'id': prompt_id, 'result': {'stopReason': 'end_turn'}})
    assert await asyncio.wait_for(prompt, 0.2) == {'stopReason': 'end_turn'}, 'terminal response must progress'
    await connection.close()
    assert not connection._deferred_permissions, 'terminal cleanup must release permissions'


@pytest.mark.asyncio
async def test_cancel_only_settles_matching_session_and_rejects_late_permissions() -> None:
    seen: asyncio.Queue[int] = asyncio.Queue()

    async def handler(params: dict[str, Any], request_id: int) -> None:
        seen.put_nowait(request_id)

    reader, writer = asyncio.StreamReader(), _Writer()
    connection = AcpConnection(reader, writer, permission_policy='defer', deferred_permission_handler=handler)
    connection.start()
    for session_id, request_id in [('s1', 91), ('s2', 92)]:
        _feed(reader, _permission(session_id, request_id))
        assert await asyncio.wait_for(seen.get(), 1) == request_id, 'callback must receive request ID'
    await connection.cancel(session_id='s1')
    await connection.cancel(session_id='s1')
    _feed(reader, _permission('s1', 93))
    assert connection.resolve_deferred_permission(92, option_id='approve-opaque'), 'other session must remain pending'
    frames = await _wait_for_frames(writer, 5)
    results = {frame['id']: frame['result'] for frame in frames if 'id' in frame}
    assert results[91] == results[93] == {'outcome': {'outcome': 'cancelled'}}, 'cancel must reject late requests'
    assert results[92]['outcome']['optionId'] == 'approve-opaque', 'other session must keep its decision'
    assert seen.empty(), 'late permission must not enter the callback'
    assert not connection.resolve_deferred_permission(91, option_id='approve-opaque'), 'late resolution must fail'
    await connection.close()


@pytest.mark.asyncio
async def test_deferred_timeout_includes_callback_execution() -> None:
    stopped = asyncio.Event()

    async def handler(params: dict[str, Any], request_id: int) -> None:
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    reader, writer = asyncio.StreamReader(), _Writer()
    connection = AcpConnection(
        reader, writer, permission_policy='defer', deferred_permission_handler=handler, deferred_permission_timeout=0.01
    )
    connection.start()
    _feed(reader, _permission('s1', 99))
    await asyncio.wait_for(stopped.wait(), 0.5)
    response = (await _wait_for_frames(writer, 1))[0]
    assert response['result'] == {'outcome': {'outcome': 'cancelled'}}, 'timeout must fail closed'
    assert not connection._closed, 'permission timeout must not close a healthy transport'
    await connection.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('terminal', ['eof', 'close', 'invalid_json'])
async def test_transport_shutdown_settles_pending_permissions_and_cannot_restart(terminal: str) -> None:
    entered, stopped = asyncio.Event(), asyncio.Event()

    async def handler(params: dict[str, Any], request_id: int) -> None:
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    reader, writer = asyncio.StreamReader(), _Writer()
    connection = AcpConnection(reader, writer, permission_policy='defer', deferred_permission_handler=handler)
    pending = asyncio.create_task(connection.request('other', {}, timeout=1))
    await _wait_for_frames(writer, 1)
    _feed(reader, _permission('s1', 99))
    await asyncio.wait_for(entered.wait(), 1)
    if terminal == 'eof':
        reader.feed_eof()
    elif terminal == 'invalid_json':
        reader.feed_data(b'broken\n')
    else:
        await connection.close()
    with pytest.raises(AcpProtocolError):
        await asyncio.wait_for(pending, 0.2)
    await asyncio.wait_for(stopped.wait(), 0.2)
    assert not connection._pending, 'shutdown must empty pending responses'
    assert not connection._deferred_permissions, 'shutdown must empty pending permissions'
    with pytest.raises(AcpProtocolError, match='closed'):
        await connection.request('after-close', {}, timeout=0.01)
    await connection.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(('policy', 'kind'), [('allow', 'allow_once'), ('reject', 'reject_once')])
async def test_automatic_permission_uses_exact_kind_and_never_guesses_ids(policy: PermissionPolicy, kind: str) -> None:
    connection = AcpConnection(asyncio.StreamReader(), _Writer(), permission_policy=policy)
    wrong_kind = 'reject_once' if policy == 'allow' else 'allow_once'
    assert connection._permission_response({'options': [{'optionId': f'{policy}-once', 'kind': wrong_kind}]}) == {
        'outcome': {'outcome': 'cancelled'}
    }, 'an ID that looks like approval must not override its kind'
    assert connection._permission_response({'options': [{'optionId': 'opaque', 'kind': kind}]}) == {
        'outcome': {'outcome': 'selected', 'optionId': 'opaque'}
    }, 'opaque IDs must be preserved'


@pytest.mark.asyncio
@pytest.mark.parametrize('decision', ['missing', 'invalid_kind', 'exception', 'malformed'])
async def test_deferred_handler_invalid_decision_fails_closed(decision: str) -> None:
    async def handler(params: dict[str, Any], request_id: int) -> dict[str, Any]:
        if decision == 'exception':
            raise RuntimeError('approval service unavailable')
        if decision == 'malformed':
            return {'outcome': 'selected'}
        return {'outcome': {'outcome': 'selected', 'optionId': 'missing' if decision == 'missing' else 'bad-kind'}}

    reader, writer = asyncio.StreamReader(), _Writer()
    connection = AcpConnection(reader, writer, permission_policy='defer', deferred_permission_handler=handler)
    connection.start()
    frame = _permission('s1', 99)
    frame['params']['options'].append({'optionId': 'bad-kind', 'kind': 'ALLOW_ONCE'})
    _feed(reader, frame)
    response = (await _wait_for_frames(writer, 1))[0]
    assert response['result'] == {'outcome': {'outcome': 'cancelled'}}, 'invalid decisions must never authorize'
    await connection.close()


@pytest.mark.asyncio
async def test_deferred_resolution_validates_options_and_is_single_use() -> None:
    entered: asyncio.Queue[int] = asyncio.Queue()

    async def handler(params: dict[str, Any], request_id: int) -> None:
        entered.put_nowait(request_id)

    reader, writer = asyncio.StreamReader(), _Writer()
    connection = AcpConnection(reader, writer, permission_policy='defer', deferred_permission_handler=handler)
    connection.start()
    for request_id in [91, 92, 93]:
        _feed(reader, _permission('s1', request_id))
        await asyncio.wait_for(entered.get(), 1)
    assert not connection.resolve_deferred_permission(91, option_id='invented'), 'unknown option must be refused'
    assert not connection.resolve_deferred_permission(999, option_id='approve-opaque'), 'unknown request must fail'
    assert connection.resolve_deferred_permission(92, option_id='approve-opaque'), 'valid choice must resolve'
    assert not connection.resolve_deferred_permission(92, option_id='deny-opaque'), 'duplicate resolution must fail'
    assert connection.cancel_deferred_permission(93), 'explicit cancellation must resolve'
    assert not connection.cancel_deferred_permission(93), 'duplicate cancellation must fail'
    frames = await _wait_for_frames(writer, 3)
    assert frames[0]['result'] == {'outcome': {'outcome': 'cancelled'}}, 'invalid resolution must fail closed'
    await connection.close()


@pytest.mark.asyncio
async def test_defer_requires_handler_before_start_but_allows_late_assignment() -> None:
    connection = AcpConnection(asyncio.StreamReader(), _Writer(), permission_policy='defer')
    with pytest.raises(ValueError, match='handler'):
        connection.start()
    with pytest.raises(ValueError, match='handler'):
        await connection.request('test', {}, timeout=0.01)

    async def handler(params: dict[str, Any], request_id: int) -> None:
        return None

    connection.deferred_permission_handler = handler
    connection.start()
    await connection.close()


@pytest.mark.asyncio
async def test_noise_is_opt_in_and_retention_is_bounded() -> None:
    reader, writer = asyncio.StreamReader(), _Writer()
    connection = AcpConnection(reader, writer, tolerate_noise=True)
    pending = asyncio.create_task(connection.request('test', {}, timeout=1))
    request_id = (await _wait_for_frames(writer, 1))[0]['id']
    reader.feed_data(((('x' * 4000) + '\n') * 150).encode())
    reader.feed_data(b'[]\n')
    _feed(reader, {'jsonrpc': '2.0', 'id': request_id, 'result': {}})
    assert await pending == {}, 'noise must not prevent a valid response'
    assert len(connection.noise) <= 100, 'noise retention must have a line bound'
    assert all(len(line) <= 2048 for line in connection.noise), 'noise retention must have a per-line bound'
    await connection.close()


@pytest.mark.asyncio
async def test_new_session_passes_standard_mcp_servers() -> None:
    reader, writer = asyncio.StreamReader(), _Writer()
    connection = AcpConnection(reader, writer)
    servers = [{'name': 'tools', 'command': 'python', 'args': ['server.py'], 'env': []}]
    pending = asyncio.create_task(connection.new_session(cwd='/workspace', mcp_servers=servers))
    request = (await _wait_for_frames(writer, 1))[0]
    assert request['params']['mcpServers'] == servers, 'MCP definitions must be passed without rewriting'
    _feed(reader, {'jsonrpc': '2.0', 'id': request['id'], 'result': {'sessionId': 's1'}})
    assert await pending == 's1', 'session response must preserve existing contract'
    await connection.close()


@pytest.mark.asyncio
async def test_next_prompt_reopens_cancelled_session_permissions() -> None:
    entered = asyncio.Event()

    async def handler(params: dict[str, Any], request_id: int) -> dict[str, Any]:
        entered.set()
        return {'outcome': {'outcome': 'selected', 'optionId': 'approve-opaque'}}

    reader, writer = asyncio.StreamReader(), _Writer()
    connection = AcpConnection(reader, writer, permission_policy='defer', deferred_permission_handler=handler)
    await connection.cancel(session_id='s1')
    prompt = asyncio.create_task(connection.prompt(session_id='s1', text='next', timeout=1))
    request = (await _wait_for_frames(writer, 2))[1]
    _feed(reader, _permission('s1', 99))
    await asyncio.wait_for(entered.wait(), 1)
    frames = await _wait_for_frames(writer, 3)
    assert frames[2]['result']['outcome']['optionId'] == 'approve-opaque', 'new prompt must permit fresh decisions'
    _feed(reader, {'jsonrpc': '2.0', 'id': request['id'], 'result': {'stopReason': 'end_turn'}})
    await prompt
    await connection.close()


@pytest.mark.asyncio
async def test_write_failure_closes_reader_and_settles_all_pending_state() -> None:
    class BrokenWriter(_Writer):
        broken = False

        async def drain(self) -> None:
            if self.broken:
                raise OSError('writer unavailable')

    entered, stopped = asyncio.Event(), asyncio.Event()

    async def handler(params: dict[str, Any], request_id: int) -> None:
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    reader, writer = asyncio.StreamReader(), BrokenWriter()
    connection = AcpConnection(reader, writer, permission_policy='defer', deferred_permission_handler=handler)
    pending = asyncio.create_task(connection.request('first', {}, timeout=1))
    await _wait_for_frames(writer, 1)
    _feed(reader, _permission('s1', 99))
    await asyncio.wait_for(entered.wait(), 1)
    writer.broken = True
    with pytest.raises(AcpProtocolError, match='write'):
        await connection.request('second', {}, timeout=1)
    with pytest.raises(AcpProtocolError, match='write'):
        await pending
    await asyncio.wait_for(connection.close(), 0.2)
    assert stopped.is_set(), 'write failure must cancel approval callback'
    assert connection._reader_task is not None, 'reader must have started before write failure'
    assert connection._reader_task.done(), 'write failure must stop reader'
    assert not connection._pending, 'write failure must settle pending responses'
    assert not connection._deferred_permissions, 'write failure must settle pending permissions'
    assert not connection._permission_tasks, 'write failure must join managed tasks'


@pytest.mark.asyncio
async def test_permission_callback_can_close_connection_without_self_join() -> None:
    finished = asyncio.Event()
    reader, writer = asyncio.StreamReader(), _Writer()
    connection = AcpConnection(reader, writer, permission_policy='defer')

    async def handler(params: dict[str, Any], request_id: int) -> None:
        await connection.close()
        finished.set()

    connection.deferred_permission_handler = handler
    connection.start()
    _feed(reader, _permission('s1', 99))
    await asyncio.wait_for(finished.wait(), 0.2)
    await asyncio.wait_for(connection.close(), 0.2)
    assert connection._closed, 'callback close must terminate connection'
    assert not connection._deferred_permissions, 'callback close must settle pending permissions'


@pytest.mark.asyncio
async def test_external_close_joins_callback_tail_after_callback_initiates_close() -> None:
    closing, release, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()
    reader, writer = asyncio.StreamReader(), _Writer()
    connection = AcpConnection(reader, writer, permission_policy='defer')

    async def handler(params: dict[str, Any], request_id: int) -> None:
        await connection.close()
        closing.set()
        await release.wait()
        finished.set()

    connection.deferred_permission_handler = handler
    connection.start()
    _feed(reader, _permission('s1', 99))
    await asyncio.wait_for(closing.wait(), 0.2)
    external_close = asyncio.create_task(connection.close())
    try:
        done, _ = await asyncio.wait({external_close}, timeout=0.01)
        assert not done, 'external close must wait for the callback that initiated shutdown'
    finally:
        release.set()
        await asyncio.wait_for(external_close, 0.2)
    assert finished.is_set(), 'external close must wait for asynchronous callback cleanup'
    assert not connection._permission_tasks, 'callback-initiated shutdown must join every managed task'


@pytest.mark.asyncio
async def test_cancelled_close_keeps_cleanup_joinable() -> None:
    entered, stopping, release, stopped = (asyncio.Event() for _ in range(4))

    async def handler(params: dict[str, Any], request_id: int) -> None:
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopping.set()
            await release.wait()
            stopped.set()

    reader, writer = asyncio.StreamReader(), _Writer()
    connection = AcpConnection(reader, writer, permission_policy='defer', deferred_permission_handler=handler)
    connection.start()
    _feed(reader, _permission('s1', 99))
    await asyncio.wait_for(entered.wait(), 1)
    closing = asyncio.create_task(connection.close())
    await asyncio.wait_for(stopping.wait(), 1)
    closing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await closing
    release.set()
    await asyncio.wait_for(connection.close(), 0.2)
    assert stopped.is_set(), 'cancelling close caller must not cancel cleanup itself'
    assert not connection._permission_tasks, 'a later close must join all cancelled callbacks'


@pytest.mark.asyncio
async def test_cancelled_permission_callback_fails_closed() -> None:
    async def handler(params: dict[str, Any], request_id: int) -> None:
        raise asyncio.CancelledError

    reader, writer = asyncio.StreamReader(), _Writer()
    connection = AcpConnection(reader, writer, permission_policy='defer', deferred_permission_handler=handler)
    connection.start()
    _feed(reader, _permission('s1', 99))
    frames = await _wait_for_frames(writer, 1)
    assert frames[0]['result'] == {'outcome': {'outcome': 'cancelled'}}, 'callback cancellation must send a response'
    await connection.close()


@pytest.mark.asyncio
async def test_outbound_request_id_and_permission_id_have_separate_namespaces() -> None:
    entered = asyncio.Event()

    async def handler(params: dict[str, Any], request_id: int) -> None:
        entered.set()

    reader, writer = asyncio.StreamReader(), _Writer()
    connection = AcpConnection(reader, writer, permission_policy='defer', deferred_permission_handler=handler)
    pending = asyncio.create_task(connection.request('test', {}, timeout=1))
    request_id = (await _wait_for_frames(writer, 1))[0]['id']
    _feed(reader, _permission('s1', request_id))
    await asyncio.wait_for(entered.wait(), 1)
    _feed(reader, {'jsonrpc': '2.0', 'id': request_id, 'result': {'response': 'ok'}})
    assert await pending == {'response': 'ok'}, 'incoming permission must not replace outbound future'
    assert connection.resolve_deferred_permission(request_id, option_id='approve-opaque'), 'permission must resolve'
    frames = await _wait_for_frames(writer, 2)
    assert frames[1]['result']['outcome']['optionId'] == 'approve-opaque', 'permission needs its own response'
    await connection.close()


@pytest.mark.asyncio
async def test_external_resolution_interrupts_callback_without_waiting_for_timeout() -> None:
    entered, stopped = asyncio.Event(), asyncio.Event()

    async def handler(params: dict[str, Any], request_id: int) -> None:
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    reader, writer = asyncio.StreamReader(), _Writer()
    connection = AcpConnection(reader, writer, permission_policy='defer', deferred_permission_handler=handler)
    connection.start()
    _feed(reader, _permission('s1', 99))
    await asyncio.wait_for(entered.wait(), 1)
    assert connection.resolve_deferred_permission(99, option_id='approve-opaque'), 'external decision must resolve'
    await asyncio.wait_for(stopped.wait(), 0.2)
    response = (await _wait_for_frames(writer, 1))[0]
    assert response['result']['outcome']['optionId'] == 'approve-opaque', 'must reply once'
    await connection.close()


@pytest.mark.asyncio
async def test_cancel_before_callback_runs_does_not_request_user_approval() -> None:
    entered = asyncio.Event()

    async def handler(params: dict[str, Any], request_id: int) -> None:
        entered.set()

    writer = _Writer()
    connection = AcpConnection(
        asyncio.StreamReader(), writer, permission_policy='defer', deferred_permission_handler=handler
    )
    connection.start()
    await connection._dispatch(_permission('s1', 99))
    await connection.cancel(session_id='s1')
    frames = await _wait_for_frames(writer, 2)
    assert not entered.is_set(), 'cancelled permission must not call user approval handler'
    assert frames[1]['result'] == {'outcome': {'outcome': 'cancelled'}}, 'cancelled permission must still respond'
    await connection.close()


@pytest.mark.asyncio
async def test_terminal_response_closes_permission_admission_before_reader_continues() -> None:
    entered = asyncio.Event()

    async def handler(params: dict[str, Any], request_id: int) -> None:
        entered.set()

    reader, writer = asyncio.StreamReader(), _Writer()
    connection = AcpConnection(reader, writer, permission_policy='defer', deferred_permission_handler=handler)
    prompt = asyncio.create_task(connection.prompt(session_id='s1', text='run', timeout=1))
    request_id = (await _wait_for_frames(writer, 1))[0]['id']
    await connection._dispatch({'jsonrpc': '2.0', 'id': request_id, 'result': {'stopReason': 'end_turn'}})
    await connection._dispatch(_permission('s1', 99))
    assert 99 not in connection._deferred_permissions, 'terminal response must reject late permissions immediately'
    await prompt
    assert not entered.is_set(), 'late permission must not run user approval handler'
    await connection.close()


@pytest.mark.asyncio
async def test_deferred_timeout_also_bounds_wait_after_callback_returns_none() -> None:
    entered = asyncio.Event()

    async def handler(params: dict[str, Any], request_id: int) -> None:
        entered.set()

    reader, writer = asyncio.StreamReader(), _Writer()
    connection = AcpConnection(
        reader, writer, permission_policy='defer', deferred_permission_handler=handler, deferred_permission_timeout=0.01
    )
    connection.start()
    _feed(reader, _permission('s1', 99))
    await asyncio.wait_for(entered.wait(), 1)
    await asyncio.wait_for(asyncio.gather(*connection._permission_tasks), 0.2)
    assert writer.frames()[0]['result'] == {'outcome': {'outcome': 'cancelled'}}, 'pending decision must time out'
    assert not connection.resolve_deferred_permission(99, option_id='approve-opaque'), 'timed-out resolve must fail'
    await connection.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('trigger', ['cancel', 'timeout', 'terminal'])
async def test_close_after_permission_cancellation_preserves_callback_async_cleanup(trigger: str) -> None:
    entered, cleaning, release, cleaned = (asyncio.Event() for _ in range(4))

    async def handler(params: dict[str, Any], request_id: int) -> None:
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaning.set()
            await release.wait()
            cleaned.set()

    reader, writer = asyncio.StreamReader(), _Writer()
    connection = AcpConnection(
        reader,
        writer,
        permission_policy='defer',
        deferred_permission_handler=handler,
        deferred_permission_timeout=0.01 if trigger == 'timeout' else 300,
    )
    prompt = asyncio.create_task(connection.prompt(session_id='s1', text='run', timeout=1))
    request_id = (await _wait_for_frames(writer, 1))[0]['id']
    _feed(reader, _permission('s1', 99))
    await asyncio.wait_for(entered.wait(), 0.2)
    if trigger == 'cancel':
        await connection.cancel(session_id='s1')
    if trigger != 'timeout':
        _feed(reader, {'jsonrpc': '2.0', 'id': request_id, 'result': {'stopReason': 'cancelled'}})
        await prompt
    await asyncio.wait_for(cleaning.wait(), 0.2)
    external_close = asyncio.create_task(connection.close())
    try:
        done, _ = await asyncio.wait({external_close}, timeout=0.01)
        assert not done, 'close must join callback cleanup rather than cancel it a second time'
    finally:
        release.set()
        await asyncio.wait_for(external_close, 0.2)
        if trigger == 'timeout':
            with pytest.raises(AcpProtocolError):
                await prompt
    assert cleaned.is_set(), 'callback finally must run to completion'
    assert not connection._permission_tasks, 'close must join all callback cleanup'


async def _initialize_request(**kwargs: Any) -> dict[str, Any]:
    reader, writer = asyncio.StreamReader(), _Writer()
    connection = AcpConnection(reader, writer)
    init_task = asyncio.create_task(connection.initialize(**kwargs))
    request = (await _wait_for_frames(writer, 1))[0]
    _feed(reader, {'jsonrpc': '2.0', 'id': request['id'], 'result': {'protocolVersion': 1}})
    await init_task
    await connection.close()
    return request


@pytest.mark.asyncio
async def test_initialize_declares_no_client_capabilities_by_default() -> None:
    request = await _initialize_request()
    assert request['params']['clientCapabilities'] == {}, 'Other agents must keep the exact handshake they had'


@pytest.mark.asyncio
async def test_initialize_declares_the_given_client_capabilities() -> None:
    capabilities = {'elicitation': {'form': {}}, '_meta': {'codebuddy.ai': {'question': True}}}
    request = await _initialize_request(client_capabilities=capabilities)
    assert request['params']['clientCapabilities'] == capabilities


def _elicitation(request_id: int, session_id: str | None = 's1') -> dict[str, Any]:
    params: dict[str, Any] = {
        'mode': 'form',
        'message': '主色调用哪个？',
        'requestedSchema': {'type': 'object', 'properties': {'color': {'type': 'string'}}},
    }
    if session_id is not None:
        params['sessionId'] = session_id
    return {'jsonrpc': '2.0', 'id': request_id, 'method': 'elicitation/create', 'params': params}


@pytest.mark.asyncio
async def test_elicitation_without_a_handler_is_cancelled() -> None:
    reader, writer = asyncio.StreamReader(), _Writer()
    connection = AcpConnection(reader, writer)
    connection.start()
    _feed(reader, _elicitation(5))
    (frame,) = await _wait_for_frames(writer, 1)
    assert frame == {'jsonrpc': '2.0', 'id': 5, 'result': {'action': 'cancel'}}
    await connection.close()


@pytest.mark.asyncio
async def test_elicitation_handler_answers_the_request() -> None:
    seen: list[dict[str, Any]] = []

    async def handler(params: dict[str, Any]) -> dict[str, Any]:
        seen.append(params)
        return {'action': 'accept', 'content': {'color': '蓝色'}, 'extra': 'dropped'}

    reader, writer = asyncio.StreamReader(), _Writer()
    connection = AcpConnection(reader, writer, elicitation_handler=handler)
    connection.start()
    _feed(reader, _elicitation(5))
    (frame,) = await _wait_for_frames(writer, 1)
    assert seen[0]['message'] == '主色调用哪个？'
    assert frame == {'jsonrpc': '2.0', 'id': 5, 'result': {'action': 'accept', 'content': {'color': '蓝色'}}}
    await connection.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('answer', [{'action': 'maybe', 'content': {'color': '蓝色'}}, None, {'content': {}}, '蓝色'])
async def test_invalid_elicitation_answer_is_cancelled(answer: Any, caplog: pytest.LogCaptureFixture) -> None:
    async def handler(params: dict[str, Any]) -> Any:
        return answer

    reader, writer = asyncio.StreamReader(), _Writer()
    connection = AcpConnection(reader, writer, elicitation_handler=handler)
    connection.start()
    _feed(reader, _elicitation(5))
    (frame,) = await _wait_for_frames(writer, 1)
    assert frame['result'] == {'action': 'cancel'}, 'A malformed Host answer must never be passed to the agent'
    assert 'Invalid ACP elicitation result' in caplog.text
    assert '蓝色' not in caplog.text, 'The user answer must never reach the logs'
    await connection.close()


@pytest.mark.asyncio
async def test_elicitation_without_a_timeout_waits_until_the_handler_answers() -> None:
    answer: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()

    async def handler(params: dict[str, Any]) -> dict[str, Any]:
        return await answer

    reader, writer = asyncio.StreamReader(), _Writer()
    connection = AcpConnection(reader, writer, elicitation_handler=handler, deferred_permission_timeout=None)
    prompt = asyncio.create_task(connection.prompt(session_id='s1', text='run', timeout=5))
    request_id = (await _wait_for_frames(writer, 1))[0]['id']
    _feed(reader, _elicitation(5))
    await asyncio.sleep(0.05)
    assert len(writer.frames()) == 1, 'With no timeout only the Host or the prompt may end the question'
    answer.set_result({'action': 'accept', 'content': {'color': '蓝色'}})
    frames = await _wait_for_frames(writer, 2)
    assert frames[1] == {'jsonrpc': '2.0', 'id': 5, 'result': {'action': 'accept', 'content': {'color': '蓝色'}}}
    _feed(reader, {'jsonrpc': '2.0', 'id': request_id, 'result': {'stopReason': 'end_turn'}})
    await prompt
    await connection.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('session_id', ['s1', None])
@pytest.mark.parametrize(
    ('trigger', 'timeout'),
    [('cancel', None), ('terminal', None), ('cancel', 300), ('terminal', 300), ('timeout', 0.01)],
)
async def test_pending_elicitation_is_cancelled_when_its_prompt_ends_or_times_out(
    trigger: str, timeout: float | None, session_id: str | None
) -> None:
    entered = asyncio.Event()

    async def handler(params: dict[str, Any]) -> dict[str, Any]:
        entered.set()
        await asyncio.Event().wait()
        raise AssertionError('unreachable')

    reader, writer = asyncio.StreamReader(), _Writer()
    connection = AcpConnection(reader, writer, elicitation_handler=handler, deferred_permission_timeout=timeout)
    prompt = asyncio.create_task(connection.prompt(session_id='s1', text='run', timeout=1))
    request_id = (await _wait_for_frames(writer, 1))[0]['id']
    _feed(reader, _elicitation(5, session_id))
    await asyncio.wait_for(entered.wait(), 0.2)
    if trigger == 'cancel':
        await connection.cancel(session_id='s1')
    elif trigger == 'terminal':
        _feed(reader, {'jsonrpc': '2.0', 'id': request_id, 'result': {'stopReason': 'end_turn'}})
        await prompt
    else:
        await asyncio.sleep(0.05)
    frames = await _wait_for_frames(writer, 3 if trigger == 'cancel' else 2)
    assert {'jsonrpc': '2.0', 'id': 5, 'result': {'action': 'cancel'}} in frames
    assert not connection._elicitations
    if not prompt.done():
        _feed(reader, {'jsonrpc': '2.0', 'id': request_id, 'result': {'stopReason': 'cancelled'}})
        await prompt
    await connection.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('session_id', ['s1', None])
async def test_elicitation_after_the_prompt_ended_is_cancelled_without_asking(session_id: str | None) -> None:
    entered = asyncio.Event()

    async def handler(params: dict[str, Any]) -> dict[str, Any]:
        entered.set()
        return {'action': 'accept', 'content': {}}

    reader, writer = asyncio.StreamReader(), _Writer()
    connection = AcpConnection(reader, writer, elicitation_handler=handler, deferred_permission_timeout=None)
    prompt = asyncio.create_task(connection.prompt(session_id='s1', text='run', timeout=1))
    request_id = (await _wait_for_frames(writer, 1))[0]['id']
    _feed(reader, {'jsonrpc': '2.0', 'id': request_id, 'result': {'stopReason': 'end_turn'}})
    await prompt
    _feed(reader, _elicitation(5, session_id))
    frames = await _wait_for_frames(writer, 2)
    assert frames[1] == {'jsonrpc': '2.0', 'id': 5, 'result': {'action': 'cancel'}}
    assert not entered.is_set()
    await connection.close()


@pytest.mark.asyncio
async def test_extension_request_sends_the_underscore_method_and_returns_its_result() -> None:
    reader, writer = asyncio.StreamReader(), _Writer()
    connection = AcpConnection(reader, writer)
    params = {'sessionId': 's1', 'toolCallId': 'call-1', 'decision': 'allow', 'answers': {'q_0': '蓝色'}}
    task = asyncio.create_task(connection.extension_request('_codebuddy.ai/resolveInterruption', params, timeout=1))
    (request,) = await _wait_for_frames(writer, 1)
    assert (request['method'], request['params']) == ('_codebuddy.ai/resolveInterruption', params)
    _feed(reader, {'jsonrpc': '2.0', 'id': request['id'], 'result': {'resolved': True}})
    assert await task == {'resolved': True}
    await connection.close()


@pytest.mark.asyncio
async def test_side_question_sends_earlier_turns_and_returns_the_answer() -> None:
    reader, writer = asyncio.StreamReader(), _Writer()
    connection = AcpConnection(reader, writer)
    connection.protocol_version = 2
    turns = [{'question': '此前的侧问', 'answer': '此前的侧答'}]
    task = asyncio.create_task(
        connection.side_question(session_id='s1', question='为什么采用这个方案？', turns=turns, timeout=1)
    )
    request = (await _wait_for_frames(writer, 1))[0]
    assert request['method'] == '_codebuddy.ai/sideQuestion'
    assert request['params'] == {'sessionId': 's1', 'question': '为什么采用这个方案？', 'turns': turns}
    _feed(reader, {'jsonrpc': '2.0', 'id': request['id'], 'result': {'answer': '因为当前实现需要'}})
    assert await task == '因为当前实现需要'
    await connection.close()


@pytest.mark.asyncio
async def test_cancelling_a_side_question_withdraws_only_that_request() -> None:
    reader, writer = asyncio.StreamReader(), _Writer()
    connection = AcpConnection(reader, writer)
    connection.protocol_version = 2
    task = asyncio.create_task(connection.side_question(session_id='s1', question='为什么？', timeout=1))
    request = (await _wait_for_frames(writer, 1))[0]
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    frames = await _wait_for_frames(writer, 2)
    assert frames[1] == {'jsonrpc': '2.0', 'method': '$/cancel_request', 'params': {'requestId': request['id']}}
    await connection.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('method', ['session/prompt', '_', ''])
async def test_extension_request_refuses_core_protocol_methods(method: str) -> None:
    connection = AcpConnection(asyncio.StreamReader(), _Writer())
    with pytest.raises(ValueError, match='must start with "_"'):
        await connection.extension_request(method, {}, timeout=1)
    await connection.close()


def test_accepted_request_id_prefers_the_turn_id_over_the_message_id() -> None:
    steer = {'messageId': 'steer-1', '_meta': {'codebuddy.ai/requestId': 'turn-1'}}
    assert accepted_request_id(steer) == 'turn-1', 'A steer names its own message but joins the running turn'
    assert accepted_request_id({'messageId': 'turn-2'}) == 'turn-2'
    assert accepted_request_id({'stopReason': 'end_turn'}) is None
    assert accepted_request_id(None) is None


def test_capability_probes_read_the_workbuddy_initialize_result() -> None:
    workbuddy = {
        'protocolVersion': 2,
        'capabilities': {
            'session': {
                'prompt': {'_meta': {'codebuddy.ai/prompt': {'modes': ['steer']}}},
                'fork': {},
                '_meta': {'codebuddy.ai/sideQuestion': {'version': 1, 'maxTurns': 20, 'tools': False}},
            },
        },
    }
    assert supports_session_fork(workbuddy)
    assert supports_prompt_steer(workbuddy)
    assert supports_side_question(workbuddy)
    assert side_question_turn_limit(workbuddy) == 20
    codebuddy = {'protocolVersion': 1, 'agentCapabilities': {'loadSession': True}}
    assert not supports_session_fork(codebuddy)
    assert not supports_prompt_steer(codebuddy)
    assert not supports_side_question(codebuddy)
    assert side_question_turn_limit(codebuddy) == 20


def _v2_permission(session_id: str, request_id: int, tool_call_id: str = 'call-1') -> dict[str, Any]:
    return {
        'jsonrpc': '2.0',
        'id': request_id,
        'method': 'session/request_permission',
        'params': {
            'sessionId': session_id,
            'title': 'Allow Bash?',
            'subject': {'type': 'tool_call', 'toolCall': {'toolCallId': tool_call_id, 'name': 'Bash'}},
            'options': [
                {'optionId': 'allow_always', 'name': 'Allow for this session', 'kind': 'allow_always'},
                {'optionId': 'allow', 'name': 'Allow', 'kind': 'allow_once'},
                {'optionId': 'reject', 'name': 'Reject', 'kind': 'reject_once'},
            ],
            '_meta': {'codebuddy.ai/permissionCategory': 'tool'},
        },
    }


def _accepted(request_id: int, turn_id: str = 'turn-1', message_id: str | None = None) -> dict[str, Any]:
    return {
        'jsonrpc': '2.0',
        'id': request_id,
        'result': {'messageId': message_id or turn_id, '_meta': {'codebuddy.ai/requestId': turn_id}},
    }


async def _v2_connection(**kwargs: Any) -> tuple[AcpConnection, asyncio.StreamReader, _Writer]:
    reader, writer = asyncio.StreamReader(), _Writer()
    connection = AcpConnection(reader, writer, **kwargs)
    init_task = asyncio.create_task(connection.initialize(protocol_version=2))
    request = (await _wait_for_frames(writer, 1))[0]
    _feed(reader, {'jsonrpc': '2.0', 'id': request['id'], 'result': {'protocolVersion': 2}})
    await init_task
    writer.data.clear()
    return connection, reader, writer


async def _accept_prompt(
    connection: AcpConnection,
    reader: asyncio.StreamReader,
    writer: _Writer,
    *,
    session_id: str = 's1',
    turn_id: str = 'turn-1',
) -> dict[str, Any]:
    count = len(writer.frames())
    prompt = asyncio.create_task(connection.prompt(session_id=session_id, text='build it', timeout=1))
    request = (await _wait_for_frames(writer, count + 1))[count]
    _feed(reader, _accepted(request['id'], turn_id))
    return await asyncio.wait_for(prompt, 1)


@pytest.mark.asyncio
async def test_v2_initialize_names_the_client_info_and_records_the_version() -> None:
    reader, writer = asyncio.StreamReader(), _Writer()
    connection = AcpConnection(reader, writer)
    capabilities = {'elicitation': {'form': {}}}
    init_task = asyncio.create_task(connection.initialize(client_capabilities=capabilities, protocol_version=2))
    request = (await _wait_for_frames(writer, 1))[0]
    assert request['params'] == {
        'protocolVersion': 2,
        'info': {'name': 'nexus', 'version': '1'},
        'capabilities': capabilities,
    }, 'wb --acp reads info and capabilities, not the version 1 names'
    result = {'protocolVersion': 2, 'capabilities': {'session': {'fork': {}}}}
    _feed(reader, {'jsonrpc': '2.0', 'id': request['id'], 'result': result})
    assert await init_task == result, 'The initialize result must reach the caller unchanged'
    assert connection.protocol_version == 2
    await connection.close()


@pytest.mark.asyncio
async def test_v2_initialize_rejects_a_peer_that_answers_version_1() -> None:
    reader, writer = asyncio.StreamReader(), _Writer()
    connection = AcpConnection(reader, writer)
    init_task = asyncio.create_task(connection.initialize(protocol_version=2))
    request = (await _wait_for_frames(writer, 1))[0]
    _feed(reader, {'jsonrpc': '2.0', 'id': request['id'], 'result': {'protocolVersion': 1}})
    with pytest.raises(AcpProtocolError, match='unexpected protocol version'):
        await init_task
    assert connection.protocol_version == 1, 'A failed negotiation must not switch the prompt semantics'
    await connection.close()


@pytest.mark.asyncio
async def test_initialize_refuses_an_unknown_protocol_version() -> None:
    connection = AcpConnection(asyncio.StreamReader(), _Writer())
    with pytest.raises(ValueError, match='protocol_version'):
        await connection.initialize(protocol_version=3)
    await connection.close()


@pytest.mark.asyncio
async def test_v2_new_session_sends_only_cwd_and_records_config_options() -> None:
    connection, reader, writer = await _v2_connection()
    pending = asyncio.create_task(connection.new_session(cwd='/workspace/task'))
    request = (await _wait_for_frames(writer, 1))[0]
    assert request['params'] == {'cwd': '/workspace/task'}, (
        'wb answers mcpServers and additionalDirectories with -32602'
    )
    options = [{'configId': 'permission_mode', 'currentValue': 'bypassPermissions', 'options': []}]
    _feed(reader, {'jsonrpc': '2.0', 'id': request['id'], 'result': {'sessionId': 's1', 'configOptions': options}})
    assert await pending == 's1'
    assert connection.config_options['s1'] == options
    await connection.close()


@pytest.mark.asyncio
async def test_v2_new_session_passes_requested_mcp_servers_to_the_peer_instead_of_dropping_them() -> None:
    connection, reader, writer = await _v2_connection()
    servers = [{'name': 'tools', 'type': 'http', 'url': 'x'}]
    pending = asyncio.create_task(connection.new_session(cwd='/workspace', mcp_servers=servers))
    request = (await _wait_for_frames(writer, 1))[0]
    assert request['params'] == {'cwd': '/workspace', 'mcpServers': servers}, 'The peer decides on the servers'
    _feed(reader, {'jsonrpc': '2.0', 'id': request['id'], 'error': {'code': -32602, 'message': 'Invalid params'}})
    with pytest.raises(AcpRemoteError):
        await pending
    await connection.close()


@pytest.mark.asyncio
async def test_v2_permission_after_the_prompt_is_accepted_is_deferred_to_the_handler() -> None:
    seen: list[int] = []

    async def handler(params: dict[str, Any], request_id: int) -> dict[str, Any]:
        seen.append(request_id)
        return {'outcome': {'outcome': 'selected', 'optionId': 'allow'}}

    connection, reader, writer = await _v2_connection(permission_policy='defer', deferred_permission_handler=handler)
    accepted = await _accept_prompt(connection, reader, writer)
    assert accepted['_meta']['codebuddy.ai/requestId'] == 'turn-1'
    _feed(reader, _v2_permission('s1', 7))
    frames = await _wait_for_frames(writer, 2)
    assert seen == [7], 'An accepted version 2 prompt is not the end of the turn; the Host must be asked'
    assert frames[1] == {'jsonrpc': '2.0', 'id': 7, 'result': {'outcome': {'outcome': 'selected', 'optionId': 'allow'}}}
    await connection.close()


@pytest.mark.asyncio
async def test_v2_steer_result_keeps_the_pending_permission_of_the_running_turn() -> None:
    entered = asyncio.Event()

    async def handler(params: dict[str, Any], request_id: int) -> None:
        entered.set()

    connection, reader, writer = await _v2_connection(permission_policy='defer', deferred_permission_handler=handler)
    await _accept_prompt(connection, reader, writer)
    _feed(reader, _v2_permission('s1', 7))
    await asyncio.wait_for(entered.wait(), 1)
    meta = {ACP_PROMPT_META: {'mode': 'steer', 'expectedRequestId': 'turn-1'}}
    steer = asyncio.create_task(connection.prompt(session_id='s1', text='smaller', timeout=1, meta=meta))
    request = (await _wait_for_frames(writer, 2))[1]
    assert request['params']['_meta'] == meta, 'The steer must name the turn it joins'
    _feed(reader, _accepted(request['id'], 'turn-1', message_id='steer-1'))
    assert accepted_request_id(await steer) == 'turn-1'
    assert connection.has_deferred_permission(7), 'A steer result must not withdraw the waiting permission'
    connection.finish_prompt('s1')
    frames = await _wait_for_frames(writer, 3)
    assert frames[2] == {'jsonrpc': '2.0', 'id': 7, 'result': {'outcome': {'outcome': 'cancelled'}}}
    await connection.close()


@pytest.mark.asyncio
async def test_v2_finish_prompt_refuses_permissions_until_the_next_prompt() -> None:
    seen: list[int] = []

    async def handler(params: dict[str, Any], request_id: int) -> dict[str, Any]:
        seen.append(request_id)
        return {'outcome': {'outcome': 'selected', 'optionId': 'allow'}}

    connection, reader, writer = await _v2_connection(permission_policy='defer', deferred_permission_handler=handler)
    await _accept_prompt(connection, reader, writer)
    connection.finish_prompt('s1')
    _feed(reader, _v2_permission('s1', 7))
    frames = await _wait_for_frames(writer, 2)
    assert frames[1]['result'] == {'outcome': {'outcome': 'cancelled'}}, 'A finished turn cannot ask for more'
    await _accept_prompt(connection, reader, writer, turn_id='turn-2')
    _feed(reader, _v2_permission('s1', 8))
    await _wait_for_frames(writer, 4)
    assert seen == [8], 'The next regular prompt reopens the session'
    await connection.close()


@pytest.mark.asyncio
async def test_v2_refused_prompt_keeps_a_cancelled_session_cancelled() -> None:
    connection, reader, writer = await _v2_connection()
    await connection.cancel(session_id='s1')
    prompt = asyncio.create_task(connection.prompt(session_id='s1', text='again', timeout=1))
    request = (await _wait_for_frames(writer, 2))[1]
    _feed(reader, {'jsonrpc': '2.0', 'id': request['id'], 'error': {'code': -32600, 'message': 'busy'}})
    with pytest.raises(AcpRemoteError):
        await prompt
    assert 's1' in connection._cancelled_sessions, 'A refused prompt must not reopen a turn that is being cancelled'
    await connection.close()


@pytest.mark.asyncio
async def test_v2_prompt_accepted_without_a_request_id_is_a_protocol_error() -> None:
    connection, reader, writer = await _v2_connection()
    prompt = asyncio.create_task(connection.prompt(session_id='s1', text='x', timeout=1))
    request = (await _wait_for_frames(writer, 1))[0]
    _feed(reader, {'jsonrpc': '2.0', 'id': request['id'], 'result': {}})
    with pytest.raises(AcpProtocolError, match='request id'):
        await prompt
    await connection.close()


@pytest.mark.asyncio
async def test_v1_prompt_refuses_meta() -> None:
    connection = AcpConnection(asyncio.StreamReader(), _Writer())
    with pytest.raises(ValueError, match='version 2'):
        await connection.prompt(session_id='s1', text='x', timeout=1, meta={ACP_PROMPT_META: {'mode': 'steer'}})
    await connection.close()


@pytest.mark.asyncio
async def test_v2_sessionless_elicitation_is_asked_while_a_turn_is_open() -> None:
    entered = asyncio.Event()

    async def handler(params: dict[str, Any]) -> dict[str, Any]:
        entered.set()
        return {'action': 'decline'}

    connection, reader, writer = await _v2_connection(elicitation_handler=handler)
    await _accept_prompt(connection, reader, writer)
    _feed(reader, _elicitation(5, session_id=None))
    await asyncio.wait_for(entered.wait(), 1)
    await connection.close()


@pytest.mark.asyncio
async def test_command_notification_is_forwarded_as_a_session_update_and_not_answered() -> None:
    reader, writer = asyncio.StreamReader(), _Writer()
    connection = AcpConnection(reader, writer)
    seen: list[dict[str, Any]] = []
    connection.add_update_listener(seen.append)
    connection.start()
    files = {'files': ['/workspace/a/report.docx'], 'focusFile': '/workspace/a/report.docx'}
    _feed(reader, {'jsonrpc': '2.0', 'method': ACP_COMMAND_METHOD, 'params': {'sessionId': 's1', 'action': 'mystery'}})
    _feed(
        reader,
        {
            'jsonrpc': '2.0',
            'method': ACP_COMMAND_METHOD,
            'params': {'sessionId': 's1', 'action': 'presentFiles', 'params': files},
        },
    )
    await _wait_for_queue(connection, 1)
    expected = {
        'sessionId': 's1',
        'update': {'sessionUpdate': ACP_COMMAND_METHOD, 'action': 'presentFiles', 'params': files},
    }
    assert seen == [expected], 'Only the three documented actions are forwarded'
    assert connection.drain_updates(session_id='s1') == [expected], 'A replayed turn must see the deliverables too'
    assert writer.frames() == [], 'A notification has no reply'
    await connection.close()


@pytest.mark.asyncio
async def test_fork_close_and_config_option_requests_track_config_options() -> None:
    connection, reader, writer = await _v2_connection()
    fork = asyncio.create_task(connection.fork_session(session_id='s1', cwd='/workspace'))
    request = (await _wait_for_frames(writer, 1))[0]
    assert (request['method'], request['params']) == ('session/fork', {'sessionId': 's1', 'cwd': '/workspace'})
    mode = [{'configId': 'mode', 'currentValue': 'craft'}]
    _feed(reader, {'jsonrpc': '2.0', 'id': request['id'], 'result': {'sessionId': 's2', 'configOptions': mode}})
    assert await fork == 's2'
    assert connection.config_options['s2'] == mode

    config = asyncio.create_task(connection.set_config_option(session_id='s2', config_id='mode', value='ask'))
    request = (await _wait_for_frames(writer, 2))[1]
    assert request['method'] == 'session/set_config_option'
    assert request['params'] == {'sessionId': 's2', 'configId': 'mode', 'type': 'id', 'value': 'ask'}
    asked = [{'configId': 'mode', 'currentValue': 'ask'}, {'configId': 'permission_mode', 'currentValue': 'default'}]
    _feed(reader, {'jsonrpc': '2.0', 'id': request['id'], 'result': {'configOptions': asked}})
    assert await config == asked
    pushed = [{'configId': 'mode', 'currentValue': 'plan'}]
    _feed(
        reader,
        {
            'jsonrpc': '2.0',
            'method': 'session/update',
            'params': {'sessionId': 's2', 'update': {'sessionUpdate': 'config_option_update', 'configOptions': pushed}},
        },
    )
    await _wait_for_queue(connection, 1)
    assert connection.config_options['s2'] == pushed, 'A pushed config change replaces the options whole'

    close = asyncio.create_task(connection.close_session(session_id='s2'))
    request = (await _wait_for_frames(writer, 3))[2]
    assert (request['method'], request['params']) == ('session/close', {'sessionId': 's2'})
    _feed(reader, {'jsonrpc': '2.0', 'id': request['id'], 'result': {}})
    await close
    assert 's2' not in connection.config_options
    assert 's2' in connection._cancelled_sessions, 'A closed session cannot ask anything any more'
    await connection.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('method', ['fork', 'close', 'config'])
async def test_version_1_connection_refuses_version_2_session_methods(method: str) -> None:
    connection = AcpConnection(asyncio.StreamReader(), _Writer())
    calls = {
        'fork': lambda: connection.fork_session(session_id='s1', cwd='/workspace'),
        'close': lambda: connection.close_session(session_id='s1'),
        'config': lambda: connection.set_config_option(session_id='s1', config_id='mode', value='ask'),
    }
    with pytest.raises(AcpProtocolError, match='version 2'):
        await calls[method]()
    await connection.close()


@pytest.mark.asyncio
async def test_peer_cancel_request_withdraws_a_pending_permission_and_elicitation() -> None:
    permission_entered, elicitation_entered = asyncio.Event(), asyncio.Event()

    async def permission_handler(params: dict[str, Any], request_id: int) -> None:
        permission_entered.set()

    async def elicitation_handler(params: dict[str, Any]) -> dict[str, Any]:
        elicitation_entered.set()
        await asyncio.Event().wait()
        raise AssertionError('unreachable')

    connection, reader, writer = await _v2_connection(
        permission_policy='defer',
        deferred_permission_handler=permission_handler,
        elicitation_handler=elicitation_handler,
    )
    await _accept_prompt(connection, reader, writer)
    _feed(reader, _v2_permission('s1', 7))
    _feed(reader, _elicitation(8))
    await asyncio.wait_for(permission_entered.wait(), 1)
    await asyncio.wait_for(elicitation_entered.wait(), 1)
    for request_id in (7, 8, 99):
        _feed(reader, {'jsonrpc': '2.0', 'method': '$/cancel_request', 'params': {'requestId': request_id}})
    frames = await _wait_for_frames(writer, 3)
    results = {frame['id']: frame['result'] for frame in frames[1:]}
    assert results == {7: {'outcome': {'outcome': 'cancelled'}}, 8: {'action': 'cancel'}}
    assert not connection.has_deferred_permission(7)
    await connection.close()


@pytest.mark.asyncio
async def test_bound_session_handlers_answer_only_their_session() -> None:
    calls: list[tuple[str, int]] = []

    def recorder(name: str) -> Any:
        async def handler(params: dict[str, Any], request_id: int) -> dict[str, Any]:
            calls.append((name, request_id))
            return {'outcome': {'outcome': 'selected', 'optionId': 'allow'}}

        return handler

    connection, reader, writer = await _v2_connection(
        permission_policy='defer', deferred_permission_handler=recorder('source')
    )
    unbind = connection.bind_session_handlers('fork', permission_handler=recorder('fork'))
    connection.bind_session_handlers('silent')
    await _accept_prompt(connection, reader, writer, session_id='src')
    await _accept_prompt(connection, reader, writer, session_id='fork', turn_id='turn-f')
    await _accept_prompt(connection, reader, writer, session_id='silent', turn_id='turn-s')
    _feed(reader, _v2_permission('fork', 7))
    _feed(reader, _v2_permission('src', 8))
    _feed(reader, _v2_permission('silent', 9))
    frames = await _wait_for_frames(writer, 6)
    results = {frame['id']: frame['result'] for frame in frames if frame.get('id') in (7, 8, 9)}
    assert sorted(calls) == [('fork', 7), ('source', 8)], 'A fork must not ask through its source channel'
    assert results[9] == {'outcome': {'outcome': 'cancelled'}}, 'A bound session without a handler cannot ask'
    with pytest.raises(ValueError, match='already has bound handlers'):
        connection.bind_session_handlers('fork')
    unbind()
    unbind()
    _feed(reader, _v2_permission('fork', 10))
    await _wait_for_frames(writer, 7)
    assert calls[-1] == ('source', 10), 'An unbound session falls back to the connection handler'
    await connection.close()


@pytest.mark.asyncio
async def test_v2_permission_carries_the_cached_tool_call_without_meta() -> None:
    received: list[dict[str, Any]] = []

    async def handler(params: dict[str, Any], request_id: int) -> None:
        received.append(params)

    connection, reader, writer = await _v2_connection(permission_policy='defer', deferred_permission_handler=handler)
    await _accept_prompt(connection, reader, writer)
    long_text = 'x' * 5000
    for update in (
        {
            'sessionUpdate': 'tool_call_update',
            'toolCallId': 'call-1',
            'status': 'pending',
            'title': 'Bash',
            'kind': 'execute',
            '_meta': {'codebuddy.ai/toolName': 'AskUserQuestion'},
        },
        {
            'sessionUpdate': 'tool_call_update',
            'toolCallId': 'call-1',
            'rawInput': {'command': 'rm -rf build', 'description': long_text},
        },
    ):
        _feed(reader, {'jsonrpc': '2.0', 'method': 'session/update', 'params': {'sessionId': 's1', 'update': update}})
    _feed(reader, _v2_permission('s1', 7))
    for _ in range(100):
        if received:
            break
        await asyncio.sleep(0)
    tool_call = received[0]['toolCall']
    assert tool_call['toolCallId'] == 'call-1', 'The request names the tool call itself'
    assert tool_call['name'] == 'Bash', 'The request names the tool itself'
    assert tool_call['rawInput']['command'] == 'rm -rf build', 'The Host needs the command it is asked to allow'
    assert tool_call['kind'] == 'execute'
    assert tool_call['title'] == 'Bash'
    assert len(tool_call['rawInput']['description']) == 2000, 'Cached tool input is cut, not kept whole'
    assert '_meta' not in tool_call, 'A tool name in _meta would make the Host take it for AskUserQuestion'
    assert received[0]['subject'] == _v2_permission('s1', 7)['params']['subject'], 'The v2 subject stays as sent'
    connection.finish_prompt('s1')
    assert 's1' not in connection._tool_calls, 'A finished turn drops its cached tool calls'
    await connection.close()


@pytest.mark.asyncio
async def test_v1_tool_updates_are_not_cached_and_permissions_pass_unchanged() -> None:
    received: list[dict[str, Any]] = []

    async def handler(params: dict[str, Any], request_id: int) -> None:
        received.append(params)

    reader, writer = asyncio.StreamReader(), _Writer()
    connection = AcpConnection(reader, writer, permission_policy='defer', deferred_permission_handler=handler)
    prompt = asyncio.create_task(connection.prompt(session_id='s1', text='run', timeout=1))
    request = (await _wait_for_frames(writer, 1))[0]
    update = {'sessionUpdate': 'tool_call', 'toolCallId': 'call-1', 'rawInput': {'command': 'ls'}}
    _feed(reader, {'jsonrpc': '2.0', 'method': 'session/update', 'params': {'sessionId': 's1', 'update': update}})
    frame = _v2_permission('s1', 7)
    _feed(reader, frame)
    for _ in range(100):
        if received:
            break
        await asyncio.sleep(0)
    assert received[0] == frame['params'], 'Version 1 permissions reach the Host exactly as sent'
    assert connection._tool_calls == {}
    _feed(reader, {'jsonrpc': '2.0', 'id': request['id'], 'result': {'stopReason': 'end_turn'}})
    await prompt
    await connection.close()
