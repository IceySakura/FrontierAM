import asyncio

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
import pytest_asyncio

from nexus.extensions.agents.codex.app_server_protocol import AppServerClient, _PendingRequest
from nexus.extensions.agents.codex.app_server_runtime import (
    _register_noninteractive_request_handlers,
    run_codex_app_server,
)
from nexus.extensions.agents.codex.events import CodexEvent, CodexEventStream, reduce_app_server_events


@pytest_asyncio.fixture(autouse=True)
async def close_client_dispatchers(monkeypatch):
    clients = []
    original = AppServerClient.__init__

    def init(client, process):
        original(client, process)
        clients.append(client)

    monkeypatch.setattr(AppServerClient, '__init__', init)
    yield
    for client in clients:
        await client._callbacks.close()


class _FakeClient(AppServerClient):
    def __init__(self) -> None:
        super().__init__(SimpleNamespace())
        self.notifications: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.stderr_lines: list[str] = []
        self.requests: list[tuple[str, dict[str, Any]]] = []
        self.notifications_sent: list[tuple[str, dict[str, Any]]] = []
        self.request_handlers: dict[str, Callable[[dict[str, Any]], Any]] = {}
        self.event_stream = CodexEventStream(source='app_server')
        self.closed = False

    def queue_notification(self, notification: dict[str, Any]) -> None:
        handlers = tuple(self._callbacks.handlers.get(notification['method'], ()))
        if handlers:
            self._callbacks.publish(notification['method'], notification.get('params') or {})
        else:
            self.notifications.put_nowait(notification)
        self.event_stream.events.append(
            CodexEvent(
                seq=len(self.event_stream.events),
                timestamp=notification.get('_received_at'),
                type=str(notification['method']).replace('/', '.'),
                data=dict(notification.get('params') or {}),
            )
        )

    async def request(self, method: str, params: dict[str, Any], *, timeout: float | None = None) -> Any:
        self.requests.append((method, params))
        if method == 'initialize':
            return {'userAgent': 'codex'}
        if method == 'thread/start':
            return {'thread': {'id': 'thread-1'}}
        if method == 'turn/start':
            while not self.notifications.empty():
                notification = self.notifications.get_nowait()
                self._callbacks.publish(notification['method'], notification.get('params') or {})
            return {'turn': {'id': 'turn-1'}}
        if method == 'turn/interrupt':
            return {}
        raise AssertionError(method)

    async def notify(self, method: str, params: dict[str, Any]) -> None:
        self.notifications_sent.append((method, params))

    def register_request_handler(self, method: str, handler: Callable[[dict[str, Any]], Any]) -> None:
        self.request_handlers[method] = handler

    async def close(self) -> None:
        self.closed = True
        await self._callbacks.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('launch_argv', [['/usr/bin/node', '/codex.js'], ['/opt/codex']])
async def test_app_server_run_returns_final_message_and_tool_timeline(
    monkeypatch: pytest.MonkeyPatch, launch_argv: list[str]
) -> None:
    client = _FakeClient()
    for notification in [
        {
            'method': 'item/started',
            'params': {
                'threadId': 'thread-1',
                'turnId': 'turn-1',
                'item': {'id': 'cmd-1', 'type': 'commandExecution'},
            },
            '_received_at': 10.0,
        },
        {
            'method': 'item/completed',
            'params': {
                'threadId': 'thread-1',
                'turnId': 'turn-1',
                'item': {
                    'id': 'cmd-1',
                    'type': 'commandExecution',
                    'command': 'pytest -q',
                    'cwd': '/workspace',
                    'exitCode': 0,
                    'status': 'completed',
                },
            },
            '_received_at': 12.0,
        },
        {
            'method': 'item/completed',
            'params': {
                'threadId': 'thread-1',
                'turnId': 'turn-1',
                'item': {'id': 'message-1', 'type': 'agentMessage', 'text': 'fixed'},
            },
            '_received_at': 13.0,
        },
        {
            'method': 'turn/completed',
            'params': {
                'threadId': 'thread-1',
                'turn': {'id': 'turn-1', 'status': 'completed', 'items': []},
            },
            '_received_at': 14.0,
        },
    ]:
        client.queue_notification(notification)

    start = AsyncMock(return_value=client)
    monkeypatch.setattr(AppServerClient, 'start', start)
    runtime = SimpleNamespace(
        launch_argv=launch_argv,
        working_dir='/workspace',
        env={'PATH': '/usr/bin'},
    )

    result = await run_codex_app_server(
        **vars(runtime),
        task='fix it',
        timeout=10,
        model='gpt-test',
        extra_cli_args=None,
        developer_instructions='be concise',
    )

    assert start.call_args.args[0] == [*launch_argv, 'app-server', '--listen', 'stdio://']
    assert result['success'] is True
    assert result['output'] == 'fixed'
    assert result['exit_code'] == 0
    assert 'termination_reason' not in result
    assert result['json_events_stdout'] == ''
    assert result['execution_time'] >= 0
    event_stream = CodexEventStream.model_validate(result['event_stream'])
    reduction = reduce_app_server_events(event_stream.events, include_tool_args=True)
    assert reduction.timeline_segments == [
        {
            'name': 'shell',
            'start_time': 10.0,
            'end_time': 12.0,
            'tool_args': {
                'command': 'pytest -q',
                'cwd': '/workspace',
                'exit_code': 0,
                'status': 'completed',
            },
        }
    ]
    assert client.notifications_sent == [('initialized', {})]
    assert client.requests[1] == (
        'thread/start',
        {
            'cwd': '/workspace',
            'approvalPolicy': 'never',
            'sandbox': 'danger-full-access',
            'model': 'gpt-test',
            'developerInstructions': 'be concise',
        },
    )
    assert client.closed is True


@pytest.mark.asyncio
async def test_app_server_registers_noninteractive_reverse_request_handlers() -> None:
    client = _FakeClient()
    _register_noninteractive_request_handlers(client)

    assert await client.request_handlers['item/commandExecution/requestApproval']({}) == {'decision': 'accept'}
    assert await client.request_handlers['item/fileChange/requestApproval']({}) == {'decision': 'accept'}
    permissions = {'network': {'enabled': True}, 'fileSystem': {'write': ['/workspace']}}
    assert await client.request_handlers['item/permissions/requestApproval']({'permissions': permissions}) == {
        'permissions': permissions,
        'scope': 'turn',
    }
    assert await client.request_handlers['item/permissions/requestApproval']({
        'permissions': {'network': None, 'fileSystem': {'read': ['/data']}}
    }) == {'permissions': {'fileSystem': {'read': ['/data']}}, 'scope': 'turn'}
    assert await client.request_handlers['item/tool/requestUserInput']({}) == {'answers': {}}
    assert await client.request_handlers['mcpServer/elicitation/request']({}) == {'action': 'decline'}
    assert await client.request_handlers['item/tool/call']({}) == {
        'success': False,
        'contentItems': [{'type': 'inputText', 'text': 'Nexus has no dynamic tool handler'}],
    }


@pytest.mark.asyncio
async def test_app_server_interrupts_turn_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _FakeClient()
    monkeypatch.setattr(AppServerClient, 'start', AsyncMock(return_value=client))
    runtime = SimpleNamespace(
        launch_argv=['/usr/bin/node', '/codex.js'],
        working_dir='/workspace',
        env={'PATH': '/usr/bin'},
    )

    result = await run_codex_app_server(
        **vars(runtime),
        task='wait forever',
        timeout=0.01,
        model=None,
        extra_cli_args=None,
        developer_instructions=None,
    )

    assert result['timed_out'] is True
    assert result['timed_out'] is True
    assert result['failure_kind'] == 'agent'
    assert client.requests[-1] == (
        'turn/interrupt',
        {'threadId': 'thread-1', 'turnId': 'turn-1'},
    )
    assert result['event_stream']['source'] == 'app_server'


@pytest.mark.asyncio
async def test_app_server_client_journals_correlated_responses_and_notifications() -> None:
    client = AppServerClient(SimpleNamespace())
    response = asyncio.get_running_loop().create_future()
    client._pending[7] = _PendingRequest('thread/start', response)

    client._dispatch_message({'id': 7, 'result': {'thread': {'id': 'thread-1'}}})
    client._dispatch_message({
        'method': 'item/completed',
        'params': {
            'threadId': 'thread-1',
            'turnId': 'turn-1',
            'item': {'id': 'message-1', 'type': 'agentMessage', 'text': 'done'},
        },
    })

    assert response.result() == {'thread': {'id': 'thread-1'}}
    assert [event.type for event in client.event_stream.events] == [
        'thread.start.response',
        'item.completed',
    ]
    assert client.event_stream.events[0].data == {
        'requestId': 7,
        'result': {'thread': {'id': 'thread-1'}},
    }
    assert [event.direction for event in client.event_stream.events] == ['inbound', 'inbound']


@pytest.mark.asyncio
async def test_app_server_client_journals_outbound_requests_and_reverse_responses() -> None:
    stdin = SimpleNamespace(write=Mock(), drain=AsyncMock())
    client = AppServerClient(SimpleNamespace(stdin=stdin))

    request_task = asyncio.create_task(client.request('thread/start', {'cwd': '/workspace'}))
    await asyncio.sleep(0)
    client._dispatch_message({'id': 1, 'result': {'thread': {'id': 'thread-1'}}})
    assert await request_task == {'thread': {'id': 'thread-1'}}

    async def reject_tool(_params: dict[str, Any]) -> dict[str, Any]:
        return {'success': False, 'contentItems': []}

    client.register_request_handler('item/tool/call', reject_tool)
    client._dispatch_message({
        'id': 'server-1',
        'method': 'item/tool/call',
        'params': {'threadId': 'thread-1', 'tool': 'lookup'},
    })
    await asyncio.gather(*tuple(client._request_handler_tasks))

    assert [(event.direction, event.type) for event in client.event_stream.events] == [
        ('outbound', 'thread.start'),
        ('inbound', 'thread.start.response'),
        ('inbound', 'item.tool.call'),
        ('outbound', 'item.tool.call.response'),
    ]
    assert client.event_stream.events[0].data == {'requestId': 1, 'cwd': '/workspace'}
    assert client.event_stream.events[-1].data == {
        'requestId': 'server-1',
        'result': {'success': False, 'contentItems': []},
    }
    assert stdin.write.call_count == 2


@pytest.mark.asyncio
async def test_app_server_client_rejects_unknown_and_failed_request_handlers() -> None:
    stdin = SimpleNamespace(write=Mock(), drain=AsyncMock())
    client = AppServerClient(SimpleNamespace(stdin=stdin))

    async def fail(_params: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError('handler failed')

    client.register_request_handler('known/failure', fail)
    client._dispatch_message({'id': 1, 'method': 'unknown/method', 'params': {}})
    client._dispatch_message({'id': 2, 'method': 'known/failure', 'params': {}})
    await asyncio.gather(*tuple(client._request_handler_tasks))

    responses = [event for event in client.event_stream.events if event.direction == 'outbound']
    assert [(event.type, event.data['error']['code']) for event in responses] == [
        ('unknown.method.response', -32601),
        ('known.failure.response', -32603),
    ]


def test_app_server_reduces_collaboration_tool_items() -> None:
    events = [
        CodexEvent(
            seq=0,
            timestamp=10.0,
            direction='inbound',
            type='item.started',
            data={'item': {'id': 'collab-1', 'type': 'collabToolCall', 'tool': 'spawn_agent'}},
        ),
        CodexEvent(
            seq=1,
            timestamp=12.0,
            direction='inbound',
            type='item.completed',
            data={
                'item': {
                    'id': 'collab-1',
                    'type': 'collabToolCall',
                    'tool': 'spawn_agent',
                    'status': 'completed',
                }
            },
        ),
    ]

    assert reduce_app_server_events(events).timeline_segments == [
        {'name': 'spawn_agent', 'start_time': 10.0, 'end_time': 12.0}
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize('finish', ['complete', 'timeout', 'cancel'])
async def test_app_server_active_timeout_pause_resume_and_cleanup(monkeypatch: pytest.MonkeyPatch, finish: str) -> None:
    from nexus.runtime.executors.command import pause_command_timeout, resume_command_timeout

    client = _FakeClient()
    monkeypatch.setattr(AppServerClient, 'start', AsyncMock(return_value=client))
    ready = asyncio.Event()
    original_request = client.request

    async def request(method, params, **kwargs):
        response = await original_request(method, params, **kwargs)
        if method == 'turn/start':
            ready.set()
        return response

    monkeypatch.setattr(client, 'request', request)
    control_id = f'app-server-{finish}'
    run = asyncio.create_task(
        run_codex_app_server(
            launch_argv=['codex'],
            working_dir='/workspace',
            env={},
            task='wait',
            timeout=0.15,
            timeout_control_id=control_id,
            model=None,
            extra_cli_args=None,
            developer_instructions=None,
        )
    )
    try:
        await asyncio.wait_for(ready.wait(), timeout=2)
        assert await pause_command_timeout(control_id)
        await asyncio.sleep(0.25)
        assert not run.done()
        if finish == 'cancel':
            run.cancel()
            with pytest.raises(asyncio.CancelledError):
                await run
        else:
            assert await resume_command_timeout(control_id)
            if finish == 'complete':
                client.queue_notification({
                    'method': 'turn/completed',
                    'params': {'threadId': 'thread-1', 'turn': {'id': 'turn-1', 'status': 'completed'}},
                })
            result = await asyncio.wait_for(run, timeout=2)
            assert result['timed_out'] is (finish == 'timeout')
            if finish == 'timeout':
                assert client.requests[-1][0] == 'turn/interrupt'
        assert not client._callbacks.handlers
        assert not client._callbacks.error_handlers
        assert client.closed
        assert not await pause_command_timeout(control_id)
        assert not await resume_command_timeout(control_id)
    finally:
        run.cancel()
        await asyncio.gather(run, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('status', 'error', 'reason'),
    [
        ('completed', None, 'ENV_DONE'),
        ('failed', {'message': 'bad tool'}, 'CODEX_TURN_FAILED'),
        ('failed', {'message': 'context length exceeded'}, 'TRUNCATION'),
    ],
)
async def test_session_handles_completion_before_start_response(monkeypatch, status, error, reason) -> None:
    client = _FakeClient()
    original_request = client.request

    async def request(method, params, **kwargs):
        if method == 'turn/start':
            client.queue_notification({
                'method': 'turn/completed',
                'params': {'threadId': 'thread-1', 'turn': {'id': 'turn-1', 'status': status, 'error': error}},
            })
            await asyncio.sleep(0)
        return await original_request(method, params, **kwargs)

    client.request = request
    monkeypatch.setattr(AppServerClient, 'start', AsyncMock(return_value=client))
    result = await run_codex_app_server(
        launch_argv=['codex'],
        working_dir='/workspace',
        env={},
        task='test',
        timeout=1,
        model=None,
        extra_cli_args=None,
        developer_instructions=None,
    )
    assert result['exit_code'] == (0 if status == 'completed' else 1)
    assert 'termination_reason' not in result
    assert not client._callbacks.handlers
    assert not client._callbacks.error_handlers


@pytest.mark.asyncio
async def test_app_server_interrupts_notification_turn_while_start_response_pending(monkeypatch) -> None:
    client = _FakeClient()
    original_request = client.request
    cancelled = asyncio.Event()

    async def request(method, params, **kwargs):
        if method == 'turn/start':
            for thread_id, turn_id in [('thread-1', 'turn-1'), ('child-thread', 'child-turn')]:
                client.queue_notification({
                    'method': 'turn/started',
                    'params': {'threadId': thread_id, 'turn': {'id': turn_id}},
                })
            try:
                await asyncio.Future()
            finally:
                cancelled.set()
        return await original_request(method, params, **kwargs)

    client.request = request
    monkeypatch.setattr(AppServerClient, 'start', AsyncMock(return_value=client))
    result = await run_codex_app_server(
        launch_argv=['codex'],
        working_dir='/workspace',
        env={},
        task='test',
        timeout=0.02,
        model=None,
        extra_cli_args=None,
        developer_instructions=None,
    )
    assert result['timed_out'] is True
    assert result['failure_kind'] == 'agent'
    assert [params for method, params in client.requests if method == 'turn/interrupt'] == [
        {'threadId': 'thread-1', 'turnId': 'turn-1'},
    ]
    assert cancelled.is_set()
    assert client.closed


@pytest.mark.asyncio
@pytest.mark.parametrize('completed', [False, True])
async def test_app_server_process_exit_preserves_turn_result_and_partial_output(monkeypatch, completed) -> None:
    from nexus.extensions.agents.codex.app_server_protocol import AppServerProcessError

    client = _FakeClient()
    original_request = client.request

    async def request(method, params, **kwargs):
        response = await original_request(method, params, **kwargs)
        if method == 'turn/start':
            client._disconnect(AppServerProcessError('process exited'))
        return response

    client.request = request
    client.queue_notification({
        'method': 'item/completed',
        'params': {
            'threadId': 'thread-1',
            'turnId': 'turn-1',
            'item': {'id': 'msg', 'type': 'agentMessage', 'text': 'partial output'},
        },
    })
    if completed:
        client.queue_notification({
            'method': 'turn/completed',
            'params': {'threadId': 'thread-1', 'turn': {'id': 'turn-1', 'status': 'completed'}},
        })

    client.stderr_lines.append('process stderr')
    monkeypatch.setattr(AppServerClient, 'start', AsyncMock(return_value=client))
    result = await run_codex_app_server(
        launch_argv=['codex'],
        working_dir='/workspace',
        env={},
        task='test',
        timeout=1,
        model=None,
        extra_cli_args=None,
        developer_instructions=None,
    )
    assert result['exit_code'] == (0 if completed else -1)
    if not completed:
        assert result['failure_kind'] == 'infrastructure'
        assert result['failure_stage'] == 'runtime_protocol'
    assert result['output'] == 'partial output'
    assert 'process stderr' in result['stderr']
    assert client.closed


@pytest.mark.asyncio
async def test_session_reuses_thread_and_filters_unrelated_turns(monkeypatch) -> None:
    from nexus.extensions.agents.codex.app_server_turn_session import AppServerTurnSession

    client = AppServerClient(SimpleNamespace())
    requests = []

    async def request(method, params, **kwargs):
        requests.append(method)
        if method == 'thread/start':
            return {'thread': {'id': 'thread-1'}}
        if method == 'turn/start':
            turn_id = f'turn-{requests.count(method)}'
            return {'turn': {'id': turn_id}}
        return {}

    monkeypatch.setattr(client, 'request', request)
    monkeypatch.setattr(client, 'notify', AsyncMock())
    async with AppServerTurnSession(client) as session:
        await session.start_thread(working_dir='/workspace', model=None, developer_instructions=None)
        for index in (1, 2):
            task = asyncio.create_task(session.run('test'))
            await asyncio.sleep(0)
            for thread_id, turn_id in [('child-thread', 'child-turn'), ('thread-1', 'old-turn')]:
                client._dispatch_message({
                    'method': 'turn/completed',
                    'params': {'threadId': thread_id, 'turn': {'id': turn_id, 'status': 'completed'}},
                })
            assert not task.done()
            turn = {'id': f'turn-{index}', 'status': 'completed'}
            client._dispatch_message({'method': 'turn/completed', 'params': {'threadId': 'thread-1', 'turn': turn}})
            assert (await task).success
            assert session.turn_id == turn['id']
            await session.interrupt()
    assert requests == ['initialize', 'thread/start', 'turn/start', 'turn/start']
    assert not client._callbacks.handlers
    assert not client._callbacks.error_handlers


@pytest.mark.asyncio
@pytest.mark.parametrize('phase', ['initialize', 'turn/start', 'waiting'])
async def test_session_transport_failure_unblocks_and_unregisters_handlers(monkeypatch, phase) -> None:
    from nexus.extensions.agents.codex.app_server_protocol import AppServerProcessError
    from nexus.extensions.agents.codex.app_server_turn_session import AppServerTurnSession

    client = AppServerClient(SimpleNamespace())

    async def send(message, **kwargs):
        method = message['method']
        if method == 'initialized':
            return
        if method == phase:
            client._disconnect(AppServerProcessError('disconnected'))
            return
        result = {'thread': {'id': 'thread-1'}} if method == 'thread/start' else {'turn': {'id': 'turn-1'}}
        client._dispatch_message({'id': message['id'], 'result': result})
        if phase == 'waiting' and method == 'turn/start':
            asyncio.get_running_loop().call_soon(client._disconnect, AppServerProcessError('disconnected'))

    monkeypatch.setattr(client, '_send', send)

    async def run():
        async with AppServerTurnSession(client) as session:
            await session.start_thread(working_dir='/workspace', model=None, developer_instructions=None)
            await asyncio.wait_for(session.run('test'), timeout=1)

    with pytest.raises(AppServerProcessError, match='disconnected'):
        await run()
    assert not client._callbacks.handlers
    assert not client._callbacks.error_handlers
    assert not client._pending
    with pytest.raises(AppServerProcessError, match='disconnected'):
        await client.request('thread/start', {})


@pytest.mark.asyncio
@pytest.mark.parametrize('completed', [True, False])
async def test_app_server_session_over_real_stdio(monkeypatch, tmp_path, completed) -> None:
    import os
    import sys

    script = """
import json, sys

def send(message):
    print(json.dumps(message), flush=True)

for line in sys.stdin:
    request = json.loads(line)
    method = request['method']
    if method == 'initialized':
        continue
    if method == 'initialize':
        result = {}
    elif method == 'thread/start':
        result = {'thread': {'id': 'thread-1'}}
    elif method == 'turn/start':
        turn = {'id': 'turn-1', 'status': 'completed'}
        send({'method': 'turn/started', 'params': {'threadId': 'thread-1', 'turn': turn}})
        send({'method': 'item/completed', 'params': {
            'threadId': 'thread-1', 'turnId': 'turn-1',
            'item': {'id': 'message', 'type': 'agentMessage', 'text': 'stdio proof'}}})
        if sys.argv[1] == 'True':
            # Exercise completion arriving before the turn/start response.
            send({'method': 'turn/completed', 'params': {'threadId': 'thread-1', 'turn': turn}})
        send({'id': request['id'], 'result': {'turn': turn}})
        print('server stderr', file=sys.stderr, flush=True)
        break
    send({'id': request['id'], 'result': result})
"""
    client = await AppServerClient.start(
        [sys.executable, '-u', '-c', script, str(completed)], cwd=str(tmp_path), env=dict(os.environ)
    )
    monkeypatch.setattr(AppServerClient, 'start', AsyncMock(return_value=client))
    result = await run_codex_app_server(
        launch_argv=['codex'],
        working_dir=str(tmp_path),
        env={},
        task='test',
        timeout=3,
        model=None,
        extra_cli_args=None,
        developer_instructions=None,
    )
    assert result['exit_code'] == (0 if completed else -1)
    if not completed:
        assert result['failure_kind'] == 'infrastructure'
        assert result['failure_stage'] == 'runtime_protocol'
    assert result['output'] == 'stdio proof'
    assert 'server stderr' in result['stderr']
    assert client.process.returncode == 0
    assert all(task.done() for task in client._reader_tasks)
    assert not client._callbacks.handlers
    assert not client._callbacks.error_handlers


@pytest.mark.asyncio
async def test_session_can_interrupt_after_local_wait_is_cancelled() -> None:
    from nexus.extensions.agents.codex.app_server_turn_session import AppServerTurnSession

    client = _FakeClient()
    async with AppServerTurnSession(client) as session:
        await session.start_thread(working_dir='/workspace', model=None, developer_instructions=None)
        waiting = asyncio.create_task(session.run('test'))
        await asyncio.sleep(0)
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
        assert session._completion is None
        with pytest.raises(RuntimeError, match='already running'):
            await session.run('another turn')
        await session.interrupt()
        assert client.requests[-1] == ('turn/interrupt', {'threadId': 'thread-1', 'turnId': 'turn-1'})
        client.queue_notification({
            'method': 'turn/completed',
            'params': {'threadId': 'thread-1', 'turn': {'id': 'turn-1', 'status': 'interrupted'}},
        })
        await client._callbacks.queue.join()
        requests_before = len(client.requests)
        await session.interrupt()
        assert len(client.requests) == requests_before


@pytest.mark.asyncio
async def test_app_server_launch_failure_returns_common_runtime_contract(monkeypatch) -> None:
    monkeypatch.setattr(AppServerClient, 'start', AsyncMock(side_effect=FileNotFoundError('missing codex')))
    result = await run_codex_app_server(
        launch_argv=['codex'],
        working_dir='/workspace',
        env={},
        task='test',
        timeout=1,
        model=None,
        extra_cli_args=None,
        developer_instructions=None,
    )
    assert result['success'] is False
    assert result['exit_code'] == -1
    assert result['timed_out'] is False
    assert result['failure_kind'] == 'infrastructure'
    assert result['failure_stage'] == 'runtime_executor'
    assert result['json_events_stdout'] == ''
    assert result['event_stream']['events'] == []
    assert result['execution_time'] >= 0
    assert 'termination_reason' not in result


@pytest.mark.asyncio
@pytest.mark.parametrize('goal_first', [True, False])
@pytest.mark.parametrize(
    ('goal_status', 'turn_status', 'success'),
    [
        ('complete', 'completed', True),
        ('complete', 'failed', False),
        ('paused', 'completed', False),
        ('blocked', 'completed', False),
        ('budgetLimited', 'interrupted', False),
        ('usageLimited', 'failed', False),
    ],
)
async def test_goal_waits_for_terminal_goal_and_drained_turn(
    monkeypatch, goal_first, goal_status, turn_status, success
):
    from nexus.extensions.agents.codex.events import classify_app_server_termination

    client = _FakeClient()
    original = client.request

    def notify(method, **params):
        client.queue_notification({'method': method, 'params': {'threadId': 'thread-1', **params}})

    async def request(method, params, **kwargs):
        if method != 'thread/goal/set':
            return await original(method, params, **kwargs)
        client.requests.append((method, params))
        notify('thread/goal/updated', goal={'status': 'active'})
        notify('turn/started', turn={'id': 'first'})
        notify('item/completed', turnId='first', item={'id': 'old', 'type': 'agentMessage', 'text': 'old answer'})
        notify('turn/completed', turn={'id': 'first', 'status': 'completed'})
        await asyncio.sleep(0)
        notify('turn/started', turn={'id': 'second'})
        # Child-thread goal notifications must not decide the root outcome.
        client.queue_notification({
            'method': 'thread/goal/updated',
            'params': {'threadId': 'child', 'goal': {'status': 'blocked'}},
        })
        if goal_first:
            notify('thread/goal/updated', goal={'status': goal_status})
        notify('turn/completed', turn={'id': 'second', 'status': turn_status})
        if not goal_first:
            notify('thread/goal/updated', goal={'status': goal_status})
        return {'goal': {'status': 'active'}}

    client.request = request
    monkeypatch.setattr(AppServerClient, 'start', AsyncMock(return_value=client))
    result = await run_codex_app_server(
        launch_argv=['codex'],
        working_dir='/workspace',
        env={},
        task='goal',
        timeout=1,
        model=None,
        extra_cli_args=None,
        developer_instructions=None,
        goal_mode=True,
        goal_token_budget=100,
    )
    assert result['success'] is success
    assert result['output'] == ''  # Do not reuse the previous turn's answer.
    assert 'termination_reason' not in result
    assert not any(method == 'turn/start' for method, _ in client.requests)
    assert client.requests[-1] == (
        'thread/goal/set',
        {'threadId': 'thread-1', 'objective': 'goal', 'status': 'active', 'tokenBudget': 100},
    )
    events = CodexEventStream.model_validate(result['event_stream']).events
    reason = classify_app_server_termination(events, result['exit_code'], result['timed_out'])
    expected = {
        'complete': 'ENV_DONE' if success else 'CODEX_TURN_FAILED',
        'paused': 'CODEX_GOAL_PAUSED',
        'blocked': 'CODEX_GOAL_BLOCKED',
        'budgetLimited': 'CODEX_GOAL_BUDGET_LIMITED',
        'usageLimited': 'CODEX_GOAL_USAGE_LIMITED',
    }
    assert reason == expected[goal_status]
    assert client.closed
    assert not client._callbacks.handlers


@pytest.mark.asyncio
async def test_goal_timeout_pauses_native_continuation_before_interrupt(monkeypatch):
    client = _FakeClient()
    original = client.request

    async def request(method, params, **kwargs):
        if method == 'thread/goal/set':
            client.requests.append((method, params))
            if params['status'] == 'active':
                client.queue_notification({
                    'method': 'turn/started',
                    'params': {'threadId': 'thread-1', 'turn': {'id': 'goal-turn'}},
                })
            return {'goal': {'status': params['status']}}
        return await original(method, params, **kwargs)

    client.request = request
    monkeypatch.setattr(AppServerClient, 'start', AsyncMock(return_value=client))
    result = await run_codex_app_server(
        launch_argv=['codex'],
        working_dir='/workspace',
        env={},
        task='goal',
        timeout=0.01,
        model=None,
        extra_cli_args=None,
        developer_instructions=None,
        goal_mode=True,
    )
    assert result['timed_out']
    assert client.requests[-2:] == [
        ('thread/goal/set', {'threadId': 'thread-1', 'status': 'paused'}),
        ('turn/interrupt', {'threadId': 'thread-1', 'turnId': 'goal-turn'}),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize('ending', ['cleared', 'disconnect', 'context_limit'])
async def test_goal_failure_does_not_report_success(monkeypatch, ending):
    from nexus.extensions.agents.codex.app_server_protocol import AppServerProcessError
    from nexus.extensions.agents.codex.events import classify_app_server_termination

    client = _FakeClient()
    original = client.request

    async def request(method, params, **kwargs):
        if method != 'thread/goal/set':
            return await original(method, params, **kwargs)
        client.queue_notification({
            'method': 'turn/started',
            'params': {'threadId': 'thread-1', 'turn': {'id': 'goal-turn'}},
        })
        if ending == 'cleared':
            client.queue_notification({'method': 'thread/goal/cleared', 'params': {'threadId': 'thread-1'}})
        elif ending == 'disconnect':
            client.queue_notification({
                'method': 'thread/goal/updated',
                'params': {'threadId': 'thread-1', 'goal': {'status': 'complete'}},
            })
            client._disconnect(AppServerProcessError('disconnected before turn completed'))
        else:
            client.queue_notification({
                'method': 'turn/completed',
                'params': {
                    'threadId': 'thread-1',
                    'turn': {
                        'id': 'goal-turn',
                        'status': 'failed',
                        'error': {'codexErrorInfo': 'contextWindowExceeded'},
                    },
                },
            })
        return {'goal': {'status': 'active'}}

    client.request = request
    monkeypatch.setattr(AppServerClient, 'start', AsyncMock(return_value=client))
    result = await run_codex_app_server(
        launch_argv=['codex'],
        working_dir='/workspace',
        env={},
        task='goal',
        timeout=1,
        model=None,
        extra_cli_args=None,
        developer_instructions=None,
        goal_mode=True,
    )
    assert not result['success']
    events = CodexEventStream.model_validate(result['event_stream']).events
    reason = classify_app_server_termination(events, result['exit_code'], result['timed_out'])
    assert (
        reason == {'cleared': 'CODEX_GOAL_CLEARED', 'disconnect': 'CODEX_ERROR', 'context_limit': 'TRUNCATION'}[ending]
    )
    assert not client._callbacks.handlers
    assert not client._callbacks.error_handlers


@pytest.mark.asyncio
async def test_notification_subscribers_run_in_order_and_unregister_independently():
    client = _FakeClient()
    seen = []

    async def common(_):
        await asyncio.sleep(0)
        seen.append('common')

    async def specific(_):
        seen.append('specific')

    unregister_common = client.register_notification_handler('turn/completed', common)
    unregister_specific = client.register_notification_handler('turn/completed', specific)
    client._dispatch_message({'method': 'turn/completed', 'params': {}})
    await client._callbacks.queue.join()
    assert seen == ['common', 'specific']
    unregister_specific()
    client._dispatch_message({'method': 'turn/completed', 'params': {}})
    await client._callbacks.queue.join()
    assert seen == ['common', 'specific', 'common']
    unregister_common()
    assert not client._callbacks.handlers


@pytest.mark.asyncio
@pytest.mark.parametrize('goal_mode', [False, True])
@pytest.mark.parametrize('outcome', ['completed', 'cancelled', 'disconnected'])
async def test_run_callbacks_are_scoped_to_execution(goal_mode, outcome):
    from nexus.extensions.agents.codex.app_server_goal_session import AppServerGoalSession
    from nexus.extensions.agents.codex.app_server_turn_session import AppServerTurnSession

    client = _FakeClient()
    original = client.request

    async def request(method, params, **kwargs):
        if method == 'thread/goal/set':
            return {'goal': {'status': 'active'}}
        return await original(method, params, **kwargs)

    client.request = request
    session = AppServerGoalSession(client) if goal_mode else AppServerTurnSession(client)
    async with session:
        await session.start_thread(working_dir='/workspace', model=None, developer_instructions=None)
        common = {method: list(handlers) for method, handlers in client._callbacks.handlers.items()}
        for _ in range(2):
            waiting = asyncio.create_task(session.run('test'))
            await asyncio.sleep(0)
            client._dispatch_message({
                'method': 'turn/started',
                'params': {'threadId': 'thread-1', 'turn': {'id': 'turn-1'}},
            })
            if outcome == 'cancelled':
                waiting.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await waiting
            elif outcome == 'disconnected':
                client._disconnect(RuntimeError('connection lost'))
                with pytest.raises(RuntimeError, match='connection lost'):
                    await waiting
            elif goal_mode:
                client._dispatch_message({
                    'method': 'thread/goal/updated',
                    'params': {'threadId': 'thread-1', 'goal': {'status': 'complete'}},
                })
                assert not waiting.done()
            client._dispatch_message({
                'method': 'turn/completed',
                'params': {'threadId': 'thread-1', 'turn': {'id': 'turn-1', 'status': 'completed'}},
            })
            if outcome == 'completed':
                assert (await waiting).success
            await client._callbacks.queue.join()
            assert client._callbacks.handlers == common
            assert session._turn_active is False
            if outcome == 'disconnected':
                break
    assert not client._callbacks.handlers


@pytest.mark.asyncio
async def test_async_notification_can_await_rpc_over_real_stdio(tmp_path):
    import os
    import sys

    script = """
import json, sys
for line in sys.stdin:
    request = json.loads(line)
    if request['method'] == 'trigger':
        for index in range(2):
            print(json.dumps({'method': 'progress', 'params': {'index': index}}), flush=True)
    print(json.dumps({'id': request['id'], 'result': request['params']}), flush=True)
"""
    client = await AppServerClient.start([sys.executable, '-u', '-c', script], cwd=str(tmp_path), env=dict(os.environ))
    seen = []

    async def common(params):
        response = await client.request('echo', params, timeout=1)
        seen.append(('common', response['index']))

    async def specific(params):
        await asyncio.sleep(0)
        seen.append(('specific', params['index']))

    client.register_notification_handler('progress', common)
    client.register_notification_handler('progress', specific)
    try:
        await client.request('trigger', {}, timeout=1)
        await asyncio.wait_for(client._callbacks.queue.join(), timeout=2)
        assert seen == [('common', 0), ('specific', 0), ('common', 1), ('specific', 1)]
    finally:
        await client.close()
    assert client._callbacks.task.done()
    assert all(task.done() for task in client._reader_tasks)
    assert not client._pending


@pytest.mark.asyncio
async def test_close_cancels_async_notification_waiting_for_rpc(tmp_path):
    import os
    import sys

    script = """
import json, sys
for line in sys.stdin:
    request = json.loads(line)
    if request['method'] == 'trigger':
        print(json.dumps({'method': 'progress', 'params': {}}), flush=True)
        print(json.dumps({'id': request['id'], 'result': {}}), flush=True)
"""
    client = await AppServerClient.start([sys.executable, '-u', '-c', script], cwd=str(tmp_path), env=dict(os.environ))
    entered = asyncio.Event()
    exited = asyncio.Event()

    async def handler(_):
        entered.set()
        try:
            await client.request('never-responds', {})
        finally:
            exited.set()

    client.register_notification_handler('progress', handler)
    try:
        await client.request('trigger', {}, timeout=1)
        await asyncio.wait_for(entered.wait(), timeout=1)
    finally:
        await asyncio.wait_for(client.close(), timeout=6)
    assert exited.is_set()
    assert client._callbacks.task.done()
    assert not client._pending
    await asyncio.wait_for(client._callbacks.queue.join(), timeout=1)


@pytest.mark.asyncio
async def test_async_notification_failure_cannot_be_overwritten_by_completion():
    from nexus.extensions.agents.codex.app_server_protocol import AppServerProcessError
    from nexus.extensions.agents.codex.app_server_turn_session import AppServerTurnSession

    client = _FakeClient()

    async def broken(_):
        await asyncio.sleep(0)
        raise ValueError('callback failed')

    client.register_notification_handler('broken', broken)
    async with AppServerTurnSession(client) as session:
        await session.start_thread(working_dir='/workspace', model=None, developer_instructions=None)
        waiting = asyncio.create_task(session.run('test'))
        await asyncio.sleep(0)
        client._dispatch_message({'method': 'broken', 'params': {}})
        client._dispatch_message({
            'method': 'turn/completed',
            'params': {'threadId': 'thread-1', 'turn': {'id': 'turn-1', 'status': 'completed'}},
        })
        with pytest.raises(AppServerProcessError, match='callback failed'):
            await asyncio.wait_for(waiting, timeout=1)
        await client._callbacks.queue.join()


@pytest.mark.asyncio
async def test_queued_notifications_do_not_call_unregistered_handlers():
    client = _FakeClient()
    entered = asyncio.Event()
    release = asyncio.Event()
    specific = AsyncMock()

    async def common(_):
        entered.set()
        await release.wait()

    client.register_notification_handler('progress', common)
    unregister = client.register_notification_handler('progress', specific)
    client._dispatch_message({'method': 'progress', 'params': {}})
    await asyncio.wait_for(entered.wait(), timeout=1)
    unregister()
    release.set()
    await asyncio.wait_for(client._callbacks.queue.join(), timeout=1)
    specific.assert_not_awaited()


@pytest.mark.asyncio
async def test_disconnect_preserves_first_error_and_only_replays_to_new_subscriber():
    client = _FakeClient()
    first = AsyncMock()
    late = AsyncMock()
    removed = AsyncMock()
    client.register_error_handler(first)
    error = RuntimeError('first disconnect')
    pending = asyncio.get_running_loop().create_future()
    client._pending[1] = _PendingRequest('test', pending)
    client._disconnect(error)
    assert pending.exception() is error  # RPC failure does not wait for callback dispatch.
    client.register_error_handler(late)
    unregister = client.register_error_handler(removed)
    unregister()
    client._disconnect(RuntimeError('later process exit'))
    await client._callbacks.queue.join()
    first.assert_awaited_once_with(error)
    late.assert_awaited_once_with(error)
    removed.assert_not_awaited()
    assert client._callbacks.error is error
    await client.close()
    first.assert_awaited_once_with(error)
    late.assert_awaited_once_with(error)


@pytest.mark.asyncio
async def test_callback_dispatcher_survives_idle_and_closes_with_no_queued_work():
    client = _FakeClient()
    task = client._callbacks.task
    handler = AsyncMock()
    client.register_notification_handler('progress', handler)
    for index in range(2):
        client._dispatch_message({'method': 'progress', 'params': {'index': index}})
        await client._callbacks.queue.join()
        await asyncio.sleep(0)
        assert client._callbacks.task is task
        assert not task.done()
    assert handler.await_count == 2
    await client.close()
    assert task.done()
