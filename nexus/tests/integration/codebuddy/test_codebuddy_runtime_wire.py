"""Exercise the public runtime through real stdio, HTTP, and Socket.IO sockets.

The ACP peer is deterministic Python, not CodeBuddy. Image/real-CLI E2E remains a
separate gate; these tests isolate the runtime protocol and lifecycle contracts.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.abc
import json
import sys

from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import socketio

from aiohttp import web

from nexus.extensions.agents.codebuddy import session_runtime
from nexus.extensions.agents.installed_common.host_rpc import (
    ACP_PERMISSION_REQUEST_EVENT,
    ACP_PERMISSION_RESOLVE_EVENT,
    HOST_TOOL_REQUEST_EVENT,
    HOST_TOOL_RESPONSE_EVENT,
)
from nexus.runtime.context import SIOBridge


pytestmark = [pytest.mark.integration, pytest.mark.asyncio]
_FIXTURE = Path(__file__).with_name('runtime_fixture.py')
_FACTORY = 'tests.integration.codebuddy.runtime_fixture::session_extension'
_PUBLIC_FACTORY = 'nexus.extensions.agents.codebuddy.host_relay::host_relay_extension'
_ECHO_TOOL = {
    'name': 'echo',
    'description': 'Echo text through the Host.',
    'inputSchema': {'type': 'object', 'properties': {'text': {'type': 'string'}}, 'required': ['text']},
}


class _RejectProjects(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> None:
        if fullname == 'nexus.extensions.projects' or fullname.startswith('nexus.extensions.projects.'):
            raise AssertionError(f'Public runtime imported project code: {fullname}')


async def test_public_runtime_imports_in_clean_process_without_projects() -> None:
    program = """
import importlib
import importlib.abc
import json
import sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'nexus.extensions.projects' or fullname.startswith('nexus.extensions.projects.'):
            raise AssertionError(fullname)
sys.meta_path.insert(0, Guard())
modules = [
    'nexus.runtime.server.services.rsio_llm_bridge',
    'nexus.runtime.server.services.openai_proxy',
    'nexus.extensions.agents.installed_common.acp',
    'nexus.extensions.agents.codebuddy.runtime_setup',
    'nexus.extensions.agents.codebuddy.session_runtime',
    'nexus.extensions.agents.installed_common.host_rpc',
    'nexus.extensions.agents.installed_common.host_tool_mcp',
    'nexus.extensions.agents.codebuddy.host_relay',
]
print(json.dumps({name: importlib.import_module(name).__file__ for name in modules}))
assert not any(name.startswith('nexus.extensions.projects.') for name in sys.modules)
"""
    process = await asyncio.create_subprocess_exec(
        sys.executable, '-c', program, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=20)
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
    assert process.returncode == 0, stderr.decode()
    imports = json.loads(stdout)
    root = Path(__file__).resolve().parents[3]
    assert len(imports) == 8
    assert all(Path(path).is_relative_to(root) for path in imports.values())


@pytest.fixture
async def wire_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> AsyncIterator[Any]:
    server = socketio.AsyncServer(async_mode='aiohttp')
    application = web.Application()
    server.attach(application)
    runner = web.AppRunner(application)
    await runner.setup()
    site = web.TCPSite(runner, '127.0.0.1', 0)
    await site.start()
    assert isinstance(site._server, asyncio.Server)
    port = site._server.sockets[0].getsockname()[1]
    client = socketio.AsyncClient(reconnection=False)
    entered: dict[str, asyncio.Event] = {}
    permissions: dict[str, asyncio.Event] = {}
    release = asyncio.Event()
    requests: list[dict[str, Any]] = []
    in_flight: set[str] = set()
    overlap: list[set[str]] = []

    @server.on('llm_request')
    async def llm_request(sid: str, data: dict[str, Any]) -> None:
        channel = data['rsio']['channel_id']
        requests.append(data)
        in_flight.add(channel)
        overlap.append(set(in_flight))
        entered.setdefault(channel, asyncio.Event()).set()
        try:
            await asyncio.wait_for(release.wait(), timeout=10)
            # NOTE(wallyshi): main Host responses omit channel; request-id correlation must suffice.
            await server.emit(
                'llm_response',
                {
                    'request_id': data['request_id'],
                    'success': True,
                    'response': {'choices': [{'message': {'content': f'{channel}|{data["messages"][0]["content"]}'}}]},
                },
                to=sid,
            )
        finally:
            in_flight.discard(channel)

    @server.on('fixture:permission')
    async def permission_request(sid: str, data: dict[str, Any]) -> None:
        permissions.setdefault(data['session_id'], asyncio.Event()).set()

    host_decisions: dict[str, str] = {}
    host_permission_requests: list[dict[str, Any]] = []
    host_permission_arrived = asyncio.Condition()
    host_tool_requests: list[dict[str, Any]] = []
    host_sids: dict[str, str] = {}

    async def resolve_permission(request: dict[str, Any], kind: str) -> None:
        option_id = next(option['optionId'] for option in request['params']['options'] if option['kind'] == kind)
        await server.emit(
            ACP_PERMISSION_RESOLVE_EVENT,
            {
                'channel_id': request['channel_id'],
                'acp_request_id': request['acp_request_id'],
                'permission_token': request['permission_token'],
                'option_id': option_id,
            },
            to=host_sids[request['channel_id']],
        )

    @server.on(ACP_PERMISSION_REQUEST_EVENT)
    async def host_permission(sid: str, data: dict[str, Any]) -> None:
        host_sids[data['channel_id']] = sid
        async with host_permission_arrived:
            host_permission_requests.append(data)
            host_permission_arrived.notify_all()
        decision = host_decisions[data['channel_id']]
        if decision != 'hold':
            await resolve_permission(data, 'allow_once' if decision == 'allow' else 'reject_once')

    async def wait_for_permission_requests(count: int) -> list[dict[str, Any]]:
        async with host_permission_arrived:
            await asyncio.wait_for(
                host_permission_arrived.wait_for(lambda: len(host_permission_requests) >= count), timeout=15
            )
        return list(host_permission_requests)

    @server.on(HOST_TOOL_REQUEST_EVENT)
    async def host_tool(sid: str, data: dict[str, Any]) -> None:
        host_tool_requests.append(data)
        await server.emit(
            HOST_TOOL_RESPONSE_EVENT,
            {
                'request_id': data['request_id'],
                'channel_id': data['channel_id'],
                'success': True,
                'result': f'host:{data["channel_id"]}:{data["arguments"]["text"]}',
            },
            to=sid,
        )

    await client.connect(f'http://127.0.0.1:{port}', transports=['websocket'])
    bridge = SIOBridge(client)

    async def install(**kwargs: Any) -> Path:
        return _FIXTURE

    monkeypatch.setattr(session_runtime, 'install_codebuddy_via_npm', install)
    monkeypatch.setattr(session_runtime, 'find_nexus_runtime_binary', lambda name: sys.executable)
    monkeypatch.setattr(session_runtime, '_SESSION_STATE_ROOT', tmp_path / 'homes')
    monkeypatch.setattr('nexus.runtime.context.require_runtime_control', lambda: SimpleNamespace(sio=bridge))
    project_guard = _RejectProjects()
    sys.meta_path.insert(0, project_guard)
    session_ids: set[str] = set()

    async def turn(
        session_id: str, text: str, *, decision: str | None = None, extension_args: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        session_ids.add(session_id)
        extension_args = dict(extension_args or {})
        if decision is not None:
            extension_args = {
                'extension_factory': _FACTORY,
                'extension_config': {'evidence': str(tmp_path / session_id / 'evidence'), 'decision': decision},
                'permission_policy': 'defer',
                'skip_permissions': False,
            }
        return await asyncio.wait_for(
            session_runtime.run_codebuddy_session_turn(
                session_id=session_id,
                prompt=text,
                working_dir=str(tmp_path / session_id),
                timeout=10,
                **extension_args,
            ),
            timeout=20,
        )

    try:
        yield SimpleNamespace(
            turn=turn,
            entered=entered,
            permissions=permissions,
            release=release,
            requests=requests,
            overlap=overlap,
            tmp_path=tmp_path,
            client=client,
            host_decisions=host_decisions,
            host_permission_requests=host_permission_requests,
            host_tool_requests=host_tool_requests,
            resolve_permission=resolve_permission,
            wait_for_permission_requests=wait_for_permission_requests,
        )
    finally:
        release.set()
        try:
            for session_id in session_ids:
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(
                        session_runtime.interrupt_codebuddy_session(session_id=session_id), timeout=5
                    )
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(session_runtime.close_codebuddy_session(session_id=session_id), timeout=10)
        finally:
            sys.meta_path.remove(project_guard)
            await client.disconnect()
            await runner.cleanup()


async def test_two_session_stdio_http_socketio_round_trips_overlap_and_preserve_context(wire_runtime: Any) -> None:
    runtime = wire_runtime
    runtime.entered['wire-a'] = asyncio.Event()
    runtime.entered['wire-b'] = asyncio.Event()
    first = asyncio.create_task(runtime.turn('wire-a', 'llm:alpha'))
    second = asyncio.create_task(runtime.turn('wire-b', 'llm:beta'))
    try:
        await asyncio.wait_for(
            asyncio.gather(runtime.entered['wire-a'].wait(), runtime.entered['wire-b'].wait()), timeout=15
        )
        assert not first.done()
        assert not second.done()
        assert {'wire-a', 'wire-b'} in runtime.overlap
        runtime.release.set()
        results = await asyncio.gather(first, second)
        assert [result['output'] for result in results] == ['wire-a|llm:alpha', 'wire-b|llm:beta']
        assert len(runtime.requests) == 2
        identities = await asyncio.gather(runtime.turn('wire-a', 'identity'), runtime.turn('wire-b', 'identity'))
        identities = [json.loads(result['output']) for result in identities]
        assert identities[0]['pid'] != identities[1]['pid']
        assert identities[0]['home'] != identities[1]['home']
        await asyncio.gather(runtime.turn('wire-a', 'remember:alpha'), runtime.turn('wire-b', 'remember:beta'))
        recalled = await asyncio.gather(runtime.turn('wire-a', 'recall'), runtime.turn('wire-b', 'recall'))
        assert [result['output'] for result in recalled] == ['alpha', 'beta']
        for session_id in ('wire-a', 'wire-b'):
            session = session_runtime._SESSIONS[session_id]
            connection = session.connection
            process = session.proc
            assert (await session_runtime.close_codebuddy_session(session_id=session_id))['success']
            assert process.returncode is not None
            assert not connection._pending
            assert session_id not in session_runtime._SESSIONS
    finally:
        for task in (first, second):
            if not task.done():
                task.cancel()
        await asyncio.gather(first, second, return_exceptions=True)


@pytest.mark.parametrize('decision', ['allow', 'reject'])
async def test_deferred_permission_controls_real_subprocess_file_write_and_mcp(
    wire_runtime: Any, decision: str
) -> None:
    runtime = wire_runtime
    session_id = f'wire-{decision}'
    runtime.release.set()
    result = await runtime.turn(session_id, 'permission:approved-value', decision=decision)
    assert result['status'] == 'ok'
    assert result['output'] == f'waiting;{"allowed" if decision == "allow" else "denied"}'
    artifact = runtime.tmp_path / session_id / 'approved.txt'
    if decision == 'allow':
        assert artifact.read_text() == 'approved-value'
    else:
        assert not artifact.exists()
    tool_result = await runtime.turn(session_id, 'mcp:payload', decision=decision)
    assert f'{session_id}:payload' in tool_result['output']
    evidence = runtime.tmp_path / session_id / 'evidence'
    assert (evidence / 'mcp-called').read_text() == 'payload'
    assert (evidence / 'permission-finished').is_file()
    session = session_runtime._SESSIONS[session_id]
    process = session.proc
    assert (await session_runtime.close_codebuddy_session(session_id=session_id))['success']
    assert process.returncode is not None
    assert (evidence / 'closed').read_text() == 'closed\n'


async def test_permission_pending_does_not_block_cancel_terminal_and_same_process_continues(wire_runtime: Any) -> None:
    runtime = wire_runtime
    session_id = 'wire-pending'
    runtime.permissions[session_id] = asyncio.Event()
    task = asyncio.create_task(runtime.turn(session_id, 'permission:must-not-write', decision='pending'))
    try:
        await asyncio.wait_for(runtime.permissions[session_id].wait(), timeout=15)
        session = session_runtime._SESSIONS[session_id]
        process = session.proc
        # NOTE(wallyshi): Cancellation must work without pre-resolving the outstanding approval.
        interrupted = await asyncio.wait_for(
            session_runtime.interrupt_codebuddy_session(session_id=session_id), timeout=5
        )
        result = await task
        assert interrupted['success']
        assert interrupted['stopped']
        assert result['status'] == 'cancelled'
        assert result['acp_stop_reason'] == 'cancelled'
        assert process.returncode is None
        assert not (runtime.tmp_path / session_id / 'approved.txt').exists()
        identity = json.loads((await runtime.turn(session_id, 'identity', decision='pending'))['output'])
        assert identity['pid'] == process.pid
        assert session_runtime._SESSIONS[session_id] is session
        assert (runtime.tmp_path / session_id / 'evidence/permission-finished').is_file()
        assert (await session_runtime.close_codebuddy_session(session_id=session_id))['success']
        assert process.returncode is not None
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize('decision', ['allow', 'reject'])
async def test_public_host_relay_routes_approvals_and_host_tools_over_socketio(
    wire_runtime: Any, decision: str
) -> None:
    runtime = wire_runtime
    runtime.release.set()
    session_id = f'relay-{decision}'
    runtime.host_decisions[session_id] = decision
    extension_args = {
        'extension_factory': _PUBLIC_FACTORY,
        'extension_config': {'relay_permissions': True, 'host_tools': {'server_name': 'host', 'tools': [_ECHO_TOOL]}},
        'permission_policy': 'defer',
        'skip_permissions': False,
        'deferred_permission_timeout': None,
    }
    result = await runtime.turn(session_id, 'permission:approved-value', extension_args=extension_args)
    assert result['status'] == 'ok'
    assert result['output'] == f'waiting;{"allowed" if decision == "allow" else "denied"}'
    artifact = runtime.tmp_path / session_id / 'approved.txt'
    assert artifact.exists() is (decision == 'allow'), 'Only the Host decision may authorize the write'
    assert [request['channel_id'] for request in runtime.host_permission_requests] == [session_id]

    tool_result = await runtime.turn(session_id, 'mcp:payload', extension_args=extension_args)
    assert tool_result['output'] == f'host:{session_id}:payload'
    assert [(request['channel_id'], request['tool_name']) for request in runtime.host_tool_requests] == [
        (session_id, 'echo')
    ]

    session = session_runtime._SESSIONS[session_id]
    process = session.proc
    assert (await session_runtime.close_codebuddy_session(session_id=session_id))['success']
    assert process.returncode is not None
    runtime.host_permission_requests.clear()
    reopened = await runtime.turn(session_id, 'permission:second', extension_args=extension_args)
    assert reopened['status'] == 'ok', 'Closing must release the Host channels for a recreated session'
    assert [request['channel_id'] for request in runtime.host_permission_requests] == [session_id]


async def test_stale_approval_from_a_closed_session_cannot_authorize_its_same_id_successor(wire_runtime: Any) -> None:
    """A recreated CLI restarts ACP request ids and keeps fixed option ids, so only the relay token tells them apart."""
    runtime = wire_runtime
    runtime.release.set()
    session_id = 'relay-recreated'
    runtime.host_decisions[session_id] = 'hold'
    extension_args = {
        'extension_factory': _PUBLIC_FACTORY,
        'extension_config': {'relay_permissions': True},
        'permission_policy': 'defer',
        'skip_permissions': False,
        'deferred_permission_timeout': None,
    }
    artifact = runtime.tmp_path / session_id / 'approved.txt'
    old_turn = asyncio.create_task(runtime.turn(session_id, 'permission:OLD-operation', extension_args=extension_args))
    new_turn: asyncio.Task[dict[str, Any]] | None = None
    try:
        [stale] = await runtime.wait_for_permission_requests(1)
        interrupted = await session_runtime.interrupt_codebuddy_session(session_id=session_id)
        assert interrupted['stopped']
        assert (await old_turn)['status'] == 'cancelled'
        assert (await session_runtime.close_codebuddy_session(session_id=session_id))['success']

        new_turn = asyncio.create_task(
            runtime.turn(session_id, 'permission:NEW-operation', extension_args=extension_args)
        )
        fresh = (await runtime.wait_for_permission_requests(2))[1]
        assert (fresh['channel_id'], fresh['acp_request_id']) == (stale['channel_id'], stale['acp_request_id']), (
            'The recreated peer must reuse the old channel and request id for this regression to be meaningful'
        )

        await runtime.resolve_permission(stale, 'allow_once')
        await asyncio.sleep(0.5)
        assert not artifact.exists(), 'An approval issued to the closed session must not authorize its successor'
        assert not new_turn.done(), 'The new request must keep waiting for its own Host decision'

        await runtime.resolve_permission(fresh, 'allow_once')
        result = await new_turn
        assert result['output'] == 'waiting;allowed'
        assert artifact.read_text() == 'NEW-operation'
    finally:
        for task in (old_turn, new_turn):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(*(task for task in (old_turn, new_turn) if task is not None), return_exceptions=True)
