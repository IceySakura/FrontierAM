"""Drive a real WorkBuddy ``wb --acp`` (ACP version 2) through ``AcpConnection`` against a stub model.

Set ``WORKBUDDY_ACP_WB_JS`` to the ``package/bin/wb.js`` of an unpacked linux-x64 WorkBuddy release
(5.11.0 or later) and put Node.js ``^22.16.0 || >=24`` on PATH; otherwise the test is skipped. The
model is a scripted loopback server, so no model gateway is involved, but wb itself still tries its
built-in remote connectors at startup.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import shutil
import signal
import subprocess

from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from nexus.extensions.agents.installed_common.acp import (
    ACP_COMMAND_METHOD,
    ACP_PROMPT_META,
    ACP_REQUEST_ID_META,
    AcpConnection,
    accepted_request_id,
    supports_prompt_steer,
    supports_session_fork,
)

from .stub_model import StubModelServer, StubReply, message_text


pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

_WB_JS_ENV = 'WORKBUDDY_ACP_WB_JS'
_MODEL_ID = 'stub-model'
_TURN_TIMEOUT = 120.0
_SCENARIOS = ('PRESENT', 'BASH', 'SLOW', 'FORKQ')
# NOTE(wallyshi): wb 5.11.0 cancels a prompt sent within about a second after a cancelled idle, without
# calling the model; waiting this long after one keeps the next turn real.
_AFTER_CANCEL_PAUSE = 1.5


def _node_supports_workbuddy() -> bool:
    node = shutil.which('node')
    if node is None:
        return False
    version = subprocess.run([node, '--version'], capture_output=True, text=True, check=False).stdout.strip()
    match = re.match(r'v(\d+)\.(\d+)', version)
    if match is None:
        return False
    major, minor = int(match.group(1)), int(match.group(2))
    return major >= 24 or (major == 22 and minor >= 16)


_WB_JS = os.environ.get(_WB_JS_ENV, '')
pytestmark.append(
    pytest.mark.skipif(
        not _WB_JS or not Path(_WB_JS).is_file() or not _node_supports_workbuddy(),
        reason=f'{_WB_JS_ENV} must name a wb.js and Node.js ^22.16.0 || >=24 must be on PATH',
    )
)


def _script(workspace: Path) -> Callable[[dict[str, Any], dict[str, str]], StubReply]:
    """Pick the reply from a scenario word in the latest user message of the conversation."""

    def reply(body: dict[str, Any], headers: dict[str, str]) -> StubReply:
        if headers.get('X-Agent-Purpose') == 'conversation_topic':
            return StubReply(text='探针标题')
        messages = body.get('messages') or []
        users = [message_text(message.get('content')) for message in messages if message.get('role') == 'user']
        scenario = next((word for text in reversed(users) for word in _SCENARIOS if word in text), '')
        tool_results = 0
        for message in reversed(messages):
            if message.get('role') == 'tool':
                tool_results += 1
            elif message.get('role') == 'user':
                break
        if scenario == 'BASH' and tool_results == 0:
            return StubReply(
                tool_name='Bash', tool_args={'command': 'echo hi > out.txt', 'description': '写入 out.txt'}
            )
        if scenario == 'SLOW':
            return StubReply(text='慢慢输出的一段话。' * 12, chunks=16, chunk_delay=0.4)
        if scenario == 'PRESENT' and tool_results == 0:
            return StubReply(
                tool_name='Write', tool_args={'file_path': str(workspace / 'report.md'), 'content': '# 报告\n'}
            )
        if scenario == 'PRESENT' and tool_results == 1:
            return StubReply(
                tool_name='present_files',
                tool_args={'files': [str(workspace / 'report.md')], 'cwd': str(workspace), 'explanation': '报告'},
            )
        if scenario == 'FORKQ':
            return StubReply(text='这是追问的回答。')
        return StubReply(text='好的。')

    return reply


@dataclass
class _Peer:
    connection: AcpConnection
    process: asyncio.subprocess.Process
    workspace: Path
    stub: StubModelServer
    updates: list[dict[str, Any]] = field(default_factory=list)
    permissions: list[tuple[int, dict[str, Any]]] = field(default_factory=list)
    hold_permissions: bool = False
    _waiters: list[tuple[Callable[[dict[str, Any]], bool], asyncio.Future[dict[str, Any]]]] = field(
        default_factory=list
    )

    def observe(self, params: dict[str, Any]) -> None:
        self.updates.append(params)
        for predicate, future in list(self._waiters):
            if not future.done() and predicate(params):
                future.set_result(params)

    async def wait_for(
        self, predicate: Callable[[dict[str, Any]], bool], *, start: int = 0, timeout: float = _TURN_TIMEOUT
    ) -> Any:
        for params in self.updates[start:]:
            if predicate(params):
                return params
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._waiters.append((predicate, future))
        try:
            return await asyncio.wait_for(future, timeout=timeout)
        finally:
            self._waiters.remove((predicate, future))

    async def idle(self, session_id: str, request_id: str) -> dict[str, Any]:
        def is_idle(params: dict[str, Any]) -> bool:
            update = params.get('update') or {}
            return (
                params.get('sessionId') == session_id
                and update.get('sessionUpdate') == 'state_update'
                and update.get('state') == 'idle'
                and (update.get('_meta') or {}).get(ACP_REQUEST_ID_META) == request_id
            )

        params = await self.wait_for(is_idle)
        self.connection.finish_prompt(session_id)
        return params['update']

    async def first_chunk(self, session_id: str, *, start: int) -> None:
        def is_chunk(params: dict[str, Any]) -> bool:
            return (
                params.get('sessionId') == session_id
                and (params.get('update') or {}).get('sessionUpdate') == 'agent_message_chunk'
            )

        await self.wait_for(is_chunk, start=start)


@pytest.fixture
async def peer(tmp_path: Path) -> AsyncIterator[_Peer]:
    workspace = tmp_path / 'workspace'
    config_dir = tmp_path / 'workbuddy'
    home = tmp_path / 'home'
    for path in (workspace, config_dir / 'app', home):
        path.mkdir(parents=True)
    stub = StubModelServer(_script(workspace))
    base_url = stub.start()
    models = {
        'models': [
            {
                'id': _MODEL_ID,
                'name': _MODEL_ID,
                'vendor': 'nexus',
                'apiKey': 'sk-local',
                'maxInputTokens': 200000,
                'maxOutputTokens': 65536,
                'url': f'{base_url}/chat/completions',
                'supportsToolCall': True,
                'supportsImages': True,
            }
        ],
        'availableModels': [_MODEL_ID],
    }
    (config_dir / 'models.json').write_text(json.dumps(models))
    (config_dir / 'app' / 'app-config.json').write_text(json.dumps({'generateMemoryEnabled': False}))
    env = {
        'PATH': os.environ.get('PATH', ''),
        'HOME': str(home),
        'CODEBUDDY_API_KEY': 'sk-local',
        'CODEBUDDY_BASE_URL': base_url,
        'WORKBUDDY_CONFIG_DIR': str(config_dir),
        'WORKBUDDY_USER_DATA_DIR': str(config_dir / 'app'),
        'NO_COLOR': '1',
        'CI': '1',
    }
    process = await asyncio.create_subprocess_exec(
        'node',
        _WB_JS,
        '--acp',
        '--model',
        f'custom-local:{_MODEL_ID}',
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=(tmp_path / 'wb.stderr.log').open('wb'),
        cwd=str(workspace),
        env=env,
        start_new_session=True,
        limit=64 * 1024 * 1024,
    )
    assert process.stdout is not None, 'wb must be spawned with a stdout pipe'
    assert process.stdin is not None, 'wb must be spawned with a stdin pipe'
    peer: _Peer

    async def handle_permission(params: dict[str, Any], request_id: int) -> dict[str, Any] | None:
        peer.permissions.append((request_id, params))
        if peer.hold_permissions:
            await asyncio.Event().wait()
        option = next(option for option in params['options'] if option.get('kind') == 'allow_once')
        return {'outcome': {'outcome': 'selected', 'optionId': option['optionId']}}

    connection = AcpConnection(
        process.stdout,
        process.stdin,
        permission_policy='defer',
        tolerate_noise=True,
        deferred_permission_handler=handle_permission,
        deferred_permission_timeout=None,
    )
    peer = _Peer(connection=connection, process=process, workspace=workspace, stub=stub)
    connection.add_update_listener(peer.observe)
    connection.start()
    try:
        yield peer
    finally:
        # NOTE(wallyshi): wb reaps the agent processes it started on stdin EOF; a signal is only the fallback.
        process.stdin.close()
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(process.wait(), timeout=15)
        for sig in (signal.SIGTERM, signal.SIGKILL):
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, sig)
                await asyncio.sleep(0.5)
        await connection.close()
        stub.stop()


async def test_workbuddy_turn_defers_permission_steers_forks_and_presents(peer: _Peer) -> None:
    connection = peer.connection
    workspace = str(peer.workspace)
    initialize = await connection.initialize(
        timeout=120, client_capabilities={'elicitation': {'form': {}}}, protocol_version=2
    )
    assert supports_session_fork(initialize), f'wb must advertise session/fork: {initialize!r}'
    assert supports_prompt_steer(initialize), f'wb must advertise steer: {initialize!r}'

    session_id = await connection.new_session(cwd=workspace, timeout=60)
    permission_values = {
        option['configId']: [choice['value'] for choice in option.get('options') or []]
        for option in connection.config_options[session_id]
    }['permission_mode']
    assert {'default', 'bypassPermissions', 'fullAccess'} <= set(permission_values), permission_values
    options = await connection.set_config_option(session_id=session_id, config_id='permission_mode', value='default')
    assert next(option for option in options if option['configId'] == 'permission_mode')['currentValue'] == 'default'

    # An accepted prompt must keep its later permission deferred to the handler until the turn is idle.
    accepted = await connection.prompt(session_id=session_id, text='请 BASH 写 out.txt', timeout=60)
    request_id = accepted_request_id(accepted)
    assert request_id, f'a version 2 prompt is accepted with a request id: {accepted!r}'
    idle = await peer.idle(session_id, request_id)
    assert idle.get('stopReason') == 'end_turn', idle
    assert len(peer.permissions) == 1, peer.permissions
    tool_call = peer.permissions[0][1].get('toolCall') or {}
    assert tool_call.get('name') == 'Bash', f'the permission names its tool: {tool_call!r}'
    assert (tool_call.get('rawInput') or {}).get('command') == 'echo hi > out.txt', (
        f'the permission carries the cached raw input: {tool_call!r}'
    )
    assert (peer.workspace / 'out.txt').read_text().strip() == 'hi'

    # A permission the peer withdraws on cancel stops waiting here.
    peer.hold_permissions = True
    accepted = await connection.prompt(session_id=session_id, text='再 BASH 一次', timeout=60)
    request_id = accepted_request_id(accepted)
    assert request_id
    for _ in range(600):
        if len(peer.permissions) == 2:
            break
        await asyncio.sleep(0.1)
    held_id = peer.permissions[1][0]
    assert connection.has_deferred_permission(held_id), 'the second permission waits on the handler'
    await connection.notify('session/cancel', {'sessionId': session_id})
    idle = await peer.idle(session_id, request_id)
    assert idle.get('stopReason') == 'cancelled', idle
    assert not connection.has_deferred_permission(held_id), '$/cancel_request withdraws the waiting permission'
    peer.hold_permissions = False
    await asyncio.sleep(_AFTER_CANCEL_PAUSE)

    # Steer into a running turn, then fork it, ask on the fork in ask mode, and close the fork.
    seen = len(peer.updates)
    accepted = await connection.prompt(session_id=session_id, text='请 SLOW 地讲', timeout=60)
    request_id = accepted_request_id(accepted)
    assert request_id
    await peer.first_chunk(session_id, start=seen)
    steered = await connection.prompt(
        session_id=session_id,
        text='顺便加一句',
        timeout=30,
        meta={ACP_PROMPT_META: {'mode': 'steer', 'expectedRequestId': request_id}},
    )
    assert accepted_request_id(steered) == request_id, f'a steer joins the running turn: {steered!r}'
    assert steered.get('messageId') != request_id, 'a steer has its own message id'
    fork_id = await connection.fork_session(session_id=session_id, cwd=workspace, timeout=60)
    assert fork_id != session_id
    await connection.set_config_option(session_id=fork_id, config_id='mode', value='ask')
    fork_accepted = await connection.prompt(session_id=fork_id, text='FORKQ 做到哪了？', timeout=60)
    fork_request_id = accepted_request_id(fork_accepted)
    assert fork_request_id
    fork_idle = await peer.idle(fork_id, fork_request_id)
    assert fork_idle.get('stopReason') == 'end_turn', fork_idle
    fork_calls = [request for request in peer.stub.requests if request['headers'].get('X-Conversation-ID') == fork_id]
    assert fork_calls, 'the fork asks the model under its own conversation id'
    assert fork_calls[-1]['headers'].get('X-Agent-Intent') == 'ask', fork_calls[-1]['headers']
    await connection.close_session(session_id=fork_id)
    idle = await peer.idle(session_id, request_id)
    assert idle.get('stopReason') == 'end_turn', idle
    steer_seen = any(
        '顺便加一句' in json.dumps(request['body'].get('messages'), ensure_ascii=False)
        for request in peer.stub.requests
        if request['headers'].get('X-Conversation-ID') == session_id
    )
    assert steer_seen, 'the steered text reaches the model within the same turn'

    # present_files reaches the client as the _codebuddy.ai/command notification, re-emitted as an update.
    accepted = await connection.prompt(session_id=session_id, text='请 PRESENT 报告', timeout=60)
    request_id = accepted_request_id(accepted)
    assert request_id
    idle = await peer.idle(session_id, request_id)
    assert idle.get('stopReason') == 'end_turn', idle
    commands = [
        params['update']
        for params in peer.updates
        if params.get('sessionId') == session_id
        and (params.get('update') or {}).get('sessionUpdate') == ACP_COMMAND_METHOD
    ]
    report = str(peer.workspace / 'report.md')
    assert any(
        command.get('action') == 'openResultView'
        and command['params'].get('targetFile') == report
        or command.get('action') == 'presentFiles'
        and report in command['params'].get('files', [])
        for command in commands
    ), commands

    await connection.close_session(session_id=session_id)
