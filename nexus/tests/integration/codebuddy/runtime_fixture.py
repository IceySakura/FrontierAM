"""A deterministic ACP subprocess and a public runtime-only session extension.

This fixture replaces the CLI, not the ACP/HTTP/Socket.IO transports. It does not
prove compatibility with the real CodeBuddy CLI; that requires image E2E tests.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import sys

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any


async def _write_frame(frame: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps({'jsonrpc': '2.0', **frame}) + '\n')
    sys.stdout.flush()


async def _serve_peer() -> None:
    session_id = 'peer-session'
    mcp_servers: list[dict[str, Any]] = []
    pending_permissions: dict[int, asyncio.Future[dict[str, Any]]] = {}
    active_prompts: set[asyncio.Task[None]] = set()
    next_permission_id = 1000
    remembered = ''

    async def prompt(frame: dict[str, Any]) -> None:
        nonlocal next_permission_id, remembered
        request_id = frame['id']
        text = frame['params']['prompt'][0]['text']
        stop_reason = 'end_turn'
        output = ''
        try:
            if text.startswith('permission:'):
                next_permission_id += 1
                permission_id = next_permission_id
                permission = asyncio.get_running_loop().create_future()
                pending_permissions[permission_id] = permission
                await _write_frame({
                    'id': permission_id,
                    'method': 'session/request_permission',
                    'params': {
                        'sessionId': session_id,
                        'toolCall': {'toolCallId': str(permission_id), 'title': text},
                        'options': [
                            {'optionId': 'opaque-deny', 'kind': 'reject_once', 'name': 'Reject'},
                            {'optionId': 'opaque-allow', 'kind': 'allow_once', 'name': 'Allow'},
                        ],
                    },
                })
                # NOTE(wallyshi): A notification behind approval exposes reader head-of-line blocking.
                await _write_frame({
                    'method': 'session/update',
                    'params': {
                        'sessionId': session_id,
                        'update': {
                            'sessionUpdate': 'agent_message_chunk',
                            'content': {'type': 'text', 'text': 'waiting;'},
                        },
                    },
                })
                try:
                    decision = (await permission)['outcome']
                finally:
                    pending_permissions.pop(permission_id, None)
                if decision.get('optionId') == 'opaque-allow':
                    Path('approved.txt').write_text(text.partition(':')[2])
                    output = 'allowed'
                else:
                    output = 'denied'
            elif text.startswith('llm:'):
                import httpx

                model_config = json.loads((Path.home() / '.codebuddy/models.json').read_text())['models'][0]
                async with httpx.AsyncClient(trust_env=False, timeout=10) as client:
                    response = await client.post(
                        model_config['url'],
                        json={'model': model_config['id'], 'messages': [{'role': 'user', 'content': text}]},
                    )
                    response.raise_for_status()
                    output = response.json()['choices'][0]['message']['content']
            elif text.startswith('mcp:'):
                from mcp import ClientSession
                from mcp.client.streamable_http import streamablehttp_client
                from mcp.types import TextContent

                async with streamablehttp_client(mcp_servers[0]['url']) as (read_stream, write_stream, _):
                    async with ClientSession(read_stream, write_stream) as client:
                        await client.initialize()
                        result = await client.call_tool('echo', {'text': text.partition(':')[2]})
                        output = ''.join(item.text for item in result.content if isinstance(item, TextContent))
            elif text.startswith('remember:'):
                remembered = text.partition(':')[2]
                output = remembered
            elif text == 'recall':
                output = remembered
            elif text == 'identity':
                output = json.dumps({'pid': os.getpid(), 'home': str(Path.home()), 'cwd': os.getcwd()})
            else:
                output = text
        except asyncio.CancelledError:
            stop_reason = 'cancelled'
        except Exception as error:
            await _write_frame({'id': request_id, 'error': {'code': -32603, 'message': str(error)}})
            return
        if output:
            await _write_frame({
                'method': 'session/update',
                'params': {
                    'sessionId': session_id,
                    'update': {'sessionUpdate': 'agent_message_chunk', 'content': {'type': 'text', 'text': output}},
                },
            })
        await _write_frame({'id': request_id, 'result': {'stopReason': stop_reason}})

    try:
        while line := await asyncio.to_thread(sys.stdin.buffer.readline):
            frame = json.loads(line)
            method = frame.get('method')
            if method == 'initialize':
                await _write_frame({
                    'id': frame['id'],
                    'result': {'protocolVersion': 1, 'agentCapabilities': {'mcpCapabilities': {'http': True}}},
                })
            elif method == 'session/new':
                mcp_servers = frame['params']['mcpServers']
                await _write_frame({'id': frame['id'], 'result': {'sessionId': session_id}})
            elif method == 'session/prompt':
                task = asyncio.create_task(prompt(frame))
                active_prompts.add(task)
                task.add_done_callback(active_prompts.discard)
            elif method == 'session/cancel':
                for task in tuple(active_prompts):
                    task.cancel()
            elif method is None:
                permission = pending_permissions.get(frame.get('id'))
                if permission is not None and not permission.done():
                    permission.set_result(frame['result'])
    finally:
        for task in active_prompts:
            task.cancel()
        await asyncio.gather(*active_prompts, return_exceptions=True)


@contextlib.asynccontextmanager
async def session_extension(
    *, session_id: str, sio: Any, connection: Any, config: dict[str, Any]
) -> AsyncIterator[Any]:
    """Expose a real MCP HTTP tool and record permission ownership without projects."""
    import socket

    import uvicorn

    from mcp.server.fastmcp import FastMCP

    from nexus.extensions.agents.codebuddy.session_runtime import CodeBuddySessionExtension

    evidence = Path(config['evidence'])
    evidence.mkdir(parents=True, exist_ok=True)
    (evidence / 'entered').write_text(session_id)
    decision = config.get('decision', 'allow')

    async def permission_handler(params: dict[str, Any], request_id: int) -> dict[str, Any]:
        (evidence / 'permission.json').write_text(json.dumps({'id': request_id, 'params': params}))
        await sio.emit('fixture:permission', {'session_id': session_id, 'request_id': request_id})
        try:
            if decision == 'pending':
                await asyncio.Event().wait()
            kind = 'allow_once' if decision == 'allow' else 'reject_once'
            selected = next(option['optionId'] for option in params['options'] if option['kind'] == kind)
            return {'outcome': {'outcome': 'selected', 'optionId': selected}}
        finally:
            (evidence / 'permission-finished').write_text(str(request_id))

    mcp = FastMCP('public-runtime-fixture', stateless_http=True, json_response=True)

    @mcp.tool()
    async def echo(text: str) -> str:
        (evidence / 'mcp-called').write_text(text)
        return f'{session_id}:{text}'

    listener = socket.socket()
    listener.bind(('127.0.0.1', 0))
    port = listener.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(mcp.streamable_http_app(), log_level='error', loop='asyncio'))
    server_task = asyncio.create_task(server.serve(sockets=[listener]))
    try:
        deadline = asyncio.get_running_loop().time() + 5
        while not server.started:
            if server_task.done():
                await server_task
                raise RuntimeError('MCP fixture stopped before serving')
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError('MCP fixture did not start')
            await asyncio.sleep(0.01)
        yield CodeBuddySessionExtension(
            permission_handler=permission_handler,
            mcp_servers=({'type': 'http', 'name': 'fixture', 'url': f'http://127.0.0.1:{port}/mcp', 'headers': []},),
        )
    finally:
        server.should_exit = True
        try:
            await asyncio.wait_for(server_task, timeout=5)
        finally:
            listener.close()
            with (evidence / 'closed').open('a') as output:
                output.write('closed\n')


def inspect_runtime() -> dict[str, Any]:
    """Return candidate identity and resource state for no-venv image verification."""
    import hashlib
    import importlib

    from importlib.metadata import version

    modules = (
        'nexus.runtime.server.services.rsio_llm_bridge',
        'nexus.runtime.server.services.openai_proxy',
        'nexus.extensions.agents.installed_common.acp',
        'nexus.extensions.agents.installed_common.host_rpc',
        'nexus.extensions.agents.installed_common.host_tool_mcp',
        'nexus.extensions.agents.codebuddy.runtime_setup',
        'nexus.extensions.agents.codebuddy.host_relay',
        'nexus.extensions.agents.codebuddy.session_runtime',
    )
    files = {}
    for name in modules:
        module = importlib.import_module(name)
        assert module.__file__ is not None, f'Expected a concrete module source for {name}'
        source = Path(module.__file__)
        files[name] = {'path': str(source), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
    runtime = importlib.import_module(modules[-1])
    return {
        'python': sys.version,
        'executable': sys.executable,
        'nexus_version': version('nexus-agent'),
        'files': files,
        'sessions': {
            name: {'pid': session.proc.pid, 'running': session.is_running, 'pending': len(session.connection._pending)}
            for name, session in runtime._SESSIONS.items()
        },
        'project_modules': sorted(name for name in sys.modules if name.startswith('nexus.extensions.projects.')),
    }


if __name__ == '__main__':
    asyncio.run(_serve_peer())
