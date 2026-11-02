from __future__ import annotations

import asyncio
import json
import os
import socket

from copy import deepcopy
from pathlib import Path
from typing import Any

import httpx
import pytest
import uvicorn

from nexus.gateway.processor import Processor
from nexus.gateway.session import GatewaySession
from nexus.gateway.trajectory_store import InMemoryTrajectoryRecordStore
from nexus.gateway.types import BackendResult, GatewayCompletionResult
from nexus.runtime.server.services.anthropic_proxy.server import AnthropicProxy


CHILD_PROMPT = 'HYCODE_GATEWAY_IDENTICAL_CHILD_PROMPT'


def _tool_call(call_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        'id': call_id,
        'type': 'function',
        'function': {'name': name, 'arguments': json.dumps(arguments)},
    }


class _HyCodeBackend:
    def __init__(self) -> None:
        self.requests: list[tuple[list[dict], dict[str, Any] | None]] = []
        self.labels: dict[str, str] = {}
        self._first_child_ready = asyncio.Event()

    async def create_completion(
        self,
        request_id: str,
        *,
        messages: list[dict] | None = None,
        request_kwargs: dict[str, Any] | None = None,
        request_meta: dict[str, Any] | None = None,
        **_: Any,
    ) -> BackendResult:
        del request_id, request_kwargs
        assert messages is not None
        self.requests.append((deepcopy(messages), deepcopy(request_meta)))
        agent_id = (request_meta or {}).get('agent_id')
        if agent_id is None:
            has_tool_result = any(message.get('role') == 'tool' for message in messages)
            if has_tool_result:
                return self._result({'role': 'assistant', 'content': 'MAIN_DONE'}, 'stop')
            return self._result(
                {
                    'role': 'assistant',
                    'content': None,
                    'tool_calls': [
                        _tool_call(
                            f'spawn-{label.lower()}',
                            'Agent',
                            {
                                'description': f'capture {label.lower()} identity',
                                'prompt': CHILD_PROMPT,
                                'subagent_type': 'general-purpose',
                                'run_in_background': False,
                            },
                        )
                        for label in ('ALPHA', 'BETA')
                    ],
                },
                'tool_calls',
            )

        agent_id = str(agent_id)
        label = self.labels.get(agent_id)
        if label is None:
            label = ('ALPHA', 'BETA')[len(self.labels)]
            self.labels[agent_id] = label
        has_tool_result = any(message.get('role') == 'tool' for message in messages)
        if has_tool_result:
            return self._result({'role': 'assistant', 'content': f'CHILD_{label}_DONE'}, 'stop')

        if len(self.labels) == 2:
            self._first_child_ready.set()
        await asyncio.wait_for(self._first_child_ready.wait(), timeout=30)
        return self._result(
            {
                'role': 'assistant',
                'content': None,
                'tool_calls': [
                    _tool_call(f'child-{label.lower()}-bash', 'Bash', {'command': f'echo CHILD_{label}_TOOL'})
                ],
            },
            'tool_calls',
        )

    @staticmethod
    def _result(message: dict[str, Any], stop_reason: str) -> BackendResult:
        return BackendResult(
            message=message,
            stop_reason=stop_reason,
            id='chatcmpl-hycode-e2e',
            model='claude-3-5-sonnet-20241022',
            usage={'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2},
        )


class _GatewayBridge:
    def __init__(self, session: GatewaySession) -> None:
        self.session = session

    async def request(self, params: dict[str, Any], timeout: float) -> dict[str, Any]:
        request = deepcopy(params)
        messages = request.pop('messages')
        request_meta = request.pop('request_meta', None)
        result = await asyncio.wait_for(
            self.session.create_completion(messages, request_kwargs=request, request_meta=request_meta),
            timeout=timeout,
        )
        return self._openai_response(result, request.get('model'))

    @staticmethod
    def _openai_response(result: GatewayCompletionResult, model: str | None) -> dict[str, Any]:
        return {
            'id': result.id,
            'object': 'chat.completion',
            'created': 1,
            'model': result.model or model,
            'choices': [
                {
                    'index': 0,
                    'finish_reason': 'tool_calls' if result.message.get('tool_calls') else result.stop_reason,
                    'message': result.message,
                }
            ],
            'usage': result.usage,
        }


@pytest.mark.integration
@pytest.mark.parametrize('binary_env', ['HYCODE_TEST_BINARY', 'HYNESS_TEST_BINARY'])
async def test_compiled_hycode_builds_distinct_same_prompt_sidechains(binary_env: str, tmp_path: Path) -> None:
    binary = Path(os.environ.get(binary_env, ''))
    if not binary.is_file():
        pytest.skip(f'set {binary_env} to the installed agent test entrypoint')

    backend = _HyCodeBackend()
    session = GatewaySession(
        'hycode-local-e2e',
        Processor(backend),
        InMemoryTrajectoryRecordStore('cli-e2e'),
        scaffold_policy='hycode',
    )
    proxy = AnthropicProxy(sio=object(), listen_port=0, conversion_log_level='WARNING')  # type: ignore[arg-type]
    proxy.bridge = _GatewayBridge(session)  # type: ignore[assignment]

    listen_socket = socket.socket()
    listen_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listen_socket.bind(('127.0.0.1', 0))
    listen_socket.listen()
    port = listen_socket.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(proxy.create_app(), log_level='warning'))
    server_task = asyncio.create_task(server.serve(sockets=[listen_socket]))
    try:
        async with httpx.AsyncClient() as client:
            for _ in range(100):
                try:
                    if (await client.get(f'http://127.0.0.1:{port}/')).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(0.02)
            else:
                pytest.fail('local AnthropicProxy did not start')

        env = {
            key: value
            for key, value in os.environ.items()
            if key.lower() not in {'http_proxy', 'https_proxy', 'all_proxy'}
            and not key.startswith(('ANTHROPIC_', 'CLAUDE_', 'HYNESS_'))
        }
        (tmp_path / 'home').mkdir()
        env.update(
            {
                'NO_PROXY': '*',
                'no_proxy': '*',
                'CLAUDE_CODE_DISABLE_BACKGROUND_TASKS': '1',
                'ANTHROPIC_BASE_URL': f'http://127.0.0.1:{port}',
                'ANTHROPIC_API_KEY': 'sk-test',
                'ANTHROPIC_MODEL': 'claude-3-5-sonnet-20241022',
                'HOME': str(tmp_path / 'home'),
                'CLAUDE_CONFIG_DIR': str(tmp_path / 'config'),
                'HYNESS_PLUGIN_REGISTRY': str(tmp_path / 'plugins.json'),
                'IS_SANDBOX': '1',
            }
        )
        if binary_env == 'HYNESS_TEST_BINARY' and os.environ.get('HYNESS_TEST_OVERLAY'):
            env['HYNESS_APP_CONFIG_OVERLAY'] = os.environ['HYNESS_TEST_OVERLAY']
        process = await asyncio.create_subprocess_exec(
            str(binary),
            '--dangerously-skip-permissions',
            '--print',
            '--output-format',
            'stream-json',
            '--verbose',
            '--max-turns=4',
            '--',
            'delegate each requested marker to an Agent',
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=120)
        except TimeoutError:
            process.terminate()
            await process.wait()
            raise
        assert process.returncode == 0, (stdout.decode(), stderr.decode())
    finally:
        server.should_exit = True
        await server_task
        listen_socket.close()

    await session.finish_pending_publications()
    trajectory = session.end('success')
    assert len(backend.labels) == 2
    assert set(trajectory.chains) == {'main', *backend.labels}
    # Concurrent requests may reach the backend in either order. Correlate
    # its response labels to actual joins, not to the request arrival order.
    final_main = next(messages for messages, meta in reversed(backend.requests) if not (meta or {}).get('agent_id'))
    join_ids = {
        label: message['tool_call_id']
        for label in backend.labels.values()
        for message in final_main
        if message.get('role') == 'tool' and f'CHILD_{label}_DONE' in str(message.get('content'))
    }
    assert set(join_ids.values()) == {'spawn-alpha', 'spawn-beta'}
    for agent_id, label in backend.labels.items():
        chain = trajectory.chains[agent_id]
        tool_use_id = join_ids[label]
        assert chain.parent_chain_id == 'main'
        assert chain.spawn is not None
        assert chain.spawn.tool_use_id == tool_use_id
        assert chain.join is not None
        assert chain.join.tool_use_id == tool_use_id
        assert chain.join.kind == 'inline'

    main_requests = [meta for _, meta in backend.requests if not (meta or {}).get('agent_id')]
    child_requests = [(messages, meta) for messages, meta in backend.requests if (meta or {}).get('agent_id')]
    first_child_requests = [
        (messages, meta)
        for messages, meta in child_requests
        if not any(message.get('role') == 'tool' for message in messages)
    ]
    assert len(main_requests) == 2
    assert len(child_requests) == 4
    assert len(first_child_requests) == 2
    assert first_child_requests[0][0] == first_child_requests[1][0]
    assert {meta['agent_id'] for _, meta in child_requests} == set(backend.labels)
    assert {meta['session_id'] for _, meta in backend.requests if meta} == {
        next(meta['session_id'] for meta in main_requests if meta)
    }
