"""OSWorld bridge extension regressions."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tarfile
import threading

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from fastapi.testclient import TestClient

from nexus.runtime.bridge.adapters import ags_osworld as extension
from nexus.runtime.bridge.adapters.ags_osworld import LLMRequestHandler as OSWorldHandler
from nexus.runtime.bridge.app import create_bridge_fastapi_app
from nexus.runtime.bridge.artifacts import _bridge_content_hash, build_app_bundle
from nexus.runtime.bridge.host_client import connect_bridge_llm_handler
from nexus.runtime.bridge.runtime_kind import NativeBridgeBundleTarget
from nexus.runtime.server.services.llm_handler import LLMRequestHandler


EXTENSION = extension.__name__


@pytest.mark.asyncio
@pytest.mark.parametrize('harness', ['claude_code', 'codex'])
async def test_profile_runs_adapter_without_shared_dispatch(monkeypatch, harness):
    from nexus.extensions.agents.installed_common import native_bridge
    from nexus.extensions.tasks.os_world.execution.claude_code import OSWorldClaudeExecutionProfile
    from nexus.extensions.tasks.os_world.execution.codex_profile import OSWorldCodexExecutionProfile

    if harness == 'claude_code':
        profile = OSWorldClaudeExecutionProfile({'guide_text': 'guide'}, {}, 'task')
    else:
        profile = OSWorldCodexExecutionProfile(
            {'guide_version': 'v1', 'model_catalog_preset': 'gpt6_catalog_v8'}, {}, 'task'
        )
    client = object()
    result = {'success': True}
    adapter_run = AsyncMock(return_value=result)
    shared_run = AsyncMock(side_effect=AssertionError('profile must not use shared dispatch'))
    monkeypatch.setattr(extension, 'call_native_bridge_runtime_agent', adapter_run)
    monkeypatch.setattr(native_bridge, 'call_native_bridge_runtime_agent', shared_run)
    assert await profile.run(client) is result
    adapter_run.assert_awaited_once_with(profile.execute, **profile.bridge_options(client))
    shared_run.assert_not_awaited()
    assert 'bridge_extension' not in adapter_run.call_args.kwargs


@pytest.mark.asyncio
@pytest.mark.parametrize('harness', ['claude_code', 'codex'])
async def test_agent_delegates_explicit_profile_to_run(monkeypatch, harness):
    import importlib

    agent_module = importlib.import_module(f'nexus.extensions.agents.{harness}.agent')
    base = importlib.import_module(f'nexus.extensions.agents.{harness}.native_bridge.adapters.base')
    client = object()
    profile = SimpleNamespace(
        run=AsyncMock(
            return_value={
                'success': True,
                'output': 'ok',
                'stderr': '',
                'exit_code': 0,
                'execution_time': 0,
                'exit_status': 'finished',
            }
        )
    )
    monkeypatch.setattr(base, 'build_execution_profile', lambda *args: profile)
    context = SimpleNamespace(get_openai_client=lambda: client)
    monkeypatch.setattr('nexus.core.require_context', lambda: context)
    if harness == 'claude_code':
        monkeypatch.setattr(agent_module, 'require_context', lambda: context)
    shared = AsyncMock(side_effect=AssertionError('explicit profile must own its execution'))
    monkeypatch.setattr(agent_module, 'call_native_bridge_runtime_agent', shared)
    input_type = (
        agent_module.ClaudeCodeNativeBridgeInput if harness == 'claude_code' else agent_module.CodexNativeBridgeInput
    )
    entry = getattr(agent_module, f'run_{harness}_native_bridge')
    output = await entry(input_type(user_prompt='task', execution_profile='fixture'))
    assert output.success
    profile.run.assert_awaited_once_with(client)
    shared.assert_not_awaited()


@pytest.mark.asyncio
async def test_bundle_build_coalesces_and_survives_waiter_cancellation(monkeypatch, tmp_path):
    from nexus.runtime.bridge import runtime_kind

    started, finish = threading.Event(), threading.Event()
    calls = []
    artifact = tmp_path / 'bridge.tar.gz'

    def build(*args):
        calls.append(args)
        started.set()
        assert finish.wait(5)
        return artifact

    monkeypatch.setattr(
        runtime_kind, 'native_bridge_bundle_target_for_runtime', lambda _: NativeBridgeBundleTarget.LINUX_X86_64_PY310
    )
    monkeypatch.setattr(extension, '_bridge_content_hash', lambda *args: 'test-hash')
    monkeypatch.setattr(extension, '_build_cached_bundle', build)
    first = asyncio.create_task(extension.build_bridge_artifact_cached(object()))
    second = asyncio.create_task(extension.build_bridge_artifact_cached(object()))
    try:
        assert await asyncio.to_thread(started.wait, 5)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        finish.set()
        assert await asyncio.wait_for(second, 5) == artifact
        assert len(calls) == 1
    finally:
        finish.set()
        await asyncio.gather(first, second, return_exceptions=True)


@pytest.mark.asyncio
async def test_host_recovery_preserves_callbacks_and_closes_stale_sessions(monkeypatch):
    from nexus.utils import socketio_proxy

    clients = []

    class Transport:
        def __init__(self, **kwargs):
            self.handlers = {}
            self.connected = False
            self.fail = not clients
            self.shutdown = AsyncMock()
            self.eio = SimpleNamespace(state='disconnected', external_http=False)
            self.eio.http = SimpleNamespace(closed=False, close=AsyncMock())
            clients.append(self)

        def event(self, fn):
            self.handlers[fn.__name__] = fn
            return fn

        def on(self, event, callback):
            self.handlers[event] = callback

        async def connect(self, *args, **kwargs):
            if self.fail:
                raise ConnectionError('handshake failed')
            self.connected = True
            await self.handlers['connect']()

    monkeypatch.setattr(socketio_proxy, 'create_socketio_client', Transport)
    host = extension._load_host_client_class()(
        base_url='http://test', initial_connect_attempts=1, keepalive_interval=None
    )
    received = AsyncMock()
    host.on('llm_request', received)
    with pytest.raises(ConnectionError, match='handshake'):
        await host.ws_connect()
    await host.ws_connect()
    assert len(clients) == 2
    clients[0].shutdown.assert_awaited_once()
    clients[0].eio.http.close.assert_awaited_once()
    await clients[1].handlers['llm_request']({'request_id': 'same'})
    received.assert_awaited_once_with({'request_id': 'same'})
    # Late disconnect from a superseded transport must not start another recovery.
    await clients[0].handlers['disconnect']()
    assert host.ws_connected
    await host.ws_disconnect()
    clients[1].eio.http.close.assert_awaited_once()
    await clients[1].handlers['disconnect']()
    assert host._reconnect_task is None
    assert host._closed
    assert not host.ws_connected


class _Wire:
    """In-memory event transport, including concurrent cancel delivery."""

    def __init__(self):
        self.handlers = {}
        self.peer = None
        self.tasks = set()

    def on(self, event, callback=None):
        def register(fn):
            self.handlers[event] = fn
            return fn

        return register(callback) if callback else register

    def off(self, event, callback=None):
        self.handlers.pop(event, None)

    async def emit(self, event, data):
        callback = self.peer.handlers.get(event)
        if callback:
            task = asyncio.create_task(callback(data))
            self.tasks.add(task)
            task.add_done_callback(self.tasks.discard)

    async def drain(self):
        if self.tasks:
            results = await asyncio.gather(*tuple(self.tasks), return_exceptions=True)
            for result in results:
                if isinstance(result, BaseException) and not isinstance(result, asyncio.CancelledError):
                    raise result


@pytest.mark.asyncio
async def test_extension_roundtrip_cancellation_and_terminal_replay():
    from nexus.core.openai_client import ContentPolicyViolationError
    from nexus.runtime.bridge.adapters.ags_osworld import LLMRequestError, RSIOLLMBridge, RSIOLLMConfig

    host, worker = _Wire(), _Wire()
    host.peer, worker.peer = worker, host
    response = {'id': 'resp-test', 'output': []}
    client = SimpleNamespace(create_responses=AsyncMock(return_value=response))
    handler = extension.create_handler(runtime=host, openai_client=client)
    handler.register()
    bridge = RSIOLLMBridge(worker, RSIOLLMConfig(cancel_timeout=0.1))
    try:
        assert await bridge.request({'_osworld_responses_body': {'input': 'hello'}}) == response
        client.create_responses.assert_awaited_once()
        client.create_responses.side_effect = ContentPolicyViolationError('policy refusal')
        with pytest.raises(LLMRequestError, match='policy refusal'):
            await bridge.request({'_osworld_responses_body': {'input': 'refused'}})

        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def block(body):
            started.set()
            try:
                await asyncio.Future()
            finally:
                cancelled.set()

        client.create_responses.side_effect = block
        pending = asyncio.create_task(bridge.request({'_osworld_responses_body': {'input': 'blocked'}}))
        await asyncio.wait_for(started.wait(), 2)
        request_id = next(iter(bridge._pending))
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        await asyncio.wait_for(cancelled.wait(), 2)
        calls = client.create_responses.await_count
        await handler.handle_llm_request({
            'request_id': request_id,
            'rsio': {'attempt': 2},
            '_osworld_responses_body': {'input': 'blocked'},
        })
        assert client.create_responses.await_count == calls
    finally:
        bridge.close()
        await handler.aclose()
        await host.drain()
        await worker.drain()


@pytest.mark.asyncio
async def test_explicit_host_factory_does_not_replace_default_handler():
    class Transport:
        def __init__(self):
            self.ws_connect = AsyncMock()

        def on(self, *args):
            pass

        def off(self, *args):
            pass

    handlers = []
    for connect in (connect_bridge_llm_handler, extension.connect_bridge_llm_handler, connect_bridge_llm_handler):
        transport = Transport()
        handler = await connect(transport, object())
        handlers.append(handler)
        handler.unregister()
    assert [type(handler) for handler in handlers] == [LLMRequestHandler, OSWorldHandler, LLMRequestHandler]
    assert not hasattr(handlers[0], '_active_request_tasks')
    assert hasattr(handlers[1], '_active_request_tasks')


@pytest.mark.asyncio
async def test_installed_agent_passes_same_extension_to_bundle_and_host(monkeypatch, tmp_path):
    from nexus.extensions.agents.installed_common import native_bridge

    runtime = object()
    client = object()
    monkeypatch.setattr(
        __import__('nexus.core.context', fromlist=['require_context']),
        'require_context',
        lambda: SimpleNamespace(get_runtime=AsyncMock(return_value=runtime), get_openai_client=lambda: client),
    )
    build = AsyncMock(return_value=tmp_path / 'bridge.tar.gz')
    monkeypatch.setattr(extension, 'build_bridge_artifact_cached', build)
    default_build = AsyncMock(side_effect=AssertionError('must use selected bundle'))
    monkeypatch.setattr(native_bridge, 'build_bridge_artifact_cached', default_build)
    launch = SimpleNamespace(base_url='http://test', local_base_url='http://test', headers={})
    adapter = SimpleNamespace(
        upload_artifact=AsyncMock(),
        launch_bridge=AsyncMock(return_value=launch),
        wait_ready=AsyncMock(),
        stop_bridge=AsyncMock(),
    )
    monkeypatch.setattr(native_bridge, '_build_adapter', lambda _: adapter)
    transport = SimpleNamespace(ws_disconnect=AsyncMock())
    monkeypatch.setattr(extension, 'BridgeHostClient', lambda **_: transport)
    handler = SimpleNamespace(
        unregister=lambda: None,
        request_outcomes={},
        last_request_error_code=None,
        last_request_finish_reason=None,
        last_request_left_tool_call_unparsed=False,
    )
    connect = AsyncMock(return_value=handler)
    monkeypatch.setattr(extension, 'connect_bridge_llm_handler', connect)
    result = await extension.call_native_bridge_runtime_agent(
        AsyncMock(return_value={'success': True}),
    )
    assert result['success'] is True
    build.assert_awaited_once_with(runtime)
    default_build.assert_not_awaited()
    assert connect.call_args.args == (transport, client)
    adapter.stop_bridge.assert_awaited_once()
    transport.ws_disconnect.assert_awaited_once()


@pytest.mark.asyncio
async def test_responses_passthrough_is_handled_only_by_osworld():
    response = {'id': 'resp-test', 'output': []}
    client = SimpleNamespace(create_responses=AsyncMock(return_value=response))
    handler = extension.create_handler(runtime=object(), openai_client=client)
    result = await handler._produce_response('req', {'_osworld_responses_body': {'input': 'hello'}}, rsio_mode=False)
    assert result == response
    client.create_responses.assert_awaited_once_with({'input': 'hello'})
    # The ordinary implementation never dispatches this private extension operation.
    assert '_osworld_responses_body' not in LLMRequestHandler._produce_response.__code__.co_consts


@pytest.mark.parametrize('protocol', ['anthropic', 'responses-chat', 'osworld-responses'])
def test_selected_sandbox_protocol_has_extension_bridge(protocol):
    app, bridge = extension.create_bridge_fastapi_app(api_protocol=protocol, launch_id='test')
    assert type(bridge).__module__ == 'nexus.runtime.bridge.adapters.ags_osworld'
    with TestClient(app) as client:
        assert client.get('/health').json()['launch_id'] == 'test'
    with pytest.raises(ValueError, match='Unsupported bridge API protocol'):
        create_bridge_fastapi_app(api_protocol='osworld-responses')


def test_anthropic_stream_succeeds_before_and_after_extension(monkeypatch):
    for factory in (create_bridge_fastapi_app, extension.create_bridge_fastapi_app, create_bridge_fastapi_app):
        app, bridge = factory()
        monkeypatch.setattr(
            bridge,
            'request',
            AsyncMock(
                return_value={
                    'choices': [
                        {
                            'message': {
                                'role': 'assistant',
                                'content': None,
                                'tool_calls': [
                                    {
                                        'id': 'call-1',
                                        'type': 'function',
                                        'function': {'name': 'Bash', 'arguments': '{"command":"echo test"}'},
                                    }
                                ],
                            },
                            'finish_reason': 'tool_calls',
                        }
                    ],
                    'usage': {'prompt_tokens': 10, 'completion_tokens': 5},
                }
            ),
        )
        with TestClient(app) as client:
            response = client.post(
                '/v1/messages',
                json={
                    'model': 'test',
                    'max_tokens': 100,
                    'stream': True,
                    'messages': [{'role': 'user', 'content': 'hello'}],
                },
            )
        assert response.status_code == 200
        events = [
            json.loads(line[6:])
            for line in response.text.splitlines()
            if line.startswith('data: ') and line != 'data: [DONE]'
        ]
        assert not any(event['type'] == 'error' for event in events)
        assert any(
            event['type'] == 'content_block_start' and event['content_block'].get('name') == 'Bash' for event in events
        )


def test_responses_chat_controls_are_scoped_to_extension(monkeypatch):
    observed = []
    for selected, protocol in ((None, 'responses'), (EXTENSION, 'responses-chat'), (None, 'responses')):
        if selected:
            app, bridge = extension.create_bridge_fastapi_app(api_protocol=protocol)
        else:
            # Main's ResponsesProxy initializes its bridge at service startup.
            from nexus.runtime.server.services.responses_proxy.server import ResponsesProxy
            from nexus.runtime.server.services.rsio_llm_bridge import RSIOLLMBridge

            bridge = RSIOLLMBridge(_Wire())
            proxy = ResponsesProxy(sio=None)
            proxy.bridge = bridge
            app = proxy.create_app()
        backend = AsyncMock(
            return_value={
                'choices': [{'message': {'role': 'assistant', 'content': 'hello'}, 'finish_reason': 'stop'}],
                'usage': {'prompt_tokens': 1, 'completion_tokens': 1},
            }
        )
        monkeypatch.setattr(bridge, 'request', backend)
        with TestClient(app) as client:
            response = client.post(
                '/v1/responses',
                json={
                    'model': 'test',
                    'input': 'hello',
                    'reasoning': {'effort': 'none'},
                    'stream': False,
                },
            )
        assert response.status_code == 200, response.text
        observed.append(backend.call_args.args[0].get('reasoning_effort'))
    assert observed == ['none', None, 'none']


def test_native_anthropic_and_responses_roundtrip(monkeypatch):
    native = {
        'id': 'msg-native',
        'type': 'message',
        'role': 'assistant',
        'model': 'test',
        'content': [{'type': 'text', 'text': 'hello'}],
        'stop_reason': 'end_turn',
        'stop_sequence': None,
        'usage': {'input_tokens': 1, 'output_tokens': 1},
    }
    app, bridge = extension.create_bridge_fastapi_app()
    monkeypatch.setattr(
        bridge,
        'request',
        AsyncMock(
            return_value={
                'provider_specific_fields': {'native_response': {'protocol': 'anthropic', 'body': native}},
            }
        ),
    )
    with TestClient(app) as client:
        response = client.post(
            '/v1/messages',
            json={
                'model': 'test',
                'max_tokens': 100,
                'messages': [{'role': 'user', 'content': 'hello'}],
            },
        )
    assert response.json() == native
    app, bridge = extension.create_bridge_fastapi_app(api_protocol='osworld-responses')
    native_response = {'id': 'resp-native', 'status': 'completed', 'output': []}
    backend = AsyncMock(return_value=native_response)
    monkeypatch.setattr(bridge, 'request', backend)
    with TestClient(app) as client:
        response = client.post('/v1/responses', json={'model': 'test', 'input': 'hello'})
    assert response.status_code == 200
    assert response.json()['id'] == 'resp-native'
    assert backend.call_args.args[0] == {'_osworld_responses_body': {'model': 'test', 'input': 'hello'}}


def test_bundle_contains_extension_without_task_registration(tmp_path, monkeypatch):
    target = NativeBridgeBundleTarget.LINUX_X86_64_PY310
    plain = tmp_path / 'plain.tar.gz'
    selected = tmp_path / 'selected.tar.gz'
    for output, builder in ((plain, build_app_bundle), (selected, extension.build_app_bundle)):
        builder(output=output, target=target, default_port=9281, include_vendor=False)
    assert _bridge_content_hash(target, 9281) != extension._bridge_content_hash(target, 9281)
    with tarfile.open(plain) as archive:
        assert not any('/tasks/os_world/' in name for name in archive.getnames())
        launcher = archive.extractfile('nexus-llm-bridge/bin/launch_bridge.sh').read().decode()
        assert '-m nexus.runtime.bridge --host' in launcher
        assert 'setsid' not in launcher
    with tarfile.open(selected) as archive:
        archive.extractall(tmp_path / 'extract', filter='data')
    root = tmp_path / 'extract' / 'nexus-llm-bridge'
    assert not (root / 'src/nexus/extensions/tasks/os_world').exists()
    assert 'setsid python3 -m ' + EXTENSION in (root / 'bin/launch_bridge.sh').read_text()
    assert EXTENSION in (root / 'bin/launch_bridge.ps1').read_text()
    subprocess.run(['bash', '-n', str(root / 'bin/launch_bridge.sh')], check=True)
    # Use installed third-party dependencies but load nexus exclusively from the bundle.
    code = """
import sys
sys.path.insert(0, sys.argv[1])
from nexus.runtime.bridge.app import create_bridge_fastapi_app
app, bridge = create_bridge_fastapi_app()
assert app.state.bridge is bridge
from nexus.runtime.bridge.adapters.ags_osworld import create_bridge_fastapi_app
for protocol in ('anthropic', 'responses-chat', 'osworld-responses'):
    app, bridge = create_bridge_fastapi_app(api_protocol=protocol)
    assert app.state.bridge is bridge
assert 'nexus.extensions.tasks.os_world.task' not in sys.modules
assert 'nexus.extensions.tools.desktop.tools' not in sys.modules
assert 'nexus.runtime.providers.ags.osworld' not in sys.modules
assert 'nexus.core.openai_client' not in sys.modules
assert 'nexus.runtime.server.services.llm_handler' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, '-I', '-c', code, str(root / 'src'), EXTENSION],
        cwd=tmp_path,
        env={**os.environ, 'PYTHONPATH': ''},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    from nexus.runtime.bridge import artifacts

    # The assembly is now in the shared runtime/bridge tree: both hashes must change.
    monkeypatch.setattr(artifacts, '_repo_root', lambda: root / 'src')
    default_hash = _bridge_content_hash(target, 9281)
    extension_hash = extension._bridge_content_hash(target, 9281)
    source = root / 'src/nexus/runtime/bridge/adapters/ags_osworld.py'
    source.write_text(source.read_text() + '\n# cache invalidation test\n')
    assert _bridge_content_hash(target, 9281) != default_hash
    assert extension._bridge_content_hash(target, 9281) != extension_hash

    extension_hash = extension._bridge_content_hash(target, 9281)
    source = root / 'src/nexus/utils/async_http.py'
    source.write_text(source.read_text() + '\n# HTTP client cache invalidation test\n')
    assert extension._bridge_content_hash(target, 9281) != extension_hash


def test_adapter_is_single_module_and_uses_shared_execution():
    from pathlib import Path

    from nexus.runtime.bridge.adapters.ags_osworld import (
        RPCReplayGuard,
        RSIOLLMBridge,
    )
    from nexus.runtime.server.rpc_replay_guard import RPCReplayGuard as DefaultGuard
    from nexus.runtime.server.services.rsio_llm_bridge import RSIOLLMBridge as DefaultBridge

    assert Path(extension.__file__).name == 'ags_osworld.py'
    root = Path(__file__).resolve().parents[5]
    assert not list((root / 'nexus/extensions/tasks/os_world/bridge').glob('*.py'))
    assert OSWorldHandler._run_heartbeat is LLMRequestHandler._run_heartbeat
    assert OSWorldHandler._inject_reasoning_items is LLMRequestHandler._inject_reasoning_items
    assert issubclass(RPCReplayGuard, DefaultGuard)
    assert issubclass(RSIOLLMBridge, DefaultBridge)
    assert RSIOLLMBridge._drive is DefaultBridge._drive
    assert RSIOLLMBridge.close is DefaultBridge.close


def test_proxy_extension_does_not_change_default_conversion_logger():
    from nexus.runtime.server.services.anthropic_proxy import converter as default

    before = default.conversion_logger.level
    extension.create_anthropic_app(bridge=SimpleNamespace(), conversion_log_level='ERROR')
    assert default.conversion_logger.level == before


@pytest.mark.parametrize('stream', [False, True])
def test_context_error_policy_is_scoped_to_osworld(stream):
    from nexus.runtime.server.services.anthropic_proxy.server import AnthropicProxy

    body = {'model': 'test', 'max_tokens': 10, 'messages': [{'role': 'user', 'content': 'hello'}], 'stream': stream}
    for selected in (False, True):
        bridge = SimpleNamespace(request=AsyncMock(side_effect=RuntimeError('prompt is too long')))
        if selected:
            app = extension.create_anthropic_app(bridge=bridge)
        else:
            proxy = AnthropicProxy(sio=None)
            proxy.bridge = bridge
            app = proxy.create_app()
        with TestClient(app) as client:
            response = client.post('/v1/messages', json=body)
        if stream:
            assert response.status_code == 200
            assert ('invalid_request_error' if selected else 'api_error') in response.text
        else:
            assert response.status_code == (400 if selected else 500)


@pytest.mark.asyncio
async def test_default_and_adapter_completion_share_parameter_contract():
    from nexus.core.openai_client import OpenAIConfig

    calls = []
    for handler_type in (LLMRequestHandler, OSWorldHandler):
        backend = SimpleNamespace(
            config=OpenAIConfig(client_args={}, request_args={'model': 'host-model', 'reasoning_effort': 'high'}),
            create_completion=AsyncMock(
                return_value={'choices': [{'message': {'role': 'assistant', 'content': 'ok'}}]}
            ),
        )
        handler = handler_type(runtime=SimpleNamespace(), openai_client=backend)
        await handler._produce_response(
            'same-request',
            {
                'model': 'harness-model',
                'reasoning_effort': 'low',
                'max_tokens': 123,
                'messages': [{'role': 'user', 'content': 'hello'}],
            },
            rsio_mode=False,
        )
        calls.append(backend.create_completion.call_args.kwargs)
    assert calls[0] == calls[1]
    assert calls[0]['max_tokens'] == 123


def test_reasoning_snapshot_is_not_aliased_to_provider_response():
    handler = OSWorldHandler(runtime=SimpleNamespace(), openai_client=SimpleNamespace())
    output = [{'type': 'reasoning', 'id': 'rs1', 'encrypted_content': 'opaque'}]
    response = {
        'choices': [
            {
                'message': {
                    'role': 'assistant',
                    'content': 'answer',
                    'reasoning_items': output,
                    'provider_specific_fields': {'responses_output_items': output},
                }
            }
        ]
    }
    handler._cache_reasoning_items_from_response(response, frontier_id='turn1')
    output[0]['encrypted_content'] = 'mutated'
    assert handler._reasoning_items_cache[0]['output_items'][0]['encrypted_content'] == 'opaque'


@pytest.mark.parametrize('stream', [False, True])
def test_native_responses_preserves_computer_calls_and_reasoning(stream):
    from fastapi import FastAPI

    request = {
        'model': 'native',
        'stream': stream,
        'tools': [{'type': 'computer_use_preview', 'display_width': 1280, 'display_height': 720}],
        'input': [
            {
                'type': 'computer_call_output',
                'call_id': 'call1',
                'output': {'type': 'computer_screenshot', 'image_url': 'data:image/png;base64,TEST'},
            }
        ],
    }
    output = [
        {'type': 'reasoning', 'id': 'rs1', 'encrypted_content': 'opaque'},
        {
            'type': 'computer_call',
            'id': 'item1',
            'call_id': 'call2',
            'action': {'type': 'click', 'x': 100, 'y': 200},
            'pending_safety_checks': [{'id': 'check1'}],
        },
    ]
    backend = SimpleNamespace(request=AsyncMock(return_value={'id': 'resp1', 'status': 'completed', 'output': output}))
    app = FastAPI()
    extension.attach_responses_endpoint(app, backend, timeout=20)
    with TestClient(app) as client:
        response = client.post('/v1/responses', json=request)
    assert response.status_code == 200
    assert backend.request.call_args.args[0] == {'_osworld_responses_body': request}
    if stream:
        events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: ')]
        terminal = next(event for event in events if event['type'] == 'response.completed')
        assert terminal['response']['output'] == output
    else:
        assert response.json()['output'] == output


def test_namespace_flattening_is_request_scoped():
    body = {'tools': [{'type': 'namespace', 'name': 'desktop', 'tools': [{'name': 'click'}]}]}
    output = [
        {'type': 'function_call', 'namespace': 'desktop', 'name': 'click', 'call_id': 'one'},
        {'type': 'function_call', 'namespace': 'other', 'name': 'click', 'call_id': 'two'},
    ]
    result = extension.flatten_calls({'output': output}, body)
    assert result['output'][0]['name'] == 'desktop__click'
    assert 'namespace' not in result['output'][0]
    assert result['output'][1] == output[1]
    assert output[0]['name'] == 'click'
