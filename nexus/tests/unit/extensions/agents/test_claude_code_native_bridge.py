from __future__ import annotations

import inspect

from pathlib import Path
from typing import Any

import pytest

from nexus.extensions.agents.claude_code.agent import (
    ClaudeCodeInput,
    ClaudeCodeNativeBridgeInput,
    run_claude_code_native_bridge,
)
from nexus.extensions.agents.claude_code.native_bridge.agent_runtime import setup_and_run_native_bridge_claude_code
from nexus.extensions.agents.installed_common.native_bridge import (
    BRIDGE_READINESS_TIMEOUT,
    call_native_bridge_runtime_agent,
)
from nexus.runtime.bridge.adapters.base import BridgeLaunchOptions, BridgeLaunchResult


class FakeRuntimeHandle:
    headers = {'Authorization': 'Bearer fake'}

    def url(self, port: int, path: str = '') -> str:
        return f'http://bridge.example:{port}{path}'


class FakeRuntime:
    handle = FakeRuntimeHandle()
    server_port = 5000


class FakeContext:
    def __init__(self, runtime: Any, openai_client: Any) -> None:
        self.runtime = runtime
        self.openai_client = openai_client
        self.runtime_ids: list[str] = []

    async def get_runtime(self, name: str = 'main') -> Any:
        self.runtime_ids.append(name)
        return self.runtime

    def get_openai_client(self, name: str = 'main') -> Any:
        assert name == 'main'
        return self.openai_client


class FakeAdapter:
    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime
        self.events: list[str] = []

    async def upload_artifact(self, artifact_path: Path) -> None:
        self.events.append(f'upload:{artifact_path.name}')

    async def launch_bridge(self, options: BridgeLaunchOptions | None = None) -> BridgeLaunchResult:
        port = 8765
        launch_options = options or BridgeLaunchOptions()
        self.events.append(
            f'launch:{port}:{launch_options.conversion_log_level}:'
            f'{launch_options.thinking_config}:{launch_options.request_timeout}:{launch_options.api_protocol}'
        )
        return BridgeLaunchResult(
            base_url=f'http://bridge.example:{port}',
            local_base_url=f'http://127.0.0.1:{port}',
            headers={'X-Bridge-Token': 'from-launch'},
        )

    async def wait_ready(self, launch: BridgeLaunchResult, timeout: float = 60.0) -> None:
        self.events.append(f'ready:{timeout}')

    async def stop_bridge(self) -> None:
        self.events.append('stop')


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('api_protocol', 'pass_bridge_client', 'selected_extension', 'ags_deployment', 'force_stop'),
    [
        ('anthropic', False, False, False, False),
        ('anthropic', False, False, True, False),
        ('responses', False, False, True, False),
        ('anthropic', False, True, True, False),
        ('anthropic', True, True, True, True),
        ('anthropic', True, True, False, False),
        ('osworld-responses', True, True, True, False),
        ('osworld-responses', False, True, False, True),
        ('responses-chat', False, True, True, False),
        ('responses-chat', True, True, False, False),
        ('responses-chat', True, True, True, True),
        ('osworld-anthropic', False, True, True, False),
    ],
)
async def test_call_native_bridge_runtime_agent_wraps_bridge_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    pass_bridge_client: bool,
    api_protocol: str,
    selected_extension: bool,
    ags_deployment: bool,
    force_stop: bool,
) -> None:
    from nexus.runtime.bridge.adapters import ags_osworld

    artifact = tmp_path / 'bridge.tar.gz'
    artifact.write_bytes(b'fake bridge artifact')
    runtime = FakeRuntime()
    from nexus.runtime.bridge.runtime_kind import NativeBridgeRuntimeKind

    monkeypatch.setattr(
        'nexus.extensions.agents.installed_common.native_bridge.classify_native_bridge_runtime',
        lambda _rt: NativeBridgeRuntimeKind.AGS_OSWORLD,
    )
    ctx = FakeContext(runtime, openai_client=object())
    adapter_holder: dict[str, Any] = {}

    monkeypatch.setattr('nexus.extensions.agents.installed_common.native_bridge.require_context', lambda: ctx)
    monkeypatch.setattr('nexus.core.context.require_context', lambda: ctx)

    async def fake_build_bridge_artifact_cached(runtime_arg: Any, **kwargs) -> Path:
        assert runtime_arg is runtime
        assert kwargs == {}
        return artifact

    monkeypatch.setattr(
        'nexus.extensions.agents.installed_common.native_bridge.build_bridge_artifact_cached',
        fake_build_bridge_artifact_cached,
    )
    monkeypatch.setattr(ags_osworld, 'build_bridge_artifact_cached', fake_build_bridge_artifact_cached)

    class FakeOSWorldAdapter(FakeAdapter, ags_osworld.AGSOSWorldBridgeAdapter):
        pass

    def fake_build_adapter(runtime_arg: Any) -> FakeAdapter:
        assert runtime_arg is runtime
        adapter = (FakeOSWorldAdapter if ags_deployment else FakeAdapter)(runtime_arg)
        adapter_holder['adapter'] = adapter
        return adapter

    monkeypatch.setattr('nexus.extensions.agents.installed_common.native_bridge._build_adapter', fake_build_adapter)

    class FakeBridgeHostClient:
        def __init__(
            self,
            *,
            base_url: str,
            headers: dict[str, str] | None,
            request_timeout: float = 60.0,
            initial_connect_attempts: int | None = None,
            trust_env: bool = True,
        ) -> None:
            self.initial_connect_attempts = initial_connect_attempts
            self.trust_env = trust_env
            self.base_url = base_url
            self.headers = headers
            self.request_timeout = request_timeout
            self.ws_connected = False
            self.disconnected = False
            adapter_holder['client'] = self

        async def ws_connect(self) -> None:
            self.ws_connected = True

        async def ws_disconnect(self) -> None:
            self.disconnected = True
            self.ws_connected = False

    monkeypatch.setattr('nexus.extensions.agents.installed_common.native_bridge.BridgeHostClient', FakeBridgeHostClient)
    monkeypatch.setattr(ags_osworld, 'BridgeHostClient', FakeBridgeHostClient)

    class FakeHandler:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.registered = False
            self.unregistered = False
            self.request_outcomes = {'request-1': None}
            self.last_request_error_code = None
            self.last_request_finish_reason = 'stop'
            self.last_request_left_tool_call_unparsed = False

        def register(self) -> None:
            self.registered = True

        def unregister(self) -> None:
            self.unregistered = True

    monkeypatch.setattr('nexus.runtime.server.services.llm_handler.LLMRequestHandler', FakeHandler)
    selected_handler_calls = []

    def create_selected_handler(**kwargs):
        selected_handler_calls.append(kwargs)
        return FakeHandler(**kwargs)

    monkeypatch.setattr(ags_osworld, 'create_handler', create_selected_handler)

    async def target(**kwargs: Any) -> dict[str, Any]:
        assert kwargs['runtime'] is runtime
        assert kwargs['launch'].local_base_url == 'http://127.0.0.1:8765'
        assert kwargs['timeout'] == 0.125
        assert 'adapter' not in kwargs
        assert kwargs['task'] == 'hello'
        if pass_bridge_client and selected_extension:
            assert kwargs['bridge_client'] is adapter_holder['client']
        else:
            assert 'bridge_client' not in kwargs
        return {'success': True, 'output': 'ok', 'stderr': '', 'exit_code': 0, 'execution_time': 1.0}

    run = ags_osworld.call_native_bridge_runtime_agent if selected_extension else call_native_bridge_runtime_agent
    result = await run(
        target,
        timeout=0.125,
        conversion_log_level='DEBUG',
        thinking_config='bedrock-claude',
        task='hello',
        api_protocol=api_protocol,
        **(
            {
                'pass_bridge_client': pass_bridge_client,
                'stop_bridge_after_run': force_stop,
                'bridge_request_timeout': 300.0 if api_protocol.startswith('osworld-') else None,
            }
            if selected_extension
            else {}
        ),
    )

    assert result['success'] is True
    assert result['llm_request_outcomes'] == {'request-1': None}
    assert result['last_llm_error_code'] is None
    assert result['last_llm_finish_reason'] == 'stop'
    assert result['last_llm_unparsed_tool_call'] is False
    assert ctx.runtime_ids == ['main']
    selected_transport = selected_extension and (
        api_protocol.startswith('osworld-') or (api_protocol == 'responses-chat' and ags_deployment)
    )
    assert adapter_holder['adapter'].events == [
        'upload:bridge.tar.gz',
        f'launch:8765:DEBUG:bedrock-claude:{300.0 if selected_extension and api_protocol.startswith("osworld-") else 0.125}:{api_protocol}',
        f'ready:{90.0 if selected_transport else BRIDGE_READINESS_TIMEOUT}',
        # Unselected runtimes keep main's lifecycle, even for AGS deployment.
    ] + (['stop'] if not selected_extension or force_stop or not ags_deployment else [])
    assert len(selected_handler_calls) == int(selected_extension)
    assert adapter_holder['client'].headers == {'X-Bridge-Token': 'from-launch'}
    assert adapter_holder['client'].request_timeout == 60.0
    assert adapter_holder['client'].initial_connect_attempts == (1 if selected_transport else None)


@pytest.mark.asyncio
async def test_claude_code_native_bridge_agent_calls_common_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    async def fake_call_native_bridge_runtime_agent(runtime_agent: Any, **kwargs: Any) -> dict[str, Any]:
        captured['runtime_agent'] = runtime_agent
        captured.update(kwargs)
        return {'success': True, 'output': 'agent-output', 'stderr': '', 'exit_code': 0, 'execution_time': 2.5}

    monkeypatch.setattr(
        'nexus.extensions.agents.claude_code.agent.call_native_bridge_runtime_agent',
        fake_call_native_bridge_runtime_agent,
    )

    output = await run_claude_code_native_bridge(
        ClaudeCodeNativeBridgeInput(
            user_prompt='write quicksort',
            working_dir='/tmp/work',
            timeout=3600,
            extra_cli_args=['--output-format', 'json'],
            pre_start_scripts=[{'interpreter': 'bash', 'code': 'echo hi'}],
            mcp_servers=[{'name': 'files', 'command': 'python', 'args': ['server.py']}],
            conversion_log_level='DEBUG',
            thinking_preset='bedrock-claude',
        )
    )

    assert output.success is True
    assert output.output == 'agent-output'
    assert 'native_bridge' not in captured
    assert captured['task'] == 'write quicksort'
    assert captured['working_dir'] == '/tmp/work'
    assert captured['extra_cli_args'] == ['--output-format', 'json']
    assert captured['pre_start_scripts'] == [
        {'interpreter': 'bash', 'code': 'echo hi', 'timeout': 300.0, 'cwd': None, 'env': None, 'fail_on_error': True}
    ]
    assert captured['mcp_servers'] == [{'name': 'files', 'command': 'python', 'args': ['server.py']}]
    assert captured['conversion_log_level'] == 'DEBUG'
    assert captured['thinking_config'] == 'bedrock-claude'
    assert captured['api_protocol'] == 'anthropic'


def test_native_bridge_input_accepts_powershell_pre_start_script() -> None:
    native_input = ClaudeCodeNativeBridgeInput(
        user_prompt='write quicksort',
        working_dir=r'C:\work',
        pre_start_scripts=[{'interpreter': 'powershell', 'code': "Set-Content answer.txt 'ok'"}],
    )

    assert native_input.pre_start_scripts[0].interpreter == 'powershell'
    with pytest.raises(ValueError, match="Input should be 'bash' or 'python'"):
        ClaudeCodeInput(
            user_prompt='write quicksort',
            working_dir='/tmp/work',
            pre_start_scripts=[{'interpreter': 'powershell', 'code': "Set-Content answer.txt 'ok'"}],
        )


def test_native_bridge_input_narrows_exposed_fields() -> None:
    # Fields intentionally dropped from the native bridge input surface.
    forbidden = {
        'claude_path',
        'claude_install_dir',
        'config_file_patches',
        'native_bridge',
        'action_event_file',
    }
    assert forbidden.isdisjoint(ClaudeCodeNativeBridgeInput.model_fields)
    supported = {'mcp_servers', 'binary_url', 'binary_checksum'}
    assert supported <= ClaudeCodeNativeBridgeInput.model_fields.keys()
    with pytest.raises(ValueError, match='Extra inputs are not permitted'):
        ClaudeCodeNativeBridgeInput(
            user_prompt='write quicksort',
            working_dir='/tmp/work',
            claude_install_dir='/tmp/claude',
        )


def test_native_bridge_runtime_signature_does_not_expose_unsupported_setup_fields() -> None:
    unsupported = {
        'adapter',
        'config_file_patches',
        'action_event_file',
    }
    parameters = inspect.signature(setup_and_run_native_bridge_claude_code).parameters
    assert unsupported.isdisjoint(parameters)
    assert {'mcp_servers', 'binary_url', 'binary_checksum'} <= parameters.keys()
    assert 'pre_start_scripts' in parameters  # supported
    assert all(parameter.kind != inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())


@pytest.mark.asyncio
async def test_native_bridge_runtime_converts_run_exception_to_agent_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingClaudeAdapter:
        async def ensure_claude(self, spec: Any) -> str:
            return '/usr/bin/claude'

        async def run_pre_start_scripts(self, *args: Any, **kwargs: Any) -> None:
            return None

        async def run_claude(self, request: Any) -> Any:
            raise RuntimeError('sandbox connection lost')

    monkeypatch.setattr(
        'nexus.extensions.agents.claude_code.native_bridge.agent_runtime.build_claude_native_bridge_adapter',
        lambda runtime: FailingClaudeAdapter(),
    )

    result = await setup_and_run_native_bridge_claude_code(
        runtime=object(),
        launch=BridgeLaunchResult(
            base_url='http://bridge.example:8765',
            local_base_url='http://127.0.0.1:8765',
        ),
        task='hello',
    )

    assert result['success'] is False
    assert result['exit_code'] == -1
    assert result['output'] == ''
    assert result['stderr'] == 'Failed to run Claude Code: sandbox connection lost'
