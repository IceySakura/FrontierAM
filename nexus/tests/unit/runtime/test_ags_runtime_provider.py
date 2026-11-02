import asyncio

from unittest.mock import AsyncMock

import pytest

from nexus.runtime import InvalidPersistentRuntimeError
from nexus.runtime.providers.ags import AGSOSWorldRuntime, AGSRawOSWorldRuntime, AGSRuntimeProvider
from nexus.runtime.providers.ags import provider as ags_provider_module
from nexus.runtime.providers.ags.handle import AGSInstanceHandle
from nexus.runtime.providers.ags.waa import AGSWindowsAgentArenaRuntime


def _make_provider() -> AGSRuntimeProvider:
    return AGSRuntimeProvider(
        secret_id='test-secret-id',
        secret_key='test-secret-key',
        role_arn='qcs::cam::uin/100000000000:roleName/TestAGSRole',
        image='ccr.ccs.tencentyun.com/demo/nexus-runtime:latest',
    )


def test_ags_provider_allows_template_mode_without_control_plane_credentials() -> None:
    provider = AGSRuntimeProvider(api_key='test-api-key', template='osworld-template')

    assert provider.api_key == 'test-api-key'
    assert provider.template == 'osworld-template'


def test_ags_provider_requires_some_credentials() -> None:
    with pytest.raises(ValueError, match='Either secret_id/secret_key or api_key'):
        AGSRuntimeProvider()


def test_ags_raw_interface_mode_defaults_code_runtime_to_native() -> None:
    provider = _make_provider()

    assert provider._infer_launch_mode({'runtime_kind': 'code', 'interface_mode': 'raw'}) == 'native'


def test_ags_nexus_interface_mode_defaults_template_code_runtime_to_native() -> None:
    """In template provision mode, runtime_kind=code must resolve to native.

    A bare code-interpreter template (native /execute on 49999) has no Nexus
    server; misrouting it to nexus_server made the provider health-check a
    non-existent server on port 8000 and hang. Template mode is detected from
    template + api_key, mirroring the user-facing `--template ... --api-key`.
    """
    provider = AGSRuntimeProvider(api_key='test-api-key', template='code-template')

    assert provider._infer_launch_mode({'runtime_kind': 'code', 'interface_mode': 'nexus'}) == 'native'


def test_ags_control_plane_code_runtime_still_defaults_to_nexus_server() -> None:
    """In control_plane mode, runtime_kind=code stays nexus_server (full Nexus runtime).

    This is the valid `nexus_server | nexus | code -> RemoteRuntime` path and must
    not regress when fixing the template-mode routing above.
    """
    provider = _make_provider()

    assert provider._infer_launch_mode({'runtime_kind': 'code', 'interface_mode': 'nexus'}) == 'nexus_server'


def test_ags_osworld_is_a_builtin_tool_type() -> None:
    """runtime_kind=osworld resolves to the built-in 'osworld' tool on native launch.

    AGS exposes osworld as a built-in tool type (like browser/code-interpreter),
    so control_plane can create it with only secret_id/secret_key — no role_arn or
    custom image. Verified end-to-end against the live control plane.
    """
    provider = _make_provider()

    launch_mode = provider._infer_launch_mode({'runtime_kind': 'osworld'})
    assert launch_mode == 'native'

    tool_type = provider._infer_tool_type({'runtime_kind': 'osworld'}, launch_mode)
    assert tool_type == 'osworld'

    # ensure_builtin must accept osworld and not raise
    provider._validate_tool_selection(launch_mode='native', tool_type='osworld', strategy='ensure_builtin')


def test_ags_raw_interface_mode_rejects_nexus_server() -> None:
    provider = _make_provider()

    with pytest.raises(ValueError, match='interface_mode="raw" requires launch_mode="native"'):
        provider._validate_interface_selection(launch_mode='nexus_server', interface_mode='raw')


def test_ags_nexus_server_config_builds_default_custom_runtime() -> None:
    provider = _make_provider()

    custom_configuration = provider._build_custom_configuration({}, 'nexus_server')
    provider._validate_custom_configuration(custom_configuration, launch_mode='nexus_server')

    assert custom_configuration['Image'] == 'ccr.ccs.tencentyun.com/demo/nexus-runtime:latest'
    assert custom_configuration['Command'] == ['/nix/runtime/nexus/deployment/runtime/start.sh']
    assert custom_configuration['Ports'] == [{'Name': 'runtime', 'Port': 8000, 'Protocol': 'TCP'}]
    assert custom_configuration['Probe']['HttpGet'] == {'Path': '/health', 'Port': 8000, 'Scheme': 'HTTP'}


@pytest.mark.parametrize(
    ('custom_configuration', 'expected_pattern'),
    [
        (
            {
                'Image': 'ccr.ccs.tencentyun.com/demo/osworld:latest',
                'Ports': [{'Name': 'server', 'Port': 5000, 'Protocol': 'TCP'}],
                'Probe': {'HttpGet': {'Path': '/health', 'Port': 5000, 'Scheme': 'HTTP'}},
            },
            'requires Command',
        ),
        (
            {
                'Image': 'ccr.ccs.tencentyun.com/demo/osworld:latest',
                'Command': ['/start.sh'],
                'Ports': [{'Name': 'server', 'Port': 5000, 'Protocol': 'TCP'}],
            },
            'requires Probe',
        ),
        (
            {
                'Image': 'ccr.ccs.tencentyun.com/demo/osworld:latest',
                'Command': ['/start.sh'],
                'Probe': {'HttpGet': {'Path': '/health', 'Port': 5000, 'Scheme': 'HTTP'}},
            },
            'requires Ports',
        ),
    ],
)
def test_ags_native_custom_runtime_requires_command_probe_and_ports(
    custom_configuration: dict[str, object],
    expected_pattern: str,
) -> None:
    provider = _make_provider()

    with pytest.raises(ValueError, match=expected_pattern):
        provider._validate_custom_configuration(custom_configuration, launch_mode='native')


@pytest.mark.parametrize(
    ('probe', 'expected_pattern'),
    [
        (
            {
                'HttpGet': {'Path': '/health', 'Port': 5000, 'Scheme': 'HTTP'},
                'ReadyTimeoutMs': 30001,
            },
            'ReadyTimeoutMs',
        ),
        (
            {
                'HttpGet': {'Path': '/health', 'Port': 5000, 'Scheme': 'HTTP'},
                'FailureThreshold': 101,
            },
            'FailureThreshold',
        ),
    ],
)
def test_ags_probe_limits_are_validated_early(probe: dict[str, object], expected_pattern: str) -> None:
    provider = _make_provider()

    with pytest.raises(ValueError, match=expected_pattern):
        provider._validate_custom_configuration(
            {
                'Image': 'ccr.ccs.tencentyun.com/demo/osworld:latest',
                'Command': ['/start.sh'],
                'Ports': [{'Name': 'server', 'Port': 5000, 'Protocol': 'TCP'}],
                'Probe': probe,
            },
            launch_mode='native',
        )


@pytest.mark.parametrize(
    ('launch_mode', 'tool_type', 'strategy', 'expected_pattern'),
    [
        ('nexus_server', 'browser', 'ensure_builtin', 'launch_mode="nexus_server"'),
        ('native', 'custom', 'ensure_builtin', 'ensure_builtin'),
        ('native', 'browser', 'ensure_custom', 'ensure_custom'),
    ],
)
def test_ags_tool_selection_rejects_invalid_combinations(
    launch_mode: str,
    tool_type: str,
    strategy: str,
    expected_pattern: str,
) -> None:
    provider = _make_provider()

    with pytest.raises(ValueError, match=expected_pattern):
        provider._validate_tool_selection(launch_mode=launch_mode, tool_type=tool_type, strategy=strategy)


@pytest.mark.parametrize('auth_mode', ['PUBLIC', 'NONE'])
@pytest.mark.asyncio
async def test_ags_acquire_skips_token_for_tokenless_auth_modes(
    monkeypatch: pytest.MonkeyPatch,
    auth_mode: str,
) -> None:
    provider = _make_provider()
    provider.configure('main', auth_mode=auth_mode)
    runtime = object()

    async def _fake_ensure_tool(*args: object, **kwargs: object) -> tuple[dict[str, str], bool]:
        return {'ToolId': 'tool-123', 'ToolName': 'tool', 'ToolType': 'custom'}, False

    async def _fake_start_instance(*args: object, **kwargs: object) -> dict[str, str]:
        return {'InstanceId': 'instance-123'}

    async def _fail_acquire_token(*args: object, **kwargs: object) -> str:
        pytest.fail('_acquire_token must not be called for tokenless auth modes')

    async def _fake_create_runtime(handle: object, *args: object, **kwargs: object) -> object:
        assert handle.access_token == ''
        return runtime

    monkeypatch.setattr(provider, '_ensure_tool', _fake_ensure_tool)
    monkeypatch.setattr(provider, '_start_instance', _fake_start_instance)
    monkeypatch.setattr(provider, '_acquire_token', _fail_acquire_token)
    monkeypatch.setattr(provider, '_create_runtime', _fake_create_runtime)

    assert await provider.acquire('main') is runtime


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('runtime_kind', 'runtime_type', 'ready_method'),
    [
        ('osworld', AGSOSWorldRuntime, '_wait_for_osworld_ready'),
        ('waa', AGSWindowsAgentArenaRuntime, '_wait_for_waa_ready'),
    ],
)
async def test_ags_reuses_running_instance_without_starting_or_stopping_it(
    monkeypatch: pytest.MonkeyPatch,
    runtime_kind: str,
    runtime_type: type,
    ready_method: str,
) -> None:
    provider = _make_provider()
    provider.configure(
        'main',
        launch_mode='native',
        runtime_kind=runtime_kind,
        tool_type=runtime_kind,
        interface_mode='nexus',
        runtime_credential={'instance_id': 'sandbox-persistent-123'},
    )
    calls: list[tuple[str, dict[str, object]]] = []

    async def fake_call_api(action: str, payload: dict[str, object]) -> dict[str, object]:
        calls.append((action, payload))
        if action == 'DescribeSandboxInstanceList':
            return {
                'InstanceSet': [
                    {
                        'InstanceId': 'sandbox-persistent-123',
                        'ToolId': 'tool-persistent',
                        'ToolName': f'persistent-{runtime_kind}',
                        'Status': 'RUNNING',
                        'Persistent': True,
                    }
                ]
            }
        if action == 'AcquireSandboxInstanceToken':
            return {'Token': 'refreshed-access-token'}
        raise AssertionError(f'unexpected AGS action: {action}')

    monkeypatch.setattr(provider, '_call_api', fake_call_api)
    monkeypatch.setattr(provider, ready_method, AsyncMock())
    monkeypatch.setattr(provider, '_close_runtime', AsyncMock())

    runtime = await provider.acquire('main')

    assert isinstance(runtime, runtime_type)
    assert runtime.handle.instance_id == 'sandbox-persistent-123'
    assert runtime.handle.access_token == 'refreshed-access-token'
    assert provider.get_runtime_id('main') == 'sandbox-persistent-123'
    assert calls == [
        (
            'DescribeSandboxInstanceList',
            {'InstanceIds': ['sandbox-persistent-123'], 'Offset': 0, 'Limit': 1},
        ),
        ('AcquireSandboxInstanceToken', {'InstanceId': 'sandbox-persistent-123'}),
    ]

    await provider.release('main')

    assert calls[-1][0] == 'AcquireSandboxInstanceToken'
    assert provider.get_runtime_id('main') is None
    assert provider.get_active_runtimes() == {}


@pytest.mark.asyncio
async def test_ags_reused_instance_must_be_running_before_token_refresh(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _make_provider()
    provider.configure(
        'main',
        runtime_credential={'instance_id': 'sandbox-stopped'},
        runtime_kind='osworld',
        tool_type='osworld',
    )
    token_mock = AsyncMock(return_value='unused-token')
    monkeypatch.setattr(
        provider,
        '_call_api',
        AsyncMock(
            return_value={
                'InstanceSet': [{'InstanceId': 'sandbox-stopped', 'Status': 'STOPPED'}],
            }
        ),
    )
    monkeypatch.setattr(provider, '_acquire_token', token_mock)

    with pytest.raises(InvalidPersistentRuntimeError, match='not running: status=STOPPED'):
        await provider.acquire('main')

    token_mock.assert_not_awaited()
    assert provider.get_active_runtimes() == {}


@pytest.mark.asyncio
async def test_ags_runtime_credential_requires_instance_id() -> None:
    provider = _make_provider()
    provider.configure('main', runtime_credential={})

    with pytest.raises(InvalidPersistentRuntimeError, match='requires instance_id'):
        await provider.acquire('main')


@pytest.mark.asyncio
async def test_ags_rejects_direct_instance_id() -> None:
    provider = _make_provider()
    provider.configure('main', instance_id='sandbox-persistent-123')

    with pytest.raises(ValueError, match='use runtime_credential.instance_id'):
        await provider.acquire('main')


@pytest.mark.asyncio
async def test_ags_template_osworld_runtime_acquire_and_release(monkeypatch: pytest.MonkeyPatch) -> None:
    created_sandboxes: list[object] = []
    create_calls: list[dict[str, object]] = []

    class FakeSandbox:
        def __init__(self) -> None:
            self.sandbox_id = 'sandbox-template-123'
            self._envd_access_token = 'template-access-token'
            self.kill_calls = 0

        def kill(self) -> bool:
            self.kill_calls += 1
            return True

    class FakeSandboxSDK:
        @staticmethod
        def create(**kwargs: object) -> FakeSandbox:
            sandbox = FakeSandbox()
            create_calls.append(kwargs)
            created_sandboxes.append(sandbox)
            return sandbox

    async def _fake_wait_for_osworld_ready(*args: object, **kwargs: object) -> None:
        return None

    monkeypatch.setattr(ags_provider_module, 'Sandbox', FakeSandboxSDK)

    provider = AGSRuntimeProvider(
        api_key='test-api-key', template='osworld-template', domain='ap-guangzhou.tencentags.com'
    )
    provider.configure('main', runtime_kind='osworld', provision_mode='template')
    monkeypatch.setattr(provider, '_wait_for_osworld_ready', _fake_wait_for_osworld_ready)

    runtime = await provider.acquire('main')

    assert isinstance(runtime, AGSOSWorldRuntime)
    assert runtime.base_url == 'https://5000-sandbox-template-123.ap-guangzhou.tencentags.com'
    assert create_calls == [
        {
            'template': 'osworld-template',
            'timeout': 1800,
            'api_key': 'test-api-key',
            'domain': 'ap-guangzhou.tencentags.com',
            'request_timeout': 1800.0,
        }
    ]

    await provider.release('main')

    assert created_sandboxes[0].kill_calls == 1


@pytest.mark.asyncio
async def test_ags_template_raw_osworld_runtime_acquire_and_release(monkeypatch: pytest.MonkeyPatch) -> None:
    created_sandboxes: list[object] = []

    class FakeSandbox:
        def __init__(self) -> None:
            self.sandbox_id = 'sandbox-template-raw-123'
            self._envd_access_token = 'template-raw-access-token'
            self.kill_calls = 0

        def kill(self) -> bool:
            self.kill_calls += 1
            return True

    class FakeSandboxSDK:
        @staticmethod
        def create(**kwargs: object) -> FakeSandbox:
            _ = kwargs
            sandbox = FakeSandbox()
            created_sandboxes.append(sandbox)
            return sandbox

    async def _fake_wait_for_osworld_ready(*args: object, **kwargs: object) -> None:
        return None

    monkeypatch.setattr(ags_provider_module, 'Sandbox', FakeSandboxSDK)

    provider = AGSRuntimeProvider(
        api_key='test-api-key',
        template='osworld-template',
        domain='ap-guangzhou.tencentags.com',
        interface_mode='raw',
    )
    provider.configure('main', runtime_kind='osworld', provision_mode='template')
    monkeypatch.setattr(provider, '_wait_for_osworld_ready', _fake_wait_for_osworld_ready)

    runtime = await provider.acquire('main')

    assert isinstance(runtime, AGSRawOSWorldRuntime)
    assert runtime.base_url == 'https://5000-sandbox-template-raw-123.ap-guangzhou.tencentags.com'
    assert runtime.chromium_url == 'https://9222-sandbox-template-raw-123.ap-guangzhou.tencentags.com'

    await provider.release('main')

    assert created_sandboxes[0].kill_calls == 1


@pytest.mark.asyncio
async def test_ags_osworld_ready_requires_screenshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Command-only OSWorld mirrors must not bypass strict screenshot readiness."""

    class FakeHttpClient:
        def __init__(self) -> None:
            self.get_calls = 0
            self.post_calls = 0

        async def get(self, url: str, **kwargs: object) -> None:
            self.get_calls += 1
            assert url.endswith('/screenshot')
            raise TimeoutError()

        async def post(self, url: str, **kwargs: object) -> None:
            self.post_calls += 1

    http_client = FakeHttpClient()
    monkeypatch.setattr(ags_provider_module, 'http_client', http_client)

    provider = _make_provider()
    provider.instance_ready_timeout = 0.0
    handle = ags_provider_module.AGSInstanceHandle(
        instance_id='sandbox-ready',
        access_token='token',
        domain='example.tencentags.com',
    )

    with pytest.raises(TimeoutError, match='AGS OSWorld runtime not ready'):
        await provider._wait_for_osworld_ready(handle, 5000)

    assert http_client.get_calls == 1
    assert http_client.post_calls == 0


def test_ags_handle_uses_same_auth_headers_for_all_ports() -> None:
    handle = AGSInstanceHandle(
        instance_id='sandbox-gateway-123',
        access_token='template-access-token',
        domain='example.invalid',
    )

    assert handle.url(5000, '/screenshot') == 'https://5000-sandbox-gateway-123.example.invalid/screenshot'
    assert handle.host(9222) == '9222-sandbox-gateway-123.example.invalid'
    assert handle.headers == {'X-Access-Token': 'template-access-token'}


def test_ags_log_data_redacts_credentials_and_nested_tokens() -> None:
    payload = {
        'api_key': 'template-key',
        'ClientToken': 'idempotency-token',
        'Instance': {'InstanceId': 'instance-1'},
        'Token': 'instance-token',
        'nested': [{'TrafficToken': 'traffic-token', 'RequestId': 'request-1'}],
    }

    assert ags_provider_module._redact_for_log(payload) == {
        'api_key': '***REDACTED***',
        'ClientToken': '***REDACTED***',
        'Instance': {'InstanceId': 'instance-1'},
        'Token': '***REDACTED***',
        'nested': [{'TrafficToken': '***REDACTED***', 'RequestId': 'request-1'}],
    }


@pytest.mark.asyncio
async def test_ags_provider_can_skip_tool_lookup_when_tool_id_is_trusted(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _make_provider()

    async def fail_find_tool(*args: object, **kwargs: object) -> None:
        raise AssertionError('tool lookup should be skipped')

    monkeypatch.setattr(provider, '_find_tool', fail_find_tool)
    monkeypatch.setattr(provider, '_list_tools', fail_find_tool)
    monkeypatch.setattr(provider, '_wait_for_tool_active', fail_find_tool)

    tool, created = await provider._ensure_tool(
        'main',
        {
            'runtime_kind': 'osworld',
            'tool_type': 'osworld',
            'tool_id': 'sdt-existing',
            'skip_tool_lookup': True,
        },
    )

    assert created is False
    assert tool == {
        'ToolId': 'sdt-existing',
        'ToolName': 'ags-osworld-main',
        'ToolType': 'osworld',
        'Status': 'ACTIVE',
    }
    assert provider._tool_ids['main'] == 'sdt-existing'


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'selection',
    [
        {'tool_id': 'sdt-existing'},
        {'tool_id': 'sdt-existing', 'skip_tool_lookup': False},
        {'skip_tool_lookup': True},
    ],
)
async def test_ags_tool_lookup_requires_both_explicit_id_and_opt_in(monkeypatch, selection) -> None:
    provider = _make_provider()
    tool = {'ToolId': 'sdt-existing', 'ToolName': 'ags-osworld-main', 'ToolType': 'osworld', 'Status': 'ACTIVE'}
    listing = AsyncMock(return_value=[tool])
    monkeypatch.setattr(provider, '_list_tools', listing)
    wait = AsyncMock(wraps=provider._wait_for_tool_active)
    monkeypatch.setattr(provider, '_wait_for_tool_active', wait)

    result, created = await provider._ensure_tool('main', {'runtime_kind': 'osworld', **selection})

    assert result == tool
    assert created is False
    assert listing.await_count == 2
    wait.assert_awaited_once_with('sdt-existing')


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'invalid',
    [
        {'tool_type': 'unknown'},
        {'tool_strategy': 'ensure_custom'},
        {'launch_mode': 'nexus_server', 'interface_mode': 'raw'},
    ],
)
async def test_ags_skip_tool_lookup_preserves_configuration_validation(monkeypatch, invalid) -> None:
    provider = _make_provider()
    lookup = AsyncMock(side_effect=AssertionError('validation must precede discovery'))
    monkeypatch.setattr(provider, '_find_tool', lookup)
    with pytest.raises(ValueError, match='AGS'):
        await provider._ensure_tool(
            'main',
            {
                'runtime_kind': 'osworld',
                'tool_id': 'sdt-existing',
                'skip_tool_lookup': True,
                **invalid,
            },
        )
    lookup.assert_not_awaited()
    assert provider._tool_ids == {}


@pytest.mark.asyncio
@pytest.mark.parametrize('invalid_tool', [False, True])
async def test_ags_platform_run_input_reaches_start_without_tool_discovery(monkeypatch, invalid_tool) -> None:
    """Task 206902 runtime configuration, sanitized; only external boundaries are mocked."""
    from nexus.runner.single import RunInput, _create_runtime_provider

    run_input = RunInput.model_validate(
        {
            'task_name': 'os_world',
            'agent_name': 'claude_code_native_bridge',
            'agent_config': {'execution_profile': 'osworld', 'execution_options': {'runtime_name': 'main'}},
            'runtime_provider_name': 'ags',
            'runtime_provider_config': {
                'secret_id': 'test-secret-id',
                'secret_key': 'test-secret-key',
                'region': 'ap-guangzhou',
                'interface_mode': 'nexus',
                'osworld_cdp_proxy_mode': 'image',
                'instance_ready_timeout': 600,
                'request_timeout': 1800,
                'purpose': 'eval',
                'batch_id': 'test-platform',
            },
            'runtime_configs': {
                'main': {
                    'provision_mode': 'control_plane',
                    'runtime_kind': 'osworld',
                    'skip_tool_lookup': True,
                    'timeout': 86400,
                    'tool_id': 'sdt-invalid' if invalid_tool else 'sdt-existing',
                    'tool_name': 'osworld-official-cc-vpc',
                    'tunnel_port': 32000,
                }
            },
        }
    )
    provider = _create_runtime_provider(run_input)
    for method in ('_find_tool', '_list_tools', '_wait_for_tool_active'):
        monkeypatch.setattr(provider, method, AsyncMock(side_effect=AssertionError('unexpected tool discovery')))
    calls = []
    sdk_error = RuntimeError('[TencentCloudSDKException] code:ResourceNotFound message:tool does not exist')

    class Request:
        def from_json_string(self, raw):
            import json

            self.payload = json.loads(raw)

    class Response:
        def to_json_string(self):
            return '{"Instance": {"InstanceId": "instance-test"}}'

    class Client:
        def StartSandboxInstance(self, request):  # noqa: N802
            calls.append(request.payload)
            if invalid_tool:
                raise sdk_error
            return Response()

    class Models:
        StartSandboxInstanceRequest = Request

    monkeypatch.setattr(provider, '_get_sdk_client', lambda: (Client(), Models))
    token = AsyncMock(return_value='test-token')
    runtime = object()
    create = AsyncMock(return_value=runtime)
    monkeypatch.setattr(provider, '_acquire_token', token)
    monkeypatch.setattr(provider, '_create_runtime', create)

    if invalid_tool:
        with pytest.raises(RuntimeError, match='AGS StartSandboxInstance failed.*ResourceNotFound') as caught:
            await provider.acquire('main')
        assert caught.value.__cause__ is sdk_error
        token.assert_not_awaited()
        create.assert_not_awaited()
        assert provider._instance_ids == {}
        assert provider._active_runtimes == {}
    else:
        assert await provider.acquire('main') is runtime
        token.assert_awaited_once_with('instance-test')
        handle, merged, tool = create.call_args.args
        assert handle.instance_id == 'instance-test'
        assert handle.access_token == 'test-token'
        assert merged['skip_tool_lookup'] is True
        assert tool['ToolType'] == 'osworld'
        assert provider._active_runtimes['main'] is runtime

    assert len(calls) == 1
    assert calls[0]['ToolId'] == run_input.runtime_configs['main']['tool_id']
    assert calls[0]['Timeout'] == '86400s'
    assert calls[0]['ClientToken']
    assert provider._tools_to_delete == {}


@pytest.mark.asyncio
async def test_ags_control_plane_acquire_stops_instance_when_runtime_ready_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _make_provider()
    provider.configure(
        'main',
        runtime_kind='osworld',
        tool_type='osworld',
        tool_id='sdt-existing',
        skip_tool_lookup=True,
    )
    stop_calls: list[dict[str, object]] = []

    async def fake_start_instance(tool: dict[str, object], merged: dict[str, object]) -> dict[str, object]:
        return {
            'InstanceId': 'instance-not-ready',
            'ToolId': tool['ToolId'],
            'ToolName': tool['ToolName'],
        }

    async def fake_acquire_token(instance_id: str) -> str:
        assert instance_id == 'instance-not-ready'
        return 'test-access-token'

    async def fake_create_runtime(*args: object, **kwargs: object) -> object:
        raise TimeoutError('runtime not ready')

    async def fake_call_api(action: str, payload: dict[str, object]) -> dict[str, object]:
        if action == 'StopSandboxInstance':
            stop_calls.append(payload)
            return {'RequestId': 'stop-request'}
        raise AssertionError(f'unexpected API call: {action}')

    monkeypatch.setattr(provider, '_start_instance', fake_start_instance)
    monkeypatch.setattr(provider, '_acquire_token', fake_acquire_token)
    monkeypatch.setattr(provider, '_create_runtime', fake_create_runtime)
    monkeypatch.setattr(provider, '_call_api', fake_call_api)

    with pytest.raises(TimeoutError, match='runtime not ready'):
        await provider.acquire('main')

    assert stop_calls == [{'InstanceId': 'instance-not-ready'}]


@pytest.mark.asyncio
async def test_ags_template_osworld_uses_standard_handle_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    create_calls: list[dict[str, object]] = []

    class FakeSandbox:
        sandbox_id = 'sandbox-template-gateway-123'
        _envd_access_token = 'template-access-token'

        def kill(self) -> bool:
            return True

    class FakeSandboxSDK:
        @staticmethod
        def create(**kwargs: object) -> FakeSandbox:
            create_calls.append(kwargs)
            return FakeSandbox()

    captured_headers: list[dict[str, str]] = []

    async def _fake_ready(handle: AGSInstanceHandle, server_port: int) -> None:
        assert server_port == 5000
        captured_headers.append(handle.headers)

    monkeypatch.setattr(ags_provider_module, 'Sandbox', FakeSandboxSDK)

    provider = AGSRuntimeProvider(
        api_key='test-api-key',
        template='osworld-suite-mcp',
        domain='example.invalid',
    )
    provider.configure('main', runtime_kind='osworld', provision_mode='template')
    monkeypatch.setattr(provider, '_wait_for_osworld_ready', _fake_ready)

    runtime = await provider.acquire('main')

    assert isinstance(runtime, AGSOSWorldRuntime)
    assert runtime.base_url == 'https://5000-sandbox-template-gateway-123.example.invalid'
    assert runtime.port_mappings[5000] == runtime.base_url
    assert runtime.handle.headers == {'X-Access-Token': 'template-access-token'}
    assert captured_headers == [
        {
            'X-Access-Token': 'template-access-token',
        }
    ]
    assert create_calls == [
        {
            'template': 'osworld-suite-mcp',
            'timeout': 1800,
            'api_key': 'test-api-key',
            'domain': 'example.invalid',
            'request_timeout': 1800.0,
        }
    ]

    await provider.release('main')


@pytest.mark.asyncio
@pytest.mark.parametrize('cleanup_fails', [False, True])
async def test_ags_template_acquire_kills_sandbox_when_cancelled(
    monkeypatch: pytest.MonkeyPatch, cleanup_fails: bool
) -> None:
    created_sandboxes: list[object] = []

    class FakeSandbox:
        def __init__(self) -> None:
            self.sandbox_id = 'sandbox-template-cancelled'
            self._envd_access_token = 'template-cancelled-access-token'
            self.kill_calls = 0

        def kill(self) -> bool:
            self.kill_calls += 1
            if cleanup_fails:
                raise RuntimeError('sandbox cleanup failed')
            return True

    class FakeSandboxSDK:
        @staticmethod
        def create(**kwargs: object) -> FakeSandbox:
            _ = kwargs
            sandbox = FakeSandbox()
            created_sandboxes.append(sandbox)
            return sandbox

    async def fake_create_runtime(*args: object, **kwargs: object) -> object:
        _ = args, kwargs
        raise asyncio.CancelledError

    monkeypatch.setattr(ags_provider_module, 'Sandbox', FakeSandboxSDK)

    provider = AGSRuntimeProvider(
        api_key='test-api-key', template='osworld-template', domain='ap-guangzhou.tencentags.com'
    )
    provider.configure('main', runtime_kind='osworld', provision_mode='template')
    monkeypatch.setattr(provider, '_create_runtime', fake_create_runtime)

    with pytest.raises(asyncio.CancelledError):
        await provider.acquire('main')

    assert created_sandboxes[0].kill_calls == 1
    assert provider._template_sandboxes == {}
    assert provider._instance_ids == {}
