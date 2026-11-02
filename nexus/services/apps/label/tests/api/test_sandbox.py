from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from fastapi.testclient import TestClient

from infra.config import ConfigHolder
from label import LabelConfig, create_app
from label.config import ProviderConfig, load_provider
from label.schemas.sandbox import Provider
from label.services.sandbox import SandboxService


_ORIGINAL_CREATE_PROVIDER = SandboxService._create_provider


class FakeHandle:
    def __init__(self) -> None:
        self.instance_id = 'ags-instance'
        self.access_token = 'tok-abc123'


class FakeRuntime:
    def __init__(self, provider: Provider) -> None:
        self.vnc_url = 'https://8006-instance.ags.example'
        self.metadata = {'ip': '10.0.0.9', 'password': 'secret'}
        self.provider = provider
        self.handle = FakeHandle()


class FakeProvider:
    def __init__(self, provider: Provider) -> None:
        self.provider = provider
        self.control_api_base_url = 'http://ratel.example'
        self.api_calls: list[tuple[Any, ...]] = []
        self.mcp_calls: list[tuple[str, dict[str, Any]]] = []
        self.released: list[str] = []

    async def acquire(self, name: str, **params: Any) -> FakeRuntime:
        return FakeRuntime(self.provider)

    async def release(self, name: str) -> None:
        self.released.append(name)

    async def _call_api(self, *args: Any) -> dict[str, Any]:
        self.api_calls.append(args)
        return {}

    async def _call_mcp(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.mcp_calls.append((tool, arguments))
        return {}

    def get_runtime_id(self, name: str) -> str:
        return {
            Provider.AGS: 'ags-instance',
            Provider.DEVCLOUD: 'devcloud-vm',
            Provider.SMARTRUN: 'smartrun-session',
        }[self.provider]

    def get_device_id(self, name: str) -> str:
        return 'android-device'


def _client(providers: dict[Provider, FakeProvider]) -> TestClient:
    def create_provider(_self: object, provider: Provider) -> FakeProvider:
        return providers[provider]

    patcher = patch('label.services.sandbox.SandboxService._create_provider', new=create_provider)
    patcher.start()
    client = TestClient(create_app(LabelConfig()))
    client.__dict__['_provider_patcher'] = patcher
    return client


def test_acquire_and_release_ags_windows() -> None:
    provider = FakeProvider(Provider.AGS)
    client = _client({Provider.AGS: provider})

    acquired = client.post('/api/v1/label/sandbox/acquire', json={'platform': 'windows', 'provider': 'ags'})
    released = client.post(
        '/api/v1/label/sandbox/release',
        json={'platform': 'windows', 'provider': 'ags', 'instance_id': 'ags-instance'},
    )

    assert acquired.status_code == 200
    assert acquired.json() == {
        'code': 0,
        'message': 'success',
        # 未启用 OTel 时无有效 span，trace_id 为 None
        'trace_id': None,
        'data': {
            'instance_id': 'ags-instance',
            'instance_ip': None,
            'vm_id': None,
            'device_id': None,
            'session_id': None,
            'remote_control_url': 'https://8006-instance.ags.example/vnc.html',
        },
    }
    assert released.json() == {'code': 0, 'message': 'success', 'trace_id': None}
    assert provider.api_calls == [('StopSandboxInstance', {'InstanceId': 'ags-instance'})]


def test_release_devcloud_uses_vm_id() -> None:
    provider = FakeProvider(Provider.DEVCLOUD)
    client = _client({Provider.DEVCLOUD: provider})

    released = client.post(
        '/api/v1/label/sandbox/release',
        json={'platform': 'macos', 'provider': 'devcloud', 'instance_ip': '10.0.0.9', 'vm_id': 'devcloud-vm'},
    )

    assert released.status_code == 200
    assert provider.api_calls == [
        ('http://ratel.example', '/mac/vm/delete', {'id': 'devcloud-vm', 'force': True}),
    ]


def test_release_smartrun_uses_device_id() -> None:
    provider = FakeProvider(Provider.SMARTRUN)
    client = _client({Provider.SMARTRUN: provider})

    released = client.post(
        '/api/v1/label/sandbox/release',
        json={
            'platform': 'android',
            'provider': 'smartrun',
            'device_id': 'android-device',
            'session_id': 'smartrun-session',
        },
    )

    assert released.status_code == 200
    assert provider.mcp_calls == [('release_device', {'device_id': 'android-device', 'json_output': True})]


def test_release_is_stateless_without_prior_acquire() -> None:
    provider = FakeProvider(Provider.AGS)
    client = _client({Provider.AGS: provider})

    released = client.post(
        '/api/v1/label/sandbox/release',
        json={'platform': 'windows', 'provider': 'ags', 'instance_id': 'never-acquired'},
    )

    assert released.status_code == 200
    assert provider.api_calls == [('StopSandboxInstance', {'InstanceId': 'never-acquired'})]


def test_acquire_devcloud_macos() -> None:
    client = _client({Provider.DEVCLOUD: FakeProvider(Provider.DEVCLOUD)})

    response = client.post('/api/v1/label/sandbox/acquire', json={'platform': 'macos', 'provider': 'devcloud'})

    assert response.status_code == 200
    assert response.json()['data'] == {
        'instance_id': None,
        'instance_ip': '10.0.0.9',
        'vm_id': 'devcloud-vm',
        'device_id': None,
        'session_id': None,
        'remote_control_url': 'vnc://10.0.0.9',
    }


def test_acquire_smartrun_android() -> None:
    client = _client({Provider.SMARTRUN: FakeProvider(Provider.SMARTRUN)})

    response = client.post('/api/v1/label/sandbox/acquire', json={'platform': 'android', 'provider': 'smartrun'})

    assert response.status_code == 200
    assert response.json()['data']['device_id'] == 'android-device'
    assert response.json()['data']['session_id'] == 'smartrun-session'
    assert response.json()['data']['remote_control_url'].endswith(
        'live-stream?deviceId=android-device&sessionId=smartrun-session'
    )


def test_rejects_invalid_provider_platform_pair() -> None:
    client = TestClient(create_app(LabelConfig()))

    response = client.post('/api/v1/label/sandbox/acquire', json={'platform': 'android', 'provider': 'ags'})

    assert response.status_code == 422


def test_release_requires_provider_identifiers() -> None:
    client = TestClient(create_app(LabelConfig()))

    response = client.post('/api/v1/label/sandbox/release', json={'platform': 'macos', 'provider': 'devcloud'})

    assert response.status_code == 422


def test_create_ags_provider_disables_auth() -> None:
    provider_config = ProviderConfig.model_validate({
        'ags': {
            'secret_id': 'secret-id',
            'secret_key': 'secret-key',
            'region': 'ap-guangzhou',
            'domain': 'ags.example.com',
        }
    })
    config = LabelConfig(provider=provider_config)
    holder = ConfigHolder(config, field='provider', loader=load_provider)
    service = SandboxService(holder)

    with patch('nexus.runtime.providers.ags.AGSRuntimeProvider') as provider_cls:
        _ORIGINAL_CREATE_PROVIDER(service, Provider.AGS)

    provider_cls.assert_called_once_with(
        secret_id='secret-id',
        secret_key='secret-key',
        region='ap-guangzhou',
        domain='ags.example.com',
        interface_mode='raw',
        auth_mode='NONE',
    )


def test_create_ags_waa_runtime_supports_standard_executors() -> None:
    from nexus.runtime.providers.ags.waa import AGSWindowsAgentArenaRuntime

    provider = SimpleNamespace(
        domain='ap-guangzhou.tencentags.com',
        request_timeout=123.0,
    )

    runtime = SandboxService._create_ags_waa_runtime(provider, 'ags-instance')

    assert isinstance(runtime, AGSWindowsAgentArenaRuntime)
    assert runtime.handle.instance_id == 'ags-instance'
    assert runtime.handle.domain == provider.domain
    assert runtime.handle.access_token == ''
    assert runtime.request_timeout == provider.request_timeout


def test_call_executor_forwards_arbitrary_executor_to_ags_waa() -> None:
    provider = FakeProvider(Provider.AGS)
    client = _client({Provider.AGS: provider})
    calls: list[tuple[str, Any]] = []

    class FakeExecutorRuntime:
        async def call_executor(self, executor_name: str, executor_args: Any) -> dict[str, Any]:
            calls.append((executor_name, executor_args))
            return {'custom': 'result'}

    with patch(
        'label.services.sandbox.SandboxService._create_ags_waa_runtime', return_value=FakeExecutorRuntime()
    ) as create_runtime:
        response = client.post(
            '/api/v1/label/sandbox/call_executor',
            json={
                'platform': 'windows',
                'provider': 'ags',
                'instance_id': 'ags-instance',
                'executor_name': 'custom_executor',
                'executor_args': {'arbitrary': ['value']},
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        'code': 0,
        'message': 'success',
        'trace_id': None,
        'data': {'custom': 'result'},
    }
    create_runtime.assert_called_once_with(provider, 'ags-instance')
    assert calls == [('custom_executor', {'arbitrary': ['value']})]


def test_call_executor_only_accepts_ags_windows() -> None:
    client = TestClient(create_app(LabelConfig()))

    response = client.post(
        '/api/v1/label/sandbox/call_executor',
        json={
            'platform': 'android',
            'provider': 'smartrun',
            'instance_id': 'device-id',
            'executor_name': 'gui_click',
            'executor_args': {'x': 1, 'y': 2},
        },
    )

    assert response.status_code == 422


def test_call_executor_maps_runtime_error() -> None:
    client = _client({Provider.AGS: FakeProvider(Provider.AGS)})

    class FailingRuntime:
        async def call_executor(self, executor_name: str, executor_args: Any) -> Any:
            raise RuntimeError('upstream failed')

    with patch('label.services.sandbox.SandboxService._create_ags_waa_runtime', return_value=FailingRuntime()):
        response = client.post(
            '/api/v1/label/sandbox/call_executor',
            json={
                'platform': 'windows',
                'provider': 'ags',
                'instance_id': 'ags-instance',
                'executor_name': 'unknown_executor',
                'executor_args': {},
            },
        )

    assert response.status_code == 502
    assert response.json() == {
        'code': 50202,
        'message': 'failed to call executor unknown_executor',
        'trace_id': None,
    }
