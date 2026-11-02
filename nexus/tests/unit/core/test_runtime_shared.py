from __future__ import annotations

import pytest

from nexus.core.context import ExecutionContext
from nexus.core.runtime_shared import RuntimeSharedStore, SharedValueNotFoundError
from nexus.runtime.context import RuntimeControl
from nexus.utils.artifact.auth import ARTIFACT_SHARED_VALUE_NAME, ArtifactCredentials
from nexus.utils.container_registry_auth import RegistryCredentials, registry_shared_value_name
from nexus.utils.cos import COS_SHARED_VALUE_NAME, CosCredentials


class _HostRuntime:
    def __init__(self, runtime_control: RuntimeControl) -> None:
        self.runtime_control = runtime_control
        self.handlers = {}

    def on(self, event, handler):  # noqa: ANN001, ANN201
        self.handlers[event] = handler
        return lambda: None

    async def emit(self, event, payload):  # noqa: ANN001, ANN201
        raise AssertionError(f'Unexpected Host emit: {event} {payload}')

    async def invoke(self, target, *args):  # noqa: ANN001, ANN201
        if target.endswith('::provide_runtime_shared_value'):
            self.runtime_control.provide_shared_value(*args)
            return
        if target.endswith('::fail_runtime_shared_value'):
            self.runtime_control.fail_shared_value(*args)
            return
        raise AssertionError(f'Unexpected invoke target: {target}')


class _RuntimeSIO:
    def __init__(self, host_runtime: _HostRuntime) -> None:
        self.host_runtime = host_runtime

    async def emit(self, event, payload):  # noqa: ANN001, ANN201
        await self.host_runtime.handlers[event](payload)


def test_runtime_shared_store_keeps_typed_values_in_memory() -> None:
    store = RuntimeSharedStore()
    value = object()

    store.set('example', value)

    assert store.get('example') is value
    assert store.get('example', object) is value
    assert store.get_optional('missing') is None


def test_runtime_shared_store_reports_missing_and_wrong_types() -> None:
    store = RuntimeSharedStore()
    store.set('example', 'value')

    with pytest.raises(SharedValueNotFoundError):
        store.get('missing')
    with pytest.raises(TypeError, match='expected int'):
        store.get('example', int)


@pytest.mark.asyncio
async def test_runtime_requests_host_shared_value_and_keeps_it_in_memory() -> None:
    runtime_control = RuntimeControl()
    host_runtime = _HostRuntime(runtime_control)
    runtime_control.sio = _RuntimeSIO(host_runtime)
    ctx = ExecutionContext()
    credentials = CosCredentials(secret_id='test-id', secret_key='test-key')
    ctx.runtime_shared.set(COS_SHARED_VALUE_NAME, credentials)
    ctx._attach_runtime_shared(host_runtime)

    first = await runtime_control.get_shared_value(
        COS_SHARED_VALUE_NAME,
        CosCredentials,
        decode=CosCredentials.from_shared_payload,
    )
    second = await runtime_control.get_shared_value(
        COS_SHARED_VALUE_NAME,
        CosCredentials,
        decode=CosCredentials.from_shared_payload,
    )

    assert first.to_shared_payload() == {'secret_id': 'test-id', 'secret_key': 'test-key'}
    assert second is first


@pytest.mark.asyncio
async def test_runtime_requests_artifact_credentials_from_host_key_file(monkeypatch, tmp_path) -> None:
    key_file = tmp_path / 'taihu-key'
    key_file.write_text('Bearer tai_pat_host', encoding='utf-8')
    monkeypatch.delenv('ARTIFACT_TAIHU_PAT', raising=False)
    monkeypatch.setenv('ARTIFACT_TAIHU_PAT_FILE', str(key_file))
    runtime_control = RuntimeControl()
    host_runtime = _HostRuntime(runtime_control)
    runtime_control.sio = _RuntimeSIO(host_runtime)
    ctx = ExecutionContext()
    ctx._attach_runtime_shared(host_runtime)

    first = await runtime_control.get_shared_value(
        ARTIFACT_SHARED_VALUE_NAME,
        ArtifactCredentials,
        decode=ArtifactCredentials.from_shared_payload,
    )
    key_file.write_text('tai_pat_changed', encoding='utf-8')
    second_runtime_control = RuntimeControl()
    second_host_runtime = _HostRuntime(second_runtime_control)
    second_runtime_control.sio = _RuntimeSIO(second_host_runtime)
    ctx._attach_runtime_shared(second_host_runtime)
    second = await second_runtime_control.get_shared_value(
        ARTIFACT_SHARED_VALUE_NAME,
        ArtifactCredentials,
        decode=ArtifactCredentials.from_shared_payload,
    )

    assert first.to_shared_payload() == {'taihu_pat': 'tai_pat_host'}
    assert second.to_shared_payload() == first.to_shared_payload()


@pytest.mark.asyncio
async def test_runtime_requests_registry_credentials_from_host() -> None:
    runtime_control = RuntimeControl()
    host_runtime = _HostRuntime(runtime_control)
    runtime_control.sio = _RuntimeSIO(host_runtime)
    ctx = ExecutionContext()
    name = registry_shared_value_name('mirrors.tencent.com')
    credentials = RegistryCredentials(
        registry='mirrors.tencent.com',
        username='test-user',
        password='test-password',
    )
    ctx.runtime_shared.set(name, credentials)
    ctx._attach_runtime_shared(host_runtime)

    received = await runtime_control.get_shared_value(
        name,
        RegistryCredentials,
        decode=RegistryCredentials.from_shared_payload,
    )

    assert received.to_shared_payload() == credentials.to_shared_payload()
