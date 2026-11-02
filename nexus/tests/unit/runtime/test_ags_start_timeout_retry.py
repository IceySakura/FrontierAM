"""Regression coverage for AGS provider-result deadlines during creation."""

import json

from unittest.mock import AsyncMock

import pytest

from nexus.runtime.providers.ags import provider as ags_module


class SdkError(Exception):
    def __init__(self, code):
        super().__init__(f'code:{code} message:provider result unavailable')
        self.code = code


def provider_with_sdk(monkeypatch, effects):
    requests = []

    class Request:
        def from_json_string(self, raw):
            self.payload = json.loads(raw)

    class Response:
        def to_json_string(self):
            return '{"Instance":{"InstanceId":"instance-test"}}'

    class Client:
        def StartSandboxInstance(self, request):  # noqa: N802
            requests.append(request)
            effect = effects.pop(0)
            if effect:
                raise effect
            return Response()

    class Models:
        StartSandboxInstanceRequest = Request

    provider = ags_module.AGSRuntimeProvider(secret_id='test-id', secret_key='test-key')
    monkeypatch.setattr(provider, '_get_sdk_client', lambda: (Client(), Models))
    sleep = AsyncMock()
    monkeypatch.setattr(ags_module.asyncio, 'sleep', sleep)
    return provider, requests, sleep


@pytest.mark.asyncio
async def test_start_timeout_retries_same_request_and_client_token(monkeypatch):
    provider, requests, sleep = provider_with_sdk(
        monkeypatch, [SdkError('FailedOperation.Timeout'), SdkError('FailedOperation.Timeout'), None]
    )
    instance = await provider._start_instance({'ToolId': 'tool-test'}, {'timeout': 10800})
    assert instance == {'InstanceId': 'instance-test'}
    assert len(requests) == 3
    assert all(request is requests[0] for request in requests)
    assert requests[0].payload['ClientToken']
    assert requests[0].payload['Timeout'] == '10800s'
    assert [call.args[0] for call in sleep.await_args_list] == [5, 10]


@pytest.mark.asyncio
async def test_start_timeout_has_finite_retry_budget(monkeypatch):
    errors = [SdkError('FailedOperation.Timeout') for _ in range(3)]
    last_error = errors[-1]
    provider, requests, sleep = provider_with_sdk(monkeypatch, errors)
    with pytest.raises(RuntimeError, match='StartSandboxInstance failed') as caught:
        await provider._start_instance({'ToolId': 'tool-test'}, {})
    assert caught.value.__cause__ is last_error
    assert len(requests) == 3
    assert sleep.await_count == 2


@pytest.mark.parametrize('code', ['LimitExceeded', 'AuthFailure', 'InvalidParameterValue', 'ResourceNotFound'])
@pytest.mark.asyncio
async def test_start_permanent_errors_are_not_retried(monkeypatch, code):
    provider, requests, sleep = provider_with_sdk(monkeypatch, [SdkError(code)])
    with pytest.raises(RuntimeError):
        await provider._start_instance({'ToolId': 'tool-test'}, {})
    assert len(requests) == 1
    sleep.assert_not_awaited()


def test_provider_timeout_is_not_retried_for_other_actions():
    assert not ags_module._is_retryable_ags_api_error('CreateSandboxTool', SdkError('FailedOperation.Timeout'))
