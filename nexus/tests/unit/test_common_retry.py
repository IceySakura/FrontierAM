import asyncio

import httpx
import openai
import pytest

from tenacity import wait_none

import nexus.utils.common as common

from nexus.core.openai_client import ContextLengthExceededError
from nexus.utils.common import configurable_retry


def _status_error(status: int) -> openai.APIStatusError:
    request = httpx.Request('POST', 'http://api.example.test/v1/chat/completions')
    response = httpx.Response(status, request=request)
    body = {'error': {'code': '1210', 'message': "messages.content.type 参数非法，取值范围 ['text']"}}
    if status == 400:
        return openai.BadRequestError(f'Error code: {status}', response=response, body=body)
    return openai.APIStatusError(f'Error code: {status}', response=response, body=body)


class _MappedBadRequestError(Exception):
    """Stands in for litellm.BadRequestError, which also exposes status_code=400."""

    status_code = 400


@pytest.mark.asyncio
async def test_configurable_retry_does_not_retry_http_400() -> None:
    attempts = 0

    async def fail() -> None:
        nonlocal attempts
        attempts += 1
        raise _status_error(400)

    with pytest.raises(openai.BadRequestError):
        await configurable_retry(15)(fail)()

    assert attempts == 1


@pytest.mark.asyncio
async def test_configurable_retry_detects_wrapped_http_400() -> None:
    # Mirrors the production chain: litellm.BadRequestError <- OpenAIError <- openai.BadRequestError.
    attempts = 0

    async def fail() -> None:
        nonlocal attempts
        attempts += 1
        try:
            raise _status_error(400)
        except openai.BadRequestError as error:
            raise RuntimeError('wrapped by provider adapter') from error

    with pytest.raises(RuntimeError, match='wrapped by provider adapter'):
        await configurable_retry(15)(fail)()

    assert attempts == 1


@pytest.mark.asyncio
async def test_configurable_retry_does_not_retry_mapped_status_code_400() -> None:
    attempts = 0

    async def fail() -> None:
        nonlocal attempts
        attempts += 1
        raise _MappedBadRequestError('text-only model rejected image_url')

    with pytest.raises(_MappedBadRequestError):
        await configurable_retry(15)(fail)()

    assert attempts == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('status', [408, 422, 429, 500, 503])
async def test_configurable_retry_still_retries_other_status_errors(monkeypatch, status: int) -> None:
    monkeypatch.setattr(common, 'wait_exponential_jitter', lambda **_kwargs: wait_none())
    attempts = 0

    async def fail_once() -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise _status_error(status)
        return 'recovered'

    assert await configurable_retry(2)(fail_once)() == 'recovered'
    assert attempts == 2


@pytest.mark.asyncio
async def test_configurable_retry_does_not_retry_context_length_error() -> None:
    attempts = 0

    async def fail() -> None:
        nonlocal attempts
        attempts += 1
        raise ContextLengthExceededError('prompt too large')

    with pytest.raises(ContextLengthExceededError, match='prompt too large'):
        await configurable_retry(15)(fail)()

    assert attempts == 1


@pytest.mark.asyncio
async def test_configurable_retry_detects_wrapped_provider_context_error() -> None:
    class ContextWindowExceededError(Exception):
        pass

    attempts = 0

    async def fail() -> None:
        nonlocal attempts
        attempts += 1
        try:
            raise ContextWindowExceededError('provider rejected context')
        except ContextWindowExceededError as error:
            raise RuntimeError('wrapped') from error

    with pytest.raises(RuntimeError, match='wrapped'):
        await configurable_retry(15)(fail)()

    assert attempts == 1


@pytest.mark.asyncio
async def test_configurable_retry_propagates_task_cancellation_without_retry(monkeypatch) -> None:
    monkeypatch.setattr(common, 'wait_exponential_jitter', lambda **_kwargs: wait_none())
    attempts = 0
    entered = asyncio.Event()

    async def operation() -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            entered.set()
            await asyncio.Future()
        return 'retried after cancellation'

    task = asyncio.create_task(configurable_retry(2)(operation)())
    await entered.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert attempts == 1


@pytest.mark.parametrize('error_type', [KeyboardInterrupt, SystemExit, GeneratorExit])
def test_configurable_retry_propagates_sync_control_flow_without_retry(monkeypatch, error_type) -> None:
    monkeypatch.setattr(common, 'wait_exponential_jitter', lambda **_kwargs: wait_none())
    attempts = 0
    error = error_type('stop')

    @configurable_retry(2)
    def operation() -> None:
        nonlocal attempts
        attempts += 1
        raise error

    with pytest.raises(error_type) as captured:
        operation()

    assert captured.value is error
    assert attempts == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('error_type', [KeyboardInterrupt, SystemExit, GeneratorExit])
async def test_configurable_retry_propagates_async_control_flow_without_retry(monkeypatch, error_type) -> None:
    monkeypatch.setattr(common, 'wait_exponential_jitter', lambda **_kwargs: wait_none())
    attempts = 0
    error = error_type('stop')

    @configurable_retry(2)
    async def operation() -> None:
        nonlocal attempts
        attempts += 1
        raise error

    with pytest.raises(error_type) as captured:
        await operation()

    assert captured.value is error
    assert attempts == 1


@pytest.mark.parametrize('error_type', [KeyboardInterrupt, SystemExit])
def test_configurable_retry_does_not_retry_wrapped_control_flow(monkeypatch, error_type) -> None:
    monkeypatch.setattr(common, 'wait_exponential_jitter', lambda **_kwargs: wait_none())
    attempts = 0
    error = error_type('stop')

    @configurable_retry(2)
    def operation() -> None:
        nonlocal attempts
        attempts += 1
        try:
            raise error
        except BaseException as cause:
            raise RuntimeError('wrapped') from cause

    with pytest.raises(RuntimeError, match='wrapped') as captured:
        operation()

    assert captured.value.__cause__ is error
    assert attempts == 1


def test_configurable_retry_retries_sync_ordinary_exceptions(monkeypatch) -> None:
    monkeypatch.setattr(common, 'wait_exponential_jitter', lambda **_kwargs: wait_none())
    attempts = 0

    @configurable_retry(2)
    def operation() -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError('temporary failure')
        return 'recovered'

    assert operation() == 'recovered'
    assert attempts == 2


@pytest.mark.asyncio
async def test_configurable_retry_retries_async_ordinary_exceptions(monkeypatch) -> None:
    monkeypatch.setattr(common, 'wait_exponential_jitter', lambda **_kwargs: wait_none())
    attempts = 0

    @configurable_retry(2)
    async def operation() -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError('temporary failure')
        return 'recovered'

    assert await operation() == 'recovered'
    assert attempts == 2


@pytest.mark.asyncio
@pytest.mark.parametrize('recover', [False, True])
async def test_call_deadline_allows_retry_and_respects_attempt_limit(monkeypatch, recover):
    monkeypatch.setattr(common, 'wait_exponential_jitter', lambda **_kwargs: wait_none())
    attempts = 0
    cancelled = 0

    @configurable_retry(2)
    @common.with_call_deadline(0.01)
    async def operation():
        nonlocal attempts, cancelled
        attempts += 1
        if recover and attempts == 2:
            return 'recovered'
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled += 1
            raise

    if recover:
        assert await operation() == 'recovered'
    else:
        with pytest.raises(TimeoutError, match='client-side deadline'):
            await operation()
    assert attempts == 2
    assert cancelled == (1 if recover else 2)


@pytest.mark.asyncio
async def test_external_cancellation_crosses_call_deadline_without_retry(monkeypatch):
    monkeypatch.setattr(common, 'wait_exponential_jitter', lambda **_kwargs: wait_none())
    entered = asyncio.Event()
    cancelled = asyncio.Event()
    attempts = 0

    @configurable_retry(2)
    @common.with_call_deadline(10)
    async def operation():
        nonlocal attempts
        attempts += 1
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    task = asyncio.create_task(operation())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled.is_set()
    assert attempts == 1


@pytest.mark.asyncio
async def test_provider_timeout_does_not_hide_external_cancellation(monkeypatch):
    monkeypatch.setattr(common, 'wait_exponential_jitter', lambda **_kwargs: wait_none())
    entered = asyncio.Event()
    attempts = 0

    @configurable_retry(2)
    @common.with_call_deadline(10)
    async def operation():
        nonlocal attempts
        attempts += 1
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError as cause:
            raise TimeoutError('provider converted cancellation') from cause

    task = asyncio.create_task(operation())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert attempts == 1


@pytest.mark.asyncio
async def test_provider_timeout_keeps_non_retryable_context_error(monkeypatch):
    monkeypatch.setattr(common, 'wait_exponential_jitter', lambda **_kwargs: wait_none())
    attempts = 0

    @configurable_retry(2)
    @common.with_call_deadline(1)
    async def operation():
        nonlocal attempts
        attempts += 1
        try:
            raise ContextLengthExceededError('too large')
        except ContextLengthExceededError as cause:
            raise TimeoutError('provider timeout') from cause

    with pytest.raises(TimeoutError):
        await operation()
    assert attempts == 1
