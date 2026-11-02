from __future__ import annotations

import asyncio
import threading
import time

from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

import httpx
import pytest

from tenacity import retry, stop_after_attempt, wait_none

from nexus.core import OpenAIConfig
from nexus.core.request_admission import RequestAttempt, RequestPermit, bind_request_admission
from nexus.distributed.policies.models import SignalObservation
from nexus.extensions.openai_clients import infer_proxy
from nexus.extensions.openai_clients import infer_proxy_local_policy as local


@pytest.fixture(autouse=True)
def fresh_registry(monkeypatch):
    monkeypatch.setattr(local, '_admissions', {})


@pytest.fixture
def clock(monkeypatch):
    now = [0.0]
    wall = local.utcnow()

    async def sleep(delay):
        now[0] += delay
        await asyncio.sleep(0)

    monkeypatch.setattr(local, 'monotonic', lambda: now[0])
    monkeypatch.setattr(local, 'utcnow', lambda: wall + timedelta(seconds=now[0]))
    monkeypatch.setattr(local, 'sleep', sleep)
    return now


def capacity(monkeypatch, responses):
    calls = []
    responses = iter(responses)

    async def read(self, config, *, max_age_seconds):
        calls.append((local.monotonic(), config))
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        now = local.utcnow()
        return SignalObservation(
            version=str(len(calls)),
            observed_at=now,
            valid_until=now + timedelta(seconds=max_age_seconds),
            values={'keyed': True, 'remaining': 600, 'factor': 1.0, **response},
        )

    monkeypatch.setattr(local.InferProxyRequestCapacitySource, 'read', read)
    return calls


def admission(**overrides):
    return local.local_request_admission(
        model='hy3-offline', api_key='test-key', config={'quota_qpm': 600, **overrides}, client_id='test-client'
    )


async def test_clients_share_one_poll_and_one_pacer_and_keep_history(monkeypatch, clock):
    calls = capacity(monkeypatch, [{'factor': 0.5}])
    first, second = admission(), admission()
    assert first.admission is second.admission
    await first.acquire(client_id='first')
    assert clock[0] == pytest.approx(0.2)
    await second.acquire(client_id='second')
    assert clock[0] == pytest.approx(0.4)
    del first, second
    await admission().acquire(client_id='new-task')
    assert clock[0] == pytest.approx(0.6)
    assert len(calls) == 1
    assert calls[0][1]['model'] == 'hy3-offline'
    assert calls[0][1]['api_key'] == 'test-key'


async def test_pause_continues_polling_and_resumes_without_accumulated_permits(monkeypatch, clock):
    calls = capacity(monkeypatch, [{'factor': 0.5}, {'remaining': 0}, {'factor': 2}])
    bound = admission()
    await bound.acquire(client_id='first')
    clock[0] = 60
    permit = await bound.acquire(client_id='paused')
    assert [time for time, _ in calls] == [0, 60, 120]
    assert clock[0] == pytest.approx(120.1)
    assert permit.is_valid(now=clock[0])
    await bound.acquire(client_id='next')
    assert clock[0] == pytest.approx(120.2)


async def test_failure_uses_only_unexpired_decision_and_does_not_leak_key(monkeypatch, clock, caplog):
    calls = capacity(monkeypatch, [{}, httpx.ReadTimeout('test-key'), {}])
    bound = admission(max_age_seconds=61)
    await bound.acquire(client_id='first')
    clock[0] = 60
    await bound.acquire(client_id='still-valid')
    assert clock[0] == 60
    clock[0] = 61
    await bound.acquire(client_id='expired')
    assert clock[0] == pytest.approx(120.1)
    assert [time for time, _ in calls] == [0, 60, 120]
    assert 'test-key' not in caplog.text


async def test_ordinary_refresh_keeps_progress_for_rates_below_one_request_per_interval(monkeypatch, clock):
    calls = capacity(monkeypatch, [{}, {}, {}])
    await admission(quota_qpm=0.5).acquire(client_id='slow')
    assert clock[0] == 120
    assert len(calls) == 3


async def test_cancelled_refresh_releases_the_poll_for_other_clients(monkeypatch):
    entered = asyncio.Event()

    async def read(self, config, *, max_age_seconds):
        entered.set()
        await asyncio.Future()

    monkeypatch.setattr(local.InferProxyRequestCapacitySource, 'read', read)
    bound = admission()
    task = asyncio.create_task(bound.acquire(client_id='cancelled'))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    calls = capacity(monkeypatch, [{}])
    await asyncio.wait_for(bound.acquire(client_id='survivor'), timeout=2)
    assert len(calls) == 1


def test_different_thread_event_loops_share_poll_and_send_spacing(monkeypatch):
    calls = capacity(monkeypatch, [{}])
    barrier = threading.Barrier(4)

    def run(index):
        bound = admission(quota_qpm=1200)
        barrier.wait(timeout=5)
        asyncio.run(bound.acquire(client_id=str(index)))
        return time.monotonic()

    with ThreadPoolExecutor(max_workers=4) as pool:
        times = sorted(pool.map(run, range(4)))
    assert len(calls) == 1
    assert all(after - before >= 0.045 for before, after in zip(times, times[1:], strict=False))


def test_groups_are_isolated_and_conflicting_settings_are_rejected():
    first = admission()
    different_key = local.local_request_admission(
        model='hy3-offline', api_key='other-key', config={'quota_qpm': 600}, client_id='other'
    )
    different_model = local.local_request_admission(
        model='hy4-preview-offline', api_key='test-key', config={'quota_qpm': 600}, client_id='other'
    )
    assert first.admission is not different_key.admission
    assert first.admission is not different_model.admission
    with pytest.raises(ValueError, match='settings must match'):
        admission(quota_qpm=300)


@pytest.mark.parametrize(
    'config',
    [
        {'quota_qpm': 0},
        {'quota_qpm': True},
        {'quota_qpm': float('nan')},
        {'poll_interval': 59},
        {'poll_interval': float('inf')},
        {'max_age_seconds': 60},
        {'capacity_url': 'file:///tmp/capacity'},
    ],
)
def test_invalid_local_settings_fail_before_network(config):
    with pytest.raises(ValueError, match='validation error'):
        admission(**config)


def client(http_client, **client_args):
    return infer_proxy.InferProxyClient(
        OpenAIConfig(
            client_args={
                'api_key': 'test-key',
                'base_url': 'https://provider.test/v1',
                'http_client': http_client,
                'max_retries': 1,
                'local_request_policy': {'quota_qpm': 6000},
                **client_args,
            },
            request_args={'model': 'hy3-offline'},
        )
    )


def completion(request):
    return httpx.Response(
        200,
        json={
            'id': 'test',
            'object': 'chat.completion',
            'created': 1,
            'model': 'hy3-offline',
            'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'ok'}, 'finish_reason': 'stop'}],
        },
    )


async def test_local_client_is_lazy_and_retries_reacquire_without_sdk_retries(monkeypatch):
    calls = capacity(monkeypatch, [{}])
    permits, sends = [], []
    original_acquire = local.InferProxyLocalAdmission.acquire

    async def acquire(self, request):
        permit = await original_acquire(self, request)
        permits.append(permit)
        return permit

    def handler(request):
        sends.append(request)
        assert len(permits) == len(sends)
        return httpx.Response(500, json={'error': 'retry'}) if len(sends) == 1 else completion(request)

    monkeypatch.setattr(local.InferProxyLocalAdmission, 'acquire', acquire)
    monkeypatch.setattr(
        infer_proxy,
        'configurable_retry',
        lambda count: retry(stop=stop_after_attempt(count), wait=wait_none(), reraise=True),
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        instance = client(http_client, max_retries=2)
        assert not local._admissions
        response = await instance._create_completion([])
    assert response.choices[0].message.content == 'ok'
    assert len(permits) == len(sends) == 2
    assert len(calls) == 1


@pytest.mark.parametrize('bind_before_creation', [True, False])
async def test_online_binding_ignores_local_config_and_never_falls_back(bind_before_creation):
    class OnlineAdmission:
        calls = 0

        async def acquire(self, request: RequestAttempt):
            self.calls += 1
            raise RuntimeError('online paused')

    online = OnlineAdmission()
    async with httpx.AsyncClient(transport=httpx.MockTransport(completion)) as http_client:
        instance = None if bind_before_creation else client(http_client, local_request_policy={'invalid': True})
        with bind_request_admission(online, execution_id='online', attempt_id='1'):
            if instance is None:
                instance = client(http_client, local_request_policy={'invalid': True})
            with pytest.raises(RuntimeError, match='online paused'):
                await instance._create_completion([])
        assert online.calls == 1
        assert instance._local_admission is None
        assert not local._admissions


async def test_bound_online_client_keeps_online_admission_outside_context():
    class OnlineAdmission:
        calls = 0

        async def acquire(self, request):
            self.calls += 1
            return RequestPermit(request.execution_id, request.attempt_id, 'online')

    online = OnlineAdmission()
    async with httpx.AsyncClient(transport=httpx.MockTransport(completion)) as http_client:
        with bind_request_admission(online, execution_id='online', attempt_id='1'):
            instance = client(http_client, local_request_policy={'invalid': True})
        await instance._create_completion([])
    assert online.calls == 1
    assert not local._admissions


async def test_unconfigured_client_and_model_override():
    async with httpx.AsyncClient(transport=httpx.MockTransport(completion)) as http_client:
        await client(http_client, local_request_policy=None)._create_completion([])
        with pytest.raises(ValueError, match='configured model'):
            await client(http_client, max_retries=15)._create_completion([], extra_body={'model': 'other'})
        with pytest.raises(ValueError, match='validation error'):
            await client(http_client, max_retries=15, local_request_policy={'quota_qpm': 0})._create_completion([])
    assert not local._admissions
