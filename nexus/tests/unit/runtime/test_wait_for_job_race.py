"""Tests for RemoteRuntime._wait_for_job reconciliation poll (SIO race fix)."""

import asyncio
import base64
import pickle

import pytest

from nexus.runtime.client.remote import RemoteRuntime
from nexus.runtime.models import InvokeJobInfo, InvokeRequest, InvokeResponse, JobStatus


def _ok_response(rid: str, value) -> InvokeResponse:
    """Build a successful InvokeResponse carrying *value* (pickled)."""
    return InvokeResponse(
        request_id=rid,
        success=True,
        result=base64.b64encode(pickle.dumps(value)).decode('utf-8'),
    )


class _FakeHTTPResponse:
    """Minimal stand-in for nexus.utils.async_http.HTTPResponse."""

    def __init__(self, payload: dict):
        self._payload = payload
        self.status_code = 200

    def json(self, **kwargs) -> dict:
        return self._payload


def _make_runtime() -> RemoteRuntime:
    rt = RemoteRuntime(base_url='http://sandbox:8000', auto_connect_ws=True)
    rt._ws_connected = True  # force SIO branch
    return rt


@pytest.mark.asyncio
async def test_reconcile_poll_returns_result_when_event_was_lost(monkeypatch):
    """Event emitted before listener registration is lost; poll must recover it.

    No SIO push ever arrives. Without the reconciliation poll this would block
    until request_timeout. With it, the first poll reads `done` from _jobs and
    returns immediately.
    """
    rid = 'race-rid-1'
    rt = _make_runtime()

    job = InvokeJobInfo(request_id=rid, status=JobStatus.DONE, response=_ok_response(rid, 42))

    async def fake_get(url, **kwargs):
        assert url.endswith(f'/api/v1/invoke/{rid}')
        return _FakeHTTPResponse(job.model_dump(mode='json'))

    monkeypatch.setattr('nexus.runtime.client.remote.http_client.get', fake_get)

    result = await asyncio.wait_for(
        rt._wait_for_job(rid, request_timeout=5.0, timeout=rt._timeout),
        timeout=2.0,  # must return well under request_timeout
    )
    assert result == 42


@pytest.mark.asyncio
async def test_reconcile_poll_raises_on_error_job(monkeypatch):
    """A job whose status is `error` is recovered by the poll and raises."""
    rid = 'race-rid-err'
    rt = _make_runtime()

    err_response = InvokeResponse(
        request_id=rid,
        success=False,
        exception=base64.b64encode(pickle.dumps(ValueError('boom'))).decode('utf-8'),
        traceback='Traceback ...',
    )
    job = InvokeJobInfo(request_id=rid, status=JobStatus.ERROR, response=err_response)

    async def fake_get(url, **kwargs):
        return _FakeHTTPResponse(job.model_dump(mode='json'))

    monkeypatch.setattr('nexus.runtime.client.remote.http_client.get', fake_get)

    with pytest.raises(ValueError, match='boom'):
        await asyncio.wait_for(
            rt._wait_for_job(rid, request_timeout=5.0, timeout=rt._timeout),
            timeout=2.0,
        )


@pytest.mark.asyncio
async def test_poll_misses_then_sio_push_resolves(monkeypatch):
    """Poll returns `running`; a later SIO push (via the registered handler) resolves."""
    rid = 'race-rid-sio'
    rt = _make_runtime()

    async def fake_get(url, **kwargs):
        # Job not finished yet at poll time.
        return _FakeHTTPResponse(InvokeJobInfo(request_id=rid, status=JobStatus.RUNNING).model_dump(mode='json'))

    monkeypatch.setattr('nexus.runtime.client.remote.http_client.get', fake_get)

    # Capture the handler registered via rt.on(...) so we can fire it like SIO would.
    captured = {}
    real_on = rt.on

    def spy_on(event, listener):
        captured['event'] = event
        captured['listener'] = listener
        return real_on(event, listener)

    monkeypatch.setattr(rt, 'on', spy_on)

    async def fire_push_soon():
        # Wait until the listener is registered, then emit like the server would.
        for _ in range(100):
            if 'listener' in captured:
                break
            await asyncio.sleep(0.01)
        captured['listener'](_ok_response(rid, 'pushed').model_dump(mode='json'))

    pusher = asyncio.create_task(fire_push_soon())
    try:
        result = await asyncio.wait_for(
            rt._wait_for_job(rid, request_timeout=5.0, timeout=rt._timeout),
            timeout=3.0,
        )
        assert result == 'pushed'
    finally:
        pusher.cancel()


@pytest.mark.asyncio
async def test_poll_exception_is_non_fatal(monkeypatch):
    """If the reconciliation poll raises, fall through to SIO wait (not crash)."""
    rid = 'race-rid-pollerr'
    rt = _make_runtime()

    async def fake_get(url, **kwargs):
        raise ConnectionError('poll endpoint down')

    monkeypatch.setattr('nexus.runtime.client.remote.http_client.get', fake_get)

    captured = {}
    real_on = rt.on

    def spy_on(event, listener):
        captured['listener'] = listener
        return real_on(event, listener)

    monkeypatch.setattr(rt, 'on', spy_on)

    async def fire_push_soon():
        for _ in range(100):
            if 'listener' in captured:
                break
            await asyncio.sleep(0.01)
        captured['listener'](_ok_response(rid, 'recovered').model_dump(mode='json'))

    pusher = asyncio.create_task(fire_push_soon())
    try:
        result = await asyncio.wait_for(
            rt._wait_for_job(rid, request_timeout=5.0, timeout=rt._timeout),
            timeout=3.0,
        )
        assert result == 'recovered'
    finally:
        pusher.cancel()


@pytest.mark.asyncio
@pytest.mark.parametrize('request_timeout', [5.0, None])
async def test_backstop_poll_recovers_lost_push_after_first_poll_misses(monkeypatch, request_timeout):
    """A later poll recovers a completed job when its SIO push is lost."""
    rid = 'race-rid-backstop'
    rt = _make_runtime()
    rt.config.backstop_poll_interval = 0.01
    poll_calls = 0
    job_running = InvokeJobInfo(request_id=rid, status=JobStatus.RUNNING)
    job_done = InvokeJobInfo(request_id=rid, status=JobStatus.DONE, response=_ok_response(rid, 'backstopped'))

    async def fake_get(url, **kwargs):
        nonlocal poll_calls
        poll_calls += 1
        payload = job_running if poll_calls == 1 else job_done
        return _FakeHTTPResponse(payload.model_dump(mode='json'))

    monkeypatch.setattr('nexus.runtime.client.remote.http_client.get', fake_get)

    result = await asyncio.wait_for(
        rt._wait_for_job(rid, request_timeout=request_timeout, timeout=rt._timeout),
        timeout=2.0,
    )

    assert result == 'backstopped'
    assert poll_calls == 2


@pytest.mark.asyncio
async def test_backstop_poll_honors_request_deadline(monkeypatch):
    rid = 'race-rid-timeout'
    rt = _make_runtime()
    rt.config.backstop_poll_interval = 0.01

    async def fake_get(url, **kwargs):
        return _FakeHTTPResponse(InvokeJobInfo(request_id=rid, status=JobStatus.RUNNING).model_dump(mode='json'))

    monkeypatch.setattr('nexus.runtime.client.remote.http_client.get', fake_get)

    with pytest.raises(TimeoutError, match=f'invoke: job {rid} timed out'):
        await asyncio.wait_for(
            rt._wait_for_job(rid, request_timeout=0.05, timeout=rt._timeout),
            timeout=1.0,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize('request_timeout', [30.0, None])
async def test_backstop_poll_preserves_external_cancellation(monkeypatch, request_timeout):
    rid = 'race-rid-cancel'
    rt = _make_runtime()
    rt.config.backstop_poll_interval = 10.0
    first_poll_done = asyncio.Event()

    async def fake_get(url, **kwargs):
        first_poll_done.set()
        return _FakeHTTPResponse(InvokeJobInfo(request_id=rid, status=JobStatus.RUNNING).model_dump(mode='json'))

    monkeypatch.setattr('nexus.runtime.client.remote.http_client.get', fake_get)
    wait_task = asyncio.create_task(rt._wait_for_job(rid, request_timeout=request_timeout, timeout=rt._timeout))
    await first_poll_done.wait()
    await asyncio.sleep(0)

    wait_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await wait_task
    assert f'invoke:done:{rid}' not in rt._event_listeners


@pytest.mark.asyncio
@pytest.mark.parametrize('request_timeout', [1.0, None])
async def test_backstop_poll_does_not_swallow_child_cancellation(monkeypatch, request_timeout):
    rid = 'race-rid-child-cancel'
    rt = _make_runtime()

    async def cancelled_get(url, **kwargs):
        get_future = asyncio.get_running_loop().create_future()
        get_future.cancel()
        await get_future

    monkeypatch.setattr('nexus.runtime.client.remote.http_client.get', cancelled_get)

    with pytest.raises(RuntimeError, match='phase=invoke_backstop_get'):
        await rt._wait_for_job(rid, request_timeout=request_timeout, timeout=rt._timeout)


@pytest.mark.asyncio
async def test_disabled_invoke_wait_timeout_waits_for_sio_result(monkeypatch):
    """A caller-owned active deadline may disable the redundant wall timeout."""
    rid = 'no-wall-timeout-rid'
    rt = _make_runtime()

    async def fake_get(url, **kwargs):
        return _FakeHTTPResponse(InvokeJobInfo(request_id=rid, status=JobStatus.RUNNING).model_dump(mode='json'))

    monkeypatch.setattr('nexus.runtime.client.remote.http_client.get', fake_get)
    captured = {}
    real_on = rt.on

    def spy_on(event, listener):
        captured['listener'] = listener
        return real_on(event, listener)

    monkeypatch.setattr(rt, 'on', spy_on)

    async def fire_push_soon():
        for _ in range(100):
            if 'listener' in captured:
                break
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.05)
        captured['listener'](_ok_response(rid, 'completed').model_dump(mode='json'))

    pusher = asyncio.create_task(fire_push_soon())
    try:
        result = await asyncio.wait_for(
            rt._wait_for_job(rid, request_timeout=None, timeout=rt._timeout),
            timeout=2.0,
        )
        assert result == 'completed'
    finally:
        pusher.cancel()


@pytest.mark.asyncio
async def test_invoke_disables_only_cumulative_wait_timeout(monkeypatch):
    """The client-only flag must preserve every HTTP transport timeout."""
    rt = _make_runtime()
    captured = {}

    async def skip_auto_connect():
        return None

    async def fake_post(url, **kwargs):
        captured['post_timeout'] = kwargs['timeout'].total
        captured['prefer'] = kwargs['headers']['Prefer']
        captured['wire_payload'] = kwargs['json']
        response = _FakeHTTPResponse({})
        response.status_code = 202
        return response

    async def fake_wait_for_job(rid, *, request_timeout, timeout):
        captured['request_timeout'] = request_timeout
        captured['wait_timeout'] = timeout.total
        return 'completed'

    monkeypatch.setattr(rt, '_try_auto_connect_sio', skip_auto_connect)
    monkeypatch.setattr(rt, '_wait_for_job', fake_wait_for_job)
    monkeypatch.setattr('nexus.runtime.client.remote.http_client.post', fake_post)

    result = await rt._invoke(
        InvokeRequest(
            request_id='disabled-invoke-wait-timeout',
            import_path='fake.module::run',
            disable_invoke_wait_timeout=True,
        )
    )

    assert result == 'completed'
    assert captured['post_timeout'] == rt.config.request_timeout
    assert captured['prefer'] == f'wait={rt.config.invoke_wait}'
    assert captured['request_timeout'] is None
    assert captured['wait_timeout'] == rt.config.request_timeout
    assert 'disable_invoke_wait_timeout' not in captured['wire_payload']


@pytest.mark.asyncio
@pytest.mark.parametrize('sio_connected', [False, True])
async def test_finite_request_timeout_still_expires(monkeypatch, sio_connected):
    """Disabling one Codex invoke must not weaken the default transport guard."""
    rid = 'finite-timeout-rid'
    rt = _make_runtime()
    rt._ws_connected = sio_connected
    rt.config.poll_interval = 0.01

    async def fake_get(url, **kwargs):
        return _FakeHTTPResponse(InvokeJobInfo(request_id=rid, status=JobStatus.RUNNING).model_dump(mode='json'))

    monkeypatch.setattr('nexus.runtime.client.remote.http_client.get', fake_get)

    with pytest.raises(TimeoutError, match='timed out'):
        await rt._wait_for_job(rid, request_timeout=0.05, timeout=rt._timeout)


@pytest.mark.asyncio
async def test_disabled_invoke_wait_timeout_supports_http_polling(monkeypatch):
    """The no-wall-timeout mode must work when SIO is unavailable."""
    rid = 'no-wall-timeout-polling-rid'
    rt = _make_runtime()
    rt._ws_connected = False
    rt.config.poll_interval = 0.01
    poll_count = 0

    async def fake_get(url, **kwargs):
        nonlocal poll_count
        poll_count += 1
        if poll_count == 1:
            return _FakeHTTPResponse(InvokeJobInfo(request_id=rid, status=JobStatus.RUNNING).model_dump(mode='json'))
        return _FakeHTTPResponse(
            InvokeJobInfo(
                request_id=rid,
                status=JobStatus.DONE,
                response=_ok_response(rid, 'polled'),
            ).model_dump(mode='json')
        )

    monkeypatch.setattr('nexus.runtime.client.remote.http_client.get', fake_get)

    result = await asyncio.wait_for(
        rt._wait_for_job(rid, request_timeout=None, timeout=rt._timeout),
        timeout=2.0,
    )

    assert result == 'polled'
    assert poll_count == 2


@pytest.mark.parametrize('sio_connected', [False, True])
async def test_disabled_wait_deadline_still_stops_for_unhealthy_runtime(monkeypatch, sio_connected):
    from nexus.runtime.client.base import RuntimeUnhealthyError

    runtime = _make_runtime()
    runtime._ws_connected = sio_connected
    runtime.config.poll_interval = 0.01
    polled = asyncio.Event()

    async def fake_get(url, **kwargs):
        polled.set()
        return _FakeHTTPResponse(
            InvokeJobInfo(request_id='unhealthy', status=JobStatus.RUNNING).model_dump(mode='json')
        )

    monkeypatch.setattr('nexus.runtime.client.remote.http_client.get', fake_get)
    waiter = asyncio.create_task(runtime._wait_for_job('unhealthy', request_timeout=None, timeout=runtime._timeout))
    try:
        await asyncio.wait_for(polled.wait(), timeout=1.0)
        runtime.mark_unhealthy('sandbox exited')
        with pytest.raises(RuntimeUnhealthyError, match='sandbox exited'):
            await asyncio.wait_for(waiter, timeout=1.0)
    finally:
        waiter.cancel()
        await asyncio.gather(waiter, return_exceptions=True)


async def test_unbounded_polling_keeps_only_one_health_waiter(monkeypatch):
    class CountingEvent(asyncio.Event):
        def __init__(self):
            super().__init__()
            self.started = 0

        async def wait(self):
            self.started += 1
            return await super().wait()

    runtime = _make_runtime()
    runtime._ws_connected = False
    runtime.config.poll_interval = 0.01
    health = CountingEvent()
    runtime._unhealthy_event = health
    poll_count = 0

    async def fake_get(url, **kwargs):
        nonlocal poll_count
        poll_count += 1
        completed = poll_count == 3
        return _FakeHTTPResponse(
            InvokeJobInfo(
                request_id='health-waiter',
                status=JobStatus.DONE if completed else JobStatus.RUNNING,
                response=_ok_response('health-waiter', 'done') if completed else None,
            ).model_dump(mode='json')
        )

    monkeypatch.setattr('nexus.runtime.client.remote.http_client.get', fake_get)
    try:
        assert (
            await asyncio.wait_for(
                runtime._wait_for_job('health-waiter', request_timeout=None, timeout=runtime._timeout), timeout=1.0
            )
            == 'done'
        )
        assert health.started == 1, 'each polling tick must reuse the existing health watcher'
    finally:
        health.set()
