from __future__ import annotations

import asyncio
import copy
import json

from pathlib import Path
from typing import Any

import aiohttp
import pytest

from pydantic import ValidationError

from nexus.core.context import ExecutionContext
from nexus.runtime.client.models import WorkerVenvConfig
from nexus.runtime.client.remote import RemoteRuntime, http_client
from nexus.utils.async_http import HTTPResponse


WORKER_CONNECT_TIMEOUT = 'Worker did not connect within 30s'


def _venv_response(venv_id: str, *, state: str = 'ready', error: Any = None) -> dict[str, Any]:
    return {'venv_id': venv_id, 'state': state, 'error': error}


def _http_response(payload: Any, *, status: int = 200) -> HTTPResponse:
    text = json.dumps(payload)
    return HTTPResponse(status, {}, text, text.encode(), 'http://runtime.test/api/v1/venv', 'utf-8')


class VenvAPI:
    """HTTP boundary for the real RemoteRuntime create/destroy methods."""

    def __init__(self, runtime: RemoteRuntime, responses: list[dict[str, Any] | BaseException]) -> None:
        self.runtime = runtime
        self.responses = iter(responses)
        self.events: list[tuple[str, str]] = []
        self.specs: list[dict[str, Any]] = []
        self.cleanup_response: HTTPResponse | BaseException | None = None

    async def post(self, url: str, *, json: dict[str, Any], **_kwargs: Any) -> HTTPResponse:
        assert url == 'http://runtime.test/api/v1/venv'
        assert self.runtime.default_venv_id is None
        response = next(self.responses)
        self.specs.append(copy.deepcopy(json))
        if isinstance(response, BaseException):
            self.events.append(('create', 'exception'))
            raise response
        self.events.append(('create', response.get('venv_id', '')))
        return _http_response(response)

    async def delete(self, url: str, **_kwargs: Any) -> HTTPResponse:
        assert self.runtime.default_venv_id is None
        venv_id = url.rsplit('/', 1)[-1]
        self.events.append(('destroy', venv_id))
        if isinstance(self.cleanup_response, BaseException):
            raise self.cleanup_response
        if self.cleanup_response is not None:
            return self.cleanup_response
        return _http_response({'status': 'destroyed', 'venv_id': venv_id})


@pytest.fixture
def setup_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr('nexus.utils.otel.auto.is_enabled', lambda: False)
    wheel = tmp_path / 'nexus.whl'
    wheel.write_bytes(b'fixed-wheel')
    builds: list[Path] = []

    async def build_artifact(source: Path, **_kwargs: Any) -> Path:
        builds.append(source)
        return wheel

    monkeypatch.setattr('nexus.runtime.client.packaging.build_artifact_cached', build_artifact)

    def setup(responses: list[dict[str, Any] | BaseException]):
        runtime = RemoteRuntime(base_url='http://runtime.test', auto_connect_ws=False)
        api = VenvAPI(runtime, responses)
        monkeypatch.setattr(http_client, 'post', api.post)
        monkeypatch.setattr(http_client, 'delete', api.delete)
        config = {'source': str(tmp_path), 'modules': [], 'extra_requirements': ['example==1.0']}
        return runtime, api, config, builds

    return setup


async def test_worker_startup_failure_is_not_retried_by_default(setup_runtime):
    runtime, api, config, _builds = setup_runtime([
        _venv_response('failed', state='failed', error=WORKER_CONNECT_TIMEOUT),
        _venv_response('unused'),
    ])

    with pytest.raises(RuntimeError, match='Worker venv creation failed: Worker did not connect within 30s'):
        await ExecutionContext()._setup_worker_venv(runtime, config)

    assert api.events == [('create', 'failed')]
    assert runtime.default_venv_id is None


@pytest.mark.parametrize('error', [WORKER_CONNECT_TIMEOUT, f'{WORKER_CONNECT_TIMEOUT}; worker still running (rc=None)'])
async def test_worker_connect_timeout_destroys_failed_venv_before_fresh_startup(setup_runtime, error):
    runtime, api, config, builds = setup_runtime([
        _venv_response('failed', state='failed', error=error),
        _venv_response('fresh'),
    ])

    result = await ExecutionContext()._setup_worker_venv(runtime, {**config, 'startup_retries': 2})

    assert result is runtime
    assert runtime.default_venv_id == 'fresh'
    assert api.events == [('create', 'failed'), ('destroy', 'failed'), ('create', 'fresh')]
    assert len(builds) == 1
    assert api.specs[0] == api.specs[1]
    assert api.specs[0]['extra_requirements'] == ['example==1.0']
    assert 'startup_retries' not in api.specs[0]


@pytest.mark.parametrize(
    ('state', 'error', 'venv_id'),
    [
        ('failed', 'pip installation failed', 'venv'),
        ('failed', f'other failure: {WORKER_CONNECT_TIMEOUT}', 'venv'),
        ('failed', 'Worker did not connect within 30seconds', 'venv'),
        ('failed', 'Worker did not connect within 30s-like failure', 'venv'),
        ('failed', 'Worker did not connect within 60s', 'venv'),
        ('failed', None, 'venv'),
        ('failed', 30, 'venv'),
        ('creating', WORKER_CONNECT_TIMEOUT, 'venv'),
        ('unknown', WORKER_CONNECT_TIMEOUT, 'venv'),
        ('failed', WORKER_CONNECT_TIMEOUT, ''),
        ('failed', WORKER_CONNECT_TIMEOUT, None),
    ],
)
async def test_other_failed_or_incomplete_venv_results_are_never_retried(setup_runtime, state, error, venv_id):
    runtime, api, config, _builds = setup_runtime([_venv_response(venv_id, state=state, error=error)])

    with pytest.raises(RuntimeError, match='Worker venv creation failed:'):
        await ExecutionContext()._setup_worker_venv(runtime, {**config, 'startup_retries': 2})

    assert api.events == [('create', venv_id)]
    assert runtime.default_venv_id is None


async def test_worker_startup_retry_budget_exhaustion_preserves_failure(setup_runtime):
    runtime, api, config, _builds = setup_runtime([
        _venv_response(name, state='failed', error=WORKER_CONNECT_TIMEOUT) for name in ['first', 'second', 'last']
    ])

    with pytest.raises(RuntimeError) as error:
        await ExecutionContext()._setup_worker_venv(runtime, {**config, 'startup_retries': 2})

    assert str(error.value) == f'Worker venv creation failed: {WORKER_CONNECT_TIMEOUT}'
    assert api.events == [
        ('create', 'first'),
        ('destroy', 'first'),
        ('create', 'second'),
        ('destroy', 'second'),
        ('create', 'last'),
    ]
    assert runtime.default_venv_id is None


@pytest.mark.parametrize(
    ('status', 'payload', 'expected_error'),
    [
        (500, {'detail': 'stop failed'}, aiohttp.ClientResponseError),
        (200, {'status': 'failed', 'venv_id': 'failed'}, RuntimeError),
        (200, {'status': 'destroyed', 'venv_id': 'different'}, RuntimeError),
        (200, {'status': 'destroyed'}, RuntimeError),
        (200, ['destroyed'], RuntimeError),
    ],
)
async def test_cleanup_http_or_ack_failure_aborts_before_another_create(setup_runtime, status, payload, expected_error):
    runtime, api, config, _builds = setup_runtime([
        _venv_response('failed', state='failed', error=WORKER_CONNECT_TIMEOUT),
        _venv_response('must-not-start'),
    ])
    api.cleanup_response = _http_response(payload, status=status)

    with pytest.raises(expected_error):
        await ExecutionContext()._setup_worker_venv(runtime, {**config, 'startup_retries': 2})

    assert api.events == [('create', 'failed'), ('destroy', 'failed')]
    assert runtime.default_venv_id is None


@pytest.mark.parametrize(
    'error', [TimeoutError(WORKER_CONNECT_TIMEOUT), asyncio.CancelledError(), OSError('transport')]
)
async def test_create_exceptions_are_never_retried(setup_runtime, error):
    runtime, api, config, _builds = setup_runtime([error])

    with pytest.raises(type(error)):
        await ExecutionContext()._setup_worker_venv(runtime, {**config, 'startup_retries': 2})

    assert api.events == [('create', 'exception')]
    assert runtime.default_venv_id is None


@pytest.mark.parametrize('error', [TimeoutError('cleanup timeout'), asyncio.CancelledError()])
async def test_cleanup_exceptions_abort_startup_recovery(setup_runtime, error):
    runtime, api, config, _builds = setup_runtime([
        _venv_response('failed', state='failed', error=WORKER_CONNECT_TIMEOUT),
        _venv_response('must-not-start'),
    ])
    api.cleanup_response = error

    with pytest.raises(type(error)):
        await ExecutionContext()._setup_worker_venv(runtime, {**config, 'startup_retries': 2})

    assert api.events == [('create', 'failed'), ('destroy', 'failed')]
    assert runtime.default_venv_id is None


@pytest.mark.parametrize('value', [-1, 0.5, True, '2', None])
def test_startup_retry_budget_requires_a_nonnegative_integer(value):
    with pytest.raises(ValidationError, match='startup_retries'):
        WorkerVenvConfig(startup_retries=value)
