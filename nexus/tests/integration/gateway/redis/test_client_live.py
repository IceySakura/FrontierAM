"""Live Redis integration tests for RedisClient.

Requires the ``redis`` package. Server resolution order:

1. ``NEXUS_REDIS_URL`` if set (existing instance)
2. else auto-start local ``redis-server`` via ``start_local_redis()``
3. else skip when no server binary is available

Missing optional dependencies are skipped; startup and connection failures propagate.

    uv sync --extra redis
    uv run pytest -o addopts="" -v -s tests/integration/gateway/redis/test_client_live.py
"""

from __future__ import annotations

import os

from unittest.mock import Mock
from uuid import uuid4

import pytest


pytest.importorskip('redis')

pytestmark = pytest.mark.integration

from nexus.gateway.trajectory_store_backends.redis_store.client import RedisClient, connect_redis  # noqa: E402
from nexus.gateway.trajectory_store_backends.redis_store.local import (  # noqa: E402
    LocalRedis,
    redis_server_available,
    start_local_redis,
)
from nexus.gateway.trajectory_store_backends.redis_store.file_storage.deploy import LocalCluster  # noqa: E402
from tests.unit.gateway.redis_store.payloads import SamplePayload  # noqa: E402


def _cleanup_prefix(redis_client, prefix: str) -> None:
    pattern = f'{prefix}:*'.encode()
    for key in redis_client.scan_iter(match=pattern):
        redis_client.delete(key)


def _resolve_redis_url() -> tuple[str, LocalRedis | None]:
    """Return (url, optional LocalRedis handle to stop later)."""
    env_url = os.environ.get('NEXUS_REDIS_URL')
    if env_url:
        raw = connect_redis(env_url)
        try:
            raw.ping()
        finally:
            raw.close()
        return env_url, None

    if not redis_server_available():
        pytest.skip('redis-server not on PATH and NEXUS_REDIS_URL unset')

    local = start_local_redis()
    return local.url, local


@pytest.mark.parametrize('outcome', ['missing-binary', 'started', 'start-failed'])
def test_resolve_local_redis_url(monkeypatch, outcome):
    monkeypatch.delenv('NEXUS_REDIS_URL', raising=False)
    monkeypatch.setattr(f'{__name__}.redis_server_available', lambda: outcome != 'missing-binary')
    failure = OSError('redis-server failed to start')
    start = Mock(return_value=Mock(url='redis://127.0.0.1:12345/0'))
    start.side_effect = failure if outcome == 'start-failed' else None
    monkeypatch.setattr(f'{__name__}.start_local_redis', start)
    if outcome == 'started':
        assert _resolve_redis_url() == (start.return_value.url, start.return_value)
    elif outcome == 'missing-binary':
        with pytest.raises(pytest.skip.Exception, match='redis-server'):
            _resolve_redis_url()
        start.assert_not_called()
    else:
        with pytest.raises((OSError, pytest.skip.Exception)) as caught:
            _resolve_redis_url()
        assert caught.value is failure
    if outcome != 'missing-binary':
        start.assert_called_once_with()


@pytest.mark.parametrize('failure_at', [None, 'connect', 'ping'])
def test_resolve_explicit_redis_url(monkeypatch, failure_at):
    url = 'redis://configured:16379/0'
    monkeypatch.setenv('NEXUS_REDIS_URL', url)
    failure = ConnectionError('configured Redis is unavailable')
    raw = Mock()
    raw.ping.side_effect = failure if failure_at == 'ping' else None
    connect = Mock(return_value=raw, side_effect=failure if failure_at == 'connect' else None)
    start = Mock()
    monkeypatch.setattr(f'{__name__}.connect_redis', connect)
    monkeypatch.setattr(f'{__name__}.start_local_redis', start)
    if failure_at is None:
        assert _resolve_redis_url() == (url, None)
    else:
        with pytest.raises((ConnectionError, pytest.skip.Exception)) as caught:
            _resolve_redis_url()
        assert caught.value is failure
    connect.assert_called_once_with(url)
    start.assert_not_called()
    if failure_at != 'connect':
        raw.ping.assert_called_once_with()
        raw.close.assert_called_once_with()


@pytest.fixture
def live_redis_client():
    url, local = _resolve_redis_url()
    raw = connect_redis(url)
    prefix = f'nexus-redis-live-{uuid4().hex[:10]}'
    _cleanup_prefix(raw, prefix)
    try:
        with LocalCluster(url, num_data_units=2) as cluster, cluster.client(key_prefix=prefix) as client:
            yield client
    finally:
        _cleanup_prefix(raw, prefix)
        raw.close()
        if local is not None:
            local.stop()


def test_live_put_get_roundtrip(live_redis_client: RedisClient):
    ref = live_redis_client.put_node(SamplePayload(text='live-hello'))
    assert isinstance(ref, str)
    assert ref is not None
    assert live_redis_client.has_node(ref)[0]

    got = live_redis_client.get_node(ref, payload_type=SamplePayload)
    assert got.text == 'live-hello'


def test_live_put_get_multi(live_redis_client: RedisClient):
    payloads = [SamplePayload(text=f'live-{i}') for i in range(5)]
    refs = live_redis_client.put_multi_node(payloads)
    assert len(refs) == 5
    assert all(isinstance(r, str) for r in refs)

    ids = [r for r in refs if r is not None]
    assert len(set(ids)) == 5
    got = live_redis_client.get_multi_node([(ref, SamplePayload) for ref in refs])
    assert [g.text for g in got] == [f'live-{i}' for i in range(5)]
