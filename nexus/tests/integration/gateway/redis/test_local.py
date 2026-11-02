"""Tests for LocalRedis launcher (skipped without redis-server)."""

from __future__ import annotations

import pytest

from nexus.gateway.trajectory_store_backends.redis_store.local import redis_server_available, start_local_redis


pytestmark = pytest.mark.integration


@pytest.mark.skipif(not redis_server_available(), reason='redis-server not on PATH')
def test_start_local_redis_accepts_ping():
    pytest.importorskip('redis')
    from nexus.gateway.trajectory_store_backends.redis_store.client import connect_redis

    local = start_local_redis()
    try:
        client = connect_redis(local.url)
        assert client.ping() is True
        client.close()
    finally:
        local.stop()
