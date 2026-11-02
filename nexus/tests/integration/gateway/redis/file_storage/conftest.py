"""Clients borrow the shared TCP blob services for each integration test."""

from uuid import uuid4

import pytest


@pytest.fixture
def client(services, redis_backend_config):
    from nexus.gateway.trajectory_store_backends.redis_store.client import RedisClient

    with RedisClient(**redis_backend_config, fs_client=services.fs, key_prefix='hybrid-' + uuid4().hex) as client:
        yield client
