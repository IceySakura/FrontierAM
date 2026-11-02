"""Real local Redis services for Gateway storage integration tests."""

import pytest

from tests.unit.gateway.redis_store.file_storage.test_publication import MemoryFS


@pytest.fixture(scope='module')
def redis_config():
    pytest.importorskip('redis')
    from nexus.gateway.trajectory_store_backends.redis_store.local import LocalRedis, redis_server_available

    if not redis_server_available():
        pytest.skip('redis-server is required for Redis integration')
    with LocalRedis() as server:
        fs = MemoryFS()
        yield {'url': server.url, 'fs_client': fs}
        fs.close()
