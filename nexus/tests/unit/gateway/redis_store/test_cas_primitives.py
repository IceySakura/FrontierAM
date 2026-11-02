from unittest.mock import Mock

import pytest

from tests.unit.gateway.redis_store.fake_redis import FakeRedis
from tests.unit.gateway.redis_store.file_storage.test_publication import MemoryFS

from nexus.gateway.trajectory_store_backends.redis_store.client import RedisClient


def test_empty_primitives_do_no_io():
    redis = Mock(spec=FakeRedis)
    client = RedisClient(redis_client=redis, fs_client=MemoryFS())
    assert client.get_node_blobs([]) == []
    assert client.compare_and_set_many([]) == []
    assert client.delete_multi_node([]) is None
    assert redis.mock_calls == []


@pytest.mark.parametrize(
    ('operation', 'error', 'message'),
    [
        ('get_node_blobs', TypeError, 'node_id must be a string'),
        ('compare_and_set_many', TypeError, 'node_id must be a string'),
        ('delete_multi_node', TypeError, 'node_id must be a string'),
    ],
)
def test_primitives_validate_entire_batch_before_io(operation, error, message):
    redis = Mock(spec=FakeRedis)
    client = RedisClient(redis_client=redis, fs_client=MemoryFS())
    refs = ['valid', None]
    batch = refs if operation != 'compare_and_set_many' else [(ref, b'old', b'new') for ref in refs]
    with pytest.raises(error, match=message):
        getattr(client, operation)(batch)
    assert redis.mock_calls == []


@pytest.mark.parametrize('operation', ['get_node_blobs', 'compare_and_set_many', 'delete_multi_node'])
def test_primitives_propagate_connection_failures(operation):
    redis = Mock(spec=FakeRedis)
    failure = ConnectionError('connection lost')
    redis.mget.side_effect = failure
    redis.pipeline.side_effect = failure
    redis.execute_command.side_effect = failure
    client = RedisClient(redis_client=redis, fs_client=MemoryFS())
    ref = 'valid'
    batch = [ref] if operation != 'compare_and_set_many' else [(ref, b'old', b'new')]
    with pytest.raises(ConnectionError) as error:
        getattr(client, operation)(batch)
    assert error.value is failure
