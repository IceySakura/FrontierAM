"""An ordinary Redis URL discovers routing without hiding connection failures."""

import sys

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nexus.gateway.trajectory_store_backends.redis_store.client import connect_redis


@pytest.fixture
def connections(monkeypatch):
    direct = Mock()
    direct.info.return_value = {'cluster_enabled': 0}
    redis = SimpleNamespace(Redis=Mock())
    redis.Redis.from_url.return_value = direct
    cluster = Mock()
    monkeypatch.setattr('nexus.gateway.trajectory_store_backends.redis_store.client._require_redis', lambda: redis)
    monkeypatch.setitem(sys.modules, 'redis.cluster', SimpleNamespace(RedisCluster=cluster))
    return direct, redis.Redis.from_url, cluster.from_url


def test_standalone_detection_reuses_connection_and_preserves_database(connections):
    direct, factory, cluster = connections
    url = 'redis://localhost:16379/2?socket_timeout=3'
    assert connect_redis(url) is direct
    factory.assert_called_once_with(url, decode_responses=False, protocol=2)
    direct.info.assert_called_once_with('cluster')
    direct.close.assert_not_called()
    cluster.assert_not_called()


@pytest.mark.parametrize(
    'url',
    [
        'redis://chief:16379/0',
        'rediss://user:password@chief:16379/0?socket_timeout=3&ssl_cert_reqs=required',
    ],
)
def test_cluster_detection_preserves_url_and_connection_options(connections, url):
    direct, factory, cluster = connections
    direct.info.return_value = {'cluster_enabled': 1}
    assert connect_redis(url, socket_connect_timeout=2) is cluster.return_value
    options = {'decode_responses': False, 'protocol': 2, 'socket_connect_timeout': 2}
    factory.assert_called_once_with(url, **options)
    cluster.assert_called_once_with(url, **options)
    direct.info.assert_called_once_with('cluster')
    direct.close.assert_called_once_with()


@pytest.mark.parametrize('error', [ConnectionError('unreachable'), PermissionError('INFO denied'), KeyError('cluster')])
def test_detection_failure_closes_probe_and_propagates_error(connections, error):
    direct, factory, cluster = connections
    direct.info.side_effect = error
    with pytest.raises(type(error)) as caught:
        connect_redis('redis://chief:16379/0')
    assert caught.value is error
    assert factory.call_count == 1
    direct.close.assert_called_once_with()
    cluster.assert_not_called()


def test_cluster_initialization_failure_does_not_fall_back_to_direct_client(connections):
    direct, factory, cluster = connections
    direct.info.return_value = {'cluster_enabled': 1}
    error = ConnectionError('cluster unavailable')
    cluster.side_effect = error
    with pytest.raises(ConnectionError, match='cluster unavailable') as caught:
        connect_redis('redis://chief:16379/0')
    assert caught.value is error
    assert factory.call_count == 1
    direct.close.assert_called_once_with()
