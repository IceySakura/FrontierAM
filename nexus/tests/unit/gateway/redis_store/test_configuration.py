import os
import subprocess
import sys

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tests.unit.gateway.redis_store.fake_redis import FakeRedis
from tests.unit.gateway.redis_store.file_storage.test_publication import MemoryFS

from nexus.gateway.trajectory_store import build_trajectory_record_store
from nexus.gateway.trajectory_store_backends.redis import RedisTrajectoryRecordStore
from nexus.gateway.trajectory_store_backends.redis_store.client import (
    RedisClient,
    connect_redis,
    parse_cluster_url,
    parse_sentinel_url,
)
from nexus.gateway.trajectory_store_backends.types import NodePayloadRef, SegmentDataRef


def test_import_and_memory_backend_do_not_require_storage_extras():
    script = """
import importlib.abc
import sys

class BlockStorageDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'redis', 'torch', 'zmq', 'hy_kv_store', 'ray'}:
            raise ModuleNotFoundError(fullname)

sys.meta_path.insert(0, BlockStorageDependencies())
from nexus.gateway.trajectory_store import build_trajectory_record_store
from nexus.gateway.trajectory_store_backends.redis_store import RedisClient
store = build_trajectory_record_store({'backend': 'memory', 'namespace': 'run'})
assert store.namespace == 'run'
try:
    build_trajectory_record_store({
        'backend': 'redis', 'namespace': 'run',
        'redis': {'url': 'redis://localhost:16379/0', 'controller_endpoint': {'host': 'localhost'}},
    })
except ImportError as exc:
    assert 'uv sync --extra redis' in str(exc)
else:
    raise AssertionError('Redis construction must require redis-py')
"""
    repo = Path(__file__).resolve().parents[4]
    subprocess.run(
        [sys.executable, '-c', script],
        env={**os.environ, 'PYTHONPATH': str(repo)},
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize('config', [[], 'redis://localhost', 1])
def test_factory_rejects_non_mapping_redis_config(config):
    with pytest.raises(ValueError, match='trajectory_store.redis must be a mapping'):
        build_trajectory_record_store({'backend': 'redis', 'namespace': 'run', 'redis': config})


def test_factory_does_not_accept_old_backend_name():
    with pytest.raises(ValueError, match='unsupported trajectory_store.backend'):
        build_trajectory_record_store({'backend': 'hykv', 'namespace': 'run'})


def test_redis_default_prefix_and_explicit_override():
    with RedisClient(redis_client=FakeRedis(), fs_client=MemoryFS()) as client:
        assert client.key_prefix == 'nexus:trajectory'
    with RedisClient(redis_client=FakeRedis(), key_prefix='custom:', fs_client=MemoryFS()) as client:
        assert client.key_prefix == 'custom'


@pytest.mark.parametrize(
    'config',
    [
        {'io_workers': 0},
        {'io_workers': 1},
        {'io_workers': 1, 'fs_client': object()},
        {'controller_endpoint': {'host': 'host', 'port': 17000}, 'fs_client': object()},
    ],
)
def test_invalid_fs_configuration_fails_before_connect(monkeypatch, config):
    connect = Mock(side_effect=AssertionError('unexpected connection'))
    monkeypatch.setattr('nexus.gateway.trajectory_store_backends.redis_store.client.connect_redis', connect)
    with pytest.raises(ValueError, match='I/O workers'):
        RedisClient(**config)
    connect.assert_not_called()


@pytest.mark.parametrize('timeout', [10, None])
@pytest.mark.parametrize('as_mapping', [True, False])
def test_fs_endpoint_creates_owned_connection_and_borrowed_client_is_not_closed(monkeypatch, timeout, as_mapping):
    pytest.importorskip('zmq')
    from nexus.gateway.trajectory_store_backends.redis_store.file_storage.types import Endpoint

    fs = Mock()
    factory = Mock(return_value=fs)
    monkeypatch.setattr('nexus.gateway.trajectory_store_backends.redis_store.file_storage.client.FSClient', factory)
    monkeypatch.setitem(sys.modules, 'ray', None)
    endpoint = {'host': '127.0.0.1', 'port': 17000}
    with RedisClient(
        redis_client=FakeRedis(),
        controller_endpoint=endpoint if as_mapping else Endpoint(**endpoint),
        fs_timeout=timeout,
    ):
        factory.assert_called_once_with(Endpoint(**endpoint), timeout=timeout, io_threads=32)
    fs.close.assert_called_once_with()
    fs.reset_mock()
    with RedisClient(redis_client=FakeRedis(), fs_client=fs, io_workers=2):
        pass
    fs.close.assert_not_called()


@pytest.fixture
def ray_head(monkeypatch):
    ray = SimpleNamespace(
        is_initialized=Mock(return_value=True),
        init=Mock(),
        nodes=Mock(
            return_value=[
                {'Alive': True, 'NodeManagerAddress': '192.0.2.8', 'Resources': {'CPU': 8}},
                {'Alive': True, 'NodeManagerAddress': '192.0.2.7', 'Resources': {'node:__internal_head__': 1}},
            ]
        ),
    )
    monkeypatch.setitem(sys.modules, 'ray', ray)
    monkeypatch.setenv('CHIEF_IP', '203.0.113.1')
    return ray


@pytest.mark.parametrize(
    'endpoint', [None, {}, {'port': 17100}, {'host': 'override'}, {'host': 'override', 'port': 17100}]
)
def test_controller_defaults_are_applied_per_field(monkeypatch, ray_head, endpoint):
    from nexus.gateway.trajectory_store_backends.redis_store.file_storage.types import Endpoint

    fs = Mock()
    factory = Mock(return_value=fs)
    monkeypatch.setattr('nexus.gateway.trajectory_store_backends.redis_store.file_storage.client.FSClient', factory)
    with RedisClient(redis_client=FakeRedis(), controller_endpoint=endpoint) as client:
        expected = {'host': '192.0.2.7', 'port': 17000, **(endpoint or {})}
        factory.assert_called_once_with(Endpoint(**expected), timeout=120, io_threads=32)
        assert client._publication._fs is fs
    fs.close.assert_called_once_with()
    assert ray_head.nodes.call_count == int('host' not in (endpoint or {}))
    ray_head.init.assert_not_called()


def test_missing_controller_host_requires_an_existing_ray_session(monkeypatch, ray_head):
    ray_head.is_initialized.return_value = False
    raw = Mock()
    factory = Mock(side_effect=AssertionError('Ray discovery must fail before connecting FS'))
    monkeypatch.setattr('nexus.gateway.trajectory_store_backends.redis_store.file_storage.client.FSClient', factory)
    with pytest.raises(RuntimeError, match='initialized Ray'):
        RedisClient(redis_client=raw)
    raw.close.assert_called_once_with()
    factory.assert_not_called()
    ray_head.init.assert_not_called()
    ray_head.nodes.assert_not_called()


def test_default_controller_failure_is_not_a_plain_redis_fallback(monkeypatch, ray_head):
    raw = Mock()
    failure = ConnectionError('Controller is unavailable')
    factory = Mock(side_effect=failure)
    monkeypatch.setattr('nexus.gateway.trajectory_store_backends.redis_store.file_storage.client.FSClient', factory)
    with pytest.raises(ConnectionError, match='Controller is unavailable') as caught:
        RedisClient(redis_client=raw)
    assert caught.value is failure
    raw.close.assert_called_once_with()


def test_borrowed_fs_bypasses_missing_controller_defaults(monkeypatch):
    monkeypatch.setitem(sys.modules, 'ray', None)
    fs = Mock()
    factory = Mock(side_effect=AssertionError('borrowed FS must not connect'))
    monkeypatch.setattr('nexus.gateway.trajectory_store_backends.redis_store.file_storage.client.FSClient', factory)
    with RedisClient(redis_client=FakeRedis(), fs_client=fs) as client:
        assert client._publication._fs is fs
    factory.assert_not_called()
    fs.close.assert_not_called()


def test_default_redis_url_uses_ray_head(monkeypatch, ray_head):
    connect = Mock()
    monkeypatch.setattr('nexus.gateway.trajectory_store_backends.redis_store.client.connect_redis', connect)
    with RedisClient(fs_client=MemoryFS()) as client:
        assert client.url == 'redis://192.0.2.7:16379/0'
    connect.assert_called_once_with('redis://192.0.2.7:16379/0')
    ray_head.init.assert_not_called()


def test_default_redis_url_requires_an_existing_ray_session(monkeypatch, ray_head):
    ray_head.is_initialized.return_value = False
    connect = Mock()
    monkeypatch.setattr('nexus.gateway.trajectory_store_backends.redis_store.client.connect_redis', connect)
    with pytest.raises(RuntimeError, match='initialized Ray'):
        RedisClient(fs_client=MemoryFS())
    connect.assert_not_called()
    ray_head.init.assert_not_called()


def test_explicit_addresses_do_not_require_ray(monkeypatch):
    from nexus.gateway.trajectory_store_backends.redis_store.file_storage.types import Endpoint

    monkeypatch.setitem(sys.modules, 'ray', None)
    connect, fs = Mock(), Mock()
    factory = Mock(return_value=fs)
    monkeypatch.setattr('nexus.gateway.trajectory_store_backends.redis_store.client.connect_redis', connect)
    monkeypatch.setattr('nexus.gateway.trajectory_store_backends.redis_store.file_storage.client.FSClient', factory)
    with RedisClient('redis://explicit:16379/0', controller_endpoint={'host': 'controller'}):
        factory.assert_called_once_with(Endpoint('controller', 17000), timeout=120, io_threads=32)
    connect.assert_called_once_with('redis://explicit:16379/0')
    fs.close.assert_called_once_with()
    connect.return_value.close.assert_called_once_with()


@pytest.mark.parametrize('endpoint', ['192.0.2.7:17000', [], 17000])
def test_controller_endpoint_rejects_non_structured_values(endpoint):
    raw = Mock()
    with pytest.raises(TypeError, match='Endpoint or mapping'):
        RedisClient(redis_client=raw, controller_endpoint=endpoint)
    raw.close.assert_called_once_with()


def test_failed_fs_initialization_closes_redis(monkeypatch):
    pytest.importorskip('zmq')
    raw = Mock()
    factory = Mock(side_effect=ValueError('invalid timeout'))
    monkeypatch.setattr('nexus.gateway.trajectory_store_backends.redis_store.file_storage.client.FSClient', factory)
    with pytest.raises(ValueError, match='invalid timeout'):
        RedisClient(redis_client=raw, controller_endpoint={'host': 'localhost', 'port': 17000})
    raw.close.assert_called_once_with()


@pytest.mark.parametrize('namespace', ['', None])
def test_direct_store_rejects_empty_namespace(namespace):
    with pytest.raises(ValueError, match='namespace'):
        RedisTrajectoryRecordStore(namespace, {'redis_client': FakeRedis()})


@pytest.mark.parametrize('ref', [NodePayloadRef(), SegmentDataRef(), object()])
async def test_read_and_release_reject_foreign_refs(ref):
    with RedisTrajectoryRecordStore('run', {'redis_client': FakeRedis(), 'fs_client': MemoryFS()}) as store:
        with pytest.raises(TypeError, match='unsupported trajectory record ref'):
            await store.get_many([ref])
        with pytest.raises(TypeError, match='unsupported trajectory record ref'):
            await store.release_many([ref])


@pytest.mark.parametrize(
    'url',
    [
        'redis://service/0?sentinels=localhost:26379',
        'redis+sentinel:///0?sentinels=localhost:26379',
        'redis+sentinel://service/0',
        'redis+sentinel://service/0?sentinels=a:1&sentinels=b:2',
    ],
)
def test_sentinel_url_rejects_invalid_configuration(url):
    with pytest.raises(ValueError):
        parse_sentinel_url(url)


@pytest.mark.parametrize(
    'url', ['redis://localhost:16379/0', 'redis+cluster:///0', 'redis+cluster://localhost:16379/1']
)
def test_cluster_url_rejects_invalid_configuration(url):
    with pytest.raises(ValueError):
        parse_cluster_url(url)


def test_sentinel_connect_preserves_binary_resp2_client(monkeypatch):
    sentinel = Mock()
    monkeypatch.setattr('nexus.gateway.trajectory_store_backends.redis_store.client._require_redis', Mock())
    monkeypatch.setitem(sys.modules, 'redis.sentinel', SimpleNamespace(Sentinel=sentinel))
    client = connect_redis('redis+sentinel://primary/2?sentinels=a:26379,b:26380')
    sentinel.assert_called_once_with([('a', 26379), ('b', 26380)], sentinel_kwargs={'protocol': 2}, protocol=2)
    sentinel.return_value.master_for.assert_called_once_with('primary', db=2, decode_responses=False, protocol=2)
    assert client is sentinel.return_value.master_for.return_value


def test_cluster_connect_routes_startup_nodes_and_rejects_nonzero_db(monkeypatch):
    cluster = Mock()
    monkeypatch.setattr('nexus.gateway.trajectory_store_backends.redis_store.client._require_redis', Mock())
    monkeypatch.setitem(
        sys.modules,
        'redis.cluster',
        SimpleNamespace(
            RedisCluster=cluster,
            ClusterNode=lambda host, port: (host, port),
        ),
    )
    url = 'redis+cluster://a:16379/0?nodes=a:16379,b:16379'
    assert connect_redis(url) is cluster.return_value
    cluster.assert_called_once_with(startup_nodes=[('a', 16379), ('b', 16379)], decode_responses=False, protocol=2)
    with pytest.raises(ValueError, match='database 0'):
        connect_redis(url, db=1)
