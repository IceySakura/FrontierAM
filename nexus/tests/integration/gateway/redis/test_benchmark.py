"""The benchmark owns local FS services and only its optional local Redis."""

import socket
import sys

from types import SimpleNamespace
from urllib.parse import urlparse

import pytest

from nexus.gateway.trajectory_store_backends.redis_store import local
from nexus.gateway.trajectory_store_backends.redis_store.client import connect_redis
from nexus.gateway.trajectory_store_backends.redis_store.file_storage import LocalCluster
from tests.stress.redis import bench_put_get


pytestmark = pytest.mark.integration


@pytest.fixture
def benchmark_resources(monkeypatch):
    if not local.redis_server_available():
        pytest.skip('redis-server is required for benchmark tests')
    monkeypatch.delenv('CHIEF_IP', raising=False)
    resources = SimpleNamespace(servers=[], clusters=[], endpoints=[], stopped=[])
    start_redis = local.start_local_redis
    start_fs = LocalCluster.start
    stop_fs = LocalCluster.stop

    def track_redis():
        server = start_redis()
        resources.servers.append(server)
        address = urlparse(server.url)
        resources.endpoints.append((address.hostname, address.port))
        stop = server.stop

        def track_stop():
            resources.stopped.append('redis')
            stop()

        monkeypatch.setattr(server, 'stop', track_stop)
        return server

    def track_fs(cluster):
        resources.clusters.append(cluster)
        start_fs(cluster)
        for endpoint in [cluster.info.controller_endpoint, *cluster.info.data_unit_endpoints]:
            resources.endpoints.append((endpoint.host, endpoint.port))
        return cluster

    def track_fs_stop(cluster):
        resources.stopped.append('fs')
        stop_fs(cluster)

    monkeypatch.setattr(local, 'start_local_redis', track_redis)
    monkeypatch.setattr(LocalCluster, 'start', track_fs)
    monkeypatch.setattr(LocalCluster, 'stop', track_fs_stop)
    try:
        yield resources
        for address in resources.endpoints:
            with socket.socket() as connection:
                assert connection.connect_ex(address) != 0, f'benchmark left {address} running'
    finally:
        for cluster in resources.clusters:
            stop_fs(cluster)
        for server in resources.servers:
            local.LocalRedis.stop(server)


@pytest.mark.parametrize('tensor', [False, True], ids=['concurrent-records', 'cpu-tensors'])
def test_default_benchmark_runs_without_chief_ip(monkeypatch, capsys, benchmark_resources, tensor):
    arguments = ['bench_put_get', '--ops', '3', '--payload-bytes', '1']
    if tensor:
        pytest.importorskip('torch')
        arguments += ['--tensor-elements', '4', '--tensor-count', '2']
    else:
        arguments += ['--clients', '2']
    monkeypatch.setattr(sys, 'argv', arguments)

    bench_put_get.main()

    output = capsys.readouterr().out
    assert 'started local redis at redis://' in output
    assert 'put: qps=' in output
    assert 'get: qps=' in output
    if tensor:
        assert 'pickle.put:' in output
        assert 'one-key.get:' in output
    assert len(benchmark_resources.clusters) == 1
    assert benchmark_resources.stopped == ['fs', 'redis']


def test_benchmark_borrows_existing_redis(monkeypatch, benchmark_resources, redis_backend_config):
    url = redis_backend_config['url']
    prefix = 'benchmark-existing'
    monkeypatch.setattr(
        sys,
        'argv',
        ['bench_put_get', '--redis-url', url, '--redis-prefix', prefix, '--ops', '3', '--clients', '2'],
    )
    raw = connect_redis(url)
    try:
        raw.set('benchmark-sentinel', b'keep')
        bench_put_get.main()
        assert raw.ping()
        assert raw.get('benchmark-sentinel') == b'keep'
        assert len(list(raw.scan_iter(match=f'{prefix}:node:*'))) == 4
        assert benchmark_resources.servers == []
        assert len(benchmark_resources.clusters) == 1
        assert benchmark_resources.stopped == ['fs']
    finally:
        raw.delete('benchmark-sentinel')
        for key in raw.scan_iter(match=f'{prefix}:*'):
            raw.delete(key)
        raw.close()


def test_worker_failure_stops_fs_before_local_redis(monkeypatch, benchmark_resources):
    monkeypatch.setattr(sys, 'argv', ['bench_put_get', '--ops', '1'])

    def fail_worker(*args):
        raise RuntimeError('benchmark worker failed')

    monkeypatch.setattr(bench_put_get, '_worker_put', fail_worker)

    with pytest.raises(RuntimeError, match='benchmark worker failed'):
        bench_put_get.main()

    assert len(benchmark_resources.clusters) == 1
    assert benchmark_resources.stopped == ['fs', 'redis']
