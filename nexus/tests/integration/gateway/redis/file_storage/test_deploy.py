"""Deployment helpers and the public example are separate from service tests."""

import json
import os
import socket
import subprocess
import sys

from pathlib import Path

import numpy as np
import pytest

from nexus.gateway.trajectory_store_backends.redis import NodeRecord
from nexus.gateway.trajectory_store_backends.redis_store.client import RedisClient, connect_redis
from nexus.gateway.trajectory_store_backends.redis_store.file_storage import (
    ClusterInfo,
    LocalCluster,
    start_local_cluster,
)


def test_local_cluster_entrypoint(redis_backend_config):
    borrowed = connect_redis(redis_backend_config['url'])
    try:
        with pytest.raises(TypeError, match='redis_client'):
            LocalCluster('redis://127.0.0.1:1/0', redis_client=borrowed)
        assert borrowed.ping()
    finally:
        borrowed.close()
    with start_local_cluster(redis_backend_config['url'], num_data_units=2) as cluster:
        for options in ({'redis_client': object()}, {'url': 'redis://127.0.0.1:1/0'}, {'fs_client': object()}):
            with pytest.raises(ValueError, match='cannot override'):
                cluster.client(**options)
        assert ClusterInfo.load(json.loads(json.dumps(cluster.info.dump()))) == cluster.info
        with cluster.client() as client:
            assert isinstance(client, RedisClient)
            payload = NodeRecord('run', 1, [1], routed_experts=b'data')
            ref = client.put_node(payload)
            assert isinstance(ref, str)
            restored = client.get_node(ref, payload_type=NodeRecord)
            assert type(restored) is NodeRecord
            np.testing.assert_equal(vars(restored), vars(payload))
            client.delete_multi_node([ref])
        with cluster.fs_client() as fs:
            assert len(fs.check_ready()) == 2
    assert cluster.info is None


def test_complete_example(redis_backend_config):
    root = Path(__file__).resolve().parents[5]
    result = subprocess.run(
        [
            sys.executable,
            str(root / 'docs/examples/redis_fs.py'),
            '--redis-url',
            redis_backend_config['url'],
            '--large-mib',
            '1',
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'Inline client: OK' in result.stdout
    assert 'Trajectory batch/projection/append/release: OK' in result.stdout


@pytest.fixture(scope='module')
def ray_runtime(tmp_path_factory):
    ray = pytest.importorskip('ray')
    ray.init(
        address='local',
        num_cpus=3,
        num_gpus=0,
        include_dashboard=False,
        object_store_memory=128 * 1024 * 1024,
        _node_ip_address='127.0.0.1',
        _temp_dir=str(tmp_path_factory.mktemp('ray-fs')),
    )
    try:
        yield ray
    finally:
        ray.shutdown()


def test_ray_cluster_cpu_actor_roundtrip(redis_backend_config, ray_runtime):
    from nexus.gateway.trajectory_store_backends.redis_store.file_storage import RayCluster

    node_id = ray_runtime.get_runtime_context().get_node_id()
    with RayCluster(
        redis_backend_config['url'], data_unit_node_ids=[node_id, node_id], controller_node_id=node_id
    ) as cluster:
        with cluster.client() as client:
            records = [
                NodeRecord('ray-check', index, [index], routed_experts=bytes([index]) * 1024 * 1024)
                for index in range(4)
            ]
            ids = client.put_multi_node(records)
            restored = client.get_multi_node([(node_id, NodeRecord) for node_id in ids])
            for actual, expected in zip(restored, records, strict=True):
                assert type(actual) is NodeRecord
                np.testing.assert_equal(vars(actual), vars(expected))
            client.delete_multi_node(ids)
        with cluster.fs_client() as fs:
            assert len(fs.check_ready()) == 2
            assert all(unit['alive'] for unit in fs.status()['units'])
    assert cluster.info is None


@pytest.mark.parametrize('injected_job', [False, True], ids=['direct', 'ray-job'])
def test_fs_up_survives_cli_exit_reuses_data_and_down_stops_services(redis_backend_config, ray_runtime, injected_job):
    root = Path(__file__).resolve().parents[5]
    host = next(node['NodeManagerAddress'] for node in ray_runtime.nodes() if node['Alive'])
    with socket.socket() as reserved:
        reserved.bind((host, 0))
        port = reserved.getsockname()[1]
    env = {
        **os.environ,
        'PYTHONPATH': str(root),
        'CHIEF_IP': host,
        'RAY_ADDRESS': ray_runtime.get_runtime_context().gcs_address,
        'RAY_OVERRIDE_JOB_RUNTIME_ENV': '1',
        'RAY_ENABLE_UV_RUN_RUNTIME_ENV': '0',
    }
    if injected_job:
        env['RAY_JOB_CONFIG_JSON_ENV_VAR'] = json.dumps({'runtime_env': {'env_vars': {'NEXUS_FS_TEST_JOB': '1'}}})
    else:
        env.pop('RAY_JOB_CONFIG_JSON_ENV_VAR', None)
    url = redis_backend_config['url'].replace('redis+cluster://', 'redis://', 1)

    def invoke(action):
        args = ['--redis-url', url] if action == 'up' else []
        return subprocess.run(
            [
                sys.executable,
                '-m',
                'nexus.cli.main',
                'redis',
                'fs',
                action,
                '--controller-port',
                str(port),
                *args,
            ],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
        )

    def endpoints(result):
        assert result.returncode == 0, result.stdout + result.stderr
        return next(
            json.loads(line)['redis_fs_endpoints']
            for line in result.stdout.splitlines()
            if line.startswith('{"redis_fs_endpoints":')
        )

    try:
        info = endpoints(invoke('up'))
        assert info['controller_endpoint'] == {'host': host, 'port': port}
        assert info['num_data_units'] == 1
        # The publishing process starts only after the up process has exited.
        with RedisClient(url=url, controller_endpoint=info['controller_endpoint']) as client:
            record = NodeRecord('managed-fs', 1, [1, 2], routed_experts=b'x' * 1024 * 1024)
            node_id = client.put_node(record)
            repeated = endpoints(invoke('up'))
            assert repeated == info
            restored = client.get_node(node_id, payload_type=NodeRecord)
            assert type(restored) is NodeRecord
            np.testing.assert_equal(vars(restored), vars(record))
            client.delete_node(node_id)
    finally:
        stopped = invoke('down')
    assert stopped.returncode == 0, stopped.stdout + stopped.stderr

    with socket.socket() as probe:
        assert probe.connect_ex((host, port)) != 0
    redis = connect_redis(url)
    try:
        assert redis.ping()
    finally:
        redis.close()
