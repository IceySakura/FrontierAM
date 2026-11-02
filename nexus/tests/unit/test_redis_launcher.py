"""Combined startup keeps the Redis driver separate from the detached FS owner."""

import importlib
import subprocess
import sys

from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest


@pytest.fixture
def launcher(monkeypatch):
    package = importlib.import_module('nexus.gateway.trajectory_store_backends.redis_store')
    deployment = object()
    nodes = {'192.0.2.1': {'NodeID': 'redis-worker'}}
    info = {'redis_url': 'redis+cluster://192.0.2.1:16379/0?nodes=192.0.2.1:16379,192.0.2.1:16380'}
    cluster = SimpleNamespace(
        connect_ray=Mock(),
        resolve_topology=Mock(return_value=deployment),
        node_map=Mock(return_value=nodes),
        start_cluster=Mock(),
        endpoint_info=Mock(return_value=info),
        publish_endpoints=Mock(),
        monitor_cluster=Mock(),
        run=Mock(return_value=0),
    )
    monkeypatch.setitem(sys.modules, f'{package.__name__}.cluster', cluster)
    monkeypatch.setattr(package, 'cluster', cluster, raising=False)
    module = importlib.import_module(f'{package.__name__}.launcher')
    child = Mock(return_value=subprocess.CompletedProcess([], 0))
    monkeypatch.setattr(module.subprocess, 'run', child)
    calls = Mock()
    for name, operation in vars(cluster).items():
        calls.attach_mock(operation, name)
    calls.attach_mock(child, 'fs')
    return SimpleNamespace(
        module=module, cluster=cluster, deployment=deployment, nodes=nodes, info=info, child=child, calls=calls
    )


def fs_command(action, *options):
    return [sys.executable, '-m', 'nexus.cli.main', 'redis', 'fs', action, *options]


@pytest.mark.parametrize('action', ['up', 'serve'])
def test_start_waits_for_redis_then_starts_fs_with_actual_cluster_url(launcher, action):
    assert launcher.module.run(action) == 0
    expected = [
        call.connect_ray(),
        call.resolve_topology(),
        call.node_map(launcher.deployment),
        call.start_cluster(launcher.nodes, launcher.deployment),
        call.endpoint_info(launcher.deployment),
        call.fs(
            fs_command(
                'up',
                '--redis-url',
                launcher.info['redis_url'],
                '--controller-port',
                '17000',
                '--startup-timeout',
                '60',
            ),
            check=False,
        ),
    ]
    if action == 'serve':
        expected.extend([
            call.publish_endpoints(launcher.info, launcher.deployment),
            call.monitor_cluster(launcher.nodes, launcher.deployment),
        ])
    assert launcher.calls.mock_calls == expected


@pytest.mark.parametrize('action', ['up', 'serve'])
def test_start_passes_only_explicit_redis_options_and_preserves_zero_data_unit_options(launcher, action):
    redis_options = {'primary_count': 6, 'replicas_per_primary': 0, 'redis_dir': '/mnt/redis'}
    assert (
        launcher.module.run(
            action,
            **redis_options,
            num_data_units=4,
            controller_port=17100,
            startup_timeout=120,
            ssd_offload_dir='/mnt/local ssd',
            memory_waterline_bytes=1024,
            memory_safe_line_bytes=0,
            request_workers=3,
            ssd_workers=5,
            copy_workers=7,
        )
        == 0
    )
    launcher.cluster.resolve_topology.assert_called_once_with(**redis_options)
    launcher.child.assert_called_once_with(
        fs_command(
            'up',
            '--redis-url',
            launcher.info['redis_url'],
            '--num-data-units',
            '4',
            '--controller-port',
            '17100',
            '--startup-timeout',
            '120',
            '--ssd-offload-dir',
            '/mnt/local ssd',
            '--memory-waterline-bytes',
            '1024',
            '--memory-safe-line-bytes',
            '0',
            '--request-workers',
            '3',
            '--ssd-workers',
            '5',
            '--copy-workers',
            '7',
        ),
        check=False,
    )


@pytest.mark.parametrize('action', ['up', 'serve'])
def test_fs_failure_returns_its_code_without_publishing_or_monitoring(launcher, action):
    launcher.child.return_value = subprocess.CompletedProcess([], 7)
    assert launcher.module.run(action) == 7
    launcher.cluster.start_cluster.assert_called_once_with(launcher.nodes, launcher.deployment)
    launcher.cluster.publish_endpoints.assert_not_called()
    launcher.cluster.monitor_cluster.assert_not_called()
    launcher.cluster.run.assert_not_called()


def test_redis_start_failure_does_not_start_fs(launcher):
    launcher.cluster.start_cluster.side_effect = RuntimeError('Redis is unavailable')
    with pytest.raises(RuntimeError, match='Redis is unavailable'):
        launcher.module.run('up')
    launcher.child.assert_not_called()


@pytest.mark.parametrize('fs_code', [0, 7])
def test_down_stops_fs_before_redis_and_preserves_failure(launcher, fs_code):
    launcher.child.return_value = subprocess.CompletedProcess([], fs_code)
    launcher.cluster.run.return_value = 3
    redis_options = {'primary_count': 6, 'redis_node_ips': '192.0.2.1,192.0.2.2', 'extra_replicas': 0}
    assert launcher.module.run('down', **redis_options, controller_port=17100) == (fs_code or 3)
    expected = [call.fs(fs_command('down', '--controller-port', '17100'), check=False)]
    if not fs_code:
        expected.append(call.run('down', **redis_options))
    assert launcher.calls.mock_calls == expected
