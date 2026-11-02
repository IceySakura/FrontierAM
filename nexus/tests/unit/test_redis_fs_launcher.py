"""Ray FS launcher owns its services and joins an existing Ray runtime."""

import json
import logging
import signal
import sys

from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


pytest.importorskip('zmq')

from nexus.gateway.trajectory_store_backends.redis_store.file_storage import deploy
from nexus.gateway.trajectory_store_backends.redis_store.file_storage.types import (
    DataUnitRegistration,
    Endpoint,
    Topology,
)


@pytest.fixture(
    params=[
        {},
        {'memory_waterline_bytes': 1024},
        {'ssd_offload_dir': '/mnt/ssd/nexus', 'memory_waterline_bytes': 1024, 'memory_safe_line_bytes': 512},
        {'memory_waterline_bytes': None, 'memory_safe_line_bytes': None},
        {'memory_safe_line_bytes': 0},
        {'request_workers': 3, 'ssd_workers': 5, 'copy_workers': 7},
    ],
    ids=[
        'default-waterlines',
        'custom-waterline-default-safe-line',
        'explicit-options',
        'explicit-none',
        'zero-safe-line',
        'independent-io-workers',
    ],
)
def data_unit_options(request):
    return request.param


@pytest.fixture
def expected_data_unit_options(data_unit_options):
    return data_unit_options


def test_local_cluster_passes_data_unit_options_to_data_units(
    monkeypatch, data_unit_options, expected_data_unit_options
):
    unit = Mock(data_unit_id='unit-0', endpoint=Endpoint('127.0.0.1', 17001))
    controller = Mock(endpoint=Endpoint('127.0.0.1', 17000))
    controller.start.return_value = controller
    factory = Mock(return_value=unit)
    monkeypatch.setattr(deploy, 'connect_redis', Mock())
    monkeypatch.setattr(deploy, 'DataUnitService', factory)
    monkeypatch.setattr(deploy, 'ControllerService', Mock(return_value=controller))
    with deploy.LocalCluster(num_data_units=1, **data_unit_options):
        factory.assert_called_once_with('unit-0', Endpoint('127.0.0.1', 0), **expected_data_unit_options)


def test_data_unit_actor_passes_data_unit_options_on_worker(monkeypatch, expected_data_unit_options):
    monkeypatch.setitem(
        sys.modules, 'ray', SimpleNamespace(util=SimpleNamespace(get_node_ip_address=lambda: '192.0.2.1'))
    )
    service = Mock()
    service.start.return_value = service
    factory = Mock(return_value=service)
    monkeypatch.setattr(deploy, 'DataUnitService', factory)
    actor = deploy._DataUnitActor('unit-1', expected_data_unit_options)
    factory.assert_called_once_with('unit-1', Endpoint('192.0.2.1', 0), **expected_data_unit_options)
    assert actor.service is service


def test_ray_cluster_passes_data_unit_options_to_actor_constructors(
    monkeypatch, data_unit_options, expected_data_unit_options
):
    unit = Mock()
    unit.registration.remote.return_value = DataUnitRegistration('unit-0', Endpoint('192.0.2.1', 17001))
    controller = Mock()
    controller.endpoint.remote.return_value = Endpoint('192.0.2.1', 17000)
    unit_actor, controller_actor = Mock(), Mock()
    unit_actor.remote.return_value = unit
    controller_actor.remote.return_value = controller
    ray = SimpleNamespace(
        is_initialized=lambda: True,
        nodes=lambda: [],
        remote=lambda **kwargs: lambda actor: unit_actor if actor is deploy._DataUnitActor else controller_actor,
        get=lambda value, **kwargs: value,
        kill=Mock(),
    )
    monkeypatch.setitem(sys.modules, 'ray', ray)
    monkeypatch.setitem(
        sys.modules, 'ray.util.scheduling_strategies', SimpleNamespace(NodeAffinitySchedulingStrategy=Mock())
    )
    with deploy.RayCluster('redis://existing/0', num_data_units=1, **data_unit_options):
        unit_actor.remote.assert_called_once_with('unit-0', expected_data_unit_options)


def test_standalone_data_unit_passes_data_unit_options(
    monkeypatch, tmp_path, data_unit_options, expected_data_unit_options
):
    endpoint = Endpoint('127.0.0.1', 17001)
    topology = Topology.create([DataUnitRegistration('unit-0', endpoint)])
    path = tmp_path / 'topology.json'
    path.write_text(json.dumps(topology.dump()))
    factory = Mock(return_value=Mock(endpoint=endpoint))
    monkeypatch.setattr(deploy, 'DataUnitService', factory)
    monkeypatch.setattr(deploy.signal, 'signal', Mock())
    monkeypatch.setattr(deploy.threading, 'Event', lambda: SimpleNamespace(wait=lambda: None, set=lambda: None))
    assert deploy.serve_service('data-unit', str(path), unit_id='unit-0', **data_unit_options) == 0
    factory.assert_called_once_with('unit-0', endpoint, **expected_data_unit_options)


def test_fs_serve_prints_endpoints_and_cleans_owned_services(monkeypatch, capsys):
    events = []
    endpoint = {'host': '192.0.2.1', 'port': 17000}

    class Cluster:
        def __init__(self, **config):
            events.append(config)
            self.redis_url = config['redis_url']
            self.info = SimpleNamespace(dump=lambda: {'controller_endpoint': endpoint})

        def __enter__(self):
            events.append('start')
            return self

        def __exit__(self, *_):
            events.append('stop')

    ray = SimpleNamespace(init=Mock(), shutdown=lambda: events.append('shutdown'))
    monkeypatch.setitem(sys.modules, 'ray', ray)
    monkeypatch.setattr(deploy, 'RayCluster', Cluster)
    monkeypatch.setattr(
        deploy.threading, 'Event', lambda: SimpleNamespace(wait=lambda: events.append('wait'), set=lambda: None)
    )
    signals = Mock(return_value=signal.SIG_DFL)
    monkeypatch.setattr(deploy.signal, 'signal', signals)
    config = {'redis_url': 'redis://existing/0', 'num_data_units': 2, 'startup_timeout': 60}
    assert deploy.serve_ray_cluster(config) == 0
    ray.init.assert_called_once_with(address='auto', log_to_driver=False, logging_level=logging.ERROR)
    assert events == [config, 'start', 'wait', 'stop', 'shutdown']
    info = json.loads(capsys.readouterr().out)['redis_fs_endpoints']
    assert info == {'redis_url': config['redis_url'], 'controller_endpoint': endpoint}
    assert signals.call_count == 4


def test_fs_serve_cleans_up_when_driver_wait_fails(monkeypatch):
    events = []

    class Cluster:
        redis_url = 'redis://existing/0'
        info = SimpleNamespace(dump=lambda: {})

        def __enter__(self):
            return self

        def __exit__(self, *_):
            events.append('stop')

    ray = SimpleNamespace(init=Mock(), shutdown=lambda: events.append('shutdown'))
    monkeypatch.setitem(sys.modules, 'ray', ray)
    monkeypatch.setattr(deploy, 'RayCluster', lambda **config: Cluster())
    monkeypatch.setattr(deploy.signal, 'signal', lambda *args: None)
    monkeypatch.setattr(
        deploy.threading, 'Event', lambda: SimpleNamespace(wait=Mock(side_effect=RuntimeError('interrupted')))
    )
    with pytest.raises(RuntimeError, match='interrupted'):
        deploy.serve_ray_cluster({'redis_url': Cluster.redis_url})
    assert events == ['stop', 'shutdown']


def test_fs_config_defaults_and_rejects_unsupported_options(monkeypatch):
    from nexus.gateway.trajectory_store_backends.redis_store.file_storage import launcher

    cluster = deploy.RayCluster('redis://existing/0')
    assert cluster.count == 2
    assert cluster._startup_timeout == 60
    with pytest.raises(TypeError, match='redis_client'):
        deploy.LocalCluster(redis_client=object())
    with pytest.raises(TypeError, match='redis_client'):
        deploy.RayCluster('redis://existing/0', redis_client=object())
    with pytest.raises(TypeError, match='redis_client'):
        deploy.serve_service('controller', 'missing-topology.json', redis_client=object())
    monkeypatch.setitem(sys.modules, 'ray', None)
    with pytest.raises(TypeError, match='unsupported'):
        launcher.run('up', unsupported=None)
    with pytest.raises(ValueError, match='one Ray node ID'):
        deploy.RayCluster('redis://existing/0', data_unit_node_ids=['one'])


@pytest.mark.parametrize('port', [0, 1, 17000, 65535])
def test_ray_cluster_accepts_fixed_or_dynamic_controller_port(port):
    cluster = deploy.RayCluster('redis://existing/0', controller_port=port)
    assert cluster._controller_port == port


@pytest.mark.parametrize('port', [-1, 65536])
def test_ray_cluster_rejects_invalid_controller_port(port):
    with pytest.raises(ValueError, match='controller_port'):
        deploy.RayCluster('redis://existing/0', controller_port=port)


def test_managed_cluster_reuses_blobs_and_rejects_reconfiguration(monkeypatch):
    from nexus.gateway.trajectory_store_backends.redis_store.file_storage import launcher

    cluster = Mock(redis_url='redis://existing/0')
    cluster.start.return_value = cluster
    cluster.fs_client.return_value = nullcontext()
    cluster.info.dump.return_value = {'controller_endpoint': {'host': '192.0.2.1', 'port': 17000}}
    factory = Mock(return_value=cluster)
    monkeypatch.setattr(launcher, 'RayCluster', factory)
    manager = launcher._ClusterManager()
    config = {'redis_url': cluster.redis_url}
    first = manager.start(config)
    assert manager.start(config.copy()) == first
    factory.assert_called_once_with(**config)
    assert cluster.fs_client.call_count == 2
    with pytest.raises(ValueError, match='already running with different options'):
        manager.start({'redis_url': 'redis://another/0'})
    cluster.stop.assert_not_called()
    assert manager.start(config) == first
    manager.stop()
    manager.stop()
    cluster.stop.assert_called_once_with()


def test_managed_cluster_propagates_startup_failure_and_can_retry(monkeypatch):
    from nexus.gateway.trajectory_store_backends.redis_store.file_storage import launcher

    cluster = Mock()
    cluster.start.side_effect = RuntimeError('Redis is unavailable')
    monkeypatch.setattr(launcher, 'RayCluster', Mock(return_value=cluster))
    manager = launcher._ClusterManager()
    with pytest.raises(RuntimeError, match='Redis is unavailable'):
        manager.start({'redis_url': 'redis://existing/0'})
    assert manager._cluster is None


@pytest.fixture
def managed_ray(monkeypatch):
    manager = Mock()
    actor = Mock()
    actor.options.return_value.remote.return_value = manager
    ray = SimpleNamespace(
        init=Mock(),
        is_initialized=lambda: True,
        shutdown=Mock(),
        nodes=lambda: [
            {'NodeManagerAddress': '192.0.2.2', 'NodeID': 'worker', 'Alive': True, 'Resources': {}},
            {
                'NodeManagerAddress': '192.0.2.3',
                'NodeID': 'dead',
                'Alive': False,
                'Resources': {'node:__internal_head__': 1},
            },
            {
                'NodeManagerAddress': '192.0.2.1',
                'NodeID': 'chief',
                'Alive': True,
                'Resources': {'node:__internal_head__': 1},
            },
        ],
        util=SimpleNamespace(
            get_node_ip_address=Mock(return_value='192.0.2.2'),
            list_named_actors=Mock(return_value=['controller-192.0.2.1-17000']),
        ),
        remote=Mock(return_value=Mock(return_value=actor)),
        get=Mock(side_effect=lambda value, **kwargs: value),
        get_actor=Mock(return_value=manager),
        kill=Mock(),
    )
    monkeypatch.setitem(sys.modules, 'ray', ray)
    monkeypatch.setitem(
        sys.modules, 'ray.util.scheduling_strategies', SimpleNamespace(NodeAffinitySchedulingStrategy=Mock())
    )
    return ray, actor, manager


@pytest.mark.parametrize('chief_ip', [None, '192.0.2.2', '192.0.2.99'])
def test_up_discovers_defaults_and_retains_services_after_cli_exit(monkeypatch, capsys, managed_ray, chief_ip):
    from nexus.gateway.trajectory_store_backends.redis_store.file_storage import launcher

    if chief_ip is None:
        monkeypatch.delenv('CHIEF_IP', raising=False)
    else:
        monkeypatch.setenv('CHIEF_IP', chief_ip)
    ray, actor, manager = managed_ray
    manager.start.remote.return_value = {'controller_endpoint': {'host': '192.0.2.1', 'port': 17000}}
    assert launcher.run('up') == 0
    config = manager.start.remote.call_args.args[0]
    assert config['redis_url'] == 'redis://192.0.2.1:16379/0'
    assert config['controller_port'] == 17000
    assert config['num_data_units'] == 2
    assert config['controller_node_id'] == 'chief'
    assert config['data_unit_node_ids'] == ['chief', 'worker']
    assert 'ssd_offload_dir' not in config
    assert 'memory_waterline_bytes' not in config
    assert 'memory_safe_line_bytes' not in config
    assert actor.options.call_args.kwargs['lifetime'] == 'detached'
    assert actor.options.call_args.kwargs['get_if_exists'] is True
    assert actor.options.call_args.kwargs['name'] == 'controller-192.0.2.1-17000'
    affinity = sys.modules['ray.util.scheduling_strategies'].NodeAffinitySchedulingStrategy
    affinity.assert_called_once_with('chief', soft=False)
    assert actor.options.call_args.kwargs['runtime_env']['pip'] == ['redis>=8.1,<9', 'pyzmq>=25']
    runtime_env = deepcopy(ray.init.call_args.kwargs['runtime_env'])
    (package_path,) = runtime_env['py_modules']
    assert (Path(package_path) / '__init__.py').samefile(launcher.nexus.__file__)
    assert json.loads(capsys.readouterr().out)['redis_fs_endpoints'] == manager.start.remote.return_value
    manager.stop.remote.assert_not_called()
    ray.kill.assert_not_called()
    ray.util.get_node_ip_address.assert_not_called()
    ray.shutdown.assert_called_once_with()


@pytest.mark.parametrize('node_count', [1, 3, 7])
def test_up_starts_one_data_unit_per_alive_node(monkeypatch, managed_ray, node_count):
    from nexus.gateway.trajectory_store_backends.redis_store.file_storage import launcher

    monkeypatch.setenv('CHIEF_IP', '192.0.2.1')
    ray, actor, manager = managed_ray
    alive = [
        {
            'NodeManagerAddress': f'192.0.2.{index + 1}',
            'NodeID': f'node-{index}',
            'Alive': True,
            'Resources': {'node:__internal_head__': 1} if index == 0 else {},
        }
        for index in range(node_count)
    ]
    ray.nodes = lambda: [
        {'NodeManagerAddress': '192.0.2.254', 'NodeID': 'dead', 'Alive': False, 'Resources': {}},
        *reversed(alive),
    ]
    manager.start.remote.return_value = {}
    assert launcher.run('up') == 0
    config = manager.start.remote.call_args.args[0]
    assert config['num_data_units'] == node_count
    assert config['data_unit_node_ids'] == [node['NodeID'] for node in alive]
    assert config['controller_node_id'] == 'node-0'


@pytest.mark.parametrize(('num_data_units', 'node_ids'), [(1, ['chief']), (4, ['chief', 'worker', 'chief', 'worker'])])
def test_up_respects_explicit_data_unit_count(monkeypatch, managed_ray, num_data_units, node_ids):
    from nexus.gateway.trajectory_store_backends.redis_store.file_storage import launcher

    monkeypatch.setenv('CHIEF_IP', '192.0.2.1')
    ray, actor, manager = managed_ray
    manager.start.remote.return_value = {}
    assert launcher.run('up', num_data_units=num_data_units) == 0
    config = manager.start.remote.call_args.args[0]
    assert config['num_data_units'] == num_data_units
    assert config['data_unit_node_ids'] == node_ids


def test_up_passes_data_unit_options_to_managed_cluster(
    monkeypatch, managed_ray, data_unit_options, expected_data_unit_options
):
    from nexus.gateway.trajectory_store_backends.redis_store.file_storage import launcher

    monkeypatch.setenv('CHIEF_IP', '192.0.2.1')
    ray, actor, manager = managed_ray
    manager.start.remote.return_value = {}
    assert launcher.run('up', **data_unit_options) == 0
    config = manager.start.remote.call_args.args[0]
    assert {
        name: value
        for name, value in config.items()
        if name
        in (
            'ssd_offload_dir',
            'memory_waterline_bytes',
            'memory_safe_line_bytes',
            'request_workers',
            'ssd_workers',
            'copy_workers',
        )
    } == expected_data_unit_options


@pytest.mark.parametrize('action', ['up', 'down'])
def test_managed_commands_require_an_alive_ray_head_and_disconnect(monkeypatch, managed_ray, action):
    from nexus.gateway.trajectory_store_backends.redis_store.file_storage import launcher

    monkeypatch.setenv('CHIEF_IP', '192.0.2.99')
    ray, actor, manager = managed_ray
    nodes = ray.nodes()
    for node in nodes:
        if node['Alive']:
            node['Resources'] = {}
    ray.nodes = lambda: nodes
    with pytest.raises(RuntimeError, match='head'):
        launcher.run(action)
    actor.options.assert_not_called()
    ray.get_actor.assert_not_called()
    ray.shutdown.assert_called_once_with()


@pytest.mark.parametrize('port', [0, -1, 65536])
def test_managed_service_requires_a_fixed_valid_port(managed_ray, port):
    from nexus.gateway.trajectory_store_backends.redis_store.file_storage import launcher

    ray, actor, manager = managed_ray
    with pytest.raises(ValueError, match='controller_port'):
        launcher.run('up', controller_port=port)
    ray.init.assert_not_called()


@pytest.mark.parametrize('stop_fails', [False, True])
def test_down_stops_the_owner_even_when_service_shutdown_fails(monkeypatch, managed_ray, stop_fails):
    from nexus.gateway.trajectory_store_backends.redis_store.file_storage import launcher

    monkeypatch.setenv('CHIEF_IP', '192.0.2.99')
    ray, actor, manager = managed_ray
    if stop_fails:
        ray.get.side_effect = RuntimeError('service stopped unexpectedly')
        with pytest.raises(RuntimeError, match='service stopped unexpectedly'):
            launcher.run('down')
    else:
        assert launcher.run('down') == 0
    ray.get_actor.assert_called_once_with('controller-192.0.2.1-17000')
    manager.stop.remote.assert_called_once_with()
    ray.kill.assert_called_once_with(manager, no_restart=True)
    ray.shutdown.assert_called_once_with()


@pytest.mark.parametrize('initially_running', [False, True])
def test_down_succeeds_when_owner_is_already_absent(monkeypatch, managed_ray, capsys, initially_running):
    from nexus.gateway.trajectory_store_backends.redis_store.file_storage import launcher

    monkeypatch.setenv('CHIEF_IP', '192.0.2.99')
    ray, actor, manager = managed_ray
    if initially_running:
        assert launcher.run('down') == 0
    ray.util.list_named_actors.return_value = []
    ray.get_actor.side_effect = ValueError('actor does not exist')
    assert launcher.run('down') == 0
    assert ray.get_actor.call_count == int(initially_running)
    assert manager.stop.remote.call_count == int(initially_running)
    assert ray.kill.call_count == int(initially_running)
    assert ray.shutdown.call_count == 1 + int(initially_running)
    stopped = [json.loads(line)['redis_fs_stopped'] for line in capsys.readouterr().out.splitlines()]
    assert stopped == [{'host': '192.0.2.1', 'port': 17000}] * (1 + int(initially_running))
