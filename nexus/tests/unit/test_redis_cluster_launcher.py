"""The packaged launcher retains topology, configuration and lifecycle semantics."""

import importlib
import json
import logging
import sys

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest


@pytest.fixture
def cluster(monkeypatch):
    monkeypatch.setenv('NODE_IP_LIST', ','.join(f'192.0.2.{i}:8' for i in range(1, 9)))
    monkeypatch.setenv('CHIEF_IP', '192.0.2.3')
    for name in ('NODE_IPS', 'CLIENT_IP', 'DIR', 'PRIMARY_COUNT', 'REPLICAS_PER_PRIMARY', 'EXTRA_REPLICAS'):
        monkeypatch.delenv('NEXUS_REDIS_' + name, raising=False)
    ray = SimpleNamespace(
        remote=lambda **options: lambda function: function,
        get=lambda refs: refs,
        util=SimpleNamespace(get_node_ip_address=lambda: '192.0.2.3'),
        init=Mock(),
        is_initialized=Mock(return_value=True),
        nodes=Mock(
            return_value=[
                {
                    'NodeManagerAddress': f'192.0.2.{index}',
                    'NodeID': f'node-{index}',
                    'Alive': True,
                    'Resources': {'node:__internal_head__': 1} if index == 3 else {},
                }
                for index in range(1, 9)
            ]
        ),
    )

    def initialize(**_options):
        ray.is_initialized.return_value = True

    ray.init.side_effect = initialize
    monkeypatch.setitem(sys.modules, 'ray', ray)
    monkeypatch.setitem(
        sys.modules, 'ray.util.scheduling_strategies', SimpleNamespace(NodeAffinitySchedulingStrategy=Mock())
    )
    name = 'nexus.gateway.trajectory_store_backends.redis_store.cluster'
    monkeypatch.delitem(sys.modules, name, raising=False)
    module = importlib.import_module(name)
    yield module
    sys.modules.pop(name, None)


@pytest.fixture
def deployment(cluster):
    return cluster.resolve_topology()


def test_explicit_topology_and_config_are_preserved(cluster):
    deployment = cluster.resolve_topology(primary_count=3, replicas_per_primary=1, extra_replicas=1)
    assert deployment.hosts == (
        '192.0.2.3',
        '192.0.2.1',
        '192.0.2.2',
        '192.0.2.4',
        '192.0.2.5',
        '192.0.2.6',
        '192.0.2.7',
    )
    assert deployment.client_ip == '192.0.2.8'
    assert (deployment.primary_count, deployment.replicas_per_primary, deployment.extra_replicas) == (3, 1, 1)
    assert deployment.remote_dir == Path('/tmp/hykv-redis-cluster')
    conf = cluster.redis_config('192.0.2.3')
    for line in (
        'bind 192.0.2.3 127.0.0.1',
        'port 16379',
        'appendonly yes',
        'appendfsync everysec',
        'cluster-announce-bus-port 26379',
        'cluster-config-file /tmp/hykv-redis-cluster/nodes.conf',
    ):
        assert line in conf
    assert json.loads(cluster.config_path().read_text()) == cluster.CONFIG
    assert cluster.config_path() == Path(cluster.__file__).with_name('redis_server_cluster.json')
    cluster.ray.init.assert_not_called()
    cluster.ray.nodes.assert_called_once_with()


def test_topology_validation_and_explicit_override(cluster):
    hosts = ','.join(f'198.51.100.{i}:1' for i in range(1, 9))
    deployment = cluster.resolve_topology(redis_node_ips=hosts, redis_client_ip='198.51.100.8')
    assert len(deployment.instances) == 8
    assert deployment.instances[0] == ('198.51.100.1', cluster.REDIS_PORT)
    assert deployment.client_ip == '198.51.100.8'
    with pytest.raises(RuntimeError, match='not in the node list'):
        cluster.resolve_topology(redis_node_ips=hosts, redis_client_ip='203.0.113.1')
    cluster.ray.init.assert_not_called()
    assert cluster.ray.nodes.call_count == 2


@pytest.mark.parametrize('chief_ip', [None, '192.0.2.8', 'not-a-ray-head'])
def test_topology_uses_ray_head_instead_of_chief_environment(cluster, monkeypatch, chief_ip):
    if chief_ip is None:
        monkeypatch.delenv('CHIEF_IP')
    else:
        monkeypatch.setenv('CHIEF_IP', chief_ip)
    deployment = cluster.resolve_topology()
    assert deployment.hosts == ('192.0.2.3', '192.0.2.1', '192.0.2.2', *(f'192.0.2.{i}' for i in range(4, 9)))
    assert deployment.client_ip == '192.0.2.3'
    cluster.ray.init.assert_not_called()


@pytest.mark.parametrize('source', ['argument', 'taiji'])
def test_explicit_host_resolution_requires_a_connected_ray_session(cluster, source):
    cluster.ray.is_initialized.return_value = False
    options = {'redis_node_ips': '192.0.2.1'} if source == 'argument' else {}
    with pytest.raises(RuntimeError, match='initialized Ray'):
        cluster.resolve_topology(**options)
    cluster.ray.init.assert_not_called()
    cluster.ray.nodes.assert_not_called()


def test_up_from_non_head_driver_uses_ray_head(cluster, monkeypatch):
    monkeypatch.setenv('CHIEF_IP', '192.0.2.8')
    monkeypatch.setattr(cluster.ray.util, 'get_node_ip_address', lambda: '192.0.2.8')
    start = Mock()
    monkeypatch.setattr(cluster, 'start_cluster', start)
    assert cluster.run('up') == 0
    nodes, deployment = start.call_args.args
    assert nodes['192.0.2.3']['Resources'] == {'node:__internal_head__': 1}
    assert deployment.hosts[0] == deployment.client_ip == '192.0.2.3'


@pytest.mark.parametrize('source', ['argument', 'taiji'])
def test_dedicated_redis_hosts_keep_order_without_adding_ray_head(cluster, monkeypatch, source):
    hosts = '192.0.2.6,192.0.2.4,192.0.2.7'
    monkeypatch.setenv('CHIEF_IP', '192.0.2.7')
    options = {'redis_node_ips': hosts} if source == 'argument' else {}
    if source == 'taiji':
        monkeypatch.setenv('NODE_IP_LIST', hosts)
    deployment = cluster.resolve_topology(**options)
    assert deployment.hosts == ('192.0.2.6', '192.0.2.4', '192.0.2.7')
    assert deployment.client_ip == '192.0.2.6'
    assert '192.0.2.3' not in deployment.hosts


def test_topology_fails_when_ray_has_no_alive_head(cluster):
    nodes = cluster.ray.nodes.return_value
    for node in nodes:
        node['Resources'] = {}
    nodes.append({
        'NodeManagerAddress': '192.0.2.254',
        'NodeID': 'dead-head',
        'Alive': False,
        'Resources': {'node:__internal_head__': 1},
    })
    with pytest.raises(RuntimeError, match='[Hh]ead'):
        cluster.resolve_topology()


@pytest.mark.parametrize('node_count', [1, 2, 3, 7])
@pytest.mark.parametrize('node_list', [None, ''])
def test_up_discovers_alive_ray_hosts_without_node_environment(cluster, monkeypatch, node_count, node_list):
    cluster.ray.is_initialized.return_value = False
    if node_list is None:
        monkeypatch.delenv('NODE_IP_LIST')
    else:
        monkeypatch.setenv('NODE_IP_LIST', node_list)
    alive = [
        {
            'NodeManagerAddress': f'192.0.2.{index}',
            'NodeID': f'node-{index}',
            'Alive': True,
            'Resources': {'node:__internal_head__': 1} if index == (3 if node_count >= 3 else 1) else {},
        }
        for index in range(1, node_count + 1)
    ]
    cluster.ray.nodes.return_value = [
        {'NodeManagerAddress': '192.0.2.254', 'NodeID': 'dead', 'Alive': False},
        *reversed(alive),
        alive[0],
    ]
    start = Mock()
    monkeypatch.setattr(cluster, 'start_cluster', start)
    assert cluster.run('up') == 0
    nodes, deployment = start.call_args.args
    expected_hosts = (
        ('192.0.2.3', *(item['NodeManagerAddress'] for item in alive if item['NodeID'] != 'node-3'))
        if node_count >= 3
        else tuple(item['NodeManagerAddress'] for item in alive)
    )
    assert deployment.hosts == expected_hosts
    assert deployment.client_ip == expected_hosts[0]
    assert deployment.node_count == deployment.primary_count == max(node_count, 3)
    assert deployment.replicas_per_primary == deployment.extra_replicas == 0
    assert nodes == {item['NodeManagerAddress']: item for item in alive}
    cluster.ray.init.assert_called_once_with(address='auto', log_to_driver=False, logging_level=logging.ERROR)


def test_discovery_reuses_an_initialized_ray_connection(cluster, monkeypatch):
    monkeypatch.delenv('NODE_IP_LIST')
    cluster.ray.is_initialized.return_value = True
    monkeypatch.setattr(cluster, 'start_cluster', Mock())
    assert cluster.run('up') == 0
    cluster.ray.init.assert_not_called()


def test_default_discovery_requires_at_least_one_alive_host(cluster, monkeypatch):
    monkeypatch.delenv('NODE_IP_LIST')
    cluster.ray.is_initialized.return_value = False
    cluster.ray.nodes.return_value = []
    with pytest.raises(RuntimeError, match='needs at least 1 task host'):
        cluster.resolve_topology()
    cluster.ray.init.assert_called_once_with(address='auto', log_to_driver=False, logging_level=logging.ERROR)


@pytest.mark.parametrize('source', ['argument', 'taiji'])
@pytest.mark.parametrize(
    ('hosts', 'instances'),
    [
        ('192.0.2.1', (('192.0.2.1', 16379), ('192.0.2.1', 16380), ('192.0.2.1', 16381))),
        ('192.0.2.1,192.0.2.2', (('192.0.2.1', 16379), ('192.0.2.2', 16379), ('192.0.2.1', 16380))),
    ],
)
def test_default_small_topology_places_three_primaries_round_robin(cluster, monkeypatch, source, hosts, instances):
    options = {'redis_node_ips': hosts} if source == 'argument' else {}
    if source == 'taiji':
        monkeypatch.setenv('NODE_IP_LIST', hosts)
    deployment = cluster.resolve_topology(**options)
    assert deployment.instances == instances
    assert deployment.primary_count == deployment.base_node_count == deployment.node_count == 3
    assert deployment.replicas_per_primary == deployment.extra_replicas == 0


def test_discovery_propagates_ray_connection_errors(cluster, monkeypatch):
    monkeypatch.delenv('NODE_IP_LIST')
    cluster.ray.is_initialized.return_value = False
    cluster.ray.init.side_effect = ConnectionError('Ray is unavailable')
    with pytest.raises(ConnectionError, match='Ray is unavailable'):
        cluster.resolve_topology()
    cluster.ray.nodes.assert_not_called()


def test_invalid_explicit_hosts_do_not_fall_back_to_discovery(cluster):
    with pytest.raises(ValueError, match='not-an-ip'):
        cluster.resolve_topology(redis_node_ips='not-an-ip')
    cluster.ray.init.assert_not_called()
    cluster.ray.nodes.assert_not_called()


@pytest.mark.parametrize('count', [3, 4, 5, 7, 8, 10])
def test_default_topology_uses_all_available_hosts_as_primaries(cluster, monkeypatch, count):
    hosts = [f'192.0.2.{index}:8' for index in range(1, count + 1)]
    monkeypatch.setenv('NODE_IP_LIST', ','.join([*hosts, hosts[0]]))
    deployment = cluster.resolve_topology()
    assert len(deployment.hosts) == deployment.node_count == deployment.base_node_count == count
    assert deployment.primary_count == count
    assert deployment.replicas_per_primary == deployment.extra_replicas == 0
    assert deployment.hosts[0] == deployment.client_ip == '192.0.2.3'


@pytest.mark.parametrize(('host_count', 'primaries'), [(1, 3), (2, 3), (8, 8)])
@pytest.mark.parametrize(('replicas', 'extras'), [(1, 0), (0, 1), (1, 1)])
def test_replica_options_do_not_change_default_primary_count(cluster, host_count, primaries, replicas, extras):
    hosts = ','.join(f'192.0.2.{index}' for index in range(1, host_count + 1))
    deployment = cluster.resolve_topology(redis_node_ips=hosts, replicas_per_primary=replicas, extra_replicas=extras)
    assert deployment.primary_count == primaries
    assert deployment.replicas_per_primary == replicas
    assert deployment.extra_replicas == extras
    assert deployment.node_count == primaries * (replicas + 1) + extras


@pytest.mark.parametrize(('host_count', 'primary_count'), [(3, 6), (3, 7), (1, 3), (2, 3)])
def test_explicit_primary_count_places_instances_round_robin(cluster, host_count, primary_count):
    hosts = [f'192.0.2.{index}' for index in range(1, host_count + 1)]
    deployment = cluster.resolve_topology(primary_count=primary_count, redis_node_ips=','.join(hosts))
    ordered = ['192.0.2.3', *hosts[:2]] if host_count == 3 else hosts
    assert deployment.instances == tuple(
        (ordered[index % host_count], cluster.REDIS_PORT + index // host_count) for index in range(primary_count)
    )
    assert deployment.hosts == tuple(ordered)
    assert deployment.node_count == deployment.primary_count == primary_count


def test_explicit_primary_count_without_available_hosts_fails(cluster, monkeypatch):
    monkeypatch.delenv('NODE_IP_LIST')
    cluster.ray.nodes.return_value = []
    with pytest.raises(RuntimeError, match='host'):
        cluster.resolve_topology(primary_count=3)


def test_replicas_and_extra_replicas_share_round_robin_instance_placement(cluster):
    deployment = cluster.resolve_topology(
        redis_node_ips='192.0.2.1,192.0.2.2,192.0.2.3', primary_count=3, replicas_per_primary=1, extra_replicas=1
    )
    assert deployment.base_node_count == 6
    assert deployment.node_count == 7
    assert deployment.instances == (
        ('192.0.2.3', 16379),
        ('192.0.2.1', 16379),
        ('192.0.2.2', 16379),
        ('192.0.2.3', 16380),
        ('192.0.2.1', 16380),
        ('192.0.2.2', 16380),
        ('192.0.2.3', 16381),
    )


@pytest.mark.parametrize(('port', 'primary_count'), [(0, 3), (55534, 3), (16379, 10001)])
def test_invalid_generated_instance_ports_fail_before_startup(cluster, monkeypatch, port, primary_count):
    monkeypatch.setattr(cluster, 'REDIS_PORT', port)
    with pytest.raises(ValueError, match='port'):
        cluster.resolve_topology(redis_node_ips='192.0.2.1', primary_count=primary_count)


def test_last_legal_cluster_bus_port_is_accepted(cluster, monkeypatch):
    monkeypatch.setattr(cluster, 'REDIS_PORT', 55533)
    deployment = cluster.resolve_topology(redis_node_ips='192.0.2.1', primary_count=3)
    assert deployment.instances[-1] == ('192.0.2.1', 55535)


def test_instance_config_isolates_data_files_and_keeps_base_directory(cluster, tmp_path):
    assert cluster.instance_dir(16379, tmp_path) == tmp_path
    assert cluster.instance_dir(16380, tmp_path) == tmp_path / '16380'
    for port, directory in [(16379, tmp_path), (16380, tmp_path / '16380')]:
        conf = cluster.redis_config('192.0.2.3', port=port, remote_dir=tmp_path)
        for line in (
            f'port {port}',
            f'cluster-announce-port {port}',
            f'cluster-announce-bus-port {port + 10000}',
            f'pidfile {directory}/redis.pid',
            f'logfile {directory}/redis.log',
            f'dir {directory}/data',
            f'cluster-config-file {directory}/nodes.conf',
        ):
            assert line in conf.splitlines()


def test_writing_remote_instance_config_preserves_cluster_metadata(cluster, monkeypatch, tmp_path):
    monkeypatch.setenv('NEXUS_REDIS_DIR', '/wrong-worker-directory')
    for port in (16379, 16380):
        directory = cluster.instance_dir(port, tmp_path)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'nodes.conf').write_text(f'persisted-node-id-{port}')
        cluster.write_config(port=port, remote_dir=str(tmp_path))
        assert (directory / 'data').is_dir()
        assert (directory / 'redis.conf').read_text() == cluster.redis_config('192.0.2.3', port, tmp_path)
        assert (directory / 'nodes.conf').read_text() == f'persisted-node-id-{port}'


@pytest.mark.parametrize(('action', 'hard'), [('start', False), ('stop', False), ('stop', True)])
def test_service_targets_only_the_selected_instance(cluster, monkeypatch, tmp_path, action, hard):
    execute = Mock(return_value=SimpleNamespace(returncode=0, stdout='', stderr=''))
    monkeypatch.setattr(cluster, 'run_command', execute)
    result = cluster.service(action, hard, port=16380, remote_dir=str(tmp_path))
    command = execute.call_args.args[0]
    if action == 'start':
        assert command == ['redis-server', str(tmp_path / '16380' / 'redis.conf')]
    elif hard:
        assert str(tmp_path / '16380' / 'redis.pid') in command[-1]
        assert str(tmp_path / 'redis.pid') not in command[-1]
    else:
        assert command == ['redis-cli', '-h', '127.0.0.1', '-p', '16380', 'SHUTDOWN', 'NOSAVE']
    assert (result['ip'], result['port']) == ('192.0.2.3', 16380)
    execute.assert_called_once()


def test_redis_cli_uses_selected_instance_port(cluster, monkeypatch):
    execute = Mock()
    monkeypatch.setattr(cluster, 'run_command', execute)
    cluster.redis_cli('PING', port=16380)
    execute.assert_called_once_with(['redis-cli', '--raw', '-h', '127.0.0.1', '-p', '16380', 'PING'], timeout=10)


def test_inspection_reads_selected_instance(cluster, monkeypatch):
    outputs = [
        'PONG',
        'cluster_state:ok',
        'role:master',
        'node-id 192.0.2.3:16380@26380 myself,master - 0 0 1 connected 0-5460',
    ]
    cli = Mock(side_effect=[SimpleNamespace(stdout=value) for value in outputs])
    monkeypatch.setattr(cluster, 'redis_cli', cli)
    state = cluster.inspect_node(port=16380, remote_dir='/explicit-dir')
    assert (state['ip'], state['port'], state['node_id']) == ('192.0.2.3', 16380, 'node-id')
    assert cli.call_args_list == [
        call('PING', port=16380),
        call('CLUSTER', 'INFO', port=16380),
        call('INFO', 'replication', port=16380),
        call('CLUSTER', 'NODES', port=16380),
    ]


def test_recovery_starts_only_the_unavailable_instance(cluster, monkeypatch, tmp_path):
    deployment = cluster.resolve_topology(primary_count=3, redis_node_ips='192.0.2.3', redis_dir=tmp_path)
    current = [
        {'ip': '192.0.2.3', 'port': 16379, 'redis_ping': 'PONG'},
        {'ip': '192.0.2.3', 'port': 16380, 'redis_ping': ''},
    ]
    monkeypatch.setattr(cluster, 'states', lambda _nodes, _deployment: current)
    monkeypatch.setattr(cluster, 'wait_healthy', lambda _nodes, _deployment: current)
    monkeypatch.setattr(cluster, 'call_on', lambda _nodes, _ip, remote, *args: remote(*args))
    execute = Mock(return_value=SimpleNamespace(returncode=0, stdout='', stderr=''))
    monkeypatch.setattr(cluster, 'run_command', execute)
    assert cluster.recover_cluster({}, deployment) == current
    execute.assert_called_once_with(['redis-server', str(tmp_path / '16380' / 'redis.conf')])


def test_host_and_instance_fanout_propagate_affinity_ports_and_directory(cluster, monkeypatch, tmp_path):
    deployment = cluster.resolve_topology(primary_count=3, redis_node_ips='192.0.2.1,192.0.2.2', redis_dir=tmp_path)
    nodes = {ip: {'NodeID': ip} for ip in deployment.hosts}
    strategy = Mock(side_effect=lambda **kwargs: kwargs)
    monkeypatch.setattr(cluster, 'NodeAffinitySchedulingStrategy', strategy)
    remote = Mock()
    remote.options.return_value = remote
    for host_args in ((), (list(deployment.instances),)):
        cluster.call_all(nodes, deployment, remote, *host_args)
        assert remote.remote.call_args_list == [call(*host_args), call(*host_args)]
        assert strategy.call_args_list == [call(node_id='192.0.2.1', soft=False), call(node_id='192.0.2.2', soft=False)]
        remote.reset_mock()
        strategy.reset_mock()
    cluster.call_instances(nodes, deployment, remote, 'stop', False)
    assert remote.remote.call_args_list == [
        call('stop', False, port=16379, remote_dir=str(tmp_path)),
        call('stop', False, port=16379, remote_dir=str(tmp_path)),
        call('stop', False, port=16380, remote_dir=str(tmp_path)),
    ]
    assert strategy.call_args_list == [
        call(node_id='192.0.2.1', soft=False),
        call(node_id='192.0.2.2', soft=False),
        call(node_id='192.0.2.1', soft=False),
    ]


def test_status_and_down_fan_out_to_every_instance(cluster, deployment, monkeypatch):
    nodes = {}
    fanout = Mock(return_value=[])
    monkeypatch.setattr(cluster, 'call_instances', fanout)
    monkeypatch.setattr(cluster, 'node_map', lambda _deployment: nodes)
    assert cluster.states(nodes, deployment) == []
    fanout.assert_called_once_with(nodes, deployment, cluster.inspect_node)
    fanout.reset_mock()
    assert cluster.run('down') == 0
    assert fanout.call_args_list == [
        call(nodes, deployment, cluster.service, 'stop', False),
        call(nodes, deployment, cluster.inspect_node),
    ]


def test_down_fails_when_an_instance_is_still_running(cluster, monkeypatch):
    monkeypatch.setattr(cluster, 'node_map', lambda _deployment: {})
    monkeypatch.setattr(cluster, 'call_instances', Mock(return_value=[]))
    running = {'ip': '192.0.2.3', 'port': 16380, 'redis_ping': 'PONG'}
    monkeypatch.setattr(cluster, 'states', Mock(return_value=[running]))
    with pytest.raises(RuntimeError, match='Redis stop failed'):
        cluster.run('down')


def test_down_succeeds_repeatedly_for_stopped_instances(cluster, deployment, monkeypatch):
    nodes = {}
    monkeypatch.setattr(cluster, 'node_map', lambda _deployment: nodes)
    stopped = {'ip': '192.0.2.3', 'port': 16380, 'redis_ping': ''}
    inspection = Mock(return_value=[stopped])
    monkeypatch.setattr(cluster, 'states', inspection)
    monkeypatch.setattr(cluster, 'call_instances', Mock(return_value=[]))
    assert cluster.run('down') == cluster.run('down') == 0
    assert inspection.call_args_list == [call(nodes, deployment), call(nodes, deployment)]


def test_published_endpoints_include_every_instance_port(cluster):
    deployment = cluster.resolve_topology(primary_count=3, redis_node_ips='192.0.2.1')
    info = cluster.endpoint_info(deployment)
    seeds = '192.0.2.1:16379,192.0.2.1:16380,192.0.2.1:16381'
    assert info['startup_nodes'] == seeds
    assert info['redis_url'] == f'redis+cluster://192.0.2.1:16379/0?nodes={seeds}'
    assert (info['redis_port'], info['cluster_bus_port']) == (16379, 26379)


@pytest.mark.parametrize('fresh_cluster', [True, False])
def test_start_cluster_routes_full_topology_and_fresh_replicas(cluster, monkeypatch, tmp_path, fresh_cluster):
    deployment = cluster.resolve_topology(
        primary_count=3, extra_replicas=1, redis_node_ips='192.0.2.3', redis_dir=tmp_path
    )
    nodes = {'192.0.2.3': {'NodeID': 'node-3'}}
    current = [
        {
            'ip': ip,
            'port': port,
            'redis_ping': 'PONG',
            'cluster': {
                'cluster_known_nodes': '1' if fresh_cluster or port == 16382 else '3',
                'cluster_slots_assigned': '0' if fresh_cluster or port == 16382 else '16384',
            },
        }
        for ip, port in deployment.instances
    ]
    current = [current[2], *current[:2], current[3]]
    initial = [dict(state, redis_ping='' if state['port'] == 16380 else 'PONG') for state in current]
    hosts = Mock(side_effect=[[{'ready': True, 'server_version': '7.0'}], []])
    instances = Mock()
    on_host = Mock(
        side_effect=[
            {'ip': '192.0.2.3', 'port': 16380, 'results': {'redis': {'returncode': 0}}},
            {'returncode': 0, 'created': True} if fresh_cluster else [{'returncode': 0}],
            {},
        ]
    )
    monkeypatch.setattr(cluster, 'call_all', hosts)
    monkeypatch.setattr(cluster, 'call_instances', instances)
    monkeypatch.setattr(cluster, 'call_on', on_host)
    monkeypatch.setattr(cluster, 'states', Mock(side_effect=[initial, current]))
    monkeypatch.setattr(cluster, 'wait_healthy', lambda _nodes, _deployment: current)
    monkeypatch.setattr(cluster.time, 'sleep', lambda _seconds: None)
    cluster.start_cluster(nodes, deployment)
    instances.assert_called_once_with(nodes, deployment, cluster.write_config)
    assert hosts.call_args_list == [
        call(nodes, deployment, cluster.ensure_redis),
        call(nodes, deployment, cluster.probe_cluster_ports, list(deployment.instances)),
    ]
    expected = (
        call(nodes, '192.0.2.3', cluster.create_cluster, deployment)
        if fresh_cluster
        else call(nodes, '192.0.2.3', cluster.add_extra_replicas, '192.0.2.3:16381', ['192.0.2.3:16382'], 3)
    )
    assert on_host.call_args_list[:2] == [
        call(nodes, '192.0.2.3', cluster.service, 'start', False, 16380, str(tmp_path)),
        expected,
    ]


def test_cluster_creation_splits_base_and_extra_instances_at_their_real_ports(cluster, monkeypatch):
    deployment = cluster.resolve_topology(
        primary_count=3, replicas_per_primary=1, extra_replicas=1, redis_node_ips='192.0.2.1,192.0.2.2,192.0.2.3'
    )
    execute = Mock(return_value={'returncode': 0, 'stdout': 'All 16384 slots covered.', 'stderr': ''})
    attach = Mock(return_value=[{'returncode': 0}])
    monkeypatch.setattr(cluster, 'run_cluster_cli', execute)
    monkeypatch.setattr(cluster, 'attach_extra_replicas', attach)
    endpoints = [f'{ip}:{port}' for ip, port in deployment.instances]
    assert cluster.create_cluster(deployment)['created']
    execute.assert_called_once_with([
        'redis-cli',
        '--cluster',
        'create',
        *endpoints[:6],
        '--cluster-replicas',
        '1',
        '--cluster-yes',
    ])
    attach.assert_called_once_with('192.0.2.3:16379', ['192.0.2.3:16381'], 3)


@pytest.mark.parametrize('options', [{'primary_count': 2}, {'replicas_per_primary': -1}, {'extra_replicas': -1}])
def test_invalid_node_counts_fail_before_startup(cluster, options):
    with pytest.raises(ValueError, match='Redis Cluster requires'):
        cluster.resolve_topology(**options)


def test_create_cluster_passes_zero_replicas_and_every_selected_host(cluster, deployment, monkeypatch):
    execute = Mock(return_value={'returncode': 0, 'stdout': 'All 16384 slots covered.', 'stderr': ''})
    monkeypatch.setattr(cluster, 'run_cluster_cli', execute)
    endpoints = [f'{host}:{port}' for host, port in deployment.instances]
    result = cluster.create_cluster(deployment)
    execute.assert_called_once_with([
        'redis-cli',
        '--cluster',
        'create',
        *endpoints,
        '--cluster-replicas',
        '0',
        '--cluster-yes',
    ])
    assert result['created']
    assert result['extra_replicas'] == []


@pytest.mark.parametrize('good_views', [2, 4, 5, 8])
def test_availability_requires_a_majority_of_the_configured_primaries(cluster, deployment, good_views):
    states = [
        {
            'redis_ping': 'PONG',
            'role': 'primary',
            'cluster': {
                'cluster_state': 'ok' if index < good_views else 'fail',
                'cluster_slots_ok': '16384',
                'cluster_size': str(deployment.primary_count),
            },
        }
        for index in range(deployment.primary_count)
    ]
    assert cluster.cluster_available(states, deployment) == (good_views >= 5)


def test_all_primary_topology_is_healthy_without_replicas(cluster, deployment):
    states = [
        {
            'redis_ping': 'PONG',
            'role': 'primary',
            'node_id': str(index),
            'slots': ['0-1'],
            'bad_node_flags': [],
            'cluster': {
                'cluster_state': 'ok',
                'cluster_slots_ok': '16384',
                'cluster_known_nodes': str(deployment.node_count),
                'cluster_size': str(deployment.primary_count),
            },
        }
        for index in range(deployment.primary_count)
    ]
    assert cluster.cluster_healthy(states, deployment)


@pytest.mark.parametrize(
    ('command', 'handler'), [('serve', 'serve_cluster'), ('up', 'start_cluster'), ('recover', 'recover_cluster')]
)
def test_dispatch_joins_existing_ray(cluster, deployment, monkeypatch, command, handler):
    cluster.ray.is_initialized.return_value = False
    cluster.ray.init.reset_mock()
    resolve = Mock(wraps=cluster.resolve_topology)
    monkeypatch.setattr(cluster, 'resolve_topology', resolve)
    calls = Mock()
    calls.attach_mock(cluster.ray.init, 'init')
    calls.attach_mock(resolve, 'resolve')
    nodes = {'nodes': 'test'}
    monkeypatch.setattr(cluster, 'node_map', lambda _deployment: nodes)
    operation = Mock()
    monkeypatch.setattr(cluster, handler, operation)
    assert cluster.run(command) == 0
    cluster.ray.init.assert_called_once_with(address='auto', log_to_driver=False, logging_level=logging.ERROR)
    assert [invocation[0] for invocation in calls.mock_calls] == ['init', 'resolve']
    operation.assert_called_once_with(nodes, deployment)


@pytest.mark.parametrize('available', [True, False])
def test_status_exit_semantics(cluster, monkeypatch, available):
    monkeypatch.setattr(cluster, 'node_map', lambda _deployment: {})
    monkeypatch.setattr(cluster, 'states', lambda _nodes, _deployment: [])
    monkeypatch.setattr(cluster, 'cluster_available', lambda _states, _deployment: available)
    if available:
        assert cluster.run('status') == 0
    else:
        with pytest.raises(SystemExit) as caught:
            cluster.run('status')
        assert caught.value.code == 1


def test_serve_exit_leaves_redis_running(cluster, deployment, monkeypatch):
    calls = []
    monkeypatch.setattr(cluster, 'start_cluster', lambda _nodes, _deployment: calls.append('up'))
    monkeypatch.setattr(cluster, 'publish_endpoints', lambda info, _deployment: calls.append(info))
    monkeypatch.setattr(cluster.signal, 'signal', lambda *args: None)
    monkeypatch.setattr(cluster.threading, 'Event', lambda: SimpleNamespace(wait=lambda timeout: True))
    monkeypatch.setattr(cluster, 'call_instances', Mock(side_effect=AssertionError('serve must not stop Redis')))
    cluster.serve_cluster({}, deployment)
    assert calls[0] == 'up'
    assert calls[1]['redis_url'].startswith('redis+cluster://192.0.2.3:16379/0?nodes=')


def test_packaged_configuration_error_propagates(cluster, monkeypatch, tmp_path):
    config = tmp_path / 'redis_server_cluster.json'
    config.write_text('invalid json')
    monkeypatch.setattr(cluster, '__file__', str(tmp_path / 'cluster.py'))
    assert cluster.config_path() == config
    with pytest.raises(json.JSONDecodeError):
        json.loads(cluster.config_path().read_text())


def test_import_does_not_resolve_runtime_topology(cluster, monkeypatch):
    monkeypatch.delenv('NODE_IP_LIST')
    cluster.ray.nodes.side_effect = AssertionError('topology must be resolved when executing a command')
    importlib.reload(cluster)
    cluster.ray.init.assert_not_called()
    cluster.ray.nodes.assert_not_called()


def test_explicit_configuration_ignores_legacy_environment(cluster, monkeypatch):
    for name in (
        'PRIMARY_COUNT',
        'REPLICAS_PER_PRIMARY',
        'EXTRA_REPLICAS',
        'NODE_IPS',
        'CLIENT_IP',
        'DIR',
        'ENDPOINT_FILE',
    ):
        monkeypatch.setenv('NEXUS_REDIS_' + name, 'invalid-legacy-value')
    importlib.reload(cluster)
    deployment = cluster.resolve_topology(primary_count=3, redis_dir='/tmp/args-only')
    assert deployment.primary_count == 3
    assert deployment.replicas_per_primary == deployment.extra_replicas == 0
    assert deployment.remote_dir == Path('/tmp/args-only')
    assert cluster.resolve_topology().primary_count == 8


def test_deployments_do_not_share_runtime_configuration(cluster, monkeypatch, tmp_path):
    first = cluster.resolve_topology(
        primary_count=3,
        replicas_per_primary=1,
        extra_replicas=1,
        redis_dir=tmp_path / 'first',
        redis_endpoint_file=tmp_path / 'first.json',
    )
    second = cluster.resolve_topology(primary_count=6, redis_node_ips='192.0.2.1', redis_dir=tmp_path / 'second')
    assert (first.primary_count, first.replicas_per_primary, first.extra_replicas) == (3, 1, 1)
    assert first.remote_dir == tmp_path / 'first'
    assert first.endpoint_file == tmp_path / 'first.json'
    assert (second.primary_count, second.replicas_per_primary, second.extra_replicas) == (6, 0, 0)
    assert second.remote_dir == tmp_path / 'second'
    assert second.instances[-1] == ('192.0.2.1', 16384)


def test_repeated_topology_resolution_refreshes_ray_head_without_changing_prior_deployment(cluster):
    first = cluster.resolve_topology(primary_count=3)
    for node in cluster.ray.nodes.return_value:
        node['Resources'] = {'node:__internal_head__': 1} if node['NodeID'] == 'node-8' else {}
    second = cluster.resolve_topology(primary_count=6)
    assert first.hosts == ('192.0.2.3', '192.0.2.1', '192.0.2.2')
    assert first.primary_count == 3
    assert second.hosts == ('192.0.2.8', '192.0.2.1', '192.0.2.2', '192.0.2.3', '192.0.2.4', '192.0.2.5')
    assert second.primary_count == 6
    cluster.ray.init.assert_not_called()


@pytest.mark.parametrize('source', ['argument', 'taiji', 'directory'])
def test_endpoint_file_uses_argument_then_platform_output_then_deployment_directory(
    cluster, monkeypatch, tmp_path, source
):
    monkeypatch.setenv('NEXUS_REDIS_ENDPOINT_FILE', str(tmp_path / 'ignored.json'))
    if source in ('argument', 'taiji'):
        monkeypatch.setenv('TAIJI_BASIC_OUTPUT_PATH', str(tmp_path / 'taiji'))
    else:
        monkeypatch.delenv('TAIJI_BASIC_OUTPUT_PATH', raising=False)
    deployment = cluster.resolve_topology(
        redis_dir=tmp_path / 'redis', redis_endpoint_file=tmp_path / 'explicit.json' if source == 'argument' else None
    )
    info = cluster.endpoint_info(deployment)
    cluster.publish_endpoints(info, deployment)
    expected = {
        'argument': tmp_path / 'explicit.json',
        'taiji': tmp_path / 'taiji' / 'redis-cluster.json',
        'directory': tmp_path / 'redis' / 'endpoints.json',
    }[source]
    assert json.loads(expected.read_text()) == info
    assert not (tmp_path / 'ignored.json').exists()
