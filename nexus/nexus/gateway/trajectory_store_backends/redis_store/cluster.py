from __future__ import annotations

import ipaddress
import json
import logging
import os
import re
import shutil
import signal
import socket
import subprocess
import threading
import time

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ray

from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from .utils import get_ray_head_ip


def config_path() -> Path:
    return Path(__file__).with_name('redis_server_cluster.json')


CONFIG = json.loads(config_path().read_text())
REDIS_PORT = CONFIG['redis_port']
CLUSTER_BUS_PORT = CONFIG['cluster_bus_port']
DEFAULT_REMOTE_DIR = Path(CONFIG['remote_dir'])
if CLUSTER_BUS_PORT != REDIS_PORT + 10000:
    raise ValueError('cluster_bus_port must equal redis_port + 10000 for the supported Redis versions')


def connect_ray() -> None:
    # Join the existing cluster without starting a separate local runtime.
    # Remote functions return structured results, so worker stdout is not needed.
    if not ray.is_initialized():
        ray.init(address='auto', log_to_driver=False, logging_level=logging.ERROR)


@dataclass(frozen=True)
class RedisDeployment:
    instances: tuple[tuple[str, int], ...]
    client_ip: str
    replicas_per_primary: int
    extra_replicas: int
    remote_dir: Path
    endpoint_file: Path | None = None

    @property
    def hosts(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(ip for ip, _port in self.instances))

    @property
    def node_count(self) -> int:
        return len(self.instances)

    @property
    def base_node_count(self) -> int:
        return self.node_count - self.extra_replicas

    @property
    def primary_count(self) -> int:
        return self.base_node_count // (self.replicas_per_primary + 1)


def resolve_topology(
    *,
    primary_count: int | None = None,
    replicas_per_primary: int | None = None,
    extra_replicas: int | None = None,
    redis_node_ips: str | None = None,
    redis_client_ip: str | None = None,
    redis_dir: str | Path | None = None,
    redis_endpoint_file: str | Path | None = None,
) -> RedisDeployment:
    # Taiji injects NODE_IP_LIST as "ip:slot,ip:slot,...".
    # Explicit node IPs can select dedicated Redis hosts.
    replicas_per_primary = CONFIG['replicas_per_primary'] if replicas_per_primary is None else replicas_per_primary
    extra_replicas = CONFIG['extra_replicas'] if extra_replicas is None else extra_replicas
    if replicas_per_primary < 0 or extra_replicas < 0:
        raise ValueError('Redis Cluster requires nonnegative replica counts')
    raw_nodes = redis_node_ips if redis_node_ips is not None else os.environ.get('NODE_IP_LIST')
    all_nodes = []
    if raw_nodes:
        for item in raw_nodes.split(','):
            # The suffix in NODE_IP_LIST is a Taiji slot count, not part of the IP.
            ip = item.strip().rsplit(':', 1)[0]
            ipaddress.ip_address(ip)
            if ip not in all_nodes:
                all_nodes.append(ip)
    else:
        connect_ray()
        all_nodes = sorted({node['NodeManagerAddress'] for node in ray.nodes() if node['Alive']})
    configured_primaries = CONFIG['primary_count'] if primary_count is None else primary_count
    primary_count = max(len(all_nodes), 3) if configured_primaries is None else int(configured_primaries)
    if primary_count < 3:
        raise ValueError('Redis Cluster requires at least 3 primaries')
    required_instances = primary_count * (replicas_per_primary + 1) + extra_replicas
    if not all_nodes:
        raise RuntimeError('Redis Cluster needs at least 1 task host')

    # Put the Ray head first only to make cluster creation deterministic. Redis
    # Cluster owns the live primary/replica roles after bootstrap.
    head = get_ray_head_ip()
    ordered = [head, *(ip for ip in all_nodes if ip != head)] if head in all_nodes else all_nodes
    redis_nodes = tuple(ordered[:required_instances])
    instances = tuple(
        (redis_nodes[index % len(redis_nodes)], REDIS_PORT + index // len(redis_nodes))
        for index in range(required_instances)
    )
    ports_by_host: dict[str, set[int]] = {ip: set() for ip in redis_nodes}
    for ip, port in instances:
        for bound_port in (port, port + 10000):
            if not 1 <= bound_port <= 65535:
                raise ValueError(f'Redis Cluster port is out of range on {ip}: {bound_port}')
            if bound_port in ports_by_host[ip]:
                raise ValueError(f'Redis Cluster ports overlap on {ip}: {bound_port}')
            ports_by_host[ip].add(bound_port)
    if redis_client_ip and redis_client_ip not in all_nodes:
        raise RuntimeError(f'--redis-client-ip is not in the node list: {redis_client_ip}')
    client = redis_client_ip or next((ip for ip in ordered if ip not in redis_nodes), redis_nodes[0])
    return RedisDeployment(
        instances,
        client,
        replicas_per_primary,
        extra_replicas,
        DEFAULT_REMOTE_DIR if redis_dir is None else Path(redis_dir),
        None if redis_endpoint_file is None else Path(redis_endpoint_file),
    )


def run_command(
    command: list[str], timeout: int = 30, input_text: str | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, input=input_text, timeout=timeout, check=False)


def cluster_cli_supported(help_text: str) -> bool:
    return 'Cluster Manager Commands' in help_text and bool(re.search(r'(?m)^\s*create\s', help_text))


def cluster_create_succeeded(output: str) -> bool:
    return bool(re.search(r'all\s+16384\s+slots\s+covered', output, re.IGNORECASE))


def redis_cli(*args: str, port: int = REDIS_PORT) -> subprocess.CompletedProcess[str]:
    return run_command(['redis-cli', '--raw', '-h', '127.0.0.1', '-p', str(port), *args], timeout=10)


def instance_dir(port: int, remote_dir: str | Path = DEFAULT_REMOTE_DIR) -> Path:
    directory = Path(remote_dir)
    return directory if port == REDIS_PORT else directory / str(port)


def redis_config(node_ip: str, port: int = REDIS_PORT, remote_dir: str | Path = DEFAULT_REMOTE_DIR) -> str:
    # nodes.conf is deliberately not generated here: Redis owns and persists
    # cluster IDs, epochs and post-failover roles in that file.
    directory = instance_dir(port, remote_dir)
    return '\n'.join([
        # Redis 5 uses the first bind address as the source address for
        # outbound Cluster bus connections. The node IP must come first;
        # putting 127.0.0.1 first makes every MEET packet unroutable.
        f'bind {node_ip} 127.0.0.1',
        'protected-mode no',
        f'port {port}',
        'timeout 0',
        'tcp-keepalive 60',
        'daemonize yes',
        'supervised no',
        f'pidfile {directory}/redis.pid',
        f'logfile {directory}/redis.log',
        'databases 1',
        'save ""',
        f'appendonly {"yes" if CONFIG["appendonly"] else "no"}',
        'appendfsync everysec',
        f'dir {directory}/data',
        'cluster-enabled yes',
        f'cluster-config-file {directory}/nodes.conf',
        f'cluster-node-timeout {CONFIG["cluster_node_timeout_ms"]}',
        'cluster-require-full-coverage yes',
        f'cluster-announce-ip {node_ip}',
        f'cluster-announce-port {port}',
        f'cluster-announce-bus-port {port + 10000}',
        'cluster-migration-barrier 1',
        '',
    ])


# One copy runs on every selected host. redis_server_launcher_prepare.sh
# installs redis-py; this task separately verifies/installs redis-server and
# redis-cli themselves.
@ray.remote(num_cpus=0.1)
def ensure_redis() -> dict[str, Any]:
    ip = ray.util.get_node_ip_address()
    before = {name: shutil.which(name) for name in ('redis-server', 'redis-cli')}
    install_command = None
    install = None
    if not all(before.values()):
        privilege = '' if os.geteuid() == 0 else 'sudo ' if shutil.which('sudo') else None
        if privilege is None:
            raise PermissionError(f'Redis is missing on {ip}, and neither root nor sudo is available')
        if shutil.which('yum'):
            install_command = f'{privilege}yum install -y redis'
        elif shutil.which('dnf'):
            install_command = f'{privilege}dnf install -y redis'
        elif shutil.which('apt-get'):
            install_command = (
                f'{privilege}apt-get update && '
                f'DEBIAN_FRONTEND=noninteractive {privilege}apt-get install -y redis-server redis-tools'
            )
        else:
            raise RuntimeError(f'Redis is missing on {ip}, and no supported package manager was found')
        install = run_command(['bash', '-lc', install_command], timeout=300)
    after = {name: shutil.which(name) for name in ('redis-server', 'redis-cli')}
    version = run_command(['bash', '-lc', 'redis-server --version; redis-cli --version'])
    server_match = re.search(r'v=(\d+)\.(\d+)', version.stdout)
    server_version = tuple(map(int, server_match.groups())) if server_match else (0, 0)
    cluster_cli = run_command(['redis-cli', '--cluster', 'help']) if after['redis-cli'] else None
    cluster_help = f'{cluster_cli.stdout}\n{cluster_cli.stderr}' if cluster_cli else ''
    # Redis 5.0.3 prints valid Cluster Manager help but may return non-zero for
    # the help action. Detect the required command from output instead of using
    # that unstable exit code. PING is intentionally not checked here: Redis is
    # started only after this task writes the generated cluster configuration.
    checks = {
        'redis_server_found': bool(after['redis-server']),
        'redis_cli_found': bool(after['redis-cli']),
        'version_command_ok': version.returncode == 0,
        'server_version_supported': server_version >= (5, 0),
        'cluster_create_supported': cluster_cli_supported(cluster_help),
    }
    return {
        'ip': ip,
        'already_installed': all(before.values()),
        'install_command': install_command,
        'install_returncode': install.returncode if install else None,
        'install_stdout_tail': install.stdout[-1000:] if install else '',
        'install_stderr_tail': install.stderr[-1000:] if install else '',
        'binary_paths': after,
        'version_returncode': version.returncode,
        'version_stdout': version.stdout.strip(),
        'version_stderr': version.stderr.strip(),
        'server_version': '.'.join(map(str, server_version)),
        'cluster_help_returncode': cluster_cli.returncode if cluster_cli else None,
        'cluster_help_stdout_tail': cluster_cli.stdout[-2000:] if cluster_cli else '',
        'cluster_help_stderr_tail': cluster_cli.stderr[-2000:] if cluster_cli else '',
        'checks': checks,
        'ready': all(checks.values()),
    }


@ray.remote(num_cpus=0.1)
def write_config(port: int = REDIS_PORT, remote_dir: str | Path = DEFAULT_REMOTE_DIR) -> str:
    ip = ray.util.get_node_ip_address()
    directory = instance_dir(port, remote_dir)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'data').mkdir(parents=True, exist_ok=True)
    (directory / 'redis.conf').write_text(redis_config(ip, port, remote_dir))
    return f'{ip}:{port}'


# This task is node-affined: localhost and the pid file refer to the selected
# host. hard=True is reserved for crash tests; normal operations use SHUTDOWN.
@ray.remote(num_cpus=0.1)
def service(
    action: str, hard: bool = False, port: int = REDIS_PORT, remote_dir: str | Path = DEFAULT_REMOTE_DIR
) -> dict[str, Any]:
    directory = instance_dir(port, remote_dir)
    if action == 'start':
        command = ['redis-server', str(directory / 'redis.conf')]
    elif action == 'stop' and hard:
        command = ['bash', '-lc', f'test -f {directory}/redis.pid && kill -9 $(cat {directory}/redis.pid)']
    elif action == 'stop':
        command = ['redis-cli', '-h', '127.0.0.1', '-p', str(port), 'SHUTDOWN', 'NOSAVE']
    else:
        raise ValueError(f'unsupported service action: {action}')
    proc = run_command(command)
    return {
        'ip': ray.util.get_node_ip_address(),
        'port': port,
        'action': action,
        'results': {
            'redis': {
                'command': command,
                'returncode': proc.returncode,
                'stdout': proc.stdout.strip(),
                'stderr': proc.stderr.strip(),
            }
        },
    }


def parse_info(text: str) -> dict[str, str]:
    return {
        key: value.strip()
        for line in text.splitlines()
        if ':' in line and not line.startswith('#')
        for key, value in [line.split(':', 1)]
    }


@ray.remote(num_cpus=0.1)
def inspect_node(port: int = REDIS_PORT, remote_dir: str | Path = DEFAULT_REMOTE_DIR) -> dict[str, Any]:
    ping_result = redis_cli('PING', port=port)
    ping = ping_result.stdout.strip()
    cluster = parse_info(redis_cli('CLUSTER', 'INFO', port=port).stdout) if ping == 'PONG' else {}
    replication = parse_info(redis_cli('INFO', 'replication', port=port).stdout) if ping == 'PONG' else {}
    nodes_output = redis_cli('CLUSTER', 'NODES', port=port).stdout if ping == 'PONG' else ''
    diagnostics = {}
    if ping != 'PONG':
        processes = run_command(['bash', '-lc', "ps -ef | grep '[r]edis-server' || true"])
        listeners = run_command([
            'bash',
            '-lc',
            f"(ss -lntp 2>/dev/null || netstat -lntp 2>/dev/null || true) | grep ':{port} ' || true",
        ])
        redis_log = run_command([
            'bash',
            '-lc',
            f'tail -n 80 {instance_dir(port, remote_dir)}/redis.log 2>/dev/null || true',
        ])
        diagnostics = {
            'ping_returncode': ping_result.returncode,
            'ping_stderr': ping_result.stderr.strip(),
            'redis_processes': processes.stdout.strip(),
            'port_listeners': listeners.stdout.strip(),
            'redis_log_tail': redis_log.stdout.strip(),
        }
    node_rows = [line.split() for line in nodes_output.splitlines()]
    self_fields = next((row for row in node_rows if 'myself' in row[2].split(',')), [])
    bad_node_flags = sorted({
        flag for row in node_rows for flag in row[2].split(',') if flag in {'fail', 'fail?', 'handshake', 'noaddr'}
    })
    flags = self_fields[2].split(',') if self_fields else []
    role = 'primary' if 'master' in flags else 'replica' if {'slave', 'replica'} & set(flags) else None
    slots = [field for field in self_fields[8:] if not field.startswith('[')] if self_fields else []
    return {
        'ip': ray.util.get_node_ip_address(),
        'port': port,
        'redis_ping': ping,
        'node_id': self_fields[0] if self_fields else None,
        'role': role,
        'master_id': self_fields[3] if role == 'replica' else None,
        'flags': flags,
        'bad_node_flags': bad_node_flags,
        'slots': slots,
        'cluster': cluster,
        'replication': replication,
        'diagnostics': diagnostics,
    }


@ray.remote(num_cpus=0.1)
def probe_cluster_ports(targets: list[tuple[str, int]]) -> dict[str, Any]:
    results = {}
    for host, data_port in targets:
        for port in (data_port, data_port + 10000):
            try:
                with socket.create_connection((host, port), timeout=2):
                    results[f'{host}:{port}'] = 'ok'
            except OSError as exc:
                results[f'{host}:{port}'] = f'{type(exc).__name__}: {exc}'
    return {'source': ray.util.get_node_ip_address(), 'results': results}


def run_cluster_cli(command: list[str], timeout: int = 60) -> dict[str, Any]:
    try:
        result = run_command(command, timeout=timeout, input_text='yes\n')
        return {
            'command': command,
            'returncode': result.returncode,
            'timed_out': False,
            'stdout': result.stdout,
            'stderr': result.stderr,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            'command': command,
            'returncode': 124,
            'timed_out': True,
            'stdout': exc.stdout.decode() if isinstance(exc.stdout, bytes) else exc.stdout or '',
            'stderr': exc.stderr.decode() if isinstance(exc.stderr, bytes) else exc.stderr or '',
        }


def primary_replica_counts(reference: str) -> tuple[list[str], Counter[str]]:
    host, port = reference.rsplit(':', 1)
    result = run_command(['redis-cli', '--raw', '-h', host, '-p', port, 'CLUSTER', 'NODES'])
    if result.returncode != 0:
        raise RuntimeError(f'cannot read Cluster nodes from {reference}: {result.stderr}')
    primary_ids = []
    replica_counts: Counter[str] = Counter()
    for line in result.stdout.splitlines():
        fields = line.split()
        flags = set(fields[2].split(','))
        if 'master' in flags and fields[8:]:
            primary_ids.append(fields[0])
        elif {'slave', 'replica'} & flags:
            replica_counts[fields[3]] += 1
    return sorted(primary_ids), replica_counts


def attach_extra_replicas(reference: str, extra_endpoints: list[str], primary_count: int) -> list[dict[str, Any]]:
    results = []
    for endpoint in extra_endpoints:
        primary_ids, replica_counts = primary_replica_counts(reference)
        if len(primary_ids) != primary_count:
            raise RuntimeError(f'expected {primary_count} primaries before adding {endpoint}: {primary_ids}')
        # Give the next spare to the least-protected primary. The spare becomes
        # a normal replica immediately and may later migrate automatically.
        primary_id = min(primary_ids, key=lambda item: (replica_counts[item], item))
        command = [
            'redis-cli',
            '--cluster',
            'add-node',
            endpoint,
            reference,
            '--cluster-slave',
            '--cluster-master-id',
            primary_id,
        ]
        result = run_cluster_cli(command)
        results.append(result)
        if result['returncode'] != 0:
            break
    return results


@ray.remote(num_cpus=0.1)
def add_extra_replicas(reference: str, extra_endpoints: list[str], primary_count: int) -> list[dict[str, Any]]:
    return attach_extra_replicas(reference, extra_endpoints, primary_count)


@ray.remote(num_cpus=0.1)
def create_cluster(deployment: RedisDeployment) -> dict[str, Any]:
    # Create the balanced base topology first. Extra nodes are then attached as
    # replicas, because passing extras to `--cluster create` can change the
    # inferred primary count instead of creating spare replicas.
    endpoints = [f'{ip}:{port}' for ip, port in deployment.instances]
    base_endpoints = endpoints[: deployment.base_node_count]
    extra_endpoints = endpoints[deployment.base_node_count :]
    command = [
        'redis-cli',
        '--cluster',
        'create',
        *base_endpoints,
        '--cluster-replicas',
        str(deployment.replicas_per_primary),
        '--cluster-yes',
    ]
    result = run_cluster_cli(command)
    output = f'{result["stdout"]}\n{result["stderr"]}'
    result['base_created'] = cluster_create_succeeded(output)
    result['extra_replicas'] = []
    if result['returncode'] == 0 and result['base_created'] and extra_endpoints:
        result['extra_replicas'] = attach_extra_replicas(base_endpoints[0], extra_endpoints, deployment.primary_count)
    result['created'] = result['base_created'] and all(item['returncode'] == 0 for item in result['extra_replicas'])
    return result


@ray.remote(num_cpus=0.1)
def smoke_node(seed: tuple[str, int]) -> dict[str, Any]:
    # -c follows MOVED redirects, so the probe succeeds regardless of which
    # primary owns this key; a plain redis-cli connection would not.
    key = f'nexus-redis-cluster-smoke-{int(time.time())}'
    target = ['redis-cli', '-c', '--raw', '-h', seed[0], '-p', str(seed[1])]
    set_result = run_command([*target, 'SET', key, 'ok'])
    get_result = run_command([*target, 'GET', key])
    result = {
        'client_ip': ray.util.get_node_ip_address(),
        'startup_node': f'{seed[0]}:{seed[1]}',
        'set': set_result.stdout.strip(),
        'get': get_result.stdout.strip(),
    }
    if result['set'] != 'OK' or result['get'] != 'ok':
        raise RuntimeError(f'smoke failed: {result}')
    return result


def node_map(deployment: RedisDeployment) -> dict[str, dict[str, Any]]:
    nodes = {node['NodeManagerAddress']: node for node in ray.nodes() if node['Alive']}
    missing = [ip for ip in (*deployment.hosts, deployment.client_ip) if ip not in nodes]
    if missing:
        raise RuntimeError(f'configured Ray nodes are unavailable: {missing}')
    return nodes


def call_on(nodes: dict[str, dict[str, Any]], ip: str, remote: Any, *args: Any) -> Any:
    # soft=False is required because all service/config operations are local to
    # the selected host and must never be rescheduled to another Ray node.
    strategy = NodeAffinitySchedulingStrategy(node_id=nodes[ip]['NodeID'], soft=False)
    return ray.get(remote.options(scheduling_strategy=strategy).remote(*args))


def call_all(nodes: dict[str, dict[str, Any]], deployment: RedisDeployment, remote: Any, *args: Any) -> list[Any]:
    # Fan out the same Ray task to every Redis host with hard node affinity.
    # ray.get keeps result order aligned with deployment.hosts and waits until
    # all hosts have responded.
    refs = []
    for ip in deployment.hosts:
        strategy = NodeAffinitySchedulingStrategy(node_id=nodes[ip]['NodeID'], soft=False)
        refs.append(remote.options(scheduling_strategy=strategy).remote(*args))
    return ray.get(refs)


def call_instances(nodes: dict[str, dict[str, Any]], deployment: RedisDeployment, remote: Any, *args: Any) -> list[Any]:
    refs = []
    for ip, port in deployment.instances:
        strategy = NodeAffinitySchedulingStrategy(node_id=nodes[ip]['NodeID'], soft=False)
        refs.append(
            remote.options(scheduling_strategy=strategy).remote(*args, port=port, remote_dir=str(deployment.remote_dir))
        )
    return ray.get(refs)


def states(nodes: dict[str, dict[str, Any]], deployment: RedisDeployment) -> list[dict[str, Any]]:
    return call_instances(nodes, deployment, inspect_node)


def print_json(label: str, value: Any) -> None:
    print(f'{label}= {json.dumps(value, ensure_ascii=False, indent=2)}')


def require_service(result: dict[str, Any], label: str) -> None:
    failed = {name: item for name, item in result['results'].items() if item['returncode'] != 0}
    if failed:
        raise RuntimeError(f'{label} failed on {result["ip"]}:{result["port"]}: {failed}')


def cluster_available(current_states: list[dict[str, Any]], deployment: RedisDeployment) -> bool:
    good_views = [
        state
        for state in current_states
        if state['redis_ping'] == 'PONG'
        and state['cluster'].get('cluster_state') == 'ok'
        and state['cluster'].get('cluster_slots_ok') == '16384'
        and state['cluster'].get('cluster_size') == str(deployment.primary_count)
    ]
    live_primaries = [state for state in current_states if state['redis_ping'] == 'PONG' and state['role'] == 'primary']
    # Require a majority of agreeing views and every slot owner to be live
    # before clients proceed after failover.
    return len(good_views) > deployment.primary_count // 2 and len(live_primaries) == deployment.primary_count


def cluster_healthy(current_states: list[dict[str, Any]], deployment: RedisDeployment) -> bool:
    if len(current_states) != deployment.node_count or not all(
        state['redis_ping'] == 'PONG' for state in current_states
    ):
        return False
    if not all(
        state['cluster'].get('cluster_state') == 'ok'
        and state['cluster'].get('cluster_slots_ok') == '16384'
        and state['cluster'].get('cluster_known_nodes') == str(deployment.node_count)
        and state['cluster'].get('cluster_size') == str(deployment.primary_count)
        and not state['bad_node_flags']
        for state in current_states
    ):
        return False
    primaries = [state for state in current_states if state['role'] == 'primary']
    replicas = [state for state in current_states if state['role'] == 'replica']
    replica_counts = Counter(state['master_id'] for state in replicas)
    return (
        len(primaries) == deployment.primary_count
        and all(primary['slots'] for primary in primaries)
        and len(replicas) == deployment.primary_count * deployment.replicas_per_primary + deployment.extra_replicas
        and all(replica_counts[primary['node_id']] >= deployment.replicas_per_primary for primary in primaries)
        and all(replica['replication'].get('master_link_status') == 'up' for replica in replicas)
    )


def wait_healthy(
    nodes: dict[str, dict[str, Any]], deployment: RedisDeployment, timeout: float = 60.0
) -> list[dict[str, Any]]:
    deadline = time.monotonic() + timeout
    latest = []
    while time.monotonic() < deadline:
        latest = states(nodes, deployment)
        if cluster_healthy(latest, deployment):
            return latest
        time.sleep(0.5)
    raise TimeoutError(f'cluster did not become fully healthy: {latest}')


def wait_available(
    nodes: dict[str, dict[str, Any]], deployment: RedisDeployment, timeout: float = 30.0
) -> list[dict[str, Any]]:
    deadline = time.monotonic() + timeout
    latest = []
    while time.monotonic() < deadline:
        latest = states(nodes, deployment)
        if cluster_available(latest, deployment):
            return latest
        time.sleep(0.5)
    raise TimeoutError(f'cluster did not become available: {latest}')


def start_cluster(nodes: dict[str, dict[str, Any]], deployment: RedisDeployment) -> None:
    prepared = call_all(nodes, deployment, ensure_redis)
    if failed := [item for item in prepared if not item['ready']]:
        raise RuntimeError(f'Redis environment preparation failed: {failed}')
    versions = {item['server_version'] for item in prepared}
    if len(versions) != 1:
        raise RuntimeError(f'Redis Server versions must match across all nodes: {versions}')
    print_json('redis_environment', prepared)

    # nodes.conf survives process restarts. Never stop healthy nodes here:
    # a blanket restart would remove all in-memory copies at once. Write the
    # static config, then start only processes that are currently absent.
    call_instances(nodes, deployment, write_config)
    current = states(nodes, deployment)
    for state in current:
        if state['redis_ping'] != 'PONG':
            started = call_on(nodes, state['ip'], service, 'start', False, state['port'], str(deployment.remote_dir))
            require_service(started, 'Redis start')
    time.sleep(2)
    current = states(nodes, deployment)
    if failed_starts := [state for state in current if state['redis_ping'] != 'PONG']:
        raise RuntimeError(f'Redis start health check failed: {failed_starts}')

    port_reports = call_all(nodes, deployment, probe_cluster_ports, list(deployment.instances))
    print_json('cluster_network', port_reports)
    failed_ports = {
        report['source']: {endpoint: status for endpoint, status in report['results'].items() if status != 'ok'}
        for report in port_reports
        if any(status != 'ok' for status in report['results'].values())
    }
    if failed_ports:
        raise RuntimeError(f'Redis Cluster port preflight failed: {failed_ports}')

    fresh_states = [
        state
        for state in current
        if state['cluster'].get('cluster_known_nodes') == '1' and state['cluster'].get('cluster_slots_assigned') == '0'
    ]
    if len(fresh_states) == deployment.node_count:
        created = call_on(nodes, deployment.hosts[0], create_cluster, deployment)
        print_json('cluster_create', created)
        if created['returncode'] != 0 or not created['created']:
            raise RuntimeError(f'Redis Cluster creation did not complete: {created}')
    elif fresh_states and len(fresh_states) <= deployment.extra_replicas:
        initialized = next(state for state in current if state not in fresh_states)
        reference = f'{initialized["ip"]}:{initialized["port"]}'
        extras = [f'{state["ip"]}:{state["port"]}' for state in fresh_states]
        added = call_on(nodes, initialized['ip'], add_extra_replicas, reference, extras, deployment.primary_count)
        print_json('cluster_extra_replicas', added)
        if any(item['returncode'] != 0 for item in added):
            raise RuntimeError(f'Extra Redis replicas did not join: {added}')
    elif not cluster_healthy(current, deployment):
        # Unknown mixed metadata is ambiguous and automatic reset can destroy a
        # recoverable cluster, so only wait for normal reconnection.
        print_json('cluster_reconnect', current)

    print_json('status', wait_healthy(nodes, deployment))
    # A separate client host may not have redis-cli; run CLI smoke from a
    # Redis node while still routing the key through the Cluster.
    print_json('smoke', call_on(nodes, deployment.hosts[-1], smoke_node, deployment.instances[0]))


def recover_cluster(nodes: dict[str, dict[str, Any]], deployment: RedisDeployment) -> list[dict[str, Any]]:
    current = states(nodes, deployment)
    for state in current:
        if state['redis_ping'] != 'PONG':
            result = call_on(nodes, state['ip'], service, 'start', False, state['port'], str(deployment.remote_dir))
            print_json('recover_redis', result)
            require_service(result, 'Redis recovery')
    recovered = wait_healthy(nodes, deployment)
    print_json('recovered', recovered)
    return recovered


def endpoint_info(deployment: RedisDeployment) -> dict[str, Any]:
    seeds = ','.join(f'{ip}:{port}' for ip, port in deployment.instances)
    seed_ip, seed_port = deployment.instances[0]
    return {
        'redis_url': f'redis+cluster://{seed_ip}:{seed_port}/0?nodes={seeds}',
        'startup_nodes': seeds,
        'redis_port': REDIS_PORT,
        'cluster_bus_port': CLUSTER_BUS_PORT,
    }


def publish_endpoints(info: dict[str, Any], deployment: RedisDeployment) -> None:
    print_json('redis_endpoints', info)
    output_root = os.environ.get('TAIJI_BASIC_OUTPUT_PATH')
    target = (
        deployment.endpoint_file
        if deployment.endpoint_file is not None
        else Path(output_root) / 'redis-cluster.json'
        if output_root
        else deployment.remote_dir / 'endpoints.json'
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(info, ensure_ascii=False, indent=2) + '\n')
    print(f'redis_endpoint_file={target}')


def serve_cluster(nodes: dict[str, dict[str, Any]], deployment: RedisDeployment) -> None:
    start_cluster(nodes, deployment)
    publish_endpoints(endpoint_info(deployment), deployment)
    monitor_cluster(nodes, deployment)


def monitor_cluster(nodes: dict[str, dict[str, Any]], deployment: RedisDeployment) -> None:
    stopped = threading.Event()

    def stop(_signum: int, _frame: Any) -> None:
        stopped.set()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    healthy_checks = 0
    last_state = None
    while not stopped.wait(CONFIG['health_check_interval_seconds']):
        current = states(nodes, deployment)
        state = (
            'healthy'
            if cluster_healthy(current, deployment)
            else 'degraded'
            if cluster_available(current, deployment)
            else 'unavailable'
        )
        healthy_checks = healthy_checks + 1 if state == 'healthy' else 0
        # Healthy heartbeats are sampled; state changes and unhealthy checks are
        # always printed so failures remain visible without flooding Taiji logs.
        if state != 'healthy' or state != last_state or healthy_checks % CONFIG['healthy_log_every_checks'] == 0:
            seed_ip, seed_port = deployment.instances[0]
            print(f'redis_cluster_{state} startup_node={seed_ip}:{seed_port}')
        last_state = state
    print('redis_cluster_service_stopped')


def run(
    command: str,
    *,
    primary_count: int | None = None,
    replicas_per_primary: int | None = None,
    extra_replicas: int | None = None,
    redis_node_ips: str | None = None,
    redis_client_ip: str | None = None,
    redis_dir: str | Path | None = None,
    redis_endpoint_file: str | Path | None = None,
) -> int:
    connect_ray()
    deployment = resolve_topology(
        primary_count=primary_count,
        replicas_per_primary=replicas_per_primary,
        extra_replicas=extra_replicas,
        redis_node_ips=redis_node_ips,
        redis_client_ip=redis_client_ip,
        redis_dir=redis_dir,
        redis_endpoint_file=redis_endpoint_file,
    )
    nodes = node_map(deployment)
    if command == 'serve':
        serve_cluster(nodes, deployment)
    elif command == 'up':
        start_cluster(nodes, deployment)
    elif command == 'down':
        print_json('down', call_instances(nodes, deployment, service, 'stop', False))
        if running := [state for state in states(nodes, deployment) if state['redis_ping'] == 'PONG']:
            raise RuntimeError(f'Redis stop failed: instances are still running: {running}')
    elif command == 'status':
        current = states(nodes, deployment)
        print_json('status', current)
        if not cluster_available(current, deployment):
            raise SystemExit(1)
    elif command == 'smoke':
        print_json('smoke', call_on(nodes, deployment.hosts[-1], smoke_node, deployment.instances[0]))
    elif command == 'recover':
        recover_cluster(nodes, deployment)
    return 0
