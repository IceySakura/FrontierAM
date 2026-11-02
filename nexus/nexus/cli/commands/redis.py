"""Redis Cluster and FS service commands; optional dependencies load on dispatch."""

from __future__ import annotations

import argparse
import json
import re
import subprocess

from importlib.resources import as_file, files
from pathlib import Path


def parse_memory_bytes(value: str) -> int:
    """Parse integer bytes or a K/M/G/T size using powers of 1024."""
    match = re.fullmatch(r'([0-9]+)([KMGT]?)', value.strip(), re.IGNORECASE)
    if match is None:
        raise argparse.ArgumentTypeError('expected integer bytes or a K/M/G/T size, e.g. 200G (1024-based units)')
    amount, unit = match.groups()
    power = 'KMGT'.index(unit.upper()) + 1 if unit else 0
    return int(amount) * 1024**power


def register_redis_cluster_arguments(parser: argparse.ArgumentParser, *, serve: bool = False) -> None:
    parser.add_argument('--primary-count', type=int, help='Total Primary instances; at least 3')
    parser.add_argument('--replicas-per-primary', type=int, help='Replica instances per Primary')
    parser.add_argument('--extra-replicas', type=int, help='Additional Replica instances')
    parser.add_argument('--redis-node-ips', help='Comma-separated Redis host IPs; defaults to Taiji or alive Ray hosts')
    parser.add_argument('--redis-client-ip', help='Host IP for Redis client checks')
    parser.add_argument('--redis-dir', help='Node-local Redis data and cluster metadata directory')
    if serve:
        parser.add_argument('--redis-endpoint-file', help='Output path for the Redis endpoint JSON')


def _register_fs_start_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--num-data-units', type=int, help='Defaults to one DataUnit per alive Ray node')
    parser.add_argument('--controller-port', type=int)
    parser.add_argument('--startup-timeout', type=float)


def register_redis_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser('redis', help='Manage Redis Cluster and FS services')
    parser.set_defaults(_redis_parser=parser)
    services = parser.add_subparsers(dest='redis_service')
    data_services = []
    for command, help_text in (
        ('up', 'Start Redis and background FS services, then wait until ready'),
        ('serve', 'Start Redis and FS services, publish endpoints and monitor Redis'),
        ('down', 'Stop FS services, then stop Redis nodes'),
    ):
        combined = services.add_parser(command, help=help_text)
        combined.set_defaults(redis_action=command)
        register_redis_cluster_arguments(combined, serve=command == 'serve')
        if command == 'down':
            combined.add_argument('--controller-port', type=int)
        else:
            _register_fs_start_arguments(combined)
            data_services.append(combined)

    cluster = services.add_parser('cluster', help='Manage the configured Redis Cluster on existing Ray hosts')
    cluster.set_defaults(_redis_parser=cluster)
    cluster_commands = cluster.add_subparsers(dest='redis_action')
    for command, help_text in (
        ('prepare', 'Prepare the Taiji environment before starting Redis'),
        ('serve', 'Start Redis, publish endpoints and monitor until interrupted'),
        ('up', 'Start or complete the configured Redis Cluster'),
        ('down', 'Stop Redis nodes'),
        ('status', 'Report cluster state; exit 1 when unavailable'),
        ('smoke', 'Run the Redis read/write smoke check'),
        ('recover', 'Restart stopped Redis nodes and wait for recovery'),
    ):
        operation = cluster_commands.add_parser(command, help=help_text)
        if command != 'prepare':
            register_redis_cluster_arguments(operation, serve=command == 'serve')

    fs = services.add_parser('fs', help='Run FS services using an existing Redis backend')
    fs.set_defaults(_redis_parser=fs)
    fs_commands = fs.add_subparsers(dest='redis_action')
    up = fs_commands.add_parser('up', help='Start background FS services on existing Ray and wait until ready')
    up.add_argument('--redis-url', help='Existing Redis URL; defaults to port 16379 on the connected Ray head')
    _register_fs_start_arguments(up)
    down = fs_commands.add_parser('down', help='Stop the FS services created by up')
    down.add_argument('--controller-port', type=int)
    serve = fs_commands.add_parser('serve', help='Run FS actors on an existing Ray cluster')
    serve.add_argument('--config', required=True, help='JSON file with RayCluster connection and placement parameters')
    data_services.append(up)
    for role in ('controller', 'data-unit'):
        service = fs_commands.add_parser(role, help=f'Run one FS {role} process')
        service.set_defaults(_redis_parser=service)
        service.add_argument('--topology', required=True, help='JSON file with cluster_id, data_units and 256 slots')
        service.add_argument('--redis-url')
        service.add_argument('--controller-host')
        service.add_argument('--controller-port', type=int)
        service.add_argument('--unit-id', required=role == 'data-unit')
        if role == 'data-unit':
            data_services.append(service)
    for data_service in data_services:
        for option, description in (
            ('request-workers', 'Ordinary request workers'),
            ('ssd-workers', 'Parallel SSD preparation workers'),
            ('copy-workers', 'Parallel checkpoint file copy workers'),
        ):
            data_service.add_argument(
                f'--{option}',
                type=int,
                default=argparse.SUPPRESS,
                help=f'{description} per DataUnit; defaults to 8',
            )
        data_service.add_argument(
            '--ssd-offload-dir', default=argparse.SUPPRESS, help='Node-local SSD directory; defaults to .hykvstore'
        )
        data_service.add_argument(
            '--memory-waterline-bytes',
            type=parse_memory_bytes,
            default=argparse.SUPPRESS,
            help='Per-DataUnit memory limit; defaults to 200G; integer bytes or K/M/G/T',
        )
        data_service.add_argument(
            '--memory-safe-line-bytes',
            type=parse_memory_bytes,
            default=argparse.SUPPRESS,
            help='Eviction target; defaults to half the limit; integer bytes or K/M/G/T, e.g. 100G',
        )


def handle_redis_command(args: argparse.Namespace) -> int:
    action = getattr(args, 'redis_action', None)
    if action is None:
        args._redis_parser.print_help()
        return 0
    service_options = {
        name: value
        for name, value in vars(args).items()
        if name
        in (
            'primary_count',
            'replicas_per_primary',
            'extra_replicas',
            'redis_node_ips',
            'redis_client_ip',
            'redis_dir',
            'redis_endpoint_file',
            'num_data_units',
            'controller_port',
            'startup_timeout',
            'redis_url',
            'controller_host',
            'unit_id',
            'ssd_offload_dir',
            'memory_waterline_bytes',
            'memory_safe_line_bytes',
            'request_workers',
            'ssd_workers',
            'copy_workers',
        )
        and value is not None
    }
    if args.redis_service == 'cluster':
        if action == 'prepare':
            resource = files('nexus.gateway.trajectory_store_backends.redis_store').joinpath(
                'redis_server_launcher_prepare.sh'
            )
            with as_file(resource) as script:
                return subprocess.run(['bash', str(script)], check=False).returncode

        from nexus.gateway.trajectory_store_backends.redis_store.cluster import run

        return run(action, **service_options)
    if args.redis_service in ('up', 'serve', 'down'):
        from nexus.gateway.trajectory_store_backends.redis_store.launcher import run

        return run(action, **service_options)
    if action in ('up', 'down'):
        from nexus.gateway.trajectory_store_backends.redis_store.file_storage.launcher import run

        return run(action, **service_options)
    if action == 'serve':
        from nexus.gateway.trajectory_store_backends.redis_store.file_storage.deploy import serve_ray_cluster

        return serve_ray_cluster(json.loads(Path(args.config).read_text()))
    from nexus.gateway.trajectory_store_backends.redis_store.file_storage.deploy import serve_service

    return serve_service(action, args.topology, **service_options)
