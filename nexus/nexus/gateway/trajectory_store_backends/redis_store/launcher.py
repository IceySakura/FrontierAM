"""Start Redis and detached FS services without sharing their Ray driver lifecycle."""

from __future__ import annotations

import subprocess
import sys


def run(
    action: str,
    *,
    num_data_units: int | None = None,
    controller_port: int = 17000,
    startup_timeout: float = 60,
    ssd_offload_dir: str | None = None,
    memory_waterline_bytes: int | None = None,
    memory_safe_line_bytes: int | None = None,
    request_workers: int | None = None,
    ssd_workers: int | None = None,
    copy_workers: int | None = None,
    **redis_options,
) -> int:
    if action not in ('up', 'serve', 'down'):
        raise ValueError(f'unknown Redis service command: {action}')

    from nexus.gateway.trajectory_store_backends.redis_store import cluster

    command = [sys.executable, '-m', 'nexus.cli.main', 'redis', 'fs']
    if action == 'down':
        result = subprocess.run([*command, 'down', '--controller-port', str(controller_port)], check=False)
        return result.returncode or cluster.run('down', **redis_options)

    cluster.connect_ray()
    deployment = cluster.resolve_topology(**redis_options)
    nodes = cluster.node_map(deployment)
    cluster.start_cluster(nodes, deployment)
    info = cluster.endpoint_info(deployment)
    command.extend(['up', '--redis-url', info['redis_url']])
    for name, value in {
        'num_data_units': num_data_units,
        'controller_port': controller_port,
        'startup_timeout': startup_timeout,
        'ssd_offload_dir': ssd_offload_dir,
        'memory_waterline_bytes': memory_waterline_bytes,
        'memory_safe_line_bytes': memory_safe_line_bytes,
        'request_workers': request_workers,
        'ssd_workers': ssd_workers,
        'copy_workers': copy_workers,
    }.items():
        if value is not None:
            command.extend([f'--{name.replace("_", "-")}', str(value)])
    result = subprocess.run(command, check=False)
    if result.returncode:
        return result.returncode
    if action == 'serve':
        cluster.publish_endpoints(info, deployment)
        cluster.monitor_cluster(nodes, deployment)
    return 0
