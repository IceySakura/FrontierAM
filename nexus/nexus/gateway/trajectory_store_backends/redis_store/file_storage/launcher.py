"""Named Ray owner for FS services that outlive the startup command."""

from __future__ import annotations

import json
import logging

from pathlib import Path
from typing import Any, cast

import nexus

from nexus.gateway.trajectory_store_backends.redis_store.file_storage.deploy import (
    ClusterInfo,
    RayCluster,
    _validate_data_unit_options,
)

from ..utils import get_ray_head_ip


class _ClusterManager:
    def __init__(self):
        self._cluster: RayCluster | None = None
        self._config: dict | None = None

    def start(self, config: dict) -> dict:
        if self._cluster is None:
            self._cluster = RayCluster(**config).start()
            self._config = config
        elif config != self._config:
            raise ValueError(
                'FS is already running with different options; use nexus redis fs down before changing them'
            )
        with self._cluster.fs_client():
            pass  # Return only after the Controller and every DataUnit answer readiness probes.
        info = cast(ClusterInfo, self._cluster.info)
        return {'redis_url': self._cluster.redis_url, **info.dump()}

    def stop(self):
        if self._cluster is not None:
            self._cluster.stop()
            self._cluster = None
            self._config = None


def run(
    action: str,
    *,
    redis_url: str | None = None,
    num_data_units: int | None = None,
    controller_port: int = 17000,
    startup_timeout: float = 60,
    **data_unit_options,
) -> int:
    _validate_data_unit_options(data_unit_options)
    import ray

    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    if action not in ('up', 'down'):
        raise ValueError(f'unknown FS command: {action}')
    if not 1 <= controller_port <= 65535:
        raise ValueError('controller_port must be between 1 and 65535 for FS up/down')
    # Ship the installed package as well as installing the FS extras on workers.
    # The participating hosts need the normal Nexus base environment and Ray.
    # Ray Job config merging deep-copies this value, so pass a path instead of a module object.
    ray.init(
        address='auto',
        namespace='nexus-redis-fs',
        log_to_driver=False,
        logging_level=logging.ERROR,
        runtime_env={'py_modules': [str(Path(nexus.__file__).resolve().parent)]} if action == 'up' else None,
    )
    try:
        host = get_ray_head_ip()
        name = f'controller-{host}-{controller_port}'
        if action == 'down':
            if name in ray.util.list_named_actors():
                manager = ray.get_actor(name)
                try:
                    ray.get(manager.stop.remote(), timeout=30)
                finally:
                    ray.kill(manager, no_restart=True)
            print(json.dumps({'redis_fs_stopped': {'host': host, 'port': controller_port}}), flush=True)
            return 0

        nodes = {node['NodeManagerAddress']: node['NodeID'] for node in ray.nodes() if node['Alive']}
        if host not in nodes:
            raise ValueError(f'FS Controller host {host!r} is not an alive Ray node')
        node_ids = [nodes[host], *(node_id for ip, node_id in sorted(nodes.items()) if ip != host)]
        if num_data_units is None:
            num_data_units = len(node_ids)
        config = {
            'redis_url': redis_url or f'redis://{host}:16379/0',
            'num_data_units': num_data_units,
            'data_unit_node_ids': [node_ids[index % len(node_ids)] for index in range(num_data_units)],
            'controller_node_id': nodes[host],
            'controller_port': controller_port,
            'startup_timeout': startup_timeout,
            **data_unit_options,
        }
        manager = (
            cast(Any, ray.remote(num_cpus=0, num_gpus=0, max_restarts=0)(_ClusterManager))
            .options(
                name=name,
                lifetime='detached',
                get_if_exists=True,
                scheduling_strategy=NodeAffinitySchedulingStrategy(nodes[host], soft=False),
                runtime_env={'pip': ['redis>=8.1,<9', 'pyzmq>=25']},
            )
            .remote()
        )
        info = ray.get(manager.start.remote(config), timeout=4 * startup_timeout)
        print(json.dumps({'redis_fs_endpoints': info}), flush=True)
        return 0
    finally:
        ray.shutdown()
