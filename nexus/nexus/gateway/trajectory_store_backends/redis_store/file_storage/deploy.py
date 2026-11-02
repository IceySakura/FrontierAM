"""Local threads or Ray actor shells; data always travels through TCP endpoints."""

from __future__ import annotations

import json
import logging
import signal
import threading

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from nexus.gateway.trajectory_store_backends.redis_store.client import connect_redis
from nexus.gateway.trajectory_store_backends.redis_store.file_storage.client import FSClient
from nexus.gateway.trajectory_store_backends.redis_store.file_storage.controller import ControllerService
from nexus.gateway.trajectory_store_backends.redis_store.file_storage.data_unit import DataUnitService
from nexus.gateway.trajectory_store_backends.redis_store.file_storage.types import (
    DataUnitRegistration,
    Endpoint,
    Topology,
)


@dataclass(frozen=True)
class ClusterInfo:
    controller_endpoint: Endpoint
    topology: Topology

    @property
    def data_unit_endpoints(self):
        return [unit.endpoint for unit in self.topology.data_units]

    @property
    def num_data_units(self):
        return len(self.topology.data_units)

    def dump(self):
        return {
            **asdict(self),
            'data_unit_endpoints': [asdict(endpoint) for endpoint in self.data_unit_endpoints],
            'num_data_units': self.num_data_units,
        }

    @classmethod
    def load(cls, data):
        return cls(Endpoint(**data['controller_endpoint']), Topology.load(data['topology']))


def _cluster_client(redis_url, info, options):
    from nexus.gateway.trajectory_store_backends.redis_store.client import RedisClient

    if info is None:
        raise RuntimeError('start the cluster first')
    if {'url', 'redis_url', 'redis_client', 'fs_client', 'controller_endpoint'} & options.keys():
        raise ValueError('cluster.client() cannot override cluster storage connections')
    return RedisClient(url=redis_url, controller_endpoint=info.controller_endpoint, **options)


def _validate_data_unit_options(options: dict[str, Any]) -> None:
    unknown = options.keys() - {
        'ssd_offload_dir',
        'memory_waterline_bytes',
        'memory_safe_line_bytes',
        'request_workers',
        'ssd_workers',
        'copy_workers',
    }
    if unknown:
        raise TypeError(f'unexpected DataUnit options: {", ".join(sorted(unknown))}')


class LocalCluster:
    """In-process services for development. The Redis backend remains external."""

    def __init__(
        self,
        redis_url: str = 'redis://127.0.0.1:16379/0',
        *,
        num_data_units: int = 2,
        host: str = '127.0.0.1',
        **data_unit_options,
    ):
        _validate_data_unit_options(data_unit_options)
        if not 1 <= num_data_units <= 256:
            raise ValueError('num_data_units must be between 1 and 256')
        self._redis = None
        self._redis_url = redis_url
        self._units = []
        self._controller = None
        self.info = None
        self._count, self._host = num_data_units, host
        self._data_unit_options = data_unit_options

    def start(self):
        if self.info is not None:
            raise RuntimeError('cluster is already running')
        if self._redis is None:
            self._redis = connect_redis(self._redis_url, socket_timeout=5, socket_connect_timeout=5)
        try:
            for index in range(self._count):
                unit = DataUnitService(f'unit-{index}', Endpoint(self._host, 0), **self._data_unit_options)
                self._units.append(unit)
                unit.start()
            topology = Topology.create([DataUnitRegistration(unit.data_unit_id, unit.endpoint) for unit in self._units])
            self._controller = ControllerService(self._redis, topology, Endpoint(self._host, 0)).start()
            for unit in self._units:
                unit.configure(topology, self._controller.endpoint)
            self.info = ClusterInfo(self._controller.endpoint, topology)
            return self
        except BaseException:
            self.stop()
            raise

    def client(self, **kwargs):
        """Return a complete Redis+FS client; the caller owns its connections."""
        return _cluster_client(self._redis_url, self.info, kwargs)

    def fs_client(self, **kwargs) -> FSClient:
        """Return only the opaque-blob interface, without a Redis payload client."""
        if self.info is None:
            raise RuntimeError('start the cluster first')
        return FSClient(self.info.controller_endpoint, **kwargs)

    def stop(self):
        for unit in reversed(self._units):
            unit.stop()
        self._units.clear()
        if self._controller is not None:
            self._controller.stop()
            self._controller = None
        if self._redis is not None:
            self._redis.close()
            self._redis = None
        self.info = None

    def __enter__(self):
        return self if self.info is not None else self.start()

    def __exit__(self, *_):
        self.stop()


class _DataUnitActor:
    def __init__(self, unit_id, data_unit_options):
        import ray

        host = ray.util.get_node_ip_address()
        self.service = DataUnitService(unit_id, Endpoint(host, 0), **data_unit_options).start()

    def registration(self):
        return DataUnitRegistration(self.service.data_unit_id, self.service.endpoint)

    def configure(self, topology, endpoint):
        self.service.configure(topology, endpoint)

    def stop(self):
        self.service.stop()


class _ControllerActor:
    def __init__(self, redis_url, topology, port):
        import ray

        self.redis = connect_redis(redis_url, socket_timeout=5, socket_connect_timeout=5)
        self.service = ControllerService(self.redis, topology, Endpoint(ray.util.get_node_ip_address(), port)).start()

    def endpoint(self):
        return self.service.endpoint

    def stop(self):
        self.service.stop()
        self.redis.close()


class RayCluster:
    """Ray only schedules services. Install this package on every participating node.

    Actors are not automatically restarted: in-memory blobs cannot survive a
    restart. Redis and advertised FS ports must be reachable across all nodes.
    """

    def __init__(
        self,
        redis_url: str,
        *,
        num_data_units: int = 2,
        data_unit_node_ids=None,
        controller_node_id: str | None = None,
        controller_port: int = 0,
        startup_timeout: float = 60,
        **data_unit_options,
    ):
        _validate_data_unit_options(data_unit_options)
        if not 1 <= num_data_units <= 256 or startup_timeout <= 0:
            raise ValueError('expected 1-256 data units and a positive startup timeout')
        if data_unit_node_ids is not None and len(data_unit_node_ids) != num_data_units:
            raise ValueError('provide one Ray node ID per data unit')
        if not 0 <= controller_port <= 65535:
            raise ValueError('controller_port must be between 0 and 65535')
        self.redis_url, self.count = redis_url, num_data_units
        self._node_ids = list(data_unit_node_ids) if data_unit_node_ids is not None else None
        self._controller_node_id = controller_node_id
        self._controller_port = controller_port
        self._startup_timeout = startup_timeout
        self._data_unit_options = data_unit_options
        self._units: list[Any] = []
        self._controller: Any = None
        self.info = None

    def start(self):
        import ray

        if not ray.is_initialized():
            raise RuntimeError('call ray.init(...) before starting RayCluster')
        if self.info is not None:
            raise RuntimeError('cluster is already running')
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        requested = set(self._node_ids or []) | ({self._controller_node_id} if self._controller_node_id else set())
        alive = {node['NodeID'] for node in ray.nodes() if node['Alive']}
        if not requested <= alive:
            raise ValueError('requested Ray node is not alive')
        try:
            actor = ray.remote(num_cpus=1, num_gpus=0, max_restarts=0)(_DataUnitActor)
            for index in range(self.count):
                placed = (
                    actor
                    if self._node_ids is None
                    else actor.options(
                        scheduling_strategy=NodeAffinitySchedulingStrategy(self._node_ids[index], soft=False)
                    )
                )
                self._units.append(placed.remote(f'unit-{index}', self._data_unit_options))
            topology = Topology.create(
                ray.get([unit.registration.remote() for unit in self._units], timeout=self._startup_timeout)
            )
            controller_actor = ray.remote(num_cpus=1, num_gpus=0, max_restarts=0)(_ControllerActor)
            if self._controller_node_id is not None:
                controller_actor = controller_actor.options(
                    scheduling_strategy=NodeAffinitySchedulingStrategy(self._controller_node_id, soft=False)
                )
            self._controller = cast(Any, controller_actor.remote(self.redis_url, topology, self._controller_port))
            endpoint = ray.get(self._controller.endpoint.remote(), timeout=self._startup_timeout)
            ray.get([unit.configure.remote(topology, endpoint) for unit in self._units], timeout=self._startup_timeout)
            self.info = ClusterInfo(endpoint, topology)
            return self
        except BaseException:
            self.stop()
            raise

    def client(self, **kwargs):
        return _cluster_client(self.redis_url, self.info, kwargs)

    def fs_client(self, **kwargs) -> FSClient:
        if self.info is None:
            raise RuntimeError('start the cluster first')
        return FSClient(self.info.controller_endpoint, **kwargs)

    def stop(self):
        import ray

        actors = [*self._units, *([self._controller] if self._controller is not None else [])]
        try:
            if actors:
                ray.get([actor.stop.remote() for actor in actors], timeout=15)
        finally:
            for actor in actors:
                ray.kill(actor, no_restart=True)
            self._units.clear()
            self._controller = None
            self.info = None

    def __enter__(self):
        return self if self.info is not None else self.start()

    def __exit__(self, *_):
        self.stop()


def start_local_cluster(*args, **kwargs) -> LocalCluster:
    """Start independent FS services; cluster.client() returns an FS-enabled RedisClient."""
    return LocalCluster(*args, **kwargs).start()


def start_ray_cluster(*args, **kwargs) -> RayCluster:
    """Start CPU-only Ray service actors in an already initialized Ray cluster."""
    return RayCluster(*args, **kwargs).start()


def serve_ray_cluster(config: dict) -> int:
    """Join existing Ray and own the FS actors for the lifetime of this driver."""
    import ray

    cluster = RayCluster(**config)
    stopped = threading.Event()
    previous = {sig: signal.signal(sig, lambda *_: stopped.set()) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        ray.init(address='auto', log_to_driver=False, logging_level=logging.ERROR)
        with cluster:
            info = cast(ClusterInfo, cluster.info)
            print(json.dumps({'redis_fs_endpoints': {'redis_url': cluster.redis_url, **info.dump()}}), flush=True)
            stopped.wait()
    finally:
        ray.shutdown()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return 0


def serve_service(
    role: str,
    topology_path: str,
    *,
    redis_url: str = 'redis://127.0.0.1:16379/0',
    controller_host: str = '127.0.0.1',
    controller_port: int = 17000,
    unit_id: str | None = None,
    **data_unit_options,
) -> int:
    """Run one service process from the same topology used by all participants."""
    _validate_data_unit_options(data_unit_options)
    topology = Topology.load(json.loads(Path(topology_path).read_text()))
    endpoint = Endpoint(controller_host, controller_port)
    redis = None
    if role == 'controller':
        redis = connect_redis(redis_url, socket_timeout=5, socket_connect_timeout=5)
        service = ControllerService(redis, topology, endpoint)
    elif role == 'data-unit':
        registration = next((unit for unit in topology.data_units if unit.data_unit_id == unit_id), None)
        if registration is None:
            raise ValueError('--unit-id must identify a unit in the topology file')
        service = DataUnitService(
            registration.data_unit_id,
            registration.endpoint,
            **data_unit_options,
        )
    else:
        raise ValueError(f'unknown FS service role: {role}')
    stopped = threading.Event()
    previous = {sig: signal.signal(sig, lambda *_: stopped.set()) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        service.start()
        if role == 'data-unit':
            cast(DataUnitService, service).configure(topology, endpoint)
        print(json.dumps({'role': role, 'endpoint': asdict(service.endpoint)}), flush=True)
        stopped.wait()
    finally:
        try:
            service.stop()
        finally:
            if redis is not None:
                redis.close()
            for sig, handler in previous.items():
                signal.signal(sig, handler)
    return 0
