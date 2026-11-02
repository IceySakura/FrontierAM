"""Out-of-line storage for Gateway trajectory records."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .trajectory_store_backends.garnet import GarnetTrajectoryRecordStore
from .trajectory_store_backends.memory import InMemoryTrajectoryRecordStore
from .trajectory_store_backends.ray import RayTrajectoryRecordStore
from .trajectory_store_backends.redis import RedisTrajectoryRecordStore
from .trajectory_store_backends.types import (
    DumpRef,
    NodePayloadRecord,
    NodePayloadRef,
    SegmentDataRecord,
    SegmentDataRef,
    TrajectoryRecordStore,
)
from .trajectory_store_backends.types import (
    Record as Record,
)
from .trajectory_store_backends.types import (
    RecordRef as RecordRef,
)


def launch_servers(config: Mapping[str, Any]) -> tuple[dict[str, Any], Any]:
    """Return client configuration and an owning handler for backend servers."""
    backend = config.get('backend')
    if backend in ('memory', 'ray', 'redis'):
        return dict(config), None
    if backend == 'garnet':
        # NOTE(jerryhrzhou): Ray deployment dependencies load only when launching Garnet.
        from .trajectory_store_backends.garnet_server import launch_garnet_servers

        return launch_garnet_servers(config)
    raise ValueError(f'unsupported trajectory_store.backend: {backend!r}')


def stop_servers(handler: Any) -> None:
    """Stop backend servers owned by a launch handler."""
    if handler is not None:
        handler.stop()


def build_trajectory_record_store(config: Mapping[str, Any]) -> TrajectoryRecordStore:
    backend = config.get('backend')
    namespace = config.get('namespace')
    if not isinstance(namespace, str) or not namespace:
        raise ValueError('trajectory_store.namespace must be a non-empty string')
    if backend == 'memory':
        return InMemoryTrajectoryRecordStore(namespace)
    if backend == 'ray':
        return RayTrajectoryRecordStore(namespace)
    if backend == 'garnet':
        garnet_config = config.get('garnet')
        if garnet_config is None:
            garnet_config = {}
        if not isinstance(garnet_config, Mapping):
            raise ValueError('trajectory_store.garnet must be a mapping when provided')
        return GarnetTrajectoryRecordStore(namespace, garnet_config)
    if backend == 'redis':
        redis_config = config.get('redis')
        if redis_config is None:
            redis_config = {}
        if not isinstance(redis_config, Mapping):
            raise ValueError('trajectory_store.redis must be a mapping when provided')
        return RedisTrajectoryRecordStore(namespace, redis_config)
    raise ValueError(f'unsupported trajectory_store.backend: {backend!r}')


__all__ = [
    'DumpRef',
    'GarnetTrajectoryRecordStore',
    'InMemoryTrajectoryRecordStore',
    'NodePayloadRecord',
    'NodePayloadRef',
    'RayTrajectoryRecordStore',
    'RedisTrajectoryRecordStore',
    'SegmentDataRecord',
    'SegmentDataRef',
    'TrajectoryRecordStore',
    'build_trajectory_record_store',
    'launch_servers',
    'stop_servers',
]
