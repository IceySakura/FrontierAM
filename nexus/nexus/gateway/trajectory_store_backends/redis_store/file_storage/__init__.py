"""Independent in-memory blob storage. No replicas, migration, or recovery.

RedisClient always uses FS, through a supplied blob client or a controller endpoint.
"""

from nexus.gateway.trajectory_store_backends.redis_store.file_storage.client import FSClient
from nexus.gateway.trajectory_store_backends.redis_store.file_storage.deploy import (
    ClusterInfo,
    LocalCluster,
    RayCluster,
    start_local_cluster,
    start_ray_cluster,
)


__all__ = ['ClusterInfo', 'FSClient', 'LocalCluster', 'RayCluster', 'start_local_cluster', 'start_ray_cluster']
