"""Discovery only: authoritative topology and heartbeat state live in Redis."""

from __future__ import annotations

import json
import time

from uuid import uuid4

from nexus.gateway.trajectory_store_backends.redis_store.file_storage.network import RouterServer
from nexus.gateway.trajectory_store_backends.redis_store.file_storage.types import Endpoint, Topology
from nexus.gateway.trajectory_store_backends.redis_store.types import RedisClientError


class ControllerService:
    def __init__(
        self,
        redis_client,
        topology: Topology,
        endpoint: Endpoint = Endpoint('127.0.0.1', 0),
        *,
        key_prefix: str = 'hykv',
        heartbeat_ttl: int = 15,
        controller_id: str | None = None,
    ):
        if heartbeat_ttl <= 0:
            raise ValueError('heartbeat_ttl must be positive')
        self._redis = redis_client
        self._initial_topology = topology
        self._prefix = f'{key_prefix.rstrip(":")}:fs:{topology.cluster_id}'
        self._topology_key = f'{self._prefix}:topology'
        self._heartbeat_ttl = heartbeat_ttl
        self.controller_id = controller_id or uuid4().hex
        self._server = RouterServer(
            endpoint, self._handle, server_id=self.controller_id, control_operations=('heartbeat',)
        )

    @property
    def endpoint(self) -> Endpoint:
        return self._server.endpoint

    def start(self):
        data = json.dumps(self._initial_topology.dump(), sort_keys=True)
        if not self._redis.set(self._topology_key, data, nx=True) and self.topology() != self._initial_topology:
            raise ValueError('cluster membership is immutable; use a new cluster ID')
        self._server.start()
        return self

    def topology(self) -> Topology:
        blob = self._redis.get(self._topology_key)
        if blob is None:
            raise RedisClientError('FS topology is missing from Redis', code='not_ready')
        return Topology.load(json.loads(blob))

    def list_data_units(self):
        return self.topology().data_units

    def receive_heartbeat(self, data_unit_id: str, endpoint: Endpoint, *, stats: dict[str, int]):
        registration = next(
            (unit for unit in self._initial_topology.data_units if unit.data_unit_id == data_unit_id), None
        )
        if registration is None or registration.endpoint != endpoint:
            raise RedisClientError('unregistered data unit', code='invalid_registration')
        report = {'received_at': time.time(), 'stats': stats}
        self._redis.set(f'{self._prefix}:heartbeat:{data_unit_id}', json.dumps(report), ex=self._heartbeat_ttl)

    def _handle(self, operation: str, body: dict, blob: bytes):
        if blob:
            raise ValueError('controller accepts metadata only')
        if operation == 'heartbeat':
            self.receive_heartbeat(
                body['data_unit_id'],
                Endpoint(**body['endpoint']),
                stats=body['stats'],
            )
            return {}, b''
        topology = self.topology()
        if operation == 'handshake':
            if body.get('protocol_version') != 1:
                raise RedisClientError('unsupported FS protocol', code='protocol_mismatch')
            return {'protocol_version': 1, 'controller_id': self.controller_id, 'cluster_id': topology.cluster_id}, b''
        if operation in ('discover', 'list_data_units'):
            return topology.dump(), b''
        if operation == 'status':
            # One-key commands remain valid across Redis Cluster slots.
            with self._redis.pipeline(transaction=False) as pipe:
                for unit in topology.data_units:
                    pipe.get(f'{self._prefix}:heartbeat:{unit.data_unit_id}')
                values = pipe.execute()
            reports = [json.loads(value) if value is not None else None for value in values]
            return {
                'units': [
                    {
                        'data_unit_id': unit.data_unit_id,
                        'alive': report is not None,
                        'stats': report['stats'] if report is not None else None,
                    }
                    for unit, report in zip(topology.data_units, reports, strict=True)
                ]
            }, b''
        raise ValueError(f'unknown controller operation: {operation}')

    def stop(self):
        self._server.stop()
        # The Redis connection is borrowed. State is retained for inspection.
