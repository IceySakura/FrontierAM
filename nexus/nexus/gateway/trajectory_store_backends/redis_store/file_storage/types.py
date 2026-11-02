"""Contracts for opaque blobs; Redis and trajectory serialization stay outside FS."""

from __future__ import annotations

import hashlib
import re

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from uuid import uuid4

from nexus.gateway.trajectory_store_backends.redis_store.types import RedisClientError


SLOT_COUNT = 256
LARGE_FIELDS = ('routed_experts', 'multi_modal_items')


@dataclass(frozen=True)
class Endpoint:
    host: str
    port: int

    def __post_init__(self):
        if not self.host or not 0 <= self.port <= 65535:
            raise ValueError('invalid endpoint')

    @property
    def tcp_address(self) -> str:
        return f'tcp://{self.host}:{self.port}'

    @property
    def bind_address(self) -> str:
        """Bind to the advertised interface; wildcard endpoints cannot be published."""
        return self.tcp_address


@dataclass(frozen=True)
class DataUnitRegistration:
    data_unit_id: str
    endpoint: Endpoint


@dataclass(frozen=True)
class Topology:
    """Immutable membership for one run. Heartbeat expiry never remaps slots."""

    cluster_id: str
    data_units: tuple[DataUnitRegistration, ...]
    slots: tuple[str, ...]

    def __post_init__(self):
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', self.cluster_id):
            raise ValueError('invalid cluster_id')
        ids = [unit.data_unit_id for unit in self.data_units]
        if not 1 <= len(ids) <= SLOT_COUNT or any(not item for item in ids) or len(set(ids)) != len(ids):
            raise ValueError('expected 1-256 unique data unit IDs')
        endpoints = [unit.endpoint for unit in self.data_units]
        if len(set(endpoints)) != len(endpoints):
            raise ValueError('duplicate data unit endpoints')
        if any(ep.port == 0 or ep.host in ('*', '0.0.0.0', '::') for ep in endpoints):
            raise ValueError('published endpoints must have reachable hosts and concrete ports')
        if len(self.slots) != SLOT_COUNT or set(self.slots) != set(ids):
            raise ValueError('all 256 slots and all units must be assigned')

    @classmethod
    def create(cls, units, cluster_id: str | None = None) -> Topology:
        units = tuple(sorted(units, key=lambda unit: unit.data_unit_id))
        if not units:
            raise ValueError('at least one data unit is required')
        return cls(
            cluster_id or uuid4().hex, units, tuple(units[i % len(units)].data_unit_id for i in range(SLOT_COUNT))
        )

    def new_url(self) -> str:
        return f'hykvfs://{self.cluster_id}/{uuid4().hex}'

    def slot(self, url: str) -> int:
        prefix = f'hykvfs://{self.cluster_id}/'
        if (
            not isinstance(url, str)
            or not url.startswith(prefix)
            or not re.fullmatch(r'[0-9a-f]{32}', url[len(prefix) :])
        ):
            raise RedisClientError('invalid URL or wrong FS cluster', code='invalid_url')
        # Built-in hash() is process-randomized and cannot be used for routing.
        return hashlib.blake2b(url.encode(), digest_size=8).digest()[0]

    def owner(self, url: str) -> DataUnitRegistration:
        owner_id = self.slots[self.slot(url)]
        return next(unit for unit in self.data_units if unit.data_unit_id == owner_id)

    def dump(self) -> dict:
        return asdict(self)

    @classmethod
    def load(cls, data: dict) -> Topology:
        units = tuple(
            DataUnitRegistration(item['data_unit_id'], Endpoint(**item['endpoint'])) for item in data['data_units']
        )
        return cls(data['cluster_id'], units, tuple(data['slots']))


class BlobClient(ABC):
    """Thread-safe client interface; only callers encode or decode payloads."""

    @abstractmethod
    def allocate_url(self) -> str: ...

    @abstractmethod
    def put(self, url: str, blob: bytes | bytearray | memoryview) -> None: ...

    @abstractmethod
    def get(self, url: str) -> bytes | bytearray | memoryview: ...

    @abstractmethod
    def delete(self, url: str) -> bool: ...

    @abstractmethod
    def close(self) -> None: ...
