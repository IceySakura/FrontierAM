"""Payload shared by benchmark clients and independently importing workers."""

from dataclasses import dataclass
from typing import Any

from nexus.gateway.trajectory_store_backends.redis_store.types import Serializable


@dataclass
class BenchmarkPayload(Serializable):
    data: Any
    text: str | None = None
