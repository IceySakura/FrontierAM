"""Importable payloads for client, codec and worker tests."""

from __future__ import annotations

import json
import struct

from dataclasses import dataclass
from typing import Any

from nexus.gateway.trajectory_store_backends.redis_store.types import Serializable


@dataclass
class SamplePayload(Serializable):
    data: Any = None
    text: str | None = None


@dataclass
class JsonPayload(Serializable):
    text: str

    def serialize(self) -> bytes:
        return json.dumps({'text': self.text}).encode()

    @staticmethod
    def deserialize(blob: bytes) -> JsonPayload:
        return JsonPayload(**json.loads(blob))


@dataclass
class BinaryPayload(Serializable):
    value: int

    def serialize(self) -> bytes:
        return struct.pack('!q', self.value)

    @staticmethod
    def deserialize(blob: bytes) -> BinaryPayload:
        (value,) = struct.unpack('!q', blob)
        return BinaryPayload(value)
