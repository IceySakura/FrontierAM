"""Redis payload serialization, IDs, and client results."""

from __future__ import annotations

import pickle

from enum import Enum
from typing import Any
from uuid import uuid4


DEFAULT_MEMORY_WATERLINE_BYTES = 200 * 1024**3


# Client-generated globally unique id (uuid4 hex).
NodeId = str


class CompareAndSetResult(Enum):
    """Single-key Lua CAS outcomes, in the same order as the submitted updates."""

    UPDATED = 1
    MISMATCH = 0
    MISSING = -1


def new_node_id() -> NodeId:
    """Return a new client-side ``node_id`` (uuid4 hex, no dashes)."""
    return uuid4().hex


class Serializable:
    """Default pickle codec; subclasses changing the format override both methods.

    Reads select the decoder through an explicitly supplied payload class.
    """

    def serialize(self) -> bytes:
        return pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def deserialize(blob: bytes) -> object:
        """Restore the concrete payload; callers validate its expected type."""
        return pickle.loads(blob)


class RedisClientError(RuntimeError):
    """Raised when a Redis operation fails."""

    def __init__(self, message: str, *, code: str | None = None, body: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.body = body or {}


def _is_transport_error(error: Exception) -> bool:
    if isinstance(error, (ConnectionError, TimeoutError)):
        return True
    if isinstance(error, RedisClientError) and error.code in ('timeout', 'transport_error', 'closed'):
        return True
    # Redis is an optional dependency when clients are injected in unit tests.
    return any(
        base.__module__ == 'redis.exceptions' and base.__name__ in ('ConnectionError', 'TimeoutError')
        for base in type(error).__mro__
    )
