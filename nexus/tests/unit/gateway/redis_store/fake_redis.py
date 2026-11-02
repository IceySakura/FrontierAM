"""In-memory Redis stand-in for unit tests (no redis-py / server required).

Implements the subset used by ``RedisClient``: get/set/mget, exists, delete,
pipeline (plus unused incr/incrby helpers).
"""

from __future__ import annotations

import hashlib
import threading

from typing import Any
from unittest.mock import Mock


class FakeRedis:
    """Minimal dict-backed Redis mock."""

    def __init__(self) -> None:
        self._kv: dict[bytes, bytes] = {}
        self._ints: dict[bytes, int] = {}
        self._dump_protections: dict[bytes, set[str]] = {}
        self._lock = threading.Lock()
        self.connection_pool = Mock(spec=['get_connection', 'release'])
        self.connection_pool.get_connection.side_effect = lambda: Mock(spec=['send_command', 'disconnect'])

    def incr(self, key: bytes) -> int:
        self._ints[key] = int(self._ints.get(key, 0)) + 1
        return self._ints[key]

    def incrby(self, key: bytes, amount: int) -> int:
        self._ints[key] = int(self._ints.get(key, 0)) + int(amount)
        return self._ints[key]

    def set(self, key: bytes, value: bytes, nx: bool = False) -> bool | None:
        if nx and key in self._kv:
            return False
        self._kv[key] = value
        return True

    def get(self, key: bytes) -> bytes | None:
        return self._kv.get(key)

    def mget(self, keys: list[bytes]) -> list[bytes | None]:
        return [self._kv.get(k) for k in keys]

    def exists(self, key: bytes) -> int:
        return 1 if key in self._kv else 0

    def delete(self, key: bytes) -> int:
        if key in self._kv:
            del self._kv[key]
            return 1
        return 0

    unlink = delete

    def pipeline(self, transaction: bool = True) -> FakePipeline:
        return FakePipeline(self)

    def execute_command(self, command, *arguments):
        """Atomic record CAS, guarded deletion and checkpoint protection scripts."""
        with self._lock:
            if command == 'SCARD':
                return len(self._dump_protections.get(arguments[0], ()))
            if command == 'UNLINK':
                return sum(self.delete(key) for key in arguments)
            if command == 'SREM':
                key, owner = arguments
                owners = self._dump_protections.get(key)
                if owners is None or owner not in owners:
                    return 0
                owners.remove(owner)
                if not owners:
                    del self._dump_protections[key]
                return 1
            assert command == 'EVAL'
            script, key_count, key, *arguments = arguments
            if script.startswith('-- nexus:dump-protect:1'):
                protection, owner = arguments
                if not self.exists(key):
                    return 0
                self._dump_protections.setdefault(protection, set()).add(owner)
                return 1
            if script.startswith('-- nexus:dump-release:1'):
                deferred, owner = arguments
                owners = self._dump_protections.get(key)
                if owners is not None:
                    owners.discard(owner)
                    if not owners:
                        del self._dump_protections[key]
                return int(not self._dump_protections.get(key) and self.exists(deferred))
            if key_count == 3:
                protection, deferred, expected = arguments
                current = self.get(key)
                if current is None:
                    return 0
                matches = (
                    hashlib.sha1(current).hexdigest() == expected
                    if 'nexus:load-delete' in script
                    else current == expected
                )
                if not matches:
                    return -1
                if self._dump_protections.get(protection):
                    self._kv[deferred] = self._kv.pop(key)
                    return 2
                return self.delete(key)
            assert key_count == 1
            expected, replacement = arguments
            current = self.get(key)
            if current is None:
                return -1
            if current != expected:
                return 0
            self.set(key, replacement)
            return 1

    def parse_response(self, connection, command_name):
        return self.execute_command(*connection.send_command.call_args.args)

    def close(self) -> None:
        return None


class FakePipeline:
    def __init__(self, redis: FakeRedis) -> None:
        self._r = redis
        self._ops: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def set(self, key: bytes, value: bytes, nx: bool = False) -> FakePipeline:
        self._ops.append(('set', (key, value), {'nx': nx}))
        return self

    def delete(self, key):
        self._ops.append(('delete', (key,), {}))
        return self

    def unlink(self, key):
        self._ops.append(('unlink', (key,), {}))
        return self

    def execute_command(self, *args):
        self._ops.append(('execute_command', args, {}))
        return self

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self._ops.clear()

    def execute(self, raise_on_error: bool = True) -> list[Any]:
        out: list[Any] = []
        for name, args, kwargs in self._ops:
            try:
                out.append(getattr(self._r, name)(*args, **kwargs))
            except Exception as exc:
                if raise_on_error:
                    raise
                out.append(exc)
        self._ops.clear()
        return out
