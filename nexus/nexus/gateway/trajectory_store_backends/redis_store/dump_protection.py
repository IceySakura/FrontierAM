"""Retain logically deleted metadata until existing dumps have captured DU pins."""

from __future__ import annotations

from binascii import crc_hqx
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from .types import RedisClientError


_REGISTER_DUMP_OWNER = """-- nexus:dump-protect:1
if redis.call('EXISTS', KEYS[1]) == 0 then return 0 end
redis.call('SADD', KEYS[2], ARGV[1])
return 1
"""
_RELEASE_DUMP_OWNER = """-- nexus:dump-release:1
redis.call('SREM', KEYS[1], ARGV[1])
if redis.call('SCARD', KEYS[1]) == 0 then
    return redis.call('EXISTS', KEYS[2])
end
return 0
"""
_REDIS_PIPELINE_BATCH_SIZE = 256


@lru_cache(maxsize=16384)
def _find_slot_tag(slot: int) -> bytes:
    # Malformed braces prevent wrapping the original key in a valid hash tag.
    # Only these unusual prefixes need a one-time search for an equivalent slot.
    index = 0
    while True:
        tag = str(index).encode()
        if crc_hqx(tag, 0) % 16384 == slot:
            return tag
        index += 1


def dump_owners_key(record_key: bytes) -> bytes:
    """Keep each protection key in the original record's Redis Cluster slot."""
    start = record_key.find(b'{')
    end = record_key.find(b'}', start + 1) if start >= 0 else -1
    if end > start + 1:
        tag = record_key[start + 1 : end]
    elif b'{' not in record_key and b'}' not in record_key:
        tag = record_key
    else:
        tag = _find_slot_tag(crc_hqx(record_key, 0) % 16384)
    return b'nexus:dump:{' + tag + b'}:' + record_key


def deferred_record_key(record_key: bytes) -> bytes:
    """Use a separate keyspace while retaining the protection key's hash slot."""
    return b'nexus:dump-deferred:' + dump_owners_key(record_key)[len(b'nexus:dump:') :]


def get_deferred_records(client, record_ids: Sequence[str]) -> list[bytes | None]:
    keys = [deferred_record_key(client._node_key(record_id)) for record_id in record_ids]
    cluster_mget = getattr(client._r, 'mget_nonatomic', None)
    return cluster_mget(keys) if callable(cluster_mget) else client._r.mget(keys)


@dataclass
class DumpProtection:
    _client: Any
    _dump_id: str
    # Only unconfirmed owner releases remain here; deferred metadata owns cleanup retries.
    _pending_release_keys: set[bytes]

    def release_records(self, record_ids: Sequence[str], *, cleanup_urls_by_record=None) -> None:
        """Release only this dump's owners; unknown ACKs remain pending for final cleanup."""
        record_keys = {self._client._node_key(record_id): record_id for record_id in record_ids}
        keys = sorted(record_keys.keys() & self._pending_release_keys)
        error = None
        for start in range(0, len(keys), _REDIS_PIPELINE_BATCH_SIZE):
            batch = keys[start : start + _REDIS_PIPELINE_BATCH_SIZE]
            with self._client._r.pipeline(transaction=False) as pipe:
                for key in batch:
                    pipe.execute_command(
                        'EVAL', _RELEASE_DUMP_OWNER, 2, dump_owners_key(key), deferred_record_key(key), self._dump_id
                    )
                try:
                    release_results = self._client._publication.execute_pipeline(pipe, raise_on_error=False)
                except Exception as exc:
                    release_results = [exc] * len(batch)
            reclaimable_record_ids = []
            for key, outcome in zip(batch, release_results, strict=True):
                if isinstance(outcome, Exception):
                    error = error or outcome
                else:
                    self._pending_release_keys.remove(key)
                    if outcome:
                        reclaimable_record_ids.append(record_keys[key])
            if reclaimable_record_ids:
                try:
                    cleanup_results = self._client._publication._reclaim_deferred_records(
                        reclaimable_record_ids, cleanup_urls_by_record=cleanup_urls_by_record
                    )
                except Exception as exc:
                    error = error or exc
                else:
                    error = error or next(
                        (item for item in cleanup_results.values() if isinstance(item, Exception)), None
                    )
        if error is not None:
            raise error


@contextmanager
def protect_dump_records(client, record_ids: Sequence[str], dump_id: str) -> Iterator[DumpProtection]:
    """Register all source records before reading payloads, without excluding writes."""
    record_ids = tuple(record_ids)
    keys = sorted({client._node_key(record_id) for record_id in record_ids})
    protection = DumpProtection(client, dump_id, set())
    try:
        for start in range(0, len(keys), _REDIS_PIPELINE_BATCH_SIZE):
            batch = keys[start : start + _REDIS_PIPELINE_BATCH_SIZE]
            with client._r.pipeline(transaction=False) as pipe:
                # Any command can have applied even when its reply is lost.
                protection._pending_release_keys.update(batch)
                for key in batch:
                    pipe.execute_command('EVAL', _REGISTER_DUMP_OWNER, 2, key, dump_owners_key(key), dump_id)
                registration_results = client._publication.execute_pipeline(pipe, raise_on_error=False)
            error = None
            for key, outcome in zip(batch, registration_results, strict=True):
                if isinstance(outcome, Exception):
                    error = error or outcome
                elif not outcome:
                    protection._pending_release_keys.remove(key)
                    error = error or RedisClientError(f'payload not found for key={key!r}', code='not_found')
            if error is not None:
                raise error
        yield protection
    finally:
        protection.release_records(record_ids)
