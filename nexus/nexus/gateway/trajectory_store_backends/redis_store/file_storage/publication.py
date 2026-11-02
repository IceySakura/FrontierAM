"""Per-record Redis reservations and immutable external blob publications."""

from __future__ import annotations

import threading

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import contextmanager, suppress
from contextvars import copy_context
from dataclasses import dataclass, field, replace
from functools import partial
from typing import TYPE_CHECKING, cast
from uuid import uuid4

from .blob_codec import decode_blob, encode_blob
from .types import LARGE_FIELDS, BlobClient
from ..client import _allocate_node_ids, _decode_payload
from ..dump_protection import deferred_record_key, dump_owners_key, get_deferred_records
from ..types import CompareAndSetResult, RedisClientError, Serializable


if TYPE_CHECKING:
    from .client import FSClient
    from ..client import RedisClient


# An explicit envelope keeps caller-selected codecs away from FS metadata.
_ENVELOPE_PREFIX = b'\x00nexus:redis-fs:1\x00'


@dataclass
class StoredNode(Serializable):
    """The record bytes use the payload's own codec, including custom codecs."""

    record: bytes
    blob_urls: dict[str, str]
    version: int = 1
    state: str = 'committed'
    write_id: str | None = None

    def serialize(self) -> bytes:
        return _ENVELOPE_PREFIX + super().serialize()

    @staticmethod
    def deserialize(blob: bytes) -> StoredNode:
        if not blob.startswith(_ENVELOPE_PREFIX):
            raise ValueError('expected a Redis FS envelope')
        value = Serializable.deserialize(blob[len(_ENVELOPE_PREFIX) :])
        if not isinstance(value, StoredNode) or value.version != 1 or value.state not in ('writing', 'committed'):
            raise ValueError('unsupported Redis FS record format')
        return value


@dataclass
class PublicationResult:
    node_id: str
    blob_urls: dict[str, str] = field(default_factory=dict)
    error: str | None = None


@dataclass
class _PreparedWrite:
    pending: bytes
    committed: bytes | None
    blobs: dict[str, bytes | bytearray]
    result: PublicationResult


_DELETE_IF_UNCHANGED = """
local current = redis.call('GET', KEYS[1])
if not current then return 0 end
if current ~= ARGV[1] then return -1 end
if redis.call('SCARD', KEYS[2]) > 0 then
    redis.call('RENAME', KEYS[1], KEYS[3])
    return 2
end
return redis.call('UNLINK', KEYS[1])
"""


def publication_errors(results: list[PublicationResult]) -> dict[int, str]:
    return {index: result.error for index, result in enumerate(results) if result.error is not None}


class RedisPublication:
    """FS composition owned by one RedisClient; injected blob clients are borrowed."""

    def __init__(self, client: RedisClient, fs: BlobClient, *, owns_fs: bool, io_workers: int):
        self._client = client
        self._fs = fs
        self._owns_fs = owns_fs
        set_retry = getattr(client._r, 'set_retry', None)
        if callable(set_retry):
            from redis.backoff import NoBackoff
            from redis.retry import Retry

            set_retry(Retry(NoBackoff(), 0))
        self._io_workers = io_workers
        self._io = ThreadPoolExecutor(max_workers=io_workers, thread_name_prefix='nexus-redis-fs')
        self._lifecycle = threading.Condition()
        self._active_operations = 0
        self._closing = False
        self._closed = False

    @contextmanager
    def operation(self):
        with self._lifecycle:
            if self._closing:
                raise RedisClientError('Redis FS client is closing or closed', code='closed')
            self._active_operations += 1
        try:
            yield
        finally:
            with self._lifecycle:
                self._active_operations -= 1
                self._lifecycle.notify_all()

    def _run_parallel(self, calls, *, return_exceptions=False):
        """Drain all admitted calls, including when later submissions fail.

        Any finished call frees its slot immediately; results keep input order.
        """
        calls = list(calls)
        results = [None] * len(calls)
        pending: dict[Future, int] = {}
        next_index = 0
        try:
            while next_index < len(calls) or pending:
                while next_index < len(calls) and len(pending) < self._io_workers:
                    try:
                        future = self._io.submit(copy_context().run, calls[next_index])
                    except Exception as exc:
                        results[next_index:] = [exc] * (len(calls) - next_index)
                        next_index = len(calls)
                        break
                    pending[future] = next_index
                    next_index += 1
                if pending:
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        index = pending.pop(future)
                        try:
                            results[index] = future.result()
                        except Exception as exc:
                            results[index] = exc
        finally:
            for future in pending:
                # A BaseException must not release connections with I/O in flight.
                with suppress(BaseException):
                    future.result()
        if not return_exceptions:
            for result in results:
                if isinstance(result, Exception):
                    raise result
        return results

    def _prepare_write(self, record, node_id):
        from ...redis import NodeRecord

        if not isinstance(record, Serializable):
            raise TypeError('payloads must be Serializable')
        urls, blobs = {}, {}
        if isinstance(record, NodeRecord):
            for name in LARGE_FIELDS:
                value = getattr(record, name)
                if value is not None:
                    url = self._fs.allocate_url()
                    urls[name] = url
                    blobs[url] = encode_blob(value)
        result = PublicationResult(node_id, urls)
        if not urls:
            return _PreparedWrite(record.serialize(), None, blobs, result)
        encoded = replace(record, **dict.fromkeys(urls)).serialize()
        envelope = StoredNode(encoded, urls, state='writing', write_id=uuid4().hex)
        return _PreparedWrite(envelope.serialize(), replace(envelope, state='committed').serialize(), blobs, result)

    @staticmethod
    def _serialize_import_metadata(record, blob_urls):
        """Describe a restore whose encoded blobs are imported directly by the DU."""
        envelope = StoredNode(record.serialize(), dict(blob_urls), state='writing', write_id=uuid4().hex)
        return envelope.serialize(), replace(envelope, state='committed').serialize()

    def _reserve(self, prepared, overwrite):
        client = self._client
        is_cluster = callable(getattr(client._r, 'mget_nonatomic', None))
        pipe = client._r.pipeline(transaction=False) if is_cluster else client._r.pipeline()
        for plan in prepared:
            pipe.set(client._node_key(plan.result.node_id), plan.pending, nx=not overwrite)
        try:
            outcomes = self.execute_pipeline(pipe, raise_on_error=False)
        except Exception as exc:
            # A connection failure cannot establish which reservations reached Redis.
            outcomes = [exc] * len(prepared)
        results = []
        for plan, outcome in zip(prepared, outcomes, strict=True):
            error = None
            if isinstance(outcome, Exception):
                error = f'{type(outcome).__name__}: {outcome}'
            elif not overwrite and not outcome:
                error = f'payload already exists for node_id={plan.result.node_id}'
            results.append(replace(plan.result, error=error))
        return results

    def execute_pipeline(self, pipe, *, raise_on_error=True):
        if not callable(getattr(self._client._r, 'mget_nonatomic', None)):
            return pipe.execute(raise_on_error=raise_on_error)
        try:
            # redis-py 8.x replays connection failures together with redirections,
            # even with retry=0. Use its routing queue with that replay disabled.
            return pipe.send_cluster_commands(
                pipe._execution_strategy.command_queue,
                raise_on_error=raise_on_error,
                allow_redirections=False,
            )
        finally:
            pipe.reset()

    def _commit_write(self, prepared):
        result = self._client.compare_and_set_many([(prepared.result.node_id, prepared.pending, prepared.committed)])[0]
        if result is not CompareAndSetResult.UPDATED:
            raise RedisClientError('publication reservation changed', code='conflict')
        return prepared.result

    def put(self, payloads, *, node_ids=None, overwrite=None):
        from ...redis import NodeRecord

        with self.operation():
            payloads = list(payloads)
            if not payloads:
                return []
            ids = _allocate_node_ids(len(payloads), node_ids)
            offload = any(
                isinstance(record, NodeRecord) and any(getattr(record, name) is not None for name in LARGE_FIELDS)
                for record in payloads
            )
            immutable = node_ids is not None or offload
            if immutable and overwrite is True:
                raise ValueError('FS publications are immutable; allocate a new storage ID')
            prepared = [self._prepare_write(record, node_id) for record, node_id in zip(payloads, ids, strict=True)]
            results = self._reserve(prepared, not immutable and (overwrite is None or bool(overwrite)))
            uploads, owners, eligible = [], [], []
            for index, (result, plan) in enumerate(zip(results, prepared, strict=True)):
                if result.error is None and plan.committed is not None:
                    eligible.append(index)
                    for url, blob in plan.blobs.items():
                        owners.append(index)
                        uploads.append(partial(self._fs.put, url, blob))
            failures = {}
            for index, outcome in zip(owners, self._run_parallel(uploads, return_exceptions=True), strict=True):
                if isinstance(outcome, Exception):
                    failures.setdefault(index, outcome)
            committing = [index for index in eligible if index not in failures]
            outcomes = self._run_parallel(
                [partial(self._commit_write, prepared[index]) for index in committing], return_exceptions=True
            )
            for index, outcome in zip(committing, outcomes, strict=True):
                if isinstance(outcome, Exception):
                    failures[index] = outcome
                else:
                    results[index] = outcome
            for index, error in failures.items():
                # Keep uncertain reservations/blobs; never guess a rollback.
                results[index] = replace(
                    results[index],
                    error=f'{type(error).__name__}: {error}; inspect publication {self._client._node_key(ids[index])!r}',
                )
            return results

    @staticmethod
    def _envelope(record_bytes):
        if not record_bytes.startswith(_ENVELOPE_PREFIX):
            return None
        envelope = StoredNode.deserialize(record_bytes)
        if envelope.state == 'writing':
            raise RedisClientError('record publication is still in progress', code='write_in_progress')
        return envelope

    def get(self, requests, *, fields=None, reference_urls=None):
        with self.operation():
            if not requests:
                return []
            selected_fields = None if fields is None else set(fields)
            locators = [None] * len(requests) if reference_urls is None else reference_urls
            calls = [partial(self._client.get_node_blobs, [node_id for node_id, _ in requests])]
            reads = {}
            for index, urls in enumerate(locators):
                for name, url in (urls or {}).items():
                    if selected_fields is None or name in selected_fields:
                        reads[index, name] = len(calls)
                        calls.append(partial(self._fs.get, url))
            results = self._run_parallel(calls, return_exceptions=True) if reads else [calls[0]()]
            record_bytes_list = results[0]
            if isinstance(record_bytes_list, Exception):
                raise record_bytes_list
            out = []
            for index, ((node_id, payload_type), record_bytes, urls) in enumerate(
                zip(requests, record_bytes_list, locators, strict=True)
            ):
                if record_bytes is None:
                    raise RedisClientError(f'payload not found for node_id={node_id}', code='not_found')
                envelope = self._envelope(record_bytes)
                stored_blob_urls = envelope.blob_urls if envelope is not None else {}
                if urls is not None and urls != stored_blob_urls:
                    raise RedisClientError('reference URLs do not match stored URLs', code='invalid_reference')
                payload = _decode_payload(node_id, payload_type, record_bytes if envelope is None else envelope.record)
                restored = {}
                for name, url in stored_blob_urls.items():
                    if selected_fields is not None and name not in selected_fields:
                        continue
                    read = reads.get((index, name))
                    value = results[read] if read is not None else self._fs.get(url)
                    if isinstance(value, RedisClientError) and value.code == 'not_found':
                        # Speculative blob reads may precede the publication CAS.
                        value = self._fs.get(url)
                    if isinstance(value, Exception):
                        raise value
                    restored[name] = decode_blob(value)
                out.append(replace(payload, **restored) if envelope is not None else payload)
            return out

    def _read_delete_snapshot(self, node_ids, missing_ok, reference_urls):
        record_bytes_list = self._client.get_node_blobs(node_ids)
        locators = [None] * len(node_ids) if reference_urls is None else reference_urls
        expected_records, blob_urls_by_record, missing_record_ids = {}, {}, set()
        for node_id, record_bytes, locator in zip(node_ids, record_bytes_list, locators, strict=True):
            if record_bytes is None:
                if not missing_ok:
                    raise RedisClientError('record not found', code='not_found')
                missing_record_ids.add(node_id)
                continue
            envelope = self._envelope(record_bytes)
            stored_blob_urls = envelope.blob_urls if envelope is not None else {}
            if locator is not None and locator != stored_blob_urls:
                raise RedisClientError('reference URLs do not match stored URLs', code='invalid_reference')
            expected_records[node_id] = record_bytes
            blob_urls_by_record[node_id] = tuple(stored_blob_urls.values())
        deferred_records = (
            dict(zip(missing_record_ids, get_deferred_records(self._client, list(missing_record_ids)), strict=True))
            if missing_record_ids
            else {}
        )
        deferred_records = {
            node_id: record_bytes for node_id, record_bytes in deferred_records.items() if record_bytes is not None
        }
        for node_id, locator in zip(node_ids, locators, strict=True):
            record_bytes = deferred_records.get(node_id)
            if record_bytes is None:
                continue
            envelope = StoredNode.deserialize(record_bytes) if record_bytes.startswith(_ENVELOPE_PREFIX) else None
            stored_blob_urls = envelope.blob_urls if envelope is not None else {}
            if locator is not None and locator != stored_blob_urls:
                raise RedisClientError('reference URLs do not match stored URLs', code='invalid_reference')
        return expected_records, blob_urls_by_record, deferred_records

    def _delete_if_unchanged(self, expected_records):
        if not expected_records:
            return {}
        with self._client._r.pipeline(transaction=False) as pipe:
            for node_id, record_bytes in expected_records.items():
                key = self._client._node_key(node_id)
                pipe.execute_command(
                    'EVAL', _DELETE_IF_UNCHANGED, 3, key, dump_owners_key(key), deferred_record_key(key), record_bytes
                )
            try:
                delete_results = self.execute_pipeline(pipe, raise_on_error=False)
            except Exception as exc:
                # Do not replay deletes whose acknowledgement is unknown.
                delete_results = [exc] * len(expected_records)
        return dict(zip(expected_records, delete_results, strict=True))

    def _reclaim_deferred_record(self, record_id, record_bytes, blob_urls):
        if blob_urls is None:
            if record_bytes is None:
                return
            envelope = StoredNode.deserialize(record_bytes) if record_bytes.startswith(_ENVELOPE_PREFIX) else None
            blob_urls = list(envelope.blob_urls.values()) if envelope is not None else []
            if envelope is not None and envelope.state == 'writing' and blob_urls:
                states = cast('FSClient', self._fs).finalize_ssd_imports(blob_urls, [])['items']
                unconfirmed_imports = {
                    url: states.get(url) for url in blob_urls if states.get(url, {}).get('state') != 'finished'
                }
                if unconfirmed_imports:
                    raise RedisClientError(
                        f'deferred import cleanup not confirmed for record_id={record_id}: {unconfirmed_imports}',
                        code='cleanup_failed',
                    )
        for url in blob_urls:
            self._fs.delete(url)
        self._client._execute_without_retry('UNLINK', deferred_record_key(self._client._node_key(record_id)))

    def _reclaim_deferred_records(self, record_ids, *, cleanup_urls_by_record=None, deferred_records=None):
        """Reclaim only records whose last dump owner has already acknowledged release."""
        cleanup_urls_by_record = cleanup_urls_by_record or {}
        if deferred_records is None:
            missing_record_ids = [record_id for record_id in record_ids if record_id not in cleanup_urls_by_record]
            deferred_records = (
                dict(zip(missing_record_ids, get_deferred_records(self._client, missing_record_ids), strict=True))
                if missing_record_ids
                else {}
            )
        calls = [
            partial(
                self._reclaim_deferred_record,
                record_id,
                deferred_records.get(record_id),
                cleanup_urls_by_record.get(record_id),
            )
            for record_id in record_ids
        ]
        return dict(zip(record_ids, self._run_parallel(calls, return_exceptions=True), strict=True))

    def delete(self, node_ids, *, missing_ok=True, reference_urls=None):
        with self.operation():
            if not node_ids:
                return
            expected_records, blob_urls_by_record, deferred_records = self._read_delete_snapshot(
                node_ids, missing_ok, reference_urls
            )
            delete_results = self._delete_if_unchanged(expected_records)
            # A Redis ACK must precede any DU delete. Keep URLs for conflicts and
            # unknown results, including when other records in the batch succeed.
            blob_urls_to_delete = {
                url for node_id, result in delete_results.items() if result == 1 for url in blob_urls_by_record[node_id]
            }
            cleanup_results = self._run_parallel(
                [partial(self._fs.delete, url) for url in blob_urls_to_delete], return_exceptions=True
            )
            if deferred_records:
                with self._client._r.pipeline(transaction=False) as pipe:
                    for node_id in deferred_records:
                        pipe.execute_command('SCARD', dump_owners_key(self._client._node_key(node_id)))
                    try:
                        owner_counts = self.execute_pipeline(pipe, raise_on_error=False)
                    except Exception as exc:
                        owner_counts = [exc] * len(deferred_records)
                reclaimable_record_ids = [
                    node_id for node_id, count in zip(deferred_records, owner_counts, strict=True) if count == 0
                ]
                cleanup_results.extend(
                    self._reclaim_deferred_records(reclaimable_record_ids, deferred_records=deferred_records).values()
                )
                cleanup_results.extend(count for count in owner_counts if isinstance(count, Exception))
            for result in delete_results.values():
                if isinstance(result, Exception):
                    raise result
                if result == -1:
                    raise RedisClientError('record changed after deletion validation', code='conflict')
                if result == 0 and not missing_ok:
                    raise RedisClientError('record already deleted', code='not_found')
            for result in cleanup_results:
                if isinstance(result, Exception):
                    raise result
                if result is False and not missing_ok:
                    raise RedisClientError('blob already missing', code='not_found')

    def close(self):
        with self._lifecycle:
            if self._closing:
                while not self._closed:
                    self._lifecycle.wait()
                return
            self._closing = True
            while self._active_operations:
                self._lifecycle.wait()
        try:
            self._io.shutdown(wait=True)
        finally:
            try:
                if self._owns_fs:
                    self._fs.close()
            finally:
                try:
                    self._client._close_redis()
                finally:
                    with self._lifecycle:
                        self._closed = True
                        self._lifecycle.notify_all()
