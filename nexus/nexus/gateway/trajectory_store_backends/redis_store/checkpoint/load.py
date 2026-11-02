"""Online checkpoint restore, with per-record acknowledgements and owned cleanup."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import pickle
import threading

from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq

from redis.exceptions import RedisError

from .format import ARRAY_DTYPES, PARQUET_BATCH_SIZE, build_schema, decode_array, parse_blob_reference
from ..client import _RECORD_CAS
from ..dump_protection import deferred_record_key, dump_owners_key
from ..file_storage.types import LARGE_FIELDS
from ..types import CompareAndSetResult, RedisClientError, _is_transport_error


logger = logging.getLogger(__name__)
_LOAD_EXCEPTIONS = (
    RedisClientError,
    RedisError,
    OSError,
    ValueError,
    TypeError,
    pickle.PickleError,
    pa.ArrowException,
)
_DELETE_IF_OWNED = """
-- nexus:load-delete:1
local current = redis.call('GET', KEYS[1])
if not current then return 0 end
if redis.sha1hex(current) ~= ARGV[1] then return -1 end
if redis.call('SCARD', KEYS[2]) > 0 then
    redis.call('RENAME', KEYS[1], KEYS[3])
    return 2
end
return redis.call('UNLINK', KEYS[1])
"""


@dataclass
class _LoadEntry:
    entry_id: int
    kind: str
    fields: tuple[str, ...] | None
    target_record_id: str = field(default_factory=lambda: uuid4().hex)
    source_paths: dict[str, str] = field(default_factory=dict)
    target_urls: dict[str, str] = field(default_factory=dict)
    ack_sha1: str | None = None
    restored_ref: object | None = None
    error: str | None = None
    outcome_unknown: bool = False
    phase: str = 'preflight'
    redis_deleted: bool = False
    cleanup_deferred: bool = False


def _validate_checkpoint(dump_refs, dump_dir):
    """Read only identity, array metadata and paths before allocating target resources."""
    from ...redis import _RedisDumpRef

    entries = {}
    for ref in dump_refs:
        if not isinstance(ref, _RedisDumpRef):
            raise TypeError('Redis load requires Redis checkpoint references')
        if not isinstance(ref.dump_id, str) or not ref.dump_id:
            raise ValueError('invalid checkpoint dump_id')
        if ref.kind not in ('node', 'segment') or not isinstance(ref.entry_id, int) or ref.entry_id < 0:
            raise ValueError('invalid checkpoint entry identity')
        if ref.fields is not None and (
            not isinstance(ref.fields, tuple) or any(not isinstance(name, str) for name in ref.fields)
        ):
            raise ValueError('invalid checkpoint fields')
        if ref.kind == 'node' and ref.fields is not None and 'token_ids' not in ref.fields:
            raise ValueError('Node checkpoint fields must include token_ids')
        previous = entries.setdefault(ref.entry_id, _LoadEntry(ref.entry_id, ref.kind, ref.fields))
        if (previous.kind, previous.fields) != (ref.kind, ref.fields):
            raise ValueError('duplicate checkpoint entry has inconsistent metadata')
    parquet = pq.ParquetFile(dump_dir / 'trajectory.parquet')
    if not parquet.schema_arrow.equals(build_schema(''), check_metadata=False):
        raise ValueError('unsupported trajectory checkpoint schema')
    metadata = parquet.schema_arrow.metadata or {}
    if metadata.get(b'nexus_format') != b'trajectory_parquet_v1':
        raise ValueError('unsupported trajectory checkpoint format')
    if any(ref.dump_id.encode() != metadata.get(b'dump_id') for ref in dump_refs):
        raise ValueError('checkpoint dump_id does not match references')
    seen = set()
    columns = ['entry_id', 'kind', 'token_ids.shape', *LARGE_FIELDS]
    for arrow_batch in parquet.iter_batches(batch_size=PARQUET_BATCH_SIZE, columns=columns):
        for row in arrow_batch.to_pylist():
            entry = entries.get(row['entry_id'])
            if entry is None:
                continue
            if entry.entry_id in seen or row['kind'] != entry.kind:
                raise ValueError('checkpoint entry kind or identity does not match references')
            seen.add(entry.entry_id)
            if entry.kind == 'node':
                if row.get('token_ids') is None or row['token_ids'].get('shape') is None:
                    raise ValueError('Node checkpoint is missing token_ids')
                for name in LARGE_FIELDS:
                    value = row.get(name)
                    if value is None or entry.fields is not None and name not in entry.fields:
                        continue
                    path = parse_blob_reference(value)
                    resolved = (dump_dir / path).resolve(strict=True)
                    if not resolved.is_relative_to(dump_dir) or not resolved.is_file():
                        raise ValueError(f'invalid checkpoint blob path: {value!r}')
                    entry.source_paths[name] = str(resolved)
    if seen != entries.keys():
        raise ValueError('checkpoint does not contain every requested entry')
    return parquet, entries


def _build_target_record(row, entry, fs_client):
    from ...redis import NodeRecord, SegmentRecord

    if entry.kind == 'segment':
        fields = {
            name: value for name, value in row['segment_fields'] or () if entry.fields is None or name in entry.fields
        }
        return SegmentRecord(row['trajectory_id'], row['segment_id'], fields)
    values = {
        name: decode_array(row[name], name, summaries=False)
        for name in ARRAY_DTYPES
        if (entry.fields is None or name in entry.fields) and row[name].is_valid
    }
    entry.target_urls = {name: fs_client.allocate_url() for name in entry.source_paths}
    return NodeRecord(row['trajectory_id'], row['node_id'], **values)


class _CheckpointLoader:
    def __init__(self, store, dump_refs, dump_dir):
        self.store, self.dump_refs, self.dump_dir = store, dump_refs, dump_dir
        self.redis_client = store._client
        self.publication = self.redis_client._publication
        self.fs_client = self.publication._fs
        self.load_id = uuid4().hex
        self.entries = {}
        self.state_lock = threading.Lock()
        self.cancel_requested = threading.Event()
        self.finished = False

    def request_cancel(self):
        with self.state_lock:
            if self.finished:
                return False
            self.cancel_requested.set()
            return True

    def try_complete(self):
        with self.state_lock:
            if self.cancel_requested.is_set():
                return False
            self.finished = True
            return True

    def build_result_body(self):
        partial_refs, errors, unknown_entries = [], {}, {}
        for index, ref in enumerate(self.dump_refs):
            entry_id = getattr(ref, 'entry_id', None)
            entry = self.entries.get(entry_id) if isinstance(entry_id, int) else None
            partial_refs.append(
                replace(entry.restored_ref) if entry is not None and entry.restored_ref is not None else None
            )
            if entry is None or entry.restored_ref is None:
                errors[index] = 'not attempted' if entry is None or entry.error is None else entry.error
            if entry is not None and entry.outcome_unknown:
                unknown_entries[index] = {
                    'entry_id': entry.entry_id,
                    'record_id': entry.target_record_id,
                    'blob_urls': dict(entry.target_urls),
                    'phase': entry.phase,
                    'error': entry.error,
                }
        return {'load_id': self.load_id, 'partial_refs': partial_refs, 'errors': errors, 'unknown': unknown_entries}

    def set_entry_error(self, entry, error, *, outcome_unknown=False):
        entry.error = f'{type(error).__name__}: {error}' if isinstance(error, BaseException) else str(error)
        entry.outcome_unknown = outcome_unknown

    def record_redis_ack(self, entry, record_bytes, *, complete=False):
        from ...redis import HybridNodeRef, NodeRef, SegmentRef

        entry.ack_sha1 = hashlib.sha1(record_bytes).hexdigest()
        if complete:
            cls = SegmentRef if entry.kind == 'segment' else HybridNodeRef if entry.target_urls else NodeRef
            kwargs = {name + '_url': url for name, url in entry.target_urls.items()}
            entry.restored_ref = cls(self.store.namespace, entry.target_record_id, **kwargs)

    def finalize_imports(self, entries):
        entries = [entry for entry in entries if not entry.cleanup_deferred]
        target_urls = [url for entry in entries for url in entry.target_urls.values()]
        if not target_urls:
            return
        keep_urls = {
            url
            for entry in entries
            if not entry.redis_deleted and (entry.restored_ref is not None or entry.outcome_unknown)
            for url in entry.target_urls.values()
        }
        try:
            states = self.fs_client.finalize_ssd_imports(target_urls, keep_urls)['items']
        except _LOAD_EXCEPTIONS as exc:
            states = {url: {'state': 'unknown', 'error': str(exc)} for url in target_urls}
        for entry in entries:
            unconfirmed_imports = {
                url: states.get(url, {})
                for url in entry.target_urls.values()
                if states.get(url, {}).get('state') != ('kept' if url in keep_urls else 'finished')
            }
            if unconfirmed_imports:
                # A failed keep never invalidates an acknowledged Redis publication.
                if entry.restored_ref is not None and not entry.redis_deleted:
                    logger.warning('Load %s retained import cleanup metadata: %s', self.load_id, unconfirmed_imports)
                else:
                    self.set_entry_error(
                        entry, f'import cleanup not confirmed: {unconfirmed_imports}', outcome_unknown=True
                    )

    def cleanup_entries(self, entries, *, cancel=False):
        """Delete Redis ownership first; only then permit removal of SSD files."""
        for entry in entries:
            if (
                entry.outcome_unknown
                or entry.redis_deleted
                or entry.ack_sha1 is None
                or entry.restored_ref is not None
                and not cancel
            ):
                continue
            entry.phase = 'cleanup'
            try:
                key = self.redis_client._node_key(entry.target_record_id)
                result = self.redis_client._execute_without_retry(
                    'EVAL', _DELETE_IF_OWNED, 3, key, dump_owners_key(key), deferred_record_key(key), entry.ack_sha1
                )
                if result not in (0, 1, 2):
                    self.set_entry_error(entry, 'Redis record changed before cleanup', outcome_unknown=True)
                else:
                    entry.restored_ref = None
                    entry.redis_deleted = True
                    entry.cleanup_deferred = result == 2
            except _LOAD_EXCEPTIONS as exc:
                self.set_entry_error(entry, exc, outcome_unknown=True)
        self.finalize_imports(entries)

    def restore_batch(self, rows):
        entries = [self.entries[row['entry_id']] for row in rows]
        reservations = {}
        for entry, row in zip(entries, rows, strict=True):
            if self.cancel_requested.is_set():
                break
            try:
                record = _build_target_record(row, entry, self.fs_client)
                entry.phase = 'reserve'
                if entry.target_urls:
                    pending_bytes, committed_bytes = self.publication._serialize_import_metadata(
                        record, entry.target_urls
                    )
                    result = self.redis_client._execute_without_retry(
                        'SET', self.redis_client._node_key(entry.target_record_id), pending_bytes, 'NX'
                    )
                    if not result:
                        self.set_entry_error(entry, f'payload already exists for node_id={entry.target_record_id}')
                        continue
                    self.record_redis_ack(entry, pending_bytes)
                    reservations[entry.entry_id] = (pending_bytes, committed_bytes)
                else:
                    record_bytes = record.serialize()
                    outcome = self.redis_client._execute_without_retry(
                        'SET', self.redis_client._node_key(entry.target_record_id), record_bytes, 'NX'
                    )
                    if not outcome:
                        raise RedisClientError('fresh load record already exists', code='conflict')
                    self.record_redis_ack(entry, record_bytes, complete=True)
            except _LOAD_EXCEPTIONS as exc:
                self.set_entry_error(entry, exc, outcome_unknown=_is_transport_error(exc))
                if entry.outcome_unknown:
                    break
        copy_calls, copy_entries = [], []
        if not self.cancel_requested.is_set():
            for entry in entries:
                if entry.entry_id in reservations:
                    entry.phase = 'copy'
                    for name, url in entry.target_urls.items():
                        copy_calls.append(partial(self.fs_client.import_file_to_ssd, url, entry.source_paths[name]))
                        copy_entries.append(entry)
            copy_results = self.publication._run_parallel(copy_calls, return_exceptions=True)
            for entry, result in zip(copy_entries, copy_results, strict=True):
                if isinstance(result, BaseException):
                    # Redis is still definitely a writing reservation; finish can
                    # drain or tombstone an uncertain import before dropping it.
                    entry.error = f'{type(result).__name__}: {result}'
            for entry in entries:
                reservation = reservations.get(entry.entry_id)
                if reservation is None or entry.error is not None or self.cancel_requested.is_set():
                    continue
                pending_bytes, committed_bytes = reservation
                entry.phase = 'commit'
                try:
                    result = CompareAndSetResult(
                        self.redis_client._execute_without_retry(
                            'EVAL',
                            _RECORD_CAS,
                            1,
                            self.redis_client._node_key(entry.target_record_id),
                            pending_bytes,
                            committed_bytes,
                        )
                    )
                    if result is not CompareAndSetResult.UPDATED:
                        raise RedisClientError('publication reservation changed', code='conflict')
                    self.record_redis_ack(entry, committed_bytes, complete=True)
                except _LOAD_EXCEPTIONS as exc:
                    self.set_entry_error(entry, exc, outcome_unknown=_is_transport_error(exc))
                    if entry.outcome_unknown:
                        break
        self.cleanup_entries(entries, cancel=self.cancel_requested.is_set())

    def run(self):
        try:
            with self.publication.operation():
                parquet, self.entries = _validate_checkpoint(self.dump_refs, self.dump_dir)
                selected_fields = (
                    None
                    if any(entry.fields is None for entry in self.entries.values())
                    else {name for entry in self.entries.values() for name in entry.fields}
                )
                columns = ['entry_id', 'kind', 'trajectory_id', 'node_id', 'segment_id', 'segment_fields']
                columns += [name for name in ARRAY_DTYPES if selected_fields is None or name in selected_fields]
                try:
                    for arrow_batch in parquet.iter_batches(batch_size=PARQUET_BATCH_SIZE, columns=columns):
                        if self.cancel_requested.is_set() or any(
                            entry.outcome_unknown for entry in self.entries.values()
                        ):
                            break
                        rows = []
                        for index in range(arrow_batch.num_rows):
                            entry_id = arrow_batch.column('entry_id')[index].as_py()
                            if entry_id in self.entries:
                                rows.append({
                                    name: arrow_batch.column(name)[index]
                                    if name in ARRAY_DTYPES
                                    else arrow_batch.column(name)[index].as_py()
                                    for name in arrow_batch.schema.names
                                })
                        if rows:
                            self.restore_batch(rows)
                except _LOAD_EXCEPTIONS as exc:
                    # A malformed later numeric batch does not discard earlier ACKs.
                    for entry in self.entries.values():
                        if entry.restored_ref is None and entry.error is None:
                            entry.error = f'{type(exc).__name__}: {exc}'
                if not self.try_complete():
                    # No RPC remains in flight. Roll back all owned ACKs, while retaining
                    # any record whose publication or conditional deletion is unknown.
                    entries = [entry for entry in self.entries.values() if not entry.outcome_unknown]
                    self.cleanup_entries(entries, cancel=True)
                    return self.build_result_body()
                body = self.build_result_body()
                if body['errors']:
                    raise RedisClientError('checkpoint load failed', code='load_failed', body=body)
                return body
        finally:
            # Errors before the first write are terminal too; a late caller
            # cancellation must not replace an already completed result.
            with self.state_lock:
                self.finished = True


async def load_checkpoint(store, dump_refs, dump_dir):
    loader = _CheckpointLoader(store, dump_refs, Path(dump_dir).resolve())
    task = asyncio.create_task(asyncio.to_thread(loader.run))
    completion = asyncio.gather(task, return_exceptions=True)
    cancellation = None
    while not completion.done():
        try:
            await asyncio.shield(completion)
        except asyncio.CancelledError as exc:
            if loader.request_cancel():
                cancellation = cancellation or exc
    if cancellation is not None:
        error = task.exception()
        body = loader.build_result_body()
        if error is not None or body['unknown']:
            asyncio.get_running_loop().call_exception_handler({
                'message': 'Checkpoint load cancellation retained uncertain records',
                'exception': error,
                'load_id': loader.load_id,
                'unknown': body['unknown'],
            })
            logger.error('Load %s cancellation retained uncertain state: %s', loader.load_id, body['unknown'])
        raise cancellation from error
    body = task.result()
    return body['partial_refs']
