"""Concurrent blob requests, serialized memory admission, and pinned SSD exports."""

from __future__ import annotations

import logging
import shutil
import threading

from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast
from uuid import uuid4

from nexus.gateway.trajectory_store_backends.redis_store.file_storage._bounded_executor import BoundedExecutor
from nexus.gateway.trajectory_store_backends.redis_store.file_storage.network import DealerClient, RouterServer
from nexus.gateway.trajectory_store_backends.redis_store.file_storage.types import Endpoint, Topology
from nexus.gateway.trajectory_store_backends.redis_store.types import DEFAULT_MEMORY_WATERLINE_BYTES, RedisClientError


_EVICTION_BATCH_SIZE = 1000
logger = logging.getLogger(__name__)


def _new_operation_future():
    future = Future()
    future.set_running_or_notify_cancel()
    return future


@dataclass
class _BlobEntry:
    size: int
    memory: bytes | None
    disk_path: Path | None = None
    dump_pin_count: int = 0
    logically_deleted: bool = False


@dataclass
class _URLLock:
    lock: threading.Lock = field(default_factory=threading.Lock)
    user_count: int = 0


@dataclass
class _DumpJob:
    pinned_entries: dict[str, _BlobEntry] = field(default_factory=dict)
    lifecycle_lock: threading.Lock = field(default_factory=threading.Lock)
    state: str = 'prepared'
    first_error: BaseException | None = None
    dump_directory: Path | None = None
    copy_future: Future | None = None
    prepare_complete: bool = True
    prepare_future: Future | None = None
    prepare_urls: frozenset[str] = frozenset()
    pending_file_count: int = 0
    released_pin_urls: set[str] = field(default_factory=set)


@dataclass
class _SSDImportJob:
    source_path: str | None = None
    state: str = 'pending'
    error: str | None = None
    import_future: Future = field(default_factory=_new_operation_future)
    published_entry: _BlobEntry | None = None
    finish_action: Literal['keep', 'abort'] | None = None
    finish_future: Future | None = None
    claimed_finish_future: Future | None = None


class DataUnitService:
    def __init__(
        self,
        data_unit_id: str,
        endpoint: Endpoint = Endpoint('127.0.0.1', 0),
        *,
        memory_waterline_bytes: int = DEFAULT_MEMORY_WATERLINE_BYTES,
        memory_safe_line_bytes: int | None = None,
        ssd_offload_dir: str | Path = '.hykvstore',
        request_workers: int = 8,
        ssd_workers: int = 8,
        copy_workers: int = 8,
    ):
        if type(memory_waterline_bytes) is not int or memory_waterline_bytes <= 0:
            raise ValueError('memory_waterline_bytes must be a positive integer')
        for name, value in (
            ('request_workers', request_workers),
            ('ssd_workers', ssd_workers),
            ('copy_workers', copy_workers),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if memory_safe_line_bytes is None:
            memory_safe_line_bytes = memory_waterline_bytes // 2
        assert type(memory_safe_line_bytes) is int, 'memory_safe_line_bytes must be an integer'
        assert 0 <= memory_safe_line_bytes <= memory_waterline_bytes, (
            'memory_safe_line_bytes must be between zero and the waterline, inclusive'
        )
        self.data_unit_id = data_unit_id
        self._memory_waterline_bytes = memory_waterline_bytes
        self._memory_safe_line_bytes = memory_safe_line_bytes
        self._ssd_offload_dir = Path(ssd_offload_dir)
        self._ssd_dir: Path | None = None
        self._store: dict[str, _BlobEntry] = {}
        self._resident_lru: OrderedDict[str, None] = OrderedDict()
        self._resident_bytes = 0
        self._used_bytes = 0
        self._ssd_bytes = 0
        self._ssd_objects = 0
        self._cached_bytes = 0
        self._cached_objects = 0
        self._metadata_lock = threading.RLock()
        self._url_locks: dict[str, _URLLock] = {}
        # Memory PUTs and refills run serially on this executor.
        self._admission_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='hykv-fs-admission')
        self._ssd_executor = BoundedExecutor(ssd_workers, 'hykv-fs-ssd')
        self._copy_executor = BoundedExecutor(copy_workers, 'hykv-fs-copy')
        self._dump_jobs: dict[str, _DumpJob] = {}
        self._ssd_imports: dict[str, _SSDImportJob] = {}
        self._started = False
        self._topology: Topology | None = None
        self._stop = threading.Event()
        self._heartbeat_thread = None
        self._controller = None
        self._server = RouterServer(
            endpoint,
            self._dispatch_request,
            server_id=data_unit_id,
            control_operations=('stats', 'get_dump_status', 'get_ssd_import_status', 'finalize_ssd_imports'),
            request_workers=request_workers,
        )

    @property
    def endpoint(self) -> Endpoint:
        return self._server.endpoint

    def start(self):
        if self._started or self._stop.is_set():
            raise RuntimeError('data unit cannot be started more than once')
        root = self._ssd_offload_dir.resolve()
        root.mkdir(parents=True, exist_ok=True)
        directory = root / uuid4().hex
        directory.mkdir()
        self._ssd_dir = directory
        try:
            self._server.start()
            self._started = True
        finally:
            if not self._started:
                self.stop()
        return self

    def _write_blob(self, url: str, blob: bytes) -> Path:
        directory = cast(Path, self._ssd_dir)
        path = directory / f'{url.rsplit("/", 1)[1]}.blob'
        temporary = directory / f'{uuid4().hex}.tmp'
        committed = False
        try:
            with temporary.open('xb') as file:
                file.write(blob)
            temporary.replace(path)
            committed = True
        finally:
            if not committed:
                temporary.unlink(missing_ok=True)
        return path

    def _make_memory_room(self, incoming_bytes: int):
        with self._metadata_lock:
            if self._resident_bytes + incoming_bytes < self._memory_waterline_bytes:
                return
        target_resident_bytes = self._memory_safe_line_bytes - incoming_bytes
        while True:
            with self._metadata_lock:
                bytes_to_reclaim = self._resident_bytes - target_resident_bytes
                if bytes_to_reclaim <= 0:
                    return
                candidates = {}
                for url in self._resident_lru:
                    entry = self._store[url]
                    candidates[url] = entry
                    bytes_to_reclaim -= entry.size
                    if bytes_to_reclaim <= 0 or len(candidates) == _EVICTION_BATCH_SIZE:
                        break
            eviction_futures = [
                self._ssd_executor.submit(self._evict_blob_from_memory, url, entry) for url, entry in candidates.items()
            ]
            wait(eviction_futures)
            for future in eviction_futures:
                future.result()

    def _evict_blob_from_memory(self, url, entry):
        with self._url_lock(url):
            with self._metadata_lock:
                if self._store.get(url) is not entry:
                    return
            self._ensure_ssd_copy_locked(url)
            with self._metadata_lock:
                entry.memory = None
                del self._resident_lru[url]
                self._resident_bytes -= entry.size
                self._cached_bytes -= entry.size
                self._cached_objects -= 1

    @contextmanager
    def _url_lock(self, url):
        with self._metadata_lock:
            slot = self._url_locks.setdefault(url, _URLLock())
            slot.user_count += 1
        try:
            with slot.lock:
                yield
        finally:
            with self._metadata_lock:
                slot.user_count -= 1
                if not slot.user_count:
                    del self._url_locks[url]

    def _snapshot_stats(self) -> dict[str, int]:
        with self._metadata_lock:
            return {
                'objects': len(self._store),
                'used_bytes': self._used_bytes,
                'resident_bytes': self._resident_bytes,
                'resident_objects': len(self._resident_lru),
                'ssd_bytes': self._ssd_bytes,
                'ssd_objects': self._ssd_objects,
                'cached_bytes': self._cached_bytes,
                'cached_objects': self._cached_objects,
            }

    def _build_heartbeat_body(self) -> dict:
        return {
            'data_unit_id': self.data_unit_id,
            'endpoint': vars(self.endpoint),
            'stats': self._snapshot_stats(),
        }

    def configure(self, topology: Topology, controller_endpoint: Endpoint, *, heartbeat_interval: float = 2):
        if self._topology is not None:
            raise RuntimeError('data unit membership cannot change during a run')
        if heartbeat_interval <= 0:
            raise ValueError('heartbeat_interval must be positive')
        registration = next((unit for unit in topology.data_units if unit.data_unit_id == self.data_unit_id), None)
        if registration is None or registration.endpoint != self.endpoint:
            raise ValueError('topology does not match the bound data unit endpoint')
        self._controller = DealerClient(controller_endpoint, timeout=3)
        self._controller.request('heartbeat', self._build_heartbeat_body())
        self._topology = topology
        self._heartbeat_thread = threading.Thread(
            target=self._run_heartbeat_loop, args=(heartbeat_interval,), daemon=True
        )
        self._heartbeat_thread.start()

    def _run_heartbeat_loop(self, interval):
        while not self._stop.wait(interval):
            try:
                cast(DealerClient, self._controller).request('heartbeat', self._build_heartbeat_body())
            except Exception:
                # Reporting failure must not erase storage or change slot ownership.
                logger.warning('FS heartbeat failed for %s', self.data_unit_id, exc_info=True)

    def _handle_request_sync(self, operation: str, body: dict, blob: bytes):
        result = self._dispatch_request(operation, body, blob)
        return result.result() if isinstance(result, Future) else result

    def _dispatch_request(self, operation: str, body: dict, blob: bytes):
        if self._topology is None:
            raise RedisClientError('data unit has not been configured', code='not_ready')
        if operation == 'handshake':
            if blob or body.get('protocol_version') != 1:
                raise RedisClientError('unsupported FS protocol', code='protocol_mismatch')
            return {
                'protocol_version': 1,
                'cluster_id': self._topology.cluster_id,
                'data_unit_id': self.data_unit_id,
            }, b''
        if operation == 'stats':
            return self._snapshot_stats(), b''
        if operation == 'prepare_dump_sources':
            return self.prepare_dump_sources(body['dump_id'], body['urls'])
        if operation == 'start_dump_copy':
            return self.start_dump_copy(body['dump_id'], body['dump_dir']), b''
        if operation == 'get_dump_status':
            return self.get_dump_status(body['dump_id']), b''
        if operation == 'finalize_dump':
            return self._ssd_executor.submit(self._finalize_dump_reply, body['dump_id'])
        if operation == 'get_ssd_import_status':
            return self.get_ssd_import_status(body['urls']), b''
        if operation == 'finalize_ssd_imports':
            return self.finalize_ssd_imports(body['urls'], body['keep_urls'])
        url = body['url']
        self._check_url_owner(url)
        if operation == 'put':
            if len(blob) > self._memory_safe_line_bytes:
                return self._ssd_executor.submit(self._put_blob_on_ssd, url, blob)
            return self._admission_executor.submit(self._put_blob_in_memory, url, blob)
        if operation == 'import_file_to_ssd':
            if blob:
                raise ValueError('import_file_to_ssd accepts a shared file path, not a payload')
            return self.import_file_to_ssd(url, body['source_path'])
        if blob:
            raise ValueError('only PUT accepts a payload')
        if operation == 'get':
            return self._get_blob(url)
        if operation == 'ensure_ssd_copy':
            return self._ssd_executor.submit(self._ensure_ssd_copy_reply, url)
        if operation == 'delete':
            return {'deleted': self._delete_blob(url)}, b''
        raise ValueError(f'unknown data operation: {operation}')

    def _check_url_owner(self, url):
        if cast(Topology, self._topology).owner(url).data_unit_id != self.data_unit_id:
            raise RedisClientError('URL belongs to another data unit', code='wrong_owner')

    def _check_url_available(self, url):
        if url in self._store:
            raise RedisClientError('blob URLs are immutable', code='already_exists')

    def _require_blob_entry(self, url):
        entry = self._store.get(url)
        if entry is None:
            raise RedisClientError('blob not found', code='not_found')
        return entry

    def _put_blob_on_ssd(self, url, blob):
        with self._url_lock(url):
            with self._metadata_lock:
                self._check_url_available(url)
            path = self._write_blob(url, blob)
            with self._metadata_lock:
                self._store[url] = _BlobEntry(len(blob), None, path)
                self._used_bytes += len(blob)
                self._ssd_bytes += len(blob)
                self._ssd_objects += 1
        return {}, b''

    def _put_blob_in_memory(self, url, blob):
        with self._url_lock(url), self._metadata_lock:
            self._check_url_available(url)
        self._make_memory_room(len(blob))
        with self._url_lock(url), self._metadata_lock:
            self._check_url_available(url)
            self._store[url] = _BlobEntry(len(blob), blob)
            self._resident_lru[url] = None
            self._resident_bytes += len(blob)
            self._used_bytes += len(blob)
        return {}, b''

    def _get_blob(self, url):
        with self._url_lock(url):
            with self._metadata_lock:
                entry = self._require_blob_entry(url)
                value, path = entry.memory, entry.disk_path
                if value is not None:
                    self._resident_lru.move_to_end(url)
                    return {}, value
            value = cast(Path, path).read_bytes()
            if len(value) != entry.size:
                raise RedisClientError('SSD blob size does not match its metadata', code='server_error')
        if entry.size > self._memory_safe_line_bytes:
            return {}, value
        # Keep this worker until refill finishes to bound cold GETs by request_workers.
        return self._admission_executor.submit(self._cache_blob_in_memory, url, entry, value).result()

    def _cache_blob_in_memory(self, url, entry, value):
        with self._url_lock(url), self._metadata_lock:
            if self._store.get(url) is not entry or entry.memory is not None:
                return {}, value
        self._make_memory_room(entry.size)
        with self._url_lock(url), self._metadata_lock:
            if self._store.get(url) is entry:
                entry.memory = value
                self._resident_lru[url] = None
                self._resident_bytes += entry.size
                self._cached_bytes += entry.size
                self._cached_objects += 1
        return {}, value

    def _ensure_ssd_copy_locked(self, url):
        """Ensure the SSD copy while the caller holds this URL lock."""
        with self._metadata_lock:
            entry = self._require_blob_entry(url)
            if entry.disk_path is not None:
                return entry
            value = cast(bytes, entry.memory)
        path = self._write_blob(url, value)
        with self._metadata_lock:
            entry.disk_path = path
            self._ssd_bytes += entry.size
            self._ssd_objects += 1
            self._cached_bytes += entry.size
            self._cached_objects += 1
        return entry

    def ensure_ssd_copy(self, url):
        """Ensure an SSD replica without changing residency or LRU order."""
        self._check_url_owner(url)
        with self._url_lock(url):
            self._ensure_ssd_copy_locked(url)

    def _ensure_ssd_copy_reply(self, url):
        self.ensure_ssd_copy(url)
        return {}, b''

    def import_file_to_ssd(self, url, source_path):
        if self._stop.is_set():
            raise RuntimeError('data unit is stopping')
        self._check_url_owner(url)
        if not isinstance(source_path, str) or not Path(source_path).is_absolute():
            raise ValueError('source_path must be an absolute shared file path')
        with self._metadata_lock:
            job = self._ssd_imports.get(url)
            if job is not None:
                if job.source_path != source_path or job.finish_action == 'abort':
                    raise RedisClientError('SSD import URL has already been finalized', code='import_conflict')
                return job.import_future
            job = _SSDImportJob(source_path=source_path)
            self._ssd_imports[url] = job
        copying = self._copy_executor.submit(self._run_ssd_import, url, job)
        copying.add_done_callback(lambda done: self._on_ssd_import_done(url, job, done))
        return job.import_future

    def _run_ssd_import(self, url, job):
        with self._url_lock(url):
            with self._metadata_lock:
                if job.finish_action == 'abort':
                    return
                self._check_url_available(url)
                job.state = 'running'
            path = cast(Path, self._ssd_dir) / f'{url.rsplit("/", 1)[1]}.blob'
            published = False
            try:
                self._copy_file_atomically(Path(job.source_path), path)
                size = path.stat().st_size
                with self._metadata_lock:
                    if job.finish_action == 'abort':
                        return
                    entry = _BlobEntry(size, None, path)
                    self._store[url] = entry
                    self._used_bytes += size
                    self._ssd_bytes += size
                    self._ssd_objects += 1
                    job.published_entry = entry
                    published = True
            finally:
                if not published:
                    path.unlink(missing_ok=True)

    def _on_ssd_import_done(self, url, job, copy_future):
        error = copy_future.exception()
        if isinstance(error, OSError):
            error = RedisClientError(f'SSD import failed: {error}', code='io_error')
        with self._metadata_lock:
            job.error = str(error) if error is not None else None
            job.state = 'failed' if error is not None else 'imported'
        if error is not None:
            job.import_future.set_exception(error)
        elif job.published_entry is None:
            job.import_future.set_exception(RedisClientError('SSD import aborted', code='import_aborted'))
        else:
            job.import_future.set_result(({}, b''))
        self._apply_import_finish_action(url, job)

    def get_ssd_import_status(self, urls):
        with self._metadata_lock:
            items = {}
            for url in urls:
                job = self._ssd_imports.get(url)
                items[url] = (
                    {'state': job.state, 'error': job.error} if job is not None else {'state': 'unknown', 'error': None}
                )
        return {'items': items}

    def finalize_ssd_imports(self, urls, keep_urls):
        urls, keep_urls = list(dict.fromkeys(urls)), set(keep_urls)
        if not keep_urls.issubset(urls):
            raise ValueError('keep_urls must be a subset of urls')
        finish_futures, imports_to_finalize, aborted_import_futures = [], [], []
        with self._metadata_lock:
            for url in urls:
                self._check_url_owner(url)
                job = self._ssd_imports.get(url)
                if url in keep_urls and job is not None and job.finish_action == 'abort':
                    raise RedisClientError('an aborted SSD import cannot be kept', code='import_conflict')
            for url in urls:
                job = self._ssd_imports.get(url)
                if job is None:
                    # Keeping an unknown outcome must allow a delayed PUT to
                    # arrive. Only an explicit abort installs a tombstone.
                    if url in keep_urls:
                        continue
                    job = _SSDImportJob(state='finished', finish_action='abort')
                    self._ssd_imports[url] = job
                    aborted_import_futures.append(job.import_future)
                changed_to_abort = job.finish_action == 'keep' and url not in keep_urls
                job.finish_action = 'keep' if url in keep_urls else 'abort'
                if (
                    job.finish_future is None
                    or changed_to_abort
                    or (job.finish_future.done() and job.finish_future.exception() is not None)
                ):
                    job.finish_future = _new_operation_future()
                    job.claimed_finish_future = None
                finish_futures.append(job.finish_future)
                imports_to_finalize.append((url, job))
        for waiting in aborted_import_futures:
            waiting.set_exception(RedisClientError('SSD import aborted', code='import_aborted'))
        for url, job in imports_to_finalize:
            self._apply_import_finish_action(url, job)
        result = _new_operation_future()
        remaining = len(finish_futures)
        lock = threading.Lock()

        def on_import_finished(done):
            nonlocal remaining
            with lock:
                remaining -= 1
                last = remaining == 0
            if last:
                result.set_result((self.get_ssd_import_status(urls), b''))

        if not finish_futures:
            result.set_result((self.get_ssd_import_status(urls), b''))
        for waiting in finish_futures:
            waiting.add_done_callback(on_import_finished)
        return result

    def _apply_import_finish_action(self, url, job):
        with self._metadata_lock:
            if (
                job.finish_action is None
                or not job.import_future.done()
                or job.claimed_finish_future is job.finish_future
            ):
                return
            finish_future = job.finish_future
            job.claimed_finish_future = finish_future
            needs_cleanup = job.finish_action == 'abort' and job.published_entry is not None
            if not needs_cleanup:
                if job.finish_action == 'keep':
                    job.state = 'failed' if job.error is not None else 'kept'
                else:
                    job.state = 'finished'
        if needs_cleanup:
            cleanup_future = self._ssd_executor.submit(self._delete_imported_blob, url, job)
            cleanup_future.add_done_callback(lambda done: self._on_import_cleanup_done(job, finish_future, done))
        else:
            finish_future.set_result(None)

    def _delete_imported_blob(self, url, job):
        with self._url_lock(url):
            with self._metadata_lock:
                if self._store.get(url) is not job.published_entry:
                    return
            self._delete_blob_locked(url)

    def _on_import_cleanup_done(self, job, finish_future, cleanup_future):
        error = cleanup_future.exception()
        with self._metadata_lock:
            if error is None:
                job.published_entry = None
                job.state = 'finished'
                job.error = None
            else:
                job.state, job.error = 'failed', str(error)
        if error is None:
            finish_future.set_result(None)
        else:
            finish_future.set_exception(error)

    def _delete_blob(self, url):
        with self._url_lock(url):
            return self._delete_blob_locked(url)

    def _delete_blob_locked(self, url):
        """Delete the blob while the caller holds this URL lock."""
        with self._metadata_lock:
            entry = self._store.get(url)
            if entry is None:
                return False
            path = entry.disk_path if not entry.dump_pin_count else None
        # A failed unlink preserves the published object and all its counters.
        if path is not None:
            path.unlink(missing_ok=True)
        with self._metadata_lock:
            if entry.disk_path is not None:
                self._ssd_bytes -= entry.size
                self._ssd_objects -= 1
                if entry.memory is not None:
                    self._cached_bytes -= entry.size
                    self._cached_objects -= 1
            if entry.memory is not None:
                del self._resident_lru[url]
                self._resident_bytes -= entry.size
            del self._store[url]
            self._used_bytes -= entry.size
            entry.logically_deleted = True
            entry.memory = None
            if path is not None:
                entry.disk_path = None
        return True

    def prepare_dump_sources(self, dump_id, urls):
        if self._stop.is_set():
            raise RuntimeError('data unit is stopping')
        urls = list(dict.fromkeys(urls))
        with self._metadata_lock:
            job = self._dump_jobs.setdefault(dump_id, _DumpJob())
            if job.prepare_future is not None and not job.prepare_future.done():
                if set(urls).issubset(job.prepare_urls | job.pinned_entries.keys()):
                    return job.prepare_future
                raise RedisClientError('dump preparation is still running', code='dump_busy')
            if job.state in ('finishing', 'finished') or (
                job.copy_future is not None and not set(urls).issubset(job.pinned_entries)
            ):
                raise RedisClientError('dump has already started', code='dump_conflict')
            result = _new_operation_future() if job.copy_future is not None else None
            pending = [url for url in urls if url not in job.pinned_entries]
            if result is None:
                result = job.prepare_future = _new_operation_future()
                job.prepare_urls = frozenset(urls)
                job.state, job.first_error, job.prepare_complete = 'preparing', None, False
                job.pending_file_count = len(pending)
        if not pending:
            with self._metadata_lock:
                if job.copy_future is None:
                    job.state, job.prepare_complete = 'prepared', True
            result.set_result((self.get_dump_status(dump_id), b''))
            return result

        def on_blob_prepared(done):
            error = done.exception()
            with self._metadata_lock:
                if error is not None and job.first_error is None:
                    job.first_error = error
                job.pending_file_count -= 1
                last = job.pending_file_count == 0
                if last:
                    error = job.first_error
                    job.state = 'failed' if error is not None else 'prepared'
                    job.prepare_complete = error is None
            if last:
                if error is not None:
                    result.set_exception(error)
                else:
                    result.set_result((self.get_dump_status(dump_id), b''))

        for url in pending:
            preparing = self._ssd_executor.submit(self._prepare_and_pin_dump_blob, job, url)
            preparing.add_done_callback(on_blob_prepared)
        return result

    def _prepare_and_pin_dump_blob(self, job, url):
        self._check_url_owner(url)
        with self._url_lock(url):
            with self._metadata_lock:
                if job.first_error is not None:
                    return
            entry = self._ensure_ssd_copy_locked(url)
            with self._metadata_lock:
                entry.dump_pin_count += 1
                job.pinned_entries[url] = entry

    @staticmethod
    def _copy_file_atomically(source, target):
        temporary = target.with_name(f'{uuid4().hex}.tmp')
        try:
            shutil.copyfile(source, temporary)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)

    def _copy_dump_blob(self, job, url, entry, dump_directory):
        with self._metadata_lock:
            if job.first_error is not None:
                return
        data = dump_directory / 'data'
        data.mkdir(parents=True, exist_ok=True)
        self._copy_file_atomically(entry.disk_path, data / f'{url.rsplit("/", 1)[1]}.blob')

    def start_dump_copy(self, dump_id, dump_dir):
        if self._stop.is_set():
            raise RuntimeError('data unit is stopping')
        with self._metadata_lock:
            job = self._dump_jobs[dump_id]
        dump_directory = Path(dump_dir).absolute()
        with job.lifecycle_lock:
            with self._metadata_lock:
                if job.dump_directory is not None and job.dump_directory != dump_directory:
                    raise RedisClientError('dump directory changed', code='dump_conflict')
                if job.state in ('running', 'complete'):
                    return self.get_dump_status(dump_id)
                if not job.prepare_complete:
                    raise RedisClientError('dump is not prepared', code='dump_conflict')
                job.dump_directory = dump_directory
                job.state, job.first_error = 'running', None
                job.copy_future = result = _new_operation_future()
                pinned_entries = list(job.pinned_entries.items())
                job.pending_file_count = len(pinned_entries)

        def on_blob_copied(done):
            error = done.exception()
            with self._metadata_lock:
                if error is not None and job.first_error is None:
                    job.first_error = error
                job.pending_file_count -= 1
                last = job.pending_file_count == 0
                if last:
                    error = job.first_error
                    job.state = 'failed' if error is not None else 'complete'
            if last:
                if error is not None:
                    result.set_exception(error)
                else:
                    result.set_result(None)

        if not pinned_entries:
            with self._metadata_lock:
                job.state = 'complete'
            result.set_result(None)
        for url, entry in pinned_entries:
            work = self._copy_executor.submit(self._copy_dump_blob, job, url, entry, dump_directory)
            work.add_done_callback(on_blob_copied)
        return self.get_dump_status(dump_id)

    def get_dump_status(self, dump_id):
        with self._metadata_lock:
            job = self._dump_jobs[dump_id]
            return {'state': job.state, 'error': str(job.first_error) if job.first_error is not None else None}

    def finalize_dump(self, dump_id):
        with self._metadata_lock:
            # An abort may overtake a timed-out prepare still queued at this DU.
            job = self._dump_jobs.setdefault(dump_id, _DumpJob(state='finished', prepare_complete=False))
        with job.lifecycle_lock:
            with self._metadata_lock:
                if (job.copy_future is not None and not job.copy_future.done()) or (
                    job.prepare_future is not None and not job.prepare_future.done()
                ):
                    raise RedisClientError('dump copy is still running', code='dump_busy')
                job.state, job.prepare_complete = 'finishing', False
            for url, entry in list(job.pinned_entries.items()):
                with self._url_lock(url):
                    with self._metadata_lock:
                        if url not in job.released_pin_urls:
                            entry.dump_pin_count -= 1
                            job.released_pin_urls.add(url)
                        path = entry.disk_path if entry.logically_deleted and not entry.dump_pin_count else None
                    if path is not None:
                        path.unlink(missing_ok=True)
                        with self._metadata_lock:
                            entry.disk_path = None
                    with self._metadata_lock:
                        del job.pinned_entries[url]
                        job.released_pin_urls.discard(url)
            with self._metadata_lock:
                job.state = 'finished'
            return self.get_dump_status(dump_id)

    def _finalize_dump_reply(self, dump_id):
        return self.finalize_dump(dump_id), b''

    def stop(self):
        self._stop.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=5)
        if self._controller is not None:
            self._controller.close()
        self._server.stop()
        self._admission_executor.shutdown(wait=True)
        self._copy_executor.wait_for_idle()
        self._ssd_executor.wait_for_idle()
        for dump_id in tuple(self._dump_jobs):
            self.finalize_dump(dump_id)
        items = self.finalize_ssd_imports(list(self._ssd_imports), []).result()[0]['items']
        unfinished = {url: item for url, item in items.items() if item['state'] != 'finished'}
        if unfinished:
            raise RedisClientError(
                'SSD import cleanup failed during stop', code='cleanup_failed', body={'items': unfinished}
            )
        self._copy_executor.shutdown()
        self._ssd_executor.shutdown()
        if self._ssd_dir is not None:
            shutil.rmtree(self._ssd_dir)
            self._ssd_dir = None
        with self._metadata_lock:
            self._store.clear()
            self._resident_lru.clear()
            self._resident_bytes = 0
            self._used_bytes = 0
            self._ssd_bytes = 0
            self._ssd_objects = 0
            self._cached_bytes = 0
            self._cached_objects = 0
