"""
File cache implementation for search and visit operations.
Based on verl/tools/utils/file_cache.py
"""

import asyncio
import hashlib
import json
import logging
import os
import queue
import threading
import time

from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path


logger = logging.getLogger(__name__)

# The cache usually lives on a shared CephFS directory where a single metadata operation can take
# seconds, while one asyncio loop drives every trajectory of a rollout worker. Async callers therefore
# never touch the disk on the loop: loads run on a bounded thread pool with a timeout (a slow load is
# treated as a miss, which only costs one API call), and saves go through a single background writer.
LOAD_TIMEOUT_S = float(os.getenv('SEARCH_STATEFUL_CACHE_LOAD_TIMEOUT', '3.0'))
LOAD_WORKERS = int(os.getenv('SEARCH_STATEFUL_CACHE_LOAD_WORKERS', '4'))
WRITE_QUEUE_SIZE = int(os.getenv('SEARCH_STATEFUL_CACHE_WRITE_QUEUE_SIZE', '1024'))
STATS_LOG_INTERVAL_S = float(os.getenv('SEARCH_STATEFUL_CACHE_STATS_INTERVAL', '300'))
META_BATCH_SIZE = 64


class FileCache:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.data_dir = os.path.join(base_dir, 'data')
        self.meta_path = os.path.join(base_dir, 'meta_data.jsonl')

        # Lock for writing to the metadata file
        self.meta_lock = threading.Lock()

        os.makedirs(self.data_dir, exist_ok=True)

        # Initialize meta file if it doesn't exist
        if not os.path.exists(self.meta_path):
            try:
                open(self.meta_path, 'w').close()
            except OSError as e:
                print(f'Error initializing meta file {self.meta_path}: {e}')
                raise

        # Records queued for the background writer, keyed by md5, so a load right after a save still hits.
        self._pending: dict[str, dict] = {}
        self._pending_lock = threading.Lock()
        self._write_queue: queue.Queue[tuple[str, str, dict]] = queue.Queue(maxsize=WRITE_QUEUE_SIZE)
        self._writer: threading.Thread | None = None
        self._writer_lock = threading.Lock()

        self._load_executor = ThreadPoolExecutor(max_workers=LOAD_WORKERS, thread_name_prefix='search-cache-load')
        self._loads_in_flight = 0

        self._stats: Counter[str] = Counter()
        self._load_latency_max_s = 0.0
        self._stats_lock = threading.Lock()
        self._last_stats_log = time.monotonic()

    def md5(self, text):
        """Generates MD5 hash for the key."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def get_path(self, key_md5):
        """Returns the full file path based on the hash."""
        return os.path.join(self.data_dir, f'{key_md5}.json')

    def load(self, key):
        """Return cached content if exists, else None. Blocks on disk IO."""
        key_md5 = self.md5(key)
        pending = self._get_pending(key_md5)
        if pending is not None:
            return pending
        return self._read(key_md5)

    async def load_async(self, key):
        """Return cached content without blocking the event loop.

        Returns None on a miss, when the disk read exceeds LOAD_TIMEOUT_S, or when all load workers are
        busy; callers fall back to the API in every case.
        """
        key_md5 = self.md5(key)
        pending = self._get_pending(key_md5)
        if pending is not None:
            self._count('load_hit_pending')
            return pending

        with self._stats_lock:
            busy = self._loads_in_flight >= LOAD_WORKERS
            if busy:
                self._stats['load_busy'] += 1
            else:
                self._loads_in_flight += 1
        if busy:
            self._maybe_log_stats()
            return None

        start = time.monotonic()
        future = self._load_executor.submit(self._read, key_md5)
        # A timed-out read keeps its thread until the disk returns; the in-flight slot is released only then.
        future.add_done_callback(lambda done: self._on_load_done(done, start))
        try:
            record = await asyncio.wait_for(asyncio.wrap_future(future), timeout=LOAD_TIMEOUT_S)
        except asyncio.TimeoutError:
            self._count('load_timeout')
            record = None
        else:
            self._count('load_hit' if record is not None else 'load_miss')
        self._maybe_log_stats()
        return record

    def save(self, key, content):
        """
        Saves content to disk synchronously using an atomic write pattern.
        Returns the record dict if successful, else None.
        """
        key_md5 = self.md5(key)
        record = self._build_record(key, key_md5, content)
        if not self._write_data(key_md5, record):
            return None
        self._append_meta([self._build_meta_line(key, key_md5, record)])
        return record

    def save_in_background(self, key, content):
        """Queue a save for the background writer and return immediately.

        The write is dropped when the queue is full; queued writes are lost if the process exits first.
        """
        key_md5 = self.md5(key)
        record = self._build_record(key, key_md5, content)
        with self._pending_lock:
            self._pending[key_md5] = record
        try:
            self._write_queue.put_nowait((key, key_md5, record))
        except queue.Full:
            self._discard_pending(key_md5, record)
            self._count('write_dropped')
            self._maybe_log_stats()
            return
        self._ensure_writer()

    def flush(self):
        """Block until every queued background write has been processed."""
        self._write_queue.join()

    def _build_record(self, key, key_md5, content):
        return {'query': key, 'query_md5': key_md5, 'timestamp': datetime.now(timezone.utc).isoformat(), **content}

    def _build_meta_line(self, key, key_md5, record):
        path = self.get_path(key_md5)
        meta = {'query': key, 'query_md5': key_md5, 'timestamp': record['timestamp'], 'file': path}
        return json.dumps(meta, ensure_ascii=False) + '\n'

    def _read(self, key_md5):
        path = self.get_path(key_md5)
        if not os.path.exists(path):
            return None
        try:
            with open(path, encoding='utf-8') as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning('Error loading cache at %s: %s', path, e)
            return None

    def _write_data(self, key_md5, record):
        path = self.get_path(key_md5)
        # Several workers may save the same key concurrently; each needs its own temporary file.
        tmp_path = f'{path}.{os.getpid()}.{threading.get_ident()}.tmp'
        try:
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump(record, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, path)
        except Exception as e:
            logger.warning('Failed to save cache data file %s: %s', path, e)
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            return False
        return True

    def _append_meta(self, lines):
        with self.meta_lock:
            try:
                with open(self.meta_path, 'a', encoding='utf-8') as mf:
                    mf.write(''.join(lines))
            except Exception as e:
                logger.warning('Failed to append %d cache meta records to %s: %s', len(lines), self.meta_path, e)

    def _get_pending(self, key_md5):
        with self._pending_lock:
            return self._pending.get(key_md5)

    def _discard_pending(self, key_md5, record):
        with self._pending_lock:
            if self._pending.get(key_md5) is record:
                del self._pending[key_md5]

    def _ensure_writer(self):
        if self._writer is not None:
            return
        with self._writer_lock:
            if self._writer is None:
                self._writer = threading.Thread(target=self._writer_loop, name='search-cache-writer', daemon=True)
                self._writer.start()

    def _writer_loop(self):
        while True:
            batch = [self._write_queue.get()]
            while len(batch) < META_BATCH_SIZE:
                try:
                    batch.append(self._write_queue.get_nowait())
                except queue.Empty:
                    break

            meta_lines = []
            for key, key_md5, record in batch:
                if self._write_data(key_md5, record):
                    meta_lines.append(self._build_meta_line(key, key_md5, record))
            if meta_lines:
                self._append_meta(meta_lines)

            for _, key_md5, record in batch:
                self._discard_pending(key_md5, record)
            with self._stats_lock:
                self._stats['write_ok'] += len(meta_lines)
                self._stats['write_failed'] += len(batch) - len(meta_lines)
            for _ in batch:
                self._write_queue.task_done()
            self._maybe_log_stats()

    def _on_load_done(self, future: Future, start: float):
        latency_s = time.monotonic() - start
        with self._stats_lock:
            self._loads_in_flight -= 1
            self._load_latency_max_s = max(self._load_latency_max_s, latency_s)
            if latency_s <= 0.5:
                self._stats['load_latency_le_0.5s'] += 1
            elif latency_s <= 1.0:
                self._stats['load_latency_le_1s'] += 1
            elif latency_s <= 2.0:
                self._stats['load_latency_le_2s'] += 1
            elif latency_s <= 3.0:
                self._stats['load_latency_le_3s'] += 1
            else:
                self._stats['load_latency_gt_3s'] += 1

    def _count(self, name):
        with self._stats_lock:
            self._stats[name] += 1

    def _maybe_log_stats(self):
        now = time.monotonic()
        with self._stats_lock:
            if now - self._last_stats_log < STATS_LOG_INTERVAL_S or not self._stats:
                return
            self._last_stats_log = now
            stats = dict(sorted(self._stats.items()))
            load_latency_max_s = self._load_latency_max_s
        logger.info(
            'search_stateful cache stats (cumulative): base_dir=%s write_queue=%d %s load_latency_max=%.2fs',
            self.base_dir,
            self._write_queue.qsize(),
            ' '.join(f'{name}={value}' for name, value in stats.items()),
            load_latency_max_s,
        )


# Default cache paths - can be overridden via configuration
_DEFAULT_SEARCH_CACHE_PATH = os.getenv(
    'DEFAULT_SEARCH_CACHE_PATH',
    '/apdcephfs_gy4/share_302736513/agentic_reasoning/search_history/search',
)
_DEFAULT_VISIT_CACHE_PATH = os.getenv(
    'DEFAULT_VISIT_CACHE_PATH',
    '/apdcephfs_gy4/share_302736513/agentic_reasoning/search_history/visit',
)

# Global cache instances - will be initialized with configurable paths
SEARCH_CACHE: FileCache | None = None
VISIT_CACHE: FileCache | None = None

# A backend is built for every tool call; reusing one FileCache per directory keeps the directory
# checks off that path and makes all calls in a process share one writer and one meta lock.
_CACHES: dict[str, FileCache] = {}
_CACHES_LOCK = threading.Lock()


def _get_local_cache_root() -> Path:
    """Return a per-user writable cache root."""
    xdg_cache_home = os.environ.get('XDG_CACHE_HOME')
    if xdg_cache_home:
        return Path(xdg_cache_home).expanduser() / 'nexus' / 'search_stateful'
    return Path.home() / '.cache' / 'nexus' / 'search_stateful'


def _create_cache(base_dir: str, cache_name: str, fallback_dir: str | None = None) -> FileCache:
    """Return the process-wide cache for base_dir, optionally falling back to a user-writable location."""
    with _CACHES_LOCK:
        cache = _CACHES.get(base_dir)
        if cache is not None:
            return cache
        try:
            cache = FileCache(base_dir)
        except OSError as exc:
            if fallback_dir is None:
                raise
            logger.warning(
                'Failed to initialize %s cache at %s (%s); falling back to %s',
                cache_name,
                base_dir,
                exc,
                fallback_dir,
            )
            cache = _CACHES.get(fallback_dir) or FileCache(fallback_dir)
            _CACHES[fallback_dir] = cache
        _CACHES[base_dir] = cache
        return cache


def initialize_caches(search_cache_path=None, visit_cache_path=None):
    """
    Initialize global cache instances with configurable paths.

    Args:
        search_cache_path: Path for search cache. If None, uses default.
        visit_cache_path: Path for visit cache. If None, uses default.
    """
    global SEARCH_CACHE, VISIT_CACHE

    search_path = search_cache_path or _DEFAULT_SEARCH_CACHE_PATH
    visit_path = visit_cache_path or _DEFAULT_VISIT_CACHE_PATH

    local_cache_root = _get_local_cache_root()
    search_fallback_path = None if search_cache_path else str(local_cache_root / 'search')
    visit_fallback_path = None if visit_cache_path else str(local_cache_root / 'visit')

    SEARCH_CACHE = _create_cache(search_path, 'search', search_fallback_path)
    VISIT_CACHE = _create_cache(visit_path, 'visit', visit_fallback_path)
    return SEARCH_CACHE, VISIT_CACHE
