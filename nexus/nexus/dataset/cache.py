"""Dataset cache paths, locking, and checksum helpers."""

from __future__ import annotations

import contextlib
import hashlib
import os
import time

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from nexus.utils.paths import DATASET_CACHE_DIR


DEFAULT_DATASET_CACHE_DIR = DATASET_CACHE_DIR


@dataclass(frozen=True)
class CachePolicy:
    """Controls cache reuse during materialization."""

    refresh: bool = False
    offline: bool = False


def dataset_cache_dir(
    *,
    namespace: str,
    dataset: str,
    version: str,
    cache_dir: Path | None = None,
) -> Path:
    root = cache_dir or DEFAULT_DATASET_CACHE_DIR
    return root / namespace / dataset / version


def blob_cache_key(
    *,
    namespace: str,
    dataset: str,
    version: str,
    split: str | None,
    item_id: str,
    root_path: str,
    sha256: str | None = None,
) -> str:
    raw = '\0'.join([namespace, dataset, version, split or '', item_id, root_path, sha256 or ''])
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


@contextlib.contextmanager
def cache_lock(path: Path, *, timeout: float = 300.0, poll_interval: float = 0.1) -> Iterator[None]:
    """Acquire an interprocess lock using exclusive file creation."""

    lock_path = path.with_suffix(f'{path.suffix}.lock')
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    fd: int | None = None
    try:
        while True:
            try:
                fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode('ascii'))
                break
            except FileExistsError:
                if time.monotonic() - started >= timeout:
                    raise TimeoutError(f'Timed out waiting for dataset cache lock: {lock_path}') from None
                time.sleep(poll_interval)
        yield
    finally:
        if fd is not None:
            os.close(fd)
            lock_path.unlink(missing_ok=True)
