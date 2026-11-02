"""Read OSWorld task metadata from its dataset URI or an explicit local manifest.

One dataset item per OSWorld task, packaged by ``scripts/build_osworld_dataset.py``
and published by ``nexus dataset publish``. This module is the generator-side half
of that arrangement: it turns the published manifest into the two things a generator
needs per task -- the ``RunInput.task_files`` spec that tells a run which blob to
read, and the ``domain``/``instruction`` metadata attached to each dataset item.

The remote dataset service carries both, so a generating machine needs no local index:
``task_source.iter_selected_tasks`` enumerates from here when a dataset is
configured, and the run host has never needed one (``task.py`` prefetches the
packaged ``task.json``). The checkout is only required by the *packaging* machine.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os

from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any

from nexus.dataset.client import DatasetClient
from nexus.dataset.uri import parse_dataset_uri
from nexus.utils.timeout import async_timeout


# Emitted as an interpolation, never the resolved secret: the batch DB is what
# E1 exports the platform dataset from, so a literal key here would be published
# with the dataset.
WEDATA_KEY_INTERPOLATION = '${oc.env:WEDATA_KEY}'

DATASET_URI_ENV_VAR = 'OSWORLD_DATASET_URI'
DATASET_MANIFEST_ENV_VAR = 'OSWORLD_DATASET_MANIFEST'
DATASET_SPLIT_ENV_VAR = 'OSWORLD_DATASET_SPLIT'
_REMOTE_INDEX_LOCK = Lock()
_REMOTE_INDEX_TIMEOUT = 120


def _parse_dataset_uri(uri: str) -> tuple[str, str, str]:
    """Split ``namespace/dataset@version`` as ``nexus dataset publish`` accepts it."""
    if '/' not in uri or '@' not in uri:
        raise ValueError(f'{DATASET_URI_ENV_VAR} must look like namespace/dataset@version, got {uri!r}')
    parsed = parse_dataset_uri(uri)
    return parsed.namespace, parsed.dataset, parsed.version


def dataset_index() -> dict[str, dict[str, Any]] | None:
    """Map task id -> ``{'files': …, 'metadata': …}``, or None when no dataset is set.

    URI alone is sufficient: DatasetClient fetches pages once and supplies
    metadata plus size/hash/remote coordinates. An explicitly selected manifest
    replaces that lookup for offline use; a bad file never falls back to remote.
    Only metadata is cached, never signed URLs or asset contents.
    """
    uri = os.getenv(DATASET_URI_ENV_VAR, '').strip()
    if not uri:
        return None

    namespace, dataset, version = _parse_dataset_uri(uri)
    split = os.getenv(DATASET_SPLIT_ENV_VAR, '').strip() or parse_dataset_uri(uri).split or 'test'

    manifest_value = os.getenv(DATASET_MANIFEST_ENV_VAR, '').strip()
    if manifest_value:
        return _load_manifest_index(manifest_value, namespace, dataset, version, split)
    # Cache by source and credential context. Never put credential values into
    # cached arguments or persist an authenticated index to a shared disk cache.
    auth_scope = hashlib.sha256(
        json.dumps([
            os.getenv(name, '')
            for name in (
                'WEDATA_KEY',
                'WEDATA_KEY_FILE',
                'TAI_PAT',
                'TAIJI_DATASET_AUTH_TOKEN',
            )
        ]).encode()
    ).hexdigest()
    with _REMOTE_INDEX_LOCK:
        return _load_remote_index(
            f'{namespace}/{dataset}@{version}',
            split,
            os.getenv('TAIJI_DATASET_SERVER_URL', '').strip() or None,
            auth_scope,
        )


def _validate_entry(
    record: dict[str, Any],
    expected: tuple[str, str, str, str],
    source: str,
) -> tuple[str, dict[str, Any]]:
    item_id = record.get('id')
    files = record.get('files')
    if not isinstance(item_id, str) or not item_id.strip() or not isinstance(files, dict):
        raise ValueError(f'{source}: each OSWorld item requires an id and files')
    remote = files.get('remote') or {}
    actual = tuple(remote.get(key) for key in ('namespace', 'dataset', 'version', 'split'))
    if actual != expected:
        raise ValueError(f'{source}: item {item_id} targets {actual}, expected {expected}')
    if files.get('type') != 'zip' or not files.get('path'):
        raise ValueError(f'{source}: item {item_id} requires a remote ZIP path')
    size = files.get('size')
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise ValueError(f'{source}: item {item_id} requires positive size metadata')
    metadata = record.get('metadata') or {}
    for key in ('domain', 'instruction'):
        if not isinstance(metadata.get(key), str) or not metadata[key].strip():
            raise ValueError(f'{source}: item {item_id} requires metadata.{key}')
    return item_id, {'files': files, 'metadata': metadata}


async def _fetch_remote_index(uri: str, split: str, server_url: str | None) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    # One resolved version for the entire pagination, including latest/dev.
    # Clients and HTTP sessions are created and closed on the same event loop.
    async with async_timeout(_REMOTE_INDEX_TIMEOUT), DatasetClient(server_url=server_url) as client:
        ref = await client.resolve_ref(uri, split=split)
        if ref.split is None:
            raise ValueError('Resolved OSWorld dataset reference requires a split')
        expected = (ref.namespace, ref.dataset, ref.version, ref.split)
        async for item in client.iter_resolved_items(ref):
            files = item.resolved_task_files()
            item_id, entry = _validate_entry(
                {
                    'id': item.id,
                    'files': files.model_dump(exclude_none=True) if files is not None else None,
                    'metadata': item.metadata,
                },
                expected,
                uri,
            )
            if item_id in index:
                raise ValueError(f'Duplicate OSWorld item id: {item_id}')
            index[item_id] = entry
    if not index:
        raise ValueError(f'{uri}: selected split contains no OSWorld items')
    return index


@lru_cache(maxsize=4)
def _load_remote_index(
    uri: str,
    split: str,
    server_url: str | None,
    _auth_scope: str,
) -> dict[str, dict[str, Any]]:
    """Bridge the synchronous generator API to DatasetClient.

    Both batch generation and run-part expansion may already be inside asyncio.
    Use a worker event loop rather than nested asyncio.run or a new task
    on the blocked caller loop. Failures are not
    cached by lru_cache; the next attempt can retry with repaired credentials.
    """
    try:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(_fetch_remote_index(uri, split, server_url))
        with ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(lambda: asyncio.run(_fetch_remote_index(uri, split, server_url))).result()
    except Exception as exc:
        # The transport exception can contain tokens or signed URLs. Surface a
        # safe error instead of serializing it into batch logs.
        raise RuntimeError(
            f'Cannot load OSWorld dataset index for {uri} (split={split}, {type(exc).__name__}). '
            'Check dataset metadata, access and WEDATA_KEY; no local fallback was used.'
        ) from None


@lru_cache(maxsize=4)
def _load_manifest_index(
    manifest_value: str,
    namespace: str,
    dataset: str,
    version: str,
    split: str,
) -> dict[str, dict[str, Any]]:
    """Parse and validate one manifest. Cached: both enumeration and _build_config
    run per task and would otherwise re-read the whole file once for every task.

    The release coordinates are cache-key arguments rather than being read from
    the environment inside, so a caller that switches dataset does not silently
    get the previous one's index back.

    The URI is cross-checked against the manifest rather than used to build the
    spec: a manifest left over from an earlier ``--build-only``, or from a
    different version, would otherwise point every run at the wrong release
    without any visible symptom.
    """
    manifest_path = Path(manifest_value).expanduser()
    if not manifest_path.is_file():
        raise ValueError(f'{DATASET_MANIFEST_ENV_VAR} does not exist: {manifest_path}')

    expected = (namespace, dataset, version, split)
    index: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(manifest_path.read_text(encoding='utf-8').splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        item_id, entry = _validate_entry(record, expected, f'{manifest_path}: line {line_number}')
        if item_id in index:
            raise ValueError(f'Duplicate OSWorld item id: {item_id}')
        index[item_id] = entry

    if not index:
        raise ValueError(f'{DATASET_MANIFEST_ENV_VAR} contains no usable items: {manifest_path}')
    return index


def dataset_task_files(task_id: str) -> dict[str, Any] | None:
    """``RunInput.task_files`` for one task, or None when no dataset is configured."""
    index = dataset_index()
    if index is None:
        return None
    entry = index.get(task_id)
    if entry is None:
        raise ValueError(
            f'Task {task_id} is not in the selected dataset index. Re-run the packaging script and republish so '
            'the dataset covers every task this run selects.'
        )
    return entry['files']


def apply_dataset_task_files(config: dict[str, Any], task_config: dict[str, Any], task_id: str) -> None:
    """Attach the dataset wiring to one generated run config, in place.

    Both halves are needed and neither works alone: ``task_files`` tells the run
    which blob to read, ``wedata_key`` is how the run authenticates to read it
    (``run_single`` may not read process env for its own configuration, so the
    key has to travel inside task_config -- see ``task.py:_apply_wedata_key``).
    """
    if dataset_index() is None:
        return
    task_config['wedata_key'] = WEDATA_KEY_INTERPOLATION
    task_files = dataset_task_files(task_id)
    if task_files is not None:
        config['task_files'] = task_files
