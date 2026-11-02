"""Select tasks from the published dataset, never from an implicit checkout.

The optional manifest replaces the online index only. Runtime task JSON and
attachments still come from task_files. Checkout enumeration belongs solely to
the dataset packaging script, not an evaluation entrypoint.
"""

from __future__ import annotations

import json
import os

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nexus.extensions.tasks.os_world.data.dataset_files import DATASET_URI_ENV_VAR, dataset_index


TEST_META_PATH_ENV_VAR = 'OSWORLD_TEST_META_PATH'

_FALSE_ENV_VALUES = {'0', 'false', 'no', 'off'}


@dataclass(frozen=True)
class OSWorldBatchTask:
    domain: str
    task_id: str
    instruction: str
    # None in dataset mode: there is no local task JSON, the run unpacks one from
    # the item. Callers put this in batch metadata only, so they omit the entry.
    task_file: Path | None = None


def env_int(name: str, default: int | None) -> int | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return int(value)


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return float(value)


def env_set(name: str) -> set[str] | None:
    value = os.getenv(name)
    if not value:
        return None
    items = {item.strip() for item in value.split(',') if item.strip()}
    return items or None


def env_limit(name: str) -> int | None:
    value = env_int(name, None)
    if value is None:
        return None
    return max(value, 0)


def env_list(name: str, default: str) -> list[str]:
    value = os.getenv(name, default)
    if not value or value.strip().lower() == 'all':
        return []
    return [item.strip() for item in value.split(',') if item.strip()]


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() not in _FALSE_ENV_VALUES


def dataset_source() -> str | None:
    """The published dataset tasks are being enumerated from, for messages and metadata."""
    return os.getenv(DATASET_URI_ENV_VAR, '').strip() or None


def test_meta_path() -> Path | None:
    """Optional curated id list; otherwise the remote dataset supplies membership."""
    value = os.getenv(TEST_META_PATH_ENV_VAR, '').strip()
    if value:
        return Path(value)
    return None


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def _load_test_meta(path: Path) -> dict[str, list[str]]:
    if not path.is_file():
        raise FileNotFoundError(f'OSWorld test meta file not found: {path}')

    raw = _load_json(path)
    if isinstance(raw, dict) and 'examples' in raw:
        raw = raw['examples']

    if isinstance(raw, list):
        grouped: dict[str, list[str]] = {}
        for item in raw:
            if not isinstance(item, dict):
                continue
            domain = item.get('domain') or item.get('snapshot')
            task_id = item.get('task_id') or item.get('id')
            if domain and task_id:
                grouped.setdefault(str(domain), []).append(str(task_id))
        return grouped

    if not isinstance(raw, dict):
        raise TypeError(f'OSWorld test meta must be a dict or list, got {type(raw).__name__}')

    meta: dict[str, list[str]] = {}
    for domain, task_ids in raw.items():
        if not isinstance(task_ids, list):
            continue
        meta[str(domain)] = [str(task_id) for task_id in task_ids]
    return meta


def _manifest_meta() -> dict[str, list[str]]:
    """Group the manifest's items by domain, the shape ``_load_test_meta`` returns.

    ``OSWORLD_TEST_META_PATH`` still wins when it is set: a manifest published
    from the full task set can then be narrowed to the curated V1 list without
    republishing. Ids the manifest does not carry are dropped here rather than
    left to fail per-run -- a missing item cannot be fetched.
    """
    index = dataset_index()
    if index is None:
        raise ValueError(f'Set {DATASET_URI_ENV_VAR} for native OSWorld task selection')
    curated = test_meta_path()
    if curated is not None:
        return {
            domain: [task_id for task_id in task_ids if task_id in index]
            for domain, task_ids in _load_test_meta(curated).items()
        }

    grouped: dict[str, list[str]] = {}
    for task_id, entry in index.items():
        domain = str(entry['metadata'].get('domain') or '')
        if domain:
            grouped.setdefault(domain, []).append(task_id)
    return grouped


def selected_meta() -> dict[str, list[str]]:
    """Domain -> task ids, after the ``OSWORLD_DOMAIN`` filter."""
    meta = _manifest_meta()
    domains = env_list('OSWORLD_DOMAIN', 'all')
    if not domains:
        return meta
    return {domain: meta[domain] for domain in domains if domain in meta}


def iter_selected_tasks(
    *, task_ids: set[str] | None = None, skip: int | None = None, limit: int | None = None
) -> Iterator[OSWorldBatchTask]:
    """Yield the tasks this run selects, in meta order, after every filter."""
    index = dataset_index() or {}
    task_id_filter = task_ids if task_ids is not None else env_set('OSWORLD_TASK_IDS')
    skip = max(env_int('OSWORLD_SKIP', 0) or 0, 0) if skip is None else skip
    limit = env_limit('OSWORLD_LIMIT') if limit is None else limit
    if limit == 0:
        return

    skipped = 0
    produced = 0
    for domain, domain_task_ids in selected_meta().items():
        for task_id in domain_task_ids:
            if task_id_filter is not None and task_id not in task_id_filter:
                continue

            entry = index.get(task_id)
            if entry is None:
                continue
            instruction = str(entry['metadata'].get('instruction') or '')

            if skipped < skip:
                skipped += 1
                continue

            yield OSWorldBatchTask(
                domain=domain,
                task_id=task_id,
                instruction=instruction,
            )
            produced += 1
            if limit is not None and produced >= limit:
                return


def batch_task_metadata(task: OSWorldBatchTask) -> dict[str, Any]:
    """Per-item batch metadata, sharing one shape across every generator.

    ``test_meta_path`` / ``task_file`` are omitted rather than nulled in dataset
    mode: they are local paths, and a null in an exported batch row reads as "we
    looked and there was nothing there" instead of "this run had no checkout".
    """
    metadata: dict[str, Any] = {
        'domain': task.domain,
        'task_id': task.task_id,
        'instruction': task.instruction,
    }
    meta_path = test_meta_path()
    if meta_path is not None:
        metadata['test_meta_path'] = str(meta_path)
    if task.task_file is not None:
        metadata['task_file'] = str(task.task_file)
    return metadata
