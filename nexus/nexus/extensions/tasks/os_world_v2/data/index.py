"""Remote V2 dataset index shared by local launch and platform JSONL export."""

import asyncio
import hashlib
import json
import os
import re

from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Any

from nexus.dataset.auth import WedataAuthenticator
from nexus.dataset.client import DatasetClient
from nexus.dataset.uri import parse_dataset_uri
from nexus.extensions.tasks.os_world_v2.adapter.source import COMMIT, RELEASE, task_id
from nexus.utils.timeout import async_timeout


def dataset_uri():
    value = os.getenv('OSWORLD_V2_DATASET_URI', '').strip()
    if not value or '@' not in value:
        raise ValueError('OSWORLD_V2_DATASET_URI must identify a published dataset: namespace/dataset@version')
    parsed = parse_dataset_uri(value)
    if parsed.version in {'latest', 'dev'}:
        raise ValueError('V2 requires an immutable published dataset version')
    return value


def _validate(record):
    metadata, files = record['metadata'], record['files']
    selected = task_id(metadata['task_id'])
    if metadata.get('source_commit') != COMMIT or metadata.get('benchmark_release') != RELEASE:
        raise ValueError('Remote V2 dataset targets an unsupported source/release')
    if not metadata.get('instruction') or not metadata.get('bundle_sha256'):
        raise ValueError('V2 dataset metadata lacks instruction or bundle digest')
    if not re.fullmatch(r'[0-9a-f]{64}', metadata['bundle_sha256']):
        raise ValueError('V2 dataset bundle digest is malformed')
    if task_id(record['id']) != selected:
        raise ValueError('V2 dataset item ID does not match task metadata')
    if files.get('type') != 'zip' or not files.get('remote') or not files.get('sha256') or not files.get('size'):
        raise ValueError('V2 requires remote ZIP with SHA256 and size')
    return selected


async def fetch(uri, split):
    result = {}
    auth = WedataAuthenticator() if os.getenv('WEDATA_KEY') else None
    async with async_timeout(180), DatasetClient(authenticator=auth) as client:
        ref = await client.resolve_ref(uri, split=split)
        async for item in client.iter_resolved_items(ref):
            files = item.resolved_task_files()
            record: dict[str, Any] = {
                'id': item.id,
                'metadata': item.metadata,
                'files': files.model_dump(exclude_none=True) if files is not None else {},
            }
            selected = _validate(record)
            remote = record['files']['remote']
            if tuple(remote.get(key) for key in ('namespace', 'dataset', 'version', 'split')) != (
                ref.namespace,
                ref.dataset,
                ref.version,
                ref.split,
            ):
                raise ValueError('V2 dataset item points outside the selected release')
            if selected in result:
                raise ValueError('Duplicate V2 task id in dataset')
            result[selected] = record
    if not result:
        raise ValueError('V2 dataset is empty')
    return result


@lru_cache(maxsize=4)
def _cached(uri, split, auth_scope):
    with ThreadPoolExecutor(max_workers=1) as pool:
        try:
            return pool.submit(asyncio.run, fetch(uri, split)).result()
        except Exception as exc:
            raise RuntimeError(f'Cannot load V2 remote dataset ({type(exc).__name__}); no checkout fallback') from None


def dataset_index():
    scope = hashlib.sha256(
        json.dumps([os.getenv(k, '') for k in ('WEDATA_KEY', 'WEDATA_KEY_FILE', 'TAI_PAT')]).encode()
    ).hexdigest()
    return _cached(dataset_uri(), os.getenv('OSWORLD_V2_DATASET_SPLIT', 'test'), scope)


def export_platform_jsonl(path: Path, *, exclude_missing_dependencies=False):
    count = 0
    with path.open('w', encoding='utf-8') as output:
        for selected, item in dataset_index().items():
            metadata = item['metadata']
            if exclude_missing_dependencies and metadata.get('missing_external_dependencies'):
                continue
            row = {
                'id': f'v21__{selected}',
                'meta_data_': {
                    'nexus_run_input': {
                        'task_config': {
                            'domain': 'tasks',
                            'task_id': selected,
                            'bundle_sha256': metadata['bundle_sha256'],
                        },
                        'task_files': item['files'],
                    },
                    'instruction': metadata['instruction'],
                    'instruction_is_display_template': metadata.get('instruction_is_display_template', False),
                    'domain': 'tasks',
                    'benchmark_release': RELEASE,
                    'source_commit': COMMIT,
                    'missing_external_dependencies': metadata.get('missing_external_dependencies', []),
                },
            }
            output.write(json.dumps(row, ensure_ascii=False) + '\n')
            count += 1
    return count
