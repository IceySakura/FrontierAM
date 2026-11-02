"""Resolve macOSWorld task membership and files through Nexus Dataset.

This branch's DatasetClient does not paginate load_items(), so enumerate IO
pages explicitly and pin the resolved version before reading any item bodies.
No local checkout or manifest is consulted.
"""

from __future__ import annotations

import asyncio
import json
import os

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from nexus.dataset.client import DatasetClient
from nexus.dataset.item import DatasetItem
from nexus.dataset.task_files import task_files_from_spec
from nexus.dataset.uri import parse_dataset_uri


def dataset_uri() -> str:
    uri = os.getenv('MACOSWORLD_DATASET_URI', '').strip()
    if not uri or '/' not in uri or '@' not in uri:
        raise ValueError('Set MACOSWORLD_DATASET_URI to namespace/dataset@version; local task roots are unsupported')
    parse_dataset_uri(uri)
    return uri


def run_async(factory):
    """Allow synchronous batch generators to run inside an existing event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(factory())
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(lambda: asyncio.run(factory())).result()


async def _fetch_items(uri: str, split: str | None) -> list[DatasetItem]:
    parsed = parse_dataset_uri(uri)
    async with DatasetClient(timeout=120) as client:
        version = await client.resolve_version(parsed)
        split = await client.resolve_split(parsed, version=version, split=split or parsed.split)
        if not split:
            raise ValueError('macOSWorld Dataset requires a split')
        pinned = f'{parsed.namespace}/{parsed.dataset}@{version}'
        ids: list[str] = []
        seen_ids: set[str] = set()
        seen_cursors: set[str] = set()
        cursor = None
        assert client.io is not None
        while True:
            page = await client.io.list_items(
                parsed.namespace, parsed.dataset, version, split=split, cursor=cursor, limit=100
            )
            for item_id in page['ids']:
                if not isinstance(item_id, str) or not item_id or item_id in seen_ids:
                    raise ValueError('Dataset returned an invalid or duplicate item id')
                ids.append(item_id)
                seen_ids.add(item_id)
            cursor = page.get('next_cursor')
            if cursor is None or cursor == '':
                break
            cursor = str(cursor)
            if cursor in seen_cursors:
                raise ValueError('Dataset returned a repeated pagination cursor')
            seen_cursors.add(cursor)
        semaphore = asyncio.Semaphore(12)

        async def fetch(item_id: str) -> DatasetItem:
            async with semaphore:
                items = await client.load_items(pinned, split=split, item_id=item_id)
                item = items[0]
                if item.id != item_id:
                    raise ValueError('Dataset response id does not match the requested item')
                return item

        return list(await asyncio.gather(*(fetch(item_id) for item_id in ids)))


def dataset_index() -> dict[str, DatasetItem]:
    uri = dataset_uri()
    split = os.getenv('MACOSWORLD_DATASET_SPLIT', '').strip() or None
    try:

        async def fetch():
            return await asyncio.wait_for(_fetch_items(uri, split), timeout=120)

        items = run_async(fetch)
        index = {}
        for item in items:
            spec = item.resolved_task_files()
            config = item.body.get('task_config', {})
            if (
                spec is None
                or spec.remote is None
                or spec.type != 'zip'
                or not spec.size
                or spec.size < 0
                or not spec.sha256
                or item.body.get('task_name') != 'mac_os_world'
                or config.get('task_id') != item.id
                or not config.get('category')
                or config.get('task_file', 'task.json') != 'task.json'
                or config.get('task_root')
            ):
                raise ValueError('Invalid macOSWorld Dataset item contract')
            if item.id in index:
                raise ValueError('Duplicate macOSWorld Dataset item')
            index[item.id] = item
        if not index:
            raise ValueError('Selected Dataset split is empty')
        return index
    except Exception as exc:
        # Transport exceptions may include credentials or signed URLs.
        raise RuntimeError(
            f'Cannot load macOSWorld Dataset ({type(exc).__name__}). '
            'Check MACOSWORLD_DATASET_URI, split, item metadata and Wedata access; no local fallback was used.'
        ) from None


def item_instruction(item: DatasetItem, language: str) -> str:
    instructions = item.metadata.get('instructions')
    if isinstance(instructions, dict):
        return str(instructions.get(language) or instructions.get('en') or '')
    if language == 'en' and item.metadata.get('instruction'):
        return str(item.metadata['instruction'])

    # v1.0.0 predates instruction metadata. Read its remote task.json rather
    # than requiring a republish or silently depending on the upstream checkout.
    async def read() -> str:
        files = task_files_from_spec(item.resolved_task_files())
        assert files is not None
        data: dict[str, Any] = json.loads(await files.read_text('task.json'))
        if data.get('id') != item.id:
            raise ValueError('Dataset task.json id mismatch')
        task = data.get('task')
        return str(task.get(language) or task.get('en') or '') if isinstance(task, dict) else str(task or '')

    try:
        return run_async(read)
    except Exception as exc:
        raise RuntimeError(
            f'Cannot read Dataset instruction ({type(exc).__name__}); no local fallback was used'
        ) from None
