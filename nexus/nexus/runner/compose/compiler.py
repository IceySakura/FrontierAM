"""Compile ordered run parts into RunInput and BatchItem objects."""

from __future__ import annotations

import asyncio
import logging

from collections.abc import Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path

from nexus.dataset.item import DatasetItem
from nexus.dataset.layout import is_local_dataset_path
from nexus.dataset.reader import DatasetReader, load_items, open_dataset
from nexus.dataset.source import parse_dataset_source_uri
from nexus.dataset.uri import parse_dataset_uri
from nexus.runner.batch.batch_config import BatchItem
from nexus.runner.compose.compose_hooks import ComposeHookChain, ComposeHookSpec
from nexus.runner.compose.merge import deep_merge
from nexus.runner.compose.model import RawRunPart, RunPart, UseSpec
from nexus.runner.compose.parser import load_run_file, parse_run_composition, parse_use_spec
from nexus.runner.compose.resolver import resolve_non_data_part
from nexus.runner.single import RunInput


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _PreparedPart:
    raw: RawRunPart
    resolved: RunPart | None


class PreparedRunParts:
    """Resolve item-independent compose parts once, then compile many dataset items."""

    def __init__(self, raw_parts: list[RawRunPart], *, compose_hooks: Sequence[ComposeHookSpec] = ()) -> None:
        self.compose_hook_specs = tuple(compose_hooks)
        self.parts = tuple(
            _PreparedPart(raw=raw, resolved=None if _is_data_part(raw) else resolve_non_data_part(raw))
            for raw in raw_parts
        )
        self._compose_hooks: ComposeHookChain | None = None

    def finish_compose_hooks(self) -> None:
        """Run the hooks' whole-batch checks once every item has been compiled."""
        if self.compose_hook_specs:
            self._hook_chain().finish()

    def _hook_chain(self) -> ComposeHookChain:
        if self._compose_hooks is None:
            self._compose_hooks = ComposeHookChain(self.compose_hook_specs)
        return self._compose_hooks

    def compile(self, item: DatasetItem | None, *, log_item: bool = True) -> RunInput:
        config: dict = {}
        applied_data = False

        if item is not None and log_item:
            _log_dataset_item(item)

        for prepared in self.parts:
            if prepared.resolved is None:
                if item is None:
                    raise ValueError(f'{prepared.raw.source} requires a dataset item')
                config = deep_merge(config, _item_run_part(item), item.merge)
                applied_data = True
                continue
            config = deep_merge(config, prepared.resolved.body, prepared.resolved.merge)

        if item is not None and not applied_data:
            config = deep_merge(config, _item_run_part(item), item.merge)
        return RunInput.from_dict(config)

    def compile_batch_item(
        self,
        item: DatasetItem | None,
        *,
        log_item: bool = True,
        preserve_metadata: bool = True,
    ) -> BatchItem:
        if item is not None and not preserve_metadata and item.metadata:
            raise ValueError('task-list v1 does not support item metadata')
        run_input = self.compile(item, log_item=log_item)
        if item is None:
            batch_item = BatchItem(config=run_input, run_id=run_input.run_id, metadata={})
        elif not preserve_metadata:
            batch_item = BatchItem(config=run_input, run_id=run_input.run_id or item.id, metadata={})
        else:
            metadata = dict(item.metadata)
            metadata.setdefault('dataset_item_id', item.id)
            batch_item = BatchItem(config=run_input, run_id=run_input.run_id or item.id, metadata=metadata)
        return self._hook_chain().apply(batch_item) if self.compose_hook_specs else batch_item


class RunComposer:
    """Resolve datasets and ordered run parts before execution."""

    def __init__(self, *, dataset_client=None) -> None:
        self._dataset_client = dataset_client
        self._query_cache: dict[tuple[str, str | None, str | None, int | None], list[DatasetItem]] = {}
        self._item_cache: dict[tuple[str, str | None, str], DatasetItem] = {}
        self._readers: dict[str, DatasetReader] = {}
        self._reader_stack = AsyncExitStack()

    async def __aenter__(self) -> RunComposer:
        await self._reader_stack.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self._reader_stack.__aexit__(exc_type, exc_val, exc_tb)
        self._readers.clear()
        self._query_cache.clear()
        self._item_cache.clear()

    async def compile_single(
        self,
        raw_parts: list[RawRunPart],
        *,
        item_id: str | None = None,
        split: str | None = None,
    ) -> RunInput:
        prepared = PreparedRunParts(raw_parts)
        items = await self._load_dataset_part(raw_parts, split=split, item_id=item_id, limit=2)
        if items:
            if item_id is None and len(items) != 1:
                raise ValueError('--id is required when the selected dataset split has multiple items')
            item = items[0]
        else:
            item = None
        return prepared.compile(item)

    async def compile_batch(
        self,
        raw_parts: list[RawRunPart],
        *,
        item_id: str | None = None,
        split: str | None = None,
        limit: int | None = None,
        compose_hooks: Sequence[ComposeHookSpec] = (),
    ) -> list[BatchItem]:
        prepared = PreparedRunParts(raw_parts, compose_hooks=compose_hooks)
        if limit == 0 and select_dataset_part(raw_parts, split=split) is not None:
            return []
        items = await self._load_dataset_part(raw_parts, split=split, item_id=item_id, limit=limit)
        batch_items = [prepared.compile_batch_item(item) for item in items or [None]]
        prepared.finish_compose_hooks()
        return batch_items

    async def compile_single_file(
        self,
        path: str | Path,
        *,
        item_id: str | None = None,
        split: str | None = None,
    ) -> RunInput:
        data, source_path = load_run_file(path)
        if 'compose' not in data:
            return RunInput.from_file(source_path)
        composition = parse_run_composition(data, source_path=source_path)
        if composition.limit is not None:
            raise ValueError(f'{source_path}: data.limit is only supported for batch compose')
        return await self.compile_single(
            composition.parts,
            item_id=item_id or composition.item_id,
            split=split or composition.split,
        )

    async def compile_batch_file(
        self,
        path: str | Path,
        *,
        split: str | None = None,
        limit: int | None = None,
        compose_hooks: Sequence[ComposeHookSpec] = (),
    ) -> list[BatchItem]:
        data, source_path = load_run_file(path)
        if 'compose' not in data:
            run_input = RunInput.from_file(source_path)
            item = BatchItem(config=run_input, run_id=run_input.run_id, metadata={})
            if not compose_hooks:
                return [item]
            chain = ComposeHookChain(compose_hooks)
            items = [chain.apply(item)]
            chain.finish()
            return items
        composition = parse_run_composition(data, source_path=source_path)
        return await self.compile_batch(
            composition.parts,
            item_id=composition.item_id,
            split=split or composition.split,
            limit=limit if limit is not None else composition.limit,
            compose_hooks=compose_hooks,
        )

    async def _load_dataset_part(
        self,
        raw_parts: list[RawRunPart],
        *,
        split: str | None,
        item_id: str | None,
        limit: int | None,
    ) -> list[DatasetItem]:
        selected = select_dataset_part(raw_parts, split=split)
        if selected is None:
            return []
        data_value, selected_split = selected
        items = await self._load_selected_items(
            data_value,
            selected_split=selected_split,
            item_id=item_id,
            limit=limit,
        )
        if not items:
            raise ValueError(
                f'No dataset items matched {data_value} '
                f'(split={selected_split or "<default>"}, id={item_id or "<all>"}, '
                f'limit={limit if limit is not None else "<none>"}). '
                'Check the split name, --id, and any filters in the dataset URI.'
            )
        return items

    async def _load_selected_items(
        self,
        data_value: str,
        *,
        selected_split: str | None,
        item_id: str | None,
        limit: int | None,
    ) -> list[DatasetItem]:
        _validate_dataset_limit(limit)
        if limit == 0:
            return []
        query_key = (data_value, selected_split, item_id, limit)
        cached = self._query_cache.get(query_key)
        if cached is not None:
            return list(cached)
        if item_id is not None:
            cached_item = self._item_cache.get((data_value, selected_split, item_id))
            if cached_item is not None:
                return [cached_item]

        logger.info(
            'Loading dataset items: source=%s split=%s id=%s limit=%s',
            data_value,
            selected_split or '<default>',
            item_id or '<all>',
            limit if limit is not None else '<none>',
        )
        reader = await self._get_reader(data_value)
        items = await load_items(
            reader,
            split=selected_split,
            item_id=item_id,
            limit=limit,
        )
        self._query_cache[query_key] = list(items)
        for item in items:
            self._item_cache[(data_value, selected_split, item.id)] = item
        logger.info(
            'Loaded dataset items: source=%s count=%s ids=%s',
            data_value,
            len(items),
            _item_ids_display(items),
        )
        return list(items)

    async def _get_reader(self, data_value: str) -> DatasetReader:
        reader = self._readers.get(data_value)
        if reader is None:
            reader = open_dataset(data_value, dataset_client=self._dataset_client)
            reader = await self._reader_stack.enter_async_context(reader)
            self._readers[data_value] = reader
        return reader


def compile_single(
    raw_parts: list[RawRunPart],
    *,
    item_id: str | None = None,
    split: str | None = None,
) -> RunInput:
    """Compile ordered run parts into a single ``RunInput``."""

    async def compile_once() -> RunInput:
        async with RunComposer() as composer:
            return await composer.compile_single(raw_parts, item_id=item_id, split=split)

    return _run_async(compile_once())


def compile_batch(
    raw_parts: list[RawRunPart],
    *,
    split: str | None = None,
    limit: int | None = None,
    compose_hooks: Sequence[ComposeHookSpec] = (),
) -> list[BatchItem]:
    """Compile ordered run parts into batch items, then apply compose hooks to each."""

    async def compile_once() -> list[BatchItem]:
        async with RunComposer() as composer:
            return await composer.compile_batch(raw_parts, split=split, limit=limit, compose_hooks=compose_hooks)

    return _run_async(compile_once())


def compile_single_file(
    path: str | Path,
    *,
    item_id: str | None = None,
    split: str | None = None,
) -> RunInput:
    """Synchronously compile a canonical RunInput or compose file."""

    async def compile_once() -> RunInput:
        async with RunComposer() as composer:
            return await composer.compile_single_file(path, item_id=item_id, split=split)

    return _run_async(compile_once())


def compile_batch_file(
    path: str | Path,
    *,
    split: str | None = None,
    limit: int | None = None,
    compose_hooks: Sequence[ComposeHookSpec] = (),
) -> list[BatchItem]:
    """Synchronously compile a canonical RunInput or compose file into batch items."""

    async def compile_once() -> list[BatchItem]:
        async with RunComposer() as composer:
            return await composer.compile_batch_file(path, split=split, limit=limit, compose_hooks=compose_hooks)

    return _run_async(compile_once())


def _compile_for_item(raw_parts: list[RawRunPart], item: DatasetItem | None) -> RunInput:
    return PreparedRunParts(raw_parts).compile(item)


def select_dataset_part(raw_parts: list[RawRunPart], *, split: str | None = None) -> tuple[str, str | None] | None:
    data_values = [_data_value(raw) for raw in raw_parts if _is_data_part(raw)]
    if not data_values:
        return None
    if len(data_values) > 1:
        raise ValueError('Only one --data/data: run part is currently supported')
    data_value, data_split = data_values[0]
    return data_value, split or data_split


def _log_dataset_item(item: DatasetItem) -> None:
    logger.info(
        'Applying dataset item: id=%s task=%s',
        item.id,
        item.body.get('task_name', '<missing>'),
    )
    task_files = item.resolved_task_files()
    if task_files is None:
        return
    remote = task_files.remote
    remote_display = (
        f'{remote.namespace}/{remote.dataset}@{remote.version}' + (f'?split={remote.split}' if remote.split else '')
        if remote is not None
        else '<local>'
    )
    logger.info(
        'Dataset item files attached: id=%s type=%s path=%s remote=%s size=%s',
        item.id,
        task_files.type,
        task_files.path,
        remote_display,
        task_files.size if task_files.size is not None else '<unknown>',
    )


def _item_run_part(item: DatasetItem) -> dict:
    body = dict(item.body)
    task_files = item.resolved_task_files()
    if task_files is not None:
        body['task_files'] = task_files.model_dump(exclude_none=True)
    return body


def _is_data_part(raw: RawRunPart) -> bool:
    if raw.kind == 'data':
        return True
    if raw.kind != 'use':
        return False
    spec = raw.value if isinstance(raw.value, UseSpec) else parse_use_spec(raw.value)
    return spec.name == 'data'


def _data_value(raw: RawRunPart) -> tuple[str, str | None]:
    if raw.kind == 'data':
        if not isinstance(raw.value, str):
            raise TypeError(f'{raw.source} dataset URI/path must be a string')
        return raw.value, _reference_split(raw.value)
    spec = raw.value if isinstance(raw.value, UseSpec) else parse_use_spec(raw.value)
    uri = str(spec.params['uri'])
    return uri, _reference_split(uri)


def _reference_split(value: str) -> str | None:
    if value.startswith('cos://'):
        from nexus.dataset.cos import CosDatasetRef

        return CosDatasetRef.parse(value).split
    source_uri = parse_dataset_source_uri(value)
    if source_uri is not None:
        values = (source_uri.query or {}).get('split', ())
        if len(values) > 1:
            raise ValueError(f'Dataset source URI has multiple split values: {value}')
        return values[0] if values else None

    path = Path(value).expanduser()
    if is_local_dataset_path(path) or path.is_absolute() or value.startswith(('./', '../', '~/', '.\\', '..\\')):
        return None
    return parse_dataset_uri(value).split


def _item_ids_display(items: list[DatasetItem]) -> str:
    if not items:
        return '<none>'
    ids = [item.id for item in items[:10]]
    suffix = f',...(+{len(items) - 10})' if len(items) > 10 else ''
    return ','.join(ids) + suffix


def _validate_dataset_limit(limit: int | None) -> None:
    if limit is None:
        return
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise ValueError('Dataset limit must be an integer')
    if limit < 0:
        raise ValueError('Dataset limit must be non-negative')


def _run_async(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise RuntimeError('Synchronous compose cannot run inside an existing event loop; use RunComposer instead')
