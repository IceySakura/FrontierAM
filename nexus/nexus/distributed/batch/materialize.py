from __future__ import annotations

import argparse
import asyncio
import contextlib
import copy
import hashlib
import json
import os
import shutil
import stat
import sys
import tempfile
import uuid

from collections.abc import AsyncIterable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, BinaryIO

from nexus.dataset.auth import WedataAuthenticator
from nexus.dataset.cache import sha256_file
from nexus.dataset.item import TaskFilesSpec
from nexus.dataset.reader import open_dataset
from nexus.dataset.zip_index import NexusZipError, write_nexus_zip_directory_to_file
from nexus.distributed.batch.data_list import (
    DEFAULT_MAX_DATA_FRAME_BYTES,
    DEFAULT_MAX_DATA_LIST_BYTES,
    DataListError,
    DataListWriter,
    DataRef,
)
from nexus.distributed.batch.task_list import (
    TaskListVersion,
    canonical_batch_item_metadata,
    canonical_batch_metadata,
    is_valid_batch_item_id,
    normalize_batch_item_id,
)
from nexus.distributed.utils.identity import canonical_json
from nexus.runner.batch.batch_config import BatchItem
from nexus.runner.batch.loader import GeneratorLoader
from nexus.runner.compose.compose_hooks import ComposeHookSpec, postprocess_generated_items
from nexus.runner.compose.model import RawRunPart


_ITEM_ID_WIDTH = 12
COMPOSE_REQUEST_MAX_BYTES = 16 * 1024 * 1024
DEFAULT_MAX_TASK_RECORD_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class MaterializedTaskList:
    path: Path
    sha256: str
    size_bytes: int
    total_items: int
    record_version: TaskListVersion = 1
    resolved_datasets: tuple[dict[str, str], ...] = ()
    batch_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MaterializedDataList:
    path: Path
    sha256: str
    size_bytes: int
    frame_count: int


@dataclass(frozen=True, slots=True)
class MaterializedBatchArtifacts:
    task_list: MaterializedTaskList
    data_list: MaterializedDataList | None = None

    @property
    def path(self) -> Path:
        return self.task_list.path

    @property
    def sha256(self) -> str:
        return self.task_list.sha256

    @property
    def size_bytes(self) -> int:
        return self.task_list.size_bytes

    @property
    def total_items(self) -> int:
        return self.task_list.total_items

    @property
    def record_version(self) -> TaskListVersion:
        return self.task_list.record_version

    @property
    def resolved_datasets(self) -> tuple[dict[str, str], ...]:
        return self.task_list.resolved_datasets

    @property
    def batch_metadata(self) -> dict[str, Any]:
        return self.task_list.batch_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description='Materialize a Nexus batch task list')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--generator', type=Path)
    source.add_argument('--compose-request', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--max-items', type=int, required=True)
    parser.add_argument('--max-bytes', type=int, required=True)
    parser.add_argument('--max-line-bytes', type=int, default=16 * 1024 * 1024)
    parser.add_argument('--task-list-version', type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    if args.generator is not None:
        result = materialize_generator(
            args.generator,
            args.output,
            max_items=args.max_items,
            max_bytes=args.max_bytes,
            max_line_bytes=args.max_line_bytes,
            task_list_version=args.task_list_version,
        )
    else:
        compose, requires_wedata = _load_compose_request(args.compose_request)
        result = asyncio.run(
            materialize_compose(
                [RawRunPart(**part) for part in compose['parts']],
                args.output,
                split=compose.get('split'),
                limit=compose.get('limit'),
                max_items=args.max_items,
                max_bytes=args.max_bytes,
                max_line_bytes=args.max_line_bytes,
                page_size=compose.get('page_size', 500),
                task_list_version=args.task_list_version,
                dataset_authenticator=WedataAuthenticator() if requires_wedata else None,
                reject_local_task_files=True,
            )
        )
    _write_result(args.result, result)


def _load_compose_request(path: Path) -> tuple[dict[str, Any], bool]:
    if path.stat().st_size > COMPOSE_REQUEST_MAX_BYTES:
        raise ValueError('batch compose request exceeds size limit')
    payload = json.loads(path.read_bytes())
    if not isinstance(payload, dict) or set(payload) != {'version', 'compose', 'requires_wedata'}:
        raise ValueError('invalid batch compose request')
    if payload['version'] != 1 or not isinstance(payload['compose'], dict):
        raise ValueError('invalid batch compose request')
    requires_wedata = payload['requires_wedata']
    if not isinstance(requires_wedata, bool):
        raise ValueError('invalid batch compose request')
    compose = payload['compose']
    if not isinstance(compose.get('parts'), list):
        raise ValueError('invalid batch compose request')
    return compose, requires_wedata


def materialize_generator(
    generator_path: Path,
    output_path: Path,
    *,
    max_items: int | None = 10_000,
    max_bytes: int = 1024 * 1024 * 1024,
    max_line_bytes: int | None = 16 * 1024 * 1024,
    task_list_version: TaskListVersion = 1,
    compose_hooks: Sequence[ComposeHookSpec] | None = None,
) -> MaterializedTaskList:
    loader = GeneratorLoader(generator_path)
    batch_metadata = loader.get_batch_metadata()
    canonical_batch_metadata(batch_metadata)

    def cloud_items() -> Iterable[BatchItem]:
        for item in postprocess_generated_items(loader.generate(), hooks=compose_hooks):
            task_files = item.config.task_files
            if task_files is not None and not task_files.is_remote:
                raise ValueError('local_generator_task_files_unsupported')
            if item.config.task_name == 'harbor_bridge_task' and (
                item.config.task_config.get('task_dir') is not None or item.metadata.get('source_path') is not None
            ):
                # The legacy Harbor generator carries submitter-local paths.
                # Dataset-backed Harbor items must use the standard task-files
                # path and data-list materialization instead.
                raise ValueError('local_generator_harbor_task_files_unsupported')
            yield item

    materialized = materialize_items(
        cloud_items(),
        output_path,
        max_items=max_items,
        max_bytes=max_bytes,
        max_line_bytes=max_line_bytes,
        task_list_version=task_list_version,
    )
    return replace(materialized, batch_metadata=copy.deepcopy(batch_metadata))


def materialize_items(
    items: Iterable[BatchItem],
    output_path: Path,
    *,
    max_items: int | None = 10_000,
    max_bytes: int = 1024 * 1024 * 1024,
    max_line_bytes: int | None = 16 * 1024 * 1024,
    task_list_version: TaskListVersion = 1,
) -> MaterializedTaskList:
    writer = _TaskListWriter(
        output_path,
        max_items=max_items,
        max_bytes=max_bytes,
        max_line_bytes=max_line_bytes,
        task_list_version=task_list_version,
    )
    with writer:
        for item in items:
            writer.write(item)
    return writer.result()


async def materialize_async_items(
    items: AsyncIterable[BatchItem],
    output_path: Path,
    *,
    max_items: int | None = 10_000,
    max_bytes: int = 1024 * 1024 * 1024,
    max_line_bytes: int | None = 16 * 1024 * 1024,
    task_list_version: TaskListVersion = 1,
) -> MaterializedTaskList:
    writer = _TaskListWriter(
        output_path,
        max_items=max_items,
        max_bytes=max_bytes,
        max_line_bytes=max_line_bytes,
        task_list_version=task_list_version,
    )
    with writer:
        async for item in items:
            writer.write(item)
    return writer.result()


async def materialize_compose(
    raw_parts: list[RawRunPart],
    output_path: Path,
    *,
    split: str | None = None,
    item_id: str | None = None,
    limit: int | None = None,
    max_items: int | None = 10_000,
    max_bytes: int = 1024 * 1024 * 1024,
    max_line_bytes: int | None = 16 * 1024 * 1024,
    page_size: int = 500,
    task_list_version: TaskListVersion = 1,
    dataset_authenticator: WedataAuthenticator | None = None,
    auto_data_list: bool = False,
    data_list_output_path: Path | None = None,
    max_data_frame_bytes: int = DEFAULT_MAX_DATA_FRAME_BYTES,
    max_task_record_bytes: int = DEFAULT_MAX_TASK_RECORD_BYTES,
    max_data_list_bytes: int = DEFAULT_MAX_DATA_LIST_BYTES,
    reject_local_task_files: bool = False,
    compose_hooks: Sequence[ComposeHookSpec] | None = None,
) -> MaterializedBatchArtifacts:
    from nexus.runner.compose.compiler import PreparedRunParts, select_dataset_part

    if auto_data_list and task_list_version != 2:
        raise ValueError('automatic compose data-list materialization requires task-list version 2')
    prepared = PreparedRunParts(raw_parts, compose_hooks=compose_hooks or ())
    selected = select_dataset_part(raw_parts, split=split)
    if selected is None:

        async def single_item() -> AsyncIterable[BatchItem]:
            yield prepared.compile_batch_item(None, preserve_metadata=task_list_version in {2, 3})
            prepared.finish_compose_hooks()

        return await _materialize_compose_items(
            single_item(),
            output_path,
            max_items=max_items,
            max_bytes=max_bytes,
            max_line_bytes=max_line_bytes,
            task_list_version=task_list_version,
            auto_data_list=auto_data_list,
            data_list_output_path=data_list_output_path,
            max_data_frame_bytes=max_data_frame_bytes,
            max_task_record_bytes=max_task_record_bytes,
            max_data_list_bytes=max_data_list_bytes,
            reject_local_task_files=reject_local_task_files,
        )

    data_value, selected_split = selected
    async with open_dataset(
        data_value,
        authenticator=dataset_authenticator,
        options={'page_size': page_size},
    ) as reader:
        descriptor = await reader.describe()

        async def dataset_items() -> AsyncIterable[BatchItem]:
            async for item in reader.iter_items(split=selected_split, item_id=item_id, limit=limit):
                yield prepared.compile_batch_item(
                    item,
                    log_item=False,
                    preserve_metadata=task_list_version in {2, 3},
                )
            prepared.finish_compose_hooks()

        result = await _materialize_compose_items(
            dataset_items(),
            output_path,
            max_items=max_items,
            max_bytes=max_bytes,
            max_line_bytes=max_line_bytes,
            task_list_version=task_list_version,
            auto_data_list=auto_data_list,
            data_list_output_path=data_list_output_path,
            max_data_frame_bytes=max_data_frame_bytes,
            max_task_record_bytes=max_task_record_bytes,
            max_data_list_bytes=max_data_list_bytes,
            reject_local_task_files=reject_local_task_files,
        )
    if descriptor.provider == 'cos':
        from nexus.dataset.cos import CosDatasetRef

        ref = CosDatasetRef.parse(descriptor.resolved_reference)
        ref.split = selected_split or descriptor.default_split
        return replace(
            result,
            task_list=replace(
                result.task_list, resolved_datasets=({'input_uri': descriptor.reference, 'resolved_uri': ref.data_uri},)
            ),
        )
    if descriptor.provider != 'taiji':
        return result
    resolved_reference = descriptor.resolved_reference
    resolved_split = selected_split or descriptor.default_split
    # ``describe()`` may have embedded the provider default split.  An
    # explicit compose split must replace that value rather than being ignored
    # merely because the descriptor already contains a ``?split=`` query.
    resolved_base = resolved_reference.split('?', 1)[0]
    resolved_reference = f'{resolved_base}?split={resolved_split}' if resolved_split is not None else resolved_base
    return replace(
        result,
        task_list=replace(
            result.task_list,
            resolved_datasets=(
                {
                    'input_uri': descriptor.reference,
                    'resolved_uri': resolved_reference,
                },
            ),
        ),
    )


async def _materialize_compose_items(
    items: AsyncIterable[BatchItem],
    output_path: Path,
    *,
    max_items: int | None,
    max_bytes: int,
    max_line_bytes: int | None,
    task_list_version: TaskListVersion,
    auto_data_list: bool,
    data_list_output_path: Path | None,
    max_data_frame_bytes: int,
    max_task_record_bytes: int,
    max_data_list_bytes: int,
    reject_local_task_files: bool,
) -> MaterializedBatchArtifacts:
    if auto_data_list:
        # Data-list mode is specifically responsible for packaging local task
        # files.  It must still strip legacy Harbor host paths from the
        # serialized run input, but cannot use `_reject_local_task_files`,
        # which intentionally rejects those files before they can be packed.
        items = _sanitize_legacy_harbor_items(items)
    elif reject_local_task_files:
        items = _reject_local_task_files(items)
    if not auto_data_list:
        task_list = await materialize_async_items(
            items,
            output_path,
            max_items=max_items,
            max_bytes=max_bytes,
            max_line_bytes=max_line_bytes,
            task_list_version=task_list_version,
        )
        return MaterializedBatchArtifacts(task_list=task_list)
    return await _materialize_auto_data_list(
        items,
        output_path,
        data_list_output_path=data_list_output_path or output_path.with_name('data_list.bin'),
        max_items=max_items,
        max_bytes=max_bytes,
        max_line_bytes=max_line_bytes,
        max_data_frame_bytes=max_data_frame_bytes,
        max_task_record_bytes=max_task_record_bytes,
        max_data_list_bytes=max_data_list_bytes,
    )


async def _reject_local_task_files(items: AsyncIterable[BatchItem]) -> AsyncIterable[BatchItem]:
    async for item in items:
        item = _sanitize_legacy_harbor_paths(item)
        task_files = item.config.task_files
        if task_files is not None and not task_files.is_remote:
            raise ValueError('local_compose_task_files_unsupported')
        yield item


async def _sanitize_legacy_harbor_items(items: AsyncIterable[BatchItem]) -> AsyncIterable[BatchItem]:
    async for item in items:
        yield _sanitize_legacy_harbor_paths(item)


def _sanitize_legacy_harbor_paths(item: BatchItem) -> BatchItem:
    """Remove legacy Harbor host paths or fail closed when no portable files exist."""

    if item.config.task_name != 'harbor_bridge_task':
        return item
    task_config = item.config.task_config
    has_task_dir = isinstance(task_config, dict) and task_config.get('task_dir') is not None
    has_source_path = item.metadata.get('source_path') is not None
    if not has_task_dir and not has_source_path:
        return item
    if item.config.task_files is None:
        raise ValueError('local_compose_harbor_task_files_unsupported')

    sanitized = item.model_copy(deep=True)
    sanitized.config.task_config.pop('task_dir', None)
    sanitized.metadata.pop('source_path', None)
    return sanitized


async def _materialize_auto_data_list(
    items: AsyncIterable[BatchItem],
    output_path: Path,
    *,
    data_list_output_path: Path,
    max_items: int | None,
    max_bytes: int,
    max_line_bytes: int | None,
    max_data_frame_bytes: int,
    max_task_record_bytes: int,
    max_data_list_bytes: int,
) -> MaterializedBatchArtifacts:
    if _same_output_path(output_path, data_list_output_path):
        raise ValueError('task_list_and_data_list_outputs_must_differ')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data_list_output_path.parent.mkdir(parents=True, exist_ok=True)
    workspace = Path(tempfile.mkdtemp(prefix='.nexus-materialize-', dir=output_path.parent))
    spool_path = workspace / 'records.jsonl'
    staged_task_list = workspace / 'tasks.jsonl'
    staged_data_list = workspace / 'data_list.bin'
    data_writer: DataListWriter | None = None
    data_manifest = None
    item_ids: set[str] = set()
    count = 0
    try:
        with spool_path.open('xb') as spool:
            async for item in items:
                if max_items is not None and count >= max_items:
                    raise ValueError('batch task list exceeds item limit')
                item_id, metadata = _resolve_item(count, item, version=2)
                if item_id in item_ids:
                    raise ValueError(f'duplicate batch item_id: {item_id}')
                canonical_batch_item_metadata(metadata)
                run_input = item.config.model_dump(mode='json', exclude={'run_id'})
                data_ref = None
                task_files = item.config.task_files
                if task_files is not None:
                    if task_files.is_remote:
                        if task_files.type != 'zip':
                            raise ValueError('remote_task_files_type_invalid')
                    else:
                        local_zip, temporary_zip = _prepare_local_task_files(
                            task_files,
                            workspace=workspace,
                            ordinal=count,
                        )
                        try:
                            if data_writer is None:
                                data_writer = DataListWriter(
                                    staged_data_list,
                                    max_frame_bytes=max_data_frame_bytes,
                                    max_bytes=max_data_list_bytes,
                                )
                                data_writer.__enter__()
                            try:
                                data_ref = data_writer.write_frame(
                                    item_id=item_id,
                                    zip_path=local_zip,
                                    root_name=task_files.root_name,
                                )
                            except DataListError as exc:
                                if exc.code == 'invalid_batch_data_list':
                                    raise ValueError(f'local_task_files_invalid_zip:{item_id}') from exc
                                raise
                        finally:
                            if temporary_zip:
                                local_zip.unlink(missing_ok=True)
                        run_input.pop('task_files', None)
                record = {
                    'item_id': item_id,
                    'run_input': run_input,
                    'metadata': metadata,
                    'data_ref': data_ref.model_dump(mode='json') if data_ref is not None else None,
                }
                spool.write(_canonical_json_bytes(record) + b'\n')
                item_ids.add(item_id)
                count += 1
            spool.flush()
            os.fsync(spool.fileno())
        if data_writer is not None:
            active_writer = data_writer
            try:
                active_writer.__exit__(None, None, None)
            finally:
                data_writer = None
            data_manifest = active_writer.result()

        version: TaskListVersion = 3 if data_manifest is not None else 2
        writer = _TaskListWriter(
            staged_task_list,
            max_items=max_items,
            max_bytes=max_bytes,
            max_line_bytes=max_line_bytes,
            task_list_version=version,
        )
        with writer, spool_path.open('rb') as spool:
            for line in spool:
                record = json.loads(line)
                data_ref = DataRef.model_validate(record['data_ref']) if record['data_ref'] is not None else None
                writer.write_resolved(
                    item_id=record['item_id'],
                    run_input=record['run_input'],
                    metadata=record['metadata'],
                    data_ref=data_ref,
                    max_task_record_bytes=max_task_record_bytes if version == 3 else None,
                )
        task_list = writer.result()

        if data_manifest is not None:
            _publish_materialized_artifacts(
                staged_task_list=staged_task_list,
                task_list_output_path=output_path,
                staged_data_list=staged_data_list,
                data_list_output_path=data_list_output_path,
            )
        else:
            staged_task_list.replace(output_path)
        task_list = replace(task_list, path=output_path)
        data_list = (
            MaterializedDataList(
                path=data_list_output_path,
                sha256=data_manifest.sha256,
                size_bytes=data_manifest.size_bytes,
                frame_count=data_manifest.frame_count,
            )
            if data_manifest is not None
            else None
        )
        return MaterializedBatchArtifacts(task_list=task_list, data_list=data_list)
    except BaseException:
        if data_writer is not None and data_writer.stream is not None:
            data_writer.__exit__(*sys.exc_info())
        raise
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def _same_output_path(first: Path, second: Path) -> bool:
    return first.expanduser().resolve(strict=False) == second.expanduser().resolve(strict=False)


def _publish_materialized_artifacts(
    *,
    staged_task_list: Path,
    task_list_output_path: Path,
    staged_data_list: Path,
    data_list_output_path: Path,
) -> None:
    """Publish the task/data pair with recoverable best-effort replacement."""

    if _same_output_path(task_list_output_path, data_list_output_path):
        raise ValueError('task_list_and_data_list_outputs_must_differ')

    destinations = (
        (staged_data_list, data_list_output_path),
        (staged_task_list, task_list_output_path),
    )
    backups: dict[Path, Path] = {}
    published: list[Path] = []
    try:
        for _staged, destination in destinations:
            if os.path.lexists(destination):
                backup = destination.with_name(f'.{destination.name}.publish-backup-{uuid.uuid4().hex}')
                os.replace(destination, backup)
                backups[destination] = backup
        for staged, destination in destinations:
            os.replace(staged, destination)
            published.append(destination)
    except BaseException:
        for destination in reversed(published):
            with contextlib.suppress(OSError):
                destination.unlink()
        for destination, backup in backups.items():
            with contextlib.suppress(OSError):
                os.replace(backup, destination)
        raise
    else:
        for backup in backups.values():
            with contextlib.suppress(OSError):
                backup.unlink()


def _prepare_local_task_files(
    spec: TaskFilesSpec,
    *,
    workspace: Path,
    ordinal: int,
) -> tuple[Path, bool]:
    if spec.source is not None and not spec.is_remote:
        from nexus.dataset.item import LocalFileRef

        source = spec.file_ref()
        if not isinstance(source, LocalFileRef):
            raise ValueError('local_task_files_reference_invalid')
        spec = spec.model_copy(update={'path': source.path})
    if not isinstance(spec.path, str) or not spec.path.strip():
        raise ValueError('local_task_files_path_empty')
    path = Path(spec.path).expanduser()
    try:
        path_stat = path.lstat()
    except FileNotFoundError:
        raise ValueError(f'local_task_files_not_found:{path}') from None
    if stat.S_ISLNK(path_stat.st_mode):
        raise ValueError(f'local_task_files_type_mismatch:{path}')
    if spec.type == 'directory':
        if not stat.S_ISDIR(path_stat.st_mode):
            raise ValueError(f'local_task_files_type_mismatch:{path}')
        archive_path = workspace / f'item-{ordinal:012d}.zip'
        try:
            write_nexus_zip_directory_to_file(path, archive_path)
        except (OSError, NexusZipError) as exc:
            raise ValueError(f'local_task_files_invalid_zip:{path}') from exc
        with archive_path.open('rb') as stream:
            os.fsync(stream.fileno())
        return archive_path, True
    if spec.type != 'zip' or not stat.S_ISREG(path_stat.st_mode):
        raise ValueError(f'local_task_files_type_mismatch:{path}')
    if spec.size is not None and spec.size != path_stat.st_size:
        raise ValueError(f'local_task_files_identity_mismatch:{path}')
    if spec.sha256 is not None and spec.sha256 != sha256_file(path):
        raise ValueError(f'local_task_files_identity_mismatch:{path}')
    return path, False


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(',', ':'),
    ).encode('utf-8')


class _TaskListWriter:
    def __init__(
        self,
        output_path: Path,
        *,
        max_items: int | None,
        max_bytes: int,
        max_line_bytes: int | None,
        task_list_version: TaskListVersion,
    ) -> None:
        if task_list_version not in {1, 2, 3}:
            raise ValueError('unsupported task-list version')
        self.output_path = output_path
        self.max_items = max_items
        self.max_bytes = max_bytes
        self.max_line_bytes = max_line_bytes
        self.task_list_version = task_list_version
        self.temporary = output_path.with_name(f'.{output_path.name}.{uuid.uuid4().hex}.tmp')
        self.digest = hashlib.sha256()
        self.total_bytes = 0
        self.item_ids: set[str] = set()
        self.count = 0
        self.stream: BinaryIO | None = None
        self.complete = False

    def __enter__(self) -> _TaskListWriter:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.temporary.open('xb')
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            if self.stream is not None:
                if exc_type is None:
                    self.stream.flush()
                    os.fsync(self.stream.fileno())
                self.stream.close()
            if exc_type is None:
                self.temporary.replace(self.output_path)
                self.complete = True
        finally:
            self.temporary.unlink(missing_ok=True)

    def write(self, item: BatchItem) -> None:
        if self.task_list_version == 1 and item.metadata:
            raise ValueError('task-list v1 does not support item metadata')
        item_id, metadata = _resolve_item(self.count, item, version=self.task_list_version)
        run_input = item.config.model_dump(mode='json', exclude={'run_id'})
        self.write_resolved(item_id=item_id, run_input=run_input, metadata=metadata)

    def write_resolved(
        self,
        *,
        item_id: str,
        run_input: dict[str, Any],
        metadata: dict[str, Any],
        data_ref: DataRef | None = None,
        max_task_record_bytes: int | None = None,
    ) -> None:
        if self.stream is None:
            raise RuntimeError('task list writer is not open')
        if self.max_items is not None and self.count >= self.max_items:
            raise ValueError('batch task list exceeds item limit')
        if self.task_list_version != 3 and data_ref is not None:
            raise ValueError('data_ref requires task-list version 3')
        if item_id in self.item_ids:
            raise ValueError(f'duplicate batch item_id: {item_id}')
        line = _encode_resolved_record(
            item_id,
            run_input,
            metadata=metadata,
            version=self.task_list_version,
            data_ref=data_ref,
        )
        if self.task_list_version == 3 and max_task_record_bytes is not None:
            frame_size = data_ref.length if data_ref is not None else 0
            identity = {
                'version': 3,
                'item_id': item_id,
                'run_input': run_input,
                'metadata': metadata,
                'data_ref': data_ref.model_dump(mode='json') if data_ref is not None else None,
            }
            identity_size = len(canonical_json(identity).encode())
            if 24 + identity_size + frame_size > max_task_record_bytes:
                raise ValueError('batch_task_record_exceeds_byte_limit')
        if self.max_line_bytes is not None and len(line) > self.max_line_bytes:
            raise ValueError('batch task list item exceeds line size limit')
        next_total = self.total_bytes + len(line)
        if next_total > self.max_bytes:
            raise ValueError('batch task list exceeds byte limit')
        self.stream.write(line)
        self.digest.update(line)
        self.item_ids.add(item_id)
        self.total_bytes = next_total
        self.count += 1

    def result(self) -> MaterializedTaskList:
        if not self.complete:
            raise RuntimeError('task list materialization is not complete')
        return MaterializedTaskList(
            path=self.output_path,
            sha256=self.digest.hexdigest(),
            size_bytes=self.total_bytes,
            total_items=self.count,
            record_version=self.task_list_version,
        )


def _default_item_id(ordinal: int) -> str:
    return f'item-{ordinal + 1:0{_ITEM_ID_WIDTH}d}'


def _resolve_item(
    ordinal: int,
    item: BatchItem,
    *,
    version: TaskListVersion,
) -> tuple[str, dict[str, Any]]:
    if item.run_id is None:
        return _default_item_id(ordinal), item.metadata
    if is_valid_batch_item_id(item.run_id):
        return item.run_id, item.metadata
    if version == 1:
        if not item.run_id.strip():
            raise ValueError('empty batch item_id')
        raise ValueError('invalid batch item_id')

    item_id = normalize_batch_item_id(item.run_id) if item.run_id.strip() else _default_item_id(ordinal)
    metadata = copy.deepcopy(item.metadata)
    nexus_metadata = metadata.get('_nexus')
    if nexus_metadata is None:
        nexus_metadata = {}
    elif not isinstance(nexus_metadata, dict):
        raise ValueError('batch item metadata _nexus must be an object')
    nexus_metadata['original_run_id'] = item.run_id
    metadata['_nexus'] = nexus_metadata
    return item_id, metadata


def _encode_record(
    item: BatchItem,
    item_id: str,
    *,
    metadata: dict[str, Any],
    version: TaskListVersion,
) -> bytes:
    run_input = item.config.model_dump(mode='json', exclude={'run_id'})
    return _encode_resolved_record(item_id, run_input, metadata=metadata, version=version)


def _encode_resolved_record(
    item_id: str,
    run_input: dict[str, Any],
    *,
    metadata: dict[str, Any],
    version: TaskListVersion,
    data_ref: DataRef | None = None,
) -> bytes:
    encoded_item_id = json.dumps(item_id, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
    encoded_run_input = json.dumps(
        run_input,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(',', ':'),
    )
    if version == 1:
        return f'{{"version":1,"item_id":{encoded_item_id},"run_input":{encoded_run_input}}}\n'.encode()
    if version not in {2, 3}:
        raise ValueError('unsupported task-list version')
    encoded_metadata = canonical_batch_item_metadata(metadata).decode()
    if version == 3:
        encoded_data_ref = (
            'null' if data_ref is None else _canonical_json_bytes(data_ref.model_dump(mode='json')).decode()
        )
        return (
            f'{{"version":3,"item_id":{encoded_item_id},"run_input":{encoded_run_input},'
            f'"metadata":{encoded_metadata},"data_ref":{encoded_data_ref}}}\n'
        ).encode()
    return (
        f'{{"version":2,"item_id":{encoded_item_id},"run_input":{encoded_run_input},"metadata":{encoded_metadata}}}\n'
    ).encode()


def _write_result(path: Path, result: MaterializedTaskList | MaterializedBatchArtifacts) -> None:
    if isinstance(result, MaterializedBatchArtifacts):
        result = result.task_list
    payload = {
        'version': 1,
        'sha256': result.sha256,
        'size_bytes': result.size_bytes,
        'total_items': result.total_items,
    }
    if result.resolved_datasets:
        payload['version'] = 3
        payload['resolved_datasets'] = list(result.resolved_datasets)
    elif result.batch_metadata:
        payload['version'] = 2
        payload['batch_metadata'] = result.batch_metadata
    content = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    try:
        with temporary.open('xb') as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
