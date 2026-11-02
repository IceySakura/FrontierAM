"""Build and publish local Nexus datasets."""

from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import importlib
import io
import json
import logging
import re
import shutil
import sys
import time
import zipfile

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp

from nexus.dataset.auth import DatasetAuthenticator, default_wedata_subject
from nexus.dataset.exceptions import UcAssetNotFoundError
from nexus.dataset.io import DATASET_MULTIPART_THRESHOLD, DatasetIoClient
from nexus.dataset.item import DatasetItem
from nexus.dataset.layout import LocalDataset, load_local_dataset
from nexus.dataset.uc import UcMetadataClient
from nexus.dataset.zip_index import read_zip_index_from_file, write_nexus_zip_directory_to_file


ITEM_ID_RE = re.compile(r'^[A-Za-z0-9_.-]+$')
SPLIT_NAME_RE = re.compile(r'^[A-Za-z0-9_.-]+$')
SNAPSHOT_RE = re.compile(r'^dev-[0-9]{8,20}$')
COMPILER_NAME = 'nexus-dataset-compiler'
COMPILER_VERSION = '0.5.0'
PYARROW_PACKAGE = 'pyarrow>=16.0.0'
DEFAULT_UPLOAD_CONCURRENCY = 5
PUBLISH_REQUEST_TIMEOUT_SECONDS = 600


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PublishFile:
    path: str
    local_path: Path
    sha256: str
    size: int
    content_type: str


@dataclass(frozen=True)
class PublishBuild:
    namespace: str
    dataset: str
    snapshot: str
    release_version: str | None
    files_root: Path
    output_dir: Path
    files: list[PublishFile]
    default_split: str
    item_count: int
    sample_item: str | None

    @property
    def file_paths(self) -> list[str]:
        return [item.path for item in self.files]


@dataclass(frozen=True)
class PublishResult:
    build: PublishBuild
    uploaded: int
    commit_response: dict[str, Any] | None
    publish_response: dict[str, Any] | None


def build_publish_layout(
    source: str | Path,
    *,
    namespace: str,
    dataset_name: str | None = None,
    release_version: str | None = None,
    split: str | None = None,
    output_dir: str | Path | None = None,
    snapshot: str | None = None,
    force: bool = True,
    with_parquet: bool = False,
) -> PublishBuild:
    """Compile a local authoring dataset into the standard upload layout."""

    logger.debug(
        'Dataset publish build started: source=%s namespace=%s dataset=%s release_version=%s split=%s output=%s '
        'with_parquet=%s',
        source,
        namespace,
        dataset_name or '<dataset.yaml name>',
        release_version,
        split or '<all>',
        output_dir or '<dataset>/_nexus_out',
        with_parquet,
    )
    local = load_local_dataset(source)
    dataset = dataset_name or _dataset_name(local)
    _validate_name('namespace', namespace)
    _validate_name('dataset', dataset)
    selected_splits = [split] if split else list(local.splits.keys())
    if not selected_splits:
        raise ValueError(f'Local dataset has no splits: {local.root}')
    for split_name in selected_splits:
        _validate_split(split_name)
    logger.debug(
        'Loaded local dataset for publish: root=%s dataset=%s default_split=%s selected_splits=%s',
        local.root,
        dataset,
        local.default_split,
        ','.join(selected_splits),
    )

    now = dt.datetime.now(dt.timezone.utc)
    now_iso = now.strftime('%Y-%m-%dT%H:%M:%SZ')
    snapshot = snapshot or _make_snapshot_id(now)
    if not SNAPSHOT_RE.match(snapshot):
        raise ValueError(f'Invalid snapshot name: {snapshot!r}; expected dev-<8-20 digits>')

    out_dir = Path(output_dir).expanduser().resolve() if output_dir else local.root / '_nexus_out'
    if out_dir.exists():
        if not out_dir.is_dir():
            raise FileExistsError(f'Output path exists and is not a directory: {out_dir}')
        if any(out_dir.iterdir()):
            if not force:
                raise FileExistsError(f'Output directory is not empty: {out_dir}')
            logger.debug('Overwriting existing dataset publish output directory: %s', out_dir)
            shutil.rmtree(out_dir)

    files_root = out_dir / 'files'
    rel_paths: list[str] = []
    split_stats: dict[str, Any] = {}
    total_items = 0
    sample_item: str | None = None

    for split_name in selected_splits:
        items = local.load_items(split=split_name)
        if not items:
            raise ValueError(f'Split {split_name!r} has no items')
        logger.debug('Building dataset split: split=%s items=%s', split_name, len(items))
        ids: list[str] = []
        seen_ids: set[str] = set()
        jsonl_records: list[dict[str, Any]] = []
        blob_paths: list[str] = []

        for item in items:
            _validate_item_id(item.id)
            if item.id in seen_ids:
                raise ValueError(f'Duplicate item id in split {split_name!r}: {item.id}')
            seen_ids.add(item.id)
            record, blob_rel = _record_for_publish(
                item,
                files_root=files_root,
                split=split_name,
                namespace=namespace,
                dataset=dataset,
                version=release_version or snapshot,
            )
            raw_rel = f'{split_name}/raw/{item.id}.json'
            _write_json(files_root / raw_rel, record)
            rel_paths.append(raw_rel)
            jsonl_records.append(record)
            ids.append(item.id)
            total_items += 1
            sample_item = sample_item or item.id
            if blob_rel:
                rel_paths.append(blob_rel)
                blob_paths.append(blob_rel.removeprefix(f'{split_name}/'))

        jsonl_rel = f'{split_name}/jsonl/{split_name}.jsonl'
        jsonl_bytes = _jsonl_bytes(jsonl_records)
        _write_bytes(files_root / jsonl_rel, jsonl_bytes)
        rel_paths.append(jsonl_rel)

        jsonl_zip_rel = f'{split_name}/jsonl/{split_name}.jsonl.zip'
        _write_bytes(files_root / jsonl_zip_rel, _jsonl_zip_bytes(split_name, jsonl_bytes))
        rel_paths.append(jsonl_zip_rel)

        parquet_files: list[str] = []
        if with_parquet:
            parquet_rel = f'{split_name}/parquet/{split_name}.parquet'
            logger.debug('Building parquet file for split: split=%s path=%s', split_name, parquet_rel)
            _write_bytes(files_root / parquet_rel, _parquet_bytes(jsonl_records))
            rel_paths.append(parquet_rel)
            parquet_files.append(f'parquet/{split_name}.parquet')

        split_stats[split_name] = {
            'count': len(items),
            'ids': ids,
            'jsonl': f'jsonl/{split_name}.jsonl.zip',
            'blobs': blob_paths,
        }
        if parquet_files:
            split_stats[split_name]['parquet'] = parquet_files

    default_split = local.default_split if local.default_split in selected_splits else selected_splits[0]
    version_manifest = {
        'schema_version': 1,
        'namespace': namespace,
        'dataset': dataset,
        'version': snapshot,
        'kind': 'dev_snapshot',
        'created_at': now_iso,
        'created_by': 'nexus dataset publish',
        'item_count': total_items,
        'default_split': default_split,
        'splits': split_stats,
        'upload_manifest_path': '_meta/upload_manifest.json',
        'compiler': {
            'name': COMPILER_NAME,
            'version': COMPILER_VERSION,
            'build_time': now_iso,
        },
    }
    _write_json(files_root / '_meta/version.json', version_manifest)
    rel_paths.append('_meta/version.json')

    upload_files = [_manifest_entry(files_root, rel) for rel in rel_paths]
    _write_json(files_root / '_meta/upload_manifest.json', {'version': snapshot, 'files': upload_files})
    rel_paths.append('_meta/upload_manifest.json')
    files = [_publish_file(files_root, rel) for rel in rel_paths]

    simulated_creator = default_wedata_subject() or 'compiler_user'
    _write_json(
        out_dir / 'register_request.json',
        {
            'namespace': namespace,
            'dataset_name': dataset,
            'creator': simulated_creator,
            'admin_list': [simulated_creator],
        },
    )
    _write_json(
        out_dir / 'sim_meta.json',
        {
            'namespace': namespace,
            'dataset': dataset,
            'snapshot': snapshot,
            'release_version': release_version,
            'item_count': total_items,
            'default_split': default_split,
            'splits': selected_splits,
            'sample_item': sample_item,
            'split_files': split_stats,
            'files': rel_paths,
        },
    )

    build = PublishBuild(
        namespace=namespace,
        dataset=dataset,
        snapshot=snapshot,
        release_version=release_version,
        files_root=files_root,
        output_dir=out_dir,
        files=files,
        default_split=default_split,
        item_count=total_items,
        sample_item=sample_item,
    )
    logger.info(
        'Built dataset package: target=%s/%s snapshot=%s files=%s items=%s output=%s',
        build.namespace,
        build.dataset,
        build.snapshot,
        len(build.files),
        build.item_count,
        build.output_dir,
    )
    return build


async def publish_local_dataset(
    source: str | Path,
    *,
    namespace: str,
    dataset_name: str | None,
    release_version: str,
    split: str | None = None,
    output_dir: str | Path | None = None,
    snapshot: str | None = None,
    force: bool = True,
    with_parquet: bool | None = None,
    build_only: bool = False,
    expires: int | None = None,
    register_asset: bool = False,
    upload_concurrency: int = DEFAULT_UPLOAD_CONCURRENCY,
    creator: str | None = None,
    admin_list: list[str] | None = None,
    visibility: str = 'PRIVATE',
    description: str | None = None,
    authenticator: DatasetAuthenticator | None = None,
) -> PublishResult:
    if upload_concurrency < 1:
        raise ValueError('upload_concurrency must be at least 1')
    resolved_with_parquet = resolve_with_parquet(with_parquet, build_only=build_only)
    logger.debug(
        'Publishing dataset: target=%s/%s@%s split=%s build_only=%s parquet=%s upload_concurrency=%s',
        namespace,
        dataset_name or '<dataset.yaml name>',
        release_version,
        split or '<all>',
        build_only,
        resolved_with_parquet,
        upload_concurrency,
    )
    build = build_publish_layout(
        source,
        namespace=namespace,
        dataset_name=dataset_name,
        release_version=release_version,
        split=split,
        output_dir=output_dir,
        snapshot=snapshot,
        force=force,
        with_parquet=resolved_with_parquet,
    )
    if build_only:
        logger.info(
            'Build-only publish completed: target=%s/%s snapshot=%s output=%s',
            build.namespace,
            build.dataset,
            build.snapshot,
            build.output_dir,
        )
        return PublishResult(build=build, uploaded=0, commit_response=None, publish_response=None)

    timeout = aiohttp.ClientTimeout(total=PUBLISH_REQUEST_TIMEOUT_SECONDS)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        io_client = DatasetIoClient(session=session, authenticator=authenticator)
        uc_client = UcMetadataClient(session=session, authenticator=authenticator)
        await _ensure_publish_asset(
            uc_client,
            build.namespace,
            build.dataset,
            creator=creator,
            admin_list=admin_list,
            visibility=visibility,
            description=description,
        )

        regular_upload_paths = [item.path for item in build.files if item.size < DATASET_MULTIPART_THRESHOLD]
        regular_upload_urls: dict[str, str] = {}
        if regular_upload_paths:
            logger.debug(
                'Signing regular upload URLs: target=%s/%s snapshot=%s files=%s expires=%s',
                build.namespace,
                build.dataset,
                build.snapshot,
                len(regular_upload_paths),
                expires,
            )
            signed = await io_client.sign_uploads(
                build.namespace,
                build.dataset,
                build.snapshot,
                regular_upload_paths,
                expires=expires,
            )
            regular_upload_urls = _extract_upload_urls(signed)
            missing = set(regular_upload_paths) - set(regular_upload_urls)
            if missing:
                raise ValueError(f'uploads:sign response did not include URLs for: {sorted(missing)}')

        uploaded = await _upload_publish_files(
            io_client,
            build,
            regular_upload_urls,
            concurrency=upload_concurrency,
            expires=expires,
        )

        logger.debug('Committing snapshot: target=%s/%s snapshot=%s', build.namespace, build.dataset, build.snapshot)
        commit_response = await io_client.commit(build.namespace, build.dataset, build.snapshot)
        logger.debug(
            'Publishing release: target=%s/%s version=%s snapshot=%s',
            build.namespace,
            build.dataset,
            release_version,
            build.snapshot,
        )
        publish_response = await io_client.publish(
            build.namespace,
            build.dataset,
            source_snapshot=build.snapshot,
            release_version=release_version,
        )
        logger.info(
            'Published dataset: target=%s/%s@%s snapshot=%s uploaded=%s',
            build.namespace,
            build.dataset,
            release_version,
            build.snapshot,
            uploaded,
        )
        return PublishResult(
            build=build,
            uploaded=uploaded,
            commit_response=commit_response,
            publish_response=publish_response,
        )


async def _upload_publish_files(
    io_client: DatasetIoClient,
    build: PublishBuild,
    regular_upload_urls: dict[str, str],
    *,
    concurrency: int,
    expires: int | None = None,
) -> int:
    if concurrency < 1:
        raise ValueError('upload concurrency must be at least 1')
    unexpected = set(regular_upload_urls) - set(build.file_paths)
    if unexpected:
        raise ValueError(f'uploads:sign response included unexpected file paths: {sorted(unexpected)}')

    total_bytes = sum(item.size for item in build.files)
    logger.debug(
        'Uploading files: target=%s/%s snapshot=%s files=%s bytes=%s concurrency=%s',
        build.namespace,
        build.dataset,
        build.snapshot,
        len(build.files),
        total_bytes,
        concurrency,
    )

    semaphore = asyncio.Semaphore(concurrency)
    progress = _upload_progress_bar(total=len(build.files))

    async def upload_one(file_info: PublishFile) -> int:
        logger.debug(
            'Uploading dataset file: path=%s size=%s content_type=%s',
            file_info.path,
            file_info.size,
            file_info.content_type,
        )
        async with semaphore:
            progress.start_file()
            try:
                signed_url = regular_upload_urls.get(file_info.path)
                if signed_url is not None:
                    await io_client.upload_signed_local_file(
                        signed_url,
                        file_info.local_path,
                        key=file_info.path,
                        content_type=file_info.content_type,
                    )
                else:
                    await io_client.upload_local_file(
                        build.namespace,
                        build.dataset,
                        build.snapshot,
                        file_info.local_path,
                        key=file_info.path,
                        content_type=file_info.content_type,
                        expires=expires,
                        max_concurrency=concurrency,
                    )
            except BaseException:
                progress.finish_file(success=False)
                raise
            progress.finish_file(success=True)
        logger.debug('Uploaded dataset file: path=%s size=%s', file_info.path, file_info.size)
        return 1

    try:
        results = await asyncio.gather(
            *(upload_one(file_info) for file_info in build.files),
            return_exceptions=True,
        )
    finally:
        progress.close()

    uploaded = 0
    for result in results:
        if isinstance(result, BaseException):
            raise result
        uploaded += result
    logger.debug(
        'Uploaded files: target=%s/%s snapshot=%s uploaded=%s bytes=%s',
        build.namespace,
        build.dataset,
        build.snapshot,
        uploaded,
        total_bytes,
    )
    return uploaded


class ConcurrentUploadProgress:
    """Single-line progress for concurrent uploads.

    The bar has three states: completed (#), uploading (~), and pending (.).
    """

    def __init__(self, total: int, *, stream: Any = None, enabled: bool | None = None, label: str = 'Uploading'):
        self.total = max(total, 0)
        self.done = 0
        self.running = 0
        self.failed = 0
        self.label = label
        self.stream = stream or sys.stderr
        self.enabled = self.stream.isatty() if enabled is None else enabled
        self.started_at = time.monotonic()
        self._last_len = 0
        self._closed = False
        self._render()

    @property
    def pending(self) -> int:
        return max(self.total - self.done - self.running - self.failed, 0)

    def start_file(self) -> None:
        if self.pending > 0:
            self.running += 1
        self._render()

    def finish_file(self, *, success: bool) -> None:
        if self.running > 0:
            self.running -= 1
        if success:
            self.done += 1
        else:
            self.failed += 1
        self._render()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.enabled:
            self.stream.write('\n')
            self.stream.flush()

    def _render(self) -> None:
        if not self.enabled:
            return
        line = self._line()
        padding = max(self._last_len - len(line), 0)
        self.stream.write('\r' + line + (' ' * padding))
        self.stream.flush()
        self._last_len = len(line)

    def _line(self) -> str:
        total = max(self.total, 1)
        pct = int((self.done / total) * 100)
        elapsed = time.monotonic() - self.started_at
        rate = self.done / elapsed if elapsed > 0 else 0.0
        remaining_items = self.total - self.done
        remaining = remaining_items / rate if rate > 0 and remaining_items > 0 else None

        suffix = (
            f'| {self.done}/{self.total} '
            f'[{_format_duration(elapsed)}<{_format_duration(remaining)}, '
            f'{rate:.2f}file/s, uploading={self.running}]'
        )
        prefix = f'{self.label}: {pct:3d}%|'
        columns = shutil.get_terminal_size((100, 20)).columns
        bar_width = max(columns - len(prefix) - len(suffix), 10)
        done_width, running_width, pending_width = self._segment_widths(bar_width)
        return f'{prefix}{"#" * done_width}{"~" * running_width}{"." * pending_width}{suffix}'

    def _segment_widths(self, bar_width: int) -> tuple[int, int, int]:
        total = max(self.total, 1)
        done_width = round((self.done / total) * bar_width)
        running_width = round((self.running / total) * bar_width)
        pending_width = bar_width - done_width - running_width
        if pending_width < 0:
            overflow = -pending_width
            pending_width = 0
            running_width = max(running_width - overflow, 0)
        return done_width, running_width, pending_width


def _format_duration(value: float | None) -> str:
    if value is None:
        return '??:??'
    seconds = max(int(value), 0)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f'{hours:d}:{minutes:02d}:{seconds:02d}'
    return f'{minutes:02d}:{seconds:02d}'


def _upload_progress_bar(total: int) -> ConcurrentUploadProgress:
    return ConcurrentUploadProgress(total)


def _dataset_name(local: LocalDataset) -> str:
    value = local.raw.get('name') or local.raw.get('dataset')
    return str(value) if value else local.root.name


def resolve_with_parquet(value: bool | None, *, build_only: bool) -> bool:
    """Generate parquet only when explicitly requested."""

    if value is not None:
        return value
    return False


async def _ensure_publish_asset(
    uc_client: UcMetadataClient,
    namespace: str,
    dataset: str,
    *,
    creator: str | None,
    admin_list: list[str] | None,
    visibility: str,
    description: str | None,
) -> dict[str, Any]:
    logger.debug('Checking dataset asset: target=%s/%s', namespace, dataset)
    try:
        asset = await uc_client.get_asset(namespace, dataset)
        logger.debug('Dataset asset exists: target=%s/%s', namespace, dataset)
        return asset
    except UcAssetNotFoundError as exc:
        logger.debug(
            'Dataset asset missing; creating: target=%s/%s',
            namespace,
            dataset,
        )
        logger.debug(
            'Dataset asset missing details: target=%s/%s code=%s http_status=%s',
            namespace,
            dataset,
            exc.code,
            exc.http_status,
        )

    resolved_creator = creator or default_wedata_subject() or ''
    if not resolved_creator:
        raise ValueError(
            'creator is required to register a missing dataset asset; configure WEDATA_KEY, set WEDATA_KEY_FILE, '
            'or pass --creator'
        )
    admins = admin_list or [resolved_creator]
    logger.debug(
        'Creating dataset asset: target=%s/%s creator=%s admins=%s visibility=%s',
        namespace,
        dataset,
        resolved_creator,
        ','.join(admins),
        visibility,
    )
    asset = await uc_client.register_nexus_asset(
        namespace,
        dataset,
        creator=resolved_creator,
        admin_list=admins,
        visibility=visibility,
        description=description,
    )
    logger.debug('Created dataset asset: target=%s/%s', namespace, dataset)
    return asset


def _record_for_publish(
    item: DatasetItem,
    *,
    files_root: Path,
    split: str,
    namespace: str,
    dataset: str,
    version: str,
) -> tuple[dict[str, Any], str | None]:
    record = item.model_dump(exclude={'source_path', 'remote_ref'}, exclude_none=True)
    if item.files is None:
        return record, None

    blob_rel = f'{split}/blobs/{item.id}.zip'
    blob_path = files_root / blob_rel
    if item.files.type == 'directory':
        task_files_path = item.local_task_files_path()
        if task_files_path is None or not task_files_path.is_dir():
            raise FileNotFoundError(f'Task files directory not found for item {item.id}: {task_files_path}')
        write_nexus_zip_directory_to_file(task_files_path, blob_path)
    elif item.files.type == 'zip':
        task_files_path = item.local_task_files_path()
        if task_files_path is None or not task_files_path.is_file():
            raise FileNotFoundError(f'Task files zip not found for item {item.id}: {task_files_path}')
        read_zip_index_from_file(task_files_path)
        blob_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(task_files_path, blob_path)
    else:
        raise ValueError(f'Unsupported files type for item {item.id}: {item.files.type}')

    digest, size = _sha256_size_file(blob_path)
    record['files'] = {
        'type': 'zip',
        'path': f'blobs/{item.id}.zip',
        'sha256': digest,
        'size': size,
        'remote': {
            'namespace': namespace,
            'dataset': dataset,
            'version': version,
            'split': split,
        },
    }
    if item.files.root_name is not None:
        record['files']['root_name'] = item.files.root_name
    return record, blob_rel


def _jsonl_bytes(records: list[dict[str, Any]]) -> bytes:
    return ''.join(json.dumps(record, ensure_ascii=False) + '\n' for record in records).encode('utf-8')


def _jsonl_zip_bytes(split: str, data: bytes) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f'{split}.jsonl', data)
    return buffer.getvalue()


def _parquet_bytes(records: list[dict[str, Any]]) -> bytes:
    pa, pq = _load_pyarrow()

    columns: list[str] = []
    for record in records:
        for key in record:
            if key not in columns:
                columns.append(key)
    data: dict[str, list[str | None]] = {}
    for column in columns:
        values: list[str | None] = []
        for record in records:
            value = record.get(column)
            if value is None:
                values.append(None)
            elif isinstance(value, (dict, list)):
                values.append(json.dumps(value, ensure_ascii=False))
            else:
                values.append(str(value))
        data[column] = values

    buffer = io.BytesIO()
    pq.write_table(pa.table(data), buffer)
    return buffer.getvalue()


def _load_pyarrow() -> tuple[Any, Any]:
    try:
        return _import_pyarrow()
    except ImportError as import_exc:
        from nexus.core.dep_checker import install_packages

        ok, message = install_packages([PYARROW_PACKAGE])
        if not ok:
            raise RuntimeError(
                f'Writing parquet requires pyarrow, and automatic install failed: {message}. '
                'Install pyarrow manually or omit --with-parquet.'
            ) from import_exc

    importlib.invalidate_caches()
    try:
        return _import_pyarrow()
    except ImportError as retry_exc:
        raise RuntimeError(
            'Automatic pyarrow install completed, but pyarrow still cannot be imported. '
            'Install pyarrow manually or omit --with-parquet.'
        ) from retry_exc


def _import_pyarrow() -> tuple[Any, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    return pa, pq


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def _write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _sha256_size(data: bytes) -> tuple[str, int]:
    return hashlib.sha256(data).hexdigest(), len(data)


def _content_type(path: str) -> str:
    if path.endswith('.zip'):
        return 'application/zip'
    if path.endswith('.parquet'):
        return 'application/vnd.apache.parquet'
    if path.endswith('.jsonl'):
        return 'application/x-ndjson'
    if path.endswith('.json'):
        return 'application/json'
    return 'application/octet-stream'


def _publish_file(files_root: Path, rel_path: str) -> PublishFile:
    path = files_root / rel_path
    digest, size = _sha256_size_file(path)
    return PublishFile(
        path=rel_path,
        local_path=path,
        sha256=digest,
        size=size,
        content_type=_content_type(rel_path),
    )


def _manifest_entry(files_root: Path, rel_path: str) -> dict[str, Any]:
    publish_file = _publish_file(files_root, rel_path)
    return {
        'path': publish_file.path,
        'sha256': publish_file.sha256,
        'size': publish_file.size,
        'content_type': publish_file.content_type,
    }


def _sha256_size_file(path: Path) -> tuple[str, int]:
    return _sha256_file(path), path.stat().st_size


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_upload_urls(data: dict[str, Any]) -> dict[str, str]:
    """Keep parsing legacy batch-sign responses for downstream callers."""

    urls = data.get('urls')
    if isinstance(urls, dict):
        return {str(path): str(url) for path, url in urls.items()}
    if isinstance(urls, list):
        result = {}
        for item in urls:
            if not isinstance(item, dict):
                continue
            path = item.get('file') or item.get('path') or item.get('key')
            url = item.get('url') or item.get('signed_url') or item.get('signedUrl')
            if path and url:
                result[str(path)] = str(url)
        return result
    return {}


def _make_snapshot_id(now: dt.datetime) -> str:
    snapshot = 'dev-' + now.strftime('%Y%m%d%H%M%S')
    if not SNAPSHOT_RE.match(snapshot):
        raise ValueError(f'Invalid generated snapshot: {snapshot}')
    return snapshot


def _validate_item_id(value: str) -> None:
    if not ITEM_ID_RE.match(value):
        raise ValueError(f'Invalid item id: {value!r}; allowed characters are [A-Za-z0-9_.-]')


def _validate_split(value: str) -> None:
    if not SPLIT_NAME_RE.match(value):
        raise ValueError(f'Invalid split name: {value!r}; allowed characters are [A-Za-z0-9_.-]')


def _validate_name(label: str, value: str) -> None:
    if not value or '/' in value or value.strip() != value:
        raise ValueError(f'Invalid {label}: {value!r}')


def run_publish(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise RuntimeError('Dataset publish cannot run inside an existing event loop')
