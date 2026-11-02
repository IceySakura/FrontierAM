"""Virtual Nexus Dataset source for existing T3-Harbor task directories."""

from __future__ import annotations

import hashlib
import logging
import os

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from nexus.dataset.item import DatasetItem, TaskFilesSpec
from nexus.dataset.reader import (
    BaseDatasetReader,
    DatasetDescriptor,
    DatasetReaderError,
    DatasetSplitDescriptor,
)
from nexus.dataset.source import DatasetSourceURI, dataset_source
from nexus.extensions.harbor_bridge.compiler import (
    compile_harbor_task,
    load_harbor_task,
)
from nexus.extensions.harbor_bridge.discovery import discover_task_dirs
from nexus.extensions.harbor_bridge.identity import (
    build_task_identity,
    task_content_sha256,
    task_run_id,
)


logger = logging.getLogger(__name__)

# Accepted T3-Harbor-specific URI query parameters. Generic selection controls such
# as split, item id, and limit belong to Run Compose / Dataset CLI arguments.
QUERY_KEYS: frozenset[str] = frozenset({
    'compile_workers',
    'exclude',
    'flat',
    'include',
    'name',
    'skip',
    'skip_invalid',
    'workers',
})


class HarborDatasetReader(BaseDatasetReader):
    """Scan a T3-Harbor tree and lazily compile tasks into Dataset Items."""

    def __init__(self, uri: DatasetSourceURI) -> None:
        super().__init__()
        if uri.authority:
            raise ValueError('t3-harbor source only supports local absolute paths')
        root = Path(uri.path).expanduser()
        if not root.is_absolute():
            raise ValueError('t3-harbor source path must be absolute; use t3-harbor:///absolute/path')
        _validate_query_keys(uri)
        self.root = root.resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f'T3-Harbor dataset directory not found: {self.root}')
        self.uri = uri
        self.flat = _query_bool(uri, 'flat', default=False)
        self.skip_invalid = _query_bool(uri, 'skip_invalid', default=False)
        self.skip = _query_int(uri, 'skip', default=0)
        self.workers = _query_int(uri, 'workers', default=_env_workers(), minimum=1)
        self.compile_workers = _query_int(uri, 'compile_workers', default=1, minimum=1)
        self.include = _query_names(uri, 'include')
        self.exclude = _query_names(uri, 'exclude')
        # Mirrors the legacy generator's HARBOR_DATASET_NAME / directory-name fallback.
        # Reported as the ``benchmark`` metadata key that trajectory exports group by.
        self.name = _query_str(uri, 'name', default='') or self.root.name
        self._task_dirs: dict[str, Path] | None = None

    async def _close(self, exc_type, exc_val, exc_tb) -> None:
        self._task_dirs = None

    async def describe(self) -> DatasetDescriptor:
        self._require_open()
        _validate_split(self.uri.split)
        task_dirs = self._discover()
        return DatasetDescriptor(
            reference=self.uri.display,
            resolved_reference=str(self.root),
            provider='t3-harbor',
            name=self.name,
            default_split='test',
            splits=[DatasetSplitDescriptor(name='test', item_count=len(task_dirs))],
            metadata={
                'path': str(self.root),
                'format': 't3-harbor',
                'consistency': 'live',
                'name': self.name,
                'flat': self.flat,
                'skip_invalid': self.skip_invalid,
                'skip': self.skip,
                'workers': self.workers,
                'compile_workers': self.compile_workers,
                'include': sorted(self.include),
                'exclude': sorted(self.exclude),
            },
        )

    async def iter_item_ids(
        self,
        *,
        split: str | None = None,
        limit: int | None = None,
    ):
        self._require_open()
        effective_split = split if split is not None else self.uri.split
        _validate_split(effective_split)
        _validate_limit(limit)
        if limit == 0:
            return
        item_ids = sorted(self._discover())
        for index, item_id in enumerate(item_ids):
            self._require_open()
            if limit is not None and index >= limit:
                return
            yield item_id

    async def iter_items(
        self,
        *,
        split: str | None = None,
        item_id: str | None = None,
        limit: int | None = None,
    ):
        self._require_open()
        effective_split = split if split is not None else self.uri.split
        _validate_split(effective_split)
        _validate_limit(limit)
        if item_id is not None and (not isinstance(item_id, str) or not item_id):
            raise DatasetReaderError('dataset_item_id_invalid', detail='item_id must be a non-empty string')
        if limit == 0:
            return
        task_dirs = self._discover()
        if item_id is not None:
            task_dir = task_dirs.get(item_id)
            if task_dir is None:
                raise DatasetReaderError('dataset_item_not_found', detail=item_id)
            selected = [(item_id, task_dir)]
        else:
            selected = sorted(task_dirs.items())

        loaded_count = 0
        failures: list[tuple[Path, Exception]] = []
        for _selected_id, task_dir, item, compile_error in self._compile_selected(selected):
            self._require_open()
            # ``limit`` counts loadable tasks, so invalid ones are replaced instead of dropped.
            if limit is not None and loaded_count >= limit:
                break
            if compile_error is not None:
                # Report each failure as it happens; compiling a large selection takes minutes.
                failures.append((task_dir, compile_error))
                logger.warning(
                    'Invalid T3-Harbor task %s: %s: %s',
                    task_dir,
                    type(compile_error).__name__,
                    str(compile_error).replace('\n', ' | '),
                )
            else:
                assert item is not None
                loaded_count += 1
                yield item

        if failures:
            if not loaded_count or not self.skip_invalid:
                raise DatasetReaderError(
                    'dataset_item_invalid',
                    detail=_invalid_tasks_message(self.root, failures, skipping=self.skip_invalid),
                )
            logger.warning(
                'Skipped %d invalid T3-Harbor task(s); loaded %d valid task(s) under %s',
                len(failures),
                loaded_count,
                self.root,
            )

    def _compile_selected(
        self,
        selected: list[tuple[str, Path]],
    ):
        """Compile a bounded window concurrently while preserving input order."""
        if self.compile_workers == 1:
            for entry in selected:
                yield self._compile_item_captured(entry)
            return

        with ThreadPoolExecutor(max_workers=self.compile_workers) as executor:
            for offset in range(0, len(selected), self.compile_workers):
                batch = selected[offset : offset + self.compile_workers]
                yield from executor.map(self._compile_item_captured, batch)

    def _compile_item_captured(
        self,
        entry: tuple[str, Path],
    ) -> tuple[str, Path, DatasetItem | None, Exception | None]:
        selected_id, task_dir = entry
        try:
            return selected_id, task_dir, self._compile_item(selected_id, task_dir), None
        except Exception as exc:
            return selected_id, task_dir, None, exc

    def _discover(self) -> dict[str, Path]:
        if self._task_dirs is not None:
            return dict(self._task_dirs)
        task_dirs = discover_task_dirs([self.root], flat=self.flat, workers=self.workers)

        discovered: dict[str, Path] = {}
        for task_dir in task_dirs:
            item_id = task_dir.name
            if self.include and item_id not in self.include:
                continue
            if item_id in self.exclude:
                continue
            previous = discovered.get(item_id)
            if previous is not None and previous != task_dir:
                raise DatasetReaderError(
                    'dataset_duplicate_item_id',
                    detail=(
                        f'{item_id!r}: {previous}, {task_dir}. Task directory names are the Dataset Item IDs; '
                        'rename one of them, point the URI at a deeper root, or add '
                        f'exclude={item_id} to select the other one.'
                    ),
                )
            discovered[item_id] = task_dir
        if self.skip:
            discovered = dict(sorted(discovered.items())[self.skip :])
        _check_run_id_collisions(discovered)
        logger.debug(
            'Discovered %d T3-Harbor task dir(s) under %s; %d selected after include/exclude/skip (flat=%s)',
            len(task_dirs),
            self.root,
            len(discovered),
            self.flat,
        )
        self._task_dirs = discovered
        return dict(discovered)

    def _compile_item(self, item_id: str, task_dir: Path) -> DatasetItem:
        loaded = load_harbor_task(task_dir)
        config = loaded.config
        compiled = compile_harbor_task(
            task_dir,
            config,
            loaded.task,
        )
        identity = build_task_identity(task_dir, item_id)
        body = compiled.dataset_body()
        task_config = dict(body['task_config'])
        task_config.pop('task_dir', None)
        body['task_config'] = task_config
        body['run_id'] = identity.run_id
        metadata = {
            'benchmark': self.name,
            **compiled.metadata,
            **identity.metadata(task_dir=task_dir, config=config, include_source_path=False),
        }
        return DatasetItem(
            id=item_id,
            body=body,
            metadata=metadata,
            files=TaskFilesSpec(type='directory', path=str(task_dir)),
        )


@dataset_source('t3-harbor')
def t3_harbor_dataset_source(uri: DatasetSourceURI) -> HarborDatasetReader:
    """Create a local virtual Dataset source for ``t3-harbor:///``."""
    return HarborDatasetReader(uri)


def _check_run_id_collisions(discovered: dict[str, Path]) -> None:
    """Reject task IDs that collapse into one batch run_id.

    The readable task-ID component normalizes punctuation, and batch generation
    only inserts run IDs it has not seen. Two tasks with the same normalized ID
    and canonical content would otherwise leave one task silently missing.
    """
    from nexus.runner.batch.batch_config import sanitize_run_id

    by_sanitized_item_id: dict[str, list[str]] = {}
    for item_id in discovered:
        readable_run_id = task_run_id(item_id, '')
        by_sanitized_item_id.setdefault(sanitize_run_id(readable_run_id), []).append(item_id)

    # Only task IDs whose readable components collapse together can produce the
    # same run ID. Hash full task identity for that small subset.
    by_run_id: dict[str, list[Path]] = {}
    for item_ids in by_sanitized_item_id.values():
        if len(item_ids) < 2:
            continue
        for item_id in item_ids:
            task_dir = discovered[item_id]
            try:
                digest = task_content_sha256(task_dir)
            except OSError:
                # Unreadable identity inputs are reported during compilation; treat each as its own
                # identity here so discovery neither crashes nor invents a collision.
                digest = hashlib.sha256(str(task_dir).encode()).hexdigest()
            run_id = sanitize_run_id(task_run_id(item_id, digest))
            by_run_id.setdefault(run_id, []).append(task_dir)

    details = '\n'.join(
        f'  - {run_id}: {", ".join(str(path) for path in sorted(paths))}'
        for run_id, paths in sorted(by_run_id.items())
        if len(paths) > 1
    )
    if details:
        raise ValueError(
            'T3-Harbor task IDs collapse into the same batch run_id after sanitization, so one of '
            f'each pair would be silently dropped from the batch. Rename one of them:\n{details}'
        )


def _invalid_tasks_message(root: Path, failures: list[tuple[Path, Exception]], *, skipping: bool) -> str:
    """Report every task that failed to compile, instead of only the first exception."""
    details = '\n'.join(
        f'  - {task_dir}: {type(exc).__name__}: {str(exc).replace(chr(10), " | ")}' for task_dir, exc in failures
    )
    if skipping:
        return f'No valid T3-Harbor tasks were loaded from {root}:\n{details}'
    return (
        f'Found {len(failures)} invalid T3-Harbor task(s) under {root}:\n{details}\n'
        'Fix these tasks or add skip_invalid=true to the t3-harbor URI to warn and skip them.'
    )


def _env_workers() -> int:
    """Read the default discovery concurrency from ``HARBOR_LOAD_WORKERS``."""
    raw = os.getenv('HARBOR_LOAD_WORKERS', '').strip()
    if not raw:
        return 32
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f'HARBOR_LOAD_WORKERS must be an integer, got {raw!r}') from exc
    return max(1, value)


def _validate_query_keys(uri: DatasetSourceURI) -> None:
    unexpected = sorted(set(uri.query or {}) - QUERY_KEYS)
    if unexpected:
        raise ValueError(f't3-harbor URI does not accept parameters: {unexpected}; supported: {sorted(QUERY_KEYS)}')


def _query_bool(uri: DatasetSourceURI, key: str, *, default: bool) -> bool:
    values = (uri.query or {}).get(key, ())
    if not values:
        return default
    if len(values) != 1:
        raise ValueError(f't3-harbor URI has multiple {key} values')
    raw = values[0].lower()
    if raw in {'1', 'true', 'yes', 'on'}:
        return True
    if raw in {'0', 'false', 'no', 'off'}:
        return False
    raise ValueError(f't3-harbor URI {key} must be true or false, got {values[0]!r}')


def _query_str(uri: DatasetSourceURI, key: str, *, default: str) -> str:
    values = (uri.query or {}).get(key, ())
    if not values:
        return default
    if len(values) != 1:
        raise ValueError(f't3-harbor URI has multiple {key} values')
    return values[0]


def _query_names(uri: DatasetSourceURI, key: str) -> set[str]:
    values = (uri.query or {}).get(key, ())
    return {name.strip() for value in values for name in value.split(',') if name.strip()}


def _query_int(uri: DatasetSourceURI, key: str, *, default: int, minimum: int = 0) -> int:
    values = (uri.query or {}).get(key, ())
    if not values:
        return default
    if len(values) != 1:
        raise ValueError(f't3-harbor URI has multiple {key} values')
    try:
        value = int(values[0])
    except ValueError as exc:
        raise ValueError(f't3-harbor URI {key} must be an integer, got {values[0]!r}') from exc
    if value < minimum:
        raise ValueError(f't3-harbor URI {key} must be at least {minimum}, got {value}')
    return value


def _validate_split(split: str | None) -> None:
    if split is not None and (not isinstance(split, str) or not split):
        raise DatasetReaderError('dataset_split_invalid', detail='split must be a non-empty string')
    if split not in (None, 'test'):
        raise DatasetReaderError('dataset_split_not_found', detail=f'only the test split is available, got {split!r}')


def _validate_limit(limit: int | None) -> None:
    if limit is None:
        return
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise DatasetReaderError('dataset_limit_invalid')
    if limit < 0:
        raise DatasetReaderError('dataset_limit_invalid')


__all__ = ['HarborDatasetReader', 't3_harbor_dataset_source']
