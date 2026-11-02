"""Materialise dataset-carried OSWorld fixtures into host caches.

The packaging script and this module share the task_files layout constants.
Authentication is provided by nexus.dataset.auth; task JSON and resource trees
are resolved from the attached dataset item.
"""

from __future__ import annotations

import logging

from pathlib import Path
from typing import Any

from nexus.core import require_context
from nexus.dataset.auth import apply_wedata_key
from nexus.dataset.exceptions import DatasetAuthenticationError, DatasetConfigurationError


logger = logging.getLogger(__name__)

__all__ = [
    'EVALUATOR_DIR',
    'SETUP_DIR',
    'TASK_FILES_EVALUATOR_PATTERN',
    'TASK_FILES_SETUP_PATTERN',
    'TASK_JSON_NAME',
    'apply_wedata_key',
    'prefetch_task_files',
    'prefetch_task_json',
]

# Layout inside a dataset item's files/ dir. The packaging script writes these
# directories and the prefetch below reads them back, so the two names live here
# once and both sides import them.
SETUP_DIR = 'setup'
EVALUATOR_DIR = 'evaluator'
TASK_JSON_NAME = 'task.json'

TASK_FILES_SETUP_PATTERN = f'{SETUP_DIR}/**'
TASK_FILES_EVALUATOR_PATTERN = f'{EVALUATOR_DIR}/**'


def _attached_task_files() -> Any | None:
    """The run's ``task_files``, or None when there is nothing to prefetch.

    ``require_context()`` raises outside a run. ``run_os_world`` always has a
    context, but ``load_os_world_task`` / ``run_setup_actions`` are also driven
    directly (unit tests, and any caller that orchestrates setup itself), and a
    best-effort cache prime must never be the reason such a call fails -- the
    whole point of these helpers is that everything downstream still downloads
    normally when the dataset carries nothing.
    """
    try:
        ctx = require_context()
    except Exception:  # noqa: BLE001 - no active run: nothing is attached
        return None
    return getattr(ctx, 'task_files', None)


def _declared_assets(task_data: dict[str, Any]) -> dict[str, set[str]]:
    """Names of packaged download/cloud_file/local_file assets declared by task.json."""
    from nexus.extensions.tasks.os_world.execution.setup import _download_cache_path

    expected: dict[str, set[str]] = {SETUP_DIR: set(), EVALUATOR_DIR: set()}

    def downloads(actions: list, tree: str) -> None:
        for action in actions:
            if not isinstance(action, dict) or action.get('type') != 'download':
                continue
            for file in (action.get('parameters') or {}).get('files') or []:
                if file.get('url'):
                    name = _download_cache_path(Path('.'), str(file['url']), str(file.get('path', ''))).name
                    expected[tree].add(f'{tree}/{name}')

    def references(node: Any) -> None:
        if isinstance(node, dict):
            # Nested evaluator rules also use "type", sometimes as a mapping
            # (e.g. Calc data-validation comparisons), not a getter name.
            node_type = node.get('type')
            if isinstance(node_type, str) and node_type in {'cloud_file', 'local_file'}:
                paths, dests = node.get('path'), node.get('dest')
                paths = paths if isinstance(paths, list) else [paths]
                dests = dests if isinstance(dests, list) else [dests]
                for path, dest in zip(paths, dests, strict=False):
                    if path and dest:
                        expected[EVALUATOR_DIR].add(f'{EVALUATOR_DIR}/{dest}')
            for value in node.values():
                references(value)
        elif isinstance(node, list):
            for value in node:
                references(value)

    evaluator = task_data.get('evaluator') or {}
    downloads(task_data.get('config') or [], SETUP_DIR)
    downloads(evaluator.get('postconfig') or [], EVALUATOR_DIR)
    references(evaluator)
    return expected


def _log_prefetch_error(pattern: str, stage: str, exc: Exception) -> None:
    status = getattr(exc, 'http_status', None)
    category = (
        'authentication'
        if isinstance(exc, (DatasetAuthenticationError, DatasetConfigurationError)) or status in {401, 403}
        else 'io'
    )
    # Exception text may contain signed URLs or credentials.
    logger.warning(
        'OSWorld asset prefetch status=failed pattern=%s stage=%s category=%s error_type=%s; '
        'prefetch incomplete, downstream asset resolution has not run',
        pattern,
        stage,
        category,
        type(exc).__name__,
    )


async def prefetch_task_files(
    setup_cache_dir: Path,
    evaluator_cache_dir: Path,
    *,
    task_data: dict[str, Any] | None = None,
) -> None:
    """Materialise dataset-carried fixtures into the two host caches.

    Lets one packaged dataset serve both offline runs (``task_files.type:
    directory``) and platform runs (``type: zip`` + ``remote``), instead of every
    task reaching Hugging Face at setup and grading time.

    Compare packaged paths with the download/reference declarations in task.json.
    Empty optional trees are informational; missing declared files and list/read
    failures are warnings. Keep the existing best-effort behaviour, but never
    claim a downstream download happened before asset resolution actually runs.
    """
    task_files = _attached_task_files()
    if task_files is None:
        logger.debug('OSWorld asset prefetch status=not_attached; no task_files configured')
        return

    expected = _declared_assets(task_data) if task_data is not None else None
    for pattern, target in (
        (TASK_FILES_SETUP_PATTERN, setup_cache_dir),
        (TASK_FILES_EVALUATOR_PATTERN, evaluator_cache_dir),
    ):
        tree = pattern.split('/')[0]
        try:
            matches = set(await task_files.list(pattern))
        except Exception as exc:
            _log_prefetch_error(pattern, 'list', exc)
            continue
        missing = expected[tree] - matches if expected is not None else set()
        if missing:
            logger.warning(
                'OSWorld asset prefetch status=missing_declared pattern=%s missing=%d packaged=%d; '
                'task.json declares assets absent from task_files; downstream asset resolution has not run',
                pattern,
                len(missing),
                len(matches),
            )
        if not matches:
            if not missing:
                logger.info(
                    'OSWorld asset prefetch status=empty pattern=%s declared=%s packaged=0; '
                    'no matching attachments to prefetch (task_files may still contain task.json)',
                    pattern,
                    0 if expected is not None else 'unknown',
                )
            continue
        try:
            target.mkdir(parents=True, exist_ok=True)
            await task_files.download(pattern, str(target))
        except Exception as exc:
            _log_prefetch_error(pattern, 'download', exc)
        else:
            logger.info(
                'OSWorld asset prefetch status=downloaded pattern=%s files=%d missing_declared=%s',
                pattern,
                len(matches),
                len(missing) if expected is not None else 'unknown',
            )


async def prefetch_task_json(staging_dir: Path) -> Path | None:
    """Pull the packaged ``task.json`` out of the dataset, if one is attached.

    Returns None only when no task_files are attached. Packaging the task JSON is
    what lets an item be self-contained: without it a platform run would still
    need the OSWorld checkout at the exact ``task_root`` baked into the config,
    which is the coupling the dataset exists to break. A plain local run (no
    ``task_files``) gets None and loads from ``task_root`` as before.
    """
    task_files = _attached_task_files()
    if task_files is None:
        return None
    target = staging_dir / TASK_JSON_NAME
    try:
        staging_dir.mkdir(parents=True, exist_ok=True)
        await task_files.download(TASK_JSON_NAME, str(target))
    except Exception as exc:
        # Dataset-backed task.json is authoritative, not a cache optimisation.
        # Do not hide authentication failures behind an unrelated task_root error.
        raise RuntimeError(
            f'OSWorld could not load task_files/task.json ({type(exc).__name__}). '
            'Check WEDATA_KEY JSON, dataset access and the attached ZIP; no task_root fallback was used.'
        ) from None
    if not target.is_file():
        raise FileNotFoundError('OSWorld task_files/task.json was not materialised; no task_root fallback was used')
    logger.info('OSWorld task definition status=downloaded source=task_files/task.json')
    return target
