#!/usr/bin/env python3
"""Build a publishable Nexus dataset from the OSWorld task set.

Bridges two models that do not line up on their own:

* OSWorld keeps its fixtures in one flat host-side cache, shared by every task.
* ``nexus dataset`` is per-item -- ``publisher._record_for_publish`` zips each
  item's ``files/`` directory into ``<split>/blobs/<item.id>.zip``, and that zip
  is what ``RunInput.task_files`` resolves to at run time.

So one dataset item == one OSWorld task, and each item carries exactly the
fixtures that task needs::

    items/<task_id>/
      item.yaml                            # body -> task_name/task_config
      files/task.json
      files/setup/<uuid5(url)>_<name>      # setup `download` assets
      files/evaluator/<dest>               # evaluator `cloud_file` reference files
      files/evaluator/<uuid5(url)>_<name>  # evaluator `postconfig` download assets

Those two subdirectory names are not arbitrary: they are ``prefetch.SETUP_DIR``
and ``prefetch.EVALUATOR_DIR``, which ``run_os_world`` downloads back into
``resolve_setup_cache_dir()`` / ``resolve_evaluator_cache_dir()``.
Shared constants keep package construction and prefetch paths consistent.

**Dataset construction requires an OSWorld checkout.** Evaluation entrypoints
enumerate the published dataset via ``task_source``. Packaging uses
``OSWORLD_EXAMPLES_ROOT`` and selection filters to build the requested subset.

Files are hard-linked out of the existing cache rather than copied, so building
the tree does not duplicate the ~600 MB already on disk. Hard links are safe here
because both the cache and the dataset directory are only ever read, and publish
reads the bytes through a fresh zip.

Usage::

    python -m nexus.extensions.tasks.os_world.scripts.build_osworld_dataset \\
        --out /data/workspace/datasets/osworld --fetch

Then publish (needs a Wedata key with subject/key/id -- see nexus/dataset/auth.py)::

    nexus dataset publish /data/workspace/datasets/osworld \\
        <namespace>/<name>@<version> --admin <subject>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

from pathlib import Path
from typing import Any

import yaml

from nexus.extensions.tasks.os_world.data import task_source
from nexus.extensions.tasks.os_world.data.paths import EVALUATOR_SUBDIR, cache_root
from nexus.extensions.tasks.os_world.data.prefetch import EVALUATOR_DIR, SETUP_DIR, TASK_JSON_NAME
from nexus.extensions.tasks.os_world.execution.setup import _download_cache_path


DEFAULT_SPLIT = 'test'


def _iter_typed_files(node: Any, wanted: str) -> list[tuple[str, str]]:
    """Collect every ``(path, dest)`` pair of getter ``type: <wanted>`` in a tree.

    ``cloud_file`` and ``local_file`` entries are nested arbitrarily deep inside
    ``expected`` / ``result`` / ``options``, and ``multi: true`` turns
    ``path``/``dest`` into parallel lists, so this walks the whole tree instead
    of matching a shape.
    """
    found: list[tuple[str, str]] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            if value.get('type') == wanted:
                raw_path, raw_dest = value.get('path'), value.get('dest')
                paths = raw_path if isinstance(raw_path, list) else [raw_path]
                dests = raw_dest if isinstance(raw_dest, list) else [raw_dest]
                for path, dest in zip(paths, dests, strict=False):
                    if path and dest:
                        found.append((str(path), str(dest)))
            for nested in value.values():
                walk(nested)
        elif isinstance(value, list):
            for nested in value:
                walk(nested)

    walk(node)
    return found


def _iter_postconfig_downloads(evaluator: Any) -> list[tuple[str, str]]:
    """Collect every ``(url, path)`` an evaluator ``postconfig`` download fetches.

    The fourth asset class, and the one that was missed until eval task 184262:
    ``evaluator.postconfig`` re-stages fixtures the grading step itself needs --
    ``eval.sh``, ``check_password.sh``, ``*_gold.xlsx`` -- after the agent has
    finished. They were not packaged, so every one of them reached Hugging Face
    from inside the runtime VPC at grading time and 15 of 361 cases were voided
    on ``evaluator postconfig downloads failed``.

    Keyed on ``(url, path)`` rather than ``(url, dest)`` like the ``cloud_file``
    pairs because these are *setup*-shaped actions, not getters: the runtime
    resolves them through ``setup._download_cache_path``, which names the cache
    entry ``uuid5(url)_<basename(path)>``. ``path`` is the in-VM destination, so
    its basename is what the name must be built from.

    Only the top level of ``postconfig`` is walked, because that is the only
    place the evaluator runs actions from (``evaluator._run_postconfig`` iterates
    the list it is handed); a recursive walk would invent assets from
    ``expected``/``result`` subtrees that nothing stages.
    """
    if not isinstance(evaluator, dict):
        return []
    found: list[tuple[str, str]] = []
    for action in evaluator.get('postconfig') or []:
        if not isinstance(action, dict) or action.get('type') != 'download':
            continue
        parameters = action.get('parameters')
        if not isinstance(parameters, dict):
            continue
        for file_spec in parameters.get('files') or []:
            if not isinstance(file_spec, dict):
                continue
            url, path = file_spec.get('url'), file_spec.get('path')
            if url and path:
                found.append((str(url), str(path)))
    return found


def _selected_tasks() -> list[tuple[str, str, Path]]:
    """Packaging alone reads an explicit source checkout; it never runs agents."""
    value = os.getenv('OSWORLD_EXAMPLES_ROOT', '').strip()
    if not value:
        raise ValueError('OSWORLD_EXAMPLES_ROOT is required on the dataset packaging machine')
    root = Path(value)
    meta_path = task_source.test_meta_path() or root.parent / 'test_nogdrive.json'
    domains = task_source.env_list('OSWORLD_DOMAIN', 'all')
    ids = task_source.env_set('OSWORLD_TASK_IDS')
    selected = []
    for domain, task_ids in task_source._load_test_meta(meta_path).items():
        if domains and domain not in domains:
            continue
        for task_id in task_ids:
            path = root / domain / f'{task_id}.json'
            if (ids is None or task_id in ids) and path.is_file():
                selected.append((domain, task_id, path))
    skip = max(task_source.env_int('OSWORLD_SKIP', 0) or 0, 0)
    limit = task_source.env_limit('OSWORLD_LIMIT')
    return selected[skip:] if limit is None else selected[skip : skip + limit]


async def _fetch_missing(
    setup_wanted: dict[Path, tuple[str, str]],
    evaluator_wanted: dict[Path, str],
    postconfig_wanted: dict[Path, tuple[str, str]],
) -> tuple[int, list[str]]:
    """Download whatever is not on disk yet. Returns (fetched, failures)."""
    from nexus.extensions.tasks.os_world.execution.setup import _ensure_cached_download
    from nexus.extensions.tasks.os_world.getters.runtime import download_url

    fetched = 0
    failures: list[str] = []

    for target, (url, path) in setup_wanted.items():
        if target.exists():
            continue
        try:
            await _ensure_cached_download(url, path, target.parent)
            fetched += 1
        except Exception as exc:  # noqa: BLE001 - report and continue; one bad URL must not abort the build
            failures.append(f'setup {target.name}: {type(exc).__name__}: {exc}')

    for target, url in evaluator_wanted.items():
        if target.exists():
            continue
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            await download_url(url, target)
            fetched += 1
        except Exception as exc:  # noqa: BLE001
            failures.append(f'evaluator {target.name}: {type(exc).__name__}: {exc}')

    # Fetched through the runtime's own helper so the name it writes is the name
    # the runtime will later look for -- _ensure_cached_download derives
    # uuid5(url)_<basename(path)> from (url, path) inside the dir it is handed,
    # which is exactly how the plan phase built `target`.
    for target, (url, path) in postconfig_wanted.items():
        if target.exists():
            continue
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            await _ensure_cached_download(url, path, target.parent)
            fetched += 1
        except Exception as exc:  # noqa: BLE001
            failures.append(f'postconfig {target.name}: {type(exc).__name__}: {exc}')

    return fetched, failures


def _resolve_repo_path(examples_root: Path, repo_path: str) -> Path:
    """Resolve a ``local_file`` path to disk.

    Mirrors the candidate order in ``getters/file.py:_resolve_local_file_source``:
    ``examples_root`` is ``<repo>/evaluation_examples/examples`` and local_file
    paths are normally written relative to ``<repo>``, but some are written with
    the ``evaluation_examples/examples`` prefix already stripped. Returns the
    first candidate that exists, else the repo-relative one so the caller reports
    a sensible missing path.
    """
    path = Path(repo_path)
    repo_root = examples_root.parent.parent
    candidates = [repo_root / path]
    if path.parts[:2] == ('evaluation_examples', 'examples'):
        candidates.append(examples_root / Path(*path.parts[2:]))
    else:
        candidates.append(examples_root / path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def _link(source: Path, target: Path) -> None:
    """Hard-link ``source`` to ``target``, falling back to a copy across devices."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    try:
        os.link(source, target)
    except OSError:
        import shutil

        shutil.copy2(source, target)


def build(args: argparse.Namespace) -> int:
    examples_root = Path(os.environ['OSWORLD_EXAMPLES_ROOT']).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    setup_cache = cache_root(args.cache_dir)
    evaluator_cache = setup_cache / EVALUATOR_SUBDIR

    if not examples_root.is_dir():
        print(f'error: examples root not found: {examples_root}', file=sys.stderr)
        return 2

    tasks = _selected_tasks()
    if not tasks:
        print(f'error: no tasks selected from {task_source.test_meta_path()}', file=sys.stderr)
        return 2

    # Plan first, fetch second, assemble third -- so a partial network failure
    # leaves a reportable gap instead of a half-built tree.
    plan: list[dict[str, Any]] = []
    setup_wanted: dict[Path, tuple[str, str]] = {}
    evaluator_wanted: dict[Path, str] = {}
    # Separate from evaluator_wanted despite landing in the same tree: these are
    # fetched through _ensure_cached_download (retries + atomic write), the same
    # helper the runtime would have used, while cloud_file assets go through
    # download_url. Keeping them apart keeps each fetch path honest.
    postconfig_wanted: dict[Path, tuple[str, str]] = {}

    for domain, task_id, task_file in tasks:
        data = json.loads(task_file.read_text(encoding='utf-8'))
        item_id = str(data.get('id') or task_file.stem)
        evaluator = data.get('evaluator') or {}

        setup_assets: list[tuple[Path, str]] = []
        for step in data.get('config') or []:
            if step.get('type') != 'download':
                continue
            for file_spec in (step.get('parameters') or {}).get('files') or []:
                url, path = file_spec.get('url'), str(file_spec.get('path', ''))
                if not url:
                    continue
                source = _download_cache_path(setup_cache, str(url), path)
                setup_wanted[source] = (str(url), path)
                setup_assets.append((source, source.name))

        evaluator_assets: list[tuple[Path, str]] = []
        for url, dest in _iter_typed_files(evaluator, 'cloud_file'):
            source = evaluator_cache / item_id / dest
            evaluator_wanted[source] = url
            evaluator_assets.append((source, dest))

        # postconfig downloads share the evaluator/** tree because that is the
        # cache the evaluator hands them: _run_postconfig_action passes the
        # per-task evaluator cache_dir into run_setup_action, so the runtime
        # looks for uuid5(url)_<basename(path)> *there*, not in the flat setup
        # cache that config-phase downloads use.
        for url, path in _iter_postconfig_downloads(evaluator):
            source = _download_cache_path(evaluator_cache / item_id, url, path)
            postconfig_wanted[source] = (url, path)
            evaluator_assets.append((source, source.name))

        # local_file reference files come from the OSWorld checkout, not a URL.
        # Resolve them against examples_root now so the platform never needs it.
        local_assets = [
            (_resolve_repo_path(examples_root, repo_path), dest)
            for repo_path, dest in _iter_typed_files(evaluator, 'local_file')
        ]

        plan.append({
            'item_id': item_id,
            'domain': domain,
            'task_id': task_id,
            'instruction': str(data.get('instruction') or ''),
            'task_json': task_file,
            'setup': setup_assets,
            'evaluator': evaluator_assets,
            'local': local_assets,
        })

    missing_setup = sum(1 for p in setup_wanted if not p.exists())
    missing_eval = sum(1 for p in evaluator_wanted if not p.exists())
    missing_postconfig = sum(1 for p in postconfig_wanted if not p.exists())
    print(f'tasks: {len(plan)}')
    print(f'setup assets:      {len(setup_wanted)} ({missing_setup} missing)')
    print(f'evaluator assets:  {len(evaluator_wanted)} ({missing_eval} missing)')
    print(f'postconfig assets: {len(postconfig_wanted)} ({missing_postconfig} missing)')

    if args.fetch and (missing_setup or missing_eval or missing_postconfig):
        print('fetching missing assets ...')
        fetched, failures = asyncio.run(_fetch_missing(setup_wanted, evaluator_wanted, postconfig_wanted))
        print(f'fetched {fetched}; {len(failures)} failed')
        for line in failures[: args.report_limit]:
            print(f'  {line}')
        if len(failures) > args.report_limit:
            print(f'  ... and {len(failures) - args.report_limit} more')

    if args.plan_only:
        return 0

    items_root = out_dir / 'items'
    items_root.mkdir(parents=True, exist_ok=True)

    written = 0
    incomplete: list[str] = []
    item_paths: list[str] = []

    for entry in plan:
        item_id = entry['item_id']
        item_dir = items_root / item_id
        files_dir = item_dir / 'files'

        gaps: list[str] = []
        for source, name in entry['setup']:
            if source.exists():
                _link(source, files_dir / SETUP_DIR / name)
            else:
                gaps.append(f'{SETUP_DIR}/{name}')
        for source, dest in entry['evaluator']:
            if source.exists():
                _link(source, files_dir / EVALUATOR_DIR / dest)
            else:
                gaps.append(f'{EVALUATOR_DIR}/{dest}')
        # local_file gold shares the evaluator/ tree: run_os_world downloads
        # evaluator/** into the same per-task cache get_cloud_file reads from, and
        # get_local_file resolves its dest there too.
        for source, dest in entry['local']:
            if source.is_file():
                _link(source, files_dir / EVALUATOR_DIR / dest)
            else:
                gaps.append(f'{EVALUATOR_DIR}/{dest} (local_file)')

        if gaps:
            incomplete.append(f'{item_id}: {", ".join(gaps[:3])}{" ..." if len(gaps) > 3 else ""}')
            if args.skip_incomplete:
                continue

        # task.json makes the item self-contained; without it a platform run
        # needs the OSWorld checkout at the exact task_root the config carries.
        files_dir.mkdir(parents=True, exist_ok=True)
        _link(entry['task_json'], files_dir / TASK_JSON_NAME)

        item_dir.joinpath('item.yaml').write_text(
            yaml.safe_dump(
                {
                    'id': item_id,
                    'files': {'type': 'directory', 'path': 'files'},
                    'body': {
                        'task_name': 'os_world',
                        # No task_root/task_file: run_os_world loads the packaged
                        # files/task.json. task_id is kept for cache-dir naming
                        # and trajectory labels.
                        'task_config': {
                            'domain': entry['domain'],
                            'task_id': entry['task_id'],
                        },
                    },
                    # domain and instruction are what task_source enumerates the
                    # published manifest by, so a generating machine needs no
                    # checkout. Dropping either sends it back to needing one.
                    'metadata': {'domain': entry['domain'], 'instruction': entry['instruction']},
                },
                sort_keys=False,
                allow_unicode=True,
            ),
            encoding='utf-8',
        )
        item_paths.append(f'items/{item_id}/item.yaml')
        written += 1

    out_dir.joinpath('dataset.yaml').write_text(
        yaml.safe_dump(
            {
                'schema_version': 1,
                'name': args.name,
                'default_split': args.split,
                'splits': {args.split: {'items': item_paths}},
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding='utf-8',
    )

    print(f'wrote {written} items to {out_dir}')
    if incomplete:
        verb = 'skipped' if args.skip_incomplete else 'written with gaps'
        print(f'{len(incomplete)} items {verb} (missing assets):')
        for line in incomplete[: args.report_limit]:
            print(f'  {line}')
        if len(incomplete) > args.report_limit:
            print(f'  ... and {len(incomplete) - args.report_limit} more')
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', required=True, help='Dataset output directory')
    parser.add_argument('--cache-dir', default=None, help='Host cache root; default resolves like the task does')
    parser.add_argument('--name', default='osworld', help='Dataset name written to dataset.yaml')
    parser.add_argument('--split', default=DEFAULT_SPLIT, help='Split name')
    parser.add_argument('--fetch', action='store_true', help='Download assets that are not cached yet')
    parser.add_argument('--plan-only', action='store_true', help='Report coverage and exit without writing')
    parser.add_argument(
        '--skip-incomplete',
        action='store_true',
        help='Omit items whose assets could not all be resolved instead of publishing them with gaps',
    )
    parser.add_argument('--report-limit', type=int, default=10, help='Max lines per problem report')
    args = parser.parse_args(argv)

    # The checkout and the id list come from the same two env vars every other
    # entry point reads, through the same resolver -- so a dataset is built from
    # the selection its own OSWORLD_* knobs describe.
    if not os.environ.get('OSWORLD_EXAMPLES_ROOT', '').strip():
        parser.error('OSWORLD_EXAMPLES_ROOT is required: this is the one machine that needs a checkout')
    return build(args)


if __name__ == '__main__':
    raise SystemExit(main())
