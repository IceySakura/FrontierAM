#!/usr/bin/env python3
"""SWE-Cosmos bugmaker batch generator for ``nexus run-batch``, run-part flavor.

Yields one :class:`BatchItem` per repo. Each item runs the whole bugmaker pipeline (bugmaker → collect → validate → gather → evaluate → issuegen) inside one root SWE-Cosmos sandbox, which itself spawns child sandboxes (or inner Docker containers) for validation and evaluation.

The sandbox, the pipeline knobs and the LLM backend come from run-part YAML under ``configs/`` — the format ``nexus run batch --use`` takes. This module keeps only what is genuinely per batch or per repo: repo iteration, include/exclude filters, resume, and the per-repo overlay (repo dict, input-file templates, prompt-cache partition).

Usage:
  uv run nexus run-batch -g nexus/extensions/tasks/swe_cosmos/generator.py -d ./outputs/swecosmos_bugmaker_batch.db

Configs (file stem under ``configs/<kind>/``, or a path):
  - RUNTIME_CONFIG: sandbox + pipeline knobs, e.g. swecosmos_bugmaker_runtime
  - LLM_CONFIG: backend, e.g. claude-opus-5.openai

Repos:
  - DATASET_PATH: required. Either a ``.jsonl`` of repo dicts (one per line, unique ``repo_name``) or a directory of per-repo ``.json`` files, read in sorted order.
  - INCLUDE_REPOS / EXCLUDE_REPOS: comma-separated repo_name or .json/.jsonl path. Include first; overlap is an error.
  - REDO_EXISTING: rerun repos whose ``result.json`` already looks done (default false, so a rerun resumes).
  - SWECOSMOS_SKIP / SWECOSMOS_LIMIT: skip the first N repos, then run at most N (default: 0 / no limit).
  - SPLIT: free-form label for this shard, recorded in batch metadata only.

Per-repo output lands under ``task_config.output_dir/<repo_name>/``: ``run_bugmaker.log``, ``<repo_name>.<bugmaker_strategy>_instances.jsonl`` (absent when the pipeline exits early) and ``result.json``.

The pipeline routes its two LLM stages to like-named clients. An LLM part that defines only ``main`` is fanned out into ``bugmaker`` and ``issuegen`` here; one that defines either stage directly has that entry used as-is, which is how the two stages run on different backends.
"""

from __future__ import annotations

import hashlib
import json
import os
import time

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from nexus import BatchItem
from nexus.extensions.tasks.swe_cosmos.load_config import load_config
from nexus.extensions.tasks.swe_cosmos.utils import BUGMAKER_CLIENT, ISSUEGEN_CLIENT, resolve_question_mark_path
from nexus.runner.compose.merge import deep_merge
from nexus.runner.single import RunInput, _resolve_env_suffix_config


# task_config fields the generator itself branches on; everything else is passed through untouched.
_PROBED_TASK_KEYS = ('output_dir', 'bugmaker_strategy', 'entity_path', 'bug_data_path', 'num_patches')


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    return default if value is None else value.strip().lower() in {'1', 'true', 'yes', 'y', 'on'}


def _resolve_scalars(values: dict[str, Any]) -> dict[str, Any]:
    """Resolve ``oc.env`` / ``oc.decode`` / ``oc.select`` and ``foo__env`` in a lifted dict.

    ``oc.select`` paths are relative to *values*, not the merged config.
    """
    resolved = OmegaConf.to_container(OmegaConf.create(values), resolve=True)
    if not isinstance(resolved, dict):
        raise TypeError(f'expected the probe to resolve to a mapping, got {type(resolved).__name__}')
    resolved = _resolve_env_suffix_config({str(key): value for key, value in resolved.items()}, path='probe')
    if not isinstance(resolved, dict):
        raise TypeError(f'expected the probe to remain a mapping after __env resolution, got {type(resolved).__name__}')
    return resolved


def _probe_task_config(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve the few task_config fields this module reads, leaving secrets in the config unresolved."""
    task_config = config.get('task_config') or {}
    return _resolve_scalars({key: task_config.get(key) for key in _PROBED_TASK_KEYS})


def _per_repo_cache_id(repo_name: str) -> str:
    """md5 of ``time:pid:repo_name`` — unique per repo and per shard (each shard is its own process)."""
    seed = f'{time.time()}:{os.getpid()}:{repo_name}'
    return hashlib.md5(seed.encode()).hexdigest()


def _api_key_with_cache_task_id(probe: dict[str, str], repo_name: str) -> str | None:
    """Return the api_key with a per-run ``cache_task_id`` appended, or None to leave it alone.

    The gateway partitions its prompt cache by this id, so two repos running concurrently under one key would otherwise cross-wire each other's prefix cache. Only Claude models go through a gateway that reads it; a non-Claude key must not be mangled.
    """
    api_key = probe['api_key']
    if not api_key or 'cache_task_id=' in api_key or 'claude' not in probe['model'].lower():
        return None
    separator = '&' if '?' in api_key else '?'
    return f'{api_key}{separator}cache_task_id={_per_repo_cache_id(repo_name)}'


# Models whose gateway partitions the prompt cache by the request body's ``prompt_cache_key``
_PROMPT_CACHE_KEY_MODELS = frozenset({
    'hy3',
    'hy3-offline',
    'hy4-preview',
    'hy4-preview-offline',
    'gpt-5.6-sol',
    'gpt-6-astra',
    'gpt-6-sol',
    'gpt-6.1-sol',
})


def _prompt_cache_key_for_repo(probe: dict[str, str], repo_name: str) -> str | None:
    """Return a per-run ``prompt_cache_key`` for models whose gateway caches by this field, or None."""
    model = probe['model'].lower()
    prefix = 'openai/'
    model_stem = model[len(prefix) :] if model.startswith(prefix) else model
    return _per_repo_cache_id(repo_name) if model_stem in _PROMPT_CACHE_KEY_MODELS else None


def _stage_clients(config: dict[str, Any], repo_name: str) -> tuple[dict[str, Any], str]:
    """Resolve the two stage clients the task looks up, plus a label of the models in play, in bugmaker-then-issuegen order.

    A stage uses its own ``openai_clients.<stage>`` entry whenever the LLM part defines one, which is how the two stages run on different backends: a stage needs a whole client (protocol, endpoint, key, request_args), not just a model name — swapping the model alone would leave the rest of the profile describing the previous one. Otherwise the stage is copied from ``main``, and ``None`` deletes ``main`` under run-part merge rules since the task looks its clients up by stage name alone.

    Each stage is probed on its own, so a stage pinned to its own backend gets the prompt-cache partition scheme that backend's gateway actually reads. The label does not depend on *repo_name*.
    """
    defined = config.get('openai_clients') or {}
    main = defined.get('main') or {}
    clients: dict[str, Any] = {'main': None}
    models: list[str] = []

    for stage in (BUGMAKER_CLIENT, ISSUEGEN_CLIENT):
        base = defined.get(stage) or main
        if not base:
            raise ValueError(f'LLM_CONFIG defines neither openai_clients.{stage} nor openai_clients.main.')
        client_config = base.get('config') or {}
        resolved = _resolve_scalars({
            'model': (client_config.get('request_args') or {}).get('model') or '',
            'api_key': (client_config.get('client_args') or {}).get('api_key') or '',
        })
        probe = {key: str(value or '') for key, value in resolved.items()}
        models.append(probe['model'])

        overlay: dict[str, Any] = {}
        if api_key := _api_key_with_cache_task_id(probe, repo_name):
            overlay['client_args'] = {'api_key': api_key}
        if prompt_cache_key := _prompt_cache_key_for_repo(probe, repo_name):
            overlay['request_args'] = {'prompt_cache_key': prompt_cache_key}
        # Merge even when the overlay is empty: the copy keeps the two stages from aliasing one dict.
        clients[stage] = deep_merge(base, {'config': overlay})

    # dict.fromkeys keeps the bugmaker -> issuegen order while collapsing the usual shared-model case.
    return clients, ' / '.join(dict.fromkeys(models))


def _extract_repo_names(path: Path) -> set[str]:
    """Pull repo_names out of a ``.jsonl`` (one object per line) or ``.json`` (list or object)."""

    def repo_name_of(obj: Any) -> str:
        """Extract a repo_name from a selector element: a bare string, or a dict's ``repo_name``."""
        if isinstance(obj, str):
            return obj.strip()
        if isinstance(obj, dict):
            return str(obj.get('repo_name', '') or '').strip()
        return ''

    names: set[str] = set()
    if path.suffix.lower() == '.jsonl':
        with path.open('r', encoding='utf-8') as f:
            for line_no, raw in enumerate(f, start=1):
                if not (line := raw.strip()):
                    continue
                if not (repo_name := repo_name_of(json.loads(line))):
                    raise ValueError(f'{path}:{line_no} has no resolvable "repo_name".')
                names.add(repo_name)
        return names
    data = json.loads(path.read_text(encoding='utf-8'))
    for idx, obj in enumerate(data if isinstance(data, list) else [data]):
        if not (repo_name := repo_name_of(obj)):
            raise ValueError(f'{path}[{idx}] has no resolvable "repo_name".')
        names.add(repo_name)
    return names


def _selected_repo_filters() -> tuple[set[str], set[str], int, int]:
    """Resolve INCLUDE_REPOS / EXCLUDE_REPOS / SWECOSMOS_SKIP / SWECOSMOS_LIMIT, rejecting any repo_name claimed by both lists.

    ``skip`` drops the first *N* repos and ``limit`` caps how many are submitted after that; 0 means unset for both. Both are applied after every other filter, so they walk the repos still outstanding rather than spending the window on ones already done.
    """
    selectors: list[set[str]] = []
    for name in ('INCLUDE_REPOS', 'EXCLUDE_REPOS'):
        names: set[str] = set()
        for item in (x.strip() for x in os.getenv(name, '').split(',')):
            if not item:
                continue
            if item.lower().endswith(('.json', '.jsonl')):
                path = Path(item)
                if not path.is_file():
                    raise ValueError(f'{name}: selector file not found: {item}')
                names.update(_extract_repo_names(path))
            else:
                names.add(item)
        selectors.append(names)
    include, exclude = selectors
    if conflict := include & exclude:
        raise ValueError(f'repo_name(s) in both INCLUDE_REPOS and EXCLUDE_REPOS: {sorted(conflict)}')

    # Unset is 0 for both. LIMIT has a floor of 1 so that an explicit 0 is an error rather than a coin flip between "run nothing" and "no limit".
    bounds: list[int] = []
    for name, minimum in (('SWECOSMOS_SKIP', 0), ('SWECOSMOS_LIMIT', 1)):
        if not (raw := os.getenv(name, '').strip()):
            bounds.append(0)
            continue
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValueError(f'{name} must be an integer, got {raw!r}') from exc
        if value < minimum:
            raise ValueError(f'{name} must be >= {minimum}, got {value}')
        bounds.append(value)
    skip, limit = bounds
    return include, exclude, skip, limit


def _result_is_done(repo_out_dir: Path) -> bool:
    """True when the repo's ``result.json`` shows it need not be re-run.

    Done means either it already issued at least one instance (prior output is kept regardless of status), or it finished cleanly with at least one bugmaker candidate and no ``Gongfeng API error`` in ``run_bugmaker.log``. A missing log does not disqualify an otherwise-finished result.
    """
    try:
        data = json.loads((repo_out_dir / 'result.json').read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(data, dict):
        return False
    if (data.get('issued') or 0) > 0:
        return True
    if data.get('status') != 'Finished' or data.get('return_code') != 0 or (data.get('unvalidated') or 0) <= 0:
        return False
    try:
        log_text = (repo_out_dir / 'run_bugmaker.log').read_text(encoding='utf-8', errors='replace')
    except OSError:
        return True
    return 'Gongfeng API error' not in log_text


def _resolved_input_path(template: str, repo_name: str, strategy: str) -> str | None:
    """Expand a per-repo input template into a real path, or None when nothing matches.

    The template takes ``{repo_name}`` / ``{bugmaker_strategy}`` and may carry one ``/?/`` wildcard segment when the task-split directory is unknown (see :func:`resolve_question_mark_path`).
    """
    try:
        formatted = template.format(repo_name=repo_name, bugmaker_strategy=strategy)
    except KeyError as exc:
        raise ValueError(
            f'unknown placeholder {exc} in {template!r}; only {{repo_name}} and {{bugmaker_strategy}} are available'
        ) from exc
    resolved = resolve_question_mark_path(formatted)
    return resolved if resolved and Path(resolved).is_file() else None


def _iter_repo_dicts() -> Iterator[tuple[str, dict[str, Any], str | None]]:
    """Yield ``(repo_name, repo_dict, host_json_path)`` from DATASET_PATH, rejecting duplicate repo_names.

    ``host_json_path`` is the repo dict's own ``.json`` file, or None for a ``.jsonl`` row (which :func:`generate_configs` then materializes, since the task uploads the dict by host path).
    """
    dataset_path = os.getenv('DATASET_PATH', '').strip()
    if not dataset_path:
        raise ValueError('DATASET_PATH is required: a .jsonl of repo dicts, or a directory of per-repo .json files')
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f'DATASET_PATH not found: {dataset_path}')

    rows: Iterator[tuple[str, dict[str, Any], str | None]]
    if path.is_file():
        rows = (
            (f'{path}:{line_no}', json.loads(line), None)
            for line_no, raw in enumerate(path.read_text(encoding='utf-8').splitlines(), start=1)
            if (line := raw.strip())
        )
    else:
        rows = ((str(f), json.loads(f.read_text(encoding='utf-8')), str(f)) for f in sorted(path.glob('*.json')))

    seen: set[str] = set()
    for origin, repo_dict, host_json_path in rows:
        repo_name = str(repo_dict.get('repo_name', '') or '').strip()
        if not repo_name:
            raise ValueError(f'{origin} is missing the required "repo_name" key.')
        if repo_name in seen:
            raise ValueError(f'Duplicate repo_name "{repo_name}" in {dataset_path} (repo_name must be unique).')
        seen.add(repo_name)
        yield repo_name, repo_dict, host_json_path


def _iter_selected_repos(settings: dict[str, Any]) -> Iterator[tuple[str, dict[str, Any], str | None, str | None]]:
    """Yield ``(repo_name, repo_dict, host_json_path, bug_data_path)`` for every repo that survives the filters.

    A repo is dropped when the include/exclude lists say so, when a completed ``result.json`` already covers it (unless ``REDO_EXISTING``), or — under ``function_compose`` — when its bug-data file is missing or holds fewer than ``num_patches`` atomic bugs to compose. ``SWECOSMOS_SKIP`` / ``SWECOSMOS_LIMIT`` then window the survivors.
    """
    include, exclude, skip, limit = _selected_repo_filters()
    output_dir = Path(settings['output_dir'])
    redo_existing = _env_bool('REDO_EXISTING')
    strategy = settings['bugmaker_strategy']
    bug_data_template = settings['bug_data_path'] or ''
    if strategy == 'function_compose' and not bug_data_template:
        raise ValueError('SWECOSMOS_BUG_DATA_PATH is required for bugmaker_strategy=function_compose')

    skipped = 0
    selected = 0
    for repo_name, repo_dict, host_json_path in _iter_repo_dicts():
        if (include and repo_name not in include) or repo_name in exclude:
            continue
        if not redo_existing and _result_is_done(output_dir / repo_name):
            continue
        bug_data_path = _resolved_input_path(bug_data_template, repo_name, strategy) if bug_data_template else None
        if strategy == 'function_compose':
            if bug_data_path is None:
                continue
            with open(bug_data_path, encoding='utf-8') as f:
                if sum(1 for line in f if line.strip()) < int(settings['num_patches']):
                    continue
        if skipped < skip:
            skipped += 1
            continue
        yield repo_name, repo_dict, host_json_path, bug_data_path
        selected += 1
        if limit and selected >= limit:
            return


def _load_parts() -> dict[str, Any]:
    """Fold the two run parts into one config: runtime, then LLM.

    The LLM profiles are shared with the SWE-Cosmos agent task and carry an ``agent_config`` block for the scaffold there. This pipeline has no scaffold, so that block is dropped; ``None`` deletes a key under run-part merge rules.
    """
    return deep_merge(load_config(), {'agent_config': None})


def generate_configs() -> Iterator[BatchItem]:
    # The run parts do not vary by repo; only the overlays below do.
    base = _load_parts()
    settings = _probe_task_config(base)
    strategy = settings['bugmaker_strategy']
    output_dir = Path(settings['output_dir'])
    entity_template = settings['entity_path'] or ''

    for repo_name, repo_dict, host_json_path, bug_data_path in _iter_selected_repos(settings):
        repo_out_dir = output_dir / repo_name
        repo_out_dir.mkdir(parents=True, exist_ok=True)
        if host_json_path is None:
            # The task uploads the repo dict by host path, so a .jsonl row needs a file of its own.
            host_json_path = str(repo_out_dir / 'repo_dict.json')
            Path(host_json_path).write_text(json.dumps(repo_dict, ensure_ascii=False, indent=2), encoding='utf-8')

        entity_path = _resolved_input_path(entity_template, repo_name, strategy) if entity_template else None
        task_config = {
            'repo_name': repo_name,
            'repo_dict_path': host_json_path,
            # Templates are per-repo: hand the task concrete paths, and blank out the ones that match nothing.
            'entity_path': entity_path or '',
            'bug_data_path': bug_data_path or '',
        }

        clients, model_label = _stage_clients(base, repo_name)
        repo_config = deep_merge(base, {'task_config': task_config, 'openai_clients': clients})
        yield BatchItem(
            config=RunInput.from_dict(repo_config),
            run_id=repo_name,
            metadata={
                'repo_name': repo_name,
                'bugmaker_strategy': strategy,
                'model': model_label,
                'split': os.getenv('SPLIT', ''),
                'output_dir': str(repo_out_dir),
            },
        )


def get_total_count() -> int:
    return sum(1 for _ in _iter_selected_repos(_probe_task_config(_load_parts())))


def get_batch_metadata() -> dict[str, Any]:
    base = _load_parts()
    settings = _probe_task_config(base)
    include, exclude, skip, limit = _selected_repo_filters()
    _, model_label = _stage_clients(base, repo_name='')
    return {
        'task_name': base.get('task_name', 'swecosmos_bugmaker_pipeline'),
        'dataset_path': os.getenv('DATASET_PATH', ''),
        'split': os.getenv('SPLIT', ''),
        'output_dir': settings['output_dir'],
        'bugmaker_strategy': settings['bugmaker_strategy'],
        'model': model_label,
        'include_repo_names': sorted(include) or None,
        'exclude_repo_names': sorted(exclude) or None,
        'skip': skip,
        'limit': limit,
    }
