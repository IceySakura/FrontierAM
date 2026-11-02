#!/usr/bin/env python3
"""SWE-Cosmos batch generator for ``nexus run-batch``, run-part flavor.

The agent / LLM / runtime blocks come from run-part YAML under ``configs/`` instead of a wall of ``os.environ.setdefault`` defaults. Each file is the format ``nexus run batch --use`` takes (``body`` / ``merge`` / ``source``), so this module folds three of them into one config and keeps only what is genuinely per batch or per instance: dataset iteration, include/exclude filters, paging, the inlined instance, and the two things that cannot be known until the instance is — its cache partition and any resource override.

Usage:
  uv run nexus run-batch -g nexus/extensions/tasks/swe/swe_cosmos/generator.py -d ./outputs/swecosmos_batch.db

Configs (file stem under ``configs/<kind>/``, or a path):
  - RUNTIME_CONFIG: sandbox, e.g. swecosmos_runtime
  - AGENT_CONFIG: scaffold, e.g. claude_code / openclaw / codex
  - LLM_CONFIG: backend, e.g. claude-opus-5.openai

Dataset:
  - DATASET_PATH: required local .jsonl
  - INCLUDE_INSTANCES / EXCLUDE_INSTANCES: comma-separated instance_id or .json/.jsonl path. Include first; overlap is an error.
  - SWE_SKIP / SWE_LIMIT: skip first N, then at most N (default: 0 / no limit)

``RUN_ORACLE`` is in the runtime profile. When set, this module drops ``agent_name``, ``agent_config`` and ``openai_clients``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from nexus import BatchItem
from nexus.extensions.tasks.swe.constants import DATASET_SWE_BENCH, SWEbenchInstance
from nexus.extensions.tasks.swe.swe_cosmos.load_config import load_config
from nexus.runner.compose.merge import deep_merge
from nexus.runner.single import RunInput, _resolve_env_suffix_config


logger = logging.getLogger(__name__)

# Instances whose resource_spec the runtime profile's default does not cover.
INSTANCE_RESOURCE_OVERRIDE = {
    # Heavy numeric workload, CPU bound.
    'scikit-learn__scikit-learn-14710': {'cpu': 16, 'memory': 16},
}


def _env_int(name: str, default: int | None = None) -> int | None:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _resolve_scalars(values: dict[str, Any]) -> dict[str, Any]:
    """Resolve ``oc.env`` / ``oc.decode`` / ``oc.select`` and ``foo__env`` in a lifted dict.

    ``oc.select`` paths are relative to *values*, not the merged config or ``body``.
    """
    resolved = OmegaConf.to_container(OmegaConf.create(values), resolve=True)
    if not isinstance(resolved, dict):
        raise TypeError(f'expected the probe to resolve to a mapping, got {type(resolved).__name__}')
    resolved = _resolve_env_suffix_config({str(key): value for key, value in resolved.items()}, path='probe')
    if not isinstance(resolved, dict):
        raise TypeError(f'expected the probe to remain a mapping after __env resolution, got {type(resolved).__name__}')
    return resolved


def _is_oracle_run(config: dict[str, Any]) -> bool:
    """Read ``task_config.run_oracle``, which the runtime profile owns."""
    run_oracle = (config.get('task_config') or {}).get('run_oracle', False)
    return bool(_resolve_scalars({'run_oracle': run_oracle})['run_oracle'])


def _probe_llm(config: dict[str, Any]) -> dict[str, str]:
    """Resolve the merged config's model and api_key, for the cache-partition decision below."""
    main = ((config.get('openai_clients') or {}).get('main') or {}).get('config') or {}
    resolved = _resolve_scalars({
        'model': (main.get('request_args') or {}).get('model') or '',
        'api_key': (main.get('client_args') or {}).get('api_key') or '',
    })
    return {key: str(value or '') for key, value in resolved.items()}


def _per_instance_cache_id(instance_id: str) -> str:
    """md5 of ``time:pid:instance_id`` — unique per instance and per shard (each shard is its own process)."""
    seed = f'{time.time()}:{os.getpid()}:{instance_id}'
    return hashlib.md5(seed.encode()).hexdigest()


def _api_key_with_cache_task_id(probe: dict[str, str], instance_id: str) -> str | None:
    """Return the api_key with a per-run ``cache_task_id`` appended, or None to leave it alone.

    The gateway partitions its prompt cache by this id, so two instances running concurrently under one key would otherwise cross-wire each other's prefix cache. Only Claude models go through a gateway that reads it; a non-Claude key must not be mangled.
    """
    api_key = probe['api_key']
    if not api_key or 'cache_task_id=' in api_key or 'claude' not in probe['model'].lower():
        return None
    separator = '&' if '?' in api_key else '?'
    return f'{api_key}{separator}cache_task_id={_per_instance_cache_id(instance_id)}'


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


def _prompt_cache_key_for_instance(probe: dict[str, str], instance_id: str) -> str | None:
    """Return a per-run ``prompt_cache_key`` for models whose gateway caches by this field, or None to leave request_args alone.

    Same uniqueness as Claude's ``cache_task_id``: two concurrent instances must not share a key, or their prefix caches cross-wire. A static value in the LLM profile would do exactly that, so the generator always overlays one here.
    """
    model = probe['model'].lower()
    prefix = 'openai/'
    model_stem = model[len(prefix) :] if model.startswith(prefix) else model
    if model_stem not in _PROMPT_CACHE_KEY_MODELS:
        return None
    return _per_instance_cache_id(instance_id)


def _extract_instance_ids(path: Path) -> set[str]:
    """Pull instance_ids out of a ``.jsonl`` (one object per line) or ``.json`` (list or object)."""

    def instance_id_of(obj: Any) -> str:
        """Extract an instance_id from a selector element: a bare string, or a dict's ``instance_id``."""
        if isinstance(obj, str):
            return obj.strip()
        if isinstance(obj, dict):
            return str(obj.get('instance_id', '') or '').strip()
        return ''

    ids: set[str] = set()
    if path.suffix.lower() == '.jsonl':
        with path.open('r', encoding='utf-8') as f:
            for line_no, raw in enumerate(f, start=1):
                if not (line := raw.strip()):
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f'{path}:{line_no} is not valid JSON: {e}') from e
                if not (instance_id := instance_id_of(obj)):
                    raise ValueError(f'{path}:{line_no} has no resolvable "instance_id".')
                ids.add(instance_id)
        return ids
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except json.JSONDecodeError as e:
        raise ValueError(f'{path} is not valid JSON: {e}') from e
    for idx, obj in enumerate(data if isinstance(data, list) else [data]):
        if not (instance_id := instance_id_of(obj)):
            raise ValueError(f'{path}[{idx}] has no resolvable "instance_id".')
        ids.add(instance_id)
    return ids


def _selected_instance_filters() -> tuple[set[str], set[str]]:
    """Resolve INCLUDE_INSTANCES / EXCLUDE_INSTANCES, rejecting any id claimed by both."""
    selectors: list[set[str]] = []
    for name in ('INCLUDE_INSTANCES', 'EXCLUDE_INSTANCES'):
        ids: set[str] = set()
        for item in (x.strip() for x in os.getenv(name, '').split(',')):
            if not item:
                continue
            if item.lower().endswith(('.json', '.jsonl')):
                path = Path(item)
                if not path.is_file():
                    raise ValueError(f'{name}: selector file not found: {item}')
                ids.update(_extract_instance_ids(path))
            else:
                ids.add(item)
        selectors.append(ids)
    include, exclude = selectors
    if conflict := include & exclude:
        raise ValueError(f'instance_id(s) in both INCLUDE_INSTANCES and EXCLUDE_INSTANCES: {sorted(conflict)}')
    return include, exclude


def _required_dataset_path() -> str:
    dataset_path = os.getenv('DATASET_PATH', '').strip()
    if not dataset_path:
        raise ValueError('DATASET_PATH is required and must be a non-empty path to a .jsonl dataset')
    return dataset_path


def _iter_selected_instances() -> Iterator[tuple[str, SWEbenchInstance, dict[str, Any]]]:
    """Yield (instance_id, instance, metadata) for every instance that survives the filters and SWE_SKIP/SWE_LIMIT."""
    dataset_path = _required_dataset_path()
    include_ids, exclude_ids = _selected_instance_filters()
    skip = _env_int('SWE_SKIP', 0) or 0
    limit = _env_int('SWE_LIMIT')

    path = Path(dataset_path)
    if not path.is_file():
        raise FileNotFoundError(f'Dataset file not found: {dataset_path}')
    if path.suffix != '.jsonl':
        raise ValueError(f'Dataset must be .jsonl for batch mode: {dataset_path}')
    instances: list[SWEbenchInstance] = []
    # Not str.splitlines(): it also splits on U+2028 etc., which JSON allows unescaped inside strings.
    with path.open(encoding='utf-8') as f:
        for lineno, raw in enumerate(f, start=1):
            if not (line := raw.strip()):
                continue
            try:
                instance = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning('Skipping malformed JSON at %s:%d: %s', dataset_path, lineno, e)
                continue
            if not isinstance(instance, dict):
                logger.warning(
                    'Skipping non-object JSON at %s:%d: got %s', dataset_path, lineno, type(instance).__name__
                )
                continue
            instances.append(instance)
    source = (
        (
            str(instance['instance_id']),
            instance,
            {
                'dataset_path': dataset_path,
                'repo': instance.get('repo', ''),
                'dataset_type': str(instance.get('dataset_type') or DATASET_SWE_BENCH),
                'language': instance.get('language', 'unknown'),
            },
        )
        for instance in instances
        if instance.get('instance_id')
    )

    skipped = 0
    produced = 0
    for instance_id, instance, metadata in source:
        if include_ids and instance_id not in include_ids:
            continue
        if instance_id in exclude_ids:
            continue
        if skipped < skip:
            skipped += 1
            continue
        yield instance_id, instance, metadata
        produced += 1
        if limit is not None and produced >= limit:
            return


def _load_parts() -> dict[str, Any]:
    """Fold the three run parts into one config: runtime, then agent, then LLM.

    An oracle run loads all three too, then has the agent stack removed. ``run_oracle`` lives in the runtime profile's ``task_config``, so it can only be read once the merge has happened.
    """
    config = load_config()
    if not _is_oracle_run(config):
        return config
    # Oracle applies the golden patch and never reaches the agent branch of the task, so the scaffold and its client are dead weight. None deletes a key under run-part merge rules.
    return deep_merge(config, {'agent_name': None, 'agent_config': None, 'openai_clients': None})


def generate_configs() -> Iterator[BatchItem]:
    # The run parts do not vary by instance; only the overlays below do.
    base = _load_parts()
    probe = _probe_llm(base)

    for instance_id, instance, instance_meta in _iter_selected_instances():
        overlay: dict[str, Any] = {
            'task_config': {
                'instance_id': instance_id,
                'instance': instance,
            }
        }

        if override := INSTANCE_RESOURCE_OVERRIDE.get(instance_id):
            overlay['runtime_provider_config'] = {'resource_spec': override}
        client_overlay: dict[str, Any] = {}
        if api_key := _api_key_with_cache_task_id(probe, instance_id):
            client_overlay['client_args'] = {'api_key': api_key}
        if prompt_cache_key := _prompt_cache_key_for_instance(probe, instance_id):
            client_overlay['request_args'] = {'prompt_cache_key': prompt_cache_key}
        if client_overlay:
            overlay['openai_clients'] = {'main': {'config': client_overlay}}

        instance_config = deep_merge(base, overlay)
        yield BatchItem(
            config=RunInput.from_dict(instance_config),
            run_id=instance_id,
            metadata={
                'instance_id': instance_id,
                'agent_name': instance_config.get('agent_name'),
                'model': probe['model'],
                **instance_meta,
            },
        )


def get_total_count() -> int:
    return sum(1 for _ in _iter_selected_instances())


def get_batch_metadata() -> dict[str, Any]:
    include_ids, exclude_ids = _selected_instance_filters()
    base = _load_parts()
    metadata: dict[str, Any] = {
        'task_name': base.get('task_name', 'swe_task'),
        'run_oracle': _is_oracle_run(base),
        'include_instance_ids': sorted(include_ids) or None,
        'exclude_instance_ids': sorted(exclude_ids) or None,
        'agent_name': base.get('agent_name'),
        'model': _probe_llm(base)['model'],
        'dataset_path': _required_dataset_path(),
    }
    return metadata
