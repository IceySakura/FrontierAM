#!/usr/bin/env python3
"""SWE example generator for ``nexus run-batch``, on the shared configs/ layer.

The agent / LLM / runtime blocks come from run-part YAML under ``configs/`` instead of a wall of ``os.environ.setdefault`` defaults. Each file is the format ``nexus run batch --use`` takes (``body`` / ``merge`` / ``source``), so this module folds three of them into one config and keeps only what is genuinely per batch or per instance: dataset iteration, include/exclude filters, paging, the inlined instance, and the two things that cannot be known until the instance is — its cache partition and any resource override.

Usage:
  uv run nexus run-batch -g nexus/extensions/configs/generator_swe_template.py -d ./outputs/swe_batch.db

Configs (file stem under ``configs/<kind>/``, or a path):
  - RUNTIME_CONFIG: sandbox, e.g. swecosmos_runtime
  - AGENT_CONFIG: scaffold, e.g. claude_code_eval / claude_code_rollout / openclaw / codex
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
import os
import time

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from nexus import BatchItem
from nexus.extensions.configs.cache_id import cache_task_overlay
from nexus.extensions.configs.load_config import load_config
from nexus.extensions.tasks.swe.constants import DATASET_SWE_BENCH, SWEbenchInstance
from nexus.runner.compose.merge import deep_merge
from nexus.runner.single import RunInput, _resolve_env_suffix_config


# Instances whose resource_spec the runtime profile's default does not cover.
INSTANCE_RESOURCE_OVERRIDE = {
    # Heavy numeric workload, CPU bound.
    'scikit-learn__scikit-learn-14710': {'cpu': 16, 'memory': 16},
}


def _env_int(name: str, default: int | None = None) -> int | None:
    """The integer ``name`` is set to, or ``default`` when it is unset.

    Raises ``ValueError`` for a value that is not an integer. Swallowing it returned
    ``default``, and ``SWE_LIMIT``'s default is ``None`` -- i.e. no limit -- so a typo
    like ``SWE_LIMIT=10x`` silently produced the whole dataset. Same bug as
    ``tasks/hitl/utils/generator_util/env.py``, fixed the same way.
    """
    value = os.getenv(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        raise ValueError(f'{name}={value!r} is not an integer') from None


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
    'us.openai.gpt-6.1-sol',
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
                if not (instance_id := instance_id_of(json.loads(line))):
                    raise ValueError(f'{path}:{line_no} has no resolvable "instance_id".')
                ids.add(instance_id)
        return ids
    data = json.loads(path.read_text(encoding='utf-8'))
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
    instances: list[SWEbenchInstance] = [
        json.loads(line) for raw in path.read_text(encoding='utf-8').splitlines() if (line := raw.strip())
    ]
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

    if include_ids:
        # An id the dataset does not have would otherwise just shrink the run: 99 questions, a score still read as
        # out of 100, exit 0. Say so before any sandbox is paid for.
        missing = include_ids - {str(i['instance_id']) for i in instances if i.get('instance_id')}
        if missing:
            raise ValueError(f'INCLUDE_INSTANCES: {len(missing)} id(s) not in {dataset_path}: {sorted(missing)[:5]}')

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
        if limit is not None and produced >= limit:
            return
        yield instance_id, instance, metadata
        produced += 1


def _langfuse_overlay(base: dict[str, Any], instance_id: str, agent_name: Any, model: str) -> dict[str, Any]:
    """Name each Langfuse trace after its instance, as the production compiler does; the YAML cannot know the id."""
    if 'langfuse' not in (base.get('processors') or {}):
        return {}
    args: dict[str, Any] = {'trace_metadata': {'instance_id': instance_id, 'agent_name': agent_name, 'model': model}}
    if not os.getenv('LANGFUSE_TRACE_NAME'):
        args['trace_name'] = instance_id
    return {'processors': {'langfuse': {'args': args}}}


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
    main_client = (base.get('openai_clients') or {}).get('main') or {}
    client_name = str(main_client.get('name') or '')
    main_client_args = _resolve_scalars((main_client.get('config') or {}).get('client_args') or {})

    for instance_id, instance, instance_meta in _iter_selected_instances():
        overlay: dict[str, Any] = {
            'task_config': {
                'instance_id': instance_id,
                'instance': instance,
            }
        }

        if override := INSTANCE_RESOURCE_OVERRIDE.get(instance_id):
            overlay['runtime_provider_config'] = {'resource_spec': override}
        # One id per trajectory, on the wire and in the metadata (see cache_id.py).
        cache_id = _per_instance_cache_id(instance_id)
        cache_overlay, effective_cache_id = cache_task_overlay(
            client_name, main_client_args, probe['api_key'], cache_id
        )
        client_overlay: dict[str, Any] = cache_overlay or {}
        if prompt_cache_key := _prompt_cache_key_for_instance(probe, instance_id):
            client_overlay['request_args'] = {'prompt_cache_key': prompt_cache_key}
        if client_overlay:
            overlay['openai_clients'] = {'main': {'config': client_overlay}}
        overlay = deep_merge(overlay, _langfuse_overlay(base, instance_id, base.get('agent_name'), probe['model']))

        instance_config = deep_merge(base, overlay)
        metadata: dict[str, Any] = {
            'instance_id': instance_id,
            'agent_name': instance_config.get('agent_name'),
            'model': probe['model'],
            **instance_meta,
        }
        if effective_cache_id is not None:
            # HyWay reads this from the nexus output and passes it to the CoT hack, which can only
            # decode encrypted reasoning on the account this id routed to. Recorded only when the
            # wire really carried it.
            metadata['cache_task_id'] = effective_cache_id
        yield BatchItem(
            config=RunInput.from_dict(instance_config),
            run_id=instance_id,
            metadata=metadata,
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
