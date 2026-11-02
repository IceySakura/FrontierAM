#!/usr/bin/env python3
"""SWE-Env *oracle* batch generator for `nexus run-batch`.

Validates the effectiveness of each instance's oracle patch (the ``patch`` field): every instance is run in oracle mode (no agent), so the task applies ``instance['patch']`` directly and grades it. A healthy dataset should end up ``resolved`` for (almost) every row.

Two things are forced for every loaded instance:
  - ``dataset_type`` is overwritten to ``"swe-env"`` so swe_task routes to the SWE-Env handler.
  - ``run_oracle`` is True, so the oracle patch (``instance['patch']``) is applied.

Each (possibly modified) instance is passed *inline* via ``task_config['instance']`` rather than by ``dataset_path`` + ``instance_id``. The inline form is the only way to override per-row fields here, and ``RunInput.from_dict`` deliberately pops ``task_config.instance`` before OmegaConf resolution, so arbitrary patch/diff text (including ``${...}``) is preserved verbatim.

Usage:
  uv run nexus run-batch -g nexus/extensions/tasks/swe/swe_env/oracle_generator.py \
      -d ./outputs/swe_env_oracle_batch.db --max-concurrent 20

Environment variables (subset of the shared SWE generator):
  - SWE_DATASET_PATH: path to the swe-env instances .jsonl (required)
  - SWE_INSTANCE_IDS: comma-separated instance_ids to run (default: all)
  - SWE_SKIP / SWE_LIMIT: paging over the (filtered) rows
  - SWE_EVAL_TIMEOUT_SECONDS: per-instance eval timeout (default: 1800)
  - SWE_RAISE_ON_RUNTIME_ERROR: re-raise infra errors so the batch layer can retry (default: false)
  - SWE_WORKER_VENV: upload the current nexus wheel into the eval sandbox (default: false)
  - SWE_RUNTIME_PROVIDER_NAME / SWE_RUNTIME_BASE_URL / SWE_RUNTIME_TAG / SWE_RUNTIME_TIMEOUT / SWE_RUNTIME_CPU / SWE_RUNTIME_MEMORY / SWE_RPC_PROTOCOL / GONGFENG_RUNTIME_TOKEN: sandbox config
"""

from __future__ import annotations

import json
import os

from collections.abc import Callable, Iterator
from typing import Any

from nexus import BatchItem
from nexus.extensions.tasks.swe.constants import DATASET_SWE_ENV, SWEbenchInstance

# Reuse the shared SWE generator's env-parsing / iteration helpers to stay in lockstep with it.
from nexus.extensions.tasks.swe.generator import (
    INSTANCE_RESOURCE_OVERRIDE,
    _env_bool,
    _env_int,
    _env_list,
    _iter_instances_from_file,
)
from nexus.runner.single import RunInput


def _selected_instances() -> Iterator[SWEbenchInstance]:
    """Yield rows from SWE_DATASET_PATH, honoring SWE_INSTANCE_IDS / SWE_SKIP / SWE_LIMIT."""
    dataset_path = os.getenv('SWE_DATASET_PATH', '').strip()
    if not dataset_path:
        raise ValueError('SWE_DATASET_PATH is required (a .jsonl of swe-env instances).')

    ids_filter = _env_list('SWE_INSTANCE_IDS')
    if ids_filter is not None:
        resolved: set[str] = set()
        for item in ids_filter:
            if os.path.isfile(item):
                # item is a file path — treat it as a JSON list of instance ids
                with open(item, encoding='utf8') as f:
                    resolved.update(json.load(f))
            else:
                resolved.add(item)
        ids_filter = resolved

    skip = _env_int('SWE_SKIP') or 0
    limit = _env_int('SWE_LIMIT')

    skipped = 0
    produced = 0
    for inst in _iter_instances_from_file(dataset_path):
        instance_id = str(inst.get('instance_id', '') or '')
        if not instance_id:
            continue
        if ids_filter is not None and instance_id not in ids_filter:
            continue
        if skipped < skip:
            skipped += 1
            continue
        yield inst
        produced += 1
        if limit is not None and produced >= limit:
            break


def build_configs(mutate_instance: Callable[[dict[str, Any]], None]) -> Iterator[BatchItem]:
    """Yield one oracle-mode BatchItem per selected instance.

    ``mutate_instance`` is applied to a *copy* of each row after ``dataset_type`` is forced to ``"swe-env"`` and before the row is embedded inline. Use it to rewrite the oracle patch (e.g. the buggy generator swaps in a no-op patch). Leave it a no-op to keep the row's patch.
    """
    runtime_provider_name = os.getenv('SWE_RUNTIME_PROVIDER_NAME', 'gongfeng')
    runtime_base_url = os.getenv('SWE_RUNTIME_BASE_URL', 'http://hyrl-sandbox.prod.woa.com:8052')
    runtime_tag = os.getenv('SWE_RUNTIME_TAG', 'current').strip() or 'current'
    runtime_token = os.getenv('GONGFENG_RUNTIME_TOKEN', '')
    runtime_cpu = _env_int('SWE_RUNTIME_CPU') or 2
    runtime_memory = _env_int('SWE_RUNTIME_MEMORY') or 10
    rpc_protocol = os.getenv('SWE_RPC_PROTOCOL', 'http')
    worker_venv = _env_bool('SWE_WORKER_VENV', default=False)
    raise_on_runtime_error = _env_bool('SWE_RAISE_ON_RUNTIME_ERROR', default=False)
    eval_timeout = _env_int('SWE_EVAL_TIMEOUT_SECONDS') or 1800
    runtime_timeout = float(_env_int('SWE_RUNTIME_TIMEOUT') or (eval_timeout + 900))

    runtime_image = (
        runtime_tag if runtime_tag == 'current' else f'mirrors.tencent.com/hunyuan_yanguan/nexus-runtime:{runtime_tag}'
    )

    for inst in _selected_instances():
        # Work on a copy so we never mutate the source dict the iterator handed us.
        instance: dict[str, Any] = {**inst}
        instance['dataset_type'] = DATASET_SWE_ENV  # force swe-env routing, ignore the row's value
        mutate_instance(instance)
        instance_id = str(instance.get('instance_id', '') or '')

        task_config: dict[str, Any] = {
            'instance_id': instance_id,
            # Inline instance: overrides dataset loading and is preserved verbatim through from_dict (it is popped before OmegaConf resolution), so patch text is safe.
            'instance': instance,
            # Force oracle: the task applies instance['patch'] (no agent / no LLM).
            'run_oracle': True,
            # Always run the real eval, even for an empty/no-op patch — never short-circuit.
            'skip_empty_patch_eval': False,
            'use_partial_score': False,
            'eval_timeout_seconds': eval_timeout,
            'raise_on_runtime_error': raise_on_runtime_error,
        }

        config: dict[str, Any] = {
            'imports': [
                'nexus.extensions.tasks.swe',
                'nexus.extensions.agents.swe_agent',
            ],
            'task_name': 'swe_task',
            'task_config': task_config,
            'runtime_provider_name': runtime_provider_name,
            'runtime_provider_config': {
                'api_base_url': runtime_base_url,
                'token': runtime_token,
                'runtime_image': runtime_image,
                'request_timeout': runtime_timeout,
                'resource_spec': {
                    'cpu': INSTANCE_RESOURCE_OVERRIDE.get(instance_id, {}).get('cpu', runtime_cpu),
                    'memory': INSTANCE_RESOURCE_OVERRIDE.get(instance_id, {}).get('memory', runtime_memory),
                },
                'rpc_protocol': rpc_protocol,
            },
            'runtime_configs': {'eval': {'worker_venv': True}} if worker_venv else {},
            'processors': {},
            # Oracle mode needs no agent / LLM.
            'agent_name': None,
            'openai_clients': {},
        }

        yield BatchItem(
            config=RunInput.model_validate(config),
            run_id=instance_id,
            metadata={'instance_id': instance_id, 'dataset_type': DATASET_SWE_ENV, 'mode': 'oracle'},
        )


def batch_metadata(mode: str) -> dict[str, Any]:
    """Shared get_batch_metadata payload; ``mode`` distinguishes oracle vs buggy."""
    ids_filter = _env_list('SWE_INSTANCE_IDS')
    return {
        'task_name': 'swe_task',
        'dataset_path': os.getenv('SWE_DATASET_PATH', ''),
        'dataset_type': DATASET_SWE_ENV,
        'mode': mode,
        'run_oracle': True,
        'instance_ids_filter': sorted(ids_filter) if ids_filter else None,
    }


def generate_configs() -> Iterator[BatchItem]:
    # Keep each row's own ``patch`` as the oracle patch.
    return build_configs(lambda _instance: None)


def get_total_count() -> int:
    return sum(1 for _ in _selected_instances())


def get_batch_metadata() -> dict[str, Any]:
    return batch_metadata('oracle')
