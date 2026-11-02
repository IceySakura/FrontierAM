#!/usr/bin/env python3
"""Create one deterministic reduce task from feature batch JSONL output."""

from __future__ import annotations

import copy
import os

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from nexus import BatchItem
from nexus.extensions.tasks.swe_any.pipeline.manifest import iter_jsonl
from nexus.extensions.tasks.swe_any.utils.batch_config import PIPELINE_CONFIG_DIR, config_patch_paths, load_stage_config
from nexus.extensions.tasks.swe_any.utils.instance import output_dir
from nexus.runner.single import RunInput


_DEFAULT_YAML = PIPELINE_CONFIG_DIR / 'post_feature.yaml'
_FEATURE_RESULTS_ENV = 'SWE_ANY_FEATURE_RESULTS'


def feature_results_path() -> Path:
    configured = os.getenv(_FEATURE_RESULTS_ENV, '').strip()
    return Path(configured) if configured else Path(output_dir()) / 'pipeline_feature_batch_results.jsonl'


def _base_config() -> dict[str, Any]:
    config = load_stage_config(_DEFAULT_YAML, config_patch_paths())
    # Keep legacy custom patches harmless for this host-only reducer.
    config.pop('instances', None)
    return config


def _feature_runs(results_path: Path) -> list[dict[str, Any]]:
    """Normalize exported batch rows without dropping failed feature tasks."""
    runs: list[dict[str, Any]] = []
    for record in iter_jsonl(results_path):
        run_id = record.get('run_id')
        status = record.get('status')
        metadata = record.get('metadata')
        task_output = record.get('task_output')
        if not isinstance(run_id, str) or not run_id:
            raise ValueError(f'{results_path}: batch record missing run_id')
        if not isinstance(status, str) or not status:
            raise ValueError(f'{results_path}: batch record {run_id!r} missing status')
        if not isinstance(metadata, dict):
            raise ValueError(f'{results_path}: batch record {run_id!r} missing metadata')
        if task_output is not None and not isinstance(task_output, dict):
            raise ValueError(f'{results_path}: batch record {run_id!r} task_output must be a mapping or null')

        output = task_output or {}
        repo_name = metadata.get('repo_name') or output.get('repo_name')
        index = metadata.get('index', output.get('index'))
        if not isinstance(repo_name, str) or not repo_name:
            raise ValueError(f'{results_path}: batch record {run_id!r} missing repo_name')
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise ValueError(f'{results_path}: batch record {run_id!r} missing non-negative integer index')

        feature_result_path = output.get('feature_result_path') or metadata.get('feature_result_path')
        if feature_result_path is not None and not isinstance(feature_result_path, str):
            raise ValueError(f'{results_path}: batch record {run_id!r} feature_result_path must be a string or null')
        instance = metadata.get('instance')
        if not isinstance(instance, dict):
            raise ValueError(f'{results_path}: batch record {run_id!r} metadata missing source instance')

        runs.append({
            'run_id': run_id,
            'batch_status': status,
            'batch_error': record.get('error'),
            'repo_name': repo_name,
            'index': index,
            'instance': instance,
            'feature_result_path': feature_result_path,
            'feature_status': output.get('status'),
            'terminal_stage': output.get('terminal_stage'),
            'feature_error': output.get('error'),
        })

    runs.sort(key=lambda run: (run['repo_name'], run['index'], run['run_id']))
    seen: set[tuple[str, int]] = set()
    for run in runs:
        key = (run['repo_name'], run['index'])
        if key in seen:
            raise ValueError(f'{results_path}: duplicate feature batch record for {key[0]}/sg{key[1]:02d}')
        seen.add(key)
    return runs


def _build_config(base: dict[str, Any], results_path: Path, feature_runs: list[dict[str, Any]]) -> RunInput:
    config = copy.deepcopy(base)
    config['task_config'].update({
        'output_dir': output_dir(),
        'feature_batch_results_path': str(results_path),
    })
    # Resolve only template configuration first; batch errors and source records
    # are opaque JSON and may contain interpolation-looking text.
    run_input = RunInput.from_dict(config)
    task_config = dict(run_input.task_config)
    task_config['feature_runs'] = copy.deepcopy(feature_runs)
    return run_input.model_copy(update={'task_config': task_config})


def generate_configs() -> Iterator[BatchItem]:
    results_path = feature_results_path()
    feature_runs = _feature_runs(results_path)
    yield BatchItem(
        config=_build_config(_base_config(), results_path, feature_runs),
        run_id='dataset',
        metadata={
            'stage': 'pipeline_post_feature',
            'feature_results': str(results_path),
            'features': len(feature_runs),
            'repos': len({run['repo_name'] for run in feature_runs}),
        },
    )


def get_total_count() -> int:
    _feature_runs(feature_results_path())
    return 1


def get_batch_metadata() -> dict[str, Any]:
    results_path = feature_results_path()
    feature_runs = _feature_runs(results_path)
    return {
        'task_name': 'swe_any_pipeline_post_feature',
        'stage': 'pipeline_post_feature',
        'feature_results': str(results_path),
        'features': len(feature_runs),
        'repos': len({run['repo_name'] for run in feature_runs}),
        'output_dir': output_dir(),
        'config_patch': ','.join(config_patch_paths()),
    }
