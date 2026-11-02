#!/usr/bin/env python3
"""Expand successful pre-feature manifests into one batch item per feature."""

from __future__ import annotations

import copy
import os

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from nexus import BatchItem
from nexus.extensions.tasks.swe_any.pipeline.manifest import iter_jsonl, manifest_run_id, read_manifest
from nexus.extensions.tasks.swe_any.utils.batch_config import (
    PIPELINE_CONFIG_DIR,
    apply_client_patch,
    attach_cache_task_id,
    config_patch_paths,
    load_stage_config,
)
from nexus.extensions.tasks.swe_any.utils.instance import host_feature_result_path, output_dir, repo_name_of
from nexus.runner.single import RunInput


_DEFAULT_YAML = PIPELINE_CONFIG_DIR / 'feature.yaml'
_PRE_RESULTS_ENV = 'SWE_ANY_PRE_FEATURE_RESULTS'
_FEATURE_STAGES = ('select', 'mask', 'eval', 'issue', 'export')


def pre_feature_results_path() -> Path:
    configured = os.getenv(_PRE_RESULTS_ENV, '').strip()
    return Path(configured) if configured else Path(output_dir()) / 'pipeline_pre_feature_batch_results.jsonl'


def _base_config() -> dict[str, Any]:
    config = load_stage_config(_DEFAULT_YAML, config_patch_paths())
    # Legacy custom patches may carry an instances block.  Feature generation is
    # driven by pre-feature batch outputs, never by a second dataset scan.
    config.pop('instances', None)
    return apply_client_patch(config)


def _manifest_entries(results_path: Path) -> list[tuple[Path, dict[str, Any]]]:
    """Return all successful pre-feature manifest rows in stable global order."""
    entries: list[tuple[Path, dict[str, Any]]] = []
    for record in iter_jsonl(results_path):
        if record.get('status') != 'success':
            continue
        task_output = record.get('task_output')
        if not isinstance(task_output, dict):
            raise ValueError(f'{results_path}: successful pre-feature run {record.get("run_id")!r} has no task_output')
        raw_manifest_path = task_output.get('manifest_path')
        if not isinstance(raw_manifest_path, str) or not raw_manifest_path:
            raise ValueError(
                f'{results_path}: successful pre-feature run {record.get("run_id")!r} has no task_output.manifest_path'
            )
        manifest_path = Path(raw_manifest_path)
        graph_path = task_output.get('graph_path')
        output_repo_name = task_output.get('repo_name')
        for row in read_manifest(manifest_path):
            instance = row.get('instance')
            if not isinstance(instance, dict):
                raise ValueError(f'{manifest_path}: feature index {row["index"]} missing instance')
            repo_name = str(row.get('repo_name') or output_repo_name or repo_name_of(instance))
            if not repo_name:
                raise ValueError(f'{manifest_path}: feature index {row["index"]} missing repo_name')
            row_graph_path = row.get('graph_path') or graph_path
            if not isinstance(row_graph_path, str) or not row_graph_path:
                raise ValueError(f'{manifest_path}: feature index {row["index"]} missing graph_path')
            if row['heuristic_row'].get('index') != row['index']:
                raise ValueError(
                    f'{manifest_path}: feature index {row["index"]} does not match heuristic_row index '
                    f'{row["heuristic_row"].get("index")!r}'
                )
            entries.append((
                manifest_path,
                {
                    **row,
                    'repo_name': repo_name,
                    'graph_path': row_graph_path,
                },
            ))

    entries.sort(
        key=lambda entry: (
            str(entry[1]['instance'].get('instance_id') or ''),
            str(entry[1]['repo_name']),
            int(entry[1]['index']),
            str(entry[0]),
        )
    )
    seen_run_ids: set[str] = set()
    for _manifest_path, row in entries:
        instance_id = str(row['instance'].get('instance_id') or '')
        if not instance_id:
            raise ValueError('manifest instance requires non-empty instance_id')
        run_id = manifest_run_id(instance_id, row['index'])
        if run_id in seen_run_ids:
            raise ValueError(f'duplicate feature run id from pre-feature manifests: {run_id}')
        seen_run_ids.add(run_id)
    return entries


def _build_config(base: dict[str, Any], manifest_path: Path, row: dict[str, Any]) -> RunInput:
    config = copy.deepcopy(base)
    instance = row['instance']
    repo_name = row['repo_name']
    index = int(row['index'])

    task_config = config['task_config']
    task_config.update({
        'repo_name': repo_name,
        'index': index,
        'graph_path': row['graph_path'],
        'manifest_path': str(manifest_path),
        'output_dir': output_dir(),
        'image': instance['image'],
        'repo_dir': str(instance.get('workdir') or '/testbed'),
        'runtime_name': 'main',
        'release_runtime': True,
    })
    for stage in _FEATURE_STAGES:
        overrides = task_config['stage_configs'].setdefault(stage, {})
        overrides.update({'runtime_name': 'main', 'max_workers': 1, 'release_runtime': False})

    instance_id = str(instance['instance_id'])
    attach_cache_task_id(config, f'{instance_id}:{index}')

    # Resolve config/env interpolation before injecting opaque manifest JSON.  A
    # feature description or patch-like field containing ``${oc.env:...}`` is
    # data and must not be interpreted as generator configuration.
    run_input = RunInput.from_dict(config)
    resolved_task_config = dict(run_input.task_config)
    resolved_task_config['instance'] = copy.deepcopy(instance)
    resolved_task_config['heuristic_row'] = copy.deepcopy(row['heuristic_row'])
    resolved_task_config['heuristic_error'] = row.get('heuristic_error')
    resolved_task_config['select_settings'] = copy.deepcopy(row.get('select_settings') or {})
    return run_input.model_copy(update={'task_config': resolved_task_config})


def generate_configs() -> Iterator[BatchItem]:
    results_path = pre_feature_results_path()
    base = _base_config()
    for manifest_path, row in _manifest_entries(results_path):
        instance = row['instance']
        instance_id = str(instance['instance_id'])
        index = int(row['index'])
        run_id = manifest_run_id(instance_id, index)
        yield BatchItem(
            config=_build_config(base, manifest_path, row),
            run_id=run_id,
            metadata={
                'instance_id': instance_id,
                'instance': instance,
                'repo_name': row['repo_name'],
                'language': instance.get('language'),
                'index': index,
                'stage': 'pipeline_feature',
                'manifest_path': str(manifest_path),
                'graph_path': row['graph_path'],
                'feature_result_path': str(host_feature_result_path(instance, index)),
            },
        )


def get_total_count() -> int:
    return len(_manifest_entries(pre_feature_results_path()))


def get_batch_metadata() -> dict[str, Any]:
    results_path = pre_feature_results_path()
    return {
        'task_name': 'swe_any_pipeline_feature',
        'stage': 'pipeline_feature',
        'pre_feature_results': str(results_path),
        'features': len(_manifest_entries(results_path)),
        'output_dir': output_dir(),
        'config_patch': ','.join(config_patch_paths()),
    }
