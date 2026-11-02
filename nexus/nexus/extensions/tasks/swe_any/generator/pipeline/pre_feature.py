#!/usr/bin/env python3
"""Generate one repo-level preparation task per selected SWE-Env instance."""

from __future__ import annotations

import copy

from collections.abc import Iterator
from typing import Any

from nexus import BatchItem
from nexus.extensions.tasks.swe_any.utils.batch_config import (
    PIPELINE_CONFIG_DIR,
    config_patch_paths,
    load_stage_config,
    pop_instance_spec,
    resolve_config,
    resolve_instance_spec,
)
from nexus.extensions.tasks.swe_any.utils.instance import (
    host_manifest_path,
    output_dir,
    repo_name_of,
    selected_instances,
)
from nexus.runner.single import RunInput


_DEFAULT_YAML = PIPELINE_CONFIG_DIR / 'pre_feature.yaml'


def _base_and_spec() -> tuple[dict[str, Any], dict[str, Any]]:
    config = load_stage_config(_DEFAULT_YAML, config_patch_paths())
    spec = resolve_instance_spec(pop_instance_spec(config))
    return config, spec


def _build_config(base: dict[str, Any], inst: dict[str, Any]) -> RunInput:
    config = copy.deepcopy(base)
    repo_name = repo_name_of(inst)
    task_config = config['task_config']
    task_config.update({
        'instance': inst,
        'repo_name': repo_name,
        'output_dir': output_dir(),
        'image': inst['image'],
        'repo_dir': str(inst.get('workdir') or '/testbed'),
        'runtime_name': 'main',
        'release_runtime': True,
    })

    stage_configs = task_config['stage_configs']
    stage_configs['parse'].update({
        'project_name': inst.get('repo') or repo_name,
        'runtime_name': 'main',
        'release_runtime': False,
    })
    stage_configs['select'].update({'runtime_name': 'main', 'max_workers': 1, 'release_runtime': False})

    # The repo sandbox must remain available through both preparation steps.
    resolved = resolve_config(config)
    parse_config = resolved['task_config']['stage_configs']['parse']
    parse_timeout = max(
        float(parse_config['request_timeout']),
        float(parse_config['index_timeout']) + float(parse_config['npm_install_timeout']) + 300.0,
    )
    config['runtime_provider_config']['request_timeout'] = max(
        float(resolved['runtime_provider_config']['request_timeout']),
        parse_timeout + 600.0,
    )
    return RunInput.from_dict(config)


def generate_configs() -> Iterator[BatchItem]:
    base, spec = _base_and_spec()
    for inst in selected_instances(spec):
        instance_id = str(inst['instance_id'])
        repo_name = repo_name_of(inst)
        yield BatchItem(
            config=_build_config(base, inst),
            run_id=instance_id,
            metadata={
                'instance_id': instance_id,
                'repo_name': repo_name,
                'language': inst.get('language'),
                'stage': 'pipeline_pre_feature',
                'manifest_path': str(host_manifest_path(inst)),
            },
        )


def get_total_count() -> int:
    _, spec = _base_and_spec()
    return sum(1 for _ in selected_instances(spec))


def get_batch_metadata() -> dict[str, Any]:
    _, spec = _base_and_spec()
    return {
        'task_name': 'swe_any_pipeline_pre_feature',
        'stage': 'pipeline_pre_feature',
        'instances': spec.get('path', ''),
        'language': spec.get('language', ''),
        'output_dir': output_dir(),
        'config_patch': ','.join(config_patch_paths()),
    }
