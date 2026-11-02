#!/usr/bin/env python3
"""SWE-Any batch generator for ``nexus run-batch`` — Export at scale.

Yields a ``swe_any_export`` BatchItem for each SWE-Env instance whose Issue already wrote
``<output_dir>/<repo_name>/issue.jsonl`` on the host (its hard precondition — Export joins Mask +
Eval + Issue per feature). Each item re-acquires the baked image and, for every ready feature,
execution-grades it (pregold FAIL / postgold PASS → FAIL_TO_PASS / PASS_TO_PASS) and, when valid,
writes a Harbor task directory under ``<output_dir>/harbor/<instance_id>/``.

Export is DETERMINISTIC (no LLM), so — unlike the other stages — it does NOT apply a
``config/client/<name>.yaml`` backend patch; it assembles the base config directly.

Environment: SWE_ANY_INSTANCES (required), SWE_ANY_CONFIG_PATCH, SWE_ANY_INSTANCE_IDS / _SKIP /
_LIMIT / _LANGUAGE, SWE_ANY_OUTPUT_DIR, GONGFENG_TOKEN.
"""

from __future__ import annotations

import copy

from collections.abc import Iterator
from typing import Any

from nexus import BatchItem
from nexus.extensions.tasks.swe_any.utils.batch_config import (
    STAGE_CONFIG_DIR,
    base_config_and_spec,
    config_patch_paths,
)
from nexus.extensions.tasks.swe_any.utils.instance import (
    host_eval_path,
    host_issue_path,
    host_mask_path,
    output_dir,
    repo_name_of,
    selected_instances,
)
from nexus.runner.single import RunInput


_DEFAULT_YAML = STAGE_CONFIG_DIR / 'export.yaml'


def _selected_export_instances(spec: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield only instances with a host ``issue.jsonl`` — Export's hard precondition."""
    out_dir = output_dir()
    for inst in selected_instances(spec):
        if host_issue_path(inst, out_dir=out_dir).is_file():
            yield inst


def _build_config(base: dict[str, Any], inst: dict[str, Any]) -> RunInput:
    config = copy.deepcopy(base)
    repo_dir = str(inst.get('workdir') or '/testbed')

    task_config = config['task_config']
    # Forward the whole SWE-Env instance so the graded SWE-bench record carries a real identity
    # (repo / base_commit / image) and a unique instance_id — apply_instance hydrates repo_name /
    # image / repo_dir from it, and build_swebench_instance reads repo / base_commit off it.
    task_config['instance'] = inst
    task_config['repo_name'] = repo_name_of(inst)
    task_config['mask_path'] = str(host_mask_path(inst))
    task_config['eval_path'] = str(host_eval_path(inst))
    task_config['issue_path'] = str(host_issue_path(inst))
    task_config['output_dir'] = output_dir()
    task_config['image'] = inst['image']
    task_config['repo_dir'] = repo_dir
    task_config['release_runtime'] = True
    return RunInput.from_dict(config)


def generate_configs() -> Iterator[BatchItem]:
    base, spec = base_config_and_spec(_DEFAULT_YAML)
    for inst in _selected_export_instances(spec):
        instance_id = str(inst['instance_id'])
        yield BatchItem(
            config=_build_config(base, inst),
            run_id=instance_id,
            metadata={
                'instance_id': instance_id,
                'repo_name': inst.get('repo_name'),
                'language': inst.get('language'),
                'stage': 'export',
                'issue_path': str(host_issue_path(inst)),
            },
        )


def get_total_count() -> int:
    _, spec = base_config_and_spec(_DEFAULT_YAML)
    return sum(1 for _ in _selected_export_instances(spec))


def get_batch_metadata() -> dict[str, Any]:
    _, spec = base_config_and_spec(_DEFAULT_YAML)
    return {
        'task_name': 'swe_any_export',
        'stage': 'export',
        'instances': spec.get('path', ''),
        'language': spec.get('language', ''),
        'output_dir': output_dir(),
        'config_patch': ','.join(config_patch_paths()),
    }
