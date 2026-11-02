#!/usr/bin/env python3
"""SWE-Any batch generator for ``nexus run-batch`` — Issue at scale.

Yields a ``swe_any_issue`` BatchItem for each SWE-Env instance whose Eval already wrote
``<output_dir>/<repo_name>/eval.jsonl`` on the host (its hard precondition — Issue joins Mask +
Eval per feature). Each item re-acquires the baked image and, for every masked+oracled feature,
git-applies the mask patches to build a fully masked tree, then runs one Claude Code pass to write a
non-leaking ``problem_statement``.

Mirrors :mod:`.eval`: starts from ``config/stage/issue.yaml`` (shared with run-single) with
``SWE_ANY_CONFIG_PATCH`` deep-merged and the client backend patch applied; only per-instance fields
and the provider timeout floor are injected here.

Environment: SWE_ANY_INSTANCES (required), SWE_ANY_CONFIG_PATCH, SWE_ANY_INSTANCE_IDS / _SKIP /
_LIMIT / _LANGUAGE, SWE_ANY_OUTPUT_DIR, OPENAI_API_KEY / GONGFENG_TOKEN.
"""

from __future__ import annotations

import copy

from collections.abc import Iterator
from typing import Any

from nexus import BatchItem
from nexus.extensions.tasks.swe_any.utils.batch_config import (
    STAGE_CONFIG_DIR,
    agentic_config_and_spec,
    attach_cache_task_id,
    config_patch_paths,
    resolve_config,
)
from nexus.extensions.tasks.swe_any.utils.instance import (
    host_eval_path,
    host_mask_path,
    host_select_path,
    output_dir,
    repo_name_of,
    selected_instances,
)
from nexus.runner.single import RunInput


_DEFAULT_YAML = STAGE_CONFIG_DIR / 'issue.yaml'


def _selected_issue_instances(spec: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield only instances with a host ``eval.jsonl`` — Issue's hard precondition."""
    out_dir = output_dir()
    for inst in selected_instances(spec):
        if host_eval_path(inst, out_dir=out_dir).is_file():
            yield inst


def _build_config(base: dict[str, Any], inst: dict[str, Any]) -> RunInput:
    config = copy.deepcopy(base)
    resolved = resolve_config(config)

    repo_name = repo_name_of(inst)
    repo_dir = str(inst.get('workdir') or '/testbed')
    agent_timeout = float(resolved['agent_config']['timeout'])

    task_config = config['task_config']
    task_config['repo_name'] = repo_name
    task_config['select_path'] = str(host_select_path(inst))
    task_config['mask_path'] = str(host_mask_path(inst))
    task_config['eval_path'] = str(host_eval_path(inst))
    task_config['output_dir'] = output_dir()
    task_config['image'] = inst['image']
    task_config['repo_dir'] = repo_dir
    task_config['release_runtime'] = True

    provider_config = config['runtime_provider_config']
    provider_config['request_timeout'] = max(
        float(resolved['runtime_provider_config']['request_timeout']),
        agent_timeout + 600.0,
    )
    attach_cache_task_id(config, inst['instance_id'])
    return RunInput.from_dict(config)


def generate_configs() -> Iterator[BatchItem]:
    base, spec = agentic_config_and_spec(_DEFAULT_YAML)
    for inst in _selected_issue_instances(spec):
        instance_id = str(inst['instance_id'])
        yield BatchItem(
            config=_build_config(base, inst),
            run_id=instance_id,
            metadata={
                'instance_id': instance_id,
                'repo_name': inst.get('repo_name'),
                'language': inst.get('language'),
                'stage': 'issue',
                'eval_path': str(host_eval_path(inst)),
            },
        )


def get_total_count() -> int:
    _, spec = agentic_config_and_spec(_DEFAULT_YAML)
    return sum(1 for _ in _selected_issue_instances(spec))


def get_batch_metadata() -> dict[str, Any]:
    _, spec = agentic_config_and_spec(_DEFAULT_YAML)
    return {
        'task_name': 'swe_any_issue',
        'stage': 'issue',
        'instances': spec.get('path', ''),
        'language': spec.get('language', ''),
        'output_dir': output_dir(),
        'config_patch': ','.join(config_patch_paths()),
    }
