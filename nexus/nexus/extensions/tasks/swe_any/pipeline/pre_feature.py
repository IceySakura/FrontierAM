"""Repo-level deterministic preparation for the feature pipeline.

Every candidate slice is written to the manifest, including heuristic failures.
That lets the feature task persist explicit Select failure and downstream
blocked rows, and lets Post reconstruct aggregates without dropping slices.
"""

from __future__ import annotations

import contextlib
import logging

from pathlib import Path
from typing import Any

from nexus.core import require_context, task_registry
from nexus.extensions.tasks.swe_any.pipeline.manifest import write_manifest
from nexus.extensions.tasks.swe_any.pipeline.schema import (
    PipelinePreFeatureTaskInput,
    PipelinePreFeatureTaskOutput,
)
from nexus.extensions.tasks.swe_any.stage.parse.runner import run_parse_operation
from nexus.extensions.tasks.swe_any.stage.parse.schema import ParseTaskInput
from nexus.extensions.tasks.swe_any.stage.select.runner import (
    prepare_select_heuristic_context,
    run_select_heuristic_slice,
)
from nexus.extensions.tasks.swe_any.stage.select.schema import SelectTaskInput
from nexus.extensions.tasks.swe_any.utils import constants
from nexus.extensions.tasks.swe_any.utils.artifacts import write_jsonl
from nexus.extensions.tasks.swe_any.utils.instance import apply_instance
from nexus.extensions.tasks.swe_any.utils.repo_git import reset_repo


logger = logging.getLogger(__name__)

_SELECT_MANIFEST_SETTINGS = (
    'rng_seed',
    'max_subgraphs',
    'max_bfs_nodes',
    'drop_bfs_prob',
    'min_bfs_per_depth',
    'max_bfs_per_depth',
    'include_tests',
    'include_external',
    'min_component_nodes',
    'max_component_nodes',
    'exclude_globs',
)


async def _no_agent(**_kwargs: Any) -> Any:
    """Satisfy Parse's shared schema while making accidental LLM use fail."""
    raise RuntimeError('pipeline pre-feature is deterministic and must not call run_agent')


async def _release_quietly(ctx: Any, runtime_name: str) -> None:
    with contextlib.suppress(Exception):
        await ctx.runtime_provider.release(name=runtime_name)


def _parse_input(task_input: PipelinePreFeatureTaskInput) -> ParseTaskInput:
    values = dict(task_input.stage_configs.get('parse', {}))
    values.update({
        'instance': task_input.instance,
        'instance_path': '',
        'repo_name': task_input.repo_name,
        'output_dir': task_input.output_dir,
        'repo_dir': task_input.repo_dir,
        'base_commit': task_input.base_commit,
        'swe_any_dir': task_input.swe_any_dir,
        'graph_filename': constants.GRAPH_FILENAME,
        'seed_policy_filename': constants.SEED_POLICY_FILENAME,
        'runtime_name': task_input.runtime_name,
        'image': task_input.image,
        'image_override': task_input.image_override,
        'download': True,
        'release_runtime': False,
        'run_agent': _no_agent,
    })
    if values.get('project_name') is None and task_input.instance is not None:
        values['project_name'] = str(task_input.instance.get('repo') or task_input.repo_name)
    return ParseTaskInput.model_validate(values)


def _select_input(task_input: PipelinePreFeatureTaskInput, graph_path: str) -> SelectTaskInput:
    host_out_dir = Path(task_input.output_dir) / task_input.repo_name
    values = dict(task_input.stage_configs.get('select', {}))
    values.update({
        'instance': task_input.instance,
        'instance_path': '',
        'repo_name': task_input.repo_name,
        'graph_path': graph_path,
        # Pre-feature never consumes an LLM-produced seed policy, including
        # one left by a prior legacy run in the same output directory.
        'seed_policy_path': str(host_out_dir / '.pipeline_no_seed_policy'),
        'output_dir': task_input.output_dir,
        'repo_dir': task_input.repo_dir,
        'base_commit': task_input.base_commit,
        'swe_any_dir': task_input.swe_any_dir,
        'heuristic_filename': constants.SELECT_HEURISTIC_FILENAME,
        'agentic_filename': constants.SELECT_AGENTIC_FILENAME,
        'heuristic_jsonl_filename': constants.SELECT_HEURISTIC_JSONL_FILENAME,
        'agentic_jsonl_filename': constants.SELECT_AGENTIC_JSONL_FILENAME,
        'unified_jsonl_filename': constants.SELECT_JSONL_FILENAME,
        'runtime_name': task_input.runtime_name,
        'image': task_input.image,
        'image_override': task_input.image_override,
        'max_workers': 1,
        'release_runtime': False,
        'run_agent': _no_agent,
    })
    return SelectTaskInput.model_validate(values)


@task_registry.register('swe_any_pipeline_pre_feature')
async def run_swe_any_pipeline_pre_feature(
    task_input: PipelinePreFeatureTaskInput,
) -> PipelinePreFeatureTaskOutput:
    """Run Parse and every deterministic Select slice in one repo runtime."""
    apply_instance(task_input)
    ctx = require_context()
    if ctx.runtime_provider is None:
        raise RuntimeError('swe_any_pipeline_pre_feature requires a runtime provider.')

    image = task_input.image_override or task_input.image
    if image:
        ctx.runtime_provider.configure(task_input.runtime_name, image=image)

    runtime_acquired = False
    try:
        runtime = await ctx.get_runtime(task_input.runtime_name)
        runtime_acquired = True
        await runtime.ws_connect()
        await reset_repo(runtime, task_input.repo_dir, base_commit=task_input.base_commit)

        parse_result = await run_parse_operation(_parse_input(task_input), runtime)
        graph_path = parse_result.get('graph_path')
        if not isinstance(graph_path, str) or not graph_path:
            raise RuntimeError('Parse did not produce a downloaded host graph')

        select_input = _select_input(task_input, graph_path)
        context = prepare_select_heuristic_context(select_input)
        seeds: list[str] = context['seeds']
        heuristic_rows: list[dict[str, Any]] = []
        manifest_rows: list[dict[str, Any]] = []
        errors: list[str] = []
        settings = {name: getattr(select_input, name) for name in _SELECT_MANIFEST_SETTINGS}

        for index, seed in enumerate(seeds):
            row, error = run_select_heuristic_slice(select_input, context, index=index, seed=seed)
            heuristic_rows.append(row)
            if error:
                errors.append(error)
            slice_dir = Path(task_input.output_dir) / task_input.repo_name / constants.slice_dirname(index)
            manifest_rows.append({
                'manifest_version': 1,
                'instance': task_input.instance or {},
                'repo_name': task_input.repo_name,
                'index': index,
                'seed': seed,
                'graph_path': graph_path,
                'heuristic_path': str(slice_dir / constants.SELECT_HEURISTIC_FILENAME),
                'heuristic_row': row,
                'heuristic_error': error,
                'select_settings': settings,
            })

        host_out_dir = Path(task_input.output_dir) / task_input.repo_name
        heuristic_path = host_out_dir / constants.SELECT_HEURISTIC_JSONL_FILENAME
        manifest_path = host_out_dir / task_input.manifest_filename
        write_jsonl(heuristic_rows, heuristic_path)
        write_manifest(manifest_path, manifest_rows)
        logger.info(
            'swe_any_pipeline_pre_feature %s: %d seeds, %d manifest features, %d heuristic errors',
            task_input.repo_name,
            len(seeds),
            len(manifest_rows),
            len(errors),
        )
        return PipelinePreFeatureTaskOutput(
            score=float(bool(manifest_rows)),
            repo_name=task_input.repo_name,
            num_features=len(manifest_rows),
            graph_path=graph_path,
            manifest_path=str(manifest_path),
            runtime_error='; '.join(errors) if errors else None,
        )
    finally:
        if runtime_acquired and task_input.release_runtime:
            await _release_quietly(ctx, task_input.runtime_name)


__all__ = ['run_swe_any_pipeline_pre_feature']
