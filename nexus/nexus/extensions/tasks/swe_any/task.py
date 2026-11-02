"""Legacy repo-level orchestration for the six SWE-Any stages.

Runtime lifecycle, repo iteration, aggregate artifacts, and metrics remain here.
Each feature's stage behavior lives in ``stage/<stage>/runner.py`` so the legacy
pipeline and feature-level pipeline share one implementation.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging

from pathlib import Path
from typing import Any

from nexus.core import require_context, task_registry
from nexus.extensions.tasks.swe_any.stage.eval.runner import run_eval_feature
from nexus.extensions.tasks.swe_any.stage.eval.schema import (
    EvalTaskInput,
    EvalTaskOutput,
    EvalVerifierVerdictCode,
    empty_eval_pieces,
    load_masked_features,
)
from nexus.extensions.tasks.swe_any.stage.export.harbor import write_dataset_toml
from nexus.extensions.tasks.swe_any.stage.export.runner import run_export_feature
from nexus.extensions.tasks.swe_any.stage.export.schema import (
    ExportTaskInput,
    ExportTaskOutput,
    empty_export_row,
    export_unified_row,
    load_export_features,
)
from nexus.extensions.tasks.swe_any.stage.issue.runner import run_issue_feature
from nexus.extensions.tasks.swe_any.stage.issue.schema import (
    IssueTaskInput,
    IssueTaskOutput,
    empty_issue_pieces,
    load_issue_features,
)
from nexus.extensions.tasks.swe_any.stage.mask.runner import run_mask_feature
from nexus.extensions.tasks.swe_any.stage.mask.schema import (
    MaskTaskInput,
    MaskTaskOutput,
    empty_mask_pieces,
    load_resolved_features,
)
from nexus.extensions.tasks.swe_any.stage.parse.runner import run_parse_operation, run_seed_policy_pass
from nexus.extensions.tasks.swe_any.stage.parse.schema import ParseTaskInput, ParseTaskOutput
from nexus.extensions.tasks.swe_any.stage.select.runner import (
    prepare_select_heuristic_context,
    run_select_feature,
)
from nexus.extensions.tasks.swe_any.stage.select.schema import (
    SelectAgenticRow,
    SelectStatusRow,
    SelectTaskInput,
    SelectTaskOutput,
    select_unified_rows,
)
from nexus.extensions.tasks.swe_any.utils import constants
from nexus.extensions.tasks.swe_any.utils.artifacts import ordered_rows, write_jsonl
from nexus.extensions.tasks.swe_any.utils.graph import dumps_json, read_json
from nexus.extensions.tasks.swe_any.utils.instance import apply_instance
from nexus.extensions.tasks.swe_any.utils.repo_git import reset_repo
from nexus.extensions.tasks.swe_any.utils.sandbox_files import upload_text


logger = logging.getLogger(__name__)


async def _release_quietly(ctx: Any, runtime_name: str) -> None:
    """Release a runtime by name, swallowing errors from cleanup."""
    with contextlib.suppress(Exception):
        await ctx.runtime_provider.release(name=runtime_name)


def _require_single_worker(task_name: str, max_workers: int, reason: str) -> None:
    if max_workers != 1:
        raise ValueError(f'{task_name} currently requires max_workers=1 ({reason}).')


@task_registry.register('swe_any_parse')
async def run_swe_any_parse(task_input: ParseTaskInput) -> ParseTaskOutput:
    """Index one repo, download its graph, and optionally generate a seed policy."""
    apply_instance(task_input)
    if task_input.instance is not None and task_input.project_name is None:
        task_input.project_name = str(task_input.instance.get('repo') or task_input.repo_name)

    ctx = require_context()
    if ctx.runtime_provider is None:
        raise RuntimeError('swe_any_parse requires a runtime provider.')
    image = task_input.image_override or task_input.image
    if image:
        ctx.runtime_provider.configure(task_input.runtime_name, image=image)

    try:
        runtime = await ctx.get_runtime(task_input.runtime_name)
        await runtime.ws_connect()
        await reset_repo(runtime, task_input.repo_dir, base_commit=task_input.base_commit)
        fields = await run_parse_operation(task_input, runtime)
        if fields['graph_path'] is None:
            policy = {'seed_policy_error': 'seed policy needs the downloaded graph (download=false)'}
            logger.warning('swe_any_parse skipping seed policy for %s: no host graph', task_input.repo_name)
        else:
            ctx.task_config['working_dir'] = task_input.repo_dir
            policy = await run_seed_policy_pass(
                task_input,
                runtime,
                graph_path=fields['graph_path'],
                host_out_dir=Path(task_input.output_dir) / task_input.repo_name,
            )
        return ParseTaskOutput(
            score=1.0 if fields['nodes_internal'] > 0 else 0.0,
            repo_name=task_input.repo_name,
            **fields,
            **policy,
        )
    finally:
        if task_input.release_runtime:
            await _release_quietly(ctx, task_input.runtime_name)


@task_registry.register('swe_any_select')
async def run_swe_any_select(task_input: SelectTaskInput) -> SelectTaskOutput:
    """Sample heuristic slices and run the Select agent serially in one repo runtime."""
    apply_instance(task_input)
    if not task_input.graph_path:
        task_input.graph_path = str(Path(task_input.output_dir) / task_input.repo_name / constants.GRAPH_FILENAME)

    ctx = require_context()
    if ctx.runtime_provider is None:
        raise RuntimeError('swe_any_select requires a runtime provider.')
    if task_input.max_workers != 1:
        raise ValueError(
            'swe_any_select currently requires max_workers=1. Concurrent Claude Code agents in one runtime '
            'each register an LLMRequestHandler for the same runtime-level llm_request event, which can duplicate '
            'model calls and responses. The semaphore/gather orchestration is retained for re-enabling concurrency '
            'after the handler is shared per runtime.'
        )

    host_out_dir = Path(task_input.output_dir) / task_input.repo_name
    host_out_dir.mkdir(parents=True, exist_ok=True)
    # Missing/corrupt input, an empty filtered graph, or unresolved seeds remain preflight failures.
    select_context = prepare_select_heuristic_context(task_input)
    seeds: list[str] = select_context['seeds']

    image = task_input.image_override or task_input.image
    if image:
        ctx.runtime_provider.configure(task_input.runtime_name, image=image)

    heuristic_rows: dict[int, dict[str, Any]] = {}
    agentic_rows: dict[int, dict[str, Any]] = {}
    status_rows: dict[int, dict[str, Any]] = {}
    errors: list[str] = []
    semaphore = asyncio.Semaphore(max(1, task_input.max_workers))
    sandbox_graph_path = constants.sandbox_path(constants.GRAPH_FILENAME, swe_any_dir=task_input.swe_any_dir)

    async def select_one(runtime: Any, index: int, seed: str) -> None:
        async with semaphore:
            heuristic, agentic, status, error = await run_select_feature(
                task_input,
                runtime,
                select_context,
                index=index,
                seed=seed,
            )
            heuristic_rows[index] = heuristic
            agentic_rows[index] = agentic
            status_rows[index] = status
            if error:
                errors.append(error)

    try:
        runtime = await ctx.get_runtime(task_input.runtime_name)
        await runtime.ws_connect()
        ctx.task_config['working_dir'] = task_input.repo_dir
        await reset_repo(runtime, task_input.repo_dir, base_commit=task_input.base_commit)
        # The agent only sees the already-filtered working graph.
        await upload_text(runtime, sandbox_graph_path, dumps_json(select_context['graph']))
        await asyncio.gather(*(select_one(runtime, index, seed) for index, seed in enumerate(seeds)))

        heuristic_path = host_out_dir / task_input.heuristic_jsonl_filename
        agentic_path = host_out_dir / task_input.agentic_jsonl_filename
        unified_path = host_out_dir / task_input.unified_jsonl_filename
        write_jsonl(ordered_rows(heuristic_rows), heuristic_path)
        write_jsonl(ordered_rows(agentic_rows), agentic_path)
        write_jsonl(
            select_unified_rows(
                ordered_rows(heuristic_rows),
                ordered_rows(agentic_rows),
                ordered_rows(status_rows),
            ),
            unified_path,
        )

        completed = sum(1 for row in status_rows.values() if row.get('completed'))
        num_agentic = sum(1 for row in agentic_rows.values() if row.get('success'))
        abandoned = sum(1 for row in agentic_rows.values() if row.get('error'))
        agentic_present = [row for row in agentic_rows.values() if row.get('success') and row.get('agentic_nodes')]
        cover_rate_mean = (
            sum(float(row.get('cover_rate') or 0.0) for row in agentic_present) / len(agentic_present)
            if agentic_present
            else 0.0
        )
        # Per-slice heuristic failures stay soft; acquire/upload/aggregate failures still propagate.
        return SelectTaskOutput(
            score=float(num_agentic > 0),
            repo_name=task_input.repo_name,
            num_subgraphs=len(seeds),
            num_completed=completed,
            num_agentic=num_agentic,
            num_abandoned=abandoned,
            cover_rate_mean=cover_rate_mean,
            seeds=seeds,
            heuristic_path=str(heuristic_path),
            agentic_path=str(agentic_path),
            unified_path=str(unified_path),
            runtime_error='; '.join(errors) if errors else None,
        )
    finally:
        if task_input.release_runtime:
            await _release_quietly(ctx, task_input.runtime_name)


@task_registry.register('swe_any_mask')
async def run_swe_any_mask(task_input: MaskTaskInput) -> MaskTaskOutput:
    """Build and refine one feature mask per resolved Select row."""
    apply_instance(task_input)
    if not task_input.graph_path:
        task_input.graph_path = str(Path(task_input.output_dir) / task_input.repo_name / constants.GRAPH_FILENAME)
    if not task_input.select_path:
        task_input.select_path = str(
            Path(task_input.output_dir) / task_input.repo_name / constants.SELECT_JSONL_FILENAME
        )

    ctx = require_context()
    if ctx.runtime_provider is None:
        raise RuntimeError('swe_any_mask requires a runtime provider.')
    _require_single_worker('swe_any_mask', task_input.max_workers, 'one Claude Code agent per runtime')

    host_out_dir = Path(task_input.output_dir) / task_input.repo_name
    host_out_dir.mkdir(parents=True, exist_ok=True)
    graph = read_json(task_input.graph_path)
    features = load_resolved_features(Path(task_input.select_path), max_features=task_input.max_features)
    if not features:
        return MaskTaskOutput(
            score=0.0,
            repo_name=task_input.repo_name,
            num_features=0,
            runtime_error=f'no resolved features in {task_input.select_path}',
        )

    image = task_input.image_override or task_input.image
    if image:
        ctx.runtime_provider.configure(task_input.runtime_name, image=image)

    heuristic_rows: dict[int, dict[str, Any]] = {}
    agentic_rows: dict[int, dict[str, Any]] = {}
    unified_rows: dict[int, dict[str, Any]] = {}
    errors: list[str] = []
    sandbox_graph_path = constants.sandbox_path(task_input.graph_filename, swe_any_dir=task_input.swe_any_dir)

    try:
        runtime = await ctx.get_runtime(task_input.runtime_name)
        await runtime.ws_connect()
        ctx.task_config['working_dir'] = task_input.repo_dir
        await reset_repo(runtime, task_input.repo_dir, base_commit=task_input.base_commit)
        await upload_text(runtime, sandbox_graph_path, dumps_json(graph))

        for index, feature_desc, feature_node_ids in features:
            # Runner-handled invoke/refine failures are soft; this catch preserves an explicit
            # per-feature failure row for unexpected prompt, file, or orchestration failures.
            try:
                heuristic, agentic, unified, error = await run_mask_feature(
                    task_input,
                    runtime,
                    graph,
                    index=index,
                    feature_desc=feature_desc,
                    feature_node_ids=feature_node_ids,
                )
                heuristic_rows[index] = heuristic
                agentic_rows[index] = agentic
                unified_rows[index] = unified
                if error:
                    errors.append(error)
            except Exception as exc:  # noqa: BLE001 - one feature never sinks the repo run
                message = f'{type(exc).__name__}: {exc}'
                logger.exception('swe_any_mask feature %s failed for %s', index, task_input.repo_name)
                errors.append(f'feature {index} failed: {message}')
                pieces = empty_mask_pieces(index, feature_node_ids, message)
                heuristic_rows[index] = pieces['heuristic']
                agentic_rows[index] = pieces['agentic']
                unified_rows[index] = pieces['unified']

        unified_path = host_out_dir / task_input.unified_jsonl_filename
        write_jsonl(ordered_rows(heuristic_rows), host_out_dir / task_input.heuristic_jsonl_filename)
        write_jsonl(ordered_rows(agentic_rows), host_out_dir / task_input.agentic_jsonl_filename)
        write_jsonl(ordered_rows(unified_rows), unified_path)

        num_applicable = sum(1 for row in unified_rows.values() if row.get('success'))
        agentic_indices = [row['index'] for row in unified_rows.values() if row.get('source') == 'agentic']
        num_agentic = len(agentic_indices)
        num_agent_failed = sum(
            1
            for row in unified_rows.values()
            if any(str(warning).startswith('refine agent failed:') for warning in (row.get('warnings') or []))
        )
        cover_rate_mean = (
            sum(float(agentic_rows.get(index, {}).get('cover_rate') or 0.0) for index in agentic_indices) / num_agentic
            if num_agentic
            else 0.0
        )
        return MaskTaskOutput(
            score=float(num_applicable > 0),
            repo_name=task_input.repo_name,
            num_features=len(features),
            num_applicable=num_applicable,
            num_agentic=num_agentic,
            num_agent_failed=num_agent_failed,
            cover_rate_mean=cover_rate_mean,
            unified_path=str(unified_path),
            runtime_error='; '.join(errors) if errors else None,
        )
    finally:
        if task_input.release_runtime:
            await _release_quietly(ctx, task_input.runtime_name)


@task_registry.register('swe_any_eval')
async def run_swe_any_eval(task_input: EvalTaskInput) -> EvalTaskOutput:
    """Build each masked feature's hidden-test oracle."""
    apply_instance(task_input)
    if not task_input.graph_path:
        task_input.graph_path = str(Path(task_input.output_dir) / task_input.repo_name / constants.GRAPH_FILENAME)
    if not task_input.mask_path:
        task_input.mask_path = str(Path(task_input.output_dir) / task_input.repo_name / constants.MASK_JSONL_FILENAME)

    ctx = require_context()
    if ctx.runtime_provider is None:
        raise RuntimeError('swe_any_eval requires a runtime provider.')
    _require_single_worker('swe_any_eval', task_input.max_workers, 'one Claude Code agent per runtime')

    host_out_dir = Path(task_input.output_dir) / task_input.repo_name
    host_out_dir.mkdir(parents=True, exist_ok=True)
    graph = read_json(task_input.graph_path)
    features = load_masked_features(Path(task_input.mask_path), max_features=task_input.max_features)
    if not features:
        return EvalTaskOutput(
            score=0.0,
            repo_name=task_input.repo_name,
            num_features=0,
            runtime_error=f'no masked features in {task_input.mask_path}',
        )

    image = task_input.image_override or task_input.image
    if image:
        ctx.runtime_provider.configure(task_input.runtime_name, image=image)

    heuristic_rows: dict[int, dict[str, Any]] = {}
    agentic_rows: dict[int, dict[str, Any]] = {}
    unified_rows: dict[int, dict[str, Any]] = {}
    errors: list[str] = []
    sandbox_graph_path = constants.sandbox_path(task_input.graph_filename, swe_any_dir=task_input.swe_any_dir)

    try:
        runtime = await ctx.get_runtime(task_input.runtime_name)
        await runtime.ws_connect()
        ctx.task_config['working_dir'] = task_input.repo_dir
        await reset_repo(runtime, task_input.repo_dir, base_commit=task_input.base_commit)
        await upload_text(runtime, sandbox_graph_path, dumps_json(graph))

        for index, feature, feature_mask_patch, feature_gold_patch in features:
            try:
                heuristic, agentic, unified, error = await run_eval_feature(
                    task_input,
                    runtime,
                    graph,
                    index=index,
                    feature=feature,
                    feature_mask_patch=feature_mask_patch,
                    feature_gold_patch=feature_gold_patch,
                )
                heuristic_rows[index] = heuristic
                agentic_rows[index] = agentic
                unified_rows[index] = unified
                if error:
                    errors.append(error)
            except Exception as exc:  # noqa: BLE001 - one feature never sinks the repo run
                message = f'{type(exc).__name__}: {exc}'
                logger.exception('swe_any_eval feature %s failed for %s', index, task_input.repo_name)
                errors.append(f'feature {index} failed: {message}')
                pieces = empty_eval_pieces(index, message)
                heuristic_rows[index] = pieces['heuristic']
                agentic_rows[index] = pieces['agentic']
                unified_rows[index] = pieces['unified']

        unified_path = host_out_dir / task_input.unified_jsonl_filename
        write_jsonl(ordered_rows(heuristic_rows), host_out_dir / task_input.heuristic_jsonl_filename)
        write_jsonl(ordered_rows(agentic_rows), host_out_dir / task_input.agentic_jsonl_filename)
        write_jsonl(ordered_rows(unified_rows), unified_path)

        num_applicable = sum(1 for row in unified_rows.values() if row.get('success'))
        num_agentic = sum(1 for row in unified_rows.values() if row.get('source') == 'agentic')
        num_agent_failed = sum(
            1
            for row in unified_rows.values()
            if any(
                str(warning).startswith(('oracle agent failed:', 'oracle verifier failed:'))
                for warning in (row.get('warnings') or [])
            )
        )
        num_verifier_rejected = sum(
            1
            for row in unified_rows.values()
            if isinstance(row.get('verifier'), dict) and row['verifier'].get('verdict') != EvalVerifierVerdictCode.CLEAN
        )
        return EvalTaskOutput(
            score=float(num_applicable > 0),
            repo_name=task_input.repo_name,
            num_features=len(features),
            num_applicable=num_applicable,
            num_agentic=num_agentic,
            num_agent_failed=num_agent_failed,
            num_verifier_rejected=num_verifier_rejected,
            unified_path=str(unified_path),
            runtime_error='; '.join(errors) if errors else None,
        )
    finally:
        if task_input.release_runtime:
            await _release_quietly(ctx, task_input.runtime_name)


@task_registry.register('swe_any_issue')
async def run_swe_any_issue(task_input: IssueTaskInput) -> IssueTaskOutput:
    """Write one non-leaking problem statement per masked and oracled feature."""
    apply_instance(task_input)
    base = Path(task_input.output_dir) / task_input.repo_name
    if not task_input.select_path:
        task_input.select_path = str(base / constants.SELECT_JSONL_FILENAME)
    if not task_input.mask_path:
        task_input.mask_path = str(base / constants.MASK_JSONL_FILENAME)
    if not task_input.eval_path:
        task_input.eval_path = str(base / constants.EVAL_JSONL_FILENAME)

    ctx = require_context()
    if ctx.runtime_provider is None:
        raise RuntimeError('swe_any_issue requires a runtime provider.')
    _require_single_worker('swe_any_issue', task_input.max_workers, 'one Claude Code agent per runtime')

    base.mkdir(parents=True, exist_ok=True)
    features = load_issue_features(
        Path(task_input.select_path),
        Path(task_input.mask_path),
        Path(task_input.eval_path),
        feature_indices=task_input.feature_indices,
        max_features=task_input.max_features,
    )
    if not features:
        return IssueTaskOutput(
            score=0.0,
            repo_name=task_input.repo_name,
            num_features=0,
            runtime_error=(
                f'no masked+oracled features to write issues for '
                f'(join of {task_input.mask_path} + {task_input.eval_path})'
            ),
        )

    image = task_input.image_override or task_input.image
    if image:
        ctx.runtime_provider.configure(task_input.runtime_name, image=image)

    agentic_rows: dict[int, dict[str, Any]] = {}
    unified_rows: dict[int, dict[str, Any]] = {}
    errors: list[str] = []

    try:
        runtime = await ctx.get_runtime(task_input.runtime_name)
        await runtime.ws_connect()
        ctx.task_config['working_dir'] = task_input.repo_dir
        await reset_repo(runtime, task_input.repo_dir, base_commit=task_input.base_commit)
        for feature in features:
            index = feature['index']
            try:
                agentic, unified, error = await run_issue_feature(task_input, runtime, feature)
                agentic_rows[index] = agentic
                unified_rows[index] = unified
                if error:
                    errors.append(error)
            except Exception as exc:  # noqa: BLE001 - one feature never sinks the repo run
                message = f'{type(exc).__name__}: {exc}'
                logger.exception('swe_any_issue feature %s failed for %s', index, task_input.repo_name)
                errors.append(f'feature {index} failed: {message}')
                pieces = empty_issue_pieces(index, message)
                agentic_rows[index] = pieces['agentic']
                unified_rows[index] = pieces['unified']

        unified_path = base / task_input.unified_jsonl_filename
        write_jsonl(ordered_rows(agentic_rows), base / task_input.agentic_jsonl_filename)
        write_jsonl(ordered_rows(unified_rows), unified_path)

        num_applicable = sum(1 for row in unified_rows.values() if row.get('success'))
        num_agent_failed = sum(
            1
            for row in unified_rows.values()
            if any(str(warning).startswith('issue agent failed:') for warning in (row.get('warnings') or []))
        )
        return IssueTaskOutput(
            score=float(num_applicable > 0),
            repo_name=task_input.repo_name,
            num_features=len(features),
            num_applicable=num_applicable,
            num_agent_failed=num_agent_failed,
            unified_path=str(unified_path),
            runtime_error='; '.join(errors) if errors else None,
        )
    finally:
        if task_input.release_runtime:
            await _release_quietly(ctx, task_input.runtime_name)


@task_registry.register('swe_any_export')
async def run_swe_any_export(task_input: ExportTaskInput) -> ExportTaskOutput:
    """Execution-grade and Harbor-emit each feature ready for export."""
    apply_instance(task_input)
    base = Path(task_input.output_dir) / task_input.repo_name
    if not task_input.mask_path:
        task_input.mask_path = str(base / constants.MASK_JSONL_FILENAME)
    if not task_input.eval_path:
        task_input.eval_path = str(base / constants.EVAL_JSONL_FILENAME)
    if not task_input.issue_path:
        task_input.issue_path = str(base / constants.ISSUE_JSONL_FILENAME)

    ctx = require_context()
    if ctx.runtime_provider is None:
        raise RuntimeError('swe_any_export requires a runtime provider.')
    _require_single_worker('swe_any_export', task_input.max_workers, 'single shared repo_dir grades serially')
    if task_input.use_hack_check:
        raise ValueError('use_hack_check is not implemented; unset SWE_ANY_USE_HACK_CHECK.')

    base.mkdir(parents=True, exist_ok=True)
    harbor_root = Path(task_input.output_dir) / constants.HARBOR_DIR
    instance = task_input.instance or {}
    features = load_export_features(
        Path(task_input.mask_path),
        Path(task_input.eval_path),
        Path(task_input.issue_path),
        max_features=task_input.max_features,
    )
    if not features:
        return ExportTaskOutput(
            score=0.0,
            repo_name=task_input.repo_name,
            num_features=0,
            runtime_error=f'no exportable features (join of mask + eval + {task_input.issue_path})',
        )

    image = task_input.image_override or task_input.image
    if image:
        ctx.runtime_provider.configure(task_input.runtime_name, image=image)

    # Shared metadata belongs to the legacy outer task, never to a feature runner.
    write_dataset_toml(harbor_root)
    unified_rows: dict[int, dict[str, Any]] = {}
    instances_out: dict[int, dict[str, Any]] = {}
    errors: list[str] = []

    try:
        runtime = await ctx.get_runtime(task_input.runtime_name)
        await runtime.ws_connect()
        ctx.task_config['working_dir'] = task_input.repo_dir
        await reset_repo(
            runtime,
            task_input.repo_dir,
            base_commit=task_input.base_commit,
        )
        for feature in features:
            index = feature['index']
            try:
                unified, instance_row, error = await run_export_feature(
                    task_input,
                    runtime,
                    feature,
                    instance=instance,
                    harbor_root=harbor_root,
                )
                unified_rows[index] = unified
                if instance_row is not None:
                    instances_out[index] = instance_row
                if error:
                    errors.append(error)
            except Exception as exc:  # noqa: BLE001 - one feature never sinks the repo run
                message = f'{type(exc).__name__}: {exc}'
                logger.exception('swe_any_export feature %s failed for %s', index, task_input.repo_name)
                errors.append(f'feature {index} failed: {message}')
                row = empty_export_row(index)
                row['runtime_error'] = message
                unified_rows[index] = export_unified_row(row)

        unified_path = base / constants.EXPORT_JSONL_FILENAME
        write_jsonl(ordered_rows(unified_rows), unified_path)
        instances_path = base / constants.EXPORT_INSTANCES_FILENAME
        write_jsonl(ordered_rows(instances_out), instances_path)

        num_exported = sum(1 for row in unified_rows.values() if row.get('success'))
        return ExportTaskOutput(
            score=float(num_exported > 0),
            repo_name=task_input.repo_name,
            num_features=len(features),
            num_exported=num_exported,
            harbor_dir=str(harbor_root),
            unified_path=str(unified_path),
            runtime_error='; '.join(errors) if errors else None,
        )
    finally:
        if task_input.release_runtime:
            await _release_quietly(ctx, task_input.runtime_name)


__all__ = [
    'EvalTaskInput',
    'EvalTaskOutput',
    'ExportTaskInput',
    'ExportTaskOutput',
    'IssueTaskInput',
    'IssueTaskOutput',
    'MaskTaskInput',
    'MaskTaskOutput',
    'ParseTaskInput',
    'ParseTaskOutput',
    'SelectAgenticRow',
    'SelectStatusRow',
    'SelectTaskInput',
    'SelectTaskOutput',
    'run_swe_any_eval',
    'run_swe_any_export',
    'run_swe_any_issue',
    'run_swe_any_mask',
    'run_swe_any_parse',
    'run_swe_any_select',
]
