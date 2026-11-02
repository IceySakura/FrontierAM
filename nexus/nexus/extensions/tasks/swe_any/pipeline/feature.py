"""One-feature, one-runtime SWE-Any pipeline orchestration.

Feature task statuses distinguish a completed valid export (``completed``), a stage-level
business outcome that blocks the funnel (``business_failure``), and an
unexpected execution failure (``runtime_failure``).  Every stage entry uses the
existing stage row contracts; no repo-level aggregate is written here.
"""

from __future__ import annotations

import contextlib
import logging

from pathlib import Path
from typing import Any

from nexus.core import require_context, task_registry
from nexus.extensions.tasks.swe_any.pipeline.manifest import write_feature_result
from nexus.extensions.tasks.swe_any.pipeline.schema import PipelineFeatureTaskInput, PipelineFeatureTaskOutput
from nexus.extensions.tasks.swe_any.stage.eval.runner import run_eval_feature
from nexus.extensions.tasks.swe_any.stage.eval.schema import EvalTaskInput, empty_eval_pieces
from nexus.extensions.tasks.swe_any.stage.export.runner import run_export_feature
from nexus.extensions.tasks.swe_any.stage.export.schema import (
    ExportTaskInput,
    empty_export_row,
    export_unified_row,
)
from nexus.extensions.tasks.swe_any.stage.issue.runner import run_issue_feature
from nexus.extensions.tasks.swe_any.stage.issue.schema import IssueTaskInput, empty_issue_pieces
from nexus.extensions.tasks.swe_any.stage.mask.runner import run_mask_feature
from nexus.extensions.tasks.swe_any.stage.mask.schema import MaskTaskInput, empty_mask_pieces
from nexus.extensions.tasks.swe_any.stage.select.runner import (
    prepare_select_heuristic_context,
    run_select_agentic_slice,
)
from nexus.extensions.tasks.swe_any.stage.select.schema import (
    SelectAgenticRow,
    SelectStatusRow,
    SelectTaskInput,
    select_unified_rows,
)
from nexus.extensions.tasks.swe_any.utils import constants
from nexus.extensions.tasks.swe_any.utils.graph import dumps_json
from nexus.extensions.tasks.swe_any.utils.instance import apply_instance
from nexus.extensions.tasks.swe_any.utils.repo_git import reset_repo
from nexus.extensions.tasks.swe_any.utils.sandbox_files import upload_text


logger = logging.getLogger(__name__)

_STAGE_ORDER = ('select', 'mask', 'eval', 'issue', 'export')
_INPUT_TYPES = {
    'select': SelectTaskInput,
    'mask': MaskTaskInput,
    'eval': EvalTaskInput,
    'issue': IssueTaskInput,
    'export': ExportTaskInput,
}
_FIXED_FIELDS = {
    'graph_filename': constants.GRAPH_FILENAME,
    'heuristic_filename': '',
    'agentic_filename': '',
    'heuristic_jsonl_filename': '',
    'agentic_jsonl_filename': '',
    'unified_jsonl_filename': '',
}
_STAGE_FILENAMES = {
    'select': (
        constants.SELECT_HEURISTIC_FILENAME,
        constants.SELECT_AGENTIC_FILENAME,
        constants.SELECT_HEURISTIC_JSONL_FILENAME,
        constants.SELECT_AGENTIC_JSONL_FILENAME,
        constants.SELECT_JSONL_FILENAME,
    ),
    'mask': (
        constants.MASK_HEURISTIC_FILENAME,
        constants.MASK_AGENTIC_FILENAME,
        constants.MASK_HEURISTIC_JSONL_FILENAME,
        constants.MASK_AGENTIC_JSONL_FILENAME,
        constants.MASK_JSONL_FILENAME,
    ),
    'eval': (
        constants.EVAL_HEURISTIC_FILENAME,
        constants.EVAL_AGENTIC_FILENAME,
        constants.EVAL_HEURISTIC_JSONL_FILENAME,
        constants.EVAL_AGENTIC_JSONL_FILENAME,
        constants.EVAL_JSONL_FILENAME,
    ),
    'issue': (
        '',
        constants.ISSUE_AGENTIC_FILENAME,
        '',
        constants.ISSUE_AGENTIC_JSONL_FILENAME,
        constants.ISSUE_JSONL_FILENAME,
    ),
    'export': ('', '', '', '', constants.EXPORT_JSONL_FILENAME),
}


async def _release_quietly(ctx: Any, runtime_name: str) -> None:
    with contextlib.suppress(Exception):
        await ctx.release_runtime(name=runtime_name)


def _feature_result_path(task_input: PipelineFeatureTaskInput) -> Path:
    return (
        Path(task_input.output_dir)
        / task_input.repo_name
        / constants.slice_dirname(task_input.index)
        / constants.FEATURE_RESULT_FILENAME
    )


def _artifact_paths(task_input: PipelineFeatureTaskInput, result_path: Path) -> dict[str, str]:
    slice_dir = result_path.parent
    return {
        'graph_path': task_input.graph_path,
        'manifest_path': task_input.manifest_path,
        'slice_dir': str(slice_dir),
        'feature_result_path': str(result_path),
        'select_heuristic_path': str(slice_dir / constants.SELECT_HEURISTIC_FILENAME),
        'select_agentic_path': str(slice_dir / constants.SELECT_AGENTIC_FILENAME),
        'mask_heuristic_path': str(slice_dir / constants.MASK_HEURISTIC_FILENAME),
        'mask_agentic_path': str(slice_dir / constants.MASK_AGENTIC_FILENAME),
        'eval_heuristic_path': str(slice_dir / constants.EVAL_HEURISTIC_FILENAME),
        'eval_agentic_path': str(slice_dir / constants.EVAL_AGENTIC_FILENAME),
        'eval_verifier_path': str(slice_dir / constants.EVAL_AGENTIC_VERIFIER_FILENAME),
        'issue_agentic_path': str(slice_dir / constants.ISSUE_AGENTIC_FILENAME),
        'grade_pregold_path': str(slice_dir / 'grade_pregold.log'),
        'grade_postgold_path': str(slice_dir / 'grade_postgold.log'),
    }


def _stage_input(stage: str, task_input: PipelineFeatureTaskInput) -> Any:
    model = _INPUT_TYPES[stage]
    values = dict(task_input.stage_configs.get(stage, {}))
    if stage == 'select':
        # These values describe the graph filtering used to create the manifest
        # row and therefore win over feature-time configuration.
        values.update(task_input.select_settings)

    base = Path(task_input.output_dir) / task_input.repo_name
    owned: dict[str, Any] = {
        'instance': task_input.instance,
        'instance_path': '',
        'repo_name': task_input.repo_name,
        'output_dir': task_input.output_dir,
        'repo_dir': task_input.repo_dir,
        'base_commit': task_input.base_commit,
        'swe_any_dir': task_input.swe_any_dir,
        'runtime_name': task_input.runtime_name,
        'image': task_input.image,
        'image_override': task_input.image_override,
        'release_runtime': False,
        'max_workers': 1,
        'graph_path': task_input.graph_path,
        # Pre-feature intentionally does not produce an LLM seed policy. Ignore
        # any stale legacy artifact while reconstructing the Select context.
        'seed_policy_path': str(base / '.pipeline_no_seed_policy'),
        'select_path': str(base / constants.SELECT_JSONL_FILENAME),
        'mask_path': str(base / constants.MASK_JSONL_FILENAME),
        'eval_path': str(base / constants.EVAL_JSONL_FILENAME),
        'issue_path': str(base / constants.ISSUE_JSONL_FILENAME),
        'run_agent': task_input.run_agent,
    }
    names = _STAGE_FILENAMES[stage]
    fixed = dict(_FIXED_FIELDS)
    for field, value in zip(
        (
            'heuristic_filename',
            'agentic_filename',
            'heuristic_jsonl_filename',
            'agentic_jsonl_filename',
            'unified_jsonl_filename',
        ),
        names,
        strict=True,
    ):
        if value:
            fixed[field] = value
    owned.update(fixed)
    values.update({name: value for name, value in owned.items() if name in model.model_fields})
    return model.model_validate(values)


def _select_pieces(
    index: int,
    heuristic: dict[str, Any],
    *,
    error: str | None = None,
    runtime_error: str | None = None,
) -> dict[str, Any]:
    agentic = SelectAgenticRow(index=index, error=error).model_dump()
    status = SelectStatusRow(index=index, completed=False, runtime_error=runtime_error).model_dump()
    unified = select_unified_rows([heuristic], [agentic], [status])[0]
    return {'heuristic': heuristic, 'agentic': agentic, 'status': status, 'unified': unified}


def _blocked_pieces(stage: str, index: int, feature_node_ids: list[str], message: str) -> dict[str, Any]:
    if stage == 'mask':
        pieces = empty_mask_pieces(index, feature_node_ids)
    elif stage == 'eval':
        pieces = empty_eval_pieces(index)
    elif stage == 'issue':
        pieces = empty_issue_pieces(index)
    elif stage == 'export':
        pieces = {'unified': export_unified_row(empty_export_row(index))}
    else:
        raise ValueError(f'cannot build blocked pieces for {stage}')
    pieces['unified']['error'] = message
    return pieces


def _runtime_failure_pieces(stage: str, index: int, feature_node_ids: list[str], message: str) -> dict[str, Any]:
    if stage == 'mask':
        return empty_mask_pieces(index, feature_node_ids, message)
    if stage == 'eval':
        return empty_eval_pieces(index, message)
    if stage == 'issue':
        return empty_issue_pieces(index, message)
    if stage == 'export':
        row = empty_export_row(index)
        row['runtime_error'] = message
        return {'unified': export_unified_row(row)}
    raise ValueError(f'cannot build runtime failure pieces for {stage}')


def _block_after(
    result: dict[str, Any],
    failed_stage: str,
    feature_node_ids: list[str],
    reason: str,
) -> None:
    failed_index = _STAGE_ORDER.index(failed_stage)
    for stage in _STAGE_ORDER[failed_index + 1 :]:
        result['stages'][stage] = _blocked_pieces(
            stage,
            int(result['index']),
            feature_node_ids,
            f'blocked by {failed_stage}: {reason}',
        )


def _row_failure(stage: str, row: dict[str, Any]) -> tuple[str, str]:
    if row.get('runtime_error'):
        return 'runtime_failure', f'{stage} runtime failure: {row["runtime_error"]}'
    if row.get('error'):
        return 'business_failure', f'{stage} business failure: {row["error"]}'
    warnings = '; '.join(str(warning) for warning in (row.get('warnings') or []))
    suffix = f': {warnings}' if warnings else ''
    return 'business_failure', f'{stage} produced no usable output{suffix}'


def _terminal_output(result: dict[str, Any], result_path: Path) -> PipelineFeatureTaskOutput:
    instance_row = result.get('instance_row')
    instance_id = instance_row.get('instance_id') if isinstance(instance_row, dict) else None
    if instance_id is None:
        export = (result.get('stages', {}).get('export') or {}).get('unified') or {}
        instance_id = export.get('instance_id') or None
    return PipelineFeatureTaskOutput(
        score=float(result.get('status') == 'completed'),
        repo_name=str(result['repo_name']),
        index=int(result['index']),
        status=str(result['status']),
        terminal_stage=result.get('terminal_stage'),
        feature_result_path=str(result_path),
        instance_id=instance_id,
        error=result.get('error'),
    )


@task_registry.register('swe_any_pipeline_feature')
async def run_swe_any_pipeline_feature(task_input: PipelineFeatureTaskInput) -> PipelineFeatureTaskOutput:
    """Run one manifest feature serially through the existing stage runners."""
    result_path = _feature_result_path(task_input)
    result: dict[str, Any] = {
        'contract_version': 1,
        'repo_name': task_input.repo_name,
        'index': task_input.index,
        'instance': task_input.instance,
        'status': 'running',
        'terminal_stage': None,
        'error': None,
        'stages': {'select': {'heuristic': task_input.heuristic_row}},
        'instance_row': None,
        'artifacts': _artifact_paths(task_input, result_path),
    }
    ctx: Any | None = None
    runtime_acquired = False
    current_stage = 'startup'
    feature_node_ids: list[str] = []

    try:
        apply_instance(task_input)
        result_path = _feature_result_path(task_input)
        result.update({
            'repo_name': task_input.repo_name,
            'instance': task_input.instance,
            'artifacts': _artifact_paths(task_input, result_path),
        })
        write_feature_result(result_path, result)

        heuristic_nodes = task_input.heuristic_row.get('heuristic_nodes')
        if not (
            isinstance(heuristic_nodes, list)
            and heuristic_nodes
            and all(
                isinstance(node, dict) and isinstance(node.get('id'), str) and node['id'] for node in heuristic_nodes
            )
        ):
            if task_input.heuristic_error:
                reason = f'heuristic select failed: {task_input.heuristic_error}'
                result['stages']['select'] = _select_pieces(
                    task_input.index,
                    task_input.heuristic_row,
                    runtime_error=task_input.heuristic_error,
                )
                terminal_status = 'runtime_failure'
            else:
                reason = 'manifest heuristic_row has no usable heuristic_nodes'
                result['stages']['select'] = _select_pieces(task_input.index, task_input.heuristic_row, error=reason)
                terminal_status = 'business_failure'
            _block_after(result, 'select', [], reason)
            result.update(status=terminal_status, terminal_stage='select', error=reason)
            return _terminal_output(result, result_path)

        ctx = require_context()
        if ctx.runtime_provider is None:
            raise RuntimeError('swe_any_pipeline_feature requires a runtime provider.')
        image = task_input.image_override or task_input.image
        if image:
            ctx.runtime_provider.configure(task_input.runtime_name, image=image)

        runtime = await ctx.get_runtime(task_input.runtime_name)
        runtime_acquired = True
        await runtime.ws_connect()
        ctx.task_config['working_dir'] = task_input.repo_dir
        await reset_repo(runtime, task_input.repo_dir, base_commit=task_input.base_commit)

        current_stage = 'select'
        select_input = _stage_input('select', task_input)
        select_context = prepare_select_heuristic_context(select_input)
        sandbox_graph_path = constants.sandbox_path(constants.GRAPH_FILENAME, swe_any_dir=task_input.swe_any_dir)
        await upload_text(runtime, sandbox_graph_path, dumps_json(select_context['graph']))
        select_agentic, select_status = await run_select_agentic_slice(
            select_input,
            runtime,
            select_context,
            task_input.heuristic_row,
        )
        select_unified = select_unified_rows(
            [task_input.heuristic_row],
            [select_agentic],
            [select_status],
        )[0]
        result['stages']['select'] = {
            'heuristic': task_input.heuristic_row,
            'agentic': select_agentic,
            'status': select_status,
            'unified': select_unified,
        }
        selected_nodes = select_unified.get('nodes')
        if select_unified.get('source') != 'agentic' or not isinstance(selected_nodes, list):
            status, reason = _row_failure('select', select_unified)
            _block_after(result, 'select', [], reason)
            result.update(status=status, terminal_stage='select', error=reason)
            return _terminal_output(result, result_path)
        feature_node_ids = [
            node['id']
            for node in selected_nodes
            if isinstance(node, dict) and isinstance(node.get('id'), str) and node['id']
        ]
        if not feature_node_ids:
            reason = 'select produced no usable feature node ids'
            _block_after(result, 'select', [], reason)
            result.update(status='business_failure', terminal_stage='select', error=reason)
            return _terminal_output(result, result_path)

        current_stage = 'mask'
        mask_heuristic, mask_agentic, mask_unified, _mask_error = await run_mask_feature(
            _stage_input('mask', task_input),
            runtime,
            select_context['graph'],
            index=task_input.index,
            feature_desc=str(select_unified.get('feature') or ''),
            feature_node_ids=feature_node_ids,
        )
        result['stages']['mask'] = {
            'heuristic': mask_heuristic,
            'agentic': mask_agentic,
            'unified': mask_unified,
        }
        masked_nodes = mask_unified.get('nodes')
        if not (mask_unified.get('success') and isinstance(masked_nodes, list) and masked_nodes):
            status, reason = _row_failure('mask', mask_unified)
            _block_after(result, 'mask', feature_node_ids, reason)
            result.update(status=status, terminal_stage='mask', error=reason)
            return _terminal_output(result, result_path)

        current_stage = 'eval'
        eval_heuristic, eval_agentic, eval_unified, _eval_error = await run_eval_feature(
            _stage_input('eval', task_input),
            runtime,
            select_context['graph'],
            index=task_input.index,
            feature=[node for node in masked_nodes if isinstance(node, str)],
            feature_mask_patch=str(mask_unified.get('feature_mask_patch') or ''),
            feature_gold_patch=str(mask_unified.get('feature_gold_patch') or ''),
        )
        result['stages']['eval'] = {
            'heuristic': eval_heuristic,
            'agentic': eval_agentic,
            'unified': eval_unified,
        }
        if not (eval_unified.get('success') and str(eval_unified.get('test_runner_script') or '').strip()):
            status, reason = _row_failure('eval', eval_unified)
            _block_after(result, 'eval', feature_node_ids, reason)
            result.update(status=status, terminal_stage='eval', error=reason)
            return _terminal_output(result, result_path)

        current_stage = 'issue'
        issue_feature = {
            'index': task_input.index,
            'feature_desc': str(select_unified.get('feature') or ''),
            'feature_ids': [node for node in masked_nodes if isinstance(node, str)],
            'feature_mask_patch': str(mask_unified.get('feature_mask_patch') or ''),
            'feature_gold_patch': str(mask_unified.get('feature_gold_patch') or ''),
            'test_mask_patch': str(eval_unified.get('test_mask_patch') or ''),
            'test_gold_patch': str(eval_unified.get('test_gold_patch') or ''),
        }
        issue_agentic, issue_unified, _issue_error = await run_issue_feature(
            _stage_input('issue', task_input),
            runtime,
            issue_feature,
        )
        result['stages']['issue'] = {'agentic': issue_agentic, 'unified': issue_unified}
        if not issue_unified.get('success'):
            status, reason = _row_failure('issue', issue_unified)
            _block_after(result, 'issue', feature_node_ids, reason)
            result.update(status=status, terminal_stage='issue', error=reason)
            return _terminal_output(result, result_path)

        current_stage = 'export'
        test_files = [path for path in (eval_unified.get('test_files') or []) if isinstance(path, str)]
        if not test_files:
            reason = 'export blocked: eval produced no non-empty test_files'
            result['stages']['export'] = _blocked_pieces('export', task_input.index, feature_node_ids, reason)
            result.update(status='business_failure', terminal_stage='export', error=reason)
            return _terminal_output(result, result_path)

        export_input = _stage_input('export', task_input)
        if export_input.use_hack_check:
            raise ValueError('use_hack_check is not implemented; unset SWE_ANY_USE_HACK_CHECK.')
        export_feature = {
            'index': task_input.index,
            'problem_statement': str(issue_unified.get('problem_statement') or ''),
            'feature_mask_patch': str(mask_unified.get('feature_mask_patch') or ''),
            'feature_gold_patch': str(mask_unified.get('feature_gold_patch') or ''),
            'test_mask_patch': str(eval_unified.get('test_mask_patch') or ''),
            'test_gold_patch': str(eval_unified.get('test_gold_patch') or ''),
            'test_runner_script': str(eval_unified.get('test_runner_script') or ''),
            'test_files': test_files,
        }
        export_unified, instance_row, _export_error = await run_export_feature(
            export_input,
            runtime,
            export_feature,
            instance=task_input.instance,
            harbor_root=Path(task_input.output_dir) / constants.HARBOR_DIR,
        )
        result['stages']['export'] = {'unified': export_unified}
        result['instance_row'] = instance_row
        if export_unified.get('harbor_path'):
            result['artifacts']['harbor_path'] = str(export_unified['harbor_path'])
        if not (export_unified.get('success') and instance_row is not None):
            status, reason = _row_failure('export', export_unified)
            result.update(status=status, terminal_stage='export', error=reason)
            return _terminal_output(result, result_path)

        result.update(status='completed', terminal_stage='export', error=None)
        return _terminal_output(result, result_path)
    except Exception as exc:  # noqa: BLE001 - persist a recoverable terminal feature result
        message = f'{type(exc).__name__}: {exc}'
        logger.exception(
            'swe_any_pipeline_feature %s/%s failed in %s',
            result.get('repo_name'),
            constants.slice_dirname(task_input.index),
            current_stage,
        )
        if current_stage == 'select':
            result['stages']['select'] = _select_pieces(
                task_input.index,
                task_input.heuristic_row,
                runtime_error=message,
            )
            _block_after(result, 'select', feature_node_ids, message)
        elif current_stage in _STAGE_ORDER:
            result['stages'][current_stage] = _runtime_failure_pieces(
                current_stage,
                task_input.index,
                feature_node_ids,
                message,
            )
            _block_after(result, current_stage, feature_node_ids, message)
        else:
            result['stages']['select'] = _select_pieces(
                task_input.index,
                task_input.heuristic_row,
                runtime_error=message,
            )
            _block_after(result, 'select', feature_node_ids, f'startup failure: {message}')
        result.update(status='runtime_failure', terminal_stage=current_stage, error=message)
        return _terminal_output(result, result_path)
    finally:
        if result.get('status') == 'running':
            message = f'task interrupted during {current_stage}'
            if current_stage == 'select':
                result['stages']['select'] = _select_pieces(
                    task_input.index,
                    task_input.heuristic_row,
                    runtime_error=message,
                )
                _block_after(result, 'select', feature_node_ids, message)
            elif current_stage in _STAGE_ORDER:
                result['stages'][current_stage] = _runtime_failure_pieces(
                    current_stage,
                    task_input.index,
                    feature_node_ids,
                    message,
                )
                _block_after(result, current_stage, feature_node_ids, message)
            else:
                result['stages']['select'] = _select_pieces(
                    task_input.index,
                    task_input.heuristic_row,
                    runtime_error=message,
                )
                _block_after(result, 'select', feature_node_ids, f'startup interruption: {message}')
            result.update(status='runtime_failure', terminal_stage=current_stage, error=message)
        write_feature_result(result_path, result)
        if ctx is not None and runtime_acquired and task_input.release_runtime:
            await _release_quietly(ctx, task_input.runtime_name)


__all__ = ['run_swe_any_pipeline_feature']
