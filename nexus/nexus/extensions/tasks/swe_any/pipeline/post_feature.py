"""Deterministic host-side reduction for SWE-Any feature pipeline runs."""

from __future__ import annotations

import json
import logging

from pathlib import Path
from typing import Any

from nexus.core import task_registry
from nexus.extensions.tasks.swe_any.pipeline.manifest import read_feature_result
from nexus.extensions.tasks.swe_any.pipeline.schema import PipelinePostFeatureTaskInput, PipelinePostFeatureTaskOutput
from nexus.extensions.tasks.swe_any.stage.eval.schema import EvalUnifiedRow, empty_eval_pieces
from nexus.extensions.tasks.swe_any.stage.export.harbor import write_dataset_toml
from nexus.extensions.tasks.swe_any.stage.export.schema import (
    ExportUnifiedRow,
    empty_export_row,
    export_unified_row,
)
from nexus.extensions.tasks.swe_any.stage.issue.schema import IssueUnifiedRow, empty_issue_pieces
from nexus.extensions.tasks.swe_any.stage.mask.schema import MaskUnifiedRow, empty_mask_pieces
from nexus.extensions.tasks.swe_any.stage.select.schema import (
    SelectAgenticRow,
    SelectStatusRow,
    SelectUnifiedRow,
    select_unified_rows,
)
from nexus.extensions.tasks.swe_any.utils import constants
from nexus.extensions.tasks.swe_any.utils.artifacts import write_jsonl


logger = logging.getLogger(__name__)

_TERMINAL_STATUSES = {'completed', 'business_failure', 'runtime_failure'}
_REPO_ARTIFACTS = {
    'select_heuristic': constants.SELECT_HEURISTIC_JSONL_FILENAME,
    'select_agentic': constants.SELECT_AGENTIC_JSONL_FILENAME,
    'select_unified': constants.SELECT_JSONL_FILENAME,
    'mask_heuristic': constants.MASK_HEURISTIC_JSONL_FILENAME,
    'mask_agentic': constants.MASK_AGENTIC_JSONL_FILENAME,
    'mask_unified': constants.MASK_JSONL_FILENAME,
    'eval_heuristic': constants.EVAL_HEURISTIC_JSONL_FILENAME,
    'eval_agentic': constants.EVAL_AGENTIC_JSONL_FILENAME,
    'eval_unified': constants.EVAL_JSONL_FILENAME,
    'issue_agentic': constants.ISSUE_AGENTIC_JSONL_FILENAME,
    'issue_unified': constants.ISSUE_JSONL_FILENAME,
    'export_unified': constants.EXPORT_JSONL_FILENAME,
}


def _row(piece: Any, *, stage: str, name: str, index: int) -> dict[str, Any]:
    if not isinstance(piece, dict) or piece.get('index') != index:
        raise ValueError(f'{stage}.{name} must be an index-matched row')
    return piece


def _validate_result(payload: dict[str, Any], *, repo_name: str, index: int) -> dict[str, Any]:
    if payload.get('repo_name') != repo_name or payload.get('index') != index:
        raise ValueError('feature result identity does not match its batch record')
    if payload.get('status') not in _TERMINAL_STATUSES:
        raise ValueError(f'feature result has non-terminal status {payload.get("status")!r}')
    stages = payload.get('stages')
    if not isinstance(stages, dict):
        raise ValueError('feature result stages must be a mapping')

    select = stages.get('select')
    mask = stages.get('mask')
    eval_stage = stages.get('eval')
    issue = stages.get('issue')
    export = stages.get('export')
    for stage, pieces in (
        ('select', select),
        ('mask', mask),
        ('eval', eval_stage),
        ('issue', issue),
        ('export', export),
    ):
        if not isinstance(pieces, dict):
            raise ValueError(f'feature result missing {stage} stage rows')

    normalized = {
        'select': {
            'heuristic': _row(select.get('heuristic'), stage='select', name='heuristic', index=index),
            'agentic': SelectAgenticRow.model_validate(
                _row(select.get('agentic'), stage='select', name='agentic', index=index)
            ).model_dump(),
            'status': SelectStatusRow.model_validate(
                _row(select.get('status'), stage='select', name='status', index=index)
            ).model_dump(),
            'unified': SelectUnifiedRow.model_validate(
                _row(select.get('unified'), stage='select', name='unified', index=index)
            ).model_dump(),
        },
        'mask': {
            'heuristic': _row(mask.get('heuristic'), stage='mask', name='heuristic', index=index),
            'agentic': _normalize_mask_agentic(_row(mask.get('agentic'), stage='mask', name='agentic', index=index)),
            'unified': MaskUnifiedRow.model_validate(
                _row(mask.get('unified'), stage='mask', name='unified', index=index)
            ).model_dump(),
        },
        'eval': {
            'heuristic': _row(eval_stage.get('heuristic'), stage='eval', name='heuristic', index=index),
            'agentic': _normalize_eval_agentic(
                _row(eval_stage.get('agentic'), stage='eval', name='agentic', index=index)
            ),
            'unified': EvalUnifiedRow.model_validate(
                _row(eval_stage.get('unified'), stage='eval', name='unified', index=index)
            ).model_dump(),
        },
        'issue': {
            'agentic': _normalize_issue_agentic(_row(issue.get('agentic'), stage='issue', name='agentic', index=index)),
            'unified': IssueUnifiedRow.model_validate(
                _row(issue.get('unified'), stage='issue', name='unified', index=index)
            ).model_dump(),
        },
        'export': {
            'unified': ExportUnifiedRow.model_validate(
                _row(export.get('unified'), stage='export', name='unified', index=index)
            ).model_dump(),
        },
    }
    instance_row = payload.get('instance_row')
    if normalized['export']['unified']['success']:
        if not isinstance(instance_row, dict) or not instance_row.get('instance_id'):
            raise ValueError('successful export is missing its instance row')
    elif instance_row is not None and not isinstance(instance_row, dict):
        raise ValueError('feature result instance_row must be a mapping or null')
    return {
        **payload,
        'stages': normalized,
        'instance_row': instance_row,
    }


def _normalize_mask_agentic(row: dict[str, Any]) -> dict[str, Any]:
    return {
        'index': row['index'],
        'cover_rate': float(row.get('cover_rate') or 0.0),
        'agentic_mask_patch': str(row.get('agentic_mask_patch') or ''),
        'agentic_gold_patch': str(row.get('agentic_gold_patch') or ''),
    }


def _normalize_eval_agentic(row: dict[str, Any]) -> dict[str, Any]:
    return {
        'index': row['index'],
        'agentic_mask_patch': str(row.get('agentic_mask_patch') or ''),
        'agentic_gold_patch': str(row.get('agentic_gold_patch') or ''),
        'agentic_runner_script': str(row.get('agentic_runner_script') or ''),
        'agentic_test_files': [path for path in (row.get('agentic_test_files') or []) if isinstance(path, str)],
        'agentic_verifier': row.get('agentic_verifier') if isinstance(row.get('agentic_verifier'), dict) else None,
    }


def _normalize_issue_agentic(row: dict[str, Any]) -> dict[str, Any]:
    return {'index': row['index'], 'problem_statement': str(row.get('problem_statement') or '')}


def _load_heuristic(output_dir: Path, repo_name: str, index: int) -> tuple[dict[str, Any], str | None]:
    path = output_dir / repo_name / constants.slice_dirname(index) / constants.SELECT_HEURISTIC_FILENAME
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(payload, dict) or payload.get('index') != index:
            raise ValueError('expected an index-matched JSON object')
        return payload, None
    except FileNotFoundError:
        return {'index': index, 'seed': '', 'meta': {}, 'heuristic_nodes': []}, f'heuristic artifact missing: {path}'
    except Exception as exc:  # noqa: BLE001 - corrupt recovery input is represented in rows
        return (
            {'index': index, 'seed': '', 'meta': {}, 'heuristic_nodes': []},
            f'heuristic artifact corrupt: {type(exc).__name__}: {exc}',
        )


def _blocked(stage: str, index: int, nodes: list[str], message: str) -> dict[str, Any]:
    if stage == 'mask':
        pieces = empty_mask_pieces(index, nodes)
    elif stage == 'eval':
        pieces = empty_eval_pieces(index)
    elif stage == 'issue':
        pieces = empty_issue_pieces(index)
    elif stage == 'export':
        pieces = {'unified': export_unified_row(empty_export_row(index))}
    else:
        raise ValueError(f'unknown blocked stage: {stage}')
    pieces['unified']['error'] = message
    return pieces


def _reconstruct(
    run: dict[str, Any],
    *,
    output_dir: Path,
    artifact_problem: str,
) -> dict[str, Any]:
    index = int(run['index'])
    heuristic, heuristic_problem = _load_heuristic(output_dir, str(run['repo_name']), index)
    parts = []
    if run.get('batch_status') != 'success':
        parts.append(
            f'batch task failure ({run.get("batch_status")}): {run.get("batch_error") or "no batch error message"}'
        )
    parts.append(artifact_problem)
    if heuristic_problem:
        parts.append(heuristic_problem)
    root = '; '.join(parts)

    select_agentic = SelectAgenticRow(index=index).model_dump()
    select_status = SelectStatusRow(index=index, completed=False, runtime_error=root).model_dump()
    select_unified = select_unified_rows([heuristic], [select_agentic], [select_status])[0]
    downstream = f'blocked by select recovery failure: {root}'
    return {
        'repo_name': run['repo_name'],
        'index': index,
        'status': 'runtime_failure',
        'terminal_stage': 'select',
        'error': root,
        'stages': {
            'select': {
                'heuristic': heuristic,
                'agentic': select_agentic,
                'status': select_status,
                'unified': select_unified,
            },
            'mask': _blocked('mask', index, [], downstream),
            'eval': _blocked('eval', index, [], downstream),
            'issue': _blocked('issue', index, [], downstream),
            'export': _blocked('export', index, [], downstream),
        },
        'instance_row': None,
    }


def _collect_result(
    run: dict[str, Any],
    *,
    output_dir: Path,
) -> tuple[dict[str, Any], str | None]:
    raw_path = run.get('feature_result_path')
    if not raw_path:
        return _reconstruct(
            run, output_dir=output_dir, artifact_problem='feature_result artifact path missing'
        ), 'missing'
    path = Path(raw_path)
    if not path.is_file():
        return (
            _reconstruct(run, output_dir=output_dir, artifact_problem=f'feature_result artifact missing: {path}'),
            'missing',
        )
    try:
        return (
            _validate_result(
                read_feature_result(path),
                repo_name=str(run['repo_name']),
                index=int(run['index']),
            ),
            None,
        )
    except Exception as exc:  # noqa: BLE001 - malformed terminal output is explicitly reconstructed
        problem = f'feature_result artifact corrupt: {type(exc).__name__}: {exc}'
        return _reconstruct(run, output_dir=output_dir, artifact_problem=problem), 'corrupt'


@task_registry.register('swe_any_pipeline_post_feature')
async def run_swe_any_pipeline_post_feature(
    task_input: PipelinePostFeatureTaskInput,
) -> PipelinePostFeatureTaskOutput:
    """Rebuild every repo aggregate and the global instance index in stable order."""
    output_dir = Path(task_input.output_dir)
    runs = sorted(
        task_input.feature_runs,
        key=lambda run: (str(run['repo_name']), int(run['index']), str(run['run_id'])),
    )
    by_repo: dict[str, dict[str, list[dict[str, Any]]]] = {}
    repo_instances: dict[str, list[dict[str, Any]]] = {}
    global_instances: list[dict[str, Any]] = []
    counts = {
        'batch_failed': 0,
        'missing': 0,
        'corrupt': 0,
        'select': 0,
        'mask': 0,
        'eval': 0,
        'issue': 0,
        'export': 0,
    }

    for run in runs:
        if run.get('batch_status') != 'success':
            counts['batch_failed'] += 1
        result, recovery = _collect_result(run, output_dir=output_dir)
        if recovery is not None:
            counts[recovery] += 1

        repo_name = str(run['repo_name'])
        repo_rows = by_repo.setdefault(repo_name, {name: [] for name in _REPO_ARTIFACTS})
        stages = result['stages']
        repo_rows['select_heuristic'].append(stages['select']['heuristic'])
        repo_rows['select_agentic'].append(stages['select']['agentic'])
        repo_rows['select_unified'].append(stages['select']['unified'])
        repo_rows['mask_heuristic'].append(stages['mask']['heuristic'])
        repo_rows['mask_agentic'].append(stages['mask']['agentic'])
        repo_rows['mask_unified'].append(stages['mask']['unified'])
        repo_rows['eval_heuristic'].append(stages['eval']['heuristic'])
        repo_rows['eval_agentic'].append(stages['eval']['agentic'])
        repo_rows['eval_unified'].append(stages['eval']['unified'])
        repo_rows['issue_agentic'].append(stages['issue']['agentic'])
        repo_rows['issue_unified'].append(stages['issue']['unified'])
        repo_rows['export_unified'].append(stages['export']['unified'])

        counts['select'] += int(bool(stages['select']['unified'].get('source') == 'agentic'))
        counts['mask'] += int(bool(stages['mask']['unified'].get('success')))
        counts['eval'] += int(bool(stages['eval']['unified'].get('success')))
        counts['issue'] += int(bool(stages['issue']['unified'].get('success')))
        counts['export'] += int(bool(stages['export']['unified'].get('success')))

        instance_row = result.get('instance_row')
        if stages['export']['unified'].get('success') and isinstance(instance_row, dict):
            repo_instances.setdefault(repo_name, []).append(instance_row)
            global_instances.append(instance_row)

    for repo_name in sorted(by_repo):
        base = output_dir / repo_name
        for artifact, filename in _REPO_ARTIFACTS.items():
            rows = sorted(by_repo[repo_name][artifact], key=lambda row: int(row['index']))
            write_jsonl(rows, base / filename)
        instances = sorted(repo_instances.get(repo_name, []), key=lambda row: str(row.get('instance_id') or ''))
        write_jsonl(instances, base / constants.EXPORT_INSTANCES_FILENAME)

    global_instances.sort(key=lambda row: str(row.get('instance_id') or ''))
    instances_path = output_dir / constants.EXPORT_INSTANCES_FILENAME
    # All input rows are collected before this write, so input/output path
    # equality cannot truncate the source while it is being read.
    write_jsonl(global_instances, instances_path)
    harbor_root = output_dir / constants.HARBOR_DIR
    write_dataset_toml(harbor_root)
    dataset_toml_path = harbor_root / 'dataset.toml'

    logger.info(
        'swe_any_pipeline_post_feature funnel: features=%d select=%d mask=%d eval=%d issue=%d '
        'export=%d batch_failed=%d missing=%d corrupt=%d',
        len(runs),
        counts['select'],
        counts['mask'],
        counts['eval'],
        counts['issue'],
        counts['export'],
        counts['batch_failed'],
        counts['missing'],
        counts['corrupt'],
    )
    return PipelinePostFeatureTaskOutput(
        score=float(counts['export'] > 0),
        num_repos=len(by_repo),
        num_features=len(runs),
        num_batch_failed=counts['batch_failed'],
        num_missing_results=counts['missing'],
        num_corrupt_results=counts['corrupt'],
        num_select_resolved=counts['select'],
        num_masked=counts['mask'],
        num_oracled=counts['eval'],
        num_issues=counts['issue'],
        num_exported=counts['export'],
        instances_path=str(instances_path),
        harbor_dir=str(harbor_root),
        dataset_toml_path=str(dataset_toml_path),
        runtime_error=None,
    )


__all__ = ['run_swe_any_pipeline_post_feature']
