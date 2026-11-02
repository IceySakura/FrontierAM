#!/usr/bin/env python3
"""Re-evaluate search trajectory JSONL with judge model.

Usage example:
    uv run python nexus/extensions/tasks/search/reevaluate_traj.py \
      --traj workspace/benchmark-run-xxx/traj_gaia_88.jsonl \
      --env-file .env.search \
      --workers 4 \
      --resume
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sqlite3

from pathlib import Path
from typing import Any

import pandas as pd

from nexus.core import ExecutionContext, OpenAIConfig, Registry, openai_client_registry
from nexus.extensions.agents.base.final_answer import generate_final_answer
from nexus.extensions.tasks.search.compute_metrics import compute_metrics
from nexus.extensions.tasks.search.evaluator import EvalMode, SearchEvaluator
from nexus.extensions.tasks.search.generator import _build_client_config
from nexus.extensions.tasks.search.run_benchmarks import DEFAULT_BENCHMARKS, DEFAULT_DATA_DIR, _parse_env_file
from nexus.extensions.tasks.search.utils import filter_excluded_results
from nexus.utils.common import serialize_object


def _default_output_paths(traj_path: Path, eval_mode: EvalMode) -> tuple[Path, Path]:
    mode_suffix = 'judge_llm' if eval_mode == EvalMode.LLM_JUDGE else eval_mode.value
    out_traj = traj_path.with_suffix(f'.{mode_suffix}.jsonl')
    benchmark_name = traj_path.stem.replace('traj_', '')
    out_metrics = traj_path.with_name(f'metrics_{benchmark_name}.{mode_suffix}.json')
    return out_traj, out_metrics


def _default_db_path(traj_path: Path) -> Path:
    benchmark_name = traj_path.stem.replace('traj_', '')
    return traj_path.with_name(f'search_{benchmark_name}.db')


def _default_dataset_path(traj_path: Path) -> Path | None:
    benchmark_name = traj_path.stem.replace('traj_', '')
    dataset_name = DEFAULT_BENCHMARKS.get(benchmark_name)
    if not dataset_name:
        return None
    return DEFAULT_DATA_DIR / dataset_name


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _parse_nexus_config(value: Any) -> Any:
    """Parse the parquet nexus column, handling dict-like and JSON-string values."""
    result = serialize_object(value)
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except (json.JSONDecodeError, ValueError):
            return {}
    return result if isinstance(result, dict) else {}


def _load_widesearch_evaluations_from_db(db_path: Path) -> dict[str, dict[str, Any]]:
    """Load per-run wide_search evaluation configs from batch DB."""
    result: dict[str, dict[str, Any]] = {}
    if not db_path.exists():
        return result

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.cursor()
        cursor.execute('SELECT run_id, config FROM runs')
        for row in cursor.fetchall():
            run_id = str(row['run_id'] or '')
            if not run_id:
                continue
            config_raw = row['config']
            if not config_raw:
                continue
            try:
                config = json.loads(config_raw)
            except json.JSONDecodeError:
                continue
            task_config = config.get('task_config') or {}
            evaluation = task_config.get('evaluation')
            if isinstance(evaluation, dict):
                result[run_id] = evaluation
    finally:
        conn.close()

    return result


def _load_dataset_configs(dataset_path: Path) -> tuple[dict[str, dict[str, Any]], str | None, str | None]:
    """Load per-instance evaluation configs and dataset-level judge config from parquet.

    Returns (widesearch_evaluations, judge_prompt_template, judge_system_prompt).
    """
    empty: tuple[dict[str, dict[str, Any]], str | None, str | None] = ({}, None, None)
    if not dataset_path.exists():
        return empty

    df = pd.read_parquet(dataset_path)
    if 'nexus' not in df.columns:
        return empty

    widesearch_evals: dict[str, dict[str, Any]] = {}
    judge_prompt_template: str | None = None
    judge_system_prompt: str | None = None

    for _, row in df.iterrows():
        nexus_config = _parse_nexus_config(row['nexus'])
        task_config = nexus_config.get('task_config') or {}

        instance_id = str(task_config.get('instance_id') or '')
        evaluation = task_config.get('evaluation')
        if instance_id and isinstance(evaluation, dict):
            widesearch_evals[instance_id] = evaluation

        if judge_prompt_template is None and isinstance(task_config.get('judge_prompt_template'), str):
            judge_prompt_template = task_config['judge_prompt_template']
        if judge_system_prompt is None and isinstance(task_config.get('judge_system_prompt'), str):
            judge_system_prompt = task_config['judge_system_prompt']

    return widesearch_evals, judge_prompt_template, judge_system_prompt


def _build_resume_cache(resume_file: Path) -> dict[str, dict[str, Any]]:
    """Build run_id -> cached evaluation payload from previous output file."""
    result: dict[str, dict[str, Any]] = {}
    if not resume_file.exists():
        return result
    for row in _load_jsonl(resume_file):
        run_id = str(row.get('run_id') or '')
        task_output = row.get('task_output') or {}
        score = task_output.get('score')
        if run_id and isinstance(score, (int, float)):
            cached_payload: dict[str, Any] = {'score': float(score)}
            evaluation_details = task_output.get('evaluation_details')
            if isinstance(evaluation_details, dict):
                cached_payload['evaluation_details'] = evaluation_details
            result[run_id] = cached_payload
    return result


def _load_excluded_instance_ids(args: argparse.Namespace) -> set[str]:
    excluded: set[str] = set()
    if args.exclude_instance_id:
        excluded.update(item.strip() for item in args.exclude_instance_id if item and item.strip())

    if args.exclude_instance_ids_file:
        path = args.exclude_instance_ids_file.resolve()
        if not path.exists():
            raise FileNotFoundError(f'Exclude instance IDs file not found: {path}')
        for raw_line in path.read_text(encoding='utf-8').splitlines():
            line = raw_line.strip()
            if not line:
                continue
            # Allow optional bullet prefix, e.g. "●instance_id".
            line = line.lstrip('●').strip()
            if line:
                excluded.add(line)

    return excluded


def _extract_eval_fields(row: dict[str, Any]) -> tuple[str | None, str | None, str]:
    task_output = row.get('task_output') or {}
    metadata = row.get('metadata') or {}
    prediction = task_output.get('prediction')
    reference = task_output.get('answer') or metadata.get('answer')
    question = metadata.get('question') or ''
    if not isinstance(prediction, str) or not isinstance(reference, str):
        return None, None, question
    return prediction, reference.strip().lower(), question


async def _recover_prediction_from_trajectory(
    sem: asyncio.Semaphore,
    row: dict[str, Any],
    index: int,
    total: int,
) -> str | None:
    task_output = row.get('task_output') or {}
    prediction = task_output.get('prediction')
    if isinstance(prediction, str) and prediction.strip() and prediction.strip().lower() != 'none':
        return prediction

    trajectory = task_output.get('trajectory')
    if not isinstance(trajectory, list) or not trajectory:
        return None

    try:
        async with sem:
            extracted_answer = await generate_final_answer(trajectory=trajectory, return_type='str')
    except Exception as e:
        print(f'[{index + 1}/{total}] failed to recover prediction from trajectory: {e}', flush=True)
        return None

    if not isinstance(extracted_answer, str) or not extracted_answer.strip():
        return None

    recovered_prediction = extracted_answer.strip()
    task_output['prediction'] = recovered_prediction
    row['task_output'] = task_output
    print(f'[{index + 1}/{total}] recovered prediction from trajectory', flush=True)
    return recovered_prediction


async def _eval_one(
    sem: asyncio.Semaphore,
    row: dict[str, Any],
    index: int,
    total: int,
    use_cache: dict[str, dict[str, Any]],
    evaluator: SearchEvaluator,
    widesearch_evaluations: dict[str, dict[str, Any]] | None = None,
) -> tuple[int, dict[str, Any], bool, bool]:
    run_id = str(row.get('run_id') or '')
    status = row.get('status')
    if status != 'success':
        return index, row, False, True

    await _recover_prediction_from_trajectory(sem=sem, row=row, index=index, total=total)
    prediction, reference, question = _extract_eval_fields(row)
    if prediction is None or reference is None:
        return index, row, False, True

    if run_id and run_id in use_cache:
        task_output = row.get('task_output') or {}
        cached_payload = use_cache[run_id]
        task_output['score'] = cached_payload['score']
        if isinstance(cached_payload.get('evaluation_details'), dict):
            task_output['evaluation_details'] = cached_payload['evaluation_details']
        row['task_output'] = task_output
        return index, row, False, False

    # For WIDESEARCH, create a per-row evaluator so that instance_id and the
    # per-instance evaluation config (from metadata) are forwarded correctly,
    # mirroring what task.py does at runtime.
    if evaluator.mode == EvalMode.WIDESEARCH:
        meta = row.get('metadata') or {}
        instance_id = str(meta.get('instance_id') or run_id or 'unknown')
        row_evaluation = meta.get('evaluation')
        if not isinstance(row_evaluation, dict) and widesearch_evaluations:
            row_evaluation = widesearch_evaluations.get(instance_id) or widesearch_evaluations.get(run_id)
        row_evaluator: SearchEvaluator = SearchEvaluator(
            evaluator.mode,
            judge_prompt_template=evaluator.judge_prompt_template,
            judge_system_prompt=evaluator.judge_system_prompt,
            widesearch_evaluation=row_evaluation if isinstance(row_evaluation, dict) else None,
            instance_id=instance_id,
        )
    else:
        row_evaluator = evaluator

    async with sem:
        score, details = await row_evaluator.evaluate_with_details(
            prediction=prediction,
            reference=reference,
            question=question,
        )
    task_output = row.get('task_output') or {}
    task_output['score'] = float(score)
    if details is not None:
        task_output['evaluation_details'] = details.to_dict()
    row['task_output'] = task_output
    print(f'[{index + 1}/{total}] run_id={run_id or "N/A"} score={float(score):.1f}', flush=True)
    return index, row, True, False


async def main_async(args: argparse.Namespace) -> int:
    traj_input = args.traj or args.traj_positional
    if traj_input is None:
        print('Error: missing trajectory file. Provide positional TRAJ or --traj.')
        return 2
    traj_path = traj_input.resolve()
    eval_mode = EvalMode(args.eval_mode)
    if not traj_path.exists():
        print(f'Error: traj file not found: {traj_path}')
        return 1

    out_traj = args.output_traj.resolve() if args.output_traj else _default_output_paths(traj_path, eval_mode)[0]
    out_metrics = (
        args.output_metrics.resolve() if args.output_metrics else _default_output_paths(traj_path, eval_mode)[1]
    )

    # Load env vars, but do not override existing env.
    if args.env_file:
        env_file = args.env_file.resolve()
        if not env_file.exists():
            print(f'Error: env file not found: {env_file}')
            return 1
        for k, v in _parse_env_file(env_file).items():
            os.environ.setdefault(k, v)

    # Register extensions before getting client class from registry.
    Registry.import_module('nexus.extensions')

    openai_clients: dict[str, Any] = {}
    if eval_mode in (EvalMode.LLM_JUDGE, EvalMode.WIDESEARCH):
        # LLM_JUDGE uses the judge client for semantic evaluation.
        # WIDESEARCH optionally uses it for LLM-based column/row alignment.
        judge_spec = _build_client_config('JUDGE_OPENAI')
        judge_cls = openai_client_registry.get(judge_spec['name'])
        judge_client = judge_cls(
            OpenAIConfig(
                client_args=judge_spec['config'].get('client_args', {}),
                request_args=judge_spec['config'].get('request_args', {}),
            )
        )
        openai_clients['judge'] = judge_client

    rows = _load_jsonl(traj_path)
    if not rows:
        print('Error: no rows found in traj file')
        return 1

    excluded_instance_ids = _load_excluded_instance_ids(args)
    if excluded_instance_ids:
        before_count = len(rows)
        rows = filter_excluded_results(rows, excluded_instance_ids)
        print(
            f'Excluded instance IDs: {len(excluded_instance_ids)} (rows: {before_count} -> {len(rows)})',
            flush=True,
        )
        if not rows:
            print('Error: all rows were excluded')
            return 1

    resume_cache: dict[str, dict[str, Any]] = {}
    if args.resume and out_traj.exists():
        resume_cache = _build_resume_cache(out_traj)
        print(f'Resume cache loaded: {len(resume_cache)} rows')

    sem = asyncio.Semaphore(max(1, args.workers))

    dataset_path = args.dataset_path.resolve() if args.dataset_path else _default_dataset_path(traj_path)
    judge_prompt_template: str | None = None
    judge_system_prompt: str | None = None
    widesearch_evaluations: dict[str, dict[str, Any]] = {}

    if dataset_path is not None:
        dataset_evals, judge_prompt_template, judge_system_prompt = _load_dataset_configs(dataset_path)
        if judge_prompt_template or judge_system_prompt:
            print(
                f'Loaded judge config from dataset: '
                f'prompt_template={bool(judge_prompt_template)}, system_prompt={bool(judge_system_prompt)}'
            )
        if eval_mode == EvalMode.WIDESEARCH and dataset_evals:
            fallback_count = 0
            for instance_id, evaluation in dataset_evals.items():
                if instance_id not in widesearch_evaluations:
                    widesearch_evaluations[instance_id] = evaluation
                    fallback_count += 1
            print(
                f'Loaded wide_search evaluation configs from dataset: {len(dataset_evals)} rows '
                f'(fallback added: {fallback_count})'
            )
    elif eval_mode == EvalMode.WIDESEARCH:
        print('Warning: no dataset path resolved for wide_search evaluation configs', flush=True)

    if eval_mode == EvalMode.WIDESEARCH:
        db_path = args.db_path.resolve() if args.db_path else _default_db_path(traj_path)
        db_evaluations = _load_widesearch_evaluations_from_db(db_path)
        if db_evaluations:
            widesearch_evaluations.update(db_evaluations)
            print(f'Loaded wide_search evaluation configs from DB: {len(db_evaluations)} rows')
        else:
            print(f'Warning: no wide_search evaluation configs loaded from DB: {db_path}', flush=True)

    evaluator = SearchEvaluator(
        eval_mode,
        judge_prompt_template=judge_prompt_template,
        judge_system_prompt=judge_system_prompt,
    )
    total = len(rows)
    reevaluated = 0
    skipped = 0

    async with ExecutionContext(openai_clients=openai_clients, task_name='search_reeval'):
        tasks = [
            _eval_one(
                sem=sem,
                row=row,
                index=i,
                total=total,
                use_cache=resume_cache,
                evaluator=evaluator,
                widesearch_evaluations=widesearch_evaluations,
            )
            for i, row in enumerate(rows)
        ]
        for coro in asyncio.as_completed(tasks):
            i, updated_row, changed, is_skipped = await coro
            rows[i] = updated_row
            if changed:
                reevaluated += 1
            if is_skipped:
                skipped += 1

    out_traj.parent.mkdir(parents=True, exist_ok=True)
    with open(out_traj, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

    metrics = compute_metrics(
        rows,
        excluded_instance_ids=excluded_instance_ids,
    )
    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    out_metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')

    print('-' * 60)
    print(f'eval_mode={eval_mode.value}')
    print(f'total={total}')
    print(f'reevaluated={reevaluated}')
    print(f'skipped={skipped}')
    print(f'accuracy={metrics.get("accuracy")}')
    if eval_mode == EvalMode.WIDESEARCH:
        print(f'f1_by_item_accuracy={metrics.get("f1_by_item_accuracy")}')
    print(f'correct={metrics.get("correct")}, evaluated={metrics.get("evaluated")}, total={metrics.get("total")}')
    print(f'out_traj={out_traj}')
    print(f'out_metrics={out_metrics}')
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Re-evaluate search trajectory JSONL by judge model.')
    parser.add_argument(
        'traj_positional',
        nargs='?',
        type=Path,
        help='Input trajectory JSONL file path (positional, equivalent to --traj)',
    )
    parser.add_argument('--traj', type=Path, default=None, help='Input trajectory JSONL file path')
    parser.add_argument(
        '--env-file', type=Path, default=Path('.env.search'), help='Env file path (default: .env.search)'
    )
    parser.add_argument('--output-traj', type=Path, default=None, help='Output trajectory JSONL path')
    parser.add_argument('--output-metrics', type=Path, default=None, help='Output metrics JSON path')
    parser.add_argument(
        '--db-path',
        type=Path,
        default=None,
        help='Optional batch DB path for loading per-run wide_search evaluation configs',
    )
    parser.add_argument(
        '--dataset-path',
        type=Path,
        default=None,
        help='Optional parquet dataset path for loading judge config and wide_search evaluation configs',
    )
    parser.add_argument('--workers', type=int, default=500, help='Concurrent evaluation requests (default: 2)')
    parser.add_argument(
        '--eval-mode',
        type=str,
        choices=tuple(mode.value for mode in EvalMode),
        default=EvalMode.LLM_JUDGE.value,
        help='Evaluation mode (default: llm_judge)',
    )
    parser.add_argument('--resume', action='store_true', help='Reuse existing scores from output traj file')
    parser.add_argument(
        '--exclude-instance-id',
        action='append',
        default=[],
        help='Instance ID to exclude from evaluation (can be repeated)',
    )
    parser.add_argument(
        '--exclude-instance-ids-file',
        type=Path,
        default=None,
        help='Text file containing instance IDs to exclude (one per line)',
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return asyncio.run(main_async(args))


if __name__ == '__main__':
    raise SystemExit(main())
