#!/usr/bin/env python3
"""Compute metrics from batch run results.

Usage:
    python scripts/compute_metrics.py [results_path] [--db db_path]

Output:
    Prints metrics summary to stdout and saves to outputs/metrics.json
"""

from __future__ import annotations

import json
import sqlite3
import statistics

from collections import Counter, defaultdict
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from nexus.extensions.tasks.search.utils import extract_instance_id, filter_excluded_results


VISIT_ERROR_MARKERS = ('[visit error]', '[visit llm error]')
VISIT_CONTEXT_LENGTH_EXCEEDED_TAG = '[visit context length exceeded]'


def load_results(results_path: Path) -> list[dict[str, Any]]:
    """Load results from JSONL file."""
    results = []
    with open(results_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def load_results_from_db(
    db_path: Path,
    *,
    include_agent_output: bool = False,
) -> list[dict[str, Any]]:
    """Load results from SQLite database.

    Args:
        db_path: Path to the SQLite database.
        include_agent_output: If True, also load agent_output (needed for
            trajectory statistics). Defaults to False for backward compatibility.
    """
    results = []
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    extra_cols = ', agent_output' if include_agent_output else ''
    cursor.execute(f"""
        SELECT run_id, status, start_time, end_time, error,
               config, metadata, task_output{extra_cols}
        FROM runs
    """)

    json_fields = ['config', 'metadata', 'task_output']
    if include_agent_output:
        json_fields.append('agent_output')

    for row in cursor.fetchall():
        result = {
            'run_id': row['run_id'],
            'status': row['status'],
            'start_time': row['start_time'],
            'end_time': row['end_time'],
            'error': row['error'],
        }

        for field in json_fields:
            value = row[field]
            result[field] = json.loads(value) if value else {}

        results.append(result)

    conn.close()
    return results


def _is_timeout(r: dict[str, Any]) -> bool:
    """Check if a result represents a timeout."""
    status = r.get('status', '')
    if status == 'timeout':
        return True
    if status == 'failed':
        error = r.get('error', '') or ''
        return 'timeout' in error.lower() or 'timed out' in error.lower()
    return False


def _has_wide_search_evaluation(r: dict[str, Any]) -> bool:
    """Check whether a result carries WideSearch structured evaluation config.

    The ``evaluation`` dict in task_config comes from the dataset itself and is
    the authoritative signal that this sample belongs to a WideSearch benchmark,
    regardless of what ``eval_mode`` was set to via environment variables.
    """
    from nexus.extensions.tasks.search.task import WIDESEARCH_EVALUATION_REQUIRED_KEYS

    evaluation = r.get('config', {}).get('task_config', {}).get('evaluation')
    return isinstance(evaluation, dict) and WIDESEARCH_EVALUATION_REQUIRED_KEYS.issubset(evaluation)


def _get_judge_status(r: dict[str, Any]) -> str | None:
    """Extract judge status from task_output when present."""
    judge_status = (r.get('task_output') or {}).get('judge_status')
    return judge_status if isinstance(judge_status, str) and judge_status else None


def _reached_max_iterations(r: dict[str, Any]) -> bool:
    """Check whether the agent terminated due to max-iteration guard."""
    exit_status = (r.get('agent_output') or {}).get('exit_status')
    if not isinstance(exit_status, str):
        return False
    normalized = exit_status.upper()
    return normalized in {'MAX_STEPS', 'MAX_ITERATIONS'}


def _aggregate_judge_status(per_run: list[dict[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for metrics in per_run:
        counter.update(metrics.get('judge_status', {}))
    return {key: counter.get(key, 0) for key in ('success', 'failed', 'missing')}


def _aggregate_max_iter_reached(per_run: list[dict[str, Any]], total: int) -> dict[str, Any]:
    count = 0
    instance_ids: list[str] = []
    seen_instance_ids: set[str] = set()
    for metrics in per_run:
        max_iter_metrics = metrics.get('max_iter_reached', {})
        count += int(max_iter_metrics.get('count', 0) or 0)
        for instance_id in max_iter_metrics.get('instance_ids', []) or []:
            if instance_id and instance_id not in seen_instance_ids:
                seen_instance_ids.add(instance_id)
                instance_ids.append(instance_id)
    return {
        'count': count,
        'rate': round(count / total, 4) if total > 0 else 0.0,
        'instance_ids': instance_ids,
    }


def _aggregate_tool_call_metrics(per_run: list[dict[str, Any]]) -> dict[str, Any]:
    total_tool_calls = sum(int(metrics.get('total_tool_calls', 0) or 0) for metrics in per_run)
    records_with_trajectory = sum(int((metrics.get('tool_calls') or {}).get('count', 0) or 0) for metrics in per_run)
    if records_with_trajectory <= 0:
        return {}
    return {
        'tool_calls': {
            'count': records_with_trajectory,
            'mean': round(total_tool_calls / records_with_trajectory, 2),
        },
        'total_tool_calls': total_tool_calls,
    }


def _aggregate_tool_error_metrics(per_run: list[dict[str, Any]]) -> dict[str, Any]:
    records_with_trajectory = sum(int((metrics.get('tool_calls') or {}).get('count', 0) or 0) for metrics in per_run)
    visit_error_count = sum(int((metrics.get('visit_errors') or {}).get('count', 0) or 0) for metrics in per_run)
    visit_context_length_exceeded_count = sum(
        int(metrics.get('visit_context_length_exceeded', 0) or 0) for metrics in per_run
    )
    if records_with_trajectory <= 0:
        return {}
    return {
        'visit_errors': {
            'count': visit_error_count,
            'rate': round(visit_error_count / records_with_trajectory, 4),
        },
        'visit_context_length_exceeded': visit_context_length_exceeded_count,
    }


def compute_metrics(
    results: list[dict[str, Any]],
    *,
    excluded_instance_ids: set[str] | None = None,
    eval_mode_override: str | None = None,
) -> dict[str, Any]:
    """Compute metrics from results.

    Args:
        results: List of result dicts from a benchmark run.
        excluded_instance_ids: Optional set of instance IDs to exclude.
        eval_mode_override: When provided, this value is injected into the
            detected eval_modes set before any mode-specific metric logic runs
            (e.g. WIDESEARCH f1_by_item aggregation).  Useful when re-evaluating
            trajectories whose original config.task_config.eval_mode differs from
            the mode used during re-evaluation.
    """

    results = filter_excluded_results(results, excluded_instance_ids)

    if not results:
        return {'error': 'No results found'}

    total = len(results)
    success_count = sum(1 for r in results if r.get('status') == 'success')
    timeout_count = sum(1 for r in results if _is_timeout(r))
    # failed_count excludes timeouts to distinguish actionable failures
    failed_count = total - success_count - timeout_count

    # Collect scores and durations
    scores = []
    durations = []
    by_source: dict[str, list[float]] = defaultdict(list)
    correct_instance_ids: list[str] = []
    incorrect_instance_ids: list[str] = []
    failed_instance_ids: list[str] = []
    timeout_instance_ids: list[str] = []
    max_iter_instance_ids: list[str] = []
    eval_details_list: list[dict[str, Any]] = []
    eval_modes: set[str] = set()
    judge_status_counter: Counter[str] = Counter()
    if any(_has_wide_search_evaluation(r) for r in results):
        eval_modes.add('wide_search')

    for r in results:
        instance_id = extract_instance_id(r)
        if _reached_max_iterations(r) and instance_id:
            max_iter_instance_ids.append(instance_id)

        # Duration
        start, end = r.get('start_time'), r.get('end_time')
        if start and end:
            try:
                start_dt = datetime.fromisoformat(start.replace('Z', '+00:00'))
                end_dt = datetime.fromisoformat(end.replace('Z', '+00:00'))
                durations.append((end_dt - start_dt).total_seconds())
            except Exception:
                pass

        # Score
        if _is_timeout(r):
            if instance_id:
                timeout_instance_ids.append(instance_id)
            continue

        if r.get('status') == 'failed':
            if instance_id:
                failed_instance_ids.append(instance_id)
            continue

        if r.get('status') != 'success':
            continue
        task_output = r.get('task_output', {})
        judge_status = _get_judge_status(r)
        judge_status_counter[judge_status or 'missing'] += 1
        score = task_output.get('score')
        if score is None:
            continue

        score_f = float(score)
        scores.append(score_f)
        if score_f >= 1.0 and instance_id:
            correct_instance_ids.append(instance_id)
        elif instance_id:
            incorrect_instance_ids.append(instance_id)

        # Collect evaluation_details for downstream aggregation
        ed = task_output.get('evaluation_details')
        if ed and isinstance(ed, dict):
            eval_details_list.append(ed)

        # Group by source
        config = r.get('config', {})
        task_config = config.get('task_config', {})
        eval_mode = task_config.get('eval_mode')
        if isinstance(eval_mode, str):
            eval_modes.add(eval_mode)
        source = task_config.get('data_source') or r.get('metadata', {}).get('data_source') or 'unknown'
        by_source[source].append(float(score))

    if eval_mode_override is not None:
        eval_modes.add(eval_mode_override)

    # Calculate accuracy over all tasks (including failed/timeout/non-evaluated tasks).
    correct = sum(1 for s in scores if s >= 1.0)
    accuracy = round(correct / total, 4) if total > 0 else 0.0

    # Build metrics
    metrics: dict[str, Any] = {
        'total': total,
        'success': success_count,
        'failed': failed_count,
        'timeout': timeout_count,
        'accuracy': accuracy,
        'correct': correct,
        'evaluated': len(scores),
        # Always include instance-level ID lists for downstream reporting consistency.
        'correct_instance_ids': [],
        'incorrect_instance_ids': [],
        'failed_instance_ids': [],
        'timeout_instance_ids': [],
    }

    # Timing (only avg)
    if durations:
        metrics['avg_time_sec'] = round(sum(durations) / len(durations), 1)

    # By source (only if multiple sources or source != unknown)
    if len(by_source) > 1 or (len(by_source) == 1 and 'unknown' not in by_source):
        metrics['by_source'] = {
            source: {
                'total': len(src_scores),
                'correct': sum(1 for s in src_scores if s >= 1.0),
                'accuracy': round(sum(1 for s in src_scores if s >= 1.0) / len(src_scores), 4),
            }
            for source, src_scores in sorted(by_source.items())
        }

    # Failed run IDs (non-timeout)
    error_ids = [r['run_id'] for r in results if r.get('status') == 'failed' and not _is_timeout(r)]
    if error_ids:
        metrics['failed_runs'] = error_ids[:10]
    metrics['failed_instance_ids'] = failed_instance_ids

    # Timeout run IDs with partial info
    timeout_runs = []
    for r in results:
        if _is_timeout(r):
            entry: dict[str, Any] = {'run_id': r['run_id']}
            instance_id = extract_instance_id(r)
            if instance_id:
                entry['instance_id'] = instance_id
            if r.get('error'):
                entry['error'] = r['error'][:200]
            task_output = r.get('task_output') or {}
            if task_output:
                entry['partial_result'] = task_output
            timeout_runs.append(entry)
    if timeout_runs:
        metrics['timeout_runs'] = timeout_runs
    metrics['timeout_instance_ids'] = timeout_instance_ids
    metrics['correct_instance_ids'] = correct_instance_ids
    metrics['incorrect_instance_ids'] = incorrect_instance_ids
    metrics['judge_status'] = {
        'success': judge_status_counter.get('success', 0),
        'failed': judge_status_counter.get('failed', 0),
        'missing': judge_status_counter.get('missing', 0),
    }
    metrics['max_iter_reached'] = {
        'count': len(max_iter_instance_ids),
        'rate': round(len(max_iter_instance_ids) / total, 4) if total > 0 else 0.0,
        'instance_ids': max_iter_instance_ids,
    }

    trajectory_stats = compute_trajectory_stats(results)
    total_steps = 0
    if trajectory_stats.get('records_with_trajectory', 0) > 0:
        metrics['steps'] = trajectory_stats['steps']
        metrics['messages'] = trajectory_stats['messages']
        metrics['tool_calls'] = trajectory_stats['tool_calls']
        metrics['total_tool_calls'] = trajectory_stats['total_tool_calls']
        metrics['avg_tool_calls_per_step'] = trajectory_stats['avg_tool_calls_per_step']
        metrics['visit_errors'] = trajectory_stats['visit_errors']
        metrics['visit_context_length_exceeded'] = trajectory_stats['visit_context_length_exceeded']
        total_steps = trajectory_stats.get('total_steps', 0)

    token_stats = _compute_token_stats(results, total_steps=total_steps)
    if token_stats:
        metrics['token_usage'] = token_stats

    # Aggregate evaluation_details numeric fields
    if eval_details_list:
        numeric_keys = [k for k, v in eval_details_list[0].items() if isinstance(v, (int, float))]
        n_evaluated = len(scores)
        n_with_details = len(eval_details_list)
        metrics['avg_evaluation_details'] = {
            k: round(sum(d.get(k, 0.0) for d in eval_details_list) / n_evaluated, 4) for k in numeric_keys
        }
        metrics['avg_evaluation_details']['n_evaluated'] = n_evaluated
        metrics['avg_evaluation_details']['n_with_details'] = n_with_details

        # For wide_search eval mode, benchmark accuracy should be the averaged f1_by_item.
        if 'wide_search' in eval_modes and 'f1_by_item' in metrics['avg_evaluation_details']:
            f1_by_item_accuracy = metrics['avg_evaluation_details']['f1_by_item']
            metrics['f1_by_item_accuracy'] = f1_by_item_accuracy
            metrics['accuracy'] = f1_by_item_accuracy

    return metrics


def compute_avg_k_metrics(
    results_per_run: list[list[dict[str, Any]]],
) -> dict[str, Any]:
    """Compute avg@k and pass@k metrics from k independent runs of the same benchmark.

    avg@k: for each instance, compute mean score across k runs (missing/failed runs
        contribute 0), then average those per-instance means across all instances.
    strict_avg@k: fraction of instances where per-instance mean score >= 1.0.
    pass@k: fraction of instances where at least one run scored >= 1.0.
    mean_accuracy: simple mean of per-run accuracies (unweighted average of runs).

    Args:
        results_per_run: List of result lists, one per run.

    Returns:
        Metrics dict with avg@k, pass@k, per-run breakdown, and aggregated stats.
    """
    k = len(results_per_run)
    if k == 0:
        return {'error': 'No runs provided'}

    if k == 1:
        return compute_metrics(results_per_run[0])

    # Collect per-instance scores across all runs.
    # Each instance keeps a fixed-length score vector of size k.
    # Missing/failed/no-score entries remain 0.0.
    instance_scores: dict[str, list[float]] = {}

    for run_idx, run_results in enumerate(results_per_run):
        for r in run_results:
            run_id = r.get('run_id')
            if not run_id:
                continue

            if run_id not in instance_scores:
                instance_scores[run_id] = [0.0] * k

            score = 0.0
            if r.get('status') == 'success':
                task_output = r.get('task_output', {})
                raw_score = task_output.get('score')
                if raw_score is not None:
                    try:
                        score = float(raw_score)
                    except (TypeError, ValueError):
                        score = 0.0

            # If duplicate rows appear for the same instance in one run,
            # keep the best observed score for stable pass@k behavior.
            if score > instance_scores[run_id][run_idx]:
                instance_scores[run_id][run_idx] = score

    n_instances = len(instance_scores)

    # avg@k (new definition): for each instance, first average over k runs;
    # then average those per-instance means across all instances.
    instance_mean_scores = [(sum(scores) / k) for scores in instance_scores.values()]
    avg_k_accuracy = round(sum(instance_mean_scores) / n_instances, 4) if n_instances > 0 else 0.0
    # Weighted "correct count" under avg@k definition; can be non-integer.
    avg_k_correct = round(sum(instance_mean_scores), 4)

    # Keep strict aggregate accuracy for backward compatibility:
    # instance is correct only when mean score >= 1.0.
    strict_correct = sum(1 for s in instance_mean_scores if s >= 1.0)
    strict_accuracy = round(strict_correct / n_instances, 4) if n_instances > 0 else 0.0

    # pass@k: at least one run correct
    pass_k_correct = sum(1 for scores in instance_scores.values() if max(scores) >= 1.0)
    pass_k_accuracy = round(pass_k_correct / n_instances, 4) if n_instances > 0 else 0.0

    # Per-instance rollout breakdown: each instance's score vector across the k
    # groups (index j == rollout group run_{j+1}), with its mean / pass / hit
    # count. This is the RL-facing signal that the aggregate metrics collapse
    # away. ``scores`` uses 0.0 for missing/failed groups, matching avg@k.
    per_instance = [
        {
            'instance_id': rid,
            'scores': scores,
            'mean': round(sum(scores) / k, 4),
            'pass': max(scores) >= 1.0,
            'n_correct': sum(1 for s in scores if s >= 1.0),
        }
        for rid, scores in sorted(instance_scores.items())
    ]

    # Per-run individual metrics
    per_run = [compute_metrics(run_results) for run_results in results_per_run]
    run_accuracies = [m.get('accuracy', 0.0) for m in per_run]
    mean_accuracy = round(sum(run_accuracies) / len(run_accuracies), 4) if run_accuracies else 0.0

    # Aggregate totals from individual runs (attempt-level count across k runs)
    total_tasks = sum(m.get('total', 0) for m in per_run)
    total_success = sum(m.get('success', 0) for m in per_run)
    total_failed = sum(m.get('failed', 0) for m in per_run)
    total_timeout = sum(m.get('timeout', 0) for m in per_run)
    avg_time = None
    times = [m['avg_time_sec'] for m in per_run if 'avg_time_sec' in m]
    if times:
        avg_time = round(sum(times) / len(times), 1)

    result: dict[str, Any] = {
        'k': k,
        'instances': n_instances,
        # Align default fields with k-run display semantics.
        'accuracy': avg_k_accuracy,
        'correct': avg_k_correct,
        'evaluated': n_instances,
        # Named metrics
        'avg_k_accuracy': avg_k_accuracy,
        'avg_k_correct': avg_k_correct,
        'strict_avg_k_accuracy': strict_accuracy,
        'strict_avg_k_correct': strict_correct,
        'pass_k_accuracy': pass_k_accuracy,
        'pass_k_correct': pass_k_correct,
        'mean_accuracy': mean_accuracy,
        # Aggregated task counts
        'total': total_tasks,
        'success': total_success,
        'failed': total_failed,
        'timeout': total_timeout,
        # Per-run breakdown
        'per_run': per_run,
        # Per-instance rollout breakdown (score vector across the k groups).
        'per_instance': per_instance,
        # Instance IDs aggregated across runs; keep deterministic order.
        'correct_instance_ids': sorted({
            iid for run_m in per_run for iid in (run_m.get('correct_instance_ids') or []) if iid
        }),
        'incorrect_instance_ids': sorted({
            iid for run_m in per_run for iid in (run_m.get('incorrect_instance_ids') or []) if iid
        }),
        'failed_instance_ids': sorted({
            iid for run_m in per_run for iid in (run_m.get('failed_instance_ids') or []) if iid
        }),
        'timeout_instance_ids': sorted({
            iid for run_m in per_run for iid in (run_m.get('timeout_instance_ids') or []) if iid
        }),
    }

    if avg_time is not None:
        result['avg_time_sec'] = avg_time
    result['judge_status'] = _aggregate_judge_status(per_run)
    result['max_iter_reached'] = _aggregate_max_iter_reached(per_run, total_tasks)
    result.update(_aggregate_tool_call_metrics(per_run))
    result.update(_aggregate_tool_error_metrics(per_run))
    result.update(_aggregate_step_metrics(per_run))
    token_agg = _aggregate_token_metrics(per_run)
    if token_agg:
        result['token_usage'] = token_agg

    return result


def _load_results_per_run_with_failures(run_results: list[Any]) -> list[list[dict[str, Any]]]:
    """Load DB rows per attempted run; failed/missing runs contribute empty results.

    This keeps k aligned with attempted runs so failed runs are counted as 0
    in avg@k/pass@k calculations rather than shrinking k.
    """
    results_per_run: list[list[dict[str, Any]]] = []
    for run_result in run_results:
        if getattr(run_result, 'status', None) != 'success':
            results_per_run.append([])
            continue
        db_path = getattr(run_result, 'db_path', None)
        if db_path and Path(db_path).exists():
            results_per_run.append(load_results_from_db(Path(db_path), include_agent_output=True))
        else:
            results_per_run.append([])
    return results_per_run


def compute_interim_multi_run_metrics(run_results: list[Any]) -> dict[str, Any]:
    """Compute interim multi-run metrics from completed run results."""
    results_per_run_so_far = _load_results_per_run_with_failures(run_results)

    if len(results_per_run_so_far) >= 2:
        return compute_avg_k_metrics(results_per_run_so_far)

    successful_so_far = [r for r in run_results if getattr(r, 'status', None) == 'success']
    if successful_so_far:
        return getattr(successful_so_far[0], 'metrics', None) or {}

    return {}


def compute_final_multi_run_metrics(run_results: list[Any]) -> dict[str, Any]:
    """Compute final multi-run metrics from all successful run results."""
    results_per_run = _load_results_per_run_with_failures(run_results)

    if len(results_per_run) < 2:
        successful = [r for r in run_results if getattr(r, 'status', None) == 'success']
        if not successful:
            return {}
        return getattr(successful[0], 'metrics', None) or {}

    return compute_avg_k_metrics(results_per_run)


# ============================================================================
# Token Usage Statistics
# ============================================================================


def _extract_agent_stats(record: dict[str, Any]) -> dict[str, int] | None:
    """Extract token usage and trajectory stats from agent_output if available.

    total_prompt_tokens stores the prompt_tokens of the LAST LLM call (trajectory peak),
    while total_completion_tokens is cumulative across all calls, so their sum equals
    the true total_tokens of the full trajectory.

    Returns dict with prompt/completion/total tokens plus step/tool_call counts,
    or None when no token data is present.
    """
    agent_output = record.get('agent_output') or {}
    prompt = agent_output.get('total_prompt_tokens')
    completion = agent_output.get('total_completion_tokens')
    if prompt is None and completion is None:
        return None
    p, c = int(prompt or 0), int(completion or 0)
    result: dict[str, int] = {'prompt_tokens': p, 'completion_tokens': c, 'total_tokens': p + c}
    for key in ('num_steps', 'num_tool_calls', 'context_reset_count'):
        val = agent_output.get(key)
        if val is not None:
            result[key] = int(val)
    return result


def _compute_token_stats(
    results: list[dict[str, Any]],
    total_steps: int = 0,
) -> dict[str, Any]:
    """Compute aggregate token usage statistics from results with agent_output.

    Args:
        results: Result records (must include agent_output for token data).
        total_steps: Total assistant steps across all records (for per-step averages).
            Falls back to sum of per-instance num_steps from agent_output when 0.

    Returns empty dict if no token data is available.
    """
    prompt_list: list[int] = []
    completion_list: list[int] = []
    total_list: list[int] = []
    context_resets = 0
    agent_total_steps = 0

    for rec in results:
        stats = _extract_agent_stats(rec)
        if stats is None:
            continue
        prompt_list.append(stats['prompt_tokens'])
        completion_list.append(stats['completion_tokens'])
        total_list.append(stats['total_tokens'])
        context_resets += stats.get('context_reset_count', 0)
        agent_total_steps += stats.get('num_steps', 0)

    if not total_list:
        return {}

    sum_prompt = sum(prompt_list)
    sum_completion = sum(completion_list)
    sum_total = sum_prompt + sum_completion
    effective_steps = total_steps or agent_total_steps

    result: dict[str, Any] = {
        'instances_with_usage': len(total_list),
        'total_prompt_tokens': sum_prompt,
        'total_completion_tokens': sum_completion,
        'total_tokens': sum_total,
        'prompt_tokens_per_instance': _descriptive_stats(prompt_list),
        'completion_tokens_per_instance': _descriptive_stats(completion_list),
        'total_tokens_per_instance': _descriptive_stats(total_list),
    }
    if effective_steps > 0:
        result['avg_tokens_per_step'] = round(sum_total / effective_steps, 1)
    if context_resets > 0:
        result['total_context_resets'] = context_resets

    return result


def _aggregate_token_metrics(per_run: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate token_usage metrics from per-run metrics for multi-run (avg@k)."""
    total_prompt = 0
    total_completion = 0
    instances_with_usage = 0
    total_context_resets = 0
    total_steps = 0.0

    for metrics in per_run:
        tu = metrics.get('token_usage') or {}
        total_prompt += int(tu.get('total_prompt_tokens', 0) or 0)
        total_completion += int(tu.get('total_completion_tokens', 0) or 0)
        instances_with_usage += int(tu.get('instances_with_usage', 0) or 0)
        total_context_resets += int(tu.get('total_context_resets', 0) or 0)
        step_stats = metrics.get('steps') or {}
        step_count = float(step_stats.get('count', 0) or 0)
        step_mean = float(step_stats.get('mean', 0) or 0)
        total_steps += step_count * step_mean

    if instances_with_usage <= 0:
        return {}

    total_tokens = total_prompt + total_completion
    result: dict[str, Any] = {
        'instances_with_usage': instances_with_usage,
        'total_prompt_tokens': total_prompt,
        'total_completion_tokens': total_completion,
        'total_tokens': total_tokens,
        'total_tokens_per_instance': {
            'count': instances_with_usage,
            'mean': round(total_tokens / instances_with_usage, 1),
        },
    }
    if total_steps > 0:
        result['avg_tokens_per_step'] = round(total_tokens / total_steps, 1)
    if total_context_resets > 0:
        result['total_context_resets'] = total_context_resets
    return result


def _aggregate_step_metrics(per_run: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate step/message metrics from per-run summaries.

    Mean/median are weighted by per-run record counts; min/max use global extrema.
    """
    result: dict[str, Any] = {}
    for field_name in ('steps', 'messages'):
        summaries: list[dict[str, Any]] = []
        for run_summary in per_run:
            field_summary = run_summary.get(field_name)
            if isinstance(field_summary, dict):
                summaries.append(field_summary)
        if not summaries:
            continue

        weights = [float(summary.get('count', 1) or 1) for summary in summaries]
        total_weight = sum(weights)
        agg: dict[str, Any] = {'runs': len(summaries), 'count': int(total_weight)}

        for key in ('mean', 'median'):
            weighted_values = [
                (float(summary[key]), weight)
                for summary, weight in zip(summaries, weights, strict=False)
                if key in summary
            ]
            if weighted_values:
                weighted_sum = sum(value * weight for value, weight in weighted_values)
                weight_sum = sum(weight for _, weight in weighted_values)
                agg[key] = round(weighted_sum / weight_sum, 2)

        min_values = [float(summary['min']) for summary in summaries if 'min' in summary]
        max_values = [float(summary['max']) for summary in summaries if 'max' in summary]
        if min_values:
            agg['min'] = round(min(min_values), 2)
        if max_values:
            agg['max'] = round(max(max_values), 2)

        result[field_name] = agg
    return result


# ============================================================================
# Trajectory Statistics
# ============================================================================


def count_agent_steps(messages: list[dict[str, Any]]) -> int:
    """Count agent steps (number of assistant turns) in a trajectory."""
    return sum(1 for m in messages if m.get('role') == 'assistant')


def _extract_tool_calls(messages: list[dict[str, Any]]) -> list[str]:
    """Extract tool names from all assistant tool_calls in a trajectory."""
    names: list[str] = []
    for m in messages:
        if m.get('role') == 'assistant' and m.get('tool_calls'):
            for tc in m['tool_calls']:
                names.append((tc.get('function') or {}).get('name', 'unknown'))
    return names


def _get_trajectory(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract message trajectory from a record, trying agent_output first."""
    messages = (record.get('agent_output') or {}).get('main_trajectory', [])
    if not messages:
        messages = (record.get('task_output') or {}).get('trajectory', [])
    return messages or []


def _extract_message_text(content: Any) -> str:
    if content is None:
        return ''
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return '\n'.join(part for item in content if (part := _extract_message_text(item)))
    if isinstance(content, dict):
        for key in ('text', 'value'):
            value = content.get(key)
            if isinstance(value, str):
                return value
        return '\n'.join(part for value in content.values() if (part := _extract_message_text(value)))
    return str(content)


def _collect_tool_error_counts(messages: list[dict[str, Any]]) -> tuple[int, int]:
    has_visit_error = False
    visit_context_length_exceeded_count = 0
    for message in messages:
        if message.get('role') != 'tool':
            continue
        tool_name = message.get('name')
        text = _extract_message_text(message.get('content'))
        if tool_name == 'visit':
            if any(marker in text for marker in VISIT_ERROR_MARKERS):
                has_visit_error = True
            visit_context_length_exceeded_count += text.count(VISIT_CONTEXT_LENGTH_EXCEEDED_TAG)
    return int(has_visit_error), visit_context_length_exceeded_count


def _descriptive_stats(values: Sequence[int | float]) -> dict[str, Any]:
    if not values:
        return {}
    n = len(values)
    result: dict[str, Any] = {
        'count': n,
        'mean': round(statistics.mean(values), 2),
        'median': round(statistics.median(values), 2),
        'min': min(values),
        'max': max(values),
        'stdev': round(statistics.stdev(values), 2) if n > 1 else 0.0,
    }
    if n >= 4:
        q = statistics.quantiles(values, n=4)
        result['p25'] = round(q[0], 2)
        result['p75'] = round(q[2], 2)
    if n >= 10:
        result['p90'] = round(sorted(values)[int(n * 0.9)], 2)
    return result


def _tool_distribution(counter: Counter, total_calls: int) -> list[dict[str, Any]]:
    return [
        {'tool': name, 'calls': cnt, 'pct': round(cnt / total_calls * 100, 2) if total_calls else 0}
        for name, cnt in counter.most_common()
    ]


def compute_trajectory_stats(
    results: list[dict[str, Any]],
    *,
    score_filter: float | None = None,
) -> dict[str, Any]:
    """Compute trajectory-level statistics from results that include agent_output.

    Args:
        results: List of result dicts (must include agent_output or task_output.trajectory).
        score_filter: If set, only include records with score >= this value.

    Returns:
        Dict with messages/steps/tool_calls stats, role and tool distributions.
    """
    msg_counts: list[int] = []
    step_counts: list[int] = []
    tool_call_counts: list[int] = []
    tool_counter: Counter = Counter()
    role_counter: Counter = Counter()
    visit_error_count = 0
    visit_context_length_exceeded_count = 0

    for rec in results:
        if score_filter is not None:
            score = (rec.get('task_output') or {}).get('score')
            if score is None or float(score) < score_filter:
                continue

        messages = _get_trajectory(rec)
        if not messages:
            continue

        msg_counts.append(len(messages))
        step_counts.append(count_agent_steps(messages))
        tools = _extract_tool_calls(messages)
        tool_call_counts.append(len(tools))
        tool_counter.update(tools)
        visit_err, context_err = _collect_tool_error_counts(messages)
        visit_error_count += visit_err
        visit_context_length_exceeded_count += context_err
        for m in messages:
            role_counter[m.get('role', 'unknown')] += 1

    total_tools = sum(tool_counter.values())
    total_steps = sum(step_counts)
    records_with_trajectory = len(msg_counts)

    return {
        'records_with_trajectory': records_with_trajectory,
        'messages': _descriptive_stats(msg_counts),
        'steps': _descriptive_stats(step_counts),
        'total_steps': total_steps,
        'tool_calls': _descriptive_stats(tool_call_counts),
        'avg_tool_calls_per_step': round(total_tools / total_steps, 2) if total_steps else 0,
        'total_tool_calls': total_tools,
        'visit_errors': {
            'count': visit_error_count,
            'rate': round(visit_error_count / records_with_trajectory, 4) if records_with_trajectory else 0.0,
        },
        'visit_context_length_exceeded': visit_context_length_exceeded_count,
        'role_distribution': dict(role_counter.most_common()),
        'tool_distribution': _tool_distribution(tool_counter, total_tools),
    }
