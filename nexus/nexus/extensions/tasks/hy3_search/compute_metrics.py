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

from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from nexus.extensions.tasks.hy3_search.utils import extract_instance_id, filter_excluded_results


def load_results(results_path: Path) -> list[dict[str, Any]]:
    """Load results from JSONL file."""
    results = []
    with open(results_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def load_results_from_db(db_path: Path) -> list[dict[str, Any]]:
    """Load results from SQLite database."""
    results = []
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        SELECT run_id, status, start_time, end_time, error,
               config, metadata, task_output
        FROM runs
    """)

    for row in cursor.fetchall():
        result = {
            'run_id': row['run_id'],
            'status': row['status'],
            'start_time': row['start_time'],
            'end_time': row['end_time'],
            'error': row['error'],
        }

        for field in ['config', 'metadata', 'task_output']:
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


def compute_metrics(
    results: list[dict[str, Any]],
    *,
    excluded_instance_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Compute metrics from results."""

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
    eval_details_list: list[dict[str, Any]] = []
    eval_modes: set[str] = set()

    for r in results:
        instance_id = extract_instance_id(r)

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
            results_per_run.append(load_results_from_db(Path(db_path)))
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
