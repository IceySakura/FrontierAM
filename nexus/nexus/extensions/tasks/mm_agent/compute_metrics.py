#!/usr/bin/env python3
"""Compute metrics from batch run results for the ``mm_agent`` task.

This module mirrors ``nexus.extensions.tasks.mm_agent.compute_metrics`` from
the standalone nexus repo, adapted for the hytuner in-tree mm_agent task:

* Scores at or below ``-100`` (the ``_UNJUDGED_SCORE`` marker used by
  ``task.py`` for environment-side failures) are **excluded** from accuracy
  and score-based aggregates so they don't poison the metric.
* In addition to success / failed / timeout counts, we also break the
  successful runs down by ``exit_status`` (``finished``, ``max_iterations``,
  ``tool_error``, ``tool_execution_error``, ``inference_error``,
  ``context_length_exceeded``, ``inference_timeout``, ``observation_timeout``,
  …) so it's easy to see why the model stopped.

Usage:
    # Single input
    python compute_metrics.py [results_path] [--db db_path] [-o metrics.json]
    # Multiple inputs (e.g. N repeated runs of the same dataset): all
    # positional paths are aggregated into a single "overall" metric, and
    # each file's per-run metric is also reported under ``per_run``.
    python compute_metrics.py run1.db run2.db run3.db [-o overall.json]
"""

from __future__ import annotations

import json
import math
import sqlite3

from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


try:
    from nexus.extensions.tasks.mm_agent.prompts import FORCE_FINISH_PROMPT
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from prompts import FORCE_FINISH_PROMPT


# Samples with score <= this threshold are considered "unjudged" / masked and
# must be excluded from any score-based aggregation. Matches
# ``_UNJUDGED_SCORE = -100.0`` in ``task.py`` and hytuner's
# ``ReasoningReward.THRESHOLD = -99`` masking convention.
_UNJUDGED_SCORE_THRESHOLD = -100.0

# Synthetic exit-status bucket assigned in post-processing when the trajectory
# contains the exact ``FORCE_FINISH_PROMPT`` injected by the main_agent
# post-iteration hook on the penultimate iteration, i.e. the model didn't
# terminate on its own and was *forced* to emit a final answer. We surface
# this as its own bucket so it doesn't blend into the regular ``finished``
# count.
_ENFORCE_FINISH_STATUS = 'enforce_finish'
# Only these original statuses are eligible to be re-tagged as
# ``enforce_finish``. Mid-run failures (e.g. ``inference_error``,
# ``tool_error``) keep their original status because they don't reflect
# a forced finish — the run died before the final user-prompt could matter.
_ENFORCE_FINISH_ELIGIBLE = {'finished', 'max_iterations'}


def load_results(results_path: Path) -> list[dict[str, Any]]:
    """Load results from a JSONL file."""
    results: list[dict[str, Any]] = []
    with open(results_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def load_results_from_db(db_path: Path) -> list[dict[str, Any]]:
    """Load results from the ``runs`` table of a SQLite database."""
    results: list[dict[str, Any]] = []
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
        for field in ('config', 'metadata', 'task_output'):
            value = row[field]
            if not value:
                result[field] = {}
                continue
            # DB columns are TEXT JSON in the normal case, but be defensive:
            # if a row was corrupted / written as raw bytes, decode with
            # replacement instead of exploding the whole run.
            if isinstance(value, (bytes, bytearray)):
                try:
                    value = value.decode('utf-8')
                except UnicodeDecodeError:
                    value = value.decode('utf-8', errors='replace')
            try:
                result[field] = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                result[field] = {}
        results.append(result)

    conn.close()
    return results


def _is_timeout(r: dict[str, Any]) -> bool:
    """Whether a result represents a timeout."""
    status = r.get('status', '')
    if status == 'timeout':
        return True
    if status == 'failed':
        error = r.get('error', '') or ''
        return 'timeout' in error.lower() or 'timed out' in error.lower()
    return False


def _extract_instance_id(r: dict[str, Any]) -> str | None:
    """Extract an instance id from a result row.

    Prefers ``task_output.instance_id`` (written by ``SearchTaskOutput``),
    then falls back to ``metadata.instance_id`` and finally ``run_id``.
    """
    task_output = r.get('task_output') or {}
    instance_id = task_output.get('instance_id')
    if instance_id:
        return str(instance_id)
    metadata = r.get('metadata') or {}
    instance_id = metadata.get('instance_id')
    if instance_id:
        return str(instance_id)
    run_id = r.get('run_id')
    return str(run_id) if run_id else None


def _extract_exit_status(r: dict[str, Any]) -> str | None:
    """Extract ``exit_status`` from a successful run's task output."""
    task_output = r.get('task_output') or {}
    exit_status = task_output.get('exit_status')
    return str(exit_status) if exit_status else None


def _has_force_finish_prompt(r: dict[str, Any]) -> bool:
    """Return whether ``task_output.trajectory`` contains FORCE_FINISH_PROMPT.

    ``main_agent`` may prepend prior ``task_input.history`` user turns to the
    trajectory, so counting user messages is not sufficient. The force-finish
    hook appends this exact synthetic user prompt; key off that content.
    """
    task_output = r.get('task_output') or {}
    trajectory = task_output.get('trajectory') or []
    if not isinstance(trajectory, list):
        return False
    for msg in trajectory:
        if isinstance(msg, dict) and msg.get('role') == 'user' and msg.get('content') == FORCE_FINISH_PROMPT:
            return True
    return False


def _derive_exit_status(
    raw_exit_status: str | None,
    has_force_finish_prompt: bool,
) -> str | None:
    """Optionally rewrite an exit_status to ``enforce_finish``.

    When the force-finish prompt is present and the original status is one of
    the "reached the end" statuses (``finished`` / ``max_iterations``), we
    relabel the bucket to ``enforce_finish``. Other statuses are passed through
    unchanged.
    """
    if not raw_exit_status:
        return raw_exit_status
    if has_force_finish_prompt and raw_exit_status in _ENFORCE_FINISH_ELIGIBLE:
        return _ENFORCE_FINISH_STATUS
    return raw_exit_status


def _extract_tool_call_names(r: dict[str, Any]) -> list[str]:
    """Extract the ordered list of tool-call function names from a result.

    Walks ``task_output.trajectory`` and collects every
    ``assistant.tool_calls[*].function.name``. Returns an empty list when the
    trajectory is missing or carries no tool calls.
    """
    task_output = r.get('task_output') or {}
    trajectory = task_output.get('trajectory') or []
    names: list[str] = []
    if not isinstance(trajectory, list):
        return names
    for msg in trajectory:
        if not isinstance(msg, dict):
            continue
        if msg.get('role') != 'assistant':
            continue
        tool_calls = msg.get('tool_calls') or []
        if not isinstance(tool_calls, list):
            continue
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            fn = tc.get('function') or {}
            name = fn.get('name') if isinstance(fn, dict) else None
            if isinstance(name, str) and name:
                names.append(name)
    return names


def _summarize_tool_usage(
    per_sample_counts: list[dict[str, int]],
    label: str | None = None,
) -> dict[str, Any] | None:
    """Aggregate per-sample tool-call counts into a usage summary.

    ``per_sample_counts`` is a list with one ``{tool_name: count}`` dict per
    sample (samples that made zero tool calls contribute an empty dict but
    are still counted as samples).

    Returns a dict with:

    * ``n_samples``: how many samples were aggregated.
    * ``samples_with_tool_calls`` / ``samples_without_tool_calls``.
    * ``total_tool_calls`` and ``avg_tool_calls_per_sample``.
    * ``tool_call_counts``: total calls per tool across all samples.
    * ``tool_call_share``: ``tool_call_counts`` normalized by total calls
      (i.e. global "what fraction of all calls went to this tool").
    * ``tool_usage_rate``: fraction of samples that called this tool at
      least once (coverage).
    * ``avg_per_sample_share``: average of per-sample shares — for each
      sample we compute ``tool_count / sample_total`` and then average
      across samples that made at least one call. This is the answer to
      "in a single trace, what fraction of calls is this tool on average?".
    """
    n_samples = len(per_sample_counts)
    if n_samples == 0:
        return None

    total_calls = 0
    tool_total: dict[str, int] = defaultdict(int)
    tool_sample_hits: dict[str, int] = defaultdict(int)
    tool_share_accum: dict[str, float] = defaultdict(float)
    samples_with_calls = 0

    for counts in per_sample_counts:
        sample_total = sum(counts.values())
        if sample_total <= 0:
            continue
        samples_with_calls += 1
        total_calls += sample_total
        for name, c in counts.items():
            tool_total[name] += c
            tool_sample_hits[name] += 1
            tool_share_accum[name] += c / sample_total

    if total_calls == 0 and samples_with_calls == 0:
        # No tool activity at all — still return the shape so callers can
        # see the bucket exists but the model never invoked any tool.
        return {
            'label': label,
            'n_samples': n_samples,
            'samples_with_tool_calls': 0,
            'samples_without_tool_calls': n_samples,
            'total_tool_calls': 0,
            'avg_tool_calls_per_sample': 0.0,
            'tool_call_counts': {},
            'tool_call_share': {},
            'tool_usage_rate': {},
            'avg_per_sample_share': {},
        }

    tool_call_counts = dict(sorted(tool_total.items(), key=lambda kv: (-kv[1], kv[0])))
    tool_call_share = (
        {name: round(c / total_calls, 4) for name, c in tool_call_counts.items()} if total_calls > 0 else {}
    )
    tool_usage_rate = {name: round(tool_sample_hits[name] / n_samples, 4) for name in tool_call_counts}
    # Per-sample share is averaged only over samples that issued at least
    # one tool call so empty traces don't drag every share toward zero.
    denom = samples_with_calls if samples_with_calls > 0 else 1
    avg_per_sample_share = {name: round(tool_share_accum[name] / denom, 4) for name in tool_call_counts}

    out: dict[str, Any] = {
        'n_samples': n_samples,
        'samples_with_tool_calls': samples_with_calls,
        'samples_without_tool_calls': n_samples - samples_with_calls,
        'total_tool_calls': total_calls,
        'avg_tool_calls_per_sample': round(total_calls / n_samples, 4),
        'tool_call_counts': tool_call_counts,
        'tool_call_share': tool_call_share,
        'tool_usage_rate': tool_usage_rate,
        'avg_per_sample_share': avg_per_sample_share,
    }
    if label is not None:
        out['label'] = label
    return out


def compute_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute metrics from a list of result rows.

    Scores ``<= _UNJUDGED_SCORE_THRESHOLD`` (-100) are excluded from all
    score/accuracy aggregates, but they are still counted under
    ``exit_status_counts`` and ``masked_*`` so callers can tell how many
    samples were dropped and why.
    """
    if not results:
        return {'error': 'No results found'}

    total = len(results)
    success_count = sum(1 for r in results if r.get('status') == 'success')
    timeout_count = sum(1 for r in results if _is_timeout(r))
    # ``failed`` here excludes timeouts so actionable failures stand out.
    failed_count = total - success_count - timeout_count

    scores: list[float] = []
    durations: list[float] = []
    by_source: dict[str, list[float]] = defaultdict(list)

    # Bookkeeping for instance-level reporting.
    failed_instance_ids: list[str] = []
    timeout_instance_ids: list[str] = []
    masked_instance_ids: list[str] = []

    # Exit-status distribution. Counted across *all* rows that carry a status
    # in ``task_output``, regardless of whether their score was masked.
    exit_status_counts: dict[str, int] = defaultdict(int)
    # Exit-status distribution restricted to masked (score <= -100) samples,
    # useful for double-checking env-side failure rates.
    masked_exit_status_counts: dict[str, int] = defaultdict(int)
    # Per-exit-status accuracy bucket (only non-masked, evaluated scores).
    scores_by_exit_status: dict[str, list[float]] = defaultdict(list)

    eval_details_list: list[dict[str, Any]] = []
    eval_modes: set[str] = set()
    masked_count = 0

    # Per-sample tool-call counts, plus a few breakdowns. Each entry maps
    # ``tool_name -> count`` for that sample. We keep the samples themselves
    # (rather than a global counter) so we can compute "average share within
    # a single trace" — which is what the user asked for.
    tool_counts_all: list[dict[str, int]] = []
    tool_counts_evaluated: list[dict[str, int]] = []
    tool_counts_correct: list[dict[str, int]] = []
    tool_counts_incorrect: list[dict[str, int]] = []
    tool_counts_by_exit: dict[str, list[dict[str, int]]] = defaultdict(list)

    for r in results:
        instance_id = _extract_instance_id(r)

        # Tool-call snapshot is collected for *every* row (success / failed /
        # timeout) so the global ``tool_usage`` reflects what the model
        # actually emitted, not just the runs that passed task-side gates.
        tc_names = _extract_tool_call_names(r)
        tc_counts: dict[str, int] = defaultdict(int)
        for n in tc_names:
            tc_counts[n] += 1
        tc_counts = dict(tc_counts)
        tool_counts_all.append(tc_counts)

        # Duration (measured for *all* rows regardless of final status).
        start, end = r.get('start_time'), r.get('end_time')
        if start and end:
            try:
                start_dt = datetime.fromisoformat(start.replace('Z', '+00:00'))
                end_dt = datetime.fromisoformat(end.replace('Z', '+00:00'))
                durations.append((end_dt - start_dt).total_seconds())
            except Exception:
                pass

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

        task_output = r.get('task_output') or {}
        raw_exit_status = _extract_exit_status(r)
        # Re-tag organic terminations as ``enforce_finish`` when the
        # trajectory shows the actual FORCE_FINISH_PROMPT user message. All
        # downstream buckets (counts / accuracy / tool usage) use the derived
        # value so the new state is first-class.
        exit_status = _derive_exit_status(raw_exit_status, _has_force_finish_prompt(r))
        if exit_status:
            exit_status_counts[exit_status] += 1

        raw_score = task_output.get('score')
        if raw_score is None:
            continue

        try:
            score_f = float(raw_score)
        except (TypeError, ValueError):
            continue

        # Drop masked samples from the score aggregates (but keep their
        # exit-status count visible so debugging stays easy).
        if score_f <= _UNJUDGED_SCORE_THRESHOLD:
            masked_count += 1
            if exit_status:
                masked_exit_status_counts[exit_status] += 1
            if instance_id:
                masked_instance_ids.append(instance_id)
            continue

        scores.append(score_f)
        if exit_status:
            scores_by_exit_status[exit_status].append(score_f)

        # Tool-usage breakdowns (only for evaluated, non-masked samples).
        tool_counts_evaluated.append(tc_counts)
        if exit_status:
            tool_counts_by_exit[exit_status].append(tc_counts)
        if score_f >= 1.0:
            tool_counts_correct.append(tc_counts)
        else:
            tool_counts_incorrect.append(tc_counts)

        ed = task_output.get('evaluation_details')
        if ed and isinstance(ed, dict):
            eval_details_list.append(ed)

        # Group by ``data_source`` for per-benchmark breakdown.
        config = r.get('config', {}) or {}
        task_config = config.get('task_config', {}) or {}
        eval_mode = task_config.get('eval_mode')
        if isinstance(eval_mode, str):
            eval_modes.add(eval_mode)
        source = task_config.get('data_source') or (r.get('metadata') or {}).get('data_source') or 'unknown'
        by_source[source].append(score_f)

    evaluated = len(scores)
    correct = sum(1 for s in scores if s >= 1.0)
    # Denominator = evaluated (non-masked) samples so -100 samples don't
    # artificially tank accuracy. Callers who prefer ``correct / total``
    # can recompute it from ``correct`` and ``total``.
    accuracy = round(correct / evaluated, 4) if evaluated > 0 else 0.0
    accuracy_over_total = round(correct / total, 4) if total > 0 else 0.0

    metrics: dict[str, Any] = {
        'total': total,
        'success': success_count,
        'failed': failed_count,
        'timeout': timeout_count,
        'masked': masked_count,
        'evaluated': evaluated,
        'correct': correct,
        'accuracy': accuracy,
        'accuracy_over_total': accuracy_over_total,
        'avg_score': round(sum(scores) / evaluated, 4) if evaluated > 0 else 0.0,
        'failed_instance_ids': failed_instance_ids,
        'timeout_instance_ids': timeout_instance_ids,
        'masked_instance_ids': masked_instance_ids,
    }

    # Exit-status distribution (sorted by count desc for readability).
    if exit_status_counts:
        metrics['exit_status_counts'] = dict(sorted(exit_status_counts.items(), key=lambda kv: (-kv[1], kv[0])))
    if masked_exit_status_counts:
        metrics['masked_exit_status_counts'] = dict(
            sorted(masked_exit_status_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        )

    # Per-exit-status accuracy (evaluated samples only).
    if scores_by_exit_status:
        metrics['by_exit_status'] = {
            status: {
                'evaluated': len(st_scores),
                'correct': sum(1 for s in st_scores if s >= 1.0),
                'accuracy': round(sum(1 for s in st_scores if s >= 1.0) / len(st_scores), 4),
                'avg_score': round(sum(st_scores) / len(st_scores), 4),
            }
            for status, st_scores in sorted(scores_by_exit_status.items())
        }

    if durations:
        metrics['avg_time_sec'] = round(sum(durations) / len(durations), 1)

    # ---- Tool-usage statistics --------------------------------------------
    # Global: tool calls aggregated across every row, regardless of status.
    tool_usage_all = _summarize_tool_usage(tool_counts_all, label='all')
    if tool_usage_all is not None:
        metrics['tool_usage'] = tool_usage_all

    tool_usage_eval = _summarize_tool_usage(tool_counts_evaluated, label='evaluated')
    if tool_usage_eval is not None:
        metrics['tool_usage_evaluated'] = tool_usage_eval

    # Correct vs. incorrect comparison (handy for spotting whether models that
    # answered correctly use a meaningfully different tool mix).
    tool_usage_split: dict[str, Any] = {}
    if tool_counts_correct:
        tu_c = _summarize_tool_usage(tool_counts_correct, label='correct')
        if tu_c is not None:
            tool_usage_split['correct'] = tu_c
    if tool_counts_incorrect:
        tu_i = _summarize_tool_usage(tool_counts_incorrect, label='incorrect')
        if tu_i is not None:
            tool_usage_split['incorrect'] = tu_i
    if tool_usage_split:
        metrics['tool_usage_by_correctness'] = tool_usage_split

    if tool_counts_by_exit:
        metrics['tool_usage_by_exit_status'] = {
            status: _summarize_tool_usage(samples, label=status)
            for status, samples in sorted(tool_counts_by_exit.items())
        }

    # Per-data-source breakdown (only meaningful when there is more than one
    # source, or when it is explicitly set).
    if len(by_source) > 1 or (len(by_source) == 1 and 'unknown' not in by_source):
        metrics['by_source'] = {
            source: {
                'total': len(src_scores),
                'correct': sum(1 for s in src_scores if s >= 1.0),
                'accuracy': round(sum(1 for s in src_scores if s >= 1.0) / len(src_scores), 4),
                'avg_score': round(sum(src_scores) / len(src_scores), 4),
            }
            for source, src_scores in sorted(by_source.items())
        }

    # First 10 failed run ids for quick triage.
    error_ids = [r['run_id'] for r in results if r.get('status') == 'failed' and not _is_timeout(r)]
    if error_ids:
        metrics['failed_runs'] = error_ids[:10]

    # Timeout run details (keep payload bounded).
    timeout_runs: list[dict[str, Any]] = []
    for r in results:
        if _is_timeout(r):
            entry: dict[str, Any] = {'run_id': r['run_id']}
            instance_id = _extract_instance_id(r)
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

    # Aggregate evaluation_details numeric fields across evaluated samples.
    if eval_details_list:
        numeric_keys = [k for k, v in eval_details_list[0].items() if isinstance(v, (int, float))]
        n_evaluated = evaluated
        n_with_details = len(eval_details_list)
        if n_evaluated > 0 and numeric_keys:
            metrics['avg_evaluation_details'] = {
                k: round(sum(d.get(k, 0.0) for d in eval_details_list) / n_evaluated, 4) for k in numeric_keys
            }
            metrics['avg_evaluation_details']['n_evaluated'] = n_evaluated
            metrics['avg_evaluation_details']['n_with_details'] = n_with_details

            if 'wide_search' in eval_modes and 'f1_by_item' in metrics['avg_evaluation_details']:
                f1_by_item_accuracy = metrics['avg_evaluation_details']['f1_by_item']
                metrics['f1_by_item_accuracy'] = f1_by_item_accuracy
                metrics['accuracy'] = f1_by_item_accuracy

    return metrics


def _load_one(path: Path) -> list[dict[str, Any]]:
    """Load results from a single ``.db`` / ``.jsonl`` file."""
    use_db = path.suffix.lower() in {'.db', '.sqlite', '.sqlite3'}
    if use_db:
        return load_results_from_db(path)
    return load_results(path)


def _print_summary(metrics: dict[str, Any], header: str | None = None) -> None:
    """Pretty-print the high-level numbers of a metrics dict."""
    if header:
        print(header)
    print(f'Total tasks:  {metrics.get("total", 0)}')
    print(f'Successful:   {metrics.get("success", 0)}')
    print(f'Failed:       {metrics.get("failed", 0)}')
    print(f'Timeout:      {metrics.get("timeout", 0)}')
    print(f'Masked(<=-100): {metrics.get("masked", 0)}')
    print(f'Evaluated:    {metrics.get("evaluated", 0)}')
    print(f'Correct:      {metrics.get("correct", 0)}')
    print(f'Accuracy (evaluated): {metrics.get("accuracy", 0):.2%}')
    print(f'Accuracy (over total):{metrics.get("accuracy_over_total", 0):.2%}')

    if 'avg_time_sec' in metrics:
        print(f'Average time: {metrics["avg_time_sec"]}s')

    if 'exit_status_counts' in metrics:
        print('Exit status counts:')
        for status, count in metrics['exit_status_counts'].items():
            print(f'  {status:<28s} {count}')

    tu = metrics.get('tool_usage')
    if tu and tu.get('total_tool_calls', 0) > 0:
        print(
            f'Tool usage (all rows): {tu["total_tool_calls"]} calls across '
            f'{tu["samples_with_tool_calls"]}/{tu["n_samples"]} samples '
            f'(avg {tu["avg_tool_calls_per_sample"]}/sample)'
        )
        print('  Per-sample avg share / global share / coverage:')
        for name, share in tu.get('avg_per_sample_share', {}).items():
            gshare = tu.get('tool_call_share', {}).get(name, 0.0)
            cov = tu.get('tool_usage_rate', {}).get(name, 0.0)
            count = tu.get('tool_call_counts', {}).get(name, 0)
            print(f'    {name:<28s} per-sample={share:.2%}  global={gshare:.2%}  coverage={cov:.2%}  (n_calls={count})')


def _mean_std(values: list[float]) -> tuple[float, float]:
    """Return ``(mean, std)`` for ``values`` (population std, 0.0 for n<=1)."""
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mean = sum(values) / n
    if n == 1:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / n
    return mean, math.sqrt(var)


def main() -> int:
    """Command-line entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Compute mm_agent metrics from batch run results')
    # Accept any number of positional inputs (.db / .jsonl). When more than
    # one is provided, we additionally compute an "overall" metric across
    # all of them (treating every run as part of one combined sample pool)
    # plus per-run accuracy mean/std.
    parser.add_argument(
        'results_path',
        nargs='*',
        help='Path(s) to results.jsonl or .db file(s). Pass multiple paths '
        'to aggregate N repeated runs into one overall metric.',
    )
    parser.add_argument(
        '--db',
        action='append',
        default=None,
        help='Path to SQLite database file. Can be specified multiple times '
        'to aggregate multiple files (combined with positional args).',
    )
    parser.add_argument(
        '--output',
        '-o',
        default=None,
        help='Output metrics file path. Defaults to "<input_stem>.json" '
        '(single input) or "<first_input_stem>.overall.json" '
        '(multiple inputs) in the same directory as the input.',
    )

    args = parser.parse_args()

    # Merge positional and --db inputs, preserving order.
    input_paths: list[Path] = []
    for p in args.results_path or []:
        input_paths.append(Path(p))
    for p in args.db or []:
        input_paths.append(Path(p))

    if not input_paths:
        parser.error('At least one results_path or --db must be provided')

    try:
        # Single-input path keeps the original behavior verbatim.
        if len(input_paths) == 1:
            input_path = input_paths[0]
            output_path = Path(args.output) if args.output else input_path.with_suffix('.json')
            results = _load_one(input_path)
            metrics = compute_metrics(results)
            _print_summary(metrics)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            print(f'Metrics saved to: {output_path}')
            return 0

        # Multi-input path: per-run + overall (combined pool).
        per_run: list[dict[str, Any]] = []
        combined: list[dict[str, Any]] = []
        for p in input_paths:
            print(f'\n=== Loading {p} ===')
            res = _load_one(p)
            m = compute_metrics(res)
            per_run.append({'path': str(p), 'metrics': m})
            combined.extend(res)
            _print_summary(m, header=f'-- per-run [{p.name}] --')

        print('\n=== Computing overall (combined) metrics ===')
        overall = compute_metrics(combined)
        _print_summary(overall, header='-- overall --')

        # Per-run accuracy mean/std (and avg_score / accuracy_over_total).
        acc_list = [float(item['metrics'].get('accuracy', 0.0)) for item in per_run]
        acc_total_list = [float(item['metrics'].get('accuracy_over_total', 0.0)) for item in per_run]
        avg_score_list = [float(item['metrics'].get('avg_score', 0.0)) for item in per_run]
        acc_mean, acc_std = _mean_std(acc_list)
        acc_total_mean, acc_total_std = _mean_std(acc_total_list)
        score_mean, score_std = _mean_std(avg_score_list)
        per_run_summary = {
            'n_runs': len(per_run),
            'accuracy_mean': round(acc_mean, 4),
            'accuracy_std': round(acc_std, 4),
            'accuracy_min': round(min(acc_list), 4) if acc_list else 0.0,
            'accuracy_max': round(max(acc_list), 4) if acc_list else 0.0,
            'accuracy_per_run': [round(v, 4) for v in acc_list],
            'accuracy_over_total_mean': round(acc_total_mean, 4),
            'accuracy_over_total_std': round(acc_total_std, 4),
            'avg_score_mean': round(score_mean, 4),
            'avg_score_std': round(score_std, 4),
        }

        print('\nPer-run accuracy:')
        for item, v in zip(per_run, acc_list, strict=False):
            print(f'  {Path(item["path"]).name:<48s} acc={v:.2%}')
        print(
            f'\nAcc mean ± std: {acc_mean:.2%} ± {acc_std:.2%} '
            f'(min={min(acc_list):.2%}, max={max(acc_list):.2%}, n={len(acc_list)})'
        )

        out_payload = {
            'inputs': [str(p) for p in input_paths],
            'per_run_summary': per_run_summary,
            'overall': overall,
            'per_run': per_run,
        }

        first = input_paths[0]
        # Default: drop run-suffix (e.g. ``__run1``) so multiple repeated
        # runs share a single ``<dataset>.overall.json``.
        default_stem = first.stem
        if '__run' in default_stem:
            default_stem = default_stem.split('__run', 1)[0]
        default_out = first.with_name(f'{default_stem}.overall.json')
        output_path = Path(args.output) if args.output else default_out
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(out_payload, f, indent=2, ensure_ascii=False)
        print(f'\nOverall metrics saved to: {output_path}')
        return 0

    except Exception as e:
        print(f'Error: {e}')
        return 1


if __name__ == '__main__':
    main()
