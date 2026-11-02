#!/usr/bin/env python3
"""Analyze claw-eval batch run results.

Reads a results.jsonl file (written by task.py after each task) and generates:
- batch_summary.json  — matches original claw-eval format
- Per-category score breakdown (terminal report)
- Per-task detailed results
- Overall statistics (pass rate, mean scores per dimension)
- Safety violations report

Usage:
    # Analyze results and print report
    python -m nexus.extensions.tasks.claw_eval.analyze_results results.jsonl

    # Also write batch_summary.json to the same directory
    python -m nexus.extensions.tasks.claw_eval.analyze_results results.jsonl --summary

    # Write batch_summary.json to a specific path
    python -m nexus.extensions.tasks.claw_eval.analyze_results results.jsonl --summary-path /path/to/summary.json
"""

from __future__ import annotations

import argparse
import json
import sys

from collections import defaultdict
from pathlib import Path
from typing import Any


# ============================================================================
# Data Loading
# ============================================================================


def load_results(path: str | Path) -> list[dict[str, Any]]:
    """Load results from a JSONL file."""
    results = []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f'WARNING: Skipping malformed line {lineno}: {exc}', file=sys.stderr)
    return results


# ============================================================================
# Analysis
# ============================================================================


def analyze(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Analyze results and produce summary statistics.

    Returns a dict compatible with the original claw-eval batch_summary.json
    format, plus extended per-category breakdowns.
    """
    if not results:
        return {'error': 'No results to analyze'}

    # Group by category
    by_category: dict[str, list[dict]] = defaultdict(list)

    # Accumulators
    all_scores: list[float] = []
    all_completion: list[float] = []
    all_robustness: list[float] = []
    all_communication: list[float] = []
    all_safety: list[float] = []

    # Token/time accumulators
    total_model_input_tokens: int = 0
    total_model_output_tokens: int = 0
    total_tokens: int = 0
    total_model_time_s: float = 0.0
    total_wall_time_s: float = 0.0
    total_model_call_count: int = 0
    total_cost: float = 0.0

    passed_count = 0
    failed_count = 0
    error_count = 0
    safety_violations: list[str] = []

    for r in results:
        # Support both formats:
        #   - task.py _write_result_jsonl() uses 'output' as the key
        #   - nexus batch runner export uses 'task_output' as the key
        output = r.get('output') or r.get('task_output') or r
        metadata = r.get('metadata', {})

        task_id = output.get('task_id', r.get('run_id', 'unknown'))
        category = metadata.get('category', output.get('category', 'unknown'))
        difficulty = metadata.get('difficulty', output.get('difficulty', 'unknown'))

        score = output.get('score', 0.0)
        completion = output.get('completion', 0.0)
        robustness = output.get('robustness', 0.0)
        communication = output.get('communication', 0.0)
        safety = output.get('safety', 1.0)
        passed = output.get('passed', False)
        error = output.get('error_message')
        trajectory_length = output.get('trajectory_length', 0)
        breakdown = output.get('breakdown', {})

        # Token/time per task
        model_input_tokens = output.get('model_input_tokens', 0)
        model_output_tokens = output.get('model_output_tokens', 0)
        task_total_tokens = output.get('total_tokens', 0)
        model_time_s = output.get('model_time_s', 0.0)
        wall_time_s = output.get('wall_time_s', 0.0)
        model_call_count = output.get('model_call_count', 0)
        cost = output.get('cost')

        entry = {
            'task_id': task_id,
            'category': category,
            'difficulty': difficulty,
            'score': score,
            'completion': completion,
            'robustness': robustness,
            'communication': communication,
            'safety': safety,
            'passed': passed,
            'error': error,
            'trajectory_length': trajectory_length,
            'breakdown': breakdown,
            'model_input_tokens': model_input_tokens,
            'model_output_tokens': model_output_tokens,
            'total_tokens': task_total_tokens,
            'model_time_s': model_time_s,
            'wall_time_s': wall_time_s,
            'model_call_count': model_call_count,
            'cost': cost,
        }

        by_category[category].append(entry)

        # Accumulate scores (include errored tasks with score=0)
        all_scores.append(score)
        all_completion.append(completion)
        all_robustness.append(robustness)
        all_communication.append(communication)
        all_safety.append(safety)

        # Accumulate token/time totals
        total_model_input_tokens += model_input_tokens
        total_model_output_tokens += model_output_tokens
        total_tokens += task_total_tokens
        total_model_time_s += model_time_s
        total_wall_time_s += wall_time_s
        total_model_call_count += model_call_count
        if cost is not None:
            total_cost += cost

        if error:
            error_count += 1
        elif passed:
            passed_count += 1
        else:
            failed_count += 1

        if safety < 1.0:
            safety_violations.append(task_id)

    total = len(results)

    def _mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    # Non-errored scores for avg_score (matching original claw-eval logic)
    non_error_scores = [
        s
        for s, r in zip(all_scores, results, strict=False)
        if not (r.get('output') or r.get('task_output') or r).get('error_message')
    ]
    avg_score = _mean(non_error_scores) if non_error_scores else 0.0

    # Per-category summaries
    category_summaries: dict[str, dict[str, Any]] = {}
    for cat, entries in sorted(by_category.items()):
        cat_scores = [e['score'] for e in entries]
        cat_completion = [e['completion'] for e in entries]
        cat_robustness = [e['robustness'] for e in entries]
        cat_passed = sum(1 for e in entries if e['passed'])
        cat_errors = sum(1 for e in entries if e.get('error'))

        category_summaries[cat] = {
            'count': len(entries),
            'passed': cat_passed,
            'failed': len(entries) - cat_passed - cat_errors,
            'errors': cat_errors,
            'mean_score': _mean(cat_scores),
            'mean_completion': _mean(cat_completion),
            'mean_robustness': _mean(cat_robustness),
            'pass_rate': cat_passed / len(entries) if entries else 0.0,
            'tasks': sorted(entries, key=lambda e: e['task_id']),
        }

    # Per-difficulty summaries
    by_difficulty: dict[str, list[dict]] = defaultdict(list)
    for cat_data in by_category.values():
        for entry in cat_data:
            by_difficulty[entry['difficulty']].append(entry)

    difficulty_summaries: dict[str, dict[str, Any]] = {}
    for diff, entries in sorted(by_difficulty.items()):
        diff_scores = [e['score'] for e in entries]
        diff_passed = sum(1 for e in entries if e['passed'])
        difficulty_summaries[diff] = {
            'count': len(entries),
            'passed': diff_passed,
            'mean_score': _mean(diff_scores),
            'pass_rate': diff_passed / len(entries) if entries else 0.0,
        }

    return {
        # Summary block (matches original claw-eval batch_summary.json)
        'tasks': total,
        'trials_per_task': 1,
        'passed': passed_count,
        'failed': failed_count,
        'errored': error_count,
        'avg_score': avg_score,
        'pass_rate': passed_count / total if total else 0.0,
        # Extended dimension means
        'mean_completion': _mean(all_completion),
        'mean_robustness': _mean(all_robustness),
        'mean_communication': _mean(all_communication),
        # Safety
        'safety_violation_count': len(safety_violations),
        'safety_violations': safety_violations,
        # Token/time totals (matching original claw-eval format)
        'total_model_input_tokens': total_model_input_tokens,
        'total_model_output_tokens': total_model_output_tokens,
        'total_tokens': total_tokens,
        'total_model_time_s': round(total_model_time_s, 2),
        'total_wall_time_s': round(total_wall_time_s, 2),
        'total_model_call_count': total_model_call_count,
        'total_cost': round(total_cost, 6) if total_cost > 0 else None,
        # Breakdowns
        'by_category': category_summaries,
        'by_difficulty': difficulty_summaries,
    }


# ============================================================================
# Reporting
# ============================================================================


def print_report(analysis: dict[str, Any]) -> None:
    """Print a formatted terminal report."""
    if 'error' in analysis:
        print(f'ERROR: {analysis["error"]}')
        return

    print()
    print('=' * 72)
    print('  CLAW-EVAL BATCH RESULTS')
    print('=' * 72)

    # Overall summary
    total = analysis['tasks']
    passed = analysis['passed']
    failed = analysis['failed']
    errored = analysis['errored']

    print(f'\n  Total tasks:        {total}')
    print(f'  Passed (≥0.75):     {passed}  ({analysis["pass_rate"]:.1%})')
    print(f'  Failed (<0.75):     {failed}')
    print(f'  Errored:            {errored}')

    print(f'\n  Avg score:          {analysis["avg_score"]:.4f}')
    print(f'  Mean completion:    {analysis["mean_completion"]:.4f}')
    print(f'  Mean robustness:    {analysis["mean_robustness"]:.4f}')
    print(f'  Mean communication: {analysis["mean_communication"]:.4f}')

    violations = analysis.get('safety_violation_count', 0)
    print(f'  Safety violations:  {violations}')
    if analysis.get('safety_violations'):
        for v in analysis['safety_violations']:
            print(f'    ⚠ {v}')

    # Token/time totals
    total_tokens = analysis.get('total_tokens', 0)
    total_input = analysis.get('total_model_input_tokens', 0)
    total_output = analysis.get('total_model_output_tokens', 0)
    total_model_time = analysis.get('total_model_time_s', 0.0)
    total_wall_time = analysis.get('total_wall_time_s', 0.0)
    total_calls = analysis.get('total_model_call_count', 0)
    total_cost = analysis.get('total_cost')

    if total_tokens > 0 or total_calls > 0:
        print(f'\n  Total model tokens: {total_tokens} ({total_input} in / {total_output} out)')
        print(f'  Total time:         wall={total_wall_time:.2f}s model={total_model_time:.2f}s')
        print(f'  Model API calls:    {total_calls}')
        if total_cost is not None:
            print(f'  Estimated cost:     ${total_cost:.4f}')

    # By difficulty
    by_difficulty = analysis.get('by_difficulty', {})
    if by_difficulty:
        print('\n' + '-' * 72)
        print('  BY DIFFICULTY')
        print('-' * 72)
        for diff, data in by_difficulty.items():
            print(
                f'  {diff:<12} | '
                f'tasks={data["count"]:>3} | '
                f'passed={data["passed"]:>3} | '
                f'rate={data["pass_rate"]:.1%} | '
                f'avg={data["mean_score"]:.3f}'
            )

    # By category with per-task details
    by_category = analysis.get('by_category', {})
    if by_category:
        print('\n' + '-' * 72)
        print('  BY CATEGORY')
        print('-' * 72)

        for cat, data in by_category.items():
            print(
                f'\n  [{cat}] ({data["count"]} tasks) — '
                f'avg={data["mean_score"]:.3f}, '
                f'pass_rate={data["pass_rate"]:.1%} '
                f'({data["passed"]}/{data["count"]})'
            )

            for task in data.get('tasks', []):
                tid = task['task_id']
                if task.get('error'):
                    err_short = task['error'][:60].replace('\n', ' ')
                    print(f'    {tid:40s}  0.00  ERR   {err_short}')
                else:
                    status = 'PASS' if task['passed'] else 'FAIL'
                    task_tokens = task.get('total_tokens', 0)
                    in_tok = task.get('model_input_tokens', 0)
                    out_tok = task.get('model_output_tokens', 0)
                    wall_t = task.get('wall_time_s', 0.0)
                    model_t = task.get('model_time_s', 0.0)
                    print(
                        f'    {tid:40s}  {task["score"]:.2f}  {status}  '
                        f'C={task["completion"]:.2f} R={task["robustness"]:.2f} '
                        f'M={task.get("communication", 0.0):.2f} S={task.get("safety", 1.0):.0f}  '
                        f'TOK={task_tokens} ({in_tok}in/{out_tok}out)  '
                        f'TIME=wall {wall_t:.2f}s model {model_t:.2f}s'
                    )

    print('\n' + '=' * 72)


def generate_batch_summary(analysis: dict[str, Any]) -> dict[str, Any]:
    """Generate a batch_summary.json compatible with original claw-eval format.

    Extracts only the top-level fields that match the original format.
    """
    return {
        'tasks': analysis['tasks'],
        'trials_per_task': analysis.get('trials_per_task', 1),
        'passed': analysis['passed'],
        'failed': analysis['failed'],
        'errored': analysis['errored'],
        'avg_score': analysis['avg_score'],
        'pass_rate': analysis.get('pass_rate', 0.0),
        'mean_completion': analysis.get('mean_completion', 0.0),
        'mean_robustness': analysis.get('mean_robustness', 0.0),
        'mean_communication': analysis.get('mean_communication', 0.0),
        'safety_violation_count': analysis.get('safety_violation_count', 0),
        # Token/time totals (matching original claw-eval batch_summary.json format)
        'total_model_input_tokens': analysis.get('total_model_input_tokens', 0),
        'total_model_output_tokens': analysis.get('total_model_output_tokens', 0),
        'total_tokens': analysis.get('total_tokens', 0),
        'total_model_time_s': analysis.get('total_model_time_s', 0.0),
        'total_wall_time_s': analysis.get('total_wall_time_s', 0.0),
        'total_model_call_count': analysis.get('total_model_call_count', 0),
        'total_cost': analysis.get('total_cost'),
        'by_category': {
            cat: {
                'count': data['count'],
                'passed': data['passed'],
                'failed': data.get('failed', 0),
                'errors': data.get('errors', 0),
                'mean_score': data['mean_score'],
                'pass_rate': data['pass_rate'],
            }
            for cat, data in analysis.get('by_category', {}).items()
        },
        'by_difficulty': analysis.get('by_difficulty', {}),
    }


# ============================================================================
# CLI
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Analyze claw-eval batch run results from results.jsonl',
    )
    parser.add_argument(
        '--results_file',
        help='Path to results.jsonl file',
    )
    parser.add_argument(
        '--summary',
        action='store_true',
        help='Write batch_summary.json to the same directory as results.jsonl',
    )
    parser.add_argument(
        '--summary-path',
        type=str,
        default=None,
        help='Write batch_summary.json to a specific path',
    )
    parser.add_argument(
        '--json',
        type=str,
        default=None,
        dest='json_output',
        help='Write full analysis as JSON to specified path',
    )
    parser.add_argument(
        '--quiet',
        '-q',
        action='store_true',
        help='Suppress terminal report output',
    )

    args = parser.parse_args()

    results_path = Path(args.results_file)
    if not results_path.exists():
        print(f'ERROR: Results file not found: {results_path}', file=sys.stderr)
        sys.exit(1)

    results = load_results(results_path)
    if not results:
        print('ERROR: No results found in file', file=sys.stderr)
        sys.exit(1)

    analysis = analyze(results)

    # Terminal report
    if not args.quiet:
        print_report(analysis)

    # Write batch_summary.json
    summary_path = args.summary_path
    if summary_path is None and args.summary:
        summary_path = str(results_path.parent / 'batch_summary.json')

    if summary_path:
        summary = generate_batch_summary(analysis)
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f'\nbatch_summary.json written to {summary_path}')

    # Write full analysis JSON
    if args.json_output:
        # Remove per-task details for compact output
        compact = {k: v for k, v in analysis.items() if k != 'by_category'}
        compact['by_category'] = {
            cat: {k: v for k, v in data.items() if k != 'tasks'}
            for cat, data in analysis.get('by_category', {}).items()
        }
        with open(args.json_output, 'w') as f:
            json.dump(compact, f, indent=2, ensure_ascii=False)
        print(f'\nFull analysis JSON written to {args.json_output}')


if __name__ == '__main__':
    main()
