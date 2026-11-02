#!/usr/bin/env python3
"""Build judge benchmark dataset from existing evaluation results.

Extracts (question, prediction, reference, gt_score, gt_reasoning) for each
case-experiment pair from the trace DB and judge_reasoning.json.

Usage:
    python3 build_judge_benchmark.py --results-dir /path/to/nexus_results
    python3 build_judge_benchmark.py --results-dir /path/to/nexus_results --only-correct
"""

import argparse
import json
import sqlite3

from pathlib import Path

import pandas as pd


DEFAULT_RESULTS_DIR = Path('/apdcephfs_gy6/share_304144884/hunyuan/debug_agent/taiji_replay/nexus_results')


def load_judge_reasoning(results_dir: Path) -> dict:
    path = results_dir / 'judge_reasoning.json'
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def load_annotations(results_dir: Path) -> dict:
    path = results_dir / 'annotations.json'
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def extract_cases_from_experiment(exp_path: Path, exp_name: str) -> list[dict]:
    """Extract question, prediction, reference, score from an experiment."""
    results_file = exp_path / 'results.jsonl'
    db_path = exp_path / 'taiji_bench.db'

    if not results_file.exists():
        return []

    # Load scores from results.jsonl
    results = {}
    with open(results_file) as f:
        for line in f:
            r = json.loads(line)
            rid = str(r['run_id'])
            task_output = r.get('task_output', {}) or {}
            results[rid] = {
                'score': task_output.get('score', 0),
                'prediction': task_output.get('prediction', ''),
                'reference': task_output.get('reference', ''),
            }

    # Load questions from DB
    questions = {}
    if db_path.exists():
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        for row in conn.execute('SELECT run_id, config FROM runs'):
            config = json.loads(row['config']) if row['config'] else {}
            tc = config.get('task_config', {})
            questions[str(row['run_id'])] = tc.get('question', '')
        conn.close()

    cases = []
    for rid, data in results.items():
        cases.append({
            'run_id': rid,
            'experiment': exp_name,
            'question': questions.get(rid, ''),
            'prediction': data['prediction'],
            'reference': data['reference'],
            'gt_score': data['score'],
        })
    return cases


def main():
    parser = argparse.ArgumentParser(description='Build judge benchmark dataset')
    parser.add_argument(
        '--results-dir', default=str(DEFAULT_RESULTS_DIR), help='Directory containing experiment results'
    )
    parser.add_argument('--output', default='judge_benchmark.parquet', help='Output parquet path')
    parser.add_argument(
        '--only-correct', action='store_true', help='Only include cases where human annotated judge as correct'
    )
    parser.add_argument(
        '--exclude-experiments', default='hy3_results0709', help='Comma-separated experiment names to exclude'
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    judge_reasoning = load_judge_reasoning(results_dir)
    annotations = load_annotations(results_dir) if args.only_correct else {}
    exclude_exps = set(args.exclude_experiments.split(',')) if args.exclude_experiments else set()

    all_cases = []

    # Scan experiments
    for d in sorted(results_dir.iterdir()):
        if not d.is_dir() or d.name in exclude_exps:
            continue
        # Direct experiment
        if (d / 'results.jsonl').exists():
            all_cases.extend(extract_cases_from_experiment(d, d.name))
        # Sub-experiments
        for sub in sorted(d.iterdir()):
            if sub.is_dir() and (sub / 'results.jsonl').exists():
                all_cases.extend(extract_cases_from_experiment(sub, f'{d.name}/{sub.name}'))

    # Attach GT reasoning from judge_reasoning.json
    for case in all_cases:
        exp_reasoning = judge_reasoning.get(case['experiment'], {})
        case['gt_reasoning'] = exp_reasoning.get(case['run_id'], '')

    # Filter: only keep cases with reasoning
    before = len(all_cases)
    all_cases = [c for c in all_cases if c['gt_reasoning']]
    print(f'Cases with reasoning: {len(all_cases)} / {before}')

    # Filter: only human-confirmed correct
    if args.only_correct:
        all_cases = [c for c in all_cases if annotations.get(c['run_id'], {}).get('label') == 'correct']
        print(f'After --only-correct filter: {len(all_cases)}')

    # Filter: skip empty predictions
    all_cases = [c for c in all_cases if c['prediction'].strip()]
    print(f'After removing empty predictions: {len(all_cases)}')

    df = pd.DataFrame(all_cases)
    output_path = Path(args.output)
    df.to_parquet(output_path, index=False)
    print(f'\nWritten {len(df)} cases to {output_path}')
    print(f'Columns: {list(df.columns)}')
    print(f'Experiments: {df["experiment"].nunique()}')
    print(f'Unique run_ids: {df["run_id"].nunique()}')
    print(f'GT score distribution: pass={int((df["gt_score"] >= 1).sum())}, fail={int((df["gt_score"] < 1).sum())}')


if __name__ == '__main__':
    main()
