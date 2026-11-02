"""Compute aggregate metrics from taiji_analysis batch results.

Usage:
    python -m nexus.extensions.tasks.taiji_analysis.compute_metrics \
        --db results/taiji_bench.db \
        --output results/metrics.json
"""

import json
import sqlite3

from collections import defaultdict
from pathlib import Path


def load_results(db_path: str) -> list[dict]:
    """Load all run results from the batch SQLite DB."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
        SELECT run_id, status, start_time, end_time, error,
               config, metadata, task_output
        FROM runs
    """)
    results = []
    for row in cursor.fetchall():
        r = {
            'run_id': row['run_id'],
            'status': row['status'],
            'start_time': row['start_time'],
            'end_time': row['end_time'],
        }
        for field in ['config', 'metadata', 'task_output']:
            value = row[field]
            r[field] = json.loads(value) if value else {}
        results.append(r)
    conn.close()
    return results


def compute_metrics(results: list[dict]) -> dict:
    """Compute aggregated metrics from batch results."""
    total = len(results)
    completed = []
    expired = 0
    errors = 0

    for r in results:
        task_output = r.get('task_output', {})
        status = task_output.get('status', r.get('status', ''))
        if status == 'expired':
            expired += 1
        elif status in ('completed', 'success'):
            completed.append(task_output)
        else:
            errors += 1

    scores = [r['score'] for r in completed if 'score' in r]
    avg_score = sum(scores) / len(scores) if scores else 0.0
    pass_count = sum(1 for s in scores if s >= 1.0)

    latencies: list[float] = []
    turns: list[int] = []
    skill_counts: dict[str, int] = defaultdict(int)

    for r in completed:
        meta = r.get('metadata', {})
        if meta.get('latency_seconds'):
            latencies.append(meta['latency_seconds'])
        if meta.get('turns_used'):
            turns.append(meta['turns_used'])
        for skill in meta.get('skills_used', []):
            skill_counts[skill] += 1

    n = len(completed) or 1

    return {
        'total': total,
        'completed': len(completed),
        'expired': expired,
        'errors': errors,
        'score': {
            'avg': round(avg_score, 4),
            'pass_rate': round(pass_count / n, 4),
        },
        'observability': {
            'avg_latency_seconds': round(sum(latencies) / len(latencies), 1) if latencies else None,
            'avg_turns': round(sum(turns) / len(turns), 1) if turns else None,
            'skill_usage_rate': {k: round(v / n, 3) for k, v in sorted(skill_counts.items(), key=lambda x: -x[1])},
        },
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Compute taiji_analysis benchmark metrics')
    parser.add_argument('--db', required=True, help='Batch results DB path')
    parser.add_argument('--output', default=None, help='Output JSON path (default: metrics.json next to DB)')
    args = parser.parse_args()

    results = load_results(args.db)
    metrics = compute_metrics(results)

    output = json.dumps(metrics, indent=2, ensure_ascii=False)
    out_path = Path(args.output) if args.output else Path(args.db).parent / 'metrics.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(output)
    print(f'Metrics saved to {out_path}')
