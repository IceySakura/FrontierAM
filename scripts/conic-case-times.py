"""Compare per-case solve times and outcomes of conic grades against a reference grade.

Usage: conic-case-times.py STORE BASE_JOB JOB [JOB ...]
Prints, per job, the cases whose outcome differs from BASE_JOB and the median time ratio over cases solved in both.
"""

import json
import statistics
import sys

from pathlib import Path


def cases(store: Path, job: str) -> dict:
    out = {}
    for case in sorted((store / 'results' / job / 'grade/verifier/cases').iterdir()):
        records = [json.loads(r.read_text()) for r in sorted(case.glob('repeat-*/record.json'))]
        times = [r['elapsed_seconds'] for r in records]
        out[case.name] = {
            'outcomes': [r['outcome'] for r in records],
            'median_seconds': statistics.median(times),
            'limit': records[0]['time_limit_seconds'],
        }
    return out


def main():
    store, base, jobs = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
    reference = cases(store, base)
    for job in jobs:
        current = cases(store, job)
        flips = {k: (reference[k]['outcomes'], v['outcomes']) for k, v in current.items()
                 if v['outcomes'] != reference[k]['outcomes']}
        ratios = [v['median_seconds'] / reference[k]['median_seconds'] for k, v in current.items()
                  if set(v['outcomes']) == set(reference[k]['outcomes']) == {'accepted'}]
        print(json.dumps({'job': job, 'outcome_changes': flips, 'solved_in_both': len(ratios),
                          'median_time_ratio': round(statistics.median(ratios), 4) if ratios else None,
                          'max_time_ratio': round(max(ratios), 4) if ratios else None}))


if __name__ == '__main__':
    main()
