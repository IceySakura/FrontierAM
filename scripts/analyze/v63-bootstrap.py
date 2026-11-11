"""Paired case bootstrap of v6.3 scores (results/INSIGHTS.md): species-tree-v3 (96 cases) and polarization-demosaicking-v2
(32 scenes).

Each resample draws cases with replacement and recomputes both reference anchors and every candidate's raw metric on
the same cases, then maps to the 100-point score. Reports the point score, its 95% interval and the interval of the
iterative - single difference. Reads only grading records in the local stores; prints summary numbers, no case data.

Usage: v63-bootstrap.py [--draws N]
"""

import argparse
import json
import sys

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import arena.tasks  # noqa: E402,F401
from arena.operator_tasks import score_100  # noqa: E402

STORE = ROOT / 'var/stores/store-v63'
GRADE_LOG = ROOT / 'var/stores/store/reruns/v63-grade.log'


def event(label: str) -> dict:
    rows = [json.loads(line) for line in GRADE_LOG.read_text().splitlines() if f'"label": "{label}"' in line]
    return rows[-1]


def species_parts(suite: str, part: str) -> dict[str, tuple[int, int]]:
    record = json.loads((STORE / 'suites' / f'{suite}.json').read_text())
    cases = {}
    for shard in record['shards']:
        summary = json.loads((STORE / 'results' / shard['job'] / 'grade' / part / 'summary.json').read_text())
        for r in summary['records']:
            cases[r['case']] = (r.get('missed_splits', r.get('possible_splits', 0)), r['possible_splits'])
    return cases


def species():
    reference = event('species-tree-v3')['suite']
    weak, strong = species_parts(reference, 'weak'), species_parts(reference, 'strong')
    cands = {s: species_parts(event(f'{s}-species-tree-v3-1')['suite'], 'candidate') for s in ('single', 'iterative')}
    ids = sorted(strong)
    def arrays(parts):
        return np.array([parts[i][0] for i in ids], float), np.array([parts[i][1] for i in ids], float)
    return ids, {'weak': arrays(weak), 'strong': arrays(strong), **{s: arrays(c) for s, c in cands.items()}}, \
        lambda m, idx: m[0][idx].sum() / m[1][idx].sum()


def polarization():
    root = STORE / 'baselines/polarization-demosaicking-v2'
    job = json.loads((root / (root / 'ACTIVE').read_text().strip() / 'baseline.json').read_text())['job']
    def rows(job_id, part):
        summary = json.loads((STORE / 'results' / job_id / 'grade' / part / 'summary.json').read_text())
        return {r['scene']: r['mean_rmse'] for r in summary['scene_rows']}
    parts = {'weak': rows(job, 'weak'), 'strong': rows(job, 'strong')}
    for s in ('single', 'iterative'):
        score = json.loads((ROOT / f'tasks/polarization-demosaicking/solutions/v6.3/gpt-6.1-sol-{s}-1/score.json').read_text())
        parts[s] = rows(score['id'], 'candidate')
    ids = sorted(parts['strong'])
    return ids, {k: (np.array([v[i] for i in ids]),) for k, v in parts.items()}, lambda m, idx: m[0][idx].mean()


def run(name, loader, draws, rng):
    ids, data, raw = loader()
    n = len(ids)
    full = np.arange(n)
    def scores(idx):
        u, w = raw(data['weak'], idx), raw(data['strong'], idx)
        return {s: score_100(raw(data[s], idx), u, w) for s in ('single', 'iterative')}
    point = scores(full)
    samples = [scores(rng.integers(0, n, n)) for _ in range(draws)]
    out = {'task': name, 'cases': n}
    for s in ('single', 'iterative'):
        values = np.array([x[s] for x in samples])
        out[s] = {'score': round(point[s], 1), 'ci95': [round(float(np.percentile(values, 2.5)), 1),
                                                        round(float(np.percentile(values, 97.5)), 1)]}
    diff = np.array([x['iterative'] - x['single'] for x in samples])
    out['iterative_minus_single_ci95'] = [round(float(np.percentile(diff, 2.5)), 1), round(float(np.percentile(diff, 97.5)), 1)]
    print(json.dumps(out))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--draws', type=int, default=5000)
    args = parser.parse_args()
    rng = np.random.default_rng(20261111)
    run('species-tree-v3', species, args.draws, rng)
    run('polarization-demosaicking-v2', polarization, args.draws, rng)


if __name__ == '__main__':
    main()
