"""Screen human reference candidates for trajectory-covariance on the fixed 3,000-input formal subset.

Candidates (tasks/trajectory-covariance/references/candidates/<name>, submission format) run in the task image
on leased development slots (scripts/ops/run_leased.py), each input in a fresh process with a
30-second limit; private inputs are mounted read-only and stay on this machine. Scoring follows the maintainer
verifier's summary: per-case loss ||S^-1/2 (H - S) S^-1/2||_F against the reference S (sqrt(d) for a failure), mean
per (model, length) cell, then the mean over the 30 cells. This is a screening; a chosen reference is graded
afterwards with the official verifier.

Usage: traj-ref-screen.py run OUT [--candidates a,b] [--workers N] [--shards K]
       traj-ref-screen.py score OUT
"""

import argparse
import collections
import json
import math
import os
import subprocess
import sys

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CANDIDATES = ROOT / 'tasks/trajectory-covariance/references/candidates'
SEAL = ROOT / '.runtime-ops/trajectory-covariance'
PRIVATE = 'results/mcmc-long-run-covariance/formal-v1/private'


def subset() -> list[dict]:
    lock = json.loads((SEAL / 'lock.json').read_text())
    root = (SEAL / lock['roots']['operator']).resolve()
    cases = json.loads((root / PRIVATE / 'manifest.json').read_text())['cases']
    chosen = set()
    for model in sorted({c['model'] for c in cases}):
        chosen.update(sorted({c['parent'] for c in cases if c['model'] == model})[:100])
    return [c for c in cases if c['parent'] in chosen], root / PRIVATE, lock['image']['id']


def run(args) -> None:
    out = args.out.resolve()
    cases, data, image = subset()
    names = args.candidates.split(',') if args.candidates else sorted(p.name for p in CANDIDATES.iterdir() if p.is_dir())
    (out / 'lists').mkdir(parents=True, exist_ok=True)
    jobs = []
    for k in range(args.shards):
        shard = cases[k::args.shards]
        (out / 'lists' / f'{k:02d}.txt').write_text(''.join(f"{c['id']} {c['input']}\n" for c in shard))
    for name in names:
        (out / name).mkdir(exist_ok=True)
        for k in range(args.shards):
            jobs.append(f'while read id input; do [ -f /out/{name}/$id.npy ] && continue; '
                        f'timeout 30 /cand/{name}/run.sh /data/$input /out/{name}/$id.npy '
                        f'>/dev/null 2>>/out/{name}/errors.log || echo "$id" >> /out/{name}/failed.txt; '
                        f'done < /out/lists/{k:02d}.txt')
    (out / 'jobs.txt').write_text('\n'.join(jobs) + '\n')
    (out / 'logs').mkdir(exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    subprocess.run([sys.executable, str(ROOT / 'scripts/ops/run_leased.py'), str(out / 'jobs.txt'),
                    str(out), '--workers', str(args.workers), '--image', image,
                    '--mount', f'{data}:/data:ro', '--mount', f'{CANDIDATES}:/cand:ro'], env=env, check=True)


def loss(h: np.ndarray, s: np.ndarray) -> float:
    w, v = np.linalg.eigh(s)
    root = v @ np.diag(w ** -0.5) @ v.T
    return float(np.linalg.norm(root @ (h - s) @ root))


def score(args) -> None:
    cases, data, _ = subset()
    results = {}
    for directory in sorted(p for p in args.out.iterdir() if p.is_dir() and p.name not in ('lists', 'logs')):
        cells, failures = collections.defaultdict(list), 0
        for c in cases:
            path = directory / f"{c['id']}.npy"
            value = None
            if path.exists():
                try:
                    h = np.load(path, allow_pickle=False)
                    if h.shape == (c['d'], c['d']) and np.isfinite(h).all():
                        value = loss(h, np.load(data / c['reference'], allow_pickle=False))
                except (ValueError, OSError):
                    value = None
            if value is None or not math.isfinite(value):
                failures += 1
                value = math.sqrt(c['d'])
            cells[(c['model'], c['length'])].append(value)
        means = {f'{m}/{n}': sum(v) / len(v) for (m, n), v in sorted(cells.items())}
        results[directory.name] = {'L': sum(means.values()) / len(means), 'failures': failures, 'cells': means}
    (args.out / 'scores.json').write_text(json.dumps(results, indent=1) + '\n')
    for name, r in sorted(results.items(), key=lambda kv: kv[1]['L']):
        print(f"{name:20s} L = {r['L']:.6f}  failures = {r['failures']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    r = sub.add_parser('run')
    r.add_argument('out', type=Path)
    r.add_argument('--candidates')
    r.add_argument('--workers', type=int, default=5)
    r.add_argument('--shards', type=int, default=6)
    s = sub.add_parser('score')
    s.add_argument('out', type=Path)
    args = parser.parse_args()
    run(args) if args.command == 'run' else score(args)


if __name__ == '__main__':
    main()
