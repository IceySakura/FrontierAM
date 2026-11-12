"""Build the galaxy-gravity-v4 public feedback pool (tasks/galaxy-gravity/analysis/B4_PRECALIBRATION.md).

One snapshot per scale and family at 65,536, 262,144 and 1,048,576 particles (15 snapshots), generated with the task's
public generator from fresh private seeds, with full-particle reference accelerations from the task package's own
direct-summation program. The 4,194,304 scale is left out: its full reference costs about 1.8e13 pair interactions per
snapshot. Small snapshots get one all-particle run; at 1,048,576 the targets are split into chunks computed in parallel
on grade leases (pinned) and concatenated in order.

Output: tasks/galaxy-gravity/versions/v4/data/public/{cases/,manifest.json,seeds.json} (Git LFS on git.woa only, never
in a model workspace). Rerun-safe: finished snapshots are kept.

Usage: PYTHONPATH=. FRONTIERAM_RUNTIME=... make_galaxy_v4_feedback.py [--chunks 7]
"""

import argparse
import hashlib
import json
import secrets
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
V3 = ROOT / '.runtime-ops/galaxy-gravity-v3/packages'
GENERATOR = V3 / 'public/public-dev/generate.py'
DIRECT_SRC = V3 / 'operator/operator/private/provenance/direct'
OUT = ROOT / 'tasks/galaxy-gravity/versions/v4/data/public'
SCALES = (65536, 262144, 1048576)
FAMILIES = ('smooth-hernquist', 'disc', 'group', 'jaffe', 'einasto')
PYTHON = ROOT / '.venv-ops/bin/python'


def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def leased(command: list[str], label: str) -> None:
    subprocess.run([str(PYTHON), str(ROOT / 'scripts/ops/with_lease.py'), '--kind', 'grade', '--pin', '--memory-gib', '2',
                    '--label', label, '--', *command], check=True, cwd=ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--chunks', type=int, default=7)
    args = parser.parse_args()
    (OUT / 'cases').mkdir(parents=True, exist_ok=True)
    build = Path(tempfile.mkdtemp(prefix='galaxy-direct-'))
    direct = build / 'direct'
    subprocess.run(['c++', '-std=c++17', '-O3', '-march=native', '-fno-math-errno', '-ffp-contract=fast',
                    str(DIRECT_SRC / 'direct.cpp'), '-o', str(direct)], check=True)
    seeds_path = OUT / 'seeds.json'
    seeds = json.loads(seeds_path.read_text()) if seeds_path.exists() else {}
    cases = []
    for n in SCALES:
        for family in FAMILIES:
            case = f'fb-{n}-{family}'
            seeds.setdefault(case, secrets.randbits(63))
            seeds_path.write_text(json.dumps(seeds, indent=1) + '\n')
            inp, ref = OUT / 'cases' / f'{case}.in', OUT / 'cases' / f'{case}.ref'
            if not (inp.exists() and inp.stat().st_size == 32 * n):
                leased([str(PYTHON), str(GENERATOR), family, str(n), str(seeds[case]), str(inp)], f'galaxy v4 feedback gen {case}')
            if not (ref.exists() and ref.stat().st_size == 24 * n):
                if n < 1048576:
                    leased([str(direct), str(inp), str(ref)], f'galaxy v4 feedback ref {case}')
                else:
                    bounds = np.linspace(0, n, args.chunks + 1, dtype=np.int64)
                    parts = []
                    for k in range(args.chunks):
                        ids = build / f'{case}-{k}.ids'
                        np.arange(bounds[k], bounds[k + 1], dtype='<i8').tofile(ids)
                        parts.append((ids, build / f'{case}-{k}.out'))
                    with ThreadPoolExecutor(max_workers=args.chunks) as pool:
                        list(pool.map(lambda p: leased([str(direct), str(inp), str(p[1]), str(p[0])],
                                                       f'galaxy v4 feedback ref {case} part'), parts))
                    with open(ref, 'wb') as out:
                        for _, part in parts:
                            out.write(part.read_bytes())
            truth = np.fromfile(ref, dtype='<f8').reshape(-1, 3)
            if truth.shape[0] != n or not np.isfinite(truth).all() or not np.linalg.norm(truth, axis=1).min() > 0:
                raise SystemExit(f'{case}: invalid reference')
            cases.append({'case': case, 'family': family, 'n': n, 'input': f'cases/{case}.in', 'truth': f'cases/{case}.ref',
                          'input_sha256': sha(inp), 'truth_sha256': sha(ref)})
            print(json.dumps({'case': case, 'done': True}), flush=True)
    (OUT / 'manifest.json').write_text(json.dumps({'scope': 'public feedback pool (maintainer-only)',
                                                   'generator_sha256': sha(GENERATOR), 'cases': cases}, indent=1) + '\n')


if __name__ == '__main__':
    sys.exit(main())
