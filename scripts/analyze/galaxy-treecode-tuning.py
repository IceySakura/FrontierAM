"""Can the 50-point galaxy reference (Treecode2 modified tree, theta 0.7, ncrit 64, -O3) be made more accurate at the
same CPU cost? Sweep theta, ncrit and -march=native on the public feedback snapshots; CPU time is the solve process's
user + sys time as the grader counts it. Run under scripts/ops/with_lease.py --pin (one leased core).

Usage: sweep.py build | sweep | verify CONFIG...   (results in /tmp/tc2-sweep/results.jsonl)
"""
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path('/data/workspace/FrontierAM')
REF = ROOT / 'tasks/galaxy-gravity/references/r4/baselines/modified'
CASES = ROOT / 'tasks/galaxy-gravity/versions/v4/data/public/cases'
LIMITS = json.loads((ROOT / 'tasks/galaxy-gravity/versions/v4/data/public/local-limits.json').read_text())
WORK = Path('/tmp/tc2-sweep')
BIN = WORK / 'bin'
OUT = WORK / 'results.jsonl'
FLAGS = {'base': [], 'native': ['-march=native'], 'fast': ['-march=native', '-ffast-math']}
NCRIT = (32, 64, 128)
THETAS = (0.55, 0.6, 0.65, 0.7)
SMALL = [c for c in LIMITS['limits'] if c.startswith(('fb-65536-', 'fb-262144-'))]
LARGE = [c for c in LIMITS['limits'] if c.startswith('fb-1048576-')]


def build() -> None:
    src = WORK / 'src'
    if not src.exists():
        src.mkdir(parents=True)
        subprocess.run(['tar', '-xzf', str(REF / 'Treecode2-16051d8.tar.gz'), '-C', str(src)], check=True)
        subprocess.run(['make', '-s', '-C', 'clib', 'ZCC=cc', 'libClib.a'], cwd=src, check=True)
        probe = (REF / 'probe_adapter.c').read_text()
        old = 'treegrav(tree, n > 64 ? 64 : n - 1);'
        assert probe.count(old) == 1
        (src / 'probe_ncrit.c').write_text(probe.replace(old, 'treegrav(tree, n > NCRIT ? NCRIT : n - 1);'))
    BIN.mkdir(exist_ok=True)
    for flag, extra in FLAGS.items():
        for ncrit in NCRIT:
            out = BIN / f'{flag}-n{ncrit}'
            if not out.exists():
                subprocess.run(['cc', '-O3', '-std=gnu17', '-DLINUX', '-DDOUBLEPREC', '-fno-strict-aliasing', *extra,
                                '-DMODTREECODE', f'-DNCRIT={ncrit}', '-I.', 'probe_ncrit.c', 'treebuild.c',
                                'treegrav_mod.c', 'clib/libClib.a', '-lm', '-o', str(out)], cwd=src, check=True)
    print('built', sorted(p.name for p in BIN.iterdir()))


def error(pred: Path, truth: Path) -> float:
    a = np.fromfile(pred, dtype='<f8').reshape(-1, 3)
    b = np.fromfile(truth, dtype='<f8').reshape(-1, 3)
    r = np.linalg.norm(a - b, axis=1) / np.linalg.norm(b, axis=1)
    return float(np.sqrt(np.mean(r * r)))


def run(binary: str, theta: float, case: str) -> dict:
    cpu = os.environ.get('LEASE_CPUSET', '0').replace('-', ',').split(',')[0]
    pred = WORK / f'pred-{os.getpid()}.bin'
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    wall = time.monotonic()
    done = subprocess.run(['taskset', '-c', cpu, str(BIN / binary), str(CASES / f'{case}.in'), str(pred), str(theta),
                           '0.01'], capture_output=True, text=True)
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    row = {'binary': binary, 'theta': theta, 'case': case, 'ok': done.returncode == 0,
           'cpu_s': round(after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime, 4),
           'wall_s': round(time.monotonic() - wall, 3), 'limit_s': LIMITS['limits'][case]}
    if row['ok']:
        row['error'] = error(pred, CASES / f'{case}.ref')
    pred.unlink(missing_ok=True)
    with OUT.open('a') as f:
        f.write(json.dumps(row) + '\n')
    print(json.dumps(row), flush=True)
    return row


def main() -> None:
    command = sys.argv[1]
    if command == 'build':
        build()
    elif command == 'sweep':
        for binary in sorted(p.name for p in BIN.iterdir()):
            for theta in THETAS:
                for case in SMALL:
                    run(binary, theta, case)
    elif command == 'fast':
        for binary in ('fast-n64', 'fast-n128'):
            for theta in (0.6, 0.65, 0.7):
                for case in SMALL:
                    run(binary, theta, case)
    elif command == 'verify':   # CONFIG = binary:theta, three runs on the 1,048,576-particle snapshots
        for config in sys.argv[2:]:
            binary, theta = config.split(':')
            for case in LARGE:
                for _ in range(3):
                    run(binary, float(theta), case)


if __name__ == '__main__':
    main()
