"""Precision rule of the v3 formal sets (DUAL_PLAN 9.4): an instance is kept only if its error floor is at least two
orders below the strong reference's typical error, e_floor <= median_s e_GPCA-RQMC(s) - 2, the median over 8 seeds at
N = 2^14 (baselines/methods.py:as_rqmc, the method of the frozen reference). The median, not the anchor's own run,
keeps the selection independent of the error the anchor happens to have on the instance.

Writes OUT_DIR/precision.json (per instance: e_floor, the 8 errors, median, margin) and prints the IDs that fail, in
the --redraw form of generate.py.

Usage: precision_rule.py INSTANCES_JSON OUT_DIR [--workers 2] [--seeds 8]
"""
import os

for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_var, '1')

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import warnings  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'baselines'))
import methods  # noqa: E402
import oracle  # noqa: E402

MARGIN = 2.0
BUDGET = 2 ** 14


def check(inst: dict, seeds: int) -> dict:
    warnings.filterwarnings('ignore')
    g = oracle.make(inst['spec'])
    ref, floor = inst['reference']['value'], inst['reference']['e_floor']
    errors = []
    for s in range(seeds):
        f = methods.Counted(g, inst['d'], BUDGET)
        est, _ = methods.as_rqmc(f, inst['d'], 1000 * s + 17)
        errors.append(math.log10(max(abs(est - ref) / abs(ref), 1e-300)))
    med = float(np.median(errors))
    return {'id': inst['id'], 'family': inst['family'], 'e_floor': floor, 'errors': errors, 'median': med,
            'margin': med - floor, 'pass': floor <= med - MARGIN}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('instances', type=Path)
    ap.add_argument('out', type=Path)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--seeds', type=int, default=8)
    args = ap.parse_args()
    data = json.loads(args.instances.read_text())
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(check, data['instances'], [args.seeds] * len(data['instances'])))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'precision.json').write_text(json.dumps({'margin': MARGIN, 'budget': BUDGET, 'rows': rows}, indent=1) + '\n')
    for r in rows:
        flag = 'ok  ' if r['pass'] else 'FAIL'
        print(f"{flag} {r['id']:40s} floor {r['e_floor']:7.2f} strong median {r['median']:6.2f} margin {r['margin']:5.2f}")
    failing = [r for r in rows if not r['pass']]
    redraw = data.get('redraws', {})
    print(f'{len(rows) - len(failing)} of {len(rows)} pass')
    if failing:
        print('redraw: ' + ' '.join(f"{r['id']}={redraw.get(r['id'], 0) + 1}" for r in failing))


if __name__ == '__main__':
    main()
