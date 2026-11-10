"""Select the KIOPS tolerance multiplier from public development cases only, by the same rule as Expokit.

Rule: tol = multiplier * rtol (author defaults otherwise: m_min = min(10, n), m_max = min(128, n + 1)). Choose the
largest multiplier on GRID such that it and every smaller grid value pass every request of every public dev case
with error_inf <= MARGIN * (atol + rtol * |y_ref|_inf). Timing plays no part, so grid values run in parallel.

Added on dataset v2 (2026-11-05): the pilot policy 0.01 * rtol, fixed on the v1 public cases, failed v2 public smoke
cases whose solution decays far below the input, where the acceptance threshold reduces to atol = 1e-3 * rtol.

Run inside the task image: python3 calibrate.py <public tools dir> <dev manifest> <output.json> [--jobs N]
"""

import argparse
import json
import subprocess
import sys
import tempfile

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

GRID = [1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 3e-4, 1e-4, 3e-5, 1e-5]
MARGIN = 0.1
HERE = Path(__file__).resolve().parent


def evaluate(tools, manifest, multiplier):
    with tempfile.TemporaryDirectory() as scratch:
        report = Path(scratch) / 'report.json'
        subprocess.run([sys.executable, str(Path(tools) / 'run_local.py'), '--manifest', manifest,
                        '--command', f'{HERE / "run.sh"} {multiplier}', '--repeats', '1', '--output', str(report)],
                       check=True, capture_output=True)
        runs = json.loads(report.read_text())['runs']
    worst, failures = 0.0, []
    for run in runs:
        if not run['valid']:
            failures.append({'id': run['id'], 'reason': run['reason'], 'stderr': run['stderr'][-300:]})
            continue
        worst = max([worst] + [e['scaled_error'] for e in run['errors']])
    return {'multiplier': multiplier, 'cases': len(runs), 'failures': failures, 'max_scaled_error': worst,
            'qualifies': not failures and worst <= MARGIN}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('tools')
    parser.add_argument('manifest')
    parser.add_argument('output')
    parser.add_argument('--jobs', type=int, default=1)
    args = parser.parse_args()
    with ThreadPoolExecutor(args.jobs) as pool:
        results = list(pool.map(lambda m: evaluate(args.tools, args.manifest, m), GRID))
    chosen = next((r['multiplier'] for i, r in enumerate(results) if all(s['qualifies'] for s in results[i:])), None)
    Path(args.output).write_text(json.dumps({'rule': ' '.join(__doc__.strip().splitlines()[2:5]), 'grid': GRID,
                                             'margin': MARGIN, 'selected_multiplier': chosen, 'results': results},
                                            indent=2) + '\n')
    for r in results:
        print(r['multiplier'], 'qualifies' if r['qualifies'] else 'no', round(r['max_scaled_error'], 4),
              len(r['failures']), 'failures')
    print('selected', chosen)


if __name__ == '__main__':
    main()
