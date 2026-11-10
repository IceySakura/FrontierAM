"""Select the Expokit tolerance multiplier from public development cases only.

Rule: tol = multiplier * rtol, with the Krylov size at the author's default m = 30. Choose the largest
multiplier on GRID such that it and every smaller grid value pass every request of every public dev case
with error_inf <= MARGIN * (atol + rtol * |y_ref|_inf). Timing plays no part in the selection.

The first version took the largest single qualifying value with MARGIN = 0.5. On public data alone, accuracy
proved non-monotonic above 1e3 (tol near anorm-scaled breakdown territory: 1e4 failed while 3e4 passed), so
the rule now requires a monotone safe region and a 10x margin. Public time differs by about 3% across 1e3-3e3.

Run inside the task image: python3 calibrate.py <public tools dir> <dev manifest> <binary> <output.json>
"""

import json
import subprocess
import sys
import tempfile

from pathlib import Path

# The first grid topped out at 100 and every value qualified, so it was extended upward (public data only).
GRID = [1e6, 3e5, 1e5, 3e4, 1e4, 3e3, 1e3, 300, 100, 30, 10, 3, 1, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001]
MARGIN = 0.1


def evaluate(tools, manifest, binary, multiplier):
    with tempfile.TemporaryDirectory() as scratch:
        report = Path(scratch) / 'report.json'
        subprocess.run([sys.executable, str(Path(tools) / 'run_local.py'), '--manifest', manifest,
                        '--command', f'{binary} --tol-multiplier {multiplier}', '--repeats', '1',
                        '--output', str(report)], check=True, capture_output=True)
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
    tools, manifest, binary, output = sys.argv[1:5]
    results = [evaluate(tools, manifest, binary, multiplier) for multiplier in GRID]
    chosen = next((r['multiplier'] for i, r in enumerate(results) if all(s['qualifies'] for s in results[i:])), None)
    Path(output).write_text(json.dumps({'rule': ' '.join(__doc__.strip().splitlines()[2:5]), 'grid': GRID, 'margin': MARGIN,
                                        'selected_multiplier': chosen, 'results': results}, indent=2) + '\n')
    for r in results:
        print(r['multiplier'], 'qualifies' if r['qualifies'] else 'no', round(r['max_scaled_error'], 4),
              len(r['failures']), 'failures')
    print('selected', chosen)


if __name__ == '__main__':
    main()
