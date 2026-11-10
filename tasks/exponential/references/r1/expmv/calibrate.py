"""Select expmv's precision argument per tolerance level from public development cases only.

Rule, fixed before running: for each level (p4, p7, p10) take the cheapest of half < single < double for which every
request of every public dev case at that level passes with error_inf <= MARGIN * (atol + rtol * |y_ref|_inf).
Timing plays no part. Run inside the task image: python3 calibrate.py <public tools dir> <dev manifest> <out.json>
"""

import json
import subprocess
import sys
import tempfile

from pathlib import Path

MARGIN = 0.1
ORDER = ['half', 'single', 'double']
LEVELS = {1e-4: 'p4', 1e-7: 'p7', 1e-10: 'p10'}
HERE = Path(__file__).resolve().parent


def run(tools, manifest, prec):
    policy = "struct('p4','%s','p7','%s','p10','%s')" % (prec, prec, prec)
    with tempfile.TemporaryDirectory() as scratch:
        report = Path(scratch) / 'r.json'
        subprocess.run([sys.executable, str(Path(tools) / 'run_local.py'), '--manifest', manifest, '--command',
                        f'{HERE / "run.sh"} "{policy}"', '--repeats', '1', '--output', str(report)],
                       check=True, capture_output=True)
        return json.loads(report.read_text())['runs']


def main():
    tools, manifest, output = sys.argv[1:4]
    by_level = {name: {} for name in LEVELS.values()}
    for prec in ORDER:
        for r in run(tools, manifest, prec):
            level = LEVELS[r['rtol']]
            scaled = [float('inf') if e['scaled_error'] is None else e['scaled_error'] for e in r['errors']]
            worst = max(scaled or [float('inf')]) if r['valid'] else float('inf')
            entry = by_level[level].setdefault(prec, {'cases': 0, 'max_scaled_error': 0.0, 'failures': 0})
            entry['cases'] += 1
            entry['failures'] += not r['valid']
            if worst > entry['max_scaled_error']:
                entry['max_scaled_error'], entry['worst_case'] = worst, [r['id'], r['reason'], r['errors'][:2]]
    chosen = {level: next((p for p in ORDER if results[p]['failures'] == 0 and results[p]['max_scaled_error'] <= MARGIN),
                          None) for level, results in by_level.items()}
    Path(output).write_text(json.dumps({'rule': __doc__.strip().splitlines()[2:4], 'margin': MARGIN,
                                        'selected': chosen, 'results': by_level}, indent=2) + '\n')
    print(json.dumps(chosen), json.dumps(by_level))


if __name__ == '__main__':
    main()
