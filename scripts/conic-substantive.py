"""Split conic results by the frozen per-case limit: cases at the 1-second floor versus the rest.

422 of 650 limits sit at the 1-second floor, where per-process startup and import time decide the outcome; the
remaining cases are where solver work dominates. Prints solved counts per stratum for both limit sets.

Usage: conic-substantive.py REPORT_CONFIG.json
"""

import importlib.util
import json
import sys

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('compare', ROOT / 'scripts/conic-suite-compare.py')
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)
LIMITS = {c['id']: c['time_limit_seconds'] for c in
          json.loads((ROOT / 'tasks/conic/original/tests/private/data/manifest.json').read_text())['instances']}
FLOOR = {k for k, v in LIMITS.items() if v == 1.0}


def strata(solved: set) -> dict:
    return {'floor_1s': len(solved & FLOOR), 'above_1s': len(solved - FLOOR)}


def main():
    config = json.loads(Path(sys.argv[1]).read_text())
    store = ROOT / config['store']
    reference = compare.load(store, config['reference_suite'])
    ref_recal = json.loads((ROOT / config['reference_recal']).read_text())['decisions']
    out = {'cases': {'floor_1s': len(FLOOR), 'above_1s': len(LIMITS) - len(FLOOR)},
           'reference': {'original_limits': strata({k for k, c in reference['cases'].items() if c['solved']}),
                         'recalibrated_limits': strata({k for k, v in ref_recal.items() if v == 'solved'})}}
    for item in config['candidates']:
        cand = compare.load(store, item['suite'])
        row = {'original_limits': strata({k for k, c in cand['cases'].items() if c['solved']})}
        ref_solved = {k for k, c in reference['cases'].items() if c['solved']}
        cand_solved = {k for k, c in cand['cases'].items() if c['solved']}
        above = set(LIMITS) - FLOOR
        b, c = len((cand_solved - ref_solved) & above), len((ref_solved - cand_solved) & above)
        row['original_above_1s_mcnemar'] = {'candidate_only': b, 'reference_only': c, 'p': compare.mcnemar(b, c)}
        if item.get('recal'):
            decisions = json.loads((ROOT / item['recal']).read_text())['decisions']
            cand_r = {k for k, v in decisions.items() if v == 'solved'}
            ref_r = {k for k, v in ref_recal.items() if v == 'solved'}
            row['recalibrated_limits'] = strata(cand_r)
            b, c = len((cand_r - ref_r) & above), len((ref_r - cand_r) & above)
            row['recalibrated_above_1s_mcnemar'] = {'candidate_only': b, 'reference_only': c, 'p': compare.mcnemar(b, c)}
        out[item['label']] = row
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
