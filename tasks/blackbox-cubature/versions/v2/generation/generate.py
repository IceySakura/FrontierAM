"""Build the blackbox-cubature v2 case sets from the author's v4 hidden splits.

Every polynomial_ridge case is replaced by a kinked_ridge_sum case drawn from the same seed (case ID, suite and
dimension unchanged); fourier_ridge and product_peak cases are copied. Private = splits B and C (144). Public = the
first two cases of split A per (family, dimension), 24 in all: each suite holds only two families, so no single
suite covers all three.

Usage: generate.py PACKAGE_ROOT OUT_DATA_DIR      (PACKAGE_ROOT = the extracted blackbox_cubature/ directory)
"""
import json
import sys

from pathlib import Path

import numpy as np

SETS = {'private': {'B': ('cross_splits/split_B/cases', ('H1', 'H2', 'H3')),
                    'C': ('cross_splits/split_C_20261114/cases', ('H1', 'H2', 'H3'))},
        'public': {'A': ('task/private_hidden/cases', ('H1', 'H2', 'H3'))}}
PUBLIC_PER_CELL = 2


def kinked_ridge_sum(case: dict) -> dict:
    rng = np.random.default_rng([int(case['seed']), 20261116])
    ridges = []
    for _ in range(int(rng.integers(2, 4))):
        v = rng.normal(0.0, 1.0, size=int(case['dimension']))
        v /= np.linalg.norm(v)
        ridges.append({'v': v.tolist(), 'u': rng.uniform(0.25, 0.75, size=v.size).tolist(),
                       'c': float(rng.uniform(0.5, 3.0)), 'p': int(rng.integers(2, 4))})
    kept = {k: case[k] for k in ('suite', 'case_id', 'dimension', 'seed', 'budget')}
    return {**kept, 'family': 'kinked_ridge_sum', 'ridges': ridges}


def main():
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    for name, splits in SETS.items():
        cases = []
        for split, (folder, suites) in splits.items():
            for suite in suites:
                for case in json.loads((root / folder / f'{suite}.json').read_text())['cases']:
                    case = kinked_ridge_sum(case) if case['family'] == 'polynomial_ridge' else case
                    cases.append({**case, 'split': split, 'suite': suite})
        if name == 'public':
            taken: dict[tuple, int] = {}
            kept = []
            for case in cases:
                cell = (case['family'], case['dimension'])
                if taken.get(cell, 0) < PUBLIC_PER_CELL:
                    taken[cell] = taken.get(cell, 0) + 1
                    kept.append(case)
            cases = kept
        (out / name).mkdir(parents=True, exist_ok=True)
        (out / name / 'cases.json').write_text(json.dumps(
            {'source': 'author v4 splits (blackbox_cubature.zip) with polynomial_ridge replaced by kinked_ridge_sum '
                       '(tasks/blackbox-cubature/versions/v2/generation/generate.py)', 'cases': cases}, indent=1) + '\n')
        print(name, len(cases), {f: sum(c['family'] == f for c in cases) for f in sorted({c['family'] for c in cases})})


if __name__ == '__main__':
    main()
