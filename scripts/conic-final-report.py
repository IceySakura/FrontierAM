"""Final conic tables: each candidate against the locally rerun reference under both limit sets.

Usage: conic-final-report.py CONFIG.json OUT.json
CONFIG: {"store": "store", "reference_suite": ..., "reference_recal": "...-reference-outcomes.json",
         "candidates": [{"label": ..., "suite": ..., "recal": "merge output or null"}]}
"""

import importlib.util
import json
import sys

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('compare', ROOT / 'scripts/conic-suite-compare.py')
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)
GROUPS = {c['id']: c['group'] for c in
          json.loads((ROOT / 'tasks/conic/original/tests/private/data/manifest.json').read_text())['instances']}


def recal_row(reference: dict, candidate: dict) -> dict:
    ref, cand = reference['decisions'], candidate['decisions']
    only_cand = [k for k in ref if cand[k] == 'solved' and ref[k] != 'solved']
    only_ref = [k for k in ref if ref[k] == 'solved' and cand[k] != 'solved']
    border = {k for k, d in candidate['detail'].items() if abs(d['slowest'] / d['limit'] - 1) <= compare.BAND}
    robust_cand = sum(k not in border for k in only_cand)
    robust_ref = sum(k not in border for k in only_ref)
    groups = {}
    for k, decision in cand.items():
        g = groups.setdefault(GROUPS[k], {'cases': 0, 'solved': 0, 'reference_solved': 0})
        g['cases'] += 1
        g['solved'] += decision == 'solved'
        g['reference_solved'] += ref[k] == 'solved'
    solved = sum(v == 'solved' for v in cand.values())
    p_b = reference['solved'] / len(ref)
    return {'solved': solved, 'success_rate': solved / len(cand), 'reference_success_rate': p_b,
            'delta': solved / len(cand) - p_b, 'S_with_local_p_B': compare.score_s(solved / len(cand), p_b),
            'solved_only_by_candidate': len(only_cand), 'solved_only_by_reference': len(only_ref),
            'mcnemar_p': compare.mcnemar(len(only_cand), len(only_ref)),
            'discordant_without_borderline': {'candidate': robust_cand, 'reference': robust_ref,
                                              'mcnemar_p': compare.mcnemar(robust_cand, robust_ref)},
            'borderline_cases': len(border), 'groups': groups}


def main():
    config = json.loads(Path(sys.argv[1]).read_text())
    store = ROOT / config['store']
    reference = compare.load(store, config['reference_suite'])
    reference_recal = json.loads((ROOT / config['reference_recal']).read_text())
    rows = []
    for item in config['candidates']:
        row = {'label': item['label'], 'original_limits': compare.compare(reference, compare.load(store, item['suite']))}
        if item.get('recal'):
            row['recalibrated_limits'] = recal_row(reference_recal, json.loads((ROOT / item['recal']).read_text()))
        rows.append(row)
    ref_cases = reference['cases']
    out = {'reference': {'original_limits': {'solved': sum(c['solved'] for c in ref_cases.values()),
                                             'borderline': sum(compare.borderline(c) for c in ref_cases.values())},
                         'recalibrated_limits': {'solved': reference_recal['solved']}},
           'candidates': rows}
    Path(sys.argv[2]).write_text(json.dumps(out, indent=2, ensure_ascii=False) + '\n')
    for row in rows:
        for name in ('original_limits', 'recalibrated_limits'):
            if name in row:
                r = row[name]
                print(row['label'], name, r['solved'], round(r['success_rate'], 4), 'ref', round(r['reference_success_rate'], 4),
                      'only cand/ref', r['solved_only_by_candidate'], r['solved_only_by_reference'],
                      'p', '%.2g' % r['mcnemar_p'], 'robust', r['discordant_without_borderline'])


if __name__ == '__main__':
    main()
