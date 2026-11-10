"""Compare full-suite conic assessments with the locally rerun reference, case by case.

Usage: conic-suite-compare.py STORE REFERENCE_SUITE CANDIDATE_SUITE [CANDIDATE_SUITE ...] [--out results/x]

A case counts as solved when all three repeats were accepted within its frozen limit (the original rule). Paired
differences use an exact two-sided McNemar test on discordant cases. A case is borderline for a submission when its
slowest repeat lies within 2% of the limit on either side; the test is repeated without cases borderline for
either side.
"""

import argparse
import ast
import json
import math
import statistics

from pathlib import Path

BAND = 0.02


def load(store: Path, suite: str) -> dict:
    report = json.loads((store / 'suites' / f'{suite}.report.json').read_text())
    if not report['complete'] or report['cases_graded'] != 650:
        raise SystemExit(f'{suite} is incomplete: {report["cases_graded"]} cases, missing {report["missing_shards"]}')
    meta = json.loads((store / 'submissions' / report['submission'] / 'meta.json').read_text())
    shards = []
    for shard in report['shards']:
        for case in json.loads((store / 'results' / shard['job'] / 'grade/verifier/summary.json').read_text())['cases']:
            shards.append(case)
    cases = {}
    for case in shards:
        times = case['elapsed_seconds']
        times = ast.literal_eval(times) if isinstance(times, str) else times
        cases[case['case_id']] = {
            'group': case['group'], 'solved': bool(case['success']), 'limit': case['time_limit_seconds'],
            'slowest': max(times) if times else None, 'median': case.get('median_elapsed_seconds'),
            'outcomes': case.get('repeat_outcomes'),
        }
    label = 'reference (Clarabel 0.11.1)' if meta.get('origin') == 'reference' else f'{meta.get("model")} {meta.get("protocol")}'
    return {'suite': suite, 'label': label, 'submission': report['submission'], 'cases': cases}


def borderline(case: dict) -> bool:
    return case['slowest'] is not None and abs(case['slowest'] / case['limit'] - 1) <= BAND


def mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def score_s(p: float, p_b: float) -> float:
    if p <= 0.9 * p_b:
        return 0.0
    if p <= p_b:
        return 5 * (p - 0.9 * p_b) / (0.1 * p_b)
    return 5 + 5 * (p - p_b) / (1 - p_b)


def compare(reference: dict, candidate: dict) -> dict:
    ref, cand = reference['cases'], candidate['cases']
    only_cand = sorted(k for k in ref if cand[k]['solved'] and not ref[k]['solved'])
    only_ref = sorted(k for k in ref if ref[k]['solved'] and not cand[k]['solved'])
    robust = [k for k in only_cand + only_ref if not borderline(ref[k]) and not borderline(cand[k])]
    robust_cand = sum(k in only_cand for k in robust)
    both = [k for k in ref if ref[k]['solved'] and cand[k]['solved']]
    ratios = [cand[k]['median'] / ref[k]['median'] for k in both if ref[k]['median'] and cand[k]['median']]
    solved = sum(c['solved'] for c in cand.values())
    p_b = sum(c['solved'] for c in ref.values()) / len(ref)
    groups = {}
    for k, c in cand.items():
        g = groups.setdefault(c['group'], {'cases': 0, 'solved': 0, 'reference_solved': 0})
        g['cases'] += 1
        g['solved'] += c['solved']
        g['reference_solved'] += ref[k]['solved']
    return {
        'label': candidate['label'], 'suite': candidate['suite'], 'submission': candidate['submission'],
        'solved': solved, 'success_rate': solved / len(cand), 'reference_success_rate': p_b,
        'delta': solved / len(cand) - p_b, 'S_with_local_p_B': score_s(solved / len(cand), p_b),
        'solved_only_by_candidate': len(only_cand), 'solved_only_by_reference': len(only_ref),
        'mcnemar_p': mcnemar(len(only_cand), len(only_ref)),
        'discordant_without_borderline': {'candidate': robust_cand, 'reference': len(robust) - robust_cand,
                                          'mcnemar_p': mcnemar(robust_cand, len(robust) - robust_cand)},
        'borderline_cases': sum(borderline(c) for c in cand.values()),
        'both_solved': len(both),
        'geomean_time_ratio_both_solved': math.exp(statistics.fmean(map(math.log, ratios))) if ratios else None,
        'groups': groups, 'cases_only_candidate': only_cand, 'cases_only_reference': only_ref,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('store', type=Path)
    parser.add_argument('reference')
    parser.add_argument('candidates', nargs='*')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    reference = load(args.store, args.reference)
    ref_cases = reference['cases']
    summary = {
        'reference': {'suite': args.reference, 'submission': reference['submission'],
                      'solved': sum(c['solved'] for c in ref_cases.values()),
                      'success_rate': sum(c['solved'] for c in ref_cases.values()) / len(ref_cases),
                      'borderline_cases': sorted(k for k, c in ref_cases.items() if borderline(c)),
                      'groups': {g: {'cases': sum(c['group'] == g for c in ref_cases.values()),
                                     'solved': sum(c['solved'] for c in ref_cases.values() if c['group'] == g)}
                                 for g in sorted({c['group'] for c in ref_cases.values()})}},
        'candidates': [compare(reference, load(args.store, s)) for s in args.candidates],
    }
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    if args.out:
        args.out.with_suffix('.json').write_text(text + '\n')
    print(json.dumps({'reference': {k: v for k, v in summary['reference'].items() if k != 'borderline_cases'}
                      | {'borderline': len(summary['reference']['borderline_cases'])},
                      'candidates': [{k: v for k, v in c.items() if not k.startswith('cases_')}
                                     for c in summary['candidates']]}, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
