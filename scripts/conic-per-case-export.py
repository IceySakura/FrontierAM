"""Export per-case conic metrics for the reference and every graded model, for the task author.

Writes:
  results/conic-per-case-runs.csv     one row per case x solver x run x repeat (limit, time, outcome, reason, residuals)
  results/conic-per-case-summary.csv  one row per case: limits and each solver's outcome under the recalibrated limits
  results/conic-per-case-stats.json   aggregate statistics used in CONIC_PER_CASE.md
"""

import csv
import json
import statistics

from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / 'tasks/conic/original/tests/private/data/manifest.json'
RECAL = ROOT / 'tasks/conic/local_recal/recal-manifest.json'
PUBLIC = ROOT / 'tasks/conic/original/environment/public-dev/manifest.json'
REF_OUTCOMES = ROOT / 'tasks/conic/local_recal/recal-manifest-reference-outcomes.json'

# solver -> runs in the order they decide a case: (run name, store, suite)
SOLVERS = {
    'reference': [('original_limits', 'store', 'suite-conic-20261104T112639Z-5805a6'),
                  ('remeasure_2x_limit', 'store-measure', 'suite-conic-20261104T130556Z-c921f0'),
                  ('remeasure_300s', 'store-measure2', 'suite-conic-20261104T162157Z-c36161')],
    'gpt-6.1-sol': [('original_limits', 'store', 'suite-conic-20261104T112642Z-4c4423'),
                    ('recalibrated_regrade', 'store-recal', 'suite-conic-20261104T192406Z-37e06b')],
    'gpt-6-astra': [('original_limits', 'store', 'suite-conic-20261104T120311Z-34c2e1'),
                    ('recalibrated_regrade', 'store-recal', 'suite-conic-20261104T234302Z-f73c31')],
    'hy4': [('original_limits', 'store', 'suite-conic-20261104T120314Z-3397f6'),
            ('recalibrated_regrade', 'store-recal', 'suite-conic-20261105T023029Z-4a9356')],
    'claude-opus-5-5': [('original_limits', 'store', 'suite-conic-20261104T112644Z-62e345'),
                        ('recalibrated_regrade', 'store-recal', 'suite-conic-20261104T201322Z-b5235a')],
}
RECAL_RESULTS = {'gpt-6.1-sol': 'results/conic-recal-gpt-6.1-sol.json', 'gpt-6-astra': 'results/conic-recal-gpt-6-astra.json',
                 'hy4': 'results/conic-recal-hy4.json', 'claude-opus-5-5': 'results/conic-recal-claude-opus-5-5.json'}


def load_runs(store: str, suite: str) -> dict:
    out = defaultdict(list)
    record = json.loads((ROOT / store / 'suites' / f'{suite}.json').read_text())
    for shard in record['shards']:
        for case in (ROOT / store / 'results' / shard['job'] / 'grade/verifier/cases').iterdir():
            for path in sorted(case.glob('repeat-*/record.json')):
                out[case.name].append(json.loads(path.read_text()))
    return out


def residuals(record: dict) -> dict:
    check = (record.get('checker') or {}).get('check') or {}
    res = check.get('residuals') or {}
    return {'primal': res.get('primal'), 'dual': res.get('dual'), 'gap_relative': res.get('gap_relative'),
            'certificate_margin': res.get('certificate_margin_normalized')}


def main():
    cases = {c['id']: c for c in json.loads(ORIGINAL.read_text())['instances']}
    recal = {c['id']: c['time_limit_seconds'] for c in json.loads(RECAL.read_text())['instances']}
    decisions = {'reference': json.loads(REF_OUTCOMES.read_text())['decisions']}
    detail = {}
    for solver, path in RECAL_RESULTS.items():
        merged = json.loads((ROOT / path).read_text())
        decisions[solver], detail[solver] = merged['decisions'], merged['detail']
    runs = {solver: [(name, load_runs(store, suite)) for name, store, suite in spec] for solver, spec in SOLVERS.items()}

    with (ROOT / 'results/conic-per-case-runs.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['case_id', 'group', 'n', 'm', 'nnz_P', 'nnz_A', 'cone_types', 'limit_original_s', 'limit_recalibrated_s',
                         'solver', 'run', 'run_limit_s', 'repeat', 'outcome', 'reason', 'status', 'elapsed_s',
                         'elapsed_over_limit', 'residual_primal', 'residual_dual', 'gap_relative', 'certificate_margin'])
        for cid in sorted(cases):
            c = cases[cid]
            base = [cid, c['group'], c['dims']['n'], c['dims']['m'], c['dims']['nnz_P'], c['dims']['nnz_A'],
                    '+'.join(c['cone_types']), c['time_limit_seconds'], recal[cid]]
            for solver, solver_runs in runs.items():
                for name, by_case in solver_runs:
                    for r in by_case.get(cid, []):
                        res = residuals(r)
                        writer.writerow(base + [solver, name, r['time_limit_seconds'], r['repeat'], r['outcome'], r.get('reason'),
                                                r.get('status'), round(r['elapsed_seconds'], 4),
                                                round(r['elapsed_seconds'] / r['time_limit_seconds'], 4),
                                                res['primal'], res['dual'], res['gap_relative'], res['certificate_margin']])

    def deciding(solver, cid):
        """Slowest repeat and failure reasons of the run that decides the case under the recalibrated limits."""
        chosen = None
        for name, by_case in runs[solver]:
            if cid in by_case:
                chosen = (name, by_case[cid])
        if solver != 'reference':
            name, recs = runs[solver][0][0], runs[solver][0][1][cid]
            if len(runs[solver]) > 1 and cid in runs[solver][1][1]:
                name, recs = runs[solver][1][0], runs[solver][1][1][cid]
            chosen = (name, recs)
        name, recs = chosen
        reasons = sorted({r.get('reason') for r in recs if r['outcome'] != 'accepted'})
        return name, max(r['elapsed_seconds'] for r in recs), reasons

    solver_names = list(SOLVERS)
    with (ROOT / 'results/conic-per-case-summary.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        header = ['case_id', 'group', 'n', 'm', 'nnz_A', 'cone_types', 'limit_original_s', 'limit_recalibrated_s']
        for s in solver_names:
            header += [f'{s}_solved', f'{s}_slowest_s', f'{s}_failure_reasons', f'{s}_deciding_run']
            if s != 'reference':
                header.append(f'{s}_time_ratio_vs_reference')
        writer.writerow(header)
        for cid in sorted(cases):
            c = cases[cid]
            row = [cid, c['group'], c['dims']['n'], c['dims']['m'], c['dims']['nnz_A'], '+'.join(c['cone_types']),
                   c['time_limit_seconds'], recal[cid]]
            ref_name, ref_slow, _ = deciding('reference', cid)
            for s in solver_names:
                name, slow, reasons = deciding(s, cid)
                solved = decisions[s][cid] == 'solved'
                row += [int(solved), round(slow, 4), ';'.join(reasons), name]
                if s != 'reference':
                    both = solved and decisions['reference'][cid] == 'solved'
                    row.append(round(slow / ref_slow, 3) if both else '')
            writer.writerow(row)

    # Aggregate statistics for the write-up.
    public = [c['time_limit_seconds'] for c in json.loads(PUBLIC.read_text())['instances']]
    band = lambda limit: '1s' if limit == 1.0 else ('300s' if limit == 300.0 else '1-300s')  # noqa: E731
    stats = {
        'limits': {'private_original': dict(Counter(band(c['time_limit_seconds']) for c in cases.values())),
                   'private_recalibrated': dict(Counter(band(v) for v in recal.values())),
                   'public_dev': dict(Counter(band(v) for v in public))},
        'solvers': {},
    }
    for s in solver_names:
        first = runs[s][0][1]
        fastest_1s = [min(r['elapsed_seconds'] for r in first[k]) for k in cases if recal[k] == 1.0 and first.get(k)]
        reasons = Counter()
        for k in cases:
            if decisions[s][k] != 'solved':
                _, _, rs = deciding(s, k)
                reasons[(band(recal[k]), ';'.join(rs) or 'slower_than_limit')] += 1
        by_group = Counter(cases[k]['group'] for k in cases if decisions[s][k] == 'solved')
        stats['solvers'][s] = {
            'solved': sum(v == 'solved' for v in decisions[s].values()),
            'solved_by_band': dict(Counter(band(recal[k]) for k in cases if decisions[s][k] == 'solved')),
            'solved_by_group': dict(by_group),
            'median_fastest_repeat_on_1s_cases': round(statistics.median(fastest_1s), 3),
            'unsolved_by_band_and_reason': {f'{b} | {r}': n for (b, r), n in sorted(reasons.items())},
        }
        if s != 'reference':
            ratios = [deciding(s, k)[1] / deciding('reference', k)[1] for k in cases
                      if decisions[s][k] == 'solved' and decisions['reference'][k] == 'solved' and recal[k] > 1.0]
            stats['solvers'][s]['time_ratio_vs_reference_on_both_solved_above_1s'] = {
                'cases': len(ratios), 'median': round(statistics.median(ratios), 3) if ratios else None,
                'geomean': round(statistics.geometric_mean(ratios), 3) if ratios else None}
    ref_first = runs['reference'][0][1]
    margins = sorted(max(r['elapsed_seconds'] for r in ref_first[k]) / cases[k]['time_limit_seconds']
                     for k in cases if all(r['outcome'] == 'accepted' for r in ref_first[k]) and cases[k]['time_limit_seconds'] > 1.0)
    stats['reference_slowest_over_original_limit_solved_above_1s'] = {
        'cases': len(margins), 'quantiles_10_50_90': [round(margins[int(q * (len(margins) - 1))], 3) for q in (0.1, 0.5, 0.9)]}
    failures = [r for k in cases for r in ref_first[k] if r.get('reason') == 'residual_tolerance_exceeded']
    gaps = sorted(residuals(r)['gap_relative'] for r in failures if residuals(r)['gap_relative'] is not None)
    stats['reference_residual_failures'] = {'repeats': len(failures),
                                            'gap_relative_quantiles_10_50_90': [gaps[int(q * (len(gaps) - 1))] for q in (0.1, 0.5, 0.9)] if gaps else None}
    (ROOT / 'results/conic-per-case-stats.json').write_text(json.dumps(stats, indent=2) + '\n')
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
