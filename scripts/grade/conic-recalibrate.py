"""Recalibrate conic time limits on this machine with the task's own policy, reference-max-with-margin-v1.

limit = clamp(1.2 x slowest of the reference's three local repeats, 1, 300) where the reference succeeds (all three
repeats accepted); 300 seconds where it does not. Local repeats come from the full-suite reference run under the
original limits and, for cases that failed at those limits, from a remeasurement with relaxed limits.

Usage: conic-recalibrate.py STORE FULL_SUITE OUTPUT_MANIFEST MEASURE_STORE:MEASURE_SUITE [...]
(later remeasurements override earlier ones for the cases they cover)
"""

import ast
import json
import sys

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL = ROOT / 'tasks/conic/versions/original/package/tests/private/data/manifest.json'


def local_runs(store: Path, suite: str) -> dict:
    record = json.loads((store / 'suites' / f'{suite}.json').read_text())
    out = {}
    for shard in record['shards']:
        summary = json.loads((store / 'results' / shard['job'] / 'grade/verifier/summary.json').read_text())
        for case in summary['cases']:
            times = case['elapsed_seconds']
            times = ast.literal_eval(times) if isinstance(times, str) else times
            out[case['case_id']] = {'success': bool(case['success']), 'slowest': max(times)}
    return out


def main():
    store, full, output = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    runs = local_runs(store, full)
    remeasured = {}
    for pair in sys.argv[4:]:
        measure_store, measure = pair.split(':', 1)
        remeasured.update(local_runs(Path(measure_store), measure))
    runs.update(remeasured)
    manifest = json.loads(ORIGINAL.read_text())
    policy = manifest['time_budget_policy']
    counts = {'solved': 0, 'unsolved': 0, 'remeasured': len(remeasured), 'remeasured_solved': 0}
    for case in manifest['instances']:
        run = runs[case['id']]
        if run['success']:
            limit = min(policy['maximum_seconds'],
                        max(policy['minimum_seconds'], policy['successful_reference_multiplier'] * run['slowest']))
            counts['solved'] += 1
            counts['remeasured_solved'] += case['id'] in remeasured
        else:
            limit = policy['unsuccessful_reference_seconds']
            counts['unsolved'] += 1
        case['time_limit_seconds'] = round(limit, 6)
    manifest['time_budget_policy'] = {**policy, 'recalibrated_on': 'this host (local rerun of the reference)',
                                      'sources': {'full_suite': full, 'remeasurements': sys.argv[4:]}}
    output.write_text(json.dumps(manifest, indent=2) + '\n')
    # Under its own recalibrated limits the reference solves exactly the cases it solved locally.
    outcomes = {k: ('solved' if v['success'] else 'unsolved') for k, v in runs.items()}
    output.with_name(output.stem + '-reference-outcomes.json').write_text(
        json.dumps({'solved': counts['solved'], 'decisions': outcomes}, indent=2) + '\n')
    print(json.dumps(counts))


if __name__ == '__main__':
    main()
