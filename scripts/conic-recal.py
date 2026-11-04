"""Evaluate conic submissions under locally recalibrated limits without regrading what is already decided.

A case is decided by one run's three repeats against one limit set. Under recalibrated limits:
  - all three repeats accepted and the slowest within the new limit: solved, from the original-limit run;
  - any repeat rejected for a reason other than time: unsolved;
  - slowest beyond the new limit: unsolved (the run was measured to that point);
  - a repeat stopped at the original limit while the new limit is larger: undecided, regraded under the new limits.

subset: STORE SUITE RECAL_MANIFEST OUT.json     write the undecided case IDs
merge:  STORE SUITE RECAL_MANIFEST [RERUN_STORE RERUN_SUITE] OUT.json
"""

import json
import sys

from pathlib import Path

ORIGINAL = Path(__file__).resolve().parents[1] / 'tasks/conic/original/tests/private/data/manifest.json'


def limits(path: Path) -> dict:
    return {c['id']: c['time_limit_seconds'] for c in json.loads(path.read_text())['instances']}


def runs(store: Path, suite: str) -> dict:
    record = json.loads((store / 'suites' / f'{suite}.json').read_text())
    out = {}
    for shard in record['shards']:
        cases = store / 'results' / shard['job'] / 'grade/verifier/cases'
        for case in cases.iterdir():
            repeats = [json.loads(p.read_text()) for p in sorted(case.glob('repeat-*/record.json'))]
            out[case.name] = [{'outcome': r['outcome'], 'reason': r.get('reason'), 'elapsed': r['elapsed_seconds']}
                              for r in repeats]
    return out


def decide(repeats: list, old: float, new: float) -> str:
    if len(repeats) != 3:
        return 'unsolved'
    if any(r['outcome'] != 'accepted' and r['reason'] != 'time_limit' for r in repeats):
        return 'unsolved'
    if max(r['elapsed'] for r in repeats) > new:
        return 'unsolved'
    if all(r['outcome'] == 'accepted' for r in repeats):
        return 'solved'
    return 'undecided' if new > old else 'unsolved'


def main():
    command, store, suite, recal = sys.argv[1], Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
    old, new = limits(ORIGINAL), limits(recal)
    first = runs(store, suite)
    decisions = {k: decide(v, old[k], new[k]) for k, v in first.items()}
    if command == 'subset':
        undecided = sorted(k for k, v in decisions.items() if v == 'undecided')
        Path(sys.argv[5]).write_text(json.dumps(undecided) + '\n')
        print(json.dumps({'suite': suite, 'undecided': len(undecided),
                          'regrade_budget_seconds': 3 * sum(new[k] for k in undecided)}))
        return
    deciding = dict(first)
    if len(sys.argv) == 8:
        rerun = runs(Path(sys.argv[5]), sys.argv[6])
        for k, v in rerun.items():
            if decisions.get(k) == 'undecided':
                decisions[k] = 'solved' if decide(v, new[k], new[k]) == 'solved' else 'unsolved'
                deciding[k] = v
    output = Path(sys.argv[-1])
    if any(v == 'undecided' for v in decisions.values()):
        raise SystemExit(f'{sum(v == "undecided" for v in decisions.values())} cases still undecided')
    detail = {k: {'slowest': max(r['elapsed'] for r in v), 'limit': new[k]} for k, v in deciding.items()}
    summary = {'suite': suite, 'limits': str(recal), 'solved': sum(v == 'solved' for v in decisions.values()),
               'cases': len(decisions), 'decisions': decisions, 'detail': detail}
    output.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k not in ('decisions', 'detail')}))


if __name__ == '__main__':
    main()
