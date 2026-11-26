"""Merge galaxy-gravity-v4 suite shards: reference anchors (E0, E50) and, optionally, a candidate's score.

Each shard part lists its cases with the mean error of the three calls (None unless all three were valid). The raw
error is the equal-weight mean over the 20 scale x family cells of the mean over each cell's snapshots; any failed
call invalidates the submission (score 0). The score uses the v3 log mapping (arena.public_v63.galaxy_score).

With --per-group (v6.3 board since 2026-11-26, user decision) a failure costs only its cell: a cell with a failed or
ungraded case takes the error of the 0-point anchor (classic tree, theta 0.8) in that cell, the other cells keep their
own, and the same mean and mapping apply; a build failure thus still scores 0. The all-or-nothing score stays in
score_complete. An infrastructure failure is not scored either way.

Usage: galaxy-v4-suite-score.py --store STORE --reference SUITE [--candidate SUITE] --out OUT.json [--per-group]
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import arena.tasks  # noqa: E402,F401  (operator_tasks must be imported through arena.tasks)
from arena.public_v63 import galaxy_score  # noqa: E402


def merged(store: Path, suite: str) -> dict:
    record = json.loads((store / 'suites' / f'{suite}.json').read_text())
    parts: dict[str, dict] = {}
    for shard in record['shards']:
        path = store / 'results' / shard['job'] / 'grade/metrics.json'
        if not path.exists():
            raise SystemExit(f'{suite}: shard {shard["job"]} has no metrics')
        for name, part in json.loads(path.read_text())['parts'].items():
            total = parts.setdefault(name, {'status': 'ok', 'reasons': [], 'cases': []})
            if part.get('status') != 'ok' and total['status'] != 'infrastructure_failure':
                total['status'] = part.get('status')
                total['reasons'].append(part.get('reason'))
            total['cases'].extend(part.get('cases') or [])
    for total in parts.values():
        cells: dict[str, list[float]] = {}
        for c in total['cases']:
            cells.setdefault(f'{c["n"]}/{c["family"]}', []).append(c['raw_error'])
        complete = total['status'] == 'ok' and all(v is not None for c in cells.values() for v in c)
        total['cells'] = {k: (sum(v) / len(v) if all(x is not None for x in v) else None) for k, v in sorted(cells.items())}
        total['raw_error'] = sum(total['cells'].values()) / len(total['cells']) if complete and total['cells'] else None
        total['cases_graded'] = len(total['cases'])
        total['max_cpu_fraction'] = max((c['cpu_s']['max'] / c['cpu_limit_s'] for c in total['cases'] if c.get('cpu_s')),
                                        default=None)
    return parts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--store', type=Path, required=True)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--candidate')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--per-group', action='store_true', help='failed cells at the 0-point anchor (see above)')
    args = parser.parse_args()
    reference = merged(args.store, args.reference)
    weak, strong = reference['weak'], reference['strong']
    result = {'reference_suite': args.reference, 'weak': weak, 'strong': strong,
              'anchors_ok': (weak['raw_error'] is not None and strong['raw_error'] is not None
                             and weak['raw_error'] > strong['raw_error'] > 0)}
    if args.candidate:
        candidate = merged(args.store, args.candidate)['candidate']
        result.update(candidate_suite=args.candidate, candidate=candidate)
        if candidate['status'] == 'submission_failure':
            result['score'] = 0.0
        elif candidate['raw_error'] is not None and result['anchors_ok']:
            result['score'] = galaxy_score(candidate['raw_error'], weak['raw_error'], strong['raw_error'])
        else:
            result['score'] = None
        if args.per_group and candidate['status'] != 'infrastructure_failure' and result['anchors_ok']:
            own = candidate['cells']
            failed = sorted(k for k in weak['cells'] if own.get(k) is None)
            cells = {k: weak['cells'][k] if k in failed else own[k] for k in weak['cells']}
            raw = sum(cells.values()) / len(cells)
            result.update(rule='per-group', score_complete=result['score'], failed_cells=failed, per_group_cells=cells,
                          per_group_raw_error=raw, score=galaxy_score(raw, weak['raw_error'], strong['raw_error']))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('weak', 'strong', 'candidate')}
                     | {'E0': weak['raw_error'], 'E50': strong['raw_error'],
                        'E': (result.get('candidate') or {}).get('raw_error')}))


if __name__ == '__main__':
    main()
