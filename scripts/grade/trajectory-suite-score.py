"""Merge trajectory-covariance suite shards: reference anchors (weak, strong) and, optionally, a candidate's score.

Each shard's metrics hold per-part cell sums (loss total and count per setting/length cell over that shard's inputs).
The raw error is the mean over cells of the cell's mean loss, as in the package's evaluation plan.

Usage: trajectory-suite-score.py --store STORE --reference SUITE [--candidate SUITE] --out OUT.json
"""

import argparse
import json
import sys

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import arena.tasks  # noqa: E402,F401  (operator_tasks must be imported through arena.tasks)
from arena.operator_tasks import score_100  # noqa: E402


def merged(store: Path, suite: str) -> dict:
    record = json.loads((store / 'suites' / f'{suite}.json').read_text())
    parts: dict[str, dict] = {}
    for shard in record['shards']:
        path = store / 'results' / shard['job'] / 'grade/metrics.json'
        if not path.exists():
            raise SystemExit(f'{suite}: shard {shard["job"]} has no metrics')
        for name, part in json.loads(path.read_text())['parts'].items():
            total = parts.setdefault(name, {'status': 'ok', 'cells': {}, 'cases': 0, 'reasons': []})
            if part.get('status') != 'ok':
                total['status'] = part.get('status')
                total['reasons'].append(part.get('reason'))
                continue
            for cell, (loss, count) in part['cell_sums'].items():
                cell_total = total['cells'].setdefault(cell, [0.0, 0])
                cell_total[0] += loss
                cell_total[1] += count
            total['cases'] += part['cases']
    for total in parts.values():
        means = [loss / count for loss, count in total['cells'].values() if count]
        total['raw_error'] = sum(means) / len(means) if means and total['status'] == 'ok' else None
    return parts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--store', type=Path, required=True)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--candidate')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    reference = merged(args.store, args.reference)
    weak, strong = reference['weak'], reference['strong']
    result = {'reference_suite': args.reference, 'weak': weak, 'strong': strong,
              'anchors_ok': weak['status'] == strong['status'] == 'ok' and weak['raw_error'] > strong['raw_error'] > 0}
    if args.candidate:
        candidate = merged(args.store, args.candidate)['candidate']
        result.update(candidate_suite=args.candidate, candidate=candidate)
        if candidate['status'] == 'submission_failure':
            result['score'] = 0.0
        elif candidate['status'] == 'ok' and result['anchors_ok']:
            result['score'] = score_100(candidate['raw_error'], weak['raw_error'], strong['raw_error'])
        else:
            result['score'] = None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('weak', 'strong', 'candidate')}
                     | {'weak_E': weak['raw_error'], 'strong_E': strong['raw_error'],
                        'E': (result.get('candidate') or {}).get('raw_error')}))


if __name__ == '__main__':
    main()
