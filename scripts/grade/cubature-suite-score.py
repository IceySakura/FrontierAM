"""Merge blackbox-cubature suite shards and score a candidate against the reference rows.

Each shard's metrics hold per-part rows (weak/strong for the reference pair, candidate otherwise). The anchors come
from a reference suite (--reference) or, by default, from the store's active baseline for the task.

v3 (two tracks, per-instance floors, arena/operator_drivers/blackbox_cubature_v3.py) also reports the tracks and
the error-bound diagnostics of the candidate.

Usage: cubature-suite-score.py --store STORE [--task blackbox-cubature-v2] [--reference SUITE] [--candidate SUITE] --out OUT.json
"""

import argparse
import json
import sys

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import arena.tasks  # noqa: E402,F401  (operator_tasks must be imported through arena.tasks)
from arena.operator_drivers import blackbox_cubature, blackbox_cubature_v3  # noqa: E402
from arena.operator_tasks import CLASSES  # noqa: E402


def merged(store: Path, suite: str) -> dict:
    record = json.loads((store / 'suites' / f'{suite}.json').read_text())
    parts: dict[str, dict] = {}
    for shard in record['shards']:
        path = store / 'results' / shard['job'] / 'grade/metrics.json'
        if not path.exists():
            raise SystemExit(f'{suite}: shard {shard["job"]} has no metrics')
        for name, part in json.loads(path.read_text())['parts'].items():
            total = parts.setdefault(name, {'status': 'ok', 'rows': [], 'reasons': [], 'max_wall_s': 0.0})
            if part.get('status') != 'ok' and total['status'] != 'submission_failure':
                total['status'] = part.get('status')
            if part.get('reason'):
                total['reasons'].append(part['reason'])
            total['rows'] += part.get('rows') or []
            total['max_wall_s'] = max(total['max_wall_s'], part.get('max_wall_s') or 0.0)
    return parts


def active_baseline(store: Path, task: str) -> dict:
    root = store / 'baselines' / task
    return json.loads((root / (root / 'ACTIVE').read_text().strip() / 'baseline.json').read_text())['metrics']['parts']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--store', type=Path, required=True)
    parser.add_argument('--task', default='blackbox-cubature',
                        choices=sorted(name for name in CLASSES if name.startswith('blackbox-cubature')))
    parser.add_argument('--reference')
    parser.add_argument('--candidate')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    v3 = args.task.startswith('blackbox-cubature-v3')
    if v3:
        gain, score_100 = (lambda rows, ref, _guard: blackbox_cubature_v3.gain(rows, ref)), blackbox_cubature_v3.score_100
    else:
        gain, score_100 = blackbox_cubature.gain, blackbox_cubature.score_100
    guard = getattr(CLASSES[args.task], 'guard', frozenset())
    reference = merged(args.store, args.reference) if args.reference else active_baseline(args.store, args.task)
    weak, strong = reference['weak'], reference['strong']
    anchors_ok = weak['status'] == strong['status'] == 'ok'
    mc = gain(weak['rows'], strong['rows'], guard) if anchors_ok else None
    result = {'reference': args.reference or 'active baseline', 'anchors_ok': anchors_ok and mc['gain'] < 0,
              'mc_gain': mc and mc['gain'], 'mc_families': mc and mc['families']}
    if args.candidate:
        candidate = merged(args.store, args.candidate)['candidate']
        result.update(candidate_suite=args.candidate, status=candidate['status'], reasons=candidate['reasons'],
                      cases=len(candidate['rows']), max_wall_s=candidate['max_wall_s'])
        if candidate['status'] == 'submission_failure':
            result['score'] = 0.0
        elif candidate['status'] == 'ok' and result['anchors_ok']:
            ids = {row['id'] for row in candidate['rows']}
            g = gain(candidate['rows'], strong['rows'], guard)
            g0 = gain([row for row in weak['rows'] if row['id'] in ids], strong['rows'], guard)
            result.update(gain=g['gain'], families=g['families'], score=score_100(g, g0))
            if v3:
                result.update(tracks=g['tracks'], mc_tracks=g0['tracks'], cells=g['cells'],
                              bounds=blackbox_cubature_v3.bound_diagnostics(candidate['rows']))
        else:
            result['score'] = None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
