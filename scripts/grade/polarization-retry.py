"""polarization-demosaicking-v3: grade once more when scenes only timed out, then compose the two gradings.

The 1.000 s per-scene limit meets short environment stalls (results/OPEN_ITEMS.md, 11-23 and 11-26): a contiguous run of
scenes times out at 1.000x s while the others take their usual time, and in another grading other scenes stall. User
decision 2026-11-26: a scene that timed out is run again, and it passes if either grading ran it within the limit.
Outputs are deterministic, so a scene that passed in both gradings must have the same per-channel RMSEs; otherwise
nothing is composed. E is the package's own aggregate over the composed rows, the score uses the store's active-baseline
anchors like every polarization score. Any failure other than a timeout is final and nothing is regraded.

Usage: polarization-retry.py --store STORE --submission SUB --out OUT_DIR
  OUT_DIR/score.json holds the first grading's job record (scripts/rounds/v63-grade.sh); it moves to score.first.json.
"""

import argparse
import csv
import importlib.util
import json
import math
import os
import subprocess
import sys

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
TASK = 'polarization-demosaicking-v3'
PACKAGE = ROOT / ('.runtime-ops/polarization-demosaicking-v3/packages/operator/'
                  'polarization-demosaicking-maintenance-vnext-1.3.0-dev/harbor/polarization-demosaicking')


def rows_of(store: Path, job: str) -> list[dict]:
    path = store / 'results' / job / 'grade/candidate/per_scene.csv'
    return list(csv.DictReader(path.open())) if path.exists() else []


def only_timeouts(rows: list[dict]) -> bool:
    failed = [r for r in rows if r['status'] != 'ok']
    return bool(failed) and all(r.get('reason') == 'timeout' for r in failed)


def compose(gradings: list[list[dict]]) -> tuple[list[dict] | None, dict]:
    """Per scene the first passing row; None when a scene never passed or passing rows disagree."""
    channels = ('rmse_0', 'rmse_45', 'rmse_90', 'rmse_135')
    by_key: dict[tuple, list[tuple[int, dict]]] = {}
    for index, rows in enumerate(gradings):
        for row in rows:
            by_key.setdefault((row['dataset'], row['scene']), []).append((index, row))
    composed, taken = [], {}
    for key, entries in by_key.items():
        ok = [(i, r) for i, r in entries if r['status'] == 'ok']
        if not ok:
            return None, {'never_within_limit': f'{key[0]}/{key[1]}'}
        first = ok[0][1]
        for _, other in ok[1:]:
            if any(not math.isclose(float(first[c]), float(other[c]), rel_tol=1e-12, abs_tol=0.0) for c in channels):
                return None, {'nondeterministic': f'{key[0]}/{key[1]}'}
        composed.append(first)
        if ok[0][0] > 0:
            taken[f'{key[0]}/{key[1]}'] = {'grading': ok[0][0] + 1, 'wall_s': float(first['wall_s'])}
    return composed, taken


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--store', type=Path, required=True)
    parser.add_argument('--submission', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    store = args.store.resolve()
    first = json.loads((args.out / 'score.json').read_text())
    first_rows = rows_of(store, first['id'])
    if first.get('status') == 'ok' or not only_timeouts(first_rows):
        return
    before = set((store / 'jobs/done').glob('*.json'))
    subprocess.run([str(ROOT / 'bench'), '--store', str(store), 'grade', '--submission', args.submission, '--task', TASK,
                    '--isolation', 'dev_idle'], cwd=ROOT, capture_output=True,
                   env={**os.environ, 'FRONTIERAM_RUNTIME': str(ROOT / '.runtime-recal')})
    new = [json.loads(p.read_text()) for p in set((store / 'jobs/done').glob('*.json')) - before]
    second = max((j for j in new if j.get('submission') == args.submission and j.get('kind') == 'candidate'),
                 key=lambda j: j.get('finished_utc') or '', default=None)
    if second is None:
        print(json.dumps({'event': 'retry_missing', 'submission': args.submission}))
        return
    composed, taken = compose([first_rows, rows_of(store, second['id'])])
    if not (args.out / 'score.first.json').exists():
        (args.out / 'score.json').rename(args.out / 'score.first.json')
    metrics = store / 'results' / second['id'] / 'grade/metrics.json'
    second['candidate'] = json.loads(metrics.read_text())['parts'].get('candidate') if metrics.exists() else {}
    if composed is not None:
        spec = importlib.util.spec_from_file_location('polarization_scoring', PACKAGE / 'scoring.py')
        scoring = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(scoring)
        aggregate = scoring.aggregate(composed, scoring.CANONICAL_CLASSIC, scoring.CANONICAL_ADVANCED)
        from arena.operator_tasks import score_100
        from arena.store import Store
        anchors = Store(store).active_baseline(TASK)['metrics']['parts']
        score = score_100(aggregate['E'], anchors['weak']['raw_error'], anchors['strong']['raw_error'])
        second['candidate'].update(status='ok', raw_error=aggregate['E'], cases_passed=len(composed), reason=None,
                                   max_wall_s=max(float(r['wall_s']) for r in composed),
                                   scene_statuses={f'{r["dataset"]}/{r["scene"]}': 'ok' for r in composed},
                                   scene_rmse={f'{r["dataset"]}/{r["scene"]}': float(r['mean_rmse']) for r in composed})
        second.update(status='ok', score=score)
    second['composed'] = {'base_job': first['id'], 'retry_job': second['id'], 'scenes_from_retry': taken,
                          'rule': 'a scene that timed out passes if either of two gradings ran it within 1 s; '
                                  'identical per-scene RMSEs required'} if composed is not None else {
        'base_job': first['id'], 'retry_job': second['id'], 'not_composed': taken}
    (args.out / 'score.json').write_text(json.dumps(second, indent=1) + '\n')
    print(json.dumps({'event': 'retried', 'submission': args.submission, 'status': second.get('status'),
                      'score': second.get('score'), 'composed': composed is not None}))


if __name__ == '__main__':
    main()
