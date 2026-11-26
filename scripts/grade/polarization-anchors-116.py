"""polarization-demosaicking-v3 anchors after the feedback split (2026-11-26, results/OPEN_ITEMS.md), derived from the
qualified 132-scene reference measurement instead of a new one.

The references are deterministic, so their per-scene RMSEs do not change between measurements; only validity depends on
timing, and the strong reference sits at the 1.000 s scene limit (0.89 s in the qualified 11-16 run; a re-measure on
11-26 19:38 with development running timed out on qiu/ball and qiu/blackstuff at 1.0009 s, every other scene ok). From
the qualified run this writes (1) a formal baseline on the 116 scored scenes, made active, and (2) the public-feedback
baseline on the 16 held-out scenes (rows per dataset), keeping the Wen5 one as a backup.

Usage: polarization-anchors-116.py SOURCE_BASELINE_ID
"""

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TASK = 'polarization-demosaicking-v3'
SPLIT = ROOT / 'tasks/polarization-demosaicking/versions/v3/data/private/feedback-split.json'
BASELINES = ROOT / 'var/stores/store-v63/baselines' / TASK
PUBLIC = ROOT / 'var/stores/store-public-v63/public-baselines' / f'{TASK}.json'


def main() -> None:
    source_id = sys.argv[1]
    source = json.loads((BASELINES / source_id / 'baseline.json').read_text())
    if not source.get('qualified'):
        raise SystemExit(f'{source_id} is not a qualified baseline')
    split = json.loads(SPLIT.read_text())
    now = datetime.now(timezone.utc)
    derived = json.loads(json.dumps(source))
    derived.update(id=f'{TASK}-{now:%Y%m%dT%H%M%SZ}-scored116', created_utc=now.isoformat(timespec='seconds'),
                   derived_from=source_id, derivation='per-scene results of the qualified run restricted to the 116 scored '
                                                      'scenes (scripts/grade/polarization-anchors-116.py)')
    for which in ('weak', 'strong'):
        part = derived['metrics']['parts'][which]
        rmse, statuses = part['scene_rmse'], part['scene_statuses']
        if any(statuses.get(s) != 'ok' for s in split['scored']):
            raise SystemExit(f'{which} reference failed a scored scene in {source_id}')
        part.update(raw_error=math.fsum(rmse[s] for s in split['scored']) / len(split['scored']), package_score=None,
                    expected_cases=len(split['scored']), cases_passed=len(split['scored']),
                    scene_rmse={s: rmse[s] for s in split['scored']}, scene_statuses={s: 'ok' for s in split['scored']})
    weak, strong = (derived['metrics']['parts'][w]['raw_error'] for w in ('weak', 'strong'))
    derived['reason'] = f'both references valid; local anchors weak {weak:.6g} > strong {strong:.6g} (116 scored scenes)'
    target = BASELINES / derived['id']
    target.mkdir()
    (target / 'baseline.json').write_text(json.dumps(derived, indent=2) + '\n')
    (BASELINES / '.ACTIVE.tmp').write_text(derived['id'] + '\n')
    (BASELINES / '.ACTIVE.tmp').replace(BASELINES / 'ACTIVE')

    parts = source['metrics']['parts']
    rows = {which: [{'id': s, 'category': s.split('/', 1)[0], 'status': 'passed', 'accuracy': parts[which]['scene_rmse'][s],
                     'seconds': None} for s in split['feedback']] for which in ('weak', 'strong')}
    if PUBLIC.exists() and not PUBLIC.with_name(f'{TASK}.wen5-20261126.json').exists():
        PUBLIC.rename(PUBLIC.with_name(f'{TASK}.wen5-20261126.json'))
    PUBLIC.write_text(json.dumps({
        'task': TASK, 'job': source['job'], 'slot': None, 'created_utc': now.isoformat(timespec='seconds'), 'provisional': False,
        'derived_from': source_id, 'weak': rows['weak'], 'strong': rows['strong'],
        'weak_raw': sum(r['accuracy'] for r in rows['weak']) / len(rows['weak']),
        'strong_raw': sum(r['accuracy'] for r in rows['strong']) / len(rows['strong'])}, indent=1) + '\n')
    print(json.dumps({'active': derived['id'], 'weak_116': weak, 'strong_116': strong,
                      'feedback_weak': sum(r['accuracy'] for r in rows['weak']) / 16,
                      'feedback_strong': sum(r['accuracy'] for r in rows['strong']) / 16}))


if __name__ == '__main__':
    main()
