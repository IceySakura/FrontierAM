"""Rescore the v6.3 polarization-demosaicking-v3 cells on the 116 scored scenes (2026-11-26, results/OPEN_ITEMS.md).

The other 16 formal scenes became the held-out public feedback (versions/v3/data/private/feedback-split.json). Outputs
are deterministic, so each cell is rescored from the per-scene results already in its score.json: valid when every
scored scene passed, E is their mean per-scene RMSE, and the score uses the active baseline, re-measured with the same
rule. The 132-scene record is kept as score.132.json. Cells without per-scene results (no build, no grading) are left
as they are.

Usage: polarization-rescore-116.py   (prints one line per cell)
"""

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
SPLIT = ROOT / 'tasks/polarization-demosaicking/versions/v3/data/private/feedback-split.json'
CELLS = ROOT / 'tasks/polarization-demosaicking/solutions/v6.3/v3'
BASELINES = ROOT / 'var/stores/store-v63/baselines/polarization-demosaicking-v3'


def main() -> None:
    import arena.tasks  # noqa: F401  (imported first: arena.operator_tasks alone hits the import cycle)
    from arena.operator_tasks import score_100
    scored = json.loads(SPLIT.read_text())['scored']
    baseline = json.loads((BASELINES / (BASELINES / 'ACTIVE').read_text().strip() / 'baseline.json').read_text())
    parts = baseline['metrics']['parts']
    if parts['weak'].get('expected_cases') != len(scored):
        raise SystemExit(f'the active baseline {baseline["id"]} is not on the {len(scored)} scored scenes')
    weak, strong = parts['weak']['raw_error'], parts['strong']['raw_error']
    for path in sorted(CELLS.glob('*/score.json')):
        old = json.loads((path.parent / 'score.132.json').read_text()) if (path.parent / 'score.132.json').exists() \
            else json.loads(path.read_text())
        candidate = old.get('candidate') or {}
        rmse, statuses = candidate.get('scene_rmse') or {}, candidate.get('scene_statuses') or {}
        if not statuses:
            print(json.dumps({'cell': path.parent.name, 'skipped': 'no per-scene results'}))
            continue
        failed = [s for s in scored if statuses.get(s) != 'ok' or s not in rmse]
        new = json.loads(json.dumps(old))
        if failed:
            error, score, status = None, 0.0, 'submission_failure'
            reason = f'{failed[0]}: {statuses.get(failed[0], "not graded")}'
        else:
            error = math.fsum(rmse[s] for s in scored) / len(scored)
            score, status, reason = score_100(error, weak, strong), 'ok', None
        new['candidate'].update(status=status, raw_error=error, reason=reason, expected_cases=len(scored),
                                cases_passed=sum(statuses.get(s) == 'ok' for s in scored), package_score=None,
                                scene_rmse={s: rmse[s] for s in scored if s in rmse},
                                scene_statuses={s: statuses[s] for s in scored if s in statuses})
        new.update(status=status, score=score, rule='scored-116', anchors_116={'weak': weak, 'strong': strong,
                                                                            'baseline': baseline['id']})
        if not (path.parent / 'score.132.json').exists():
            path.rename(path.parent / 'score.132.json')
        path.write_text(json.dumps(new, indent=1) + '\n')
        print(json.dumps({'cell': path.parent.name, 'old': old.get('score'), 'old_status': old.get('status'),
                          'new': round(score, 3), 'status': status, 'E': error, 'reason': reason}))


if __name__ == '__main__':
    main()
