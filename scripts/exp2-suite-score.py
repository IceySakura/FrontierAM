"""Score an exponential-v2 candidate suite against a panel suite (both sharded the same way over the grading slots).

Every case was measured for the whole panel and for the candidate on the same slot (bench suite pins shard i to slot
i). This merges the shards of each suite, builds the per-case virtual best of the panel, and reports the score
100 / (1 + R) of the v1 scorer, the geometric speed ratio with its hierarchical bootstrap interval, the winner per
case and every failure. Writes results to OUT/score.json with merged run reports beside it.

Usage: exp2-suite-score.py --store STORE --panel PANEL_SUITE --candidate CANDIDATE_SUITE --out OUT
"""

import argparse
import json
import sys

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import arena.tasks  # noqa: E402,F401  (registers the tasks)
from arena.exponential_v2 import speed_ratio  # noqa: E402
from arena.tasks import TASKS  # noqa: E402


def merged_parts(store: Path, suite: str, out: Path) -> dict:
    record = json.loads((store / 'suites' / f'{suite}.json').read_text())
    parts: dict[str, dict] = {}
    for shard in record['shards']:
        metrics = json.loads((store / 'results' / shard['job'] / 'grade/metrics.json').read_text())
        for name, part in metrics['parts'].items():
            merged = parts.setdefault(name, {'status': 'ok', 'aggregates': {}, 'runs': [], 'reasons': []})
            if part.get('status') != 'ok':
                merged['status'] = part.get('status')
                merged['reasons'].append(part.get('reason'))
                continue
            merged['aggregates'].update(part['aggregates'])
            merged['runs'] += json.loads(Path(part['report']).read_text())['runs']
    for name, merged in parts.items():
        report = out / f'{suite}-{name}.json'
        report.write_text(json.dumps({'runs': merged.pop('runs'), 'aggregates': merged['aggregates']}) + '\n')
        merged['report'] = str(report)
    return parts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--store', type=Path, default=ROOT / 'store')
    parser.add_argument('--panel', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    task = TASKS['exponential-v2']
    cases = task.cases()
    panel = {'kind': 'panel', 'parts': merged_parts(args.store, args.panel, args.out)}
    candidate = merged_parts(args.store, args.candidate, args.out)['candidate']
    ok, reason = task.baseline_ok(panel)
    reference, winners, uncovered = task.virtual_best(panel)
    result = {'panel_suite': args.panel, 'candidate_suite': args.candidate, 'panel_ok': ok, 'panel_reason': reason,
              'uncovered_cases': uncovered,
              'member_valid_cases': {m: sum(a['valid'] for a in p['aggregates'].values())
                                     for m, p in panel['parts'].items()}}
    if candidate['status'] != 'ok':
        result.update(status='invalid_submission', score=0.0, reason=candidate['reasons'])
    elif uncovered:
        result.update(status='unmeasured', score=None, reason='panel does not cover every case')
    else:
        scored = task.scoring().score_submission(cases, candidate['aggregates'], reference)
        result.update(status=scored['status'], score=scored['score'], relative_time=scored['relative_time'],
                      failed_cases=scored['failed_cases'], winners=winners,
                      speed_ratio=speed_ratio(cases, candidate, panel) if scored['status'] == 'ok' else None,
                      case_results=scored['case_results'])
    (args.out / 'score.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('case_results', 'winners')}, indent=1))


if __name__ == '__main__':
    main()
