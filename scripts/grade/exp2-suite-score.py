"""Score an exponential-v2 candidate suite against a panel suite (both sharded the same way over the grading slots).

Every case was measured for the whole panel and for the candidate on the same slot (bench suite pins shard i to slot
i). This merges the shards of each suite, builds the per-case virtual best of the panel, and reports the score
100 / (1 + R) of the v1 scorer, the geometric speed ratio with its hierarchical bootstrap interval, the winner per
case and every failure. Writes results to OUT/score.json with merged run reports beside it.

With --per-case (v6.3 board since 2026-11-26, user decision) a failed case no longer zeroes the submission: each
scored case gets 100 / (1 + r) when valid and 0 when not, averaged with the scorer's five-level equal weights (workloads
within mother, mothers within family, families within scenario, scenarios within precision, precisions); a failed
control still invalidates. A candidate as fast as the panel's per-case best still scores 50. The whole-submission
score of the v1 rule stays in score_complete.

Usage: exp2-suite-score.py --store STORE [--panel-store STORE] --panel PANEL_SUITE --candidate CANDIDATE_SUITE --out OUT
       [--per-case]
"""

import argparse
import json
import sys

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import arena.tasks  # noqa: E402,F401  (registers the tasks)
from arena.exponential_v2 import per_case_score, speed_ratio  # noqa: E402
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
    parser.add_argument('--store', type=Path, default=ROOT / 'var/stores/store', help='store holding the candidate suite')
    parser.add_argument('--panel-store', type=Path, help='store holding the panel suite (default: --store)')
    parser.add_argument('--panel', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--task', default='exponential-v2', help='exponential-v3 validates with the v6.3 limits')
    parser.add_argument('--extra-panel', action='append', default=[], metavar='STORE:SUITE',
                        help='another panel measurement (e.g. one added member) joining the virtual best')
    parser.add_argument('--per-case', action='store_true', help='per-case scores (see above) instead of all-or-nothing')
    parser.add_argument('--recheck', action='append', default=[], metavar='SUITE',
                        help='a quiet re-measurement of some candidate cases (scripts/grade/exp-cpu-recheck.py); '
                             'its aggregates replace those of the same cases')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    task = TASKS[args.task]
    cases = task.cases()
    panel = {'kind': 'panel', 'parts': merged_parts(args.panel_store or args.store, args.panel, args.out)}
    for extra in args.extra_panel:
        store, suite = extra.split(':', 1)
        panel['parts'].update(merged_parts(Path(store), suite, args.out))
    candidate = merged_parts(args.store, args.candidate, args.out)['candidate']
    rechecked = {}
    for suite in args.recheck:
        for case_id, aggregate in merged_parts(args.store, suite, args.out)['candidate']['aggregates'].items():
            candidate['aggregates'][case_id], rechecked[case_id] = aggregate, suite
    ok, reason = task.baseline_ok(panel)
    reference, winners, uncovered = task.virtual_best(panel)
    result = {'panel_suite': args.panel, 'candidate_suite': args.candidate, 'panel_ok': ok, 'panel_reason': reason,
              'uncovered_cases': uncovered,
              'member_valid_cases': {m: sum(a['valid'] for a in p['aggregates'].values())
                                     for m, p in panel['parts'].items()},
              **({'rechecked_cases': rechecked} if rechecked else {})}
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
        if args.per_case:
            result.update(rule='per-case', score_complete=scored['score'], controls_valid=scored['controls_valid'],
                          score=per_case_score(cases, scored['case_results']) if scored['controls_valid'] else 0.0,
                          status='ok' if scored['controls_valid'] else 'invalid_submission',
                          valid_scored_cases=sum(1 for c in cases if c['role'] == 'scored'
                                                 and scored['case_results'][c['id']]['valid']),
                          scored_cases=sum(1 for c in cases if c['role'] == 'scored'))
    (args.out / 'score.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('case_results', 'winners')}, indent=1))


if __name__ == '__main__':
    main()
