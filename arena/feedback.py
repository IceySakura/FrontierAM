"""Protocol v6.3 test feedback: one report format for every task (results/PROTOCOL_V63.md).

A task adapter (arena/public_v63.py) turns its grader's per-case output into rows

    {'id', 'category', 'status': passed | wrong | timeout | crash, 'accuracy': float | None, 'seconds': float | None}

for the submission and for the reference on the same public cases. The report gives the public-set score (the task's
final mapping, anchored on the reference measured on the public set) and, for tasks whose cases fall into categories,
one row per category: case count, status counts, accuracy and speed only as ratios to the reference, and the
category's score. Per-case results are never returned to the agent.
"""

from __future__ import annotations

import math

from collections import Counter
from typing import Callable

STATUSES = ('passed', 'wrong', 'timeout', 'crash')
CRASH = {'crash', 'nonzero_exit', 'oom', 'process_limit', 'entry_error', 'invalid_source', 'harness_error',
         'build_failed', 'build_error'}


def status_of(raw: str | None, *, correct: bool = True) -> str:
    """Map a grader's per-case status onto the four feedback statuses."""
    if raw in (None, '') or raw in CRASH:
        return 'crash'
    if 'timeout' in raw or raw in ('time_limit', 'cpu_limit'):
        return 'timeout'
    if raw == 'ok':
        return 'passed' if correct else 'wrong'
    return 'wrong'


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _geomean(values: list[float]) -> float | None:
    values = [v for v in values if v and v > 0 and math.isfinite(v)]
    return math.exp(sum(map(math.log, values)) / len(values)) if values else None


def compare(rows: list[dict], reference: dict[str, dict], timed: bool) -> dict:
    """Status counts plus accuracy and speed ratios (submission / reference) over one set of cases."""
    counts = Counter(r['status'] for r in rows)
    mine = [r['accuracy'] for r in rows if r['accuracy'] is not None]
    theirs = [reference[r['id']]['accuracy'] for r in rows
              if r['accuracy'] is not None and reference.get(r['id'], {}).get('accuracy') is not None]
    accuracy = None
    if mine and theirs and len(mine) == len(theirs) and _mean(theirs):
        accuracy = _mean(mine) / _mean(theirs)
    speed = None
    if timed:
        speed = _geomean([r['seconds'] / reference[r['id']]['seconds'] for r in rows
                          if r['status'] == 'passed' and r.get('seconds')
                          and reference.get(r['id'], {}).get('status') == 'passed' and reference[r['id']].get('seconds')])
    return {'cases': len(rows), **{s: counts.get(s, 0) for s in STATUSES}, 'accuracy_ratio': accuracy, 'speed_ratio': speed}


def build(task: str, rows: list[dict], reference: list[dict], *, score: float | None, categorized: bool,
          accuracy_name: str | None, lower_is_better: bool = True, timed: bool = True,
          category_score: Callable[[list[dict], list[dict]], float | None] | None = None,
          not_evaluated: int = 0, note: str = '') -> dict:
    ref = {r['id']: r for r in reference}
    report = {'task': task, 'status': 'ok', 'score': score, 'accuracy': accuracy_name, 'lower_is_better': lower_is_better,
              'timed': timed, 'total': compare(rows, ref, timed), 'categories': {}, 'not_evaluated': not_evaluated,
              'note': note}
    if categorized:
        for name in sorted({r['category'] for r in rows}):
            subset = [r for r in rows if r['category'] == name]
            report['categories'][name] = {**compare(subset, ref, timed),
                                          'score': category_score(subset, [ref[r['id']] for r in subset if r['id'] in ref])
                                          if category_score else None}
    return report


def invalid(task: str, reason: str) -> dict:
    return {'task': task, 'status': 'invalid_submission', 'score': 0.0, 'reason': reason[:500], 'categories': {}}


def unmeasured(task: str, reason: str) -> dict:
    return {'task': task, 'status': 'unmeasured', 'score': None, 'reason': reason[:500], 'categories': {}}


def _ratio(value: float | None) -> str:
    return '—' if value is None else f'{value:.3g}'


def _score(value: float | None) -> str:
    return '—' if value is None else f'{value:.1f}'


def render(report: dict) -> str:
    lines = [f'# Test feedback ({report["task"]})', '']
    if report['status'] == 'invalid_submission':
        return '\n'.join(lines + [f'Invalid submission (public-set score 0): {report.get("reason")}', ''])
    if report['status'] != 'ok':
        return '\n'.join(lines + [f'Not measured (evaluator side, not a verdict on your submission): {report.get("reason")}', ''])
    total = report['total']
    lines += [f'Public-set score: **{_score(report["score"])}** / 100 (reference = 50, same mapping as the final score).', '']
    accuracy = (f'{report["accuracy"]}: you / reference ({"lower" if report["lower_is_better"] else "higher"} is better)'
                if report['accuracy'] else None)
    speed = 'time: you / reference (lower is faster)' if report['timed'] else None
    head = ['category', 'cases', 'passed / wrong / timeout / crash'] + ([accuracy] if accuracy else []) \
        + ([speed] if speed else []) + ['score']

    def row(name: str, part: dict, score: float | None) -> str:
        cells = [name, str(part['cases']), ' / '.join(str(part[s]) for s in STATUSES)]
        cells += [_ratio(part['accuracy_ratio'])] if accuracy else []
        cells += [_ratio(part['speed_ratio'])] if speed else []
        return '| ' + ' | '.join(cells + [_score(score)]) + ' |'

    lines += ['| ' + ' | '.join(head) + ' |', '|' + ' --- |' * len(head)]
    lines += [row(name, part, part['score']) for name, part in report['categories'].items()]
    lines.append(row('**total**', total, report['score']))
    if report.get('not_evaluated'):
        lines += ['', f'{report["not_evaluated"]} cases were not evaluated before the time cap; the figures cover the '
                      f'evaluated cases only.']
    if report.get('note'):
        lines += ['', report['note']]
    if report.get('timed'):
        lines += ['', 'Times are measured while other attempts may be developing on the same machine; the final evaluation '
                      'is timed on an idle machine, so treat the time ratios here as approximate.']
    return '\n'.join(lines) + '\n'
