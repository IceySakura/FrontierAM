"""Protocol v6.2 public-set evaluation: grade a source tree on a task's public set and compare it with the baseline.

The public-feedback graders are the formal ones sealed over public data (scripts/prepare.py --conic-public-repeats,
FRONTIERAM_EXP2_SEAL with prepare_exponential_v2.py --repeats); they run in their own store, so nothing here touches
held-out data or formal results. `public-baseline` measures the reference once; `public-eval` grades one source tree
on all or part of the public set (sharded over the grading slots like a suite) and writes report.json / report.md.
"""

from __future__ import annotations

import json
import math
import shutil
import threading

from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .store import Store, read_json, utc, write_json
from .tasks import TASKS, tar_directory

PUBLIC_TASKS = ('conic', 'exponential-v2')


def _grade_suite(task, store: Store, runtime: dict, *, submission: str | None, case_ids: list[str] | None) -> dict:
    """Queue the cases as shards on the grading slots (same partition as full-set runs) and grade them now."""
    from .cli import cmd_suite, context, worker_loop

    layout, broker, _ = context()
    slots = ','.join(s['id'] for s in layout['slots'] if s['kind'] == 'grade')
    cases_file = None
    if case_ids:
        cases_file = store.root / 'public-eval-cases.json'
        cases_file.write_text(json.dumps(sorted(case_ids)))
    record = cmd_suite(Namespace(store=store.root, task=task.name, shards=len(slots.split(',')), slots=slots,
                                 isolation='shared', hold=False, cases=str(cases_file) if cases_file else None,
                                 reference=submission is None, submission=submission))
    jobs = [shard['job'] for shard in record['shards']]
    args = Namespace(task=[task.name], drain=True)
    stop = threading.Event()
    with ThreadPoolExecutor(len(jobs)) as pool:
        for future in [pool.submit(worker_loop, layout, broker, runtime, store, args, stop, [job]) for job in jobs]:
            future.result()
    return record


def _shards(store: Store, record: dict):
    for shard in record['shards']:
        result = store.result_dir(shard['job'])
        metrics = result / 'grade/metrics.json'
        yield shard['job'], result, read_json(metrics) if metrics.exists() else None


def conic_rows(store: Store, record: dict) -> tuple[dict, list[str]]:
    rows, missing = {}, []
    for job, result, metrics in _shards(store, record):
        if not metrics or metrics.get('success_rate') is None:
            missing.append(job)
            continue
        for case in metrics['cases']:
            reasons = []
            for path in sorted((result / 'grade/verifier/cases' / case['case_id']).glob('repeat-*/record.json')):
                run = read_json(path)
                reasons.append(run.get('reason'))
            elapsed = [t for t in case.get('elapsed_seconds') or [] if t is not None]
            rows[case['case_id']] = {
                'id': case['case_id'], 'group': case['group'], 'limit_s': case['time_limit_seconds'],
                'solved': bool(case['success']), 'status': case.get('status'),
                'seconds': max(elapsed) if elapsed else None,
                'reason': None if case['success'] else next((r for r in reasons if r not in (None, 'solved_verified')),
                                                             reasons[0] if reasons else None),
            }
    return rows, missing


def exp2_parts(store: Store, record: dict) -> tuple[dict, list[str]]:
    parts, missing = {}, []
    for job, _, metrics in _shards(store, record):
        if not metrics:
            missing.append(job)
            continue
        for name, part in metrics['parts'].items():
            merged = parts.setdefault(name, {'status': 'ok', 'aggregates': {}, 'reasons': []})
            if part.get('status') != 'ok':
                merged['status'] = part.get('status')
                merged['reasons'].append(part.get('reason'))
                continue
            merged['aggregates'].update(part['aggregates'])
    return parts, missing


def conic_score(p: float, p_b: float) -> float:
    """The conic mapping (reference 50, 0.9 x reference 0, all solved 100) anchored to the public reference rate."""
    if p_b <= 0:
        return 100.0 if p > 0 else 50.0
    if p <= 0.9 * p_b:
        return 0.0
    if p <= p_b:
        return 50 * (p - 0.9 * p_b) / (0.1 * p_b)
    return 100.0 if p_b >= 1 else 50 + 50 * (p - p_b) / (1 - p_b)


def baseline_path(store: Store, task: str) -> Path:
    return store.root / 'public-baselines' / f'{task}.json'


def cmd_public_baseline(args) -> None:
    from .runtime import load_runtime

    store, runtime, task = Store(args.store), load_runtime(), TASKS[args.task]
    if baseline_path(store, task.name).exists() and not args.replace:
        raise SystemExit(f'{task.name} already has a public baseline; pass --replace to measure another')
    if args.from_suite:  # re-summarise an already graded reference suite, e.g. after a scorer-side correction
        record = read_json(store.root / 'suites' / f'{args.from_suite}.json')
        if not record.get('reference'):
            raise SystemExit(f'{args.from_suite} is not a reference suite')
    else:
        record = _grade_suite(task, store, runtime, submission=None, case_ids=None)
    if task.name == 'conic':
        rows, missing = conic_rows(store, record)
        summary = {'cases': len(rows), 'solved': sum(r['solved'] for r in rows.values())}
    else:
        parts, missing = exp2_parts(store, record)
        _, winners, uncovered = task.virtual_best({'parts': parts})
        summary = {'cases': len(task.cases()), 'uncovered': uncovered,
                   'wins': {m: list(winners.values()).count(m) for m in parts}}
        if uncovered:
            missing.append(f'uncovered:{len(uncovered)}')
    baseline = {'task': task.name, 'suite': record['id'], 'created_utc': utc(), 'complete': not missing,
                'missing': missing, 'runtime_lock_sha256': runtime['lock_sha256'], **summary}
    write_json(baseline_path(store, task.name), baseline)
    print(json.dumps(baseline, indent=1))


def _select(task, runtime, groups: list[str] | None, cases: list[str] | None) -> list[dict]:
    if task.name == 'conic':
        universe = [{'id': c['id'], 'group': c['group']} for c in task.suite(runtime)]
    else:
        universe = [{'id': c['id'], 'group': c['family']} for c in task.cases()]
    chosen = [c for c in universe if (not groups or c['group'] in groups) and (not cases or c['id'] in cases)]
    unknown = set(cases or ()) - {c['id'] for c in universe}
    if unknown or not chosen:
        raise SystemExit(f'no such public cases or groups: {sorted(unknown)[:5] or groups}')
    return chosen


def cmd_public_eval(args) -> None:
    from .runtime import load_runtime

    store, runtime, task = Store(args.store), load_runtime(), TASKS[args.task]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    baseline = read_json(baseline_path(store, task.name))
    if not baseline['complete']:
        raise SystemExit('the public baseline is incomplete')
    selection = _select(task, runtime, args.groups.split(',') if args.groups else None,
                        args.cases.split(',') if args.cases else None)
    ids = [c['id'] for c in selection]
    submission_id, directory = store.new_submission(task.name, {'origin': 'public-eval', 'label': args.label})
    tar_directory(Path(args.source).resolve(), directory / 'incoming.tar')
    stage = directory / 'stage'
    try:
        task.stage(runtime, directory / 'incoming.tar', stage)
    except Exception as error:  # the formal staging rejects the archive: the same verdict a formal submission gets
        report = {'task': task.name, 'status': 'invalid_submission', 'reason': str(error)[:500], 'cases': []}
        write_json(out / 'report.json', report)
        (out / 'report.md').write_text(f'# Public evaluation\n\nInvalid submission: {report["reason"]}\n')
        print(json.dumps({k: v for k, v in report.items() if k != 'cases'}))
        return
    shutil.move(stage / 'source.tar', directory / 'source.tar')
    shutil.move(stage / 'source_manifest.json', directory / 'source_manifest.json')
    shutil.rmtree(stage)
    (directory / 'incoming.tar').unlink()
    store.update_submission(submission_id, status='frozen')
    record = _grade_suite(task, store, runtime, submission=submission_id,
                          case_ids=None if len(ids) == len(_select(task, runtime, None, None)) else ids)
    reference = read_json(store.root / 'suites' / f'{baseline["suite"]}.json')
    report = (conic_report if task.name == 'conic' else exp2_report)(task, store, record, reference, ids)
    report.update(task=task.name, suite=record['id'], submission=submission_id, created_utc=utc(),
                  subset=len(ids) != len(_select(task, runtime, None, None)))
    write_json(out / 'report.json', report)
    (out / 'report.md').write_text(render(report))
    print(json.dumps({k: v for k, v in report.items() if k not in ('cases', 'groups')}))


def conic_report(task, store, record, reference, ids) -> dict:
    rows, missing = conic_rows(store, record)
    base, _ = conic_rows(store, reference)
    if missing or set(rows) != set(ids):
        return {'status': 'unmeasured', 'reason': f'grading incomplete: {missing}', 'cases': list(rows.values())}
    groups: dict[str, dict] = {}
    for case_id in ids:
        row, ref = rows[case_id], base[case_id]
        row.update(baseline_solved=ref['solved'], baseline_seconds=ref['seconds'])
        group = groups.setdefault(row['group'], {'cases': 0, 'solved': 0, 'baseline_solved': 0})
        group['cases'] += 1
        group['solved'] += row['solved']
        group['baseline_solved'] += ref['solved']
    p = sum(rows[i]['solved'] for i in ids) / len(ids)
    p_b = sum(base[i]['solved'] for i in ids) / len(ids)
    return {'status': 'ok', 'success_rate': p, 'baseline_success_rate': p_b, 'score': conic_score(p, p_b),
            'repeats': 1, 'groups': groups, 'cases': [rows[i] for i in ids],
            'note': 'One repeat per case; the formal evaluation runs three and needs all three to pass.'}


def exp2_report(task, store, record, reference, ids) -> dict:
    parts, missing = exp2_parts(store, record)
    panel, _ = exp2_parts(store, reference)
    candidate = parts.get('candidate')
    if missing or candidate is None:
        return {'status': 'unmeasured', 'reason': f'grading incomplete: {missing}', 'cases': []}
    cases = [c for c in task.cases() if c['id'] in set(ids)]
    if candidate['status'] != 'ok':
        return {'status': 'invalid_submission', 'score': 0.0, 'reason': candidate['reasons'], 'cases': []}
    best, winners, uncovered = task.virtual_best({'parts': panel})
    scoring = task.scoring()
    if any(c['role'] == 'scored' for c in cases):
        result = scoring.score_submission(cases, candidate['aggregates'], {c['id']: best[c['id']] for c in cases})
    else:  # controls only: the scorer needs a scored session for R, so report per-case verdicts without a score
        verdicts = {}
        for c in cases:
            record = candidate['aggregates'].get(c['id'])
            verdict = dict(scoring.validate_measurement(record, scenario=c['scenario']))
            if verdict['valid']:
                verdict.update(time_seconds=record['time_seconds'],
                               relative_time=record['time_seconds'] / best[c['id']]['time_seconds'])
            verdict['reference_time_seconds'] = best[c['id']]['time_seconds']
            verdicts[c['id']] = verdict
        failed = {k: v.get('reason') for k, v in verdicts.items() if not v['valid']}
        result = {'status': 'ok' if not failed else 'invalid_submission', 'score': None, 'relative_time': None,
                  'failed_cases': failed, 'case_results': verdicts}
    rows, groups = [], {}
    for case in cases:
        verdict = result['case_results'][case['id']]
        row = {'id': case['id'], 'family': case['family'], 'mother': case['mother'], 'scenario': case['scenario'],
               'precision': case['precision'], 'role': case['role'], 'valid': verdict['valid'],
               'reason': None if verdict['valid'] else verdict.get('reason'),
               'seconds': verdict.get('time_seconds'), 'baseline_seconds': verdict.get('reference_time_seconds'),
               'baseline_method': winners.get(case['id']), 'relative_time': verdict.get('relative_time')}
        rows.append(row)
        group = groups.setdefault(case['family'], {'cases': 0, 'valid': 0, 'log_ratios': []})
        group['cases'] += 1
        group['valid'] += row['valid']
        if row['relative_time']:
            group['log_ratios'].append(math.log(row['relative_time']))
    for group in groups.values():
        ratios = group.pop('log_ratios')
        group['geometric_relative_time'] = math.exp(sum(ratios) / len(ratios)) if ratios else None
    return {'status': result['status'], 'score': result['score'], 'relative_time': result['relative_time'],
            'failed_cases': result['failed_cases'], 'repeats': 1, 'groups': groups, 'cases': rows,
            'note': 'One timed repeat per case (the formal evaluation takes the median of three). relative_time is '
                    'your time over the fastest published panel method on that case.'}


def render(report: dict) -> str:
    lines = [f'# Public evaluation ({report["task"]})', '']
    if report['status'] not in ('ok', 'invalid_submission'):
        return '\n'.join(lines + [f'Not measured: {report.get("reason")}', ''])
    score = report.get('score')
    lines += [f'- status: **{report["status"]}**', f'- public-set score (100-point scale, baseline = 50): **{score:.1f}**'
              if score is not None else '- public-set score: none',
              f'- cases evaluated: {len(report["cases"])}' + (' (subset)' if report.get('subset') else ''),
              f'- note: {report.get("note", "")}', '']
    if report['task'] == 'conic':
        lines += [f'Solved {report["success_rate"]:.3f} of the evaluated cases; the baseline solved '
                  f'{report["baseline_success_rate"]:.3f} of the same cases.', '',
                  '| group | cases | you solved | baseline solved |', '| --- | --- | --- | --- |']
        lines += [f'| {g} | {v["cases"]} | {v["solved"]} | {v["baseline_solved"]} |' for g, v in sorted(report['groups'].items())]
        lines += ['', '| case | group | limit s | you | your max s | reason | baseline | baseline max s |',
                  '| --- | --- | --- | --- | --- | --- | --- | --- |']
        for r in report['cases']:
            lines.append(f'| {r["id"]} | {r["group"]} | {r["limit_s"]} | {"solved" if r["solved"] else "failed"} | '
                         f'{_num(r["seconds"])} | {r["reason"] or ""} | {"solved" if r["baseline_solved"] else "failed"} | '
                         f'{_num(r["baseline_seconds"])} |')
    else:
        if report.get('failed_cases'):
            lines += ['Failed cases (any failure makes the formal submission score 0):', '']
            lines += [f'- {k}: {v}' for k, v in sorted(report['failed_cases'].items())] + ['']
        lines += ['| family | cases | valid | geometric mean of your time / fastest panel time |', '| --- | --- | --- | --- |']
        lines += [f'| {g} | {v["cases"]} | {v["valid"]} | {_num(v["geometric_relative_time"])} |'
                  for g, v in sorted(report['groups'].items())]
        lines += ['', '| case | role | valid | reason | your s | fastest panel s | panel method | ratio |',
                  '| --- | --- | --- | --- | --- | --- | --- | --- |']
        for r in report['cases']:
            lines.append(f'| {r["id"]} | {r["role"]} | {r["valid"]} | {r["reason"] or ""} | {_num(r["seconds"])} | '
                         f'{_num(r["baseline_seconds"])} | {r["baseline_method"]} | {_num(r["relative_time"])} |')
    return '\n'.join(lines) + '\n'


def _num(value) -> str:
    return '' if value is None else f'{value:.3g}'
