"""Protocol v6.3 public-set evaluation: grade one source tree on a task's public set on one grade slot.

Every test of an iterative attempt runs as one priority job on one grade slot (the first free one; tests go ahead of
held-out grading, arena/slots.py), in the task's public store, through the same staging and grader as formal grading. The task adapter below turns the
grader's per-case output into feedback rows (arena/feedback.py); the reference rows come from the public baseline
measured once per task (`bench public-baseline-v63`), and the public score uses the task's final mapping anchored on
it. Nothing here reads held-out data.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import threading
import time

from argparse import Namespace
from pathlib import Path

from . import feedback
from .operator_tasks import score_100
from .public_eval import _harvest_conic, conic_rows, conic_score, exp2_parts
from .store import Store, read_json, utc, write_json
from .tasks import TASKS, tar_directory

ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / 'var/stores/store-public-v63'
# Public data switches (the graders read them at grading time) and the store each task's public grading uses.
PUBLIC = {
    'conic': {'family': 'conic', 'store': ROOT / 'var/stores/store-public-conic',
              'env': {'FRONTIERAM_RUNTIME': str(ROOT / '.runtime-conic-public')}},
    'exponential-v2': {'family': 'exponential', 'store': ROOT / 'var/stores/store-public-exp2',
                       'env': {'FRONTIERAM_EXP2_SEAL': str(ROOT / '.runtime-exp2-public')}},
    # Same public cases (a subset) and panel as v2, so the v2 public panel measurement is the reference.
    'exponential-v3': {'family': 'exponential', 'store': ROOT / 'var/stores/store-public-exp2', 'baseline_task': 'exponential-v2',
                       'env': {'FRONTIERAM_EXP3_SEAL': str(ROOT / '.runtime-exp3-public')}},
    'galaxy-gravity-v2': {'family': 'galaxy-gravity', 'store': STORE, 'env': {'FRONTIERAM_GALAXY_DATA': 'public'}},
    'galaxy-gravity-v3': {'family': 'galaxy-gravity', 'store': STORE, 'env': {'FRONTIERAM_GALAXY_DATA': 'public'}},
    'polarization-demosaicking': {'family': 'polarization-demosaicking', 'store': STORE,
                                  'env': {'FRONTIERAM_POLAR_DATA': 'wen'}},
    'polarization-demosaicking-v2': {'family': 'polarization-demosaicking', 'store': STORE,
                                     'env': {'FRONTIERAM_POLAR_DATA': 'wen'}},
    'species-tree-v2': {'family': 'species-tree', 'store': STORE, 'env': {'FRONTIERAM_SPECIES_V2_DATA': 'public'}},
    'species-tree-v3': {'family': 'species-tree', 'store': STORE, 'env': {'FRONTIERAM_SPECIES_V2_DATA': 'public'}},
    'trajectory-covariance': {'family': 'trajectory-covariance', 'store': STORE,
                              'env': {'FRONTIERAM_TRAJ_POOL': 'feedback'}},
    'trajectory-covariance-v2': {'family': 'trajectory-covariance', 'store': STORE,
                                 'env': {'FRONTIERAM_TRAJ_POOL': 'feedback'}},
}
SUITE_FAMILIES = ('conic', 'exponential')


def family(task: str) -> str:
    return PUBLIC[task]['family']


def config(task: str) -> dict:
    """The task's feedback declaration in tasks/<family>/meta.json (v63_data.feedback)."""
    meta = read_json(ROOT / 'tasks' / PUBLIC[task]['family'] / 'meta.json')
    return meta['v63_data']['feedback']


def baseline_path(task: str) -> Path:
    return STORE / 'public-baselines' / f'{task}.json'


# Grading on one slot -------------------------------------------------------------------------------------------
def _abort(store: Store, job: str) -> None:
    """Stop a job past the cap: its grading scope and every container on its slot (leased by this job alone)."""
    subprocess.run(['systemctl', 'stop', f'frontieram-grade-{job}.scope'], capture_output=True, timeout=120)
    spec = store.result_dir(job) / 'spec.json'
    if spec.exists():
        host = read_json(spec)['placement']['docker_host']
        names = subprocess.run(['docker', '--host', host, 'ps', '-aq'], capture_output=True, text=True, timeout=60).stdout.split()
        if names:
            subprocess.run(['docker', '--host', host, 'rm', '-f', *names], capture_output=True, timeout=120)
    for state in ('pending', 'running'):
        path = store.root / 'jobs' / state / f'{job}.json'
        if path.exists():
            data = read_json(path)
            data['aborted'] = 'v6.3 test evaluation reached its cap'
            write_json(store.root / 'jobs/failed' / f'{job}.json', data)
            path.unlink()


def _run(store: Store, runtime: dict, jobs: list[str], cap: float | None) -> list[str]:
    """Grade the queued jobs now; returns the jobs still unfinished at the cap (seconds)."""
    from .cli import context, worker_loop

    layout, broker, _ = context()
    stop = threading.Event()
    threads = {job: threading.Thread(target=worker_loop, args=(layout, broker, runtime, store, Namespace(task=None, drain=True),
                                                               stop, [job]), daemon=True) for job in jobs}
    for thread in threads.values():
        thread.start()
    deadline = None  # the cap counts from the moment grading holds its slot, not from the wait in the queue
    while any(t.is_alive() for t in threads.values()) and (deadline is None or time.monotonic() < deadline):
        if deadline is None and cap is not None and any((store.result_dir(j) / 'spec.json').exists() for j in jobs):
            deadline = time.monotonic() + cap
        time.sleep(2)
    stop.set()
    return [job for job, thread in threads.items() if thread.is_alive()]


def queue(task, store: Store, runtime: dict, submission: str | None, slot: str | None) -> dict:
    """One priority job on one grade slot: a one-shard suite for the suite tasks, a plain job for operator tasks."""
    from .cli import cmd_suite, freeze_reference

    if family(task.name) in SUITE_FAMILIES:
        return cmd_suite(Namespace(store=store.root, task=task.name, shards=1, slots=slot, isolation='shared', hold=False,
                                   cases=None, reference=submission is None, submission=submission, priority=True))
    kind = 'candidate' if submission else 'baseline'
    submission = submission or freeze_reference(store, runtime, task, kind='baseline')
    job = store.enqueue(submission, kind=kind, isolation='shared', note='v6.3 public', extra={'slot': slot, 'priority': True})
    return {'id': job, 'task': task.name, 'submission': submission, 'shards': [{'job': job}]}


# Adapters: grader output -> feedback rows ----------------------------------------------------------------------
def _conic(store: Store, record: dict, partial: dict) -> list[dict]:
    rows, _ = conic_rows(store, record)
    for found in partial.values():
        for case_id, run in found.items():
            rows[case_id] = {'id': case_id, 'group': run.get('group'), 'solved': run.get('outcome') == 'accepted',
                             'status': run.get('status'), 'seconds': run.get('elapsed_seconds'), 'reason': run.get('reason')}
    out = []
    for r in rows.values():
        reason = str(r.get('reason') or r.get('status') or '')
        status = 'passed' if r['solved'] else 'timeout' if 'timeout' in reason or 'time_limit' in reason else \
            'crash' if any(k in reason for k in ('crash', 'nonzero', 'exit', 'oom', 'signal')) else 'wrong'
        out.append({'id': r['id'], 'category': r['group'], 'status': status, 'accuracy': 1.0 if r['solved'] else 0.0,
                    'seconds': r.get('seconds')})
    return out


def _exp2(task, store: Store, record: dict, partial: dict) -> tuple[list[dict], dict]:
    parts, _ = exp2_parts(store, record)
    candidate = parts.setdefault('candidate', {'status': 'ok', 'aggregates': {}, 'reasons': []})
    for found in partial.values():
        candidate['aggregates'].update(found)
    scoring = task.scoring()
    rows = []
    for case in task.cases():
        measured = candidate['aggregates'].get(case['id'])
        if measured is None:
            continue
        verdict = dict(scoring.validate_measurement(measured, scenario=case['scenario']))
        reason = str(verdict.get('reason') or '')
        status = 'passed' if verdict['valid'] else 'timeout' if 'time' in reason and 'limit' in reason else \
            'crash' if any(k in reason for k in ('crash', 'exit', 'protocol', 'memory', 'non_ok', 'reported')) else 'wrong'
        rows.append({'id': case['id'], 'category': case['family'], 'status': status, 'accuracy': None,
                     'seconds': measured.get('time_seconds') if verdict['valid'] else None})
    return rows, candidate


def _galaxy(out: Path) -> list[dict]:
    summary = read_json(out / 'summary.json')
    cases: dict[str, dict] = {}
    for rec in summary.get('records', []):
        row = cases.setdefault(rec['case'], {'id': rec['case'], 'category': rec['family'], 'errors': [], 'seconds': 0.0,
                                             'status': 'passed'})
        row['seconds'] = max(row['seconds'], rec.get('cpu_s') or 0.0)
        status = feedback.status_of(rec['status'])
        if status != 'passed' and row['status'] == 'passed':
            row['status'] = status
        if rec.get('error') is not None:
            row['errors'].append(float(rec['error']))
    return [{'id': r['id'], 'category': r['category'], 'status': r['status'],
             'accuracy': sum(r['errors']) / len(r['errors']) if r['status'] == 'passed' and r['errors'] else None,
             'seconds': r['seconds']} for r in cases.values()]


def _polarization(out: Path) -> list[dict]:
    summary = read_json(out / 'summary.json')
    rows = []
    for r in summary.get('scene_rows', []):
        status = 'wrong' if r['status'] == 'submission_failure' and 'timeout' not in str(r.get('reason')) else \
            feedback.status_of(r['status'])
        rows.append({'id': r['scene'], 'category': 'all', 'status': status,
                     'accuracy': r.get('mean_rmse') if status == 'passed' else None, 'seconds': r.get('wall_s')})
    return rows


def _species(out: Path) -> list[dict]:
    summary = read_json(out / 'summary.json')
    return [{'id': r['case'], 'category': 'all', 'status': feedback.status_of(r['status']),
             'accuracy': r.get('normalized_rf', 1.0), 'seconds': r.get('wall_s')} for r in summary.get('records', [])]


def _trajectory(task, out: Path) -> list[dict]:
    receipt = out / 'verifier/receipt.json'
    records = {r['id']: r for r in read_json(receipt).get('cases', [])} if receipt.exists() else {}
    manifest = task.root() / 'results/mcmc-long-run-covariance/formal-v1/private/manifest.json'
    cases = {c['id']: c for c in read_json(manifest)['cases']}
    rows = []
    for case_id, rec in records.items():
        case = cases[case_id]
        status = feedback.status_of(rec.get('status'))
        loss = rec.get('loss') if status == 'passed' and rec.get('loss') is not None else math.sqrt(case['d'])
        rows.append({'id': case_id, 'category': case['model'], 'length': case['length'], 'status': status,
                     'accuracy': loss, 'seconds': rec.get('wall_s')})
    return rows


def _rows(task, store: Store, record: dict, partial: dict, which: str = 'candidate') -> list[dict]:
    if family(task.name) == 'conic':
        return _conic(store, record, partial)
    out = store.result_dir(record['shards'][0]['job']) / 'grade' / which
    if not (out / 'summary.json').exists():
        return []
    return {'galaxy-gravity': _galaxy, 'polarization-demosaicking': _polarization,
            'species-tree': _species}.get(family(task.name), lambda o: _trajectory(task, o))(out)


# Raw metrics and the final mapping -----------------------------------------------------------------------------
def raw_error(task: str, rows: list[dict]) -> float | None:
    """The task's own aggregate over the given rows; None if a failure makes the formal submission invalid."""
    if task.startswith(('galaxy-gravity', 'polarization-demosaicking')) and any(r['status'] != 'passed' for r in rows):
        return None
    if task.startswith('galaxy-gravity'):
        families: dict[str, list[float]] = {}
        for r in rows:
            families.setdefault(r['category'], []).append(r['accuracy'])
        return sum(sum(v) / len(v) for v in families.values()) / len(families)
    if task == 'trajectory-covariance':
        cells: dict[tuple, list[float]] = {}
        for r in rows:
            cells.setdefault((r['category'], r['length']), []).append(r['accuracy'])
        return sum(sum(v) / len(v) for v in cells.values()) / len(cells)
    return sum(r['accuracy'] for r in rows) / len(rows)


def galaxy_score(raw: float, weak: float, strong: float) -> float:
    """galaxy-gravity v3 log mapping: weak reference 0, strong reference 50, 1e-15 (double precision) 100."""
    if raw >= weak:
        return 0.0
    if raw >= strong:
        return 50 * math.log(weak / raw) / math.log(weak / strong)
    return min(100.0, 50 + 50 * math.log(strong / raw) / math.log(strong / 1e-15))


def mapped(task: str, rows: list[dict], weak: list[dict], strong: list[dict]) -> float | None:
    if not rows:
        return None
    raw, low, high = raw_error(task, rows), raw_error(task, weak), raw_error(task, strong)
    if raw is None:
        return 0.0
    if low is None or high is None:
        return None
    return (galaxy_score if task.startswith('galaxy-gravity') else score_100)(raw, low, high)


# Public baselines ------------------------------------------------------------------------------------------------
def cmd_public_baseline_v63(args) -> None:
    """Measure the task's references on its public set on one grade slot; the rows anchor every test's feedback."""
    from .runtime import load_runtime

    task, runtime = TASKS[args.task], load_runtime()
    os.environ.update(PUBLIC[task.name]['env'])
    if baseline_path(task.name).exists() and not args.replace:
        raise SystemExit(f'{task.name} already has a v6.3 public baseline; pass --replace to measure another')
    store = Store(PUBLIC[task.name]['store'])
    if family(task.name) in SUITE_FAMILIES:  # measured for v6.2 in the task's public store (arena/public_eval.py)
        old = read_json(store.root / 'public-baselines' / f'{PUBLIC[task.name].get("baseline_task", task.name)}.json')
        baseline = {'task': task.name, 'suite': old['suite'], 'source': 'v6.2 public baseline', 'created_utc': utc()}
    else:
        record = queue(task, store, runtime, None, args.slot)
        unfinished = _run(store, runtime, [record['shards'][0]['job']], None)
        rows = {which: _rows(task, store, record, {}, which) for which in ('weak', 'strong')}
        if unfinished or not all(rows.values()):
            raise SystemExit(f'{task.name}: public baseline incomplete (job {record["shards"][0]["job"]})')
        baseline = {'task': task.name, 'job': record['shards'][0]['job'], 'slot': args.slot, 'created_utc': utc(),
                    'provisional': bool(args.provisional), 'weak': rows['weak'], 'strong': rows['strong'],
                    'weak_raw': raw_error(task.name, rows['weak']), 'strong_raw': raw_error(task.name, rows['strong'])}
    write_json(baseline_path(task.name), baseline)
    print(json.dumps({k: v for k, v in baseline.items() if k not in ('weak', 'strong')}))


def _reference(task, store: Store) -> dict:
    """Reference rows (and, for the suite tasks, what their scoring needs) from the task's public baseline."""
    baseline = read_json(baseline_path(task.name))
    if family(task.name) == 'conic':
        record = read_json(store.root / 'suites' / f'{baseline["suite"]}.json')
        return {'rows': _conic(store, record, {})}
    if family(task.name) == 'exponential':
        record = read_json(store.root / 'suites' / f'{baseline["suite"]}.json')
        panel, _ = exp2_parts(store, record)
        best, _, _ = task.virtual_best({'parts': panel})
        rows = [{'id': c['id'], 'category': c['family'], 'status': 'passed', 'accuracy': None,
                 'seconds': best[c['id']]['time_seconds']} for c in task.cases() if c['id'] in best]
        return {'rows': rows, 'best': best}
    return {'rows': baseline['strong'], 'weak': baseline['weak']}


# One test --------------------------------------------------------------------------------------------------------
def evaluate(task, store: Store, runtime: dict, source: Path, slot: str, cap: float | None, label: str) -> dict:
    cfg = config(task.name)
    submission, directory = store.new_submission(task.name, {'origin': 'v6.3 test', 'label': label})
    tar_directory(source.resolve(), directory / 'incoming.tar')
    stage = directory / 'stage'
    try:
        task.stage(runtime, directory / 'incoming.tar', stage)
    except Exception as error:  # the formal staging rejects it: the verdict a formal submission would get
        return feedback.invalid(task.name, str(error))
    shutil.move(stage / 'source.tar', directory / 'source.tar')
    shutil.move(stage / 'source_manifest.json', directory / 'source_manifest.json')
    shutil.rmtree(stage)
    (directory / 'incoming.tar').unlink()
    store.update_submission(submission, status='frozen')
    record = queue(task, store, runtime, submission, slot)
    job = record['shards'][0]['job']
    unfinished = _run(store, runtime, [job], cap)
    partial = {}
    if unfinished:
        if family(task.name) == 'conic':
            partial[job] = _harvest_conic(store, job)
        elif family(task.name) == 'exponential':
            report = store.result_dir(job) / 'grade/evaluation.json'
            partial[job] = read_json(report).get('aggregates', {}) if report.exists() else {}
        _abort(store, job)
    reference = _reference(task, store)
    if family(task.name) == 'exponential':
        rows, candidate = _exp2(task, store, record, partial)
    else:
        rows, candidate = _rows(task, store, record, partial), None
    if not rows:
        result = store.result_dir(job) / 'grade' / 'candidate' / 'summary.json'
        reason = read_json(result).get('reason') if result.exists() else None
        if result.exists() and read_json(result).get('status') == 'submission_failure':
            return feedback.invalid(task.name, str(reason or 'the submission failed before any case ran'))
        return feedback.unmeasured(task.name, 'no case finished within the cap' if unfinished else str(reason or 'grading failed'))
    ids = {r['id'] for r in rows}
    ref_rows = [r for r in reference['rows'] if r['id'] in ids]
    total_cases = len(reference['rows'])

    if family(task.name) == 'conic':
        def score_of(mine, theirs):
            return conic_score(sum(r['accuracy'] for r in mine) / len(mine), sum(r['accuracy'] for r in theirs) / len(theirs)) \
                if mine and theirs else None
    elif family(task.name) == 'exponential':
        def score_of(mine, theirs):
            if not mine:
                return None
            if any(r['status'] != 'passed' for r in mine):
                return 0.0
            best = {r['id']: r['seconds'] for r in theirs}
            ratios = [r['seconds'] / best[r['id']] for r in mine if best.get(r['id'])]
            return 100 / (1 + sum(ratios) / len(ratios)) if ratios else None
    else:
        weak = {r['id']: r for r in reference['weak']}

        def score_of(mine, theirs):
            return mapped(task.name, mine, [weak[r['id']] for r in mine if r['id'] in weak], theirs)

    total = score_of(rows, ref_rows)
    if family(task.name) == 'exponential' and all(r['status'] == 'passed' for r in rows):
        cases = [c for c in task.cases() if c['id'] in ids]
        if any(c['role'] == 'scored' for c in cases):  # the formal hierarchical mean, as in the final score
            total = task.scoring().score_submission(cases, candidate['aggregates'],
                                                    {c['id']: reference['best'][c['id']] for c in cases})['score']
    return feedback.build(task.name, rows, ref_rows, score=total, categorized=bool(cfg.get('category')),
                          accuracy_name=cfg.get('accuracy'), lower_is_better=cfg.get('lower_is_better', True),
                          timed=cfg.get('timed', True), category_score=score_of,
                          not_evaluated=max(0, total_cases - len(rows)), note=cfg.get('note', ''))


def cmd_public_eval_v63(args) -> None:
    from .runtime import load_runtime

    task, runtime = TASKS[args.task], load_runtime()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    os.environ.update(PUBLIC[task.name]['env'])
    report = evaluate(task, Store(PUBLIC[task.name]['store']), runtime, Path(args.source), args.slot, args.cap,
                      args.label or 'v6.3 test')
    report['created_utc'] = utc()
    write_json(out / 'report.json', report)
    (out / 'report.md').write_text(feedback.render(report))
    print(json.dumps({k: report.get(k) for k in ('task', 'status', 'score')}), flush=True)
    os._exit(0)  # daemon grading threads of an aborted job must not keep the process alive
