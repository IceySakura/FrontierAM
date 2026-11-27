"""Test-time scaling curves of the v6.3 iterative setting: the private score of each test snapshot an agent submitted.

During an iterative attempt the agent only sees public feedback; every `submit --test` froze its source
(dev-runs/<run>/evaluations/test-NN/source.tar). After the attempt has ended, this grades those frozen snapshots on the
private set with the formal grader, in a separate store (var/stores/store-curves, with copies of the formal store's
active baselines, so anchors and pairing are the formal ones). Nothing is reported back to any agent and no formal
score changes. The curve of a cell is the private score of its latest test snapshot at each time (user decision
2026-11-26), ending at the formal score.

  inventory          snapshots of the board's iterative cells -> var/raw/scale-curves/inventory.json
  register TASK...   stage every snapshot into the curves store as a submission (re-validated like any archive)
  grade TASK...      grade the registered snapshots on private (slot broker; polarization only with development idle)
  collect            private score and raw metric per snapshot -> var/raw/scale-curves/scores.json
  regrade SUB...     grade snapshots again after abnormal load; the new job replaces the old one
  curves             per-cell step curves, the per-model average over the tasks, summary rows and SVG panels
                     -> var/raw/scale-curves/{curves.json,svg/}

Usage: FRONTIERAM_RUNTIME=$PWD/.runtime-recal scale-curves.py inventory|register|grade|collect|curves [TASK ...] [--workers N]
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

FORMAL = ROOT / 'var/stores/store-v63'
CURVES = ROOT / 'var/stores/store-curves'
OUT = ROOT / 'var/raw/scale-curves'
TAGS = {'gpt-6.2-sol': 'sol62', 'gpt-6-astra': 'astra', 'gpt-6.1-sol': 'sol', 'claude-opus-5-5': 'opus', 'hy4': 'hy4',
        'qwen4-max': 'qwen4', 'qwen3.8-max-0902': 'qwen', 'minimax-m3.1-flash': 'minimax'}
# task -> (family, v6.3 result-group version)
TASKS = {'continuum-estimation': ('continuum-estimation', 'v1'), 'fluorescence-deconvolution': ('fluorescence-deconvolution', 'v1'),
         'blackbox-cubature-v2': ('blackbox-cubature', 'v2'), 'ihvp-sequence-v4': ('ihvp-sequence', 'v4'),
         'polarization-demosaicking-v3': ('polarization-demosaicking', 'v3'), 'conic': ('conic', 'v3'),
         'trajectory-covariance-v2': ('trajectory-covariance', 'v2')}
SUITE_TASKS = {'blackbox-cubature-v2', 'conic', 'trajectory-covariance-v2'}
# Costly tasks (results/OPEN_ITEMS.md, 11-27 plan): only the latest snapshot at each checkpoint is graded, so their curves
# are exact at the checkpoints; conic on a fixed stratified third of its instances, scored against the reference's and
# the weak anchor's rates on the same instances.
CHECKPOINTS = (15, 30, 60, 90, 120, 180, 240)
CHECKPOINT_TASKS = {'conic', 'trajectory-covariance-v2', 'species-tree-v3', 'exponential-v3', 'galaxy-gravity-v4'}
CONIC_RUNTIME = ROOT / '.runtime-conic-v3'
CONIC_MANIFEST = CONIC_RUNTIME / 'tasks/conic/original/tests/private/data/manifest.json'
CONIC_WEAK = FORMAL / 'suites/suite-conic-20261126T100025Z-507120.report.json'   # HiGHS weak anchor, 165 of 650
TRAJECTORY_REFERENCE = 'suite-trajectory-covariance-v2-20261110T120744Z-41b11a'
DIRECT_TASKS = {'ihvp-sequence-v4'}            # graded by the operator driver on private, outside the job queue
DEV_IDLE_TASKS = {'polarization-demosaicking-v3'}


def board_cells(task: str) -> list[dict]:
    family, version = TASKS[task]
    meta = json.loads((ROOT / 'tasks' / family / 'meta.json').read_text())
    group = next(g for g in meta['result_groups'] if g.get('protocol') == 'v6.3' and g.get('version') == version)
    return [r for r in group['results'] if r['setting'] == '迭代' and r['model'] in TAGS]


def inventory() -> list[dict]:
    rows = []
    for task in TASKS:
        for cell in board_cells(task):
            n = cell.get('attempt', 1)
            log = ROOT / f'var/stores/store/reruns/v63-{TAGS[cell["model"]]}-iterative-{task}-{n}.log'
            finished = [json.loads(line) for line in log.read_text().splitlines() if 'develop_finished' in line]
            run = finished[-1]['run']
            result = json.loads((FORMAL / 'dev-runs' / run / 'develop_result.json').read_text())
            tests = [e for e in result.get('evaluations') or [] if e.get('kind') == 'test']
            latest = {max((e['n'] for e in tests if (e.get('requested_at_seconds') or 0) <= 60 * m), default=None)
                      for m in CHECKPOINTS} - {None}
            for e in tests:
                tar = FORMAL / 'dev-runs' / run / 'evaluations' / f'test-{e["n"]:02d}' / 'source.tar'
                rows.append({'task': task, 'model': cell['model'], 'attempt': n, 'run': run, 'test': e['n'],
                             't_s': e.get('requested_at_seconds'), 'public_score': e.get('score'),
                             'public_status': e.get('status'), 'archive': str(tar.relative_to(ROOT)),
                             'archive_present': tar.is_file(), 'checkpoint': e['n'] in latest})
            rows.append({'task': task, 'model': cell['model'], 'attempt': n, 'run': run, 'test': None,
                         't_s': result.get('development_seconds'), 'final': True, 'submission': cell.get('submission'),
                         'formal_score': cell.get('score'), 'stop': result.get('agent_stop_reason')})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'inventory.json').write_text(json.dumps(rows, indent=1) + '\n')
    print(json.dumps({'cells': sum(1 for r in rows if r.get('final')), 'snapshots': sum(1 for r in rows if not r.get('final'))}))
    return rows


def key(row: dict) -> str:
    return f'{row["run"]}/test-{row["test"]:02d}'


def load(name: str) -> dict:
    path = OUT / name
    return json.loads(path.read_text()) if path.exists() else {}


def save(name: str, data: dict) -> None:
    tmp = OUT / (name + '.tmp')
    tmp.write_text(json.dumps(data, indent=1) + '\n')
    tmp.replace(OUT / name)


def prepare_store(task: str) -> None:
    """The curves store grades with the formal store's active baseline (anchors, paired reference)."""
    from arena.store import Store
    Store(CURVES)
    if task in DIRECT_TASKS or task == 'conic':   # conic is scored from its manifest and the weak-anchor report
        return
    if task == 'trajectory-covariance-v2':        # its scorer reads the reference suite from the candidate's store
        record = FORMAL / 'suites' / f'{TRAJECTORY_REFERENCE}.json'
        if not (CURVES / 'suites' / record.name).exists():
            (CURVES / 'suites').mkdir(parents=True, exist_ok=True)
            shutil.copy2(record, CURVES / 'suites' / record.name)
            for shard in json.loads(record.read_text())['shards']:
                metrics = CURVES / 'results' / shard['job'] / 'grade/metrics.json'
                metrics.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(FORMAL / 'results' / shard['job'] / 'grade/metrics.json', metrics)
        return
    active = (FORMAL / 'baselines' / task / 'ACTIVE').read_text().strip()
    # a baseline derived from another measurement (polarization's 116 scored scenes) has no reference submission
    for src, dst in ((FORMAL / 'baselines' / task, CURVES / 'baselines' / task),
                     (FORMAL / 'submissions' / active, CURVES / 'submissions' / active)):
        if src.exists() and not dst.exists():
            shutil.copytree(src, dst, symlinks=True)


def register(tasks: list[str]) -> None:
    from arena.runtime import load_runtime, sha
    from arena.store import Store
    from arena.tasks import TASKS as ARENA_TASKS
    runtime, store, registry = load_runtime(), Store(CURVES), load('registry.json')
    if 'conic' in tasks and runtime['profile_path'] != str(CONIC_RUNTIME / 'conic-profile.json'):
        raise SystemExit(f'register conic with FRONTIERAM_RUNTIME={CONIC_RUNTIME}')
    for row in json.loads((OUT / 'inventory.json').read_text()):
        if row.get('final') or row['task'] not in tasks or key(row) in registry or not row['archive_present']:
            continue
        if row['task'] in CHECKPOINT_TASKS and not row.get('checkpoint'):
            continue
        task = ARENA_TASKS[row['task']]
        prepare_store(row['task'])
        origin = {'origin': 'scale-snapshot', **{k: row[k] for k in ('model', 'run', 'test', 't_s', 'public_score')},
                  'protocol': 'v6.3', 'setting': 'iterative'}
        submission_id, directory = store.new_submission(task.name, origin)
        stage = directory / 'stage'
        try:
            task.stage(runtime, ROOT / row['archive'], stage)
            shutil.move(stage / 'source.tar', directory / 'source.tar')
            shutil.move(stage / 'source_manifest.json', directory / 'source_manifest.json')
            store.update_submission(submission_id, status='frozen', source_sha256=sha(directory / 'source.tar'))
            registry[key(row)] = submission_id
        except Exception as error:     # a snapshot the formal freeze would reject counts as an invalid candidate
            store.update_submission(submission_id, status='invalid_submission', reason=str(error)[:300])
            registry[key(row)] = submission_id
        finally:
            shutil.rmtree(stage, ignore_errors=True)
        save('registry.json', registry)
    print(json.dumps({'registered': len(registry)}))


def bench(*args: str, runtime: Path | None = None) -> list[dict]:
    env = {**os.environ, 'FRONTIERAM_RUNTIME': str(runtime)} if runtime else None
    done = subprocess.run([str(ROOT / 'bench'), '--store', str(CURVES), *args], cwd=ROOT, capture_output=True, text=True,
                          env=env)
    return [json.loads(line) for line in done.stdout.splitlines() if line.startswith('{')]


def conic_subset() -> Path:
    """A fixed third of the 650 conic instances, every third by hash within each cone family x reference-solved stratum
    (the reference left unsolved exactly the instances whose limit is the 18 s unsolved limit)."""
    path = OUT / 'conic-subset.json'
    if not path.exists():
        strata: dict[tuple, list[str]] = {}
        for row in json.loads(CONIC_MANIFEST.read_text())['instances']:
            strata.setdefault((row['group'], row['time_limit_seconds'] == 18.0), []).append(row['id'])
        chosen = [i for ids in strata.values() for i in sorted(ids, key=lambda x: hashlib.sha256(x.encode()).hexdigest())[::3]]
        path.write_text(json.dumps(sorted(chosen)) + '\n')
    return path


def conic_score(report: dict) -> dict:
    """Weak-anchor mapping (scripts/rounds/v63-grade.sh) with p, p0 and p_B all on the subset's instances."""
    subset = set(json.loads(conic_subset().read_text()))
    solved_by_reference = {r['id'] for r in json.loads(CONIC_MANIFEST.read_text())['instances'] if r['time_limit_seconds'] != 18.0}
    weak = {c['case_id'] for c in json.loads(CONIC_WEAK.read_text())['cases'] if c['success']}
    cases = [c for c in report['cases'] if c['case_id'] in subset]
    if len(cases) != len(subset) or not report.get('complete'):
        return {'score': None, 'status': 'incomplete'}
    p = sum(c['success'] for c in cases) / len(subset)
    p0, pb = len(weak & subset) / len(subset), len(solved_by_reference & subset) / len(subset)
    score = 0.0 if p <= p0 else 50 * (p - p0) / (pb - p0) if p <= pb else 50 + 50 * (p - pb) / (1 - pb)
    return {'score': score, 'status': 'ok', 'raw_error': p, 'p0': p0, 'p_B': pb, 'cases': len(subset)}


def grade(tasks: list[str], workers: int) -> None:
    from arena.store import Store
    store, registry, queued = Store(CURVES), load('registry.json'), load('queued.json')
    rows = [r for r in json.loads((OUT / 'inventory.json').read_text()) if not r.get('final') and r['task'] in tasks]
    for task in tasks:
        pending = [registry[key(r)] for r in rows if r['task'] == task and key(r) in registry and registry[key(r)] not in queued]
        pending = [s for s in pending if store.submission(s).get('status') == 'frozen']
        if task in DIRECT_TASKS:
            grade_direct(task, pending, workers)
            continue
        for submission in pending:
            if task in SUITE_TASKS:
                subset = ['--cases', str(conic_subset())] if task == 'conic' else []
                record = bench('suite', '--task', task, '--submission', submission, '--shards', '7', '--isolation', 'shared',
                               '--cap-s', '3600', *subset, runtime=CONIC_RUNTIME if task == 'conic' else None)
                queued[submission] = {'suite': record[-1]['id']}
            else:
                isolation = 'dev_idle' if task in DEV_IDLE_TASKS else 'shared'
                queued[submission] = {'job': store.enqueue(submission, kind='candidate', isolation=isolation,
                                                           note='scale-curves')}
            save('queued.json', queued)
        if pending:
            env = {**os.environ, 'FRONTIERAM_RUNTIME': str(CONIC_RUNTIME)} if task == 'conic' else None
            subprocess.run([str(ROOT / 'bench'), '--store', str(CURVES), 'grade-worker', '--task', task, '--workers',
                            str(workers), '--drain'], cwd=ROOT, check=False, env=env)


def grade_direct(task: str, submissions: list[str], workers: int) -> None:
    """ihvp-sequence-v4: the operator driver on the private split, as scripts/analyze/ihvp-v4-runs.py final."""
    import tarfile
    from concurrent.futures import ThreadPoolExecutor

    import arena.tasks  # noqa: F401
    from arena.cli import context, placement_for
    from arena.slots import Broker
    from arena.tasks import TASKS as ARENA_TASKS

    layout, _, _ = context()
    broker, arena_task, queued = Broker(layout), ARENA_TASKS[task], load('queued.json')

    def one(submission):
        out = CURVES / 'ihvp-v4' / f'{submission}-private'
        if (out / 'summary.json').exists():
            return
        source = CURVES / 'ihvp-v4' / 'sources' / submission
        if not source.exists():
            source.mkdir(parents=True)
            with tarfile.open(CURVES / 'submissions' / submission / 'source.tar') as tar:
                tar.extractall(source, filter='data')
        shutil.rmtree(out, ignore_errors=True)
        with broker.acquire('grade', arena_task.grade_memory, f'scale curves {submission}', isolation='shared') as lease:
            arena_task.run_grader(source, out, placement_for(lease, layout), split='private')

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(one, submissions))
    for submission in submissions:
        queued[submission] = {'direct': f'ihvp-v4/{submission}-private'}
    save('queued.json', queued)


def regrade(submissions: list[str]) -> None:
    """Grade snapshots once more (their first grading met abnormal machine load, results/OPEN_ITEMS.md 11-26); the new job
    replaces the old one, which is kept as superseded."""
    from arena.store import Store
    store, queued = Store(CURVES), load('queued.json')
    for submission in submissions:
        task = store.submission(submission)['task']
        before = {p.name for p in (CURVES / 'jobs/done').glob('*.json')}
        bench('grade', '--submission', submission, '--task', task, '--isolation',
              'dev_idle' if task in DEV_IDLE_TASKS else 'shared')
        new = [json.loads(p.read_text()) for p in (CURVES / 'jobs/done').glob('*.json') if p.name not in before]
        job = next((j['id'] for j in new if j.get('submission') == submission), None)
        if job:
            queued[submission] = {'job': job, 'superseded': (queued.get(submission) or {}).get('job')}
            save('queued.json', queued)
        print(json.dumps({'regraded': submission, 'job': job}))


def collect() -> None:
    """Private score per snapshot, read from the same records the formal publication uses."""
    from arena.store import Store
    store, registry, queued = Store(CURVES), load('registry.json'), load('queued.json')
    scores = {}
    for snapshot, submission in registry.items():
        meta = store.submission(submission)
        if meta.get('status') == 'invalid_submission':
            scores[snapshot] = {'submission': submission, 'score': 0.0, 'status': 'invalid_submission'}
            continue
        q = queued.get(submission) or {}
        if 'job' in q:
            path = CURVES / 'results' / q['job'] / 'score.json'
            if path.exists():
                s = json.loads(path.read_text())
                scores[snapshot] = {'submission': submission, 'score': s.get('score'), 'status': s.get('status'),
                                    'raw_error': (s.get('candidate') or {}).get('raw_error', s.get('raw_error'))}
        elif 'suite' in q:
            task = json.loads((CURVES / 'suites' / f'{q["suite"]}.json').read_text())['task']
            out = OUT / 'suite-scores' / f'{q["suite"]}.json'
            out.parent.mkdir(parents=True, exist_ok=True)
            if not out.exists() and task == 'conic':
                bench('suite-report', '--suite', q['suite'], runtime=CONIC_RUNTIME)
                report = CURVES / 'suites' / f'{q["suite"]}.report.json'
                result = conic_score(json.loads(report.read_text())) if report.exists() else {'status': 'incomplete'}
                if result['status'] == 'ok':
                    out.write_text(json.dumps(result) + '\n')
            elif not out.exists():
                scorer, extra = {'blackbox-cubature-v2': ('cubature-suite-score.py', ['--task', 'blackbox-cubature-v2']),
                                 'trajectory-covariance-v2': ('trajectory-suite-score.py', ['--reference', TRAJECTORY_REFERENCE])}[task]
                subprocess.run([str(ROOT / '.venv-ops/bin/python'), str(ROOT / 'scripts/grade' / scorer), '--store', str(CURVES),
                                *extra, '--candidate', q['suite'], '--out', str(out)], cwd=ROOT, capture_output=True, check=False)
            if out.exists():
                s = json.loads(out.read_text())
                raw = s.get('gain') if task == 'blackbox-cubature-v2' else s.get('raw_error', (s.get('candidate') or {}).get('raw_error'))
                scores[snapshot] = {'submission': submission, 'score': s.get('score'), 'status': s.get('status'), 'raw_error': raw}
        elif 'direct' in q:
            summary = CURVES / q['direct'] / 'summary.json'
            if summary.exists():
                scores[snapshot] = {'submission': submission, **ihvp_score(summary)}
    save('scores.json', scores)
    print(json.dumps({'scored': len(scores), 'registered': len(registry)}))


def ihvp_score(summary: Path) -> dict:
    """Same rule and private anchors as scripts/analyze/ihvp-v4-runs.py scores (first cg / lbfgs measurement)."""
    def mean_loss(path: Path):
        rows = json.loads(path.read_text()).get('records', [])
        ok = [r for r in rows if r['status'] == 'ok' and r.get('energy_loss') is not None]
        return (sum(r['energy_loss'] for r in ok) / len(ok) if ok and len(ok) == len(rows) else None), len(ok), len(rows)
    e0 = mean_loss(FORMAL / 'ihvp-v4/cg-private/summary.json')[0]
    e50 = mean_loss(FORMAL / 'ihvp-v4/lbfgs-private/summary.json')[0]
    loss, valid, n = mean_loss(summary)
    if loss is None:
        return {'score': 0.0, 'status': 'submission_failure', 'raw_error': None, 'valid': valid, 'cases': n}
    score = 0.0 if loss >= e0 else 50 * (e0 - loss) / (e0 - e50) if loss > e50 else 50 + 50 * (e50 - loss) / e50
    return {'score': score, 'status': 'ok', 'raw_error': loss, 'valid': valid, 'cases': n}


MINUTES = 240
MARKS = (60, 120, 180, 240)


def cells() -> dict:
    """(task, model) -> attempt, its tests in time order (public and private score) and its end (formal score)."""
    scores, out = load('scores.json'), {}
    for row in json.loads((OUT / 'inventory.json').read_text()):
        cell = out.setdefault((row['task'], row['model']), {'task': row['task'], 'model': row['model'],
                                                             'attempt': row['attempt'], 'run': row['run'], 'tests': []})
        if row.get('final'):
            cell.update(end=(row['t_s'] or MINUTES * 60) / 60, formal=row.get('formal_score'), stop=row.get('stop'))
        else:
            s = scores.get(key(row)) or {}
            cell['tests'].append({'test': row['test'], 'minute': (row['t_s'] or 0) / 60, 'public': row.get('public_score'),
                                  'private': s.get('score'), 'status': s.get('status')})
    for cell in out.values():
        cell['tests'].sort(key=lambda t: t['minute'])
    return out


def at(cell: dict, minute: float, which: str = 'private') -> float | None:
    """The score of the latest test snapshot requested by `minute` (0 before the first test); the formal score from the
    end of the attempt on. A snapshot not graded yet keeps the previous value."""
    if minute >= cell['end'] and which == 'private' and cell.get('formal') is not None:
        return cell['formal']
    value = 0.0
    for test in cell['tests']:
        if test['minute'] <= minute and test[which] is not None:
            value = test[which]
    return value


def curves_svg(title: str, lines: dict[str, list[tuple[float, float]]], dots: dict[str, list[tuple[float, float]]],
               names: dict[str, str]) -> str:
    """Step curves (x = minutes, y = 0-100, reference 50 dashed), one colour per model; dots: public scores."""
    sys.path.insert(0, str(ROOT / 'scripts/site'))
    from board_stats import COLORS
    width, height, left, pad = 560, 240, 34, 22
    x = lambda m: left + (width - left - pad) * m / MINUTES
    y = lambda s: height - pad - (height - 2 * pad) * max(0.0, min(100.0, s)) / 100
    parts = [f'<text x="{left}" y="14" font-size="12" font-weight="600">{title}</text>',
             f'<line x1="{left}" y1="{y(0)}" x2="{width - pad}" y2="{y(0)}" stroke="#bbb"/>',
             f'<line x1="{left}" y1="{y(50)}" x2="{width - pad}" y2="{y(50)}" stroke="#bbb" stroke-dasharray="4 3"/>']
    for s in (0, 50, 100):
        parts.append(f'<text x="{left - 4}" y="{y(s) + 4}" text-anchor="end" font-size="10" fill="#888">{s}</text>')
    for m in (0, 60, 120, 180, 240):
        parts.append(f'<text x="{x(m)}" y="{height - 6}" text-anchor="middle" font-size="10" fill="#888">{m}</text>')
    for model, points in lines.items():
        colour = COLORS.get(model, '#6b7280')
        path = ' '.join(f'{x(m):.1f},{y(s):.1f}' for m, s in points)
        parts.append(f'<polyline points="{path}" fill="none" stroke="{colour}" stroke-width="1.6">'
                     f'<title>{names.get(model, model)}</title></polyline>')
        for m, s in dots.get(model, []):
            parts.append(f'<circle cx="{x(m):.1f}" cy="{y(s):.1f}" r="2.2" fill="none" stroke="{colour}">'
                         f'<title>{names.get(model, model)} public {s:.1f}（{m:.0f} 分钟）</title></circle>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
            f'font-family="sans-serif">{"".join(parts)}</svg>\n')


def build_curves() -> None:
    """Curves of the latest test snapshot's private score per cell, their average over the tasks per model, summary rows."""
    names = {m: m for m in TAGS}
    grid = [m / 2 for m in range(0, 2 * MINUTES + 1)]
    data = cells()
    steps = {k: [(m, at(c, m)) for m in grid] for k, c in data.items()}
    summary = []
    for (task, model), cell in sorted(data.items()):
        graded = [t for t in cell['tests'] if t['private'] is not None]
        gaps = [t['public'] - t['private'] for t in graded if t['public'] is not None]
        summary.append({'task': task, 'model': model, 'attempt': cell['attempt'], 'tests': len(cell['tests']),
                        'graded': len(graded), 'end_minute': round(cell['end'], 1), 'formal': cell.get('formal'),
                        'best_snapshot': max((t['private'] for t in graded), default=None),
                        'first_test_minute': round(cell['tests'][0]['minute'], 1) if cell['tests'] else None,
                        **{f'at_{m}': at(cell, m) for m in MARKS},
                        'mean_public_minus_private': sum(gaps) / len(gaps) if gaps else None})
    models = sorted({m for _, m in data}, key=list(TAGS).index)
    tasks = [t for t in TASKS if any(k[0] == t for k in data)]
    average = {m: [(g, sum(at(data[(t, m)], g) for t in tasks) / len(tasks)) for g in grid]
               for m in models if all((t, m) in data for t in tasks)}
    svg = OUT / 'svg'
    svg.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        lines = {m: steps[(task, m)] for m in models if (task, m) in steps}
        dots = {m: [(t['minute'], t['public']) for t in data[(task, m)]['tests'] if t['public'] is not None]
                for m in models if (task, m) in data}
        (svg / f'{task}.svg').write_text(curves_svg(task, lines, dots, names))
    (svg / 'average.svg').write_text(curves_svg(f'{len(tasks)} 题平均', average, {}, names))
    save('curves.json', {'tasks': tasks, 'models': models, 'marks': list(MARKS), 'summary': summary,
                         'average_at': {m: {str(g): s for g, s in pts if g in MARKS} for m, pts in average.items()},
                         'cells': {f'{t}/{m}': c for (t, m), c in data.items()}})
    print(json.dumps({'cells': len(data), 'tasks': tasks, 'average_models': len(average)}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=('inventory', 'register', 'grade', 'regrade', 'collect', 'curves'))
    parser.add_argument('tasks', nargs='*')
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    tasks = args.tasks or list(TASKS)
    if args.command == 'inventory':
        inventory()
    elif args.command == 'register':
        register(tasks)
    elif args.command == 'grade':
        grade(tasks, args.workers)
    elif args.command == 'curves':
        build_curves()
    elif args.command == 'regrade':   # TASK... are submission ids here
        regrade(args.tasks)
    else:
        collect()


if __name__ == '__main__':
    main()
