"""FrontierAM bench: isolated concurrent development, separate grading, baselines and scores.

    ./bench host setup|status          kernel CPU partition, grading-only Docker daemon, slot layout
    ./bench prepare --codex PATH       seal the runtime (unchanged legacy entry point)
    ./bench develop --task T --model M [--count N]     develop only; frozen sources enter the grading queue
    ./bench submit --task T --source DIR               queue any frozen source directory
    ./bench grade-worker [--workers N] [--drain]       grade queued submissions on grade slots
    ./bench baseline --task T                          grade the frozen reference; activate it for scoring
    ./bench interference --task T --levels 1,2,4       measure timing drift of concurrent grading
    ./bench smoke --task both [--parallel K]           fixture attempts through the full concurrent pipeline
    ./bench status | leaderboard --task T | rescore | recover
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import threading
import time

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import machine
from .runtime import ROOT, child_environment, load_runtime, sha
from .slots import Broker, Lease
from .store import Store, new_id, read_json, utc, write_json
from .tasks import TASKS, Placement


MODELS = [
    'claude-opus-5-5',
    'deepseek-v4-pro',
    'hy4-preview',
    'minimax-m3.1-flash',
    'step-5-preview',
    'gpt-6.1-sol',
    'gpt-6-astra',
    'gemini-4-argon',
    'hy4',
]
DEFAULT_BASE_URL = 'http://api.aihub.woa.com/standard/v1'
PRINT_LOCK = threading.Lock()


def say(**fields) -> None:
    with PRINT_LOCK:
        print(json.dumps(fields, ensure_ascii=False), flush=True)


def placement_for(lease: Lease, layout: dict) -> Placement:
    return Placement(lease.cpuset, lease.cgroup_parent, lease.slot['docker_host'], lease.id)


def run_in_slot(lease: Lease, spec: dict, spec_path: Path, log: Path, extra_fds: tuple[int, ...] = ()) -> int:
    """Run one job pinned to the slot's host CPU inside the partition; the lease descriptor goes with it."""
    write_json(spec_path, spec)
    unit = f'frontieram-{spec["mode"]}-{spec_path.parent.name}'[:200]
    command = [
        'systemd-run',
        '--scope',
        '--quiet',
        '--collect',
        f'--unit={unit}',
        f'--slice={machine.SLICES["host"]}',
        '-p',
        f'AllowedCPUs={lease.host_cpu}',
        '--',
        sys.executable,
        '-m',
        'arena.jobs',
        str(spec_path),
    ]
    with log.open('ab') as output:
        process = subprocess.run(
            command,
            cwd=ROOT,
            env=child_environment(),
            stdout=output,
            stderr=output,
            pass_fds=(*lease.fds, *extra_fds),
            check=False,
        )
    return process.returncode


def context() -> tuple[dict, Broker, dict]:
    layout = machine.load()
    machine.require_ready(layout)
    runtime = load_runtime()
    for key in ('grade_docker_hosts', 'conic_reference_root'):
        if key not in runtime:
            raise RuntimeError('This .runtime predates per-slot grading daemons; rename it and run ./bench prepare')
    if set(runtime['grade_docker_hosts']) != set(layout['docker_hosts']['grade']):
        raise RuntimeError('The sealed runtime loaded images into other grading daemons; run ./bench prepare again')
    return layout, Broker(layout), runtime


# Development -------------------------------------------------------------------------------------------------------


def develop_one(layout, broker, runtime, store, args, index: int) -> dict:
    task = TASKS[args.task]
    run_id = new_id(f'dev-{task.name}')
    output = store.root / 'dev-runs' / run_id
    output.mkdir(parents=True)
    with broker.acquire('dev', task.dev_memory, f'develop {run_id}', timeout=args.wait) as lease:
        placement = placement_for(lease, layout)
        say(event='develop_started', run=run_id, task=task.name, model=args.model, slot=lease.id)
        started = time.time()
        spec = {
            'mode': 'develop',
            'task': task.name,
            'placement': placement.__dict__,
            'output': str(output),
            'model': args.model,
            'base_url': args.base_url,
            'key_file': str(args.key_file) if args.key_file else None,
            'smoke': args.smoke,
            'protocol': getattr(args, 'protocol', 'v4'),
            'development_seconds': getattr(args, 'development_seconds', None),
        }
        code = run_in_slot(lease, spec, output / 'spec.json', output / 'job.log')
    result_path = output / 'develop_result.json'
    result = read_json(result_path) if result_path.exists() else {'task_status': 'unmeasured', 'reason': 'job_failed'}
    result.update(job_exit_code=code, wall_seconds=time.time() - started, slot=lease.id)
    origin = {
        'origin': 'agent',
        'model': args.model,
        'protocol': getattr(args, 'protocol', 'v4'),
        'smoke': args.smoke or bool(getattr(args, 'development_seconds', None)),
        'dev_run': run_id,
        'dev_slot': lease.id,
        'runtime_lock_sha256': runtime['lock_sha256'],
        'development': result,
    }
    submission_id, directory = store.new_submission(task.name, origin)
    frozen = output / 'task'
    if result.get('task_status') == 'verification_completed' and (frozen / 'source.tar').is_file():
        # Only the canonical frozen archive crosses into grading; it is re-validated when staged.
        shutil.copy2(frozen / 'source.tar', directory / 'source.tar')
        shutil.copy2(frozen / 'source_manifest.json', directory / 'source_manifest.json')
        store.update_submission(submission_id, status='frozen', source_sha256=sha(directory / 'source.tar'))
        job = store.enqueue(submission_id) if not args.no_grade else None
    elif result.get('task_status') == 'invalid_submission':
        store.update_submission(submission_id, status='invalid_submission', score=0.0)
        job = None
    else:
        store.update_submission(submission_id, status='unmeasured')
        job = None
    say(event='develop_finished', run=run_id, submission=submission_id, status=store.submission(submission_id)['status'],
        job=job, agent_stop=result.get('agent_stop_reason'))
    return {'submission': submission_id, 'job': job, 'result': result}


def cmd_develop(args) -> list[dict]:
    layout, broker, runtime = context()
    if not args.smoke and not args.key_file:
        raise SystemExit('Supply --key-file PATH or AIHUB_KEY_FILE (a file path, not the key itself)')
    store = Store(args.store)
    with ThreadPoolExecutor(args.count) as pool:
        futures = [pool.submit(develop_one, layout, broker, runtime, store, args, i) for i in range(args.count)]
        return [f.result() for f in futures]


def cmd_submit(args) -> None:
    store = Store(args.store)
    runtime = load_runtime()
    task = TASKS[args.task]
    submission_id, directory = store.new_submission(task.name, {'origin': 'manual', 'label': args.label})
    from .tasks import tar_directory

    tar_directory(Path(args.source).resolve(), directory / 'incoming.tar')
    stage = directory / 'stage'
    task.stage(runtime, directory / 'incoming.tar', stage)
    shutil.move(stage / 'source.tar', directory / 'source.tar')
    shutil.move(stage / 'source_manifest.json', directory / 'source_manifest.json')
    shutil.rmtree(stage)
    (directory / 'incoming.tar').unlink()
    store.update_submission(submission_id, status='frozen', source_sha256=sha(directory / 'source.tar'))
    say(event='submitted', submission=submission_id, job=store.enqueue(submission_id))


# Grading -----------------------------------------------------------------------------------------------------------


def uid_tasks(uid: int) -> int:
    """Tasks (threads included) owned by a real UID on the whole host: what RLIMIT_NPROC counts."""
    count = 0
    for status in Path('/proc').glob('[0-9]*/task/[0-9]*/status'):
        try:
            for line in status.read_text().splitlines():
                if line.startswith('Uid:'):
                    count += int(line.split()[1]) == uid
                    break
        except OSError:
            continue
    return count


def grade_job(layout, broker, runtime, store, job) -> dict:
    task = TASKS[job['task']]
    results = store.result_dir(job['id'])
    archive = store.submission_dir(job['submission']) / 'source.tar'
    if task.paired and job['kind'] != 'baseline' and store.active_baseline(task.name) is None:
        return store.finish(job, 'failed', status='unmeasured', reason='no_active_baseline_to_pair_with')
    host = machine.status(layout)
    if not host['ready']:
        # Timing under lost isolation is not comparable; leave the submission ungraded rather than mis-scored.
        failed = [k for k, v in host['checks'].items() if not v]
        write_json(results / 'host_not_ready.json', host)
        say(event='grade_refused', job=job['id'], failed_checks=failed)
        return store.finish(job, 'failed', status='unmeasured', reason='host_isolation_lost:' + ','.join(failed))
    if task.grader_uid is not None:
        waited = False
        while (count := uid_tasks(task.grader_uid)) > 128 - task.grader_uid_headroom:
            if not waited:
                say(event='grade_waiting_uid_headroom', job=job['id'], uid=task.grader_uid, tasks=count)
                waited = True
            time.sleep(30)
    # Interference probes measure concurrency on purpose; a job may carry an escalated policy from the load gate.
    policy = 'shared' if job['kind'] == 'probe' else job.get('isolation') or task.grade_isolation
    say(event='grade_waiting', job=job['id'], task=task.name, isolation=policy)
    # A job may name its slot, e.g. a control sentinel placed on the noisiest slot of a concurrent round.
    with broker.acquire('grade', task.grade_memory, f'grade {job["id"]}', slot_id=job.get('slot'),
                        isolation=policy) as lease:
        placement = placement_for(lease, layout)
        say(event='grade_started', job=job['id'], task=task.name, kind=job['kind'], slot=lease.id)
        paired = None
        if task.paired:
            # Chosen only once the slot is held, so a baseline activated while this job waited is the one used.
            # A baseline is paired with itself; everything else with the reference its score is anchored to.
            active = store.active_baseline(task.name)
            paired = archive if job['kind'] == 'baseline' else store.submission_dir(active['id']) / 'source.tar'
        before = broker.leases()
        started = time.time()
        spec = {
            'mode': 'grade',
            'task': task.name,
            'placement': placement.__dict__,
            'output': str(results / 'grade'),
            'archive': str(archive),
            'paired_reference': str(paired) if paired else None,
            'case_ids': job.get('case_ids'),
        }
        code = run_in_slot(lease, spec, results / 'spec.json', results / 'job.log', extra_fds=(job['lock_fd'],))
        isolation = {'policy': lease.isolation, 'held_idle': [slot_id for slot_id, _ in lease.held]}
        after = broker.leases()
    metrics_path = results / 'grade/metrics.json'
    record = {
        'job': job['id'],
        'task': task.name,
        'kind': job['kind'],
        'submission': job['submission'],
        'slot': lease.id,
        'placement': placement.__dict__,
        'layout_sha256': layout['layout_sha256'],
        'runtime_lock_sha256': runtime['lock_sha256'],
        'source_sha256': sha(archive),
        'paired_reference_sha256': sha(paired) if paired else None,
        'job_exit_code': code,
        'isolation': isolation,
        'host_checks_before': host['checks'],
        'grading_wall_seconds': time.time() - started,
        'concurrent_leases_at_start': [r for r in before if r.get('slot') != lease.id],
        'concurrent_leases_at_end': [r for r in after if r.get('slot') != lease.id],
    }
    if code or not metrics_path.exists():
        record.update(status='unmeasured', reason='grading_job_failed')
        write_json(results / 'result.json', record)
        return store.finish(job, 'failed', status='unmeasured', reason='grading_job_failed')
    metrics = read_json(metrics_path)
    write_json(results / 'result.json', record)
    if job['kind'] == 'baseline':
        return finish_baseline(store, runtime, task, job, record, metrics)
    if job['kind'] == 'probe':
        return store.finish(job, 'done', status='probe', note=job.get('note'))
    if job['kind'] == 'shard':
        # One part of a full-suite assessment; `suite-report` merges the parts once all are graded.
        say(event='shard_graded', job=job['id'], suite=job.get('suite'), success_rate=metrics.get('success_rate'))
        return store.finish(job, 'done', status='shard', suite=job.get('suite'))
    if job['kind'] == 'sentinel':
        baseline = store.active_baseline(task.name)
        score = task.score(runtime, metrics, baseline)
        write_json(results / 'score.json', {**score, 'baseline': (baseline or {}).get('id')})
        say(event='sentinel', job=job['id'], task=task.name, score=score.get('score'), slot=lease.id)
        return store.finish(job, 'done', status='sentinel', score=score.get('score'))
    score = task.score(runtime, metrics, store.active_baseline(task.name))
    write_json(results / 'score.json', {**score, 'baseline': (store.active_baseline(task.name) or {}).get('id')})
    gate = task.load_gate
    if gate is not None and policy != 'quiet' and (score.get('load_factor') or 0) > gate:
        # Pairing cancels load only as far as candidate and reference respond to it alike; above the gate that
        # assumption is not trusted, so the submission is measured again with the machine kept quiet.
        regrade = store.enqueue(job['submission'], isolation='quiet', note=f'load gate {score["load_factor"]:.3f}')
        say(event='load_gate', job=job['id'], load_factor=score['load_factor'], gate=gate, regrade=regrade)
        return store.finish(job, 'done', status='superseded_by_quiet_regrade', regrade=regrade,
                            load_factor=score['load_factor'])
    store.update_submission(job['submission'], status='graded', score=score.get('score'), score_status=score.get('status'))
    say(event='graded', job=job['id'], submission=job['submission'], score=score.get('score'), status=score.get('status'))
    return store.finish(job, 'done', status=score.get('status'), score=score.get('score'))


def finish_baseline(store, runtime, task, job, record, metrics) -> dict:
    ok, reason = task.baseline_ok(metrics)
    baseline_id = job['submission']
    directory = store.baseline_root(task.name) / baseline_id
    directory.mkdir(parents=True, exist_ok=True)
    report = store.result_dir(job['id']) / 'grade/evaluation.json'
    if report.exists():
        shutil.copy2(report, directory / 'evaluation.json')
    baseline = {
        'id': baseline_id,
        'task': task.name,
        'qualified': ok,
        'reason': reason,
        'job': job['id'],
        'created_utc': utc(),
        'slot': record['slot'],
        'layout_sha256': record['layout_sha256'],
        'runtime_lock_sha256': record['runtime_lock_sha256'],
        'concurrent_leases_at_start': record['concurrent_leases_at_start'],
        'metrics': metrics,
    }
    write_json(directory / 'baseline.json', baseline)
    if ok and (store.active_baseline(task.name) is None or job.get('note') == 'activate'):
        # Scoring is done here against the store baseline. The sealed worker's own calibration lock is tied to one
        # Docker daemon ID and cannot follow per-slot daemons, so the worker reports local_calibration_required.
        store.activate_baseline(task.name, baseline_id)
    say(event='baseline_finished', task=task.name, baseline=baseline_id, qualified=ok, reason=reason,
        active=(store.active_baseline(task.name) or {}).get('id'))
    return store.finish(job, 'done', status='baseline', qualified=ok)


def worker_loop(layout, broker, runtime, store, args, stop: threading.Event, job_ids=None) -> list[dict]:
    done = []
    while not stop.is_set():
        job = None
        if job_ids is None:
            job = store.claim(tasks=args.task)
        else:
            for job_id in list(job_ids):
                job = store.claim(job_id=job_id)
                if job is not None:
                    break
        if job is None:
            if args.drain or job_ids is not None:
                return done
            time.sleep(5)
            continue
        try:
            done.append(grade_job(layout, broker, runtime, store, job))
        except Exception as error:
            done.append(store.finish(job, 'failed', status='unmeasured', reason=f'{type(error).__name__}: {error}'))
            say(event='grade_failed', job=job['id'], error=f'{type(error).__name__}: {error}')
    return done


def run_workers(args, job_ids=None) -> list[dict]:
    layout, broker, runtime = context()
    store = Store(args.store)
    store.recover()
    grade_slots = sum(s['kind'] == 'grade' for s in layout['slots'])
    workers = min(args.workers or grade_slots, grade_slots)
    stop = threading.Event()
    with ThreadPoolExecutor(workers) as pool:
        futures = [
            pool.submit(worker_loop, layout, broker, runtime, store, args, stop, job_ids) for _ in range(workers)
        ]
        try:
            return [row for f in futures for row in f.result()]
        except KeyboardInterrupt:
            stop.set()
            raise


def cmd_grade(args) -> None:
    store = Store(args.store)
    job = store.enqueue(args.submission)
    args.drain = True
    run_workers(args, job_ids=[job])


def enqueue_reference(store, runtime, task, *, kind: str, note: str | None = None, isolation: str | None = None) -> str:
    return store.enqueue(freeze_reference(store, runtime, task, kind=kind), kind=kind, note=note, isolation=isolation)


def freeze_reference(store, runtime, task, *, kind: str) -> str:
    submission_id, directory = store.new_submission(task.name, {'origin': 'reference', 'kind': kind})
    archive = task.reference_archive(runtime, directory)
    stage = directory / 'stage'
    task.stage(runtime, archive, stage)
    shutil.move(stage / 'source.tar', directory / 'source.tar')
    shutil.move(stage / 'source_manifest.json', directory / 'source_manifest.json')
    shutil.rmtree(stage)
    archive.unlink()
    store.update_submission(submission_id, status='frozen', source_sha256=sha(directory / 'source.tar'))
    return submission_id


def enqueue_archive(store, runtime, task, archive: Path, *, kind: str) -> str:
    """Queue a frozen canonical source.tar (for example from another store) after re-validating it."""
    submission_id, directory = store.new_submission(task.name, {'origin': 'archive', 'label': str(archive)})
    stage = directory / 'stage'
    task.stage(runtime, archive, stage)
    shutil.move(stage / 'source.tar', directory / 'source.tar')
    shutil.move(stage / 'source_manifest.json', directory / 'source_manifest.json')
    shutil.rmtree(stage)
    store.update_submission(submission_id, status='frozen', source_sha256=sha(directory / 'source.tar'))
    return store.enqueue(submission_id, kind=kind)


def cmd_baseline(args) -> None:
    store = Store(args.store)
    runtime = load_runtime()
    task = TASKS[args.task]
    if store.active_baseline(task.name) and not args.replace:
        raise SystemExit(f'{task.name} already has an active baseline; pass --replace to measure and activate another')
    # Load factors divide by baseline times, so the baseline itself is always measured on a quiet machine.
    jobs = [enqueue_reference(store, runtime, task, kind='baseline', note='activate' if args.replace else None,
                              isolation='quiet')]
    args.drain, args.task = True, [task.name]
    run_workers(args, job_ids=jobs)


def cmd_sentinel(args) -> None:
    """Grade the reference again under the task's real policy; a stable environment reproduces the baseline."""
    store = Store(args.store)
    runtime = load_runtime()
    task = TASKS[args.task]
    if store.active_baseline(task.name) is None:
        raise SystemExit('No active baseline to compare against')
    jobs = [enqueue_reference(store, runtime, task, kind='sentinel') for _ in range(args.count)]
    args.drain, args.task = True, [task.name]
    run_workers(args, job_ids=jobs)


def shard_cases(instances: list[dict], count: int) -> list[list[str]]:
    """Balance worst-case time: longest limit first, each case to the shard with the least total limit so far."""
    shards: list[list[str]] = [[] for _ in range(count)]
    totals = [0.0] * count
    for case in sorted(instances, key=lambda c: (-c['time_limit_seconds'], c['id'])):
        index = totals.index(min(totals))
        shards[index].append(case['id'])
        totals[index] += case['time_limit_seconds']
    return shards


def cmd_suite(args) -> None:
    """Queue a full held-out suite assessment of one submission (or the reference) as shards on separate slots."""
    store = Store(args.store)
    runtime = load_runtime()
    task = TASKS[args.task]
    if not hasattr(task, 'suite'):
        raise SystemExit(f'{task.name} has no shardable held-out suite')
    shards = shard_cases(task.suite(runtime), args.shards)
    slots = args.slots.split(',') if args.slots else [None] * len(shards)
    if len(slots) != len(shards):
        raise SystemExit('--slots needs one slot per shard')
    if args.cases:
        # A subset keeps every case in its full-suite shard, hence on the same slot as in full-suite runs.
        subset = set(json.loads(Path(args.cases).read_text()))
        kept = [([i for i in ids if i in subset], slot) for ids, slot in zip(shards, slots)]
        if sum(len(ids) for ids, _ in kept) != len(subset):
            raise SystemExit('--cases names instances outside the held-out suite')
        shards, slots = [ids for ids, _ in kept if ids], [slot for ids, slot in kept if ids]
    suite_id = new_id(f'suite-{task.name}')
    submission = freeze_reference(store, runtime, task, kind='suite') if args.reference else args.submission
    # The same partition and the same shard-to-slot map for every suite put each case on the same slot for the
    # reference and every candidate, so per-slot conditions apply to both sides alike.
    jobs = [
        store.enqueue(submission, kind='shard', isolation=args.isolation, note=f'{suite_id} shard {i + 1}/{len(shards)}',
                      extra={'suite': suite_id, 'case_ids': ids, 'slot': slot},
                      queue='held' if args.hold else 'pending')
        for i, (ids, slot) in enumerate(zip(shards, slots))
    ]
    record = {'id': suite_id, 'task': task.name, 'submission': submission, 'reference': args.reference,
              'created_utc': utc(), 'shards': [{'job': j, 'cases': len(ids)} for j, ids in zip(jobs, shards)],
              'runtime_lock_sha256': runtime['lock_sha256']}
    write_json(store.root / 'suites' / f'{suite_id}.json', record)
    say(event='suite_queued', **record)


def cmd_suite_report(args) -> None:
    """Merge the shards of a suite: success counts over the whole held-out set and per group."""
    store = Store(args.store)
    record = read_json(store.root / 'suites' / f'{args.suite}.json')
    cases, missing = [], []
    for shard in record['shards']:
        metrics_path = store.result_dir(shard['job']) / 'grade/metrics.json'
        done = store.root / 'jobs/done' / f'{shard["job"]}.json'
        if not done.exists() or not metrics_path.exists():
            missing.append(shard['job'])
            continue
        metrics = read_json(metrics_path)
        if metrics.get('success_rate') is None:
            missing.append(shard['job'])
            continue
        cases.extend(metrics['cases'])
    groups: dict[str, dict] = {}
    for case in cases:
        group = groups.setdefault(case['group'], {'cases': 0, 'successes': 0})
        group['cases'] += 1
        group['successes'] += bool(case['success'])
    solved = sum(bool(c['success']) for c in cases)
    report = {**record, 'complete': not missing, 'missing_shards': missing, 'cases_graded': len(cases),
              'successes': solved, 'success_rate': solved / len(cases) if cases else None, 'groups': groups,
              'cases': sorted(cases, key=lambda c: c['case_id'])}
    write_json(store.root / 'suites' / f'{args.suite}.report.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'cases'}, indent=2, ensure_ascii=False))


def cmd_rescore(args) -> None:
    store = Store(args.store)
    runtime = load_runtime()
    for job in store.jobs('done'):
        if job.get('kind') != 'candidate':
            continue
        task = TASKS[job['task']]
        metrics = read_json(store.result_dir(job['id']) / 'grade/metrics.json')
        baseline = store.active_baseline(task.name)
        score = task.score(runtime, metrics, baseline)
        write_json(store.result_dir(job['id']) / 'score.json', {**score, 'baseline': (baseline or {}).get('id')})
        store.update_submission(job['submission'], status='graded', score=score.get('score'),
                                score_status=score.get('status'))
        say(job=job['id'], submission=job['submission'], score=score.get('score'), status=score.get('status'))


# Interference ------------------------------------------------------------------------------------------------------


def dev_load(layout, broker, count: int, stop: threading.Event, leases: list) -> None:
    """Memory-bandwidth and cache pressure on development slots, the worst case a busy agent fleet produces."""
    image = 'frontieram-exponential:port-v1'
    hog = (
        'import numpy as np,time\n'
        'a=np.ones(64*2**20//8);b=np.empty_like(a)\n'
        'while True: np.copyto(b,a); a+=1e-9\n'
    )
    for index in range(count):
        lease = broker.acquire('dev', 2**30, f'interference dev-load {index}', timeout=60)
        leases.append(lease)
        subprocess.run(
            ['docker', '--host', layout['docker_hosts']['dev'], 'run', '-d', '--rm', '--name',
             f'frontieram-devload-{lease.id}', '--network', 'none', '--cpus', '1', '--cpuset-cpus', lease.cpuset,
             '--cgroup-parent', lease.cgroup_parent, '--memory', '1g', '--entrypoint', 'python3', image, '-c', hog],
            check=True, capture_output=True,
        )


def cmd_interference(args) -> None:
    layout, broker, runtime = context()
    store = Store(args.store)
    task = TASKS[args.task]
    baseline = store.active_baseline(task.name)
    report = {'task': task.name, 'created_utc': utc(), 'baseline': baseline and baseline['id'], 'levels': []}
    for level in [int(x) for x in args.levels.split(',')]:
        stop, leases = threading.Event(), []
        if args.dev_load:
            dev_load(layout, broker, args.dev_load, stop, leases)
        try:
            candidates = [enqueue_archive(store, runtime, task, Path(p), kind='probe') for p in args.candidate or []]
            jobs = candidates + [
                enqueue_reference(store, runtime, task, kind='probe', note=f'level-{level}')
                for _ in range(max(0, level - len(candidates)))
            ]
            args.drain, args.workers, args.task_filter = True, level, None
            run_workers(args, job_ids=jobs)
        finally:
            for lease in leases:
                subprocess.run(['docker', '--host', layout['docker_hosts']['dev'], 'rm', '-f',
                                f'frontieram-devload-{lease.id}'], capture_output=True)
                lease.release()
        rows = []
        for job_id in jobs:
            metrics = read_json(store.result_dir(job_id) / 'grade/metrics.json')
            entry = {'job': job_id, 'slot': read_json(store.result_dir(job_id) / 'result.json')['slot'],
                     'valid_runs': metrics['valid_runs'],
                     'source': store.submission(read_json(store.result_dir(job_id) / 'result.json')['submission'])
                     .get('label', 'reference')}
            if baseline is not None and task.name == 'exponential' and metrics['aggregates']:
                ratios = {case: metrics['aggregates'][case]['time_seconds'] / ref['time_seconds']
                          for case, ref in baseline['metrics']['aggregates'].items()}
                scored = task.score(runtime, metrics, baseline)
                entry.update(
                    score_vs_baseline=scored.get('anchored_score', scored.get('score')),
                    paired_score=scored['score'] if scored.get('scoring') == 'paired' else None,
                    load_factor=scored.get('load_factor'),
                    time_ratio_median=statistics.median(ratios.values()),
                    time_ratio_max=max(ratios.values()),
                    time_ratio_min=min(ratios.values()),
                )
            if baseline is not None and task.name == 'conic' and metrics.get('cases'):
                reference = {c['case_id']: c for c in baseline['metrics']['cases']}
                flipped = [c['case_id'] for c in metrics['cases'] if c['success'] != reference[c['case_id']]['success']]
                ratios = [
                    statistics.median(c['elapsed_seconds']) / statistics.median(reference[c['case_id']]['elapsed_seconds'])
                    for c in metrics['cases']
                    if c['success'] and reference[c['case_id']]['success']
                ]
                entry.update(
                    success_rate=metrics['success_rate'],
                    baseline_success_rate=baseline['metrics']['success_rate'],
                    flipped_cases=flipped,
                    time_ratio_median=statistics.median(ratios) if ratios else None,
                    time_ratio_max=max(ratios) if ratios else None,
                )
            rows.append(entry)
        report['levels'].append({'concurrent_grades': level, 'dev_load_slots': args.dev_load, 'runs': rows})
        say(level=level, dev_load=args.dev_load, runs=rows)
    path = store.root / 'interference' / f'{new_id(task.name)}.json'
    write_json(path, report)
    say(event='interference_report', path=str(path))


# Smoke -------------------------------------------------------------------------------------------------------------


def smoke_checks(store, task, submission_id) -> dict:
    meta = store.submission(submission_id)
    development = meta.get('development', {})
    checks = {
        'development_frozen': meta.get('status') in ('frozen', 'graded'),
        'agent_eligible': development.get('agent_eligible') is True,
        'tool_history': development.get('fixture_tool_history_ok') is True,
    }
    jobs = [j for j in store.jobs('done') + store.jobs('failed') if j['submission'] == submission_id]
    if not jobs:
        checks['graded'] = False
        return checks
    results = store.result_dir(jobs[-1]['id'])
    metrics = read_json(results / 'grade/metrics.json') if (results / 'grade/metrics.json').exists() else {}
    checks['graded'] = jobs[-1].get('status') not in (None, 'unmeasured')
    if task.name == 'exponential':
        checks['numerical_runs'] = metrics.get('valid_runs') == metrics.get('total_runs') == 69
    else:
        checks['numerical_runs'] = metrics.get('total_runs') == 60
        checks['valid_failure_assessment'] = metrics.get('success_rate') == 0
        paths = list((results / 'grade/verifier/cases').glob('*/repeat-*/stdout.log'))
        checks['all_boundary_probes'] = len(paths) == 60 and all(
            len(p := json.loads(path.read_text())['boundary_fixture']) == 6 and all(p.values()) for path in paths
        )
    placement = read_json(results / 'result.json')['placement']
    environment = read_json(results / 'grade/environment.json')
    slot = next(s for s in machine.load()['slots'] if s['id'] == placement['slot'])
    checks['grader_pinned_to_slot_sibling'] = environment['host_affinity'] == [slot['host_cpu']]
    return checks


def cmd_smoke(args) -> None:
    tasks = ['exponential', 'conic'] if args.task == 'both' else [args.task]
    args.smoke, args.model, args.key_file, args.no_grade, args.wait = True, MODELS[0], None, False, None
    args.base_url = 'http://127.0.0.1/v1'
    store = Store(args.store)
    developed = []
    with ThreadPoolExecutor(len(tasks)) as pool:
        for task_name, rows in zip(
            tasks,
            pool.map(lambda name: cmd_develop(argparse.Namespace(**{**vars(args), 'task': name, 'count': args.parallel})),
                     tasks),
        ):
            developed += [(task_name, row) for row in rows]
    jobs = [row['job'] for _, row in developed if row['job']]
    args.drain, args.task = True, tasks
    run_workers(args, job_ids=jobs)
    report = {}
    for task_name, row in developed:
        checks = smoke_checks(store, TASKS[task_name], row['submission'])
        report[row['submission']] = {'task': task_name, 'passed': all(checks.values()), 'checks': checks}
    passed = all(entry['passed'] for entry in report.values())
    path = store.root / 'smoke' / f'{new_id("smoke")}.json'
    write_json(path, {'passed': passed, 'real_model_calls': 0, 'submissions': report})
    say(event='smoke_finished', passed=passed, report=str(path))
    if not passed:
        raise SystemExit(1)


# Reporting ---------------------------------------------------------------------------------------------------------


def cmd_status(args) -> None:
    layout = machine.load()
    store = Store(args.store)
    host = machine.status(layout)
    leases = Broker(layout).leases()
    print(json.dumps({
        'host_ready': host['ready'],
        'checks': host['checks'],
        'slots': host['slots'],
        'leased': [{k: r.get(k) for k in ('slot', 'owner', 'memory_bytes')} for r in leases],
        'memory_reserved_gib': round(sum(r.get('memory_bytes', 0) for r in leases) / 2**30, 1),
        'memory_budget_gib': host['memory_budget_gib'],
        'queue': {state: len(store.jobs(state)) for state in ('pending', 'running', 'done', 'failed')},
        'baselines': {name: (store.active_baseline(name) or {}).get('id') for name in TASKS},
    }, indent=2, ensure_ascii=False))


def cmd_leaderboard(args) -> None:
    store = Store(args.store)
    rows = []
    for meta in store.submissions(args.task):
        if meta.get('origin') == 'reference' or meta.get('smoke'):
            continue
        rows.append({k: meta.get(k) for k in ('id', 'model', 'status', 'score', 'score_status', 'created_utc')})
    rows.sort(key=lambda r: (r['score'] is None, -(r['score'] or 0)))
    baseline = store.active_baseline(args.task)
    print(json.dumps({'task': args.task, 'baseline': baseline and baseline['id'], 'submissions': rows},
                     indent=2, ensure_ascii=False))


def cmd_report(args) -> None:
    """Markdown summary for acceptance: baselines, sentinels, leaderboards, development and grading evidence."""
    store = Store(args.store)
    lines = [f'# FrontierAM campaign report ({utc()})', '']
    for name, task in TASKS.items():
        active = store.active_baseline(name)
        lines += [f'## {name}', '', f'grading isolation: `{task.grade_isolation}`', '']
        for path in sorted(store.baseline_root(name).glob('*/baseline.json')):
            b = read_json(path)
            metric = b['metrics'].get('success_rate', f'{b["metrics"].get("valid_runs")}/{b["metrics"].get("total_runs")}')
            mark = ' **(active)**' if active and b['id'] == active['id'] else ''
            lines.append(f'- baseline `{b["id"]}`{mark}: qualified={b["qualified"]} ({b["reason"]}), '
                         f'metric={metric}, slot={b["slot"]}')
        sentinels = [j for j in store.jobs('done') if j['task'] == name and j.get('kind') == 'sentinel']
        for j in sentinels:
            lines.append(f'- sentinel `{j["id"]}` (reference re-graded): score={j.get("score")} vs ideal 50.0')
        lines += [
            '',
            '| protocol | model | harness | attempt | development | dev s | HTTP | status | score | grading | detail '
            '| submission |',
            '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |',
        ]
        candidate_jobs = [j for j in store.jobs('done') + store.jobs('failed') if j.get('kind') == 'candidate']
        jobs = {}
        for job in sorted(candidate_jobs, key=lambda j: j['id']):
            if job.get('status') != 'superseded_by_quiet_regrade':
                jobs[job['submission']] = job
        attempts: dict[tuple, int] = {}
        agent_rows = sorted(
            (m for m in store.submissions(name) if m.get('origin') == 'agent' and not m.get('smoke')),
            key=lambda m: (m.get('protocol', 'v4'), m.get('model'), m['created_utc']),
        )
        for meta in agent_rows:
            protocol = meta.get('protocol', 'v4')
            key = (protocol, meta['model'])
            attempts[key] = attempts.get(key, 0) + 1
            conditions = store.root / 'dev-runs' / meta.get('dev_run', '') / 'v6_conditions.json'
            harness = read_json(conditions)['harness'] if conditions.is_file() else 'codex'
            if (store.root / 'dev-runs' / meta.get('dev_run', '') / 'CANCELLED.json').is_file():
                continue
            dev = meta.get('development', {})
            job = jobs.get(meta['id'])
            detail = grading = ''
            if job and (store.result_dir(job['id']) / 'score.json').exists():
                score = read_json(store.result_dir(job['id']) / 'score.json')
                result = read_json(store.result_dir(job['id']) / 'result.json')
                grading = (result.get('isolation') or {}).get('policy', '')
                if 'delta_vs_baseline' in score:
                    detail = f'baseline p={score["baseline_success_rate"]}, delta={score["delta_vs_baseline"]:+.2f}'
                elif score.get('scoring') == 'paired':
                    anchored = score.get('anchored_score')
                    detail = (f'paired; anchored {anchored:.2f}; load factor {score.get("load_factor"):.3f}'
                              if anchored is not None else 'paired')
                elif score.get('reason'):
                    detail = str(score['reason'])
            elif job:
                detail = str(job.get('reason'))
            elif dev.get('task_status') == 'unmeasured':
                agent_result = store.root / 'dev-runs' / meta.get('dev_run', '') / 'task/agent/agent_result.json'
                errors = dev.get('error_categories') or (
                    read_json(agent_result).get('error_categories', []) if agent_result.is_file() else []
                )
                detail = 'development ended abnormally: ' + ', '.join(sorted(set(errors)))
            seconds = dev.get('development_seconds')
            score_text = f'{meta["score"]:.2f}' if isinstance(meta.get('score'), float) else str(meta.get('score'))
            lines.append(
                f'| {protocol} | {meta.get("model")} | {harness} | {attempts[key]} | {dev.get("agent_stop_reason")} | '
                f'{round(seconds) if seconds else ""} | {dev.get("http_attempts")} | {meta.get("status")} | '
                f'{score_text} | {grading} | {detail} | `{meta["id"]}` |'
            )
        lines.append('')
    queue = {state: len(store.jobs(state)) for state in ('pending', 'running', 'done', 'failed')}
    lines += ['## Queue', '', f'{queue}', '']
    failed = store.jobs('failed')
    if failed:
        lines += ['Failed (unmeasured) jobs:', ''] + [f'- `{j["id"]}` {j["task"]}: {j.get("reason")}' for j in failed]
    print('\n'.join(lines))


def cmd_host(args) -> None:
    if args.action == 'plan':
        print(json.dumps(machine.plan(**host_plan_options(args)), indent=2))
    elif args.action == 'setup':
        print(json.dumps(machine.install(machine.plan(**host_plan_options(args))), indent=2))
    else:
        print(json.dumps(machine.status(), indent=2))


def host_plan_options(args) -> dict:
    return {
        'housekeeping_cores': args.housekeeping_cores,
        'grade_cores': args.grade_cores,
        'host_reserve_gib': args.host_reserve_gib,
        'grade_placement': args.grade_placement,
    }


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == 'prepare':
        raise SystemExit(subprocess.call([sys.executable, str(ROOT / 'scripts/prepare.py'), *sys.argv[2:]]))
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--store', default=None, help='store root (default $FRONTIERAM_STORE or ./store)')
    sub = parser.add_subparsers(dest='command', required=True)
    host = sub.add_parser('host')
    host.add_argument('action', choices=['plan', 'setup', 'status'])
    host.add_argument('--housekeeping-cores', type=int, default=2)
    host.add_argument('--grade-cores', type=int, default=4)
    host.add_argument('--host-reserve-gib', type=float, default=6.0)
    host.add_argument('--grade-placement', choices=['packed', 'spread'], default='packed')
    develop = sub.add_parser('develop')
    develop.add_argument('--task', choices=sorted(TASKS), required=True)
    develop.add_argument('--model', choices=MODELS, required=True)
    develop.add_argument('--count', type=int, default=1, help='independent attempts, each on its own dev slot')
    develop.add_argument('--key-file', type=Path, default=os.environ.get('AIHUB_KEY_FILE'))
    develop.add_argument('--base-url', default=DEFAULT_BASE_URL)
    develop.add_argument('--no-grade', action='store_true', help='freeze only; do not queue for grading')
    develop.add_argument('--wait', type=float, default=None, help='seconds to wait for a free dev slot')
    develop.add_argument('--protocol', choices=['v4', 'v5', 'v5.1', 'v6', 'v6.1'], default='v4',
                         help='v4: Responses converted to Chat; v5: native Responses (native_bridge.py)')
    develop.add_argument('--development-seconds', type=float, default=None,
                         help='shorter budget for a validation run; such submissions never enter the leaderboard')
    develop.set_defaults(smoke=False)
    submit = sub.add_parser('submit')
    submit.add_argument('--task', choices=sorted(TASKS), required=True)
    submit.add_argument('--source', required=True)
    submit.add_argument('--label', default=None)
    for name in ('grade-worker', 'grade'):
        p = sub.add_parser(name)
        p.add_argument('--workers', type=int, default=None, help='default: one per grade slot')
        p.add_argument('--task', action='append', choices=sorted(TASKS))
        if name == 'grade':
            p.add_argument('--submission', required=True)
        else:
            p.add_argument('--drain', action='store_true', help='exit when the queue is empty')
    baseline = sub.add_parser('baseline')
    baseline.add_argument('--task', choices=sorted(TASKS), required=True)
    baseline.add_argument('--replace', action='store_true')
    baseline.add_argument('--workers', type=int, default=1)
    sentinel = sub.add_parser('sentinel')
    sentinel.add_argument('--task', choices=sorted(TASKS), required=True)
    sentinel.add_argument('--count', type=int, default=1)
    sentinel.add_argument('--workers', type=int, default=1)
    interference = sub.add_parser('interference')
    interference.add_argument('--task', choices=sorted(TASKS), default='exponential')
    interference.add_argument('--levels', default='1,2,4')
    interference.add_argument('--dev-load', type=int, default=0, help='dev slots running a memory-bandwidth hog')
    interference.add_argument('--candidate', action='append', help='source.tar graded as a probe next to references')
    interference.add_argument('--workers', type=int, default=None)
    smoke = sub.add_parser('smoke')
    smoke.add_argument('--task', choices=['exponential', 'conic', 'both'], default='both')
    smoke.add_argument('--parallel', type=int, default=1, help='concurrent fixture attempts per task')
    smoke.add_argument('--workers', type=int, default=None)
    suite = sub.add_parser('suite', help='queue a sharded full held-out suite assessment')
    suite.add_argument('--task', choices=sorted(TASKS), default='conic')
    target = suite.add_mutually_exclusive_group(required=True)
    target.add_argument('--submission')
    target.add_argument('--reference', action='store_true')
    suite.add_argument('--shards', type=int, default=4)
    suite.add_argument('--isolation', default='shared', choices=['shared', 'grade_pool', 'quiet'])
    suite.add_argument('--slots', help='comma-separated grade slot per shard, e.g. grade-06,grade-14,grade-22,grade-30')
    suite.add_argument('--hold', action='store_true', help='write the shards to jobs/held instead of the queue')
    suite.add_argument('--cases', help='JSON list of case IDs: grade this subset, each case in its full-suite shard')
    suite_report = sub.add_parser('suite-report')
    suite_report.add_argument('--suite', required=True)
    sub.add_parser('status')
    sub.add_parser('report')
    sub.add_parser('recover')
    sub.add_parser('rescore')
    board = sub.add_parser('leaderboard')
    board.add_argument('--task', choices=sorted(TASKS), required=True)
    args = parser.parse_args()
    handlers = {
        'host': cmd_host,
        'develop': cmd_develop,
        'submit': cmd_submit,
        'grade-worker': lambda a: run_workers(a),
        'grade': cmd_grade,
        'baseline': cmd_baseline,
        'sentinel': cmd_sentinel,
        'interference': cmd_interference,
        'smoke': cmd_smoke,
        'status': cmd_status,
        'report': cmd_report,
        'recover': lambda a: say(requeued=Store(a.store).recover()),
        'rescore': cmd_rescore,
        'suite': cmd_suite,
        'suite-report': cmd_suite_report,
        'leaderboard': cmd_leaderboard,
    }
    handlers[args.command](args)


if __name__ == '__main__':
    main()
