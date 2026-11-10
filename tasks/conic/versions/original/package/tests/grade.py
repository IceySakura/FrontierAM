#!/usr/bin/env python3
"""Standalone Harbor verifier; candidate code never runs as the grading user."""
import argparse
import ctypes
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import resource
import shutil
import signal
import stat
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'lib'))
sys.path.insert(0, str(HERE))
from conic_eval import worker
from conic_eval.success_scoring import SCORING_VERSION, aggregate_cases, compute_score

VERSION = 'standard-v1'
MEMORY = 8 * 2**30
LOG = Path('/logs/verifier')
WORK = Path('/run/conic')


class SubmissionError(ValueError):
    pass


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def snapshot(source, destination):
    if source.is_symlink() or not source.is_dir():
        raise SubmissionError('Missing ordinary submission directory')
    destination.mkdir()
    files = []
    for path in sorted(source.rglob('*')):
        info = path.lstat()
        rel = path.relative_to(source)
        target = destination / rel
        if stat.S_ISDIR(info.st_mode):
            target.mkdir(exist_ok=True)
        elif stat.S_ISREG(info.st_mode):
            target.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(fd, 'rb') as src, target.open('xb') as dst:
                opened = os.fstat(src.fileno())
                if (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino):
                    raise SubmissionError('Unstable source file')
                shutil.copyfileobj(src, dst)
            target.chmod((info.st_mode & 0o777) | 0o600)
            files.append({'path': str(rel), 'sha256': digest(target), 'bytes': info.st_size})
        else:
            raise SubmissionError(f'Source contains link or special file: {rel}')
    if not (destination / 'README.md').is_file():
        raise SubmissionError('Submission requires README.md')
    for path in [destination, *destination.rglob('*')]:
        os.chown(path, 1000, 1000, follow_symlinks=False)
    return files


def readonly_source(source):
    for path in [source, *source.rglob('*')]:
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            try:
                target = path.resolve(strict=True)
            except (RuntimeError, OSError) as error:
                raise SubmissionError('Build created an invalid or cyclic link') from error
            if not target.is_relative_to(source.resolve()):
                raise SubmissionError('Build created a link outside submission')
            os.chown(path, 0, 0, follow_symlinks=False)
        elif stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode):
            os.chown(path, 0, 0)
            path.chmod(0o555 if path.is_dir() else 0o444 | (info.st_mode & 0o111))
        else:
            raise SubmissionError('Build created a special file')


def reset_directory(path, uid=1000):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(mode=0o700)
    os.chown(path, uid, uid)


def phase(config):
    """Reuse the frozen timer/poll/exit/output logic in a fresh trusted process."""
    from runtime_sandbox import child_setup
    worker.AUDIT = Path(config['audit'])
    worker.SUBMISSION = Path(config['source'])
    worker.INPUT = Path(config['input'])
    worker.OUTPUT = Path(config['output']) / 'result.npz'
    worker.AUDIT.mkdir(parents=True, mode=0o700, exist_ok=True)
    worker.AUDIT.chmod(0o700)
    os.sched_setaffinity(0, {pinned_cpu()})
    # Reap detached candidate children after their original parent exits.
    if ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0):
        raise OSError(ctypes.get_errno(), 'PR_SET_CHILD_SUBREAPER')
    if config['mode'] == 'check':
        resource.setrlimit(resource.RLIMIT_AS, (4 * 2**30, 4 * 2**30))
        # Preload the packaged checker before the frozen worker's import path hint.
        import conicbench.model, conicbench.check
        write(worker.AUDIT / 'check.json', worker.check_candidate())
        return
    paths = {k: config[k] for k in ('source', 'input', 'tmp', 'output')}
    paths['input'] = config['input'] if config['mode'] == 'solve' else None
    worker.child_limits = lambda: child_setup(paths, config['mode'])
    real_popen = subprocess.Popen

    def candidate_popen(*args, **kwargs):
        kwargs['env'].update(HOME=config['tmp'], TMPDIR=config['tmp'])
        process = real_popen(*args, **kwargs)
        try:
            Path(f'/proc/{process.pid}/oom_score_adj').write_text('1000')
        except FileNotFoundError:
            pass  # The candidate already exited.
        return process

    # Do not edit the frozen supervisor or change its timer start/end positions.
    import types
    worker.subprocess = types.SimpleNamespace(Popen=candidate_popen,
        DEVNULL=subprocess.DEVNULL, SubprocessError=subprocess.SubprocessError)
    scan = worker.candidate_processes
    last_scan, quota_error = [0.0], [None]

    def monitored_processes():
        processes = scan()
        now = time.monotonic()
        if processes and now - last_scan[0] >= 0.1:
            last_scan[0] = now
            for key in ('tmp', 'output'):
                used = 0
                for path in Path(config[key]).rglob('*'):
                    try:
                        info = path.lstat()
                    except FileNotFoundError:
                        continue
                    if stat.S_ISREG(info.st_mode):
                        used += info.st_size
                    if used > 2**30:
                        quota_error[0] = key + '_directory_limit'
                        break
                if quota_error[0]:
                    for pid in processes:
                        try:
                            os.kill(pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    break
        return processes

    worker.candidate_processes = monitored_processes
    report = worker.supervise(config['mode'], config['time_limit'])
    before = report['cgroup_memory_before'].get('memory.events') or {}
    after = report['cgroup_memory_after'].get('memory.events') or {}
    if after.get('oom_kill', 0) > before.get('oom_kill', 0):
        report.update(process_status='process_error', reason='memory_limit')
    if quota_error[0]:
        report.update(process_status='process_error', reason=quota_error[0])
    report['runtime_version'] = VERSION
    if report.get('output_audit_path'):
        report['output_audit_path'] = str(worker.AUDIT / 'solution.npz')
    write(worker.AUDIT / 'process.json', report)


def run_phase(mode, directory, limit):
    directory.mkdir(parents=True, exist_ok=False)
    config = dict(mode=mode, audit=str(directory), source=str(WORK / 'submission'),
        input=str(WORK / 'input/instance.npz'), output=str(WORK / 'output'),
        tmp=str(WORK / 'tmp'), time_limit=limit)
    write(directory / 'phase.json', config)
    with (directory / 'supervisor.log').open('wb') as output:
        proc = subprocess.Popen([sys.executable, '-I', str(__file__), '--phase',
            str(directory / 'phase.json')], stdout=output, stderr=output, start_new_session=True)
        try:
            code = proc.wait(timeout=limit + (0 if mode == 'check' else 45))
        except BaseException:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            worker.kill_candidates(None)
            raise
    path = directory / ('check.json' if mode == 'check' else 'process.json')
    if code or not path.is_file():
        raise RuntimeError(f'{mode} supervisor failed; see {directory}/supervisor.log')
    return json.loads(path.read_text())


def pinned_cpu():
    """The host slot broker leases exactly one CPU; its index is not part of the contract."""
    allowed = os.sched_getaffinity(0)
    if len(allowed) != 1:
        raise RuntimeError('Verifier must be pinned to exactly one CPU; refusing to choose one silently')
    return next(iter(allowed))


def preflight():
    if sys.platform != 'linux' or os.geteuid() != 0:
        raise RuntimeError('Requires a fresh Linux verifier container with root supervisor')
    cpu = pinned_cpu()
    if worker.candidate_processes():
        raise RuntimeError('Verifier already contains candidate UID1000 processes')
    cg = Path('/sys/fs/cgroup')
    maximum = (cg / 'memory.max').read_text().strip()
    quota, period = (cg / 'cpu.max').read_text().split()
    if maximum != str(MEMORY) or quota == 'max' or int(quota) != int(period):
        raise RuntimeError('Verifier must have cgroup v2 8GiB and one CPU quota')
    from runtime_sandbox import child_setup
    # Exercise the actual restrictions before classifying candidate failures.
    proof = WORK / 'sandbox-check'
    proof.mkdir(mode=0o755)
    dummy = WORK / 'input/instance.npz'
    dummy.write_bytes(b'sandbox-preflight')
    dummy.chmod(0o444)
    paths = dict(source=str(proof), input=str(dummy), tmp=str(WORK / 'tmp'), output=str(WORK / 'output'))
    subprocess.run(['/usr/bin/true'], user=1000, group=1000, extra_groups=[],
        preexec_fn=lambda: child_setup(paths, 'solve'), check=True, timeout=10)
    return {'runtime_version': VERSION, 'platform': os.uname()._asdict() if hasattr(os.uname(), '_asdict') else list(os.uname()),
        'cpu_affinity': [cpu], 'cpu_max': [quota, period], 'memory_max': int(maximum),
        'memory_swap_max': (cg / 'memory.swap.max').read_text().strip(),
        'memory_swap_current': (cg / 'memory.swap.current').read_text().strip(),
        'host_swap_total': next(line.split(':',1)[1].strip() for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('SwapTotal:')),
        'memory_scope': 'whole verifier cgroup including supervisor and charged page cache',
        'temporary_storage': 'fresh disk directories, 1GiB sampled quota, not tmpfs',
        'timing': 'unchanged frozen worker before Popen to detected exit, excludes build/check/setup',
        'frozen_worker_sha256': digest(HERE / 'lib/conic_eval/worker.py')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', type=Path)
    args = parser.parse_args()
    if args.phase:
        phase(json.loads(args.phase.read_text()))
        return 0
    LOG.mkdir(parents=True, exist_ok=True)
    LOG.chmod(0o700)
    for name in ('reward.json', 'reward.txt'):
        (LOG / name).unlink(missing_ok=True)
    HERE.chmod(0o700)
    config = json.loads((HERE / 'config.json').read_text())
    mode = config['mode']
    if mode not in ('candidate', 'pilot'):
        raise ValueError('Unknown evaluation mode')
    score_context = dict(formal_score=False,
        score_status='pilot' if mode == 'pilot' else 'provisional')
    try:
        WORK.mkdir(mode=0o755, exist_ok=False)
        for name in ('tmp', 'output'):
            reset_directory(WORK / name)
        (WORK / 'input').mkdir(mode=0o755)
        environment = preflight()
        environment['calibration_fingerprint'] = hashlib.sha256(json.dumps({
            'runtime_version': VERSION, 'memory_max': environment['memory_max'],
            'cpu_max': environment['cpu_max'], 'memory_swap_max': environment['memory_swap_max'],
            'kernel': os.uname().release, 'machine': os.uname().machine,
            'python': sys.version,
            'host_boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'cpu_info_sha256': digest('/proc/cpuinfo'),
            'numerical_libraries': {name: {'version': importlib.metadata.version(name),
                'native_sha256': {str(f): digest(importlib.metadata.distribution(name).locate_file(f))
                    for f in importlib.metadata.distribution(name).files if '.so' in str(f)}}
                for name in ('numpy', 'scipy')},
            'manifest_sha256': digest(HERE / 'private/data/manifest.json'),
            'implementation': {str(p.relative_to(HERE)): digest(p) for p in
                [HERE / 'grade.py', HERE / 'runtime_sandbox.py', *sorted((HERE / 'lib').rglob('*.py'))]}
            }, sort_keys=True).encode()).hexdigest()
        write(LOG / 'runtime.json', environment)
        lock = json.loads((HERE / 'private/package-lock.json').read_text())
        for name, expected in lock['files'].items():
            if digest(HERE / name) != expected['sha256']:
                raise RuntimeError(f'Packaged resource hash mismatch: {name}')
        anchor = json.loads((HERE / 'private/score-anchor.json').read_text())
        manifest = json.loads((HERE / 'private/data/manifest.json').read_text())
        if any(value.get('scoring_version') != SCORING_VERSION for value in (config, anchor, manifest)):
            raise ValueError('Score anchor and manifest must use ' + SCORING_VERSION)
        if anchor.get('evaluation_manifest_sha256') != digest(HERE / 'private/data/manifest.json'):
            raise ValueError('Score anchor does not match evaluation manifest')
        compute_score(0, anchor['p_B'])  # Validate the frozen reference before running candidate code.
        rows = manifest['instances']
        time_limits = {}
        for row in rows:
            limit = row.get('time_limit_seconds')
            if (type(limit) not in (int, float) or not math.isfinite(limit)
                    or not 1 <= limit <= 300 or row['id'] in time_limits):
                raise ValueError('Invalid or duplicate per-instance time budget')
            time_limits[row['id']] = limit
        calibrated = (anchor.get('runtime_recalibrated') is True and
            anchor.get('runtime_version') == VERSION and
            anchor.get('runtime_fingerprint') == environment['calibration_fingerprint'])
        score_context.update(formal_score=mode == 'candidate' and calibrated,
            score_status='pilot' if mode == 'pilot' else 'calibrated' if calibrated else 'provisional',
            scoring_version=SCORING_VERSION,
            score_anchor={'p_B': anchor['p_B'],
                'calibration_status': anchor.get('calibration_status')})
        write(LOG / 'scoring.json', score_context)
        source_files = snapshot(Path('/workspace/submission'), WORK / 'submission')
        write(LOG / 'source.json', source_files)
        build = run_phase('build', LOG / 'build', 1800)
        if build['process_status'] == 'evaluation_error':
            raise RuntimeError('Build supervisor error')
        if build['process_status'] != 'completed':
            raise SubmissionError('Build failed: ' + str(build.get('reason')))
        readonly_source(WORK / 'submission')
        if mode == 'pilot':
            ids = config['case_ids']
            if not ids or len(set(ids)) != len(ids):
                raise ValueError('Pilot requires explicit unique case IDs')
            rows = [row for row in rows if row['id'] in ids]
            if len(rows) != len(ids):
                raise ValueError('Unknown pilot ID')
        elif len(rows) != 650 or config.get('case_ids'):
            raise ValueError('Candidate scoring requires all650 instances')
        cases = {row['id']: dict(case_id=row['id'], group=row['group'], repeats=[]) for row in rows}
        for repeat in (1, 2, 3):
            for row in rows:
                for name in ('tmp', 'output'):
                    reset_directory(WORK / name)
                source = HERE / 'private/data' / row['path']
                input_path = WORK / 'input/instance.npz'
                shutil.copyfile(source, input_path)
                input_path.chmod(0o444)
                directory = LOG / 'cases' / row['id'] / f'repeat-{repeat}'
                limit = time_limits[row['id']]
                measured = run_phase('solve', directory, limit)
                if measured['process_status'] == 'evaluation_error':
                    raise RuntimeError('Solve supervisor failed')
                if measured['process_status'] == 'entry_error':
                    raise SubmissionError('Invalid run entry: ' + str(measured.get('reason')))
                record = dict(repeat=repeat, case_id=row['id'], input_sha256=row['sha256'],
                    outcome='algorithm_failure', status=None, elapsed_seconds=measured['elapsed_seconds'],
                    time_limit_seconds=limit,
                    reason=measured.get('reason'), process=measured)
                if measured['process_status'] == 'completed' and measured['elapsed_seconds'] > limit:
                    record['reason'] = 'time_limit'
                if measured['process_status'] == 'completed' and measured['elapsed_seconds'] <= limit:
                    if (directory / 'solution.npz').is_file():
                        shutil.copyfile(directory / 'solution.npz', WORK / 'output/result.npz')
                        checked = run_phase('check', directory / 'checker', 60)
                        if checked['outcome'] == 'evaluation_error':
                            raise RuntimeError('Independent checker failed: ' + str(checked.get('error')))
                        status = checked.get('status')
                        record.update(checker=checked, output_sha256=digest(directory / 'solution.npz'),
                            status=status if status in ('solved', 'primal_infeasible', 'dual_infeasible', 'unresolved') else None,
                            outcome='accepted' if checked['outcome'] == 'verified' else 'algorithm_failure',
                            reason=(checked.get('check') or {}).get('reason', checked['outcome']))
                    else:
                        record['reason'] = 'missing_output'
                if digest(input_path) != row['sha256']:
                    raise RuntimeError('Input modified')
                write(directory / 'record.json', record)
                cases[row['id']]['repeats'].append(record)
                write(LOG / 'progress.json', dict(case_id=row['id'], repeat=repeat, outcome=record['outcome']))
        summary = aggregate_cases(list(cases.values()), expected_case_ids=[row['id'] for row in rows],
            time_limits=time_limits)
        if not summary['valid_assessment']:
            raise RuntimeError('Incomplete/invalid evaluation records')
        summary.update(runtime_version=VERSION, mode=mode, S=None, **score_context)
        if mode == 'pilot':
            reward = {'pilot_success_rate': summary['p']}
        else:
            summary.update(compute_score(summary['p'], anchor['p_B']))
            reward = {'reward': summary['S'] / 10}
        write(LOG / 'summary.json', summary)
        write(LOG / 'reward.json', reward)
        return 0
    except SubmissionError as error:
        write(LOG / 'summary.json', dict(outcome='submission_error', reason=str(error), S=0, **score_context))
        write(LOG / 'reward.json', {'pilot_success_rate' if mode == 'pilot' else 'reward': 0.0})
        return 0
    except BaseException as error:
        write(LOG / 'error.json', dict(outcome='evaluation_error', reason=f'{type(error).__name__}: {error}', S=None))
        return 2
    finally:
        worker.kill_candidates(None)


if __name__ == '__main__':
    raise SystemExit(main())
