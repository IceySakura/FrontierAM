"""Trusted Linux container supervisor; invoke with Python's isolated (-I) mode."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import stat
import subprocess
import sys
import time


AUDIT = Path('/audit')
SUBMISSION = Path('/submission')
INPUT = Path('/input/instance.npz')
OUTPUT = Path('/output/result.npz')
CANDIDATE_UID = 1000
MAX_OUTPUT_BYTES = 1024 ** 3
MAX_LOG_BYTES = 4 * 1024 ** 2  # Each stream; combined limit is 8 MiB.
POLL_SECONDS = 0.01
THREAD_ENV = {name: '1' for name in (
    'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
    'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS', 'RAYON_NUM_THREADS')}
os.environ.update(THREAD_ENV)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_report(name, report):
    temporary = AUDIT / (name + '.tmp')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    temporary.replace(AUDIT / name)


def cgroup_memory():
    """cgroup v2 memory measures include the whole container, not just RSS."""
    base = Path('/sys/fs/cgroup')
    result = {}
    for name in ('memory.current', 'memory.peak', 'memory.events'):
        try:
            text = (base / name).read_text().strip()
            result[name] = ({k: int(v) for k, v in (line.split() for line in text.splitlines())}
                            if name == 'memory.events' else int(text))
        except (OSError, ValueError):
            result[name] = None
    return result


def candidate_processes():
    """Scan every UID-1000 process, including children that detached with setsid."""
    processes = {}
    for path in Path('/proc').iterdir():
        if not path.name.isdecimal():
            continue
        try:
            fields = dict(line.split(':', 1) for line in (path / 'status').read_text().splitlines())
            if CANDIDATE_UID not in map(int, fields['Uid'].split()):
                continue
            if fields['State'].strip().startswith(('Z', 'X')):
                continue
            processes[int(path.name)] = int(fields.get('VmRSS', '0 kB').split()[0]) * 1024
        except (FileNotFoundError, ProcessLookupError):
            continue
    return processes


def reap_orphans():
    while True:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return
        if pid == 0:
            return


def kill_candidates(process):
    """Kill the session and all candidate UIDs; the container is a final boundary."""
    if process is not None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    deadline = time.perf_counter() + 2.0
    while True:
        remaining = candidate_processes()
        for pid in remaining:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        # Never let waitpid(-1) steal Popen's child and lose its actual exit code.
        if process is None or process.poll() is not None:
            reap_orphans()
        if not remaining:
            return True
        if time.perf_counter() >= deadline:
            return False
        time.sleep(POLL_SECONDS)


def child_limits():
    resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_OUTPUT_BYTES, MAX_OUTPUT_BYTES))


def executable(path):
    return path.is_file() and os.access(path, os.X_OK)


def save_candidate_output(report):
    """Copy only a bounded, stable regular file after every candidate has stopped."""
    try:
        info = OUTPUT.lstat()
    except FileNotFoundError:
        report['output_present'] = False
        return
    report['output_present'] = True
    if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_OUTPUT_BYTES:
        report.update(process_status='process_error', reason='invalid_output')
        return
    fd = os.open(OUTPUT, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as source:
        opened = os.fstat(source.fileno())
        if (not stat.S_ISREG(opened.st_mode) or opened.st_size > MAX_OUTPUT_BYTES
                or (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino)):
            report.update(process_status='process_error', reason='invalid_output')
            return
        digest = hashlib.sha256()
        copied = 0
        with (AUDIT / 'solution.npz').open('wb') as destination:
            while chunk := source.read(1024 * 1024):
                copied += len(chunk)
                if copied > MAX_OUTPUT_BYTES:
                    report.update(process_status='process_error', reason='invalid_output')
                    break
                digest.update(chunk)
                destination.write(chunk)
        if copied > MAX_OUTPUT_BYTES:
            (AUDIT / 'solution.npz').unlink()
        else:
            report.update(output_bytes=copied, output_sha256=digest.hexdigest(),
                          output_audit_path='/audit/solution.npz')


def supervise(mode, time_limit):
    command = (['./build.sh'] if mode == 'build' else
               ['./run.sh', '--input', str(INPUT), '--output', str(OUTPUT)])
    report = {'mode': mode, 'command': command, 'cwd': str(SUBMISSION),
              'time_limit_seconds': time_limit, 'started_at': utc_now(),
              'finished_at': None, 'elapsed_seconds': 0.0, 'exit_code': None,
              'process_status': 'entry_error', 'reason': None, 'peak_rss_bytes': 0,
              'kernel_peak_rss_bytes': None, 'max_output_bytes': MAX_OUTPUT_BYTES,
              'max_log_bytes_per_stream': MAX_LOG_BYTES,
              'rss_note': 'Sampled sum of live UID-1000 RSS; shared pages may be counted more than once.',
              'kernel_peak_rss_note': 'Linux RUSAGE_CHILDREN maximum child RSS, not a summed process-tree peak.',
              'cgroup_memory_before': cgroup_memory()}
    process = None
    started = None
    try:
        build_path = SUBMISSION / 'build.sh'
        if mode == 'build' and not (build_path.exists() or build_path.is_symlink()):
            report['command'] = []
            report['reason'] = 'no_build_required' if executable(SUBMISSION / 'run.sh') else 'invalid_run_entry'
            report['process_status'] = 'completed' if executable(SUBMISSION / 'run.sh') else 'entry_error'
            report['exit_code'] = 0 if report['process_status'] == 'completed' else None
            return report
        if not executable(SUBMISSION / command[0].removeprefix('./')):
            report['reason'] = 'missing_or_nonexecutable_entry'
            return report
        environment = {'PATH': '/usr/local/bin:/usr/bin:/bin', 'HOME': '/tmp',
                       'TMPDIR': '/tmp', 'LANG': 'C.UTF-8', 'LC_ALL': 'C.UTF-8',
                       'PYTHONDONTWRITEBYTECODE': '1', **THREAD_ENV}
        with (AUDIT / 'stdout.log').open('wb') as stdout, (AUDIT / 'stderr.log').open('wb') as stderr:
            started = time.perf_counter()
            try:
                process = subprocess.Popen(command, cwd=SUBMISSION, env=environment,
                    stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                    user=CANDIDATE_UID, group=CANDIDATE_UID, extra_groups=[],
                    start_new_session=True, preexec_fn=child_limits)
            except (OSError, subprocess.SubprocessError) as error:
                report.update(reason='entry_start_failed', error=f'{type(error).__name__}: {error}')
                report['elapsed_seconds'] = time.perf_counter() - started
                return report
            while True:
                processes = candidate_processes()
                report['peak_rss_bytes'] = max(report['peak_rss_bytes'], sum(processes.values()))
                exit_code = process.poll()
                elapsed = time.perf_counter() - started
                sizes = [os.fstat(stream.fileno()).st_size for stream in (stdout, stderr)]
                if max(sizes) > MAX_LOG_BYTES or sum(sizes) > 2 * MAX_LOG_BYTES:
                    report.update(process_status='process_error', reason='output_limit')
                    break
                if exit_code is not None:
                    report['elapsed_seconds'] = elapsed
                    report['exit_code'] = exit_code
                    report.update(process_status='completed' if exit_code == 0 else 'process_error',
                                  reason=None if exit_code == 0 else 'nonzero_exit')
                    if candidate_processes():
                        report.update(process_status='process_error', reason='background_processes')
                    break
                if elapsed >= time_limit:
                    report.update(process_status='timeout', reason='time_limit')
                    break
                time.sleep(POLL_SECONDS)
            cleaned = kill_candidates(process)
            if report['exit_code'] is None:
                report['exit_code'] = process.poll()
                report['elapsed_seconds'] = time.perf_counter() - started
            if not cleaned:
                report.update(process_status='evaluation_error', reason='candidate_cleanup_failed')
            for stream in (stdout, stderr):
                stream.truncate(min(os.fstat(stream.fileno()).st_size, MAX_LOG_BYTES))
        if mode == 'build' and report['process_status'] == 'completed' and not executable(SUBMISSION / 'run.sh'):
            report.update(process_status='entry_error', reason='invalid_run_entry_after_build')
        if mode == 'solve' and cleaned:
            save_candidate_output(report)
    except Exception as error:
        report.update(process_status='evaluation_error', reason='supervisor_error',
                      error=f'{type(error).__name__}: {error}')
        if started is not None and not report['elapsed_seconds']:
            report['elapsed_seconds'] = time.perf_counter() - started
    finally:
        if process is not None:
            try:
                kill_candidates(process)
            except Exception as error:
                report.update(process_status='evaluation_error', reason='candidate_cleanup_failed',
                              error=f'{type(error).__name__}: {error}')
        report['finished_at'] = utc_now()
        report['kernel_peak_rss_bytes'] = int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024)
        report['cgroup_memory_after'] = cgroup_memory()
        write_report('process.json', report)
    return report


def check_candidate():
    sys.path.insert(0, '/opt/evaluator')
    import numpy as np
    from conicbench.model import load_problem
    from conicbench.check import check_solution

    report = {'outcome': 'evaluation_error', 'status': None, 'check': None,
              'input_sha256': None, 'result_sha256': None, 'started_at': utc_now()}
    try:
        with INPUT.open('rb') as stream:
            report['input_sha256'] = hashlib.file_digest(stream, 'sha256').hexdigest()
        problem = load_problem(INPUT)
    except Exception as error:
        report['error'] = f'Input: {type(error).__name__}: {error}'
        return report
    try:
        fd = os.open(OUTPUT, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, 'rb') as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_OUTPUT_BYTES:
                raise ValueError('Result must be a regular file no larger than 1 GiB')
            report['result_sha256'] = hashlib.file_digest(stream, 'sha256').hexdigest()
            stream.seek(0)
            with np.load(stream, allow_pickle=False) as data:
                raw = data['metadata_json']
                if raw.ndim != 0 or raw.dtype.kind != 'U':
                    raise ValueError('metadata_json must be a scalar Unicode array')
                metadata = json.loads(str(raw))
                if (not isinstance(metadata, dict) or type(metadata.get('schema_version')) is not int
                        or metadata['schema_version'] != 1):
                    raise ValueError('Result metadata must have schema_version=1')
                json.dumps(metadata, allow_nan=False)
                result = {k: metadata[k] for k in ('status', 'objective_primal', 'objective_dual') if k in metadata}
                result.update({k: data[k] for k in ('x', 's', 'z') if k in data})
        if isinstance(result.get('status'), str):
            report['status'] = result['status']
    except Exception as error:
        report.update(outcome='rejected', error=f'Result: {type(error).__name__}: {error}')
        return report
    try:
        checked = check_solution(problem, result, tol=1e-8)
        report['check'] = checked
        report['outcome'] = ('verified' if checked['passed'] else 'evaluation_error'
                             if checked['reason'] == 'evaluation_error' else 'unresolved'
                             if result.get('status') == 'unresolved' else 'rejected')
    except Exception as error:
        report['error'] = f'Checker: {type(error).__name__}: {error}'
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', required=True, choices=('build', 'solve', 'check'))
    parser.add_argument('--time-limit', required=True, type=float)
    args = parser.parse_args()
    if not math.isfinite(args.time_limit) or args.time_limit <= 0:
        parser.error('--time-limit must be finite and positive')
    AUDIT.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(AUDIT, 0o700)
    if os.geteuid() != 0 or sys.platform != 'linux':
        raise RuntimeError('The trusted worker must run as root in its isolated Linux container')
    if args.mode == 'check':
        started = time.perf_counter()
        try:
            report = check_candidate()
        except Exception as error:
            report = {'outcome': 'evaluation_error', 'status': None,
                      'error': f'Checker setup: {type(error).__name__}: {error}'}
        report.update(finished_at=utc_now(), elapsed_seconds=time.perf_counter() - started,
                      cgroup_memory_after=cgroup_memory())
        write_report('check.json', report)
    else:
        supervise(args.mode, args.time_limit)
    return 0  # The trusted audit records, not this exit code, classify the candidate.


if __name__ == '__main__':
    raise SystemExit(main())
