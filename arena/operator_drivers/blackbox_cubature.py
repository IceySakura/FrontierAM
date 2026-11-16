"""Interactive grading for blackbox-cubature (the author's v4 protocol).

The candidate runs in a sandbox container on the grading slot and asks for function values over stdin/stdout; this
process is the oracle (the author's value_and_exact from grade_hidden.py) and enforces the budget, batch size, point
bounds and the per-case wall limit. Unlike the author's verifier, the candidate never runs beside the cases or the
grader: they stay on the host.

Each case is a fresh process (docker exec) in one container per grading job; afterwards every process of the
candidate's user is killed, so only files the candidate writes itself carry over between cases, as in the author's
verifier. The metric is the author's: the median over families of each family's median log10 absolute error.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import secrets
import selectors
import shutil
import subprocess
import tempfile
import time

from contextlib import contextmanager
from pathlib import Path

import numpy as np

BUDGET = 2048
MAX_BATCH = 512
CASE_WALL_S = 120
MEMORY = 2 * 2**30
UID = 1000


def load_oracle(grade_hidden: Path):
    spec = importlib.util.spec_from_file_location('cubature_grade_hidden', grade_hidden)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # value_and_exact recomputes the exact moment on every point; at d = 8 that is most of a case's wall time.
    moment, cache = module.linear_moment, {}

    def linear_moment(v, u, power):
        key = (tuple(map(float, v)), tuple(map(float, u)), power)
        if key not in cache:
            cache[key] = moment(v, u, power)
        return cache[key]

    module.linear_moment = linear_moment
    return module.value_and_exact


@contextmanager
def sandbox(docker_host: str, image: str, cpuset: str, cgroup_parent: str, submission: Path, scratch: Path):
    """One container per grading job; the submission is a private writable copy owned by the candidate user."""
    work = Path(tempfile.mkdtemp(prefix='cubature-', dir=scratch))
    copy = work / 'submission'
    shutil.copytree(submission, copy, symlinks=True)
    for base, dirs, files in os.walk(copy):
        for name in [base, *(os.path.join(base, n) for n in dirs + files)]:
            os.lchown(name, UID, UID)
    name = f'frontieram-cubature-{secrets.token_hex(6)}'
    docker = ['docker', '-H', docker_host]
    subprocess.run([*docker, 'run', '-d', '--name', name, '--network', 'none', '--cpus', '1', '--cpuset-cpus', cpuset,
                    '--cgroup-parent', cgroup_parent, '--memory', str(MEMORY), '--memory-swap', str(MEMORY),
                    '--pids-limit', '256', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges:true',
                    '--user', f'{UID}:{UID}', '--log-driver', 'none', '--read-only',
                    '--tmpfs', '/tmp:rw,exec,nosuid,nodev,size=512m,mode=1777',
                    '--tmpfs', f'/home/worker:rw,exec,nosuid,nodev,size=256m,uid={UID},gid={UID}',
                    '--mount', f'type=bind,src={copy},dst=/workspace/submission',
                    '--workdir', '/workspace/submission', '--entrypoint', 'sleep', image, 'infinity'],
                   check=True, capture_output=True, timeout=120)
    try:
        yield docker, name
    finally:
        subprocess.run([*docker, 'rm', '-f', name], capture_output=True, timeout=120)
        shutil.rmtree(work, ignore_errors=True)


def reset(docker: list, name: str) -> None:
    # SIGKILL to every process of the candidate user; the container's init (sleep) ignores it.
    subprocess.run([*docker, 'exec', '-u', f'{UID}:{UID}', name, 'sh', '-c', 'kill -KILL -1 2>/dev/null; true'],
                   capture_output=True, timeout=60)


def run_case(docker: list, name: str, case: dict, oracle, timeout: float = CASE_WALL_S) -> dict:
    d = int(case['dimension'])
    exact = oracle(case, np.zeros(d))[1]
    row = {'suite': case.get('suite'), 'case_id': case['case_id'], 'family': case['family'], 'dimension': d,
           'seed': int(case['seed']), 'exact': exact}
    proc = subprocess.Popen([*docker, 'exec', '-i', '-u', f'{UID}:{UID}', '-w', '/workspace/submission', name,
                             './run.sh'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            bufsize=0)
    started, queried, estimate, pending = time.monotonic(), 0, None, b''
    selector = selectors.DefaultSelector()
    selector.register(proc.stdout, selectors.EVENT_READ)
    try:
        init = {'type': 'init', 'dimension': d, 'budget': BUDGET, 'seed': int(case['seed']), 'bounds': [[0, 1]] * d}
        proc.stdin.write((json.dumps(init) + '\n').encode())
        proc.stdin.flush()
        while estimate is None:
            remaining = timeout - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError('wall timeout')
            if not selector.select(min(0.5, remaining)):
                if proc.poll() is not None:
                    raise RuntimeError('candidate exited before result')
                continue
            chunk = os.read(proc.stdout.fileno(), 65536)
            if not chunk:
                raise RuntimeError('candidate closed stdout before result')
            pending += chunk
            while b'\n' in pending and estimate is None:
                raw, pending = pending.split(b'\n', 1)
                if not raw.strip():
                    continue
                message = json.loads(raw.decode('utf-8'))
                if message.get('type') == 'query':
                    points = np.asarray(message.get('points', []), dtype=float)
                    if points.ndim != 2 or points.shape[1] != d or not 0 < len(points) <= MAX_BATCH:
                        raise ValueError('malformed query')
                    if queried + len(points) > BUDGET or not np.all(np.isfinite(points)) or np.any(points < 0) \
                            or np.any(points > 1):
                        raise ValueError('budget or point-boundary violation')
                    values = [oracle(case, p)[0] for p in points]
                    queried += len(points)
                    proc.stdin.write((json.dumps({'type': 'values', 'values': values}) + '\n').encode())
                    proc.stdin.flush()
                elif message.get('type') == 'result':
                    estimate = float(message.get('estimate'))
                    if not math.isfinite(estimate):
                        raise ValueError('non-finite result')
                else:
                    raise ValueError('unknown message')
        elapsed = time.monotonic() - started
        try:
            proc.wait(timeout=max(1.0, timeout - elapsed))
        except subprocess.TimeoutExpired:
            pass
        error = abs(estimate - exact)
        row.update(estimate=estimate, abs_error=error, relative_error=error / max(abs(exact), 1e-300),
                   log10_abs_error=math.log10(max(error, 1e-300)), queries=queried, ok=True, elapsed_sec=elapsed)
    except Exception as exc:
        row.update(estimate=None, abs_error=float('inf'), relative_error=float('inf'), log10_abs_error=float('inf'),
                   queries=queried, ok=False, error=f'{type(exc).__name__}: {exc}', elapsed_sec=time.monotonic() - started)
    finally:
        reset(docker, name)
        if proc.poll() is None:
            proc.kill()
        proc.wait()
    return row


def metrics(rows: list[dict]) -> dict:
    families: dict[str, list[float]] = {}
    for row in rows:
        families.setdefault(row['family'], []).append(row['log10_abs_error'])
    medians = {family: float(np.median(values)) for family, values in sorted(families.items())}
    failures = sum(1 for row in rows if not row['ok'])
    return {'cases': len(rows), 'failures': failures, 'failure_rate': failures / max(1, len(rows)),
            'family_median_log10_abs_error': medians,
            'family_balanced_median_log10_abs_error': float(np.median(list(medians.values()))) if medians else None,
            'max_elapsed_sec': max((row['elapsed_sec'] for row in rows), default=None)}


def grade(submission: Path, cases: list[dict], oracle, *, docker_host: str, image: str, cpuset: str,
          cgroup_parent: str, scratch: Path) -> dict:
    scratch.mkdir(parents=True, exist_ok=True)
    with sandbox(docker_host, image, cpuset, cgroup_parent, submission, scratch) as (docker, name):
        rows = [run_case(docker, name, case, oracle) for case in cases]
    return {'rows': rows, **metrics(rows)}
