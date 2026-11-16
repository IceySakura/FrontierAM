"""Interactive grading for blackbox-cubature (the author's v4 protocol) and its 100-point score.

The candidate runs in a sandbox container on the grading slot and asks for function values over stdin/stdout; this
process is the oracle (the author's value_and_exact from grade_hidden.py) and enforces the budget, batch size, point
bounds and the per-case wall limit. Unlike the author's verifier, the candidate never runs beside the cases or the
grader: they stay on the host.

Each case is a fresh process (docker exec) in one container per grading job; afterwards every process of the
candidate's user is killed, so only files the candidate writes itself carry over between cases, as in the author's
verifier. Any failed case makes the submission invalid, as in the author's verifier.

Score (results/OPEN_ITEMS.md, pending the author's confirmation): per case e = log10 relative error clipped to
[-15, 0]; gain g = e_reference - e against the locally rerun HumanPortfolio-v1 on the same case, clipped to [-2, 2],
and to [-2, 0] for polynomial_ridge (exactly integrable once the ridge direction is found, so it can only lose
points); mean per (family, dimension) cell, mean over dimensions, mean over families gives G. G >= 0 scores
50 + 50 G / G_max (G_max = 4/3: every fourier and product-peak cell 100x better); G < 0 scores 50 (G - G0) / -G0,
where G0 is plain Monte Carlo's gain on the same cases (0 points).

Usage: blackbox_cubature.py ORACLE CASES SUBMISSION OUT --image ID --cpuset C --cgroup-parent P [--case-limit S]
"""

from __future__ import annotations

import argparse
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

from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np

BUDGET = 2048
MAX_BATCH = 512
MEMORY = 2 * 2**30
UID = 1000

FLOOR = -15.0
CLIP = 2.0
GUARD_FAMILIES = {'polynomial_ridge'}


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


def case_id(case: dict) -> str:
    return f'{case["split"]}/{case["case_id"]}'


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
    try:
        subprocess.run([*docker, 'run', '-d', '--name', name, '--network', 'none', '--cpus', '1', '--cpuset-cpus', cpuset,
                        '--cgroup-parent', cgroup_parent, '--memory', str(MEMORY), '--memory-swap', str(MEMORY),
                        '--pids-limit', '256', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges:true',
                        '--user', f'{UID}:{UID}', '--log-driver', 'none', '--read-only',
                        '--tmpfs', '/tmp:rw,exec,nosuid,nodev,size=512m,mode=1777',
                        '--tmpfs', f'/home/worker:rw,exec,nosuid,nodev,size=256m,uid={UID},gid={UID}',
                        '--mount', f'type=bind,src={copy},dst=/workspace/submission',
                        '--workdir', '/workspace/submission', '--entrypoint', 'sleep', image, 'infinity'],
                       check=True, capture_output=True, timeout=120)
        yield docker, name
    finally:
        subprocess.run([*docker, 'rm', '-f', name], capture_output=True, timeout=120)
        shutil.rmtree(work, ignore_errors=True)


def reset(docker: list, name: str) -> None:
    # SIGKILL to every process of the candidate user; the container's init (sleep) ignores it.
    subprocess.run([*docker, 'exec', '-u', f'{UID}:{UID}', name, 'sh', '-c', 'kill -KILL -1 2>/dev/null; true'],
                   capture_output=True, timeout=60)


def run_case(docker: list, name: str, case: dict, oracle, timeout: float) -> dict:
    d = int(case['dimension'])
    exact = oracle(case, np.zeros(d))[1]
    row = {'id': case_id(case), 'family': case['family'], 'dimension': d}
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
                raise TimeoutError(f'wall limit {timeout:g} s')
            if not selector.select(min(0.5, remaining)):
                if proc.poll() is not None:
                    raise RuntimeError('program exited before its result')
                continue
            chunk = os.read(proc.stdout.fileno(), 65536)
            if not chunk:
                raise RuntimeError('program closed stdout before its result')
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
                    raise ValueError('unknown message type')
        elapsed = time.monotonic() - started
        relative = abs(estimate - exact) / max(abs(exact), 1e-300)
        row.update(ok=True, log10_rel_error=math.log10(max(relative, 1e-300)), queries=queried, elapsed_sec=elapsed)
    except Exception as exc:
        row.update(ok=False, error=f'{type(exc).__name__}: {exc}', queries=queried, elapsed_sec=time.monotonic() - started)
    finally:
        reset(docker, name)
        if proc.poll() is None:
            proc.kill()
        proc.wait()
    return row


def grade(submission: Path, cases: list[dict], oracle, *, docker_host: str, image: str, cpuset: str,
          cgroup_parent: str, scratch: Path, case_limit: float) -> dict:
    if not os.access(submission / 'run.sh', os.X_OK):
        return {'status': 'submission_failure', 'reason': 'no executable run.sh in the submission', 'rows': []}
    scratch.mkdir(parents=True, exist_ok=True)
    with sandbox(docker_host, image, cpuset, cgroup_parent, submission, scratch) as (docker, name):
        rows = [run_case(docker, name, case, oracle, case_limit) for case in cases]
    failed = [row for row in rows if not row['ok']]
    return {'status': 'submission_failure' if failed else 'ok', 'cases': len(rows), 'failures': len(failed),
            'reason': f'{len(failed)} of {len(rows)} cases failed; first {failed[0]["id"]}: {failed[0]["error"]}' if failed else None,
            'max_wall_s': max((row['elapsed_sec'] for row in rows), default=None), 'rows': rows}


# Score --------------------------------------------------------------------------------------------------------------
def clipped_error(row: dict) -> float:
    return min(0.0, max(FLOOR, row['log10_rel_error']))


def gain(rows: list[dict], reference: list[dict]) -> dict:
    """Family-balanced mean of clipped per-case gains over the reference, matched by case id."""
    by_id = {row['id']: row for row in reference}
    if {row['id'] for row in rows} - set(by_id):
        raise ValueError('cases without a reference measurement')
    cells: dict[tuple, list[float]] = defaultdict(list)
    for row in rows:
        high = 0.0 if row['family'] in GUARD_FAMILIES else CLIP
        g = clipped_error(by_id[row['id']]) - clipped_error(row)
        cells[(row['family'], row['dimension'])].append(min(high, max(-CLIP, g)))
    families: dict[str, list[float]] = defaultdict(list)
    for (family, _), values in sorted(cells.items()):
        families[family].append(float(np.mean(values)))
    per_family = {family: float(np.mean(values)) for family, values in families.items()}
    return {'gain': float(np.mean(list(per_family.values()))), 'families': per_family,
            'max_gain': CLIP * sum(f not in GUARD_FAMILIES for f in per_family) / len(per_family)}


def score_100(candidate: dict, mc: dict) -> float:
    g, g0, top = candidate['gain'], mc['gain'], candidate['max_gain']
    if g >= 0:
        return min(100.0, 50.0 + 50.0 * g / top)
    return max(0.0, 50.0 * (g - g0) / -g0)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('oracle', type=Path)
    parser.add_argument('cases', type=Path)
    parser.add_argument('submission', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--image', required=True)
    parser.add_argument('--cpuset', required=True)
    parser.add_argument('--cgroup-parent', required=True)
    parser.add_argument('--case-limit', type=float, default=30.0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cases = json.loads(args.cases.read_text())
    try:
        summary = grade(args.submission, cases, load_oracle(args.oracle), docker_host=os.environ['DOCKER_HOST'],
                        image=args.image, cpuset=args.cpuset, cgroup_parent=args.cgroup_parent,
                        scratch=args.out / 'scratch', case_limit=args.case_limit)
    except Exception as exc:
        summary = {'status': 'infrastructure_failure', 'reason': f'{type(exc).__name__}: {exc}', 'rows': []}
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')


if __name__ == '__main__':
    main()
