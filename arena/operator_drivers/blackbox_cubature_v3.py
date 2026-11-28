"""Interactive grading for blackbox-cubature v3 (Gaussian measure, theory + experiment task) and its 100-point score.

The candidate runs in a sandbox container on the grading slot and asks for function values over stdin/stdout; this
process holds the instances (tasks/blackbox-cubature/versions/v3/data/<set>/instances.json) and the oracle
(versions/v3/generation/oracle.py: f(z) = g(Qz), Q a hidden rotation), and enforces the budget, the batch size, the
coordinate bound and the per-instance wall limit. Each instance is a fresh process (docker exec) in one container per
grading job; afterwards every process of the candidate's user is killed. Any failed instance makes the submission
invalid, as in v2.

Protocol (versions/v3/workspace/problem.md): init {dimension, budget, max_batch, seed, time_limit_s}; any number of
query {points} -> values {values}; one result {estimate, error_bound?, confidence?}.

Score: per instance e = log10 relative error clipped to [e_floor, 0], e_floor the instance's reference resolution
(generate.py); gain g = e_reference - e against the locally rerun GPCA-RQMC reference on the same instance, clipped
to [-2, 2]; mean per cell ((family, d) on the hard and curved tracks, (family, configuration) on the option track),
mean over a family's cells, mean over a track's families, mean of the tracks present (two up to v3.5, three from v3.6)
gives G. G >= 0 scores 50 + 25 G (every
instance 100x better than the reference: 100); G < 0 scores 50 (G - G0) / -G0, where G0 is plain Monte Carlo's gain
on the same instances (0 points). Error bounds are diagnostics only (coverage and tightness), never scored.

Usage: blackbox_cubature_v3.py INSTANCES SUBMISSION OUT --image ID --cpuset C --cgroup-parent P [--case-limit S]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import secrets
import selectors
import shutil
import subprocess
import sys
import tempfile
import time

from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np

VERSION = Path(__file__).resolve().parents[2] / 'tasks/blackbox-cubature/versions/v3'
BUDGET = 2 ** 14
MAX_BATCH = 4096
BOUND = 40.0  # |z_i| <= BOUND: beyond it the Gaussian density underflows to zero in double precision
MEMORY = 2 * 2**30
UID = 1000
CLIP = 2.0
LINE = 64 * 2**20  # longest accepted protocol line (a full batch at d = 64 is about 6 MB)


def load_oracle():
    sys.path.insert(0, str(VERSION / 'generation'))
    import oracle  # noqa: E402

    return oracle.make


def cell(instance: dict) -> str:
    return f'L{instance["L"]}-m{instance["m"]}' if instance['track'] == 'options' else f'd{instance["d"]}'


@contextmanager
def sandbox(docker_host: str, image: str, cpuset: str, cgroup_parent: str, submission: Path, scratch: Path):
    """One container per grading job. The submission is a private copy owned by root and mounted read-only, so
    instances cannot pass information through it; /tmp and the home directory are wiped between instances (reset)."""
    work = Path(tempfile.mkdtemp(prefix='cubature-v3-', dir=scratch))
    work.chmod(0o755)
    copy = work / 'submission'
    shutil.copytree(submission, copy, symlinks=True)
    for base, dirs, files in os.walk(copy):
        for name in [base, *(os.path.join(base, n) for n in dirs + files)]:
            os.lchown(name, 0, 0)
            if not os.path.islink(name):
                mode = os.stat(name).st_mode
                os.chmod(name, (mode | 0o444 | (0o111 if mode & 0o111 or os.path.isdir(name) else 0)) & ~0o222)
    name = f'frontieram-cubature-v3-{secrets.token_hex(6)}'
    docker = ['docker', '-H', docker_host]
    try:
        subprocess.run([*docker, 'run', '-d', '--name', name, '--network', 'none', '--cpus', '1', '--cpuset-cpus', cpuset,
                        '--cgroup-parent', cgroup_parent, '--memory', str(MEMORY), '--memory-swap', str(MEMORY),
                        '--pids-limit', '256', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges:true',
                        '--user', f'{UID}:{UID}', '--log-driver', 'none', '--read-only',
                        '--tmpfs', '/tmp:rw,exec,nosuid,nodev,size=512m,mode=1777',
                        '--tmpfs', f'/home/worker:rw,exec,nosuid,nodev,size=256m,uid={UID},gid={UID}',
                        '--mount', f'type=bind,src={copy},dst=/workspace/submission,readonly',
                        '--workdir', '/workspace/submission', '--entrypoint', 'sleep', image, 'infinity'],
                       check=True, capture_output=True, timeout=120)
        yield docker, name
    finally:
        subprocess.run([*docker, 'rm', '-f', name], capture_output=True, timeout=120)
        shutil.rmtree(work, ignore_errors=True)


def reset(docker: list, name: str) -> None:
    # SIGKILL to every process of the candidate user (the container's init, sleep, is exempt), then wipe the writable
    # directories so the next instance starts clean.
    exec_ = [*docker, 'exec', '-u', f'{UID}:{UID}', name, 'sh', '-c']
    subprocess.run([*exec_, 'kill -KILL -1 2>/dev/null; true'], capture_output=True, timeout=60)
    subprocess.run([*exec_, 'find /tmp /home/worker -mindepth 1 -delete 2>/dev/null; true'], capture_output=True,
                   timeout=60)


class OracleFault(RuntimeError):
    """The oracle itself failed (an evaluator defect, never the candidate's fault)."""


def check_points(message: dict, d: int, queried: int) -> np.ndarray:
    try:
        points = np.asarray(message.get('points', []), dtype=float)
    except (TypeError, ValueError):
        raise ValueError('malformed query: points must be a list of numeric rows') from None
    if points.ndim != 2 or points.shape[1] != d or not 0 < len(points) <= MAX_BATCH:
        raise ValueError(f'malformed query: expected 1 to {MAX_BATCH} rows of {d} coordinates')
    if queried + len(points) > BUDGET:
        raise ValueError(f'budget exceeded: {queried} + {len(points)} > {BUDGET}')
    if not np.all(np.isfinite(points)) or np.any(np.abs(points) > BOUND):
        raise ValueError(f'point outside the domain: coordinates must be finite with |z_i| <= {BOUND:g}')
    return points


def check_result(message: dict) -> dict:
    estimate = message.get('estimate')
    if isinstance(estimate, bool) or not isinstance(estimate, (int, float)) or not math.isfinite(estimate):
        raise ValueError('result: estimate must be a finite number')
    out = {'estimate': float(estimate), 'error_bound': None, 'confidence': None}
    bound = message.get('error_bound')
    if bound is not None:
        if isinstance(bound, bool) or not isinstance(bound, (int, float)) or not math.isfinite(bound) or bound <= 0:
            raise ValueError('result: error_bound must be a positive finite number when given')
        confidence = message.get('confidence')
        if not (confidence == 'deterministic' or (isinstance(confidence, (int, float)) and not isinstance(confidence, bool)
                                                   and 0 < confidence < 1)):
            raise ValueError('result: confidence must be "deterministic" or a number in (0, 1) when error_bound is given')
        out.update(error_bound=float(bound), confidence=confidence)
    return out


def run_case(docker: list, name: str, instance: dict, make, timeout: float) -> dict:
    d = int(instance['d'])
    f = make(instance['spec'])
    truth, floor = float(instance['reference']['value']), float(instance['reference']['e_floor'])
    row = {'id': instance['id'], 'track': instance['track'], 'family': instance['family'], 'cell': cell(instance),
           'dimension': d, 'e_floor': floor}
    proc = subprocess.Popen([*docker, 'exec', '-i', '-u', f'{UID}:{UID}', '-w', '/workspace/submission', name,
                             './run.sh'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            bufsize=0)
    started, queried, result, pending = time.monotonic(), 0, None, b''
    selector = selectors.DefaultSelector()
    selector.register(proc.stdout, selectors.EVENT_READ)
    try:
        init = {'type': 'init', 'dimension': d, 'budget': BUDGET, 'max_batch': MAX_BATCH, 'seed': int(instance['seed']),
                'time_limit_s': timeout}
        proc.stdin.write((json.dumps(init) + '\n').encode())
        proc.stdin.flush()
        while result is None:
            remaining = timeout - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError(f'wall limit {timeout:g} s')
            if not selector.select(min(0.5, remaining)):
                if proc.poll() is not None:
                    raise RuntimeError('program exited before its result')
                continue
            chunk = os.read(proc.stdout.fileno(), 1 << 20)
            if not chunk:
                raise RuntimeError('program closed stdout before its result')
            pending += chunk
            if len(pending) > LINE and b'\n' not in pending:
                raise ValueError('protocol line too long')
            while b'\n' in pending and result is None:
                raw, pending = pending.split(b'\n', 1)
                if not raw.strip():
                    continue
                try:
                    message = json.loads(raw.decode('utf-8'))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    raise ValueError('stdout line is not a JSON protocol message') from None
                kind = message.get('type') if isinstance(message, dict) else None
                if kind == 'query':
                    points = check_points(message, d, queried)
                    try:
                        values = np.asarray(f(points), dtype=float)
                    except Exception as exc:
                        raise OracleFault(f'oracle failed: {type(exc).__name__}: {exc}') from exc
                    if values.shape != (len(points),) or not np.all(np.isfinite(values)):
                        raise OracleFault('oracle returned non-finite values')
                    queried += len(points)
                    proc.stdin.write((json.dumps({'type': 'values', 'values': values.tolist()}) + '\n').encode())
                    proc.stdin.flush()
                elif kind == 'result':
                    result = check_result(message)
                else:
                    raise ValueError(f'unknown message type {kind!r}')
        elapsed = time.monotonic() - started
        error = abs(result['estimate'] - truth)
        raw_log = math.log10(max(error / abs(truth), 1e-300))
        row.update(ok=True, estimate=result['estimate'], log10_rel_error=raw_log, e=min(0.0, max(floor, raw_log)),
                   queries=queried, elapsed_sec=elapsed, error_bound=result['error_bound'],
                   confidence=result['confidence'])
        if result['error_bound'] is not None:
            row.update(covered=error <= result['error_bound'],
                       bound_ratio=result['error_bound'] / max(error, 1e-300 * abs(truth)))
    except OracleFault as exc:
        row.update(ok=False, oracle_fault=True, error=str(exc), queries=queried, elapsed_sec=time.monotonic() - started)
    except Exception as exc:
        row.update(ok=False, error=f'{type(exc).__name__}: {exc}', queries=queried, elapsed_sec=time.monotonic() - started)
    finally:
        reset(docker, name)
        if proc.poll() is None:
            proc.kill()
        proc.wait()
    return row


def grade(submission: Path, instances: list[dict], make, *, docker_host: str, image: str, cpuset: str,
          cgroup_parent: str, scratch: Path, case_limit: float) -> dict:
    if not os.access(submission / 'run.sh', os.X_OK):
        return {'status': 'submission_failure', 'reason': 'no executable run.sh in the submission', 'rows': []}
    scratch.mkdir(parents=True, exist_ok=True)
    with sandbox(docker_host, image, cpuset, cgroup_parent, submission, scratch) as (docker, name):
        rows = [run_case(docker, name, instance, make, case_limit) for instance in instances]
    faults = [row for row in rows if row.get('oracle_fault')]
    if faults:
        return {'status': 'infrastructure_failure', 'reason': f'oracle fault on {faults[0]["id"]}: {faults[0]["error"]}',
                'rows': rows}
    failed = [row for row in rows if not row['ok']]
    return {'status': 'submission_failure' if failed else 'ok', 'cases': len(rows), 'failures': len(failed),
            'reason': f'{len(failed)} of {len(rows)} instances failed; first {failed[0]["id"]}: {failed[0]["error"]}' if failed else None,
            'max_wall_s': max((row['elapsed_sec'] for row in rows), default=None),
            'bounds': bound_diagnostics(rows), 'rows': rows}


# Score --------------------------------------------------------------------------------------------------------------
def clipped_error(row: dict) -> float:
    return min(0.0, max(float(row['e_floor']), float(row['log10_rel_error'])))


def gain(rows: list[dict], reference: list[dict]) -> dict:
    """Two-track mean of family means of cell means of clipped per-instance gains over the reference (matched by id)."""
    by_id = {row['id']: row for row in reference}
    if {row['id'] for row in rows} - set(by_id):
        raise ValueError('instances without a reference measurement')
    cells: dict[tuple, list[float]] = defaultdict(list)
    for row in rows:
        g = clipped_error(by_id[row['id']]) - clipped_error(row)
        cells[(row['track'], row['family'], row['cell'])].append(min(CLIP, max(-CLIP, g)))
    families: dict[tuple, list[float]] = defaultdict(list)
    for (track, family, _), values in sorted(cells.items()):
        families[(track, family)].append(float(np.mean(values)))
    tracks: dict[str, list[float]] = defaultdict(list)
    per_family = {}
    for (track, family), values in sorted(families.items()):
        per_family[family] = float(np.mean(values))
        tracks[track].append(per_family[family])
    per_track = {track: float(np.mean(values)) for track, values in tracks.items()}
    return {'gain': float(np.mean(list(per_track.values()))), 'tracks': per_track, 'families': per_family,
            'cells': {f'{family}/{c}': float(np.mean(v)) for (_, family, c), v in sorted(cells.items())}}


def score_100(candidate: dict, mc: dict) -> float:
    g, g0 = candidate['gain'], mc['gain']
    if g >= 0:
        return min(100.0, 50.0 + 25.0 * g)
    return max(0.0, 50.0 * (g - g0) / -g0)


def bound_diagnostics(rows: list[dict]) -> dict:
    """Coverage and tightness of the claimed error bounds (not scored)."""
    claimed = [r for r in rows if r.get('ok') and r.get('error_bound') is not None]
    out = {'instances': len(rows), 'with_bound': len(claimed)}
    if claimed:
        out['coverage'] = float(np.mean([r['covered'] for r in claimed]))
        out['median_bound_ratio'] = float(np.median([r['bound_ratio'] for r in claimed]))
        by_level = defaultdict(list)
        for r in claimed:
            by_level[str(r['confidence'])].append(r['covered'])
        out['coverage_by_confidence'] = {k: {'n': len(v), 'coverage': float(np.mean(v))} for k, v in sorted(by_level.items())}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('instances', type=Path)
    parser.add_argument('submission', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--image', required=True)
    parser.add_argument('--cpuset', required=True)
    parser.add_argument('--cgroup-parent', required=True)
    parser.add_argument('--case-limit', type=float, default=30.0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    instances = json.loads(args.instances.read_text())
    try:
        summary = grade(args.submission, instances, load_oracle(), docker_host=os.environ['DOCKER_HOST'],
                        image=args.image, cpuset=args.cpuset, cgroup_parent=args.cgroup_parent,
                        scratch=args.out / 'scratch', case_limit=args.case_limit)
    except Exception as exc:
        summary = {'status': 'infrastructure_failure', 'reason': f'{type(exc).__name__}: {exc}', 'rows': []}
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')


if __name__ == '__main__':
    main()
