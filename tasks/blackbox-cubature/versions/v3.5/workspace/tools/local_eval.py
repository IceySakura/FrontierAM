#!/usr/bin/env python3
"""Run a submission on the sample instances under the formal protocol and limits, and score it like the final score.

    python3 /workspace/tools/local_eval.py [--submission /workspace/submission] [--only ID_SUBSTRING ...]

Each instance runs ./run.sh in a fresh process with the same init message, budget (16384 points), batch limit (4096),
coordinate bound (|z_i| <= 40) and wall limit (30 s) as the formal evaluation; queries go to the sample functions
outside this container. Reports per instance: estimate, log10 relative error e (clipped to [e_floor, 0]), the gain over
the reference GPCA-RQMC on the same instance, queries, seconds, and the error-bound check; then the sample score with
the final mapping (references measured once on the samples, in /workspace/samples/samples.json). Differences from the
formal run: this container is not the grading sandbox (it is shared with your tools), and the samples are few, so the
sample score is only indicative.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import selectors
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from oracle_client import SampleOracle  # noqa: E402

BUDGET, MAX_BATCH, BOUND, LIMIT, CLIP = 2 ** 14, 4096, 40.0, 30.0, 2.0
SAMPLES = Path(os.environ.get('CUBATURE_SAMPLES', '/workspace/samples/samples.json'))


def run_instance(submission: Path, inst: dict, limit: float) -> dict:
    f = SampleOracle(inst['id'])
    d = inst['dimension']
    proc = subprocess.Popen(['./run.sh'], cwd=submission, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=None, bufsize=0)
    sel = selectors.DefaultSelector()
    sel.register(proc.stdout, selectors.EVENT_READ)
    start, used, pending, result = time.monotonic(), 0, b'', None
    row = {'id': inst['id']}
    try:
        init = {'type': 'init', 'dimension': d, 'budget': BUDGET, 'max_batch': MAX_BATCH, 'seed': inst['seed'],
                'time_limit_s': limit}
        proc.stdin.write((json.dumps(init) + '\n').encode())
        proc.stdin.flush()
        while result is None:
            left = limit - (time.monotonic() - start)
            if left <= 0:
                raise TimeoutError(f'wall limit {limit:g} s')
            if not sel.select(min(0.5, left)):
                if proc.poll() is not None:
                    raise RuntimeError('program exited before its result')
                continue
            chunk = os.read(proc.stdout.fileno(), 1 << 20)
            if not chunk:
                raise RuntimeError('program closed stdout before its result')
            pending += chunk
            while b'\n' in pending and result is None:
                raw, pending = pending.split(b'\n', 1)
                if not raw.strip():
                    continue
                msg = json.loads(raw)
                if msg.get('type') == 'query':
                    pts = np.asarray(msg['points'], dtype=float)
                    if pts.ndim != 2 or pts.shape[1] != d or not 0 < len(pts) <= MAX_BATCH:
                        raise ValueError(f'malformed query: expected 1 to {MAX_BATCH} rows of {d} coordinates')
                    if used + len(pts) > BUDGET:
                        raise ValueError(f'budget exceeded: {used} + {len(pts)} > {BUDGET}')
                    if not np.all(np.isfinite(pts)) or np.any(np.abs(pts) > BOUND):
                        raise ValueError(f'point outside the domain (|z_i| <= {BOUND:g})')
                    vals = f(pts)
                    used += len(pts)
                    proc.stdin.write((json.dumps({'type': 'values', 'values': vals.tolist()}) + '\n').encode())
                    proc.stdin.flush()
                elif msg.get('type') == 'result':
                    result = msg
                else:
                    raise ValueError(f'unknown message type {msg.get("type")!r}')
        est = float(result['estimate'])
        if not math.isfinite(est):
            raise ValueError('non-finite estimate')
        truth = inst['reference_value']
        err = abs(est - truth)
        raw_e = math.log10(max(err / abs(truth), 1e-300))
        row.update(ok=True, estimate=est, e=min(0.0, max(inst['e_floor'], raw_e)), raw_e=raw_e, queries=used,
                   seconds=time.monotonic() - start)
        bound = result.get('error_bound')
        if bound is not None:
            row.update(bound=float(bound), confidence=result.get('confidence'), covered=err <= float(bound))
    except Exception as exc:
        row.update(ok=False, error=f'{type(exc).__name__}: {exc}', queries=used, seconds=time.monotonic() - start)
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()
    return row


def mean_gain(rows: list[dict], insts: dict, key: str) -> dict:
    cells = defaultdict(list)
    for r in rows:
        i = insts[r['id']]
        ref = min(0.0, max(i['e_floor'], i[key]))
        cells[(i['track'], i['family'], i['cell'])].append(min(CLIP, max(-CLIP, ref - r['e'])))
    fams = defaultdict(list)
    for (t, fam, _), v in cells.items():
        fams[(t, fam)].append(np.mean(v))
    tracks = defaultdict(list)
    for (t, fam), v in fams.items():
        tracks[t].append(np.mean(v))
    per_track = {t: float(np.mean(v)) for t, v in tracks.items()}
    return {'G': float(np.mean(list(per_track.values()))), 'tracks': per_track}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--submission', type=Path, default=Path('/workspace/submission'))
    ap.add_argument('--only', nargs='*', help='run only instances whose ID contains one of these substrings')
    ap.add_argument('--limit', type=float, default=LIMIT)
    args = ap.parse_args()
    data = json.loads(SAMPLES.read_text())
    insts = {i['id']: i for i in data['instances']}
    chosen = [i for i in data['instances'] if not args.only or any(s in i['id'] for s in args.only)]
    rows = []
    for inst in chosen:
        row = run_instance(args.submission.resolve(), inst, args.limit)
        rows.append(row)
        if row['ok']:
            gain = min(CLIP, max(-CLIP, min(0.0, max(inst['e_floor'], inst['e_reference'])) - row['e']))
            extra = f"  bound {'ok' if row['covered'] else 'MISSED'}" if 'bound' in row else ''
            print(f"{inst['id']:38s} e={row['e']:7.2f} (floor {inst['e_floor']:6.2f}) gain {gain:+5.2f}  "
                  f"{row['queries']:5d} q  {row['seconds']:5.1f} s{extra}", flush=True)
        else:
            print(f"{inst['id']:38s} FAILED: {row['error']}", flush=True)
    if any(not r['ok'] for r in rows):
        print('\nA failed instance makes the whole formal submission score 0.')
        return
    mine, mc = mean_gain(rows, insts, 'e_reference'), mean_gain([{**r, 'e': insts[r['id']]['e_mc']} for r in rows], insts, 'e_reference')
    g, g0 = mine['G'], mc['G']
    score = min(100.0, 50 + 25 * g) if g >= 0 else max(0.0, 50 * (g - g0) / -g0) if g0 < 0 else float('nan')
    print(f"\nG = {g:+.3f} (tracks: {', '.join(f'{t} {v:+.3f}' for t, v in sorted(mine['tracks'].items()))}); "
          f"Monte Carlo G0 = {g0:+.3f}")
    print(f'sample score {score:.1f} / 100 (reference = 50; indicative only, {len(rows)} instances)')


if __name__ == '__main__':
    main()
