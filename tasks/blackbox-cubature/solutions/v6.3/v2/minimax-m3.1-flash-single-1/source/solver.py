#!/usr/bin/env python3
"""Black-box cubature over [0,1]^d with a fixed function-evaluation budget.

Protocol: JSON Lines on stdin/stdout.  The program asks for function values in
batches (at most 512 points per message), then reports a single finite estimate.

Method
------
A scrambled Sobol' net supplies the whole point budget.  Several estimators are
formed from that *same* point set (so model selection costs no extra function
evaluations):

  * the plain quasi-Monte Carlo mean (always available, acts as a floor);
  * a Legendre polynomial-chaos / control-variate estimator whose truncation
    degree is chosen by a half-split consistency criterion;
  * a kernel-quadrature estimator (Nystrom) with a bandwidth selected the same
    way.

The half-split criterion compares how well a method reproduces an integral
estimate from two disjoint halves of the sample; the method with the smallest
split discrepancy is refitted on the full sample.
"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from solver_core import (  # noqa: E402
    sobol_points, select_and_estimate, kq_weights, median_heuristic_h,
)

MAX_BATCH = 512


def _readline():
    line = sys.stdin.readline()
    if not line:
        raise SystemExit('evaluator closed stdin')
    return json.loads(line)


def _query(points):
    sys.stdout.write(json.dumps({'type': 'query', 'points': points.tolist()}) + '\n')
    sys.stdout.flush()
    msg = _readline()
    if msg.get('type') != 'values':
        raise SystemExit('unexpected reply: %r' % (msg,))
    vals = np.asarray(msg.get('values', []), dtype=float)
    if vals.size != points.shape[0] or not np.all(np.isfinite(vals)):
        raise SystemExit('invalid values payload')
    return vals


def _estimate(X, y, d, seed):
    """Return a finite estimate, degrading gracefully."""
    fallback = float(np.mean(y))
    try:
        est, info = select_and_estimate(X, y, d, seed)
        if est is not None and math.isfinite(est):
            return est, info
    except Exception:
        pass
    try:
        est, info = _kernel_estimate(X, y, d, seed)
        if est is not None and math.isfinite(est):
            return est, info
    except Exception:
        pass
    return fallback, {'method': 'mean-fallback'}


def _kernel_estimate(X, y, d, seed):
    """Nystrom kernel quadrature with a median-heuristic bandwidth."""
    rng = np.random.default_rng(seed + 4242)
    h = median_heuristic_h(X, d, rng)
    best = None
    for fac in (0.7, 1.0, 1.6, 2.6):
        try:
            w = kq_weights(X, X, h * fac, m=min(384, X.shape[0]))
            est = float(np.dot(w, y))
        except Exception:
            continue
        if not math.isfinite(est):
            continue
        # self-consistency: quadrature of the constant function must be ~1
        score = abs(float(np.sum(w)) - 1.0)
        if best is None or score < best[0]:
            best = (score, est)
    if best is None:
        raise RuntimeError('kernel quadrature failed')
    return best[1], {'method': 'kq'}


def main():
    init = _readline()
    if init.get('type') != 'init':
        raise SystemExit('expected init message')
    d = int(init['dimension'])
    budget = int(init.get('budget', 2048))
    seed = int(init.get('seed', 0))
    bounds = init.get('bounds') or [[0.0, 1.0]] * d
    lo = np.asarray([float(b[0]) for b in bounds], dtype=float)
    hi = np.asarray([float(b[1]) for b in bounds], dtype=float)

    n = int(min(budget, 2048))
    if d < 1:
        raise SystemExit('bad dimension')
    n = max(1, n)

    X = sobol_points(d, n, seed)
    X = lo + X * (hi - lo)
    X = np.clip(X, lo, hi)

    y = np.empty(0)
    chunks = []
    for start in range(0, n, MAX_BATCH):
        chunks.append(X[start:start + MAX_BATCH])
    for chunk in chunks:
        y = np.concatenate([y, _query(chunk)]) if y.size else _query(chunk)

    est, info = _estimate(X, y, d, seed)
    if not math.isfinite(est):
        est = float(np.mean(y))
    sys.stdout.write(json.dumps({'type': 'result', 'estimate': float(est)}) + '\n')
    sys.stdout.flush()


if __name__ == '__main__':
    main()
