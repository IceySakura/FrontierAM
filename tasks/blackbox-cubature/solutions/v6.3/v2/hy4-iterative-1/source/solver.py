#!/usr/bin/env python3
"""Black-box cubature solver with a fixed 2048-point oracle budget.

JSON-lines protocol: init -> (query -> values)* -> result.
All estimation logic is in estimator.py; this file only manages I/O, the
point budget, bounds mapping and hard-failure fallbacks.
"""
from __future__ import annotations
import json, math, sys, time
import numpy as np

START = time.perf_counter()
TIME_LIMIT = 24.0


def log(*a):
    # stderr is not part of the protocol; keep it quiet in production
    pass


class Oracle:
    def __init__(self, d, budget, seed, bounds):
        self.d = int(d)
        self.budget = int(budget)
        self.seed = int(seed)
        self.lo = np.asarray([b[0] for b in bounds], float)
        self.hi = np.asarray([b[1] for b in bounds], float)
        self.volume = float(np.prod(self.hi - self.lo))
        self.used = 0
        self.all_x = []
        self.all_y = []
        self.qmc_x = []
        self.qmc_y = []

    def note_qmc(self, P, y):
        self.qmc_x.append(np.asarray(P).copy())
        self.qmc_y.append(np.asarray(y).copy())

    @property
    def remaining(self):
        return self.budget - self.used

    def out_of_time(self):
        return time.perf_counter() - START > TIME_LIMIT

    def values(self, P):
        """P: (m,d) array in the unit cube. Returns (m,) values."""
        P = np.asarray(P, float)
        if P.ndim != 2 or P.shape[1] != self.d:
            raise ValueError('bad point shape')
        m = len(P)
        if m == 0 or m > self.remaining:
            raise ValueError('bad batch size or budget')
        if not np.all(np.isfinite(P)) or np.any(P < 0) or np.any(P > 1):
            raise ValueError('non-finite or out-of-range points')
        X = self.lo + P * (self.hi - self.lo)
        out = np.empty(m)
        for s in range(0, m, 512):
            chunk = X[s:s + 512]
            print(json.dumps({'type': 'query', 'points': chunk.tolist()},
                             separators=(',', ':')), flush=True)
            line = sys.stdin.readline()
            if not line:
                raise RuntimeError('oracle closed stdin')
            ans = json.loads(line)
            v = np.asarray(ans.get('values', []), dtype=float)
            if v.shape != (len(chunk),):
                raise RuntimeError('bad values shape')
            out[s:s + 512] = v
        self.used += m
        self.all_x.append(P.copy())
        self.all_y.append(out.copy())
        if not np.all(np.isfinite(out)):
            out = np.where(np.isfinite(out), out, 0.0)
        return out

    def fallback(self):
        ya = self.qmc_y if self.qmc_y else self.all_y
        if not ya:
            return 0.0
        y = np.concatenate(ya)
        y = y[np.isfinite(y)]
        if not len(y):
            return 0.0
        e = float(np.mean(y)) * self.volume
        return e if math.isfinite(e) else 0.0


def main():
    line = sys.stdin.readline()
    if not line:
        print(json.dumps({'type': 'result', 'estimate': 0.0}), flush=True)
        return
    init = json.loads(line)
    d = int(init['dimension'])
    budget = int(init['budget'])
    seed = int(init.get('seed', 0))
    bounds = init.get('bounds', [[0.0, 1.0]] * d)
    if len(bounds) != d:
        bounds = [[0.0, 1.0]] * d
    oracle = Oracle(d, budget, seed, bounds)
    estimate = None
    try:
        import estimator
        estimate = estimator.solve(oracle)
        estimate = float(estimate)
        if not math.isfinite(estimate):
            estimate = None
    except Exception as exc:
        log('solver error', repr(exc))
        estimate = None
    if estimate is None or not math.isfinite(estimate):
        estimate = oracle.fallback()
    if not math.isfinite(estimate):
        estimate = 0.0
    print(json.dumps({'type': 'result', 'estimate': float(estimate)}), flush=True)


if __name__ == '__main__':
    main()
