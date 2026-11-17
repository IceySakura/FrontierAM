#!/usr/bin/env python3
"""JSON-Lines oracle client for fixed-budget black-box cubature."""
from __future__ import annotations
import json, math, sys
import numpy as np

from cubature_core import solve


class Oracle:
    def __init__(self, dimension, budget, bounds, stdin, stdout):
        self.d = int(dimension)
        self.budget = int(budget)
        self.used = 0
        self.bounds = bounds
        self.stdin = stdin
        self.stdout = stdout
        self.values = []
        lo = np.array([float(b[0]) for b in bounds], dtype=float)
        hi = np.array([float(b[1]) for b in bounds], dtype=float)
        if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
            lo = np.zeros(self.d); hi = np.ones(self.d)
        self.lo = lo; self.hi = hi
        vol = float(np.prod(np.abs(hi - lo)))
        self.volume = vol if math.isfinite(vol) else 1.0
        self.unit_bounds = bool(np.allclose(lo, 0.0) and np.allclose(hi, 1.0))

    def _send(self, obj):
        self.stdout.write((json.dumps(obj, separators=(',', ':'), allow_nan=False) + '\n').encode('utf-8'))
        self.stdout.flush()

    def _read_values(self, n):
        while True:
            line = self.stdin.readline()
            if not line:
                raise RuntimeError('oracle closed stdin')
            if not line.strip():
                continue
            msg = json.loads(line.decode('utf-8'))
            if msg.get('type') != 'values':
                raise RuntimeError('unexpected response')
            vals = np.asarray(msg.get('values', []), dtype=float)
            if len(vals) != n or not np.all(np.isfinite(vals)):
                raise RuntimeError('invalid values')
            return vals

    def query(self, points):
        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != self.d:
            raise ValueError('wrong point shape')
        n = len(pts)
        if n == 0:
            return np.empty(0, dtype=float)
        pts = np.clip(np.nan_to_num(pts, nan=0.5, posinf=1.0, neginf=0.0), 0.0, 1.0)
        if self.used + n > self.budget:
            raise RuntimeError('point budget exceeded')
        out = []
        for start in range(0, n, 512):
            chunk = pts[start:start + 512]
            if self.unit_bounds:
                real = chunk
            else:
                real = self.lo + (self.hi - self.lo) * chunk
                real = np.clip(real, np.minimum(self.lo, self.hi), np.maximum(self.lo, self.hi))
            self._send({'type': 'query', 'points': real.tolist()})
            vals = self._read_values(len(chunk))
            out.append(vals)
            self.values.extend(vals.tolist())
            self.used += len(chunk)
        return np.concatenate(out)

    def fallback(self):
        if self.values:
            return float(np.mean(self.values)) * self.volume
        return 0.0


def main():
    inp = sys.stdin.buffer
    out = sys.stdout.buffer
    init = json.loads(inp.readline().decode('utf-8'))
    d = int(init.get('dimension', 1))
    budget = int(init.get('budget', 2048))
    seed = int(init.get('seed', 0))
    bounds = init.get('bounds') or [[0.0, 1.0]] * d
    if len(bounds) != d:
        bounds = [[0.0, 1.0]] * d
    oracle = Oracle(d, budget, bounds, inp, out)
    try:
        estimate = solve(oracle.query, d, budget, seed)
        if not isinstance(estimate, (int, float, np.floating)) or not math.isfinite(float(estimate)):
            estimate = oracle.fallback()
        else:
            estimate = float(estimate) * oracle.volume
    except Exception:
        estimate = oracle.fallback()
    if not math.isfinite(estimate):
        estimate = 0.0
    out.write((json.dumps({'type': 'result', 'estimate': estimate}, separators=(',', ':'), allow_nan=False) + '\n').encode('utf-8'))
    out.flush()


if __name__ == '__main__':
    main()
