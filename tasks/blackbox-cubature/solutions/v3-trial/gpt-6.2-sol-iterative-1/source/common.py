"""Protocol and shared numerical primitives."""
import json
import math
import os
import sys
import time

import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc


class Session:
    def __init__(self):
        self.started = time.monotonic()
        init = json.loads(sys.stdin.readline())
        if init["type"] != "init":
            raise ValueError("expected init")
        self.d = int(init["dimension"])
        self.budget = int(init["budget"])
        self.max_batch = int(init["max_batch"])
        self.seed = int(init["seed"])
        self.limit = float(init.get("time_limit_s", 30.0))
        self.used = 0
        self.rng = np.random.default_rng([self.seed, 3889, self.d])
        self.best = None

    @property
    def remaining(self):
        return self.budget - self.used

    @property
    def elapsed(self):
        return time.monotonic() - self.started

    def query(self, points):
        points = np.atleast_2d(np.asarray(points, dtype=float))
        if points.shape[1] != self.d or len(points) > self.remaining:
            raise ValueError("bad query shape or size")
        if not np.all(np.isfinite(points)):
            raise ValueError("nonfinite location")
        if np.max(np.abs(points), initial=0) > 40.0:
            raise ValueError("location out of range")
        y = np.empty(len(points))
        for i in range(0, len(points), self.max_batch):
            if self.elapsed > min(self.limit-2.6, 26.5):
                raise TimeoutError("reserve time for a result")
            p = points[i:i+self.max_batch]
            # Seventeen significant decimal digits preserve binary64.
            s = json.dumps(p.tolist(), separators=(",", ":"), allow_nan=False)
            sys.stdout.write('{"type":"query","points":' + s + "}\n")
            sys.stdout.flush()
            r = json.loads(sys.stdin.readline())
            if r["type"] != "values" or len(r["values"]) != len(p):
                raise ValueError("bad values reply")
            y[i:i+len(p)] = r["values"]
            self.used += len(p)
        return y

    def finish(self, val):
        if val is None or not math.isfinite(val):
            val = self.best
        if val is None or not math.isfinite(val):
            val = 0.0
        sys.stdout.write(json.dumps({"type": "result", "estimate": float(val)}, allow_nan=False) + "\n")
        sys.stdout.flush()


def log(*args):
    if os.environ.get("CUBATURE_DEBUG") == "1":
        print(*args, file=sys.stderr, flush=True)


def sobol(n, d, rng):
    if n <= 0:
        return np.empty((0, d))
    eng = qmc.Sobol(d=d, scramble=True, bits=32, rng=rng)
    p = ndtri(np.clip(eng.random_base2((n-1).bit_length())[:n], 0.5**33, 1-0.5**33))
    return p


def normalize(a):
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-300)


def orth(v):
    """Householder basis with first column v (v is a unit vector)."""
    w = v.copy()
    s = 1.0 if w[0] >= 0 else -1.0
    w[0] += s
    q = np.eye(len(v)) - (2.0 / (w @ w)) * np.outer(w, w)
    q[:, 0] = v
    return q


def linear_charge(x, y):
    y = y - np.mean(y)
    return x.T @ y


def main_fallback(s, x, y, basis=None):
    """Use an independent Gaussian randomized net, if there is room left."""
    d, n = s.d, s.remaining
    if n < 32 or s.elapsed > min(s.limit-4.0, 23.0):
        return s.best
    if basis is None:
        e = np.eye(d)
        order = np.argsort(y)
        take = min(18, len(order), max(0, (n-6400)//(d+1)))
        if take > 0:
            ids = np.r_[order[:take//3], order[-take//3:], s.rng.integers(len(x),size=take-2*(take//3))]
            base = x[ids]
            eps = 0.015
            g = ((s.query((base[:, None]+eps*e).reshape(-1,d)).reshape(-1,d)-y[ids,None])/eps)
            gnorm = np.linalg.norm(g,axis=1)
            if np.max(gnorm) > 500*np.median(gnorm) and np.sum(gnorm > 0)>4:
                g *= np.minimum(1., 5*np.median(gnorm[gnorm > 0])/np.maximum(gnorm,1e-300))[:,None]
            trend = linear_charge(x,y)
            trend /= max(np.linalg.norm(trend),1e-300)
            cov = g.T @ g + np.mean(gnorm * gnorm) * np.outer(trend,trend)
            basis = np.linalg.eigh(cov)[1][:,::-1]
        else:
            basis = orth(normalize(linear_charge(x,y)))
    n = s.remaining
    vals = []
    weights = []
    while n > 0 and s.elapsed < min(s.limit-4.0, 24.0):
        if vals and n < 32:
            break
        k = min(4096, 1 << (n.bit_length()-1))
        p = sobol(k, d, s.rng) @ basis.T
        v = s.query(p)
        vals.append(float(np.mean(v)))
        weights.append(k)
        n -= k
    if vals:
        return float(np.average(vals, weights=weights))
    return s.best
