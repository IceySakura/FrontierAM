"""Protocol, independently scrambled Gaussian nets, and pilot fitting."""

import json
import math
import sys
import time

import numpy as np
from scipy.linalg import eigh, solve
from scipy.special import ndtr, ndtri
from scipy.stats import qmc


class Oracle:
    def __init__(self):
        self.started = time.monotonic()
        init = json.loads(sys.stdin.readline())
        self.d = int(init["dimension"])
        self.remaining = int(init["budget"])
        self.max_batch = int(init["max_batch"])
        self.limit = float(init.get("time_limit_s", 30.0))
        self.rng = np.random.default_rng(int(init["seed"]))

    def query(self, points):
        points = np.asarray(points, dtype=float).reshape(-1, self.d)
        if len(points) > self.remaining:
            raise ValueError("query budget")
        points = np.clip(points, -40, 40)
        out = np.empty(len(points))
        for i in range(0, len(points), self.max_batch):
            p = points[i:i+self.max_batch]
            print(json.dumps({"type": "query", "points": p.tolist()}, separators=(",", ":"),
                             allow_nan=False), flush=True)
            ans = json.loads(sys.stdin.readline())
            out[i:i+len(p)] = ans["values"]
            self.remaining -= len(p)
        return out

    def left(self):
        return self.limit - (time.monotonic() - self.started)

    def result(self, value):
        print(json.dumps({"type": "result", "estimate": float(value)}, allow_nan=False), flush=True)


def uniform(n, d, rng):
    if d == 0:
        return np.empty((n, 0))
    return qmc.Sobol(d, scramble=True, rng=rng, bits=32).random_base2((n-1).bit_length())[:n]


def gaussian(n, d, rng):
    return ndtri(np.clip(uniform(n, d, rng), 2.0**-34, 1-2.0**-34))


def blocks(n, minsize=1):
    while n >= minsize:
        k = 1 << (n.bit_length()-1)
        yield k
        n -= k


def fd_gradients(f, x, y=None, delta=0.00015):
    n, d = x.shape
    if y is None:
        y = f.query(x)
    return (f.query(x[:, None, :] + delta*np.eye(d)).reshape(n, d)-y[:, None])/delta


def unit(x):
    return x / max(1e-16, np.linalg.norm(x))


def basis_from(gg, first=None):
    d = gg.shape[0]
    if first is None:
        a, V = eigh(gg)
        return V[:, ::-1].copy()
    first = unit(first)
    if np.linalg.norm(first) < .5:
        return basis_from(gg)
    w = first.copy()
    s = 1.0 if w[0] >= 0 else -1.0
    w[0] += s
    U = (np.eye(d)-2*np.outer(w,w)/(w@w))[:, 1:]
    a, V = eigh(U.T @ gg @ U)
    return np.column_stack((first, U @ V[:, ::-1]))


def fit_ridge(X, y, reg):
    A = X.T @ X
    A.flat[::len(A)+1] += reg
    return solve(A, X.T @ y, assume_a="pos", check_finite=False)


def grad_basis(f, X, Y, m, positive=False, residual=None):
    if positive:
        ix = np.flatnonzero(Y > max(1e-9, np.mean(Y)*0.05))
    else:
        ix = np.arange(len(X))
    if len(ix) == 0:
        return np.eye(f.d), np.empty((0, f.d)), ix
    ix = ix[f.rng.permutation(len(ix))[:m]]
    G = fd_gradients(f, X[ix], Y[ix])
    if residual is not None:
        G -= residual(X[ix])
    good = np.all(np.isfinite(G), axis=1)
    ix, G = ix[good], G[good]
    scale = np.linalg.norm(G, axis=1)
    if len(scale):
        cap = np.quantile(scale, .88)*2 + 1e-15
        G = G*np.minimum(1, cap/np.maximum(scale, 1e-15))[:, None]
    moment = G.T @ G
    if positive and len(G):
        first = np.mean(G/np.maximum(np.linalg.norm(G,axis=1)[:, None],1e-12), axis=0)
    else:
        first = None
    return basis_from(moment, first), G, ix


def net_integral(f, V, proxy=None, mean=0.0, fallback=None):
    n = int(f.remaining)
    acc = []
    weights = []
    if n <= 0:
        return mean if fallback is None else fallback
    for k in blocks(n):
        if acc and (k < 16 or f.left() < 4.0):
            break
        z = gaussian(k, f.d, f.rng) @ V.T
        y = f.query(z)
        if proxy is not None:
            y = y - proxy(z)
        if np.all(np.isfinite(y)):
            m = np.mean(y)
            acc.append(m)
            weights.append(k**1.35)
    return mean + np.average(acc, weights=weights) if acc else fallback


def linear_call_mean(c, b, strike):
    s = np.linalg.norm(b)
    t = (c - math.log(strike))/max(s, 1e-12)
    return np.exp(c + .5*s*s)*ndtr(t+s) - strike*ndtr(t)
