"""Sampling and small regression utilities shared by the estimators."""

import math
import time
import numpy as np
from scipy import linalg, special
from scipy.stats import qmc


def unit_net(n, d, rng):
    if n == 0:
        return np.empty((0, d))
    net = qmc.Sobol(d, scramble=True, seed=int(rng.integers(1, 2**32)))
    return net.random_base2((n - 1).bit_length())[:n]


def quantile(u):
    return special.ndtri(np.clip(u, 2.0**-40, 1.0 - 2.0**-40))


def normal_net(n, d, rng):
    return quantile(unit_net(n, d, rng))


def rotation(cov, first=None):
    d = len(cov)
    cov = (cov + cov.T) * .5
    if first is None:
        return linalg.eigh(cov, check_finite=False)[1][:, ::-1].copy()
    first = first / max(np.linalg.norm(first), 1e-100)
    if not np.isfinite(first).all() or np.linalg.norm(first) < .5:
        return rotation(cov)
    base = linalg.qr(first[:, None], mode="full", check_finite=False)[0][:, 1:]
    return np.column_stack([first, base @ rotation(base.T @ cov @ base)])


def ridge(x, y, penalty=1e-3, weights=None, center=True):
    """Ridge penalty on mean squared residual, with optional feature scaling."""
    x = np.asarray(x)
    y = np.asarray(y)
    if len(x) == 0:
        return np.zeros(x.shape[1]), 0.
    if weights is None:
        weights = np.full(len(x), 1. / len(x))
    else:
        weights = weights / max(np.sum(weights), 1e-100)
    if center:
        mx, my = weights @ x, weights @ y
    else:
        mx, my = np.zeros(x.shape[1]), 0.
    x0 = x - mx
    xx = x0.T @ (weights[:, None] * x0)
    xx.flat[::len(xx)+1] += penalty
    coef = linalg.solve(xx, x0.T @ (weights * (y-my)), assume_a="pos", check_finite=False)
    return coef, my - mx @ coef


def logistic(x, y, penalty=.02, iterations=10):
    n, d = x.shape
    x = np.column_stack((np.ones(n), x))
    par = np.zeros(d+1)
    yy = np.clip(np.mean(y), 1.e-4, 1.-1.e-4)
    par[0] = math.log(yy / (1.-yy))
    reg = np.broadcast_to(penalty, (d,)).copy()
    reg = np.r_[1.e-5, reg]
    for _ in range(iterations):
        p = special.expit(x @ par)
        a = x.T @ ((p*(1.-p))[:, None] * x) / n
        a.flat[::len(a)+1] += reg
        b = (x.T @ (y-p)) / n - reg * par
        step = linalg.solve(a, b, assume_a="pos", check_finite=False)
        par += np.clip(step, -4., 4.)
        if np.max(np.abs(step)) < 1.e-5:
            break
    return par


def quadratic_features(x, diagonal=True, width=None):
    d = x.shape[1] if width is None else min(width, x.shape[1])
    i, j = np.triu_indices(d, 1)
    cols = [x, x[:, i] * x[:, j]]
    if diagonal:
        cols += [(x*x - 1.) * (2.**-.5)]
    return np.column_stack(cols)


def powers(n, low=128):
    while n >= low:
        m = 1 << (n.bit_length()-1)
        yield m
        n -= m
    if n:
        yield n


class Work:
    """Numerical work for one run, with points and random streams kept local."""

    def __init__(self, session, deadline=None):
        self.session = session
        self.d = session.d
        self.rng = np.random.default_rng([session.seed, 7193])
        self.fallback = 0.
        self.deadline = deadline

    @property
    def remaining(self):
        return self.session.remaining

    def query(self, points):
        points = np.asarray(points, dtype=float)
        if not np.isfinite(points).all():
            raise FloatingPointError("nonfinite sample coordinate")
        if self.deadline is not None:
            # Do not start a large exchange close to the wall limit. The
            # check is outside the JSON-lines write/read pair.
            allowance = .9 + min(3.0, .0000045 * points.size)
            if time.monotonic() + allowance >= self.deadline:
                raise RuntimeError("time reserve reached")
        return self.session.query(np.clip(points, -40., 40.))

    def normal(self, n, d=None):
        return normal_net(n, self.d if d is None else d, self.rng)

    def unit(self, n, d=None):
        return unit_net(n, self.d if d is None else d, self.rng)


def local_gradient(w, x, y, h=0.003):
    x = np.atleast_2d(x)
    d = w.d
    if len(x) == 0:
        return np.empty((0, d))
    v = np.eye(d) * h
    ys = w.query((x[:, None, :] + v).reshape(-1, d)).reshape(len(x), d)
    return (ys - np.asarray(y)[:, None]) / h


def residual_average(w, basis, residual, test_symmetry=True):
    pilot = 0.
    ntest = 0
    pair = False
    if test_symmetry and w.remaining > 3300:
        z = w.normal(128) @ basis.T
        both = np.vstack([z,-z])
        vals = residual(both,w.query(both))
        ntest = len(vals)
        even = .5*(vals[:len(z)]+vals[len(z):])
        pair = np.var(even) < .29*max(float(np.var(vals)),1.e-30)
        pilot = vals.mean()
    n = w.remaining
    if pair:
        z = w.normal((n+1)//2) @ basis.T
        z = np.vstack([z,-z[:n-len(z)]])
    else:
        z = w.normal(n) @ basis.T
    main = np.mean(residual(z,w.query(z)))
    pilot_weight = .30*ntest/max(n+ntest,1)
    return float((1.-pilot_weight)*main + pilot_weight*pilot)
