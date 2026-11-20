"""Oracle accounting, sample designs, and elementary surrogate models."""

import json
import math
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations_with_replacement

import numpy as np
from scipy import linalg, special
from scipy.stats import qmc


class StopSampling(Exception):
    pass


class OracleError(Exception):
    pass


def exchange(points):
    print(json.dumps({"type": "query", "points": points.tolist()},
                     separators=(",", ":"), allow_nan=False), flush=True)
    try:
        msg = json.loads(sys.stdin.readline())
        if msg.get("type") != "values":
            raise OracleError("expected values")
        return msg["values"]
    except (TypeError, AttributeError, KeyError, ValueError) as e:
        raise OracleError("invalid oracle response") from e


class Samples:
    def __init__(self, init, evaluate=exchange):
        self.start = time.monotonic()
        self.d = int(init["dimension"])
        self.budget = int(init["budget"])
        seed = int(init.get("seed", 0)) % (1 << 64)
        self.rng = np.random.default_rng(seed)
        self.evaluate = evaluate
        bounds = np.asarray(init["bounds"], dtype=float)
        self.lo, self.hi = bounds.T
        self.span = self.hi - self.lo
        self.volume = math.prod(self.span)
        self.issued = self.used = self.nu = 0
        self.x = np.empty((self.budget, self.d))
        self.y = np.empty(self.budget)
        self.uy = np.empty(self.budget)
        self.scale = 1.
        self.route = ""
        self.ux = qmc.Sobol(self.d, seed=self.rng).random_base2(
            max(0, (self.budget - 1).bit_length()))[:self.budget]

    @property
    def left(self):
        return self.budget - self.issued

    def time_left(self):
        return 24.0 - (time.monotonic() - self.start)

    def ask(self, x):
        x = np.ascontiguousarray(x, dtype=float).reshape(-1, self.d)
        n = len(x)
        if n > self.left or not np.all(np.isfinite(x)):
            raise StopSampling()
        x = np.clip(x, 0., 1.)
        out = np.empty(n)
        for i in range(0, n, 512):
            p = x[i:i+512]
            self.issued += len(p)
            y = np.asarray(self.evaluate(np.clip(self.lo + p*self.span, self.lo, self.hi)),
                           dtype=float)
            if y.shape != (len(p),) or not np.all(np.isfinite(y)):
                raise OracleError("invalid function values")
            self.x[self.used:self.used+len(p)] = p
            self.y[self.used:self.used+len(p)] = out[i:i+len(p)] = y
            self.used += len(p)
        return out

    def uniform(self, n):
        n = min(n, self.left)
        x = self.ux[self.nu:self.nu+n]
        y = self.ask(x)
        self.uy[self.nu:self.nu+n] = y
        self.nu += n
        return x, y

    def design(self, n, d=None):
        if not n:
            return np.empty((0, self.d if d is None else d))
        return qmc.Sobol(self.d if d is None else d, seed=self.rng).random_base2(
            (n-1).bit_length())[:n]

    def set_scale(self):
        self.scale = max(float(np.max(np.abs(self.y[:self.used]))), 1.e-280)

    def data(self):
        return self.x[:self.used], self.y[:self.used]/self.scale

    def quick(self):
        # Use an intact power-of-two Sobol block where possible.
        if self.nu == 0:
            if self.used:
                return stable_mean(self.y[:self.used])
            if self.left:
                self.uniform(min(256, self.left))
        if self.nu == 0:
            raise StopSampling()
        n = 1 << (self.nu.bit_length()-1)
        return stable_mean(self.uy[:n])


@dataclass
class Model:
    predict: object
    mean: float
    name: str


def stable_mean(y):
    y = np.asarray(y)
    a = max(np.max(np.abs(y), initial=0.), 1.e-300)
    return float(np.mean(y/a)*a)


def least(a, b):
    return linalg.lstsq(a, b, cond=2.e-13, lapack_driver="gelsy",
                        check_finite=False)[0]


@lru_cache(maxsize=32)
def gauss(n):
    x, w = special.roots_legendre(n)
    return (x+1.)/2., w/2.


@lru_cache(maxsize=12)
def gauss_extended(n):
    """Polish the nodes and weights in extended precision."""
    z = special.roots_legendre(n)[0].astype(np.longdouble)
    for it in range(3):
        p, q = np.ones_like(z), z.copy()
        for k in range(2, n+1):
            p, q = q, ((2*k-1)*z*q-(k-1)*p)/k
        dz = n*(p-z*q)/(1-z*z)
        if it < 2:
            z -= q/dz
    w = 1./((1-z*z)*dz*dz)
    w /= np.sum(w)
    return (1.+z)*.5, w


@lru_cache(maxsize=32)
def powers(d, order):
    indices = [()]
    for k in range(1, order+1):
        indices.extend(combinations_with_replacement(range(d), k))
    a = np.array([np.bincount(p, minlength=d) for p in indices])
    cols = tuple((j, k, np.flatnonzero(a[:, j] == k))
                 for j in range(d) for k in range(1, order+1)
                 if np.any(a[:, j] == k))
    return a, cols


def legendre_features(x, order, a=None, cols=None):
    x = np.asarray(x)
    if a is None:
        a, cols = powers(x.shape[1], order)
    z = (2.*x-1.)
    if order == 0:
        return np.ones((len(x), 1))
    v = [np.ones_like(z), z]
    for k in range(2, order+1):
        v.append(((2*k-1)*z*v[-1]-(k-1)*v[-2])/k)
    f = np.ones((len(x), len(a)))
    for j, k, col in cols:
        f[:, col] *= v[k][:, j, None]
    return f


def fit_polynomial(x, y, orders=(1, 2, 3)):
    d = x.shape[1]
    for order in orders:
        p = math.comb(d+order, order)
        if p*1.4 > len(x) or p > 400:
            continue
        f = legendre_features(x, order)
        c = least(f, y)
        if np.max(np.abs(f @ c-y)) <= 7.e-13*np.max(np.abs(y), initial=0.):
            return Model(lambda z, c=c, o=order: legendre_features(z, o) @ c,
                         float(c[0]), "polynomial")
    return None


def log_quadratic_features(x):
    t = x-.5
    return np.column_stack((np.ones(len(x)), t, t*t))


def log_quad_moments(b, a):
    """Log of the coordinate integrals of exp(b*(x-.5)+a*(x-.5)^2)."""
    b, a = np.atleast_1d(b), np.atleast_1d(a)
    z, w = gauss(96)
    z = z[:, None]-.5
    result = special.logsumexp(z*b + z*z*a, axis=0, b=w[:, None])
    mask = a < -4.
    if np.any(mask):
        A = -a[mask]
        m = b[mask]/(2.*A)
        # Use the closer tail of the normal CDF in each coordinate.
        s = np.sqrt(2.*A)
        u = s*(.5-np.abs(m))
        l = s*(-.5-np.abs(m))
        u, l = special.log_ndtr(u), special.log_ndtr(l)
        result[mask] = (A*m*m + .5*np.log(np.pi/A)
                        + u + np.log(-np.expm1(np.minimum(l-u, -1.e-300))))
    return result


def exp_quadratic(c, sign=1., name="exponential quadratic"):
    d = (len(c)-1)//2
    log_mean = c[0]+np.sum(log_quad_moments(c[1:d+1], c[d+1:]))
    def predict(x):
        return sign*np.exp(np.minimum(log_quadratic_features(x)@c, 700.))
    return Model(predict, float(sign*np.exp(min(log_mean, 700.))), name)


def fit_exp_quadratic(x, y):
    sign = 1. if np.max(y) > -np.min(y) else -1.
    u = y*sign
    if np.any(u < 0.):
        return None
    good = u > 1.e-290
    if np.count_nonzero(good) < 2*(1+2*x.shape[1]):
        return None
    f = log_quadratic_features(x)
    log_u = np.log(u[good])
    c = least(f[good], log_u)
    test = f@c
    if np.max(np.abs(test[good]-log_u)) > 2.e-10:
        return None
    if np.any(test[~good] > -660.):
        return None
    return exp_quadratic(c, sign)


def agrees(pred, y, rel=2.e-11, floor=3.e-14):
    if np.shape(pred) != np.shape(y):
        return False
    return bool(np.all(np.abs(pred-y) <= floor + rel*np.maximum(np.abs(y), np.abs(pred))))
