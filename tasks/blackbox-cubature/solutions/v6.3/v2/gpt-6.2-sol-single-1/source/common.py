import itertools
import json
import math
import sys
import time
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from numpy.polynomial.legendre import legvander
from scipy.linalg import lstsq
from scipy.special import roots_legendre
from scipy.stats import qmc


class NoBudget(Exception):
    pass


class Deadline(Exception):
    pass


class Oracle:
    def __init__(self, init, started):
        self.d = int(init["dimension"])
        self.budget = int(init["budget"])
        self.used = 0
        self.started = started
        self.bounds = np.asarray(init["bounds"], dtype=float)
        self.lo = self.bounds[:, 0]
        self.width = self.bounds[:, 1] - self.lo
        self.volume = math.prod(self.width)
        self.unit = bool(np.all(self.lo == 0) and np.all(self.width == 1))
        self.rng = np.random.default_rng(int(init.get("seed", 0)) % (1 << 64))
        self.cache = {}
        self.global_net = None
        self.global_n = 0
        self.uniform_x = []
        self.uniform_y = []
        self.note = None
        self.in_exchange = False
        self.expired = False
        self.provisional = None

    @property
    def left(self):
        return self.budget - self.used

    @property
    def elapsed(self):
        return time.monotonic() - self.started

    def send(self, pts):
        response = []
        for start in range(0, len(pts), 512):
            if self.expired or self.elapsed > 24.75:
                raise Deadline
            p = pts[start:start + 512]
            if not self.unit:
                p = np.clip(self.lo + p * self.width, self.lo, self.bounds[:, 1])
            obj = {"type": "query", "points": p.tolist()}
            msg = json.dumps(obj, separators=(",", ":"), allow_nan=False)
            self.in_exchange = True
            try:
                print(msg, flush=True)
                self.used += len(p)
                got = json.loads(sys.stdin.readline())
            finally:
                self.in_exchange = False
            y = np.asarray(got["values"], dtype=float)
            if got.get("type") != "values" or y.shape != (len(p),) or not np.all(np.isfinite(y)):
                raise ValueError("invalid values response")
            response.append(y)
        return np.concatenate(response) if response else np.empty(0)

    def sample(self, points):
        pts = np.clip(np.asarray(points, dtype=float).reshape(-1, self.d), 0., 1.)
        if not np.all(np.isfinite(pts)):
            raise ValueError("invalid sample coordinate")
        # Cache exact repeats within and across batches, including corners.
        keys, new = [], {}
        for p in pts:
            key = p.tobytes()
            keys.append(key)
            if key not in self.cache and key not in new:
                new[key] = p
        if len(new) > self.left:
            raise NoBudget
        if new:
            values = self.send(np.array(list(new.values())))
            self.cache.update(zip(new, values))
        return np.array([self.cache[key] for key in keys])

    def new_sobol(self, n, d=None):
        d = self.d if d is None else d
        if not n:
            return np.empty((0, d))
        seed = int(self.rng.integers(2**32))
        return qmc.Sobol(d, scramble=True, seed=seed).random_base2((n - 1).bit_length())[:n]

    def uniform(self, n):
        if not n:
            return np.empty((0, self.d)), np.empty(0)
        if self.global_net is None or len(self.global_net) < self.budget:
            self.global_net = self.new_sobol(self.budget)
        x = self.global_net[self.global_n:self.global_n + n]
        if len(x) != n:
            raise NoBudget
        y = self.sample(x)
        self.global_n += n
        self.uniform_x.append(x)
        self.uniform_y.append(y)
        return x, y

    def data(self):
        if not self.cache:
            return np.empty((0, self.d)), np.empty(0)
        x = np.frombuffer(b"".join(self.cache), dtype=np.float64).reshape(-1, self.d)
        y = np.fromiter(self.cache.values(), float)
        return x, y

    def fallback(self):
        if self.uniform_y:
            y = np.concatenate(self.uniform_y)
        else:
            y = self.data()[1]
        return stable_mean(y)

    def best_effort(self):
        return self.provisional[1] if self.provisional is not None else self.fallback()

    def on_alarm(self, signum, frame):
        self.expired = True
        if not self.in_exchange and self.cache:
            raise Deadline


def stable_mean(y):
    scale = np.max(np.abs(y))
    return float(scale * np.mean(y / scale)) if scale > 0 else 0.


@dataclass
class Model:
    predict: object
    integral: float
    name: str
    error: float = np.inf


def residual(model, x, y):
    with np.errstate(all="ignore"):
        delta = np.asarray(model.predict(x)) - y
    if delta.shape != y.shape or not np.isfinite(model.integral) or not np.all(np.isfinite(delta)):
        return np.inf, np.inf
    return float(np.sqrt(np.mean(delta**2))), float(np.max(np.abs(delta)))


def solve_ls(a, b, cond=2e-13):
    return lstsq(a, b, cond=cond, check_finite=False, lapack_driver="gelsy")[0]


@lru_cache(None)
def multiindices(d, degree, order):
    out = [[0] * d]
    for n in range(1, degree + 1):
        for combo in itertools.combinations_with_replacement(range(d), n):
            if len(set(combo)) <= order:
                index = [0] * d
                for j in combo:
                    index[j] += 1
                out.append(index)
    return np.asarray(out).T.copy()


def legmat(x, ids):
    z = 2 * np.asarray(x) - 1
    v = legvander(z, int(ids.max()))
    a = np.ones((len(z), ids.shape[1]))
    for j, row in enumerate(ids):
        if any(row):
            a *= v[:, j, row]
    return a


@lru_cache(None)
def gl(n):
    return roots_legendre(n)


def gl_interval(n, a=0., b=1.):
    z, w = gl(n)
    return (a + b) * .5 + z * ((b - a) * .5), w * ((b - a) * .5)


_GKX = np.array([0.9914553711208126, 0.9491079123427585, 0.8648644233597691,
                 0.7415311855993944, 0.5860872354676911, 0.4058451513773972,
                 0.2077849550078985, 0.])
_GKW = np.array([0.02293532201052922, 0.06309209262997855, 0.1047900103222502,
                 0.1406532597155259, 0.1690047266392679, 0.1903505780647854,
                 0.2044329400752989, 0.2094821410847278])
_GW = np.array([0., 0.1294849661688697, 0., 0.2797053914892767, 0.,
                0.3818300505051189, 0., 0.4179591836734694])
GK_X = np.r_[-_GKX[:-1], _GKX[::-1]]
GK_H = np.r_[_GKW[:-1], _GKW[::-1]]
GK_L = np.r_[_GW[:-1], _GW[::-1]]
GK_LOW = np.nonzero(GK_L)[0]


class AxisQuad:
    """Embedded quadrature; values have shape (intervals, 15)."""
    def __init__(self, projection=None):
        self.proj = projection
        if projection is not None:
            self.hmat = np.linalg.inv(legvander(GK_X, 14))
            self.lmat = np.linalg.inv(legvander(GK_X[GK_LOW], 6))

    def evaluate(self, ab, fn):
        ab = np.asarray(ab)
        h = (ab[:, 1] - ab[:, 0]) * .5
        m = (ab[:, 1] + ab[:, 0]) * .5
        z = m[:, None] + h[:, None] * GK_X
        ys = np.asarray(fn(z.ravel())).reshape(len(ab), 15)
        if self.proj is None:
            large = h * (ys @ GK_H)
            small = h * (ys @ GK_L)
        else:
            moments = np.array([self.proj.panel_moments(a, b) for a, b in ab])
            large = np.einsum("ij,ij->i", moments @ self.hmat, ys)
            small = np.einsum("ij,ij->i", moments[:, :7] @ self.lmat, ys[:, GK_LOW])
        return large, np.abs(large - small)

    def adaptive(self, fn, left, breaks=(), maxpts=450, tol=5e-13, batch=4):
        br = np.r_[0., np.unique(np.clip(breaks, 0., 1.)), 1.]
        panels = [(a, b) for a, b in zip(br[:-1], br[1:]) if b-a > 5e-14]
        if len(panels) * 15 > min(maxpts, left()):
            panels = [(0., 1.)]
        if left() < len(panels) * 15:
            raise NoBudget
        v, e = self.evaluate(panels, fn)
        state = [(a, b, x, y) for (a, b), x, y in zip(panels, v, e)]
        used = 15 * len(panels)
        while min(maxpts - used, left()) >= 30:
            val = math.fsum(i[2] for i in state)
            err = math.fsum(i[3] for i in state)
            if err <= tol * max(abs(val), 1e-290):
                break
            state.sort(key=lambda x: x[3])
            maxe = state[-1][3]
            spl = []
            while len(spl) < min(batch, (maxpts - used)//30, left()//30) and state:
                if spl and state[-1][3] < maxe * .2:
                    break
                spl.append(state.pop())
            panels = []
            for a, b, *_ in spl:
                m = (a + b) / 2
                panels.extend([(a, m), (m, b)])
            v, e = self.evaluate(panels, fn)
            state.extend((a, b, x, y) for (a, b), x, y in zip(panels, v, e))
            used += len(panels) * 15
        return math.fsum(i[2] for i in state), math.fsum(i[3] for i in state)
