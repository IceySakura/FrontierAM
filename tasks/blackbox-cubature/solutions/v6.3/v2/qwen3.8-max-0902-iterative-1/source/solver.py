#!/usr/bin/env python3
"""Black-box cubature of an unknown f over [0,1]^d under a hard query budget.

Strategy
--------
1.  A QMC pilot (prefix of a randomised digital net) gives a scale estimate, a
    validation set, a control-variate sample and a safe fallback integral.
2.  Coordinate slices through an anchor point (taken at max |f| of the pilot)
    give exact 1-D sections s_i(t) = f(anchor with x_i = t).
3.  Structural hypotheses are tested with algebraic residual tests on points
    whose coordinates all lie on the slice-node grid, so a prediction needs no
    interpolation and the test is exact to round-off:
        f = const, quadratic, c + sum_i g_i(x_i), prod_i g_i(x_i),
        c + prod_i g_i(x_i), G(v.x)   (single ridge)
4.  The surviving hypothesis is refined with the remaining budget: adaptive
    Clenshaw-Curtis quadrature of its 1-D pieces, with greedy allocation
    across axes and automatic localisation of kinks/jumps/peaks.  The integral
    of the representation is then available analytically.
5.  Every candidate is scored by an honest error estimate and corrected by a
    QMC control variate (mean residual on the pilot points), so an imperfect
    hypothesis degrades gracefully towards plain QMC instead of blowing up.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np

DBG = bool(os.environ.get("CUB_DEBUG"))
T_START = time.time()
TIME_LIMIT = 21.0
MAX_QUERY = 512
N0 = 6                 # initial Clenshaw-Curtis order per 1-D interval
NMAX = 96              # max p-refinement order
MAXIV = 400            # max intervals per channel
NSPLIT = 14            # max feature splits per interval lineage

_CC = {}


def timed_out(margin=0.0):
    return (time.time() - T_START) > (TIME_LIMIT - margin)


class GiveUp(Exception):
    pass


# --------------------------------------------------------------------------
# Clenshaw-Curtis (Chebyshev-Lobatto) nodes and weights on [-1,1]
# --------------------------------------------------------------------------
def clencurt(N):
    got = _CC.get(N)
    if got is not None:
        return got
    k = np.arange(N + 1)
    theta = np.pi * k / N
    x = np.cos(theta)
    w = np.zeros(N + 1)
    ii = np.arange(1, N)
    v = np.ones(N - 1)
    if N % 2 == 0:
        w[0] = 1.0 / (N * N - 1)
        w[N] = w[0]
        for kk in range(1, N // 2):
            v -= 2.0 * np.cos(2 * kk * theta[ii]) / (4 * kk * kk - 1)
        v -= np.cos(N * theta[ii]) / (N * N - 1)
    else:
        w[0] = 1.0 / (N * N)
        w[N] = 1.0 / (N * N)
        for kk in range(1, (N - 1) // 2 + 1):
            v -= 2.0 * np.cos(2 * kk * theta[ii]) / (4 * kk * kk - 1)
    w[ii] = 2.0 * v / N
    out = (np.ascontiguousarray(x[::-1]), np.ascontiguousarray(w[::-1]))
    _CC[N] = out
    return out


def lob_nodes(a, b, N):
    x, _ = clencurt(N)
    return 0.5 * (a + b) + 0.5 * (b - a) * x


# --------------------------------------------------------------------------
# protocol
# --------------------------------------------------------------------------
class Oracle:
    def __init__(self, d, budget, lo, hi):
        self.d = int(d)
        self.budget = int(budget)
        self.lo = np.asarray(lo, float)
        self.hi = np.asarray(hi, float)
        self.used = 0
        self.cache = {}
        self.nq = 0

    @property
    def remaining(self):
        return self.budget - self.used

    def get(self, pts):
        """Return (values, ok).  Caches; only new points cost budget."""
        pts = np.asarray(pts, float).reshape(-1, self.d)
        if len(pts) == 0:
            return np.zeros(0), True
        keys = [tuple(np.round(p, 11)) for p in pts]
        out = np.full(len(pts), np.nan)
        need = []
        seen = set()
        for i, k in enumerate(keys):
            v = self.cache.get(k)
            if v is None:
                if k not in seen:
                    seen.add(k)
                    need.append(i)
            else:
                out[i] = v
        ok = True
        if need:
            if len(need) > self.remaining:
                need = need[:max(0, self.remaining)]
                ok = False
            if need:
                q = np.clip(pts[need], self.lo, self.hi)
                got = self._query(q)
                for j, i in enumerate(need):
                    fv = float(got[j])
                    self.cache[keys[i]] = fv
                for i, k in enumerate(keys):
                    if k in self.cache:
                        out[i] = self.cache[k]
            else:
                ok = False
        return out, ok

    def _query(self, q):
        vals = []
        for s in range(0, len(q), MAX_QUERY):
            if timed_out():
                raise GiveUp('time')
            chunk = np.ascontiguousarray(q[s:s + MAX_QUERY])
            sys.stdout.write(json.dumps({"type": "query", "points": chunk.tolist()}) + "\n")
            sys.stdout.flush()
            line = sys.stdin.readline()
            if not line:
                raise GiveUp('eof')
            msg = json.loads(line)
            v = np.asarray(msg.get("values", []), dtype=float)
            if v.size != len(chunk) or not np.all(np.isfinite(v)):
                raise GiveUp('bad values')
            vals.append(v)
            self.nq += 1
        self.used += len(q)
        return np.concatenate(vals) if len(vals) > 1 else vals[0]


def net_points(d, n, seed):
    """Randomised digital net, prefix-consistent in n."""
    s = int(seed) % (2 ** 31)
    try:
        from qmcpy.discrete_distribution import DigitalNetB2
        p = np.asarray(DigitalNetB2(d, seed=s)(n), dtype=float)
        if p.shape == (n, d) and np.all(np.isfinite(p)):
            return np.clip(p, 0.0, 1.0)
    except Exception:
        pass
    try:
        from scipy.stats.qmc import Sobol
        p = np.asarray(Sobol(d=d, scramble=True, seed=s).random(n), dtype=float)
        if p.shape == (n, d):
            return np.clip(p, 0.0, 1.0)
    except Exception:
        pass
    return np.random.default_rng(seed).random((n, d))


# --------------------------------------------------------------------------
# local polynomial interpolation, vectorised over query points
# --------------------------------------------------------------------------
def local_interp(nodes, vals, xq, k=8):
    nodes = np.asarray(nodes, float)
    vals = np.asarray(vals, float)
    xq = np.atleast_1d(np.asarray(xq, float))
    n = len(nodes)
    if n == 0:
        return np.zeros_like(xq)
    if n == 1:
        return np.full_like(xq, vals[0])
    k = int(min(k, n))
    idx = np.searchsorted(nodes, xq)
    start = np.clip(idx - (k + 1) // 2, 0, n - k)
    sel = start[:, None] + np.arange(k)[None, :]
    xs = nodes[sel]
    ys = vals[sel]
    # affine-rescale each stencil to [-1,1]: barycentric weights are invariant
    # but their raw magnitudes span hundreds of orders for clustered nodes.
    ctr = 0.5 * (xs[:, 0] + xs[:, -1])
    hw = np.maximum(0.5 * (xs[:, -1] - xs[:, 0]), 1e-300)
    zs = (xs - ctr[:, None]) / hw[:, None]
    zq = (xq - ctr) / hw
    diff = zq[:, None] - zs
    exact = np.abs(diff) < 1e-13
    tmp = zs[:, :, None] - zs[:, None, :]
    ar = np.arange(k)
    tmp[:, ar, ar] = 1.0
    with np.errstate(divide='ignore', over='ignore', invalid='ignore', under='ignore'):
        w = 1.0 / np.prod(tmp, axis=2)
        w = w / np.max(np.abs(w), axis=1, keepdims=True)
        safe = np.where(exact, 1.0, diff)
        num = np.sum(np.where(exact, 0.0, w * ys / safe), axis=1)
        den = np.sum(np.where(exact, 0.0, w) / safe, axis=1)
        out = num / den
    hit = exact.any(axis=1)
    if np.any(hit):
        j = np.argmax(exact[hit], axis=1)
        out[hit] = ys[hit][np.arange(int(hit.sum())), j]
    bad = ~np.isfinite(out)
    if np.any(bad):
        out[bad] = np.interp(xq[bad], nodes, vals)
    return out


# --------------------------------------------------------------------------
# 1-D adaptive integration channel
# --------------------------------------------------------------------------
class Iv:
    __slots__ = ('a', 'b', 'N', 'Neff', 'probes', 'q', 'err', 'nsplit', 'lastN',
                 'lasterr', 'stall', 'frozen')

    def __init__(self, a, b, N, nsplit=0):
        self.a = float(a)
        self.b = float(b)
        self.N = int(N)
        self.Neff = 0
        self.probes = 0
        self.q = 0.0
        self.err = np.inf
        self.nsplit = int(nsplit)
        self.lastN = 0
        self.lasterr = None
        self.stall = 0
        self.frozen = False


class Channel:
    """Adaptive Clenshaw-Curtis integration of a scalar function of t on [lo,hi].

    ``off`` is subtracted from raw samples; ``weight`` (t-array -> w-array) is
    multiplied in.  Raw samples are cached in ``tv`` and shared with the
    feature-localisation probes.
    """

    def __init__(self, lo, hi, off=0.0, weight=None, tol=None):
        self.lo = float(lo)
        self.hi = float(hi)
        self.off = float(off)
        self.weight = weight
        self.ivs = [Iv(lo, hi, N0)]
        self.tv = {}
        self.Q = 0.0
        self.err = np.inf
        self.mag = 1.0
        self.done = False
        self.ok = True
        self.tol = tol
        self.rounds = 0

    # -- sampling ---------------------------------------------------------
    def y_of(self, ts):
        y = np.array([self.tv[float(t)] for t in ts]) - self.off
        if self.weight is not None:
            y = y * self.weight(np.asarray(ts, float))
        return y

    def needed(self):
        out = []
        seen = set()
        for iv in self.ivs:
            for t in lob_nodes(iv.a, iv.b, iv.N):
                key = round(float(t), 12)
                if key not in self.tv and key not in seen:
                    seen.add(key)
                    out.append(float(t))
        return out

    def update(self, ts, fs):
        for t, f in zip(ts, fs):
            self.tv[round(float(t), 12)] = float(f)

    def usable(self, iv):
        """Largest nested order <= iv.N whose nodes are all available."""
        N = int(iv.N)
        while N >= 1:
            ts = lob_nodes(iv.a, iv.b, N)
            if all(round(float(t), 12) in self.tv for t in ts):
                return N, ts
            N //= 2
        return 0, None

    def recompute(self):
        Q = 0.0
        E = 0.0
        mag = 0.0
        self.ok = True
        for iv in self.ivs:
            N, ts = self.usable(iv)
            if N < 1:
                self.Q, self.err, self.ok = 0.0, np.inf, False
                return
            iv.Neff = N
            keys = [round(float(t), 12) for t in ts]
            y = self.y_of(keys)
            _, w = clencurt(N)
            q = 0.5 * (iv.b - iv.a) * float(np.dot(w, y))
            iv.q = q
            if N >= 2 and N % 2 == 0:
                _, w2 = clencurt(N // 2)
                q2 = 0.5 * (iv.b - iv.a) * float(np.dot(w2, y[::2]))
                iv.err = abs(q - q2)
            else:
                iv.err = abs(q) * 0.5
            if iv.lasterr is not None and N != iv.lastN:
                if iv.err > iv.lasterr * 0.15:
                    iv.stall += 1
                else:
                    iv.stall = 0
            iv.lastN = N
            iv.lasterr = iv.err
            Q += q
            E += iv.err
            mag = max(mag, float(np.max(np.abs(y))))
        # do not chase round-off: freeze intervals that are already exact enough
        for iv in self.ivs:
            if iv.err <= 1e-14 * mag * (iv.b - iv.a) + 1e-300:
                iv.err = 0.0
                E -= 0.0
        self.Q = Q
        self.err = sum(iv.err for iv in self.ivs)
        self.mag = mag

    # -- feature localisation --------------------------------------------
    def _probes_in(self, iv):
        ts = sorted(k for k in self.tv if iv.a - 1e-13 <= k <= iv.b + 1e-13)
        return np.asarray(ts, float), np.asarray([self.tv[k] for k in ts], float)

    def _feature(self, iv):
        """Locate a kink/jump/peak inside iv.

        Returns (probe_points, split_point).  A kink (slope discontinuity) is
        located in closed form by intersecting the neighbouring secant lines;
        a jump is bracketed by bisection, one probe per call."""
        ts, ys = self._probes_in(iv)
        if len(ts) < 3:
            return [0.5 * (iv.a + iv.b)], None
        dt = np.diff(ts)
        s = np.diff(ys) / np.where(dt > 0, dt, 1e-300)
        n = len(s)
        score = np.zeros(n)
        for j in range(n):
            sl = s[j - 1] if j > 0 else s[j]
            sr = s[j + 1] if j < n - 1 else s[j]
            score[j] = abs(s[j] - sl) + abs(s[j] - sr)
        j = int(np.argmax(score))
        sl = s[j - 1] if j > 0 else s[j]
        sr = s[j + 1] if j < n - 1 else s[j]
        gapw = ts[j + 1] - ts[j]
        if gapw <= 1e-8 * max(iv.b - iv.a, 1e-300):
            return [], 0.5 * (ts[j] + ts[j + 1])
        dslope = abs(sr - sl)
        scale = max(abs(sl), abs(sr), 1e-300)
        if dslope > 1e-7 * scale:
            c = (ys[j + 1] - ys[j] - sr * ts[j + 1] + sl * ts[j]) / (sl - sr)
            if ts[j] < c < ts[j + 1]:
                return [], float(c)
        lo, hi = ts[j], ts[j + 1]
        return [lo + (hi - lo) * f for f in (0.25, 0.5, 0.75)], None

    # -- refinement -------------------------------------------------------
    def worst(self):
        bi, be = -1, -1.0
        for i, iv in enumerate(self.ivs):
            if iv.frozen:
                continue
            if iv.err > be:
                be, bi = iv.err, i
        return bi, be

    def refine_guarded(self, limit):
        """refine() but revert the step if it would need more than ``limit`` points."""
        if self.done:
            return []
        snap = list(self.ivs)
        i, _ = self.worst()
        snapN = self.ivs[i].N if 0 <= i < len(self.ivs) else None
        newts = self.refine()
        if len(newts) > limit:
            self.ivs = snap
            if snapN is not None and 0 <= i < len(self.ivs):
                self.ivs[i].N = snapN
            return []
        return newts

    def missing_cost(self):
        return len(self.missing())

    def missing(self):
        out = []
        for iv in self.ivs:
            for t in lob_nodes(iv.a, iv.b, iv.N):
                if round(float(t), 12) not in self.tv:
                    out.append(float(t))
        return out

    def tol_eff(self):
        if self.tol is not None:
            return self.tol
        return 1e-15 * max(self.mag, abs(self.Q), 1e-300)

    def refine(self):
        """Plan one refinement step.  Returns new t values to evaluate."""
        if self.done:
            return []
        self.rounds += 1
        i, e = self.worst()
        if i < 0 or e <= self.tol_eff():
            self.done = True
            return []
        iv = self.ivs[i]
        width = iv.b - iv.a

        def freeze():
            iv.frozen = True
            if all(v.frozen or v.err <= self.tol_eff() for v in self.ivs):
                self.done = True
            return []

        # (a) cheap initial p-exploration
        if iv.N < 24 and iv.stall == 0:
            iv.N *= 2
            return self.needed()
        # (b) refinement is not paying off -> hunt for a kink/jump/peak and split
        if iv.stall >= 1:
            if iv.nsplit < NSPLIT and width > 1e-10 and len(self.ivs) < MAXIV:
                probes, c = self._feature(iv)
                if c is None:
                    out = [t for t in probes if round(float(t), 12) not in self.tv]
                    if out and iv.probes < 40:
                        iv.probes += 1
                        return out
                    c = probes[0] if probes else 0.5 * (iv.a + iv.b)
                if iv.a + 1e-11 < c < iv.b - 1e-11:
                    ns = iv.nsplit + 1
                    self.ivs[i] = Iv(iv.a, c, N0, ns)
                    self.ivs.append(Iv(c, iv.b, N0, ns))
                    self.ivs.sort(key=lambda v: v.a)
                    return self.needed()
            return freeze()
        # (c) smooth and still converging: raise the order
        if iv.N < NMAX:
            iv.N *= 2
            return self.needed()
        # (d) last resort: bisect
        if width > 1e-11 and len(self.ivs) < MAXIV:
            m = 0.5 * (iv.a + iv.b)
            self.ivs[i] = Iv(iv.a, m, N0, iv.nsplit)
            self.ivs.append(Iv(m, iv.b, N0, iv.nsplit))
            self.ivs.sort(key=lambda v: v.a)
            return self.needed()
        return freeze()

    def nodes_sorted(self):
        ks = sorted(self.tv.keys())
        return np.array(ks), np.array([self.tv[k] for k in ks])

    def eval(self, xq):
        """Interval-aware interpolation: never build a stencil across a split,
        so kinks and jumps do not produce wild overshoot."""
        xq = np.atleast_1d(np.asarray(xq, float))
        out = np.empty(len(xq))
        ivs = self.ivs
        edges = np.array([iv.a for iv in ivs] + [ivs[-1].b])
        idx = np.clip(np.searchsorted(edges, xq, side='right') - 1, 0, len(ivs) - 1)
        for j, iv in enumerate(ivs):
            m = idx == j
            if not np.any(m):
                continue
            N = getattr(iv, 'Neff', 0) or self.usable(iv)[0]
            if N < 1:
                out[m] = np.nan
                continue
            ts = lob_nodes(iv.a, iv.b, N)
            ys = np.array([self.tv[round(float(t), 12)] for t in ts]) - self.off
            out[m] = local_interp(ts, ys, xq[m], k=min(8, len(ts)))
        bad = ~np.isfinite(out)
        if np.any(bad):
            ts, ys = self.nodes_sorted()
            if len(ts):
                out[bad] = np.interp(xq[bad], ts, ys - self.off)
            else:
                out[bad] = 0.0
        return out


# --------------------------------------------------------------------------
# ridge helpers
# --------------------------------------------------------------------------
def ridge_density(v):
    v = np.asarray(v, float)
    d = len(v)
    shift = float(np.sum(np.minimum(v, 0.0)))
    w = np.abs(v)
    w = np.where(w < 1e-300, 1e-300, w)
    norm = 1.0 / (math.factorial(d - 1) * float(np.prod(w)))
    knots = np.zeros(1 << d)
    signs = np.zeros(1 << d)
    for mask in range(1 << d):
        c = 0.0
        s = 1.0
        for i in range(d):
            if (mask >> i) & 1:
                c += w[i]
                s = -s
        knots[mask] = c + shift
        signs[mask] = s * norm
    pw = float(d - 1)
    tlo = shift
    thi = shift + float(np.sum(w))

    def pdf(t):
        t = np.atleast_1d(np.asarray(t, float))
        base = np.maximum(t[:, None] - knots[None, :], 0.0)
        terms = base if pw == 1.0 else base ** pw
        p = terms @ signs
        p = np.where((t >= tlo - 1e-12) & (t <= thi + 1e-12), p, 0.0)
        return np.maximum(p, 0.0)

    return pdf, tlo, thi, knots


def ridge_points(v, t):
    """Points x in [0,1]^d with v.x = t (vectorised bisection on a clipped ray)."""
    v = np.asarray(v, float)
    d = len(v)
    t = np.atleast_1d(np.asarray(t, float))
    c = np.full(d, 0.5)
    nrm = float(np.linalg.norm(v))
    if nrm < 1e-300:
        return np.tile(c, (len(t), 1))
    direction = v / nrm
    span = float(np.sum(np.abs(v)))
    lam_lo = np.full(len(t), -2.0 * span - 2.0)
    lam_hi = np.full(len(t), 2.0 * span + 2.0)
    for _ in range(52):
        mid = 0.5 * (lam_lo + lam_hi)
        x = np.clip(c + mid[:, None] * direction[None, :], 0.0, 1.0)
        less = (x @ v) < t
        lam_lo = np.where(less, mid, lam_lo)
        lam_hi = np.where(less, lam_hi, mid)
    lam = 0.5 * (lam_lo + lam_hi)
    return np.clip(c + lam[:, None] * direction[None, :], 0.0, 1.0)


# --------------------------------------------------------------------------
# solver
# --------------------------------------------------------------------------
class Solver:
    def __init__(self, init):
        self.d = int(init["dimension"])
        self.budget = int(init.get("budget", 2048))
        self.seed = int(init.get("seed", 0) or 0)
        bnds = init.get("bounds") or [[0, 1]] * self.d
        self.lo = np.asarray([float(b[0]) for b in bnds])
        self.hi = np.asarray([float(b[1]) for b in bnds])
        if np.any(self.hi <= self.lo):
            self.lo = np.zeros(self.d)
            self.hi = np.ones(self.d)
        self.vol = float(np.prod(self.hi - self.lo))
        self.ora = Oracle(self.d, self.budget, self.lo, self.hi)
        self.best = None
        self.f_scale = 1.0
        self.Xc = np.zeros((0, self.d))
        self.Fc = np.zeros(0)
        self.models = []

    # -- utility ----------------------------------------------------------
    def to_box(self, u):
        u = np.asarray(u, float).reshape(-1, self.d)
        return self.lo[None, :] + u * (self.hi - self.lo)[None, :]

    def from_box(self, x):
        x = np.asarray(x, float).reshape(-1, self.d)
        return (x - self.lo[None, :]) / (self.hi - self.lo)[None, :]

    def add_cv_points(self, U, F):
        if len(U) == 0:
            return
        self.Xc = np.concatenate([self.Xc, np.asarray(U, float).reshape(-1, self.d)])
        self.Fc = np.concatenate([self.Fc, np.asarray(F, float).reshape(-1)])

    def qmc_record(self, tag='qmc'):
        if len(self.Fc) == 0:
            return
        m = float(np.mean(self.Fc)) * self.vol
        sd = float(np.std(self.Fc)) / math.sqrt(max(1, len(self.Fc))) * self.vol
        self.models.append({'name': tag, 'I': m, 'err': max(sd, 1e-300), 'eval': None})

    def evaluate(self):
        """Score every candidate with control-variate correction; keep best."""
        self.best = None
        for m in self.models:
            try:
                I = float(m['I'])
                err = float(m['err'])
                beta = 1.0
                if m['eval'] is not None and len(self.Fc) >= 8:
                    r = np.asarray(self.Fc - m['eval'](self.Xc), float)
                    if np.all(np.isfinite(r)):
                        rrms = float(np.sqrt(np.mean(r ** 2)))
                        m['rrms'] = rrms
                        if m.get('exact') and rrms < 1e-7 * self.f_scale:
                            beta = 0.0     # structure verified: correction only adds noise
                        if beta:
                            I = I + float(np.mean(r)) * self.vol
                            err = err + rrms / math.sqrt(len(r)) * self.vol
                        else:
                            err = max(err, rrms * 1e-3 * self.vol)
                if not np.isfinite(I) or not np.isfinite(err):
                    continue
                m['eff'] = err
                if self.best is None or err < self.best[1]:
                    self.best = (I, err, m['name'])
            except Exception:
                continue

    # -- main -------------------------------------------------------------
    def solve(self):
        d = self.d
        B = self.budget
        # ---------------- pilot ----------------
        npow = max(5, int(math.floor(math.log2(max(8, min(B // 4, 1024))))))
        n_pilot = min(2 ** npow, B)
        n_pilot = max(16, min(n_pilot, self.ora.remaining))
        nnet = 2 ** int(math.floor(math.log2(max(2, B))))
        self.net = net_points(d, nnet, self.seed)
        self.nnet = 0
        n_pilot = min(n_pilot, len(self.net), self.ora.remaining)
        Xp = self.net[:n_pilot]
        Fp, ok = self.ora.get(self.to_box(Xp))
        n = min(len(Xp), len(Fp))
        Xp, Fp = Xp[:n], Fp[:n]
        if n == 0:
            return None
        self.add_cv_points(Xp, Fp)
        self.nnet = n
        self.qmc_record('qmc-pilot')
        self.evaluate()

        self.scale = float(np.sqrt(np.mean(Fp ** 2))) + 1e-300
        self.f_scale = self.scale
        ia = int(np.argmax(np.abs(Fp)))
        anchor = Xp[ia].copy()
        fb = float(Fp[ia])
        self.anchor, self.fb = anchor, fb

        if timed_out(4.0) or self.ora.remaining < 8 * d + 32:
            self.spend_qmc()
            self.evaluate()
            return self.best[0] if self.best else None

        # ---------------- slices + exact test grid ----------------
        nodes0 = 0.5 + 0.5 * clencurt(N0)[0]
        sl = []
        for i in range(d):
            for t in nodes0:
                p = anchor.copy()
                p[i] = t
                sl.append(p)
        rng = np.random.default_rng(self.seed + 9973)
        n_grid = int(min(96, max(16, self.ora.remaining - len(sl) - 4 * d - 64)))
        n_grid = max(16, n_grid)
        gu = rng.integers(0, len(nodes0), size=(n_grid, d))
        gp = nodes0[gu]
        allp = np.concatenate([np.asarray(sl), gp], axis=0)
        vals, ok = self.ora.get(self.to_box(allp))
        if not ok:
            self.spend_qmc()
            self.evaluate()
            return self.best[0] if self.best else None
        svals = vals[:len(sl)].reshape(d, len(nodes0))
        gvals = vals[len(sl):]
        self.nodes0, self.svals0 = nodes0, svals
        self.gu, self.gp, self.gvals = gu, gp, gvals
        if not np.all(np.isfinite(svals)) or not np.all(np.isfinite(gvals)):
            self.spend_qmc()
            self.evaluate()
            return self.best[0] if self.best else None

        # ---------------- structural tests ----------------
        tests = []
        for fn in (self.test_const, self.test_quad, self.test_add,
                   self.test_prod, self.test_prodc):
            try:
                r = fn()
            except Exception:
                r = None
            if r is not None:
                tests.append(r)
        tests.sort(key=lambda t: t[0])
        self.tests = tests
        best_struct = tests[0] if tests else None

        # ---------------- ridge probe (only if structure unclear) --------
        ridge = None
        if best_struct is None or best_struct[0] > 1e-7:
            try:
                ridge = self.test_ridge()
            except Exception:
                ridge = None

        # ---------------- commit ----------------
        sres = best_struct[0] if best_struct is not None else 1e30
        accepted = sres <= 0.4
        exact = sres <= 1e-10
        if accepted:
            kind = best_struct[1]['kind']
            try:
                if kind in ('add', 'prod', 'prodc'):
                    self.refine_axes(kind, best_struct[1], sres, exact)
                elif 'I' in best_struct[1]:
                    self.models.append({
                        'name': kind, 'I': float(best_struct[1]['I']),
                        'err': max(sres * self.f_scale * self.vol, 1e-300),
                        'eval': best_struct[1].get('eval'), 'exact': exact})
                    self.evaluate()
            except GiveUp:
                raise
            except Exception:
                pass
        if ridge is not None:
            try:
                self.refine_ridge(ridge)
            except GiveUp:
                raise
            except Exception:
                pass
        if self.ora.remaining > 24 and (not accepted or sres > 1e-6) and ridge is None:
            self.spend_qmc()
        self.audit()
        self.evaluate()
        if DBG:
            sys.stderr.write("tests=%s ridge=%s best=%s used=%d models=%s\n" % (
                [(t[2], float('%.3g' % t[0])) for t in tests],
                'yes' if ridge else 'no',
                self.best[2] if self.best else None, self.ora.used,
                [(m['name'], float('%.4g' % m['I']), float('%.3g' % m['err']),
                  float('%.3g' % m.get('rrms', -1)), m.get('eff', 0.0)) for m in self.models]))
        return self.best[0] if self.best else None

    # -- residual measure on the exact test grid --------------------------
    def resid(self, pred, truth=None):
        truth = self.gvals if truth is None else truth
        r = np.asarray(pred, float) - np.asarray(truth, float)
        if not np.all(np.isfinite(r)):
            return 1e30
        return float(np.sqrt(np.mean(r ** 2)) / self.f_scale)

    # -- individual structural tests --------------------------------------
    def test_const(self):
        r = self.resid(np.full(len(self.gvals), self.fb))
        mu = float(np.mean(self.Fc))
        return (r, {'kind': 'const', 'I': mu * self.vol,
                    'eval': lambda Z: np.full(len(np.asarray(Z).reshape(-1, self.d)), mu)},
                'const')

    def test_quad(self):
        d = self.d
        X, F = self.Xc, self.Fc
        npar = 1 + d + d * (d + 1) // 2
        if len(X) < npar + 4:
            return None

        def basis(Z):
            Z = np.asarray(Z, float).reshape(-1, d)
            cols = [np.ones(len(Z))]
            for i in range(d):
                cols.append(Z[:, i])
            for i in range(d):
                for j in range(i, d):
                    cols.append(Z[:, i] * Z[:, j])
            return np.stack(cols, axis=1)

        A = basis(X)
        c, *_ = np.linalg.lstsq(A, F, rcond=None)
        if not np.all(np.isfinite(c)):
            return None
        r = self.resid(basis(self.gp) @ c)
        I = c[0]
        k = 1
        for i in range(d):
            I += c[k] * 0.5
            k += 1
        for i in range(d):
            for j in range(i, d):
                I += c[k] * (1.0 / 3.0 if i == j else 0.25)
                k += 1
        return (r, {'kind': 'quad', 'I': I * self.vol, 'eval': lambda Z: basis(Z) @ c}, 'quad')

    def test_add(self):
        s = self.svals0
        fb = self.fb
        acc = np.full(len(self.gp), float(fb))
        for i in range(self.d):
            acc = acc + (s[i][self.gu[:, i]] - fb)
        r = self.resid(acc)
        return (r, {'kind': 'add'}, 'add')

    def test_prod(self):
        if abs(self.fb) < 1e-10 * self.f_scale:
            return None
        s = self.svals0
        p = np.ones(len(self.gp))
        for i in range(self.d):
            p = p * s[i][self.gu[:, i]]
        p = p / (self.fb ** (self.d - 1))
        r = self.resid(p)
        return (r, {'kind': 'prod'}, 'prod')

    def test_prodc(self):
        s = self.svals0
        fb = self.fb
        amp = max(self.f_scale, float(np.max(np.abs(self.gvals))), 1e-12)
        d1 = self.d - 1

        def res_at(c):
            if abs(fb - c) < 1e-9 * amp:
                return 1e30, None
            sc = s - c
            p = np.ones(len(self.gp))
            for i in range(self.d):
                p = p * sc[i][self.gu[:, i]]
            p = p / ((fb - c) ** d1) + c
            return self.resid(p), p

        best = (1e30, 0.0)
        grid = np.unique(np.concatenate([
            [0.0, fb], np.linspace(-2.5 * amp, 2.5 * amp, 101),
            np.linspace(float(np.min(self.gvals)) - amp, float(np.max(self.gvals)) + amp, 61)]))
        for c in grid:
            r, _ = res_at(float(c))
            if r < best[0]:
                best = (r, float(c))
        step = amp / 40.0
        r0, c0 = best
        for _ in range(80):
            improved = False
            for cc in (c0 - step, c0 + step):
                r, _ = res_at(float(cc))
                if r < r0:
                    r0, c0, improved = r, float(cc), True
            if not improved:
                step *= 0.5
            if step < 1e-13 * amp:
                break
        return (r0, {'kind': 'prodc', 'c': c0}, 'prodc')

    # -- ridge detection ---------------------------------------------------
    def test_ridge(self):
        """Detect f(x) = G(v.x) from finite-difference gradients.

        Probe points are spread over the whole range of |f| (not just its
        maximum, where the gradient of an oscillatory ridge vanishes)."""
        d = self.d
        if d < 2:
            return None
        order = np.argsort(np.abs(self.Fc))
        cand = [self.Xc[order[int(q * (len(order) - 1))]]
                for q in (0.35, 0.5, 0.65, 0.8, 0.9, 0.97)]
        bases = []
        for u in cand:
            if all(np.max(np.abs(u - b)) > 0.05 for b in bases):
                bases.append(u.copy())
            if len(bases) >= 5:
                break
        if len(bases) < 2:
            return None
        h = 1e-4
        grads = []
        cost = 2 * d * len(bases)
        if self.ora.remaining < cost + 128:
            bases = bases[:max(2, (self.ora.remaining - 128) // (2 * d))]
            if len(bases) < 2:
                return None
        pts = []
        for b in bases:
            for i in range(d):
                pp = b.copy(); pp[i] = float(np.clip(pp[i] + h, 0.0, 1.0))
                pm = b.copy(); pm[i] = float(np.clip(pm[i] - h, 0.0, 1.0))
                pts.append(pp); pts.append(pm)
        pts = np.asarray(pts)
        if self.ora.remaining < len(pts) + 32:
            return None
        vals, ok = self.ora.get(self.to_box(pts))
        if not ok or not np.all(np.isfinite(vals)):
            return None
        for k in range(len(bases)):
            g = np.zeros(d)
            for i in range(d):
                j = 2 * (k * d + i)
                dt = pts[j][i] - pts[j + 1][i]
                if abs(dt) > 1e-9:
                    g[i] = (vals[j] - vals[j + 1]) / dt
            ng = float(np.linalg.norm(g))
            if ng > 1e-6 * self.f_scale:
                grads.append(g)
        if len(grads) < 2:
            return None
        G = np.asarray(grads)
        try:
            U, sv, Vt = np.linalg.svd(G, full_matrices=False)
        except Exception:
            return None
        tot = float(np.sum(sv ** 2))
        if tot <= 0:
            return None
        # a single ridge means every gradient is parallel to v
        if sv[0] ** 2 / tot < 0.995:
            return None
        v = Vt[0].copy()
        nv = float(np.linalg.norm(v))
        if nv < 1e-12:
            return None
        v = v / nv
        # sign convention: largest-magnitude component positive
        k = int(np.argmax(np.abs(v)))
        if v[k] < 0:
            v = -v
        spread = float(np.max(np.linalg.norm(G / np.maximum(
            np.linalg.norm(G, axis=1, keepdims=True), 1e-300)
            * np.sign((G / np.maximum(np.linalg.norm(G, axis=1, keepdims=True), 1e-300))
                      @ v)[:, None] - v[None, :], axis=1)))
        return {'v': v, 'res': spread}

    # -- refinement of axis-based structural models ------------------------
    def make_channels(self, kind, info):
        d = self.d
        chs = []
        off = info.get('c', 0.0) if kind == 'prodc' else 0.0
        for i in range(d):
            ch = Channel(0.0, 1.0, off=off)
            for t, y in zip(self.nodes0, self.svals0[i]):
                ch.tv[round(float(t), 12)] = float(y)
            chs.append(ch)
        for ch in chs:
            ch.recompute()
        return chs

    def refine_axes(self, kind, info, sresid, exact=False):
        d = self.d
        chs = self.make_channels(kind, info)
        c0 = info.get('c', 0.0)
        base = (self.fb - c0) if kind == 'prodc' else self.fb
        pexp = float(self.fb - c0) ** (d - 1) if kind == 'prodc' else float(self.fb) ** (d - 1)
        reserve = min(64, max(0, self.ora.remaining // 16))

        def build():
            S = np.array([c.Q for c in chs])
            E = np.array([c.err for c in chs])
            if not np.all(np.isfinite(S)) or abs(base) < 1e-300:
                return None
            if kind == 'add':
                I = float(np.sum(S) - (d - 1) * self.fb)
                err = float(np.sum(E))
                floor = 1e-2 * max(float(np.max(np.abs(S))), 1e-300)
                prio = E / np.maximum(np.abs(S), floor)
            else:
                I = float(np.prod(S)) / pexp
                if kind == 'prodc':
                    I += c0
                floor = 1e-2 * max(float(np.max(np.abs(S))), 1e-300)
                rel = float(np.sum(E / np.maximum(np.abs(S), floor)))
                err = abs(I) * rel
                prio = E / np.maximum(np.abs(S), floor)
            if not np.isfinite(I):
                return None
            err += sresid * self.f_scale
            return I, err, prio

        def evaluator(U):
            U = np.asarray(U, float).reshape(-1, d)
            if kind == 'add':
                acc = np.zeros(len(U))
                for i in range(d):
                    acc = acc + chs[i].eval(U[:, i])
                return acc - (d - 1) * self.fb
            p = np.ones(len(U))
            for i in range(d):
                p = p * chs[i].eval(U[:, i])
            p = p / pexp
            if kind == 'prodc':
                p = p + c0
            return p

        model = {'name': kind, 'I': 0.0, 'err': 1e30, 'eval': evaluator, 'exact': exact}
        self.models.append(model)

        def publish():
            st = build()
            if st is None:
                return
            I, err, _ = st
            if not (np.isfinite(I) and np.isfinite(err)):
                return
            if err * self.vol < model['err']:
                model['I'] = I * self.vol
                model['err'] = err * self.vol

        publish()
        for _ in range(200):
            if timed_out(2.0):
                break
            avail = self.ora.remaining - reserve
            if avail < 8:
                break
            st = build()
            if st is None:
                break
            prio = np.asarray(st[2], float)
            cap = min(480, avail)
            plan = []
            plan_ch = []
            pset = set()
            order = np.argsort(-prio)
            pmax = float(prio[order[0]]) if len(order) else 0.0
            gate = pmax * 1e-3
            for _pass in range(2):
                for i in order:
                    if len(plan) >= cap:
                        break
                    if pmax > 0.0 and float(prio[i]) < gate:
                        continue
                    ch = chs[i]
                    if ch.done or ch.err <= ch.tol_eff():
                        ch.done = True
                        continue
                    room = cap - len(plan)
                    if room <= 0:
                        break
                    if plan and ch.missing_cost() > room:
                        continue
                    newts = ch.refine_guarded(avail - len(plan))
                    if not newts:
                        continue
                    fresh = [t for t in newts
                             if (int(i), round(float(t), 12)) not in pset]
                    if not fresh:
                        continue
                    for t in fresh:
                        pset.add((int(i), round(float(t), 12)))
                    plan.extend(fresh)
                    plan_ch.extend([int(i)] * len(fresh))
            if DBG:
                sys.stderr.write("   round n=%d plan=%d prio=%s errs=%s done=%s rem=%d\n" % (
                    _, len(plan), ['%.2g' % x for x in np.asarray(prio, float)],
                    ['%.2g' % c.err for c in chs], [c.done for c in chs], self.ora.remaining))
            if not plan:
                break
            pts = np.empty((len(plan), d))
            pts[:] = self.anchor[None, :]
            pts[np.arange(len(plan)), plan_ch] = np.asarray(plan)
            vals, ok = self.ora.get(self.to_box(pts))
            good = np.isfinite(vals)
            for j in range(len(plan)):
                if good[j]:
                    chs[plan_ch[j]].update([plan[j]], [vals[j]])
            for ch in chs:
                ch.recompute()
            publish()
            if not ok or not np.all(good):
                break
        # repair: never leave a planned refinement half-evaluated
        for _ in range(40):
            if timed_out(1.0):
                break
            need = []
            need_ch = []
            for i, ch in enumerate(chs):
                for t in ch.missing():
                    need.append(t)
                    need_ch.append(i)
            if not need or len(need) > self.ora.remaining:
                break
            pts = np.empty((len(need), d))
            pts[:] = self.anchor[None, :]
            pts[np.arange(len(need)), need_ch] = np.asarray(need)
            vals, ok = self.ora.get(self.to_box(pts))
            for j in range(len(need)):
                if np.isfinite(vals[j]):
                    chs[need_ch[j]].update([need[j]], [vals[j]])
            for ch in chs:
                ch.recompute()
        publish()
        if DBG:
            sys.stderr.write("  axes: %s\n" % [(float('%.3g' % c.err), float('%.6g' % c.Q),
                                                [(iv.N, float('%.1e' % iv.err)) for iv in c.ivs][:6],
                                                len(c.ivs)) for c in chs])
        self.evaluate()

    # -- ridge refinement --------------------------------------------------
    def refine_ridge(self, ridge):
        d = self.d
        v = ridge['v']
        pdf, tlo, thi, knots = ridge_density(v)
        span = thi - tlo
        if span <= 0:
            return
        ch = Channel(tlo, thi, weight=pdf)
        if len(knots) <= 64:
            inner = sorted(set(float(k) for k in knots
                               if tlo + 1e-9 * span < k < thi - 1e-9 * span))
            edges = [tlo] + inner + [thi]
            ivs = [Iv(a, b, N0) for a, b in zip(edges[:-1], edges[1:])
                   if b - a > 1e-13 * max(1.0, span)]
            if ivs:
                ch.ivs = ivs
        reserve = min(64, max(0, self.ora.remaining // 16))

        def evaluator(U):
            U = np.asarray(U, float).reshape(-1, d)
            return ch.eval(U @ v)

        model = {'name': 'ridge', 'I': 0.0, 'err': 1e30, 'eval': evaluator}
        self.models.append(model)

        def rpub():
            e = ch.err * self.vol + ridge.get('res', 0.0) * self.f_scale * self.vol
            if np.isfinite(ch.Q) and np.isfinite(e) and e < model['err']:
                model['I'] = ch.Q * self.vol
                model['err'] = e

        for _ in range(200):
            if timed_out(2.0):
                break
            rpub()
            self.evaluate()
            avail = self.ora.remaining - reserve
            if avail < 8 or ch.done or ch.err <= ch.tol_eff():
                break
            plan = []
            cap = min(480, avail)
            while len(plan) < cap and not ch.done:
                if ch.err <= ch.tol_eff():
                    ch.done = True
                    break
                newts = ch.refine_guarded(avail - len(plan))
                if not newts:
                    break
                plan.extend(newts)
            if not plan:
                break
            pts = ridge_points(v, np.asarray(plan))
            vals, ok = self.ora.get(self.to_box(pts))
            good = np.isfinite(vals)
            if np.any(good):
                gi = np.flatnonzero(good)
                ch.update([plan[j] for j in gi], vals[gi])
            ch.recompute()
            if not ok or not np.all(good):
                break
        for _ in range(40):
            if timed_out(1.0):
                break
            need = ch.missing()
            if not need or len(need) > self.ora.remaining:
                break
            pts = ridge_points(v, np.asarray(need))
            vals, ok = self.ora.get(self.to_box(pts))
            gi = np.flatnonzero(np.isfinite(vals))
            if len(gi) == 0:
                break
            ch.update([need[j] for j in gi], vals[gi])
            ch.recompute()
        ch.recompute()
        model['I'] = ch.Q * self.vol
        model['err'] = ch.err * self.vol + ridge.get('res', 0.0) * self.f_scale * self.vol
        self.evaluate()

    # -- extra QMC / audit -------------------------------------------------
    def more_net(self, nmax):
        """Evaluate the next prefix chunk of the digital net."""
        n = int(min(nmax, len(self.net) - self.nnet, self.ora.remaining))
        if n <= 0:
            return 0
        U = self.net[self.nnet:self.nnet + n]
        F, ok = self.ora.get(self.to_box(U))
        k = min(len(F), len(U))
        if k > 0:
            self.add_cv_points(U[:k], F[:k])
            self.nnet += k
        return k

    def spend_qmc(self):
        self.more_net(self.ora.remaining)
        self.qmc_record('qmc-full')

    def audit(self):
        """Fresh net points: independent control-variate sample / model check."""
        if self.ora.remaining >= 8:
            self.more_net(self.ora.remaining)
        self.qmc_record('qmc-all')


def main():
    line = sys.stdin.readline()
    if not line:
        return
    init = json.loads(line)
    s = Solver(init)
    est = None
    try:
        est = s.solve()
    except GiveUp:
        pass
    except Exception:
        pass
    try:
        s.evaluate()
        if s.best is not None:
            est = s.best[0]
    except Exception:
        pass
    if est is None or not np.isfinite(est):
        try:
            est = float(np.mean(s.Fc)) * s.vol if len(s.Fc) else 0.0
        except Exception:
            est = 0.0
    if est is None or not np.isfinite(est):
        est = 0.0
    sys.stdout.write(json.dumps({"type": "result", "estimate": float(est)}) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
