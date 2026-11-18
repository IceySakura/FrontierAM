#!/usr/bin/env python3
"""Black-box cubature on [0,1]^d under a hard function-evaluation budget.

Cheap Sobol probe, then structure detection validated with *exact*
(interpolation-free) residuals for two model classes:

  (a) rank-1 product   f(x) = c + prod_i (h_i(x_i)-c)/(f(b)-c)^(d-1)
  (b) rank-1 ridge     f(x) = G(v.x), integrated against the exact
      box-spline density of v.x over the unit cube,

each collapsing the d-dimensional problem to 1-D adaptive Gauss-Legendre
quadratures that reach near machine precision.  If neither validates, fall
back to scrambled-Sobol QMC with a polynomial least-squares rule that is
only trusted when it demonstrably beats the sampling noise.

The answer is always  exact integral of the model + QMC mean of (f - model),
so an imperfect model degrades gracefully instead of failing.
"""
import heapq
import json
import math
import sys
import time
from itertools import combinations

import numpy as np

_GL = {}
_BW = {}
MAXB = 512
LD = np.longdouble
TIME_LIMIT = 21.5
_DEADLINE = [time.time() + TIME_LIMIT]


def _ok(margin=0.0):
    dl = _DEADLINE[0]
    return True if dl is None else time.time() < dl - margin

NODES = {2: 10, 3: 9, 4: 8, 5: 7, 6: 6, 7: 6, 8: 6}


def gl(n):
    r = _GL.get(n)
    if r is None:
        x, w = np.polynomial.legendre.leggauss(n)
        r = ((x + 1.0) / 2.0, w / 2.0)
        _GL[n] = r
    return r


def baryw(n):
    r = _BW.get(n)
    if r is None:
        x, _ = gl(n)
        diff = x[:, None] - x[None, :]
        np.fill_diagonal(diff, 1.0)
        r = 1.0 / np.prod(diff, axis=1)
        _BW[n] = r
    return r


def bary_eval(xs, bw, ys, xq):
    dx = xq[:, None] - xs[None, :]
    small = np.abs(dx) < 1e-13
    dxs = np.where(small, 1.0, dx)
    num = (bw[None, :] * ys[None, :] / dxs).sum(axis=1)
    den = (bw[None, :] / dxs).sum(axis=1)
    out = num / den
    if small.any():
        rows, cols = np.nonzero(small)
        out[rows] = ys[cols]
    return out


def ev(o, X):
    """Evaluate points in protocol-legal chunks."""
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if len(X) <= MAXB:
        return o(X)
    out = np.empty(len(X))
    for s in range(0, len(X), MAXB):
        c = X[s:s + MAXB]
        out[s:s + len(c)] = o(c)
    return out


_SOB = {}


def sobol(d, n, seed):
    """Nested scrambled Sobol: sobol(d,n)[ :m] == sobol(d,m) for m<=n."""
    from scipy.stats import qmc
    return qmc.Sobol(d, scramble=True, seed=int(seed) % (2 ** 31)).random(n)


def sobol_seq(d, seed, total):
    k = (d, seed, total)
    r = _SOB.get(k)
    if r is None:
        r = sobol(d, total, seed)
        _SOB[k] = r
    return r


# --------------------------------------------------------------------------
class A1D:
    """Adaptive Gauss-Legendre quadrature of phi*wfun on [lo,hi].

    Cells keep the raw phi values so a piecewise-polynomial interpolant of
    phi can be evaluated anywhere (needed for model validation)."""

    def __init__(self, phi, n=6, lo=0.0, hi=1.0, wfun=None):
        self.phi = phi
        self.wfun = wfun
        self.n = n
        self.lo = lo
        self.hi = hi
        self.cells = []
        self.heap = []
        self._x, self._w = gl(n)
        self._bw = baryw(n)

    def add_initial(self, breaks=()):
        edges = [self.lo]
        for t in sorted(float(x) for x in breaks):
            if edges[-1] + 1e-13 < t < self.hi - 1e-13:
                edges.append(t)
        edges.append(self.hi)
        for a, b in zip(edges[:-1], edges[1:]):
            self._new(a, b, math.inf)

    def _new(self, a, b, err):
        pts = a + (b - a) * self._x
        vals = self.phi(pts)
        w = self._w if self.wfun is None else self._w * self.wfun(pts)
        Q = float(np.dot(w, vals)) * (b - a)
        c = {"a": a, "b": b, "Q": Q, "err": err, "pts": pts, "vals": vals,
             "ver": 0, "live": True, "ierr": math.inf}
        i = len(self.cells)
        self.cells.append(c)
        heapq.heappush(self.heap, (-(err if np.isfinite(err) else 1e300), c["ver"], i))
        return i

    def worst(self):
        while self.heap:
            ne, ver, i = self.heap[0]
            c = self.cells[i]
            if c["live"] and c["ver"] == ver:
                return i
            heapq.heappop(self.heap)
        return None

    def refine(self, i=None):
        if i is None:
            i = self.worst()
            if i is None:
                return None
        c = self.cells[i]
        a, b = c["a"], c["b"]
        if b - a < 1e-16:
            c["err"] = 0.0
            c["ver"] += 1
            return 0.0
        m = 0.5 * (a + b)
        c["live"] = False
        il = self._new(a, m, math.inf)
        ir = self._new(m, b, math.inf)
        e = 0.5 * abs(self.cells[il]["Q"] + self.cells[ir]["Q"] - c["Q"])
        e = max(e, 1e-300)
        for j in (il, ir):
            cc = self.cells[j]
            cc["err"] = e
            cc["ver"] += 1
            # measured interpolation error of the parent at this child's nodes
            pred = bary_eval(c["pts"], self._bw, c["vals"], cc["pts"])
            cc["ierr"] = float(np.max(np.abs(pred - cc["vals"])))
            heapq.heappush(self.heap, (-e, cc["ver"], j))
        return e

    def interp_err(self):
        """Max measured pointwise interpolation error over live cells."""
        e = 0.0
        for c in self.cells:
            if c["live"]:
                if np.isfinite(c["ierr"]):
                    e = max(e, c["ierr"])
        return e

    def value(self):
        return sum(c["Q"] for c in self.cells if c["live"])

    def error(self):
        e = 0.0
        for c in self.cells:
            if c["live"]:
                if not np.isfinite(c["err"]):
                    return math.inf
                e += c["err"]
        return e

    def eval_at(self, xq):
        xq = np.atleast_1d(np.asarray(xq, float))
        out = np.zeros_like(xq)
        live = sorted((c for c in self.cells if c["live"]), key=lambda c: c["a"])
        if not live:
            return out
        edges = np.array([c["a"] for c in live] + [live[-1]["b"]])
        idx = np.clip(np.searchsorted(edges, xq, side="right") - 1, 0, len(live) - 1)
        for k, c in enumerate(live):
            sel = idx == k
            if sel.any():
                out[sel] = bary_eval(c["pts"], self._bw, c["vals"], xq[sel])
        return out


# --------------------------------------------------------------------------
class BoxSpline:
    """Density of s = v.x for x ~ U[0,1]^d."""

    def __init__(self, v):
        v = np.asarray(v, float)
        neg = v < 0
        w = np.abs(v).astype(LD)
        self.w = np.abs(v)
        self.off = float(-w[neg].sum())
        self.S = float(w.sum())
        d = len(v)
        self.d = d
        prod = float(np.prod(w))
        self.coef = 1.0 / (math.factorial(d - 1) * prod) if prod > 0 else 0.0
        self.subs = []
        for r in range(d + 1):
            sgn = -1.0 if (r % 2) else 1.0
            for J in combinations(range(d), r):
                self.subs.append((LD(sgn), math.fsum([float(w[j]) for j in J]) if r else LD(0)))

    def knots(self, limit=40):
        ks = set()
        if 2 ** self.d <= limit:
            for _, W in self.subs:
                ks.add(W)
        else:
            d = self.d
            for r in (1, d - 1):
                for J in combinations(range(d), r):
                    ks.add(sum(self.w[j] for j in J))
        return sorted(t for t in ks if 1e-12 < t < self.S - 1e-12)

    def __call__(self, s):
        # extended precision: the alternating subset sum cancels badly for d>=5
        s = np.asarray(s, LD)
        t0 = s - self.off
        out = np.zeros_like(t0)
        dm1 = self.d - 1
        for sgn, W in self.subs:
            u = t0 - W
            out += sgn * np.where(u > 0.0, u ** dm1, 0.0)
        return np.asarray(out * self.coef, dtype=np.float64)


# --------------------------------------------------------------------------
def _fiber(o, d, b, i):
    def phi(t):
        X = np.tile(b, (len(t), 1))
        X[:, i] = t
        return ev(o, X)
    return phi


def _nodes_of(a1):
    """All (node, value) pairs evaluated so far for each coordinate."""
    out = []
    for q in a1:
        nn, vv = [], []
        for c in q.cells:
            nn.append(c["pts"])
            vv.append(c["vals"])
        out.append((np.concatenate(nn), np.concatenate(vv)))
    return out


def _exact_prod_resid(o, a1, cval, Fb, d, K, seed=0):
    """Exact (interpolation free) residual of the rank-1 product model at K
    points built from already-evaluated fibre nodes."""
    nd = _nodes_of(a1)
    rng = np.random.default_rng(1234 + seed)
    X = np.empty((K, d))
    P = np.ones(K)
    for i in range(d):
        nn, vv = nd[i]
        j = rng.integers(0, len(nn), size=K)
        X[:, i] = nn[j]
        P *= (vv[j] - cval)
    D = (Fb - cval) ** (d - 1)
    M = cval + P / D
    f = ev(o, X)
    return f - M, X, M


def try_product(o, d, P0, f0, scale, seed=0, n=None, verbose=False):
    """f(x) = c + prod_i (h_i(x_i)-c) / (f(b)-c)^(d-1).  Estimate or None."""
    n = n or NODES.get(d, 6)
    K = 48
    if o.remaining < d * n + K + 6 * n * d + 128:
        return None
    span = max(float(np.max(f0) - np.min(f0)), scale, 1e-30)
    ref_scale = scale + span

    med = float(np.median(f0))
    cands = [np.full(d, 0.5)]
    for k in np.argsort(-np.abs(f0 - med))[:3]:
        cands.append(P0[k].copy())
    best = None
    for b in cands:
        v = float(ev(o, b.reshape(1, -1))[0])
        if best is None or abs(v - med) > abs(best[0] - med):
            best = (v, b)
    Fb, b = best

    a1 = [A1D(_fiber(o, d, b, i), n=n) for i in range(d)]
    for q in a1:
        q.add_initial()
    # two refinement rounds so that localised structure (narrow spikes, jumps)
    # shows up in the fibre node set used by the exact residual test below
    for _ in range(2):
        if not _ok():
            break
        for q in a1:
            if o.remaining < 2 * n:
                break
            q.refine()

    # ---- pick the constant c with an exact residual (K oracle calls)
    lo_t, hi_t = float(np.min(f0)), float(np.max(f0))
    grid = np.concatenate([[0.0], np.linspace(lo_t - 0.6 * span, hi_t + 0.6 * span, 121)])
    Xk = None
    results = []
    # one evaluation set, reused: build X from nodes once per c (nodes are fixed)
    nd = _nodes_of(a1)
    rng = np.random.default_rng(1234 + seed)
    pick = [rng.integers(0, len(nd[i][0]), size=K) for i in range(d)]
    Xk = np.empty((K, d))
    Hk = np.empty((K, d))
    for i in range(d):
        Xk[:, i] = nd[i][0][pick[i]]
        Hk[:, i] = nd[i][1][pick[i]]
    fk = ev(o, Xk)
    cval, mres = 0.0, math.inf
    for c in grid:
        D = (Fb - c) ** (d - 1)
        if not np.isfinite(D) or abs(D) < 1e-280:
            continue
        with np.errstate(over="ignore", invalid="ignore"):
            M = c + np.prod(Hk - c, axis=1) / D
        if not np.all(np.isfinite(M)):
            continue
        m = float(np.mean(np.abs(fk - M)) / (ref_scale + np.max(np.abs(M))))
        if m < mres:
            mres, cval = m, float(c)
    step = max(span / 120.0, 1e-14)
    for _ in range(45):
        imp = False
        for s in (cval - step, cval + step):
            D = (Fb - s) ** (d - 1)
            if not np.isfinite(D) or abs(D) < 1e-280:
                continue
            with np.errstate(over="ignore", invalid="ignore"):
                M = s + np.prod(Hk - s, axis=1) / D
            if not np.all(np.isfinite(M)):
                continue
            m = float(np.mean(np.abs(fk - M)) / (ref_scale + np.max(np.abs(M))))
            if m < mres:
                mres, cval, imp = m, float(s), True
        if not imp:
            step *= 0.5
            if step < 1e-15 * max(1.0, abs(cval)):
                break
    if verbose:
        print(f"  prod exact-resid: c={cval:.5g} mres={mres:.3e} pts={o.used}")
    if mres > 1e-7:
        return None

    # ---- full refinement, sensitivity weighted
    def total_of(I, c):
        D = (Fb - c) ** (d - 1)
        return c + float(np.prod(I - c)) / D

    def sens(I, c):
        D = (Fb - c) ** (d - 1)
        P = np.prod(I - c)
        out = np.zeros(d)
        for i in range(d):
            den = I[i] - c
            out[i] = 0.0 if abs(den) < 1e-300 else abs(P / den / D)
        return out

    Kr = 64
    reserve = Kr + 8
    guard = 0
    while o.remaining > reserve + 2 * n and guard < 100000 and _ok():
        guard += 1
        before = o.used
        I = np.array([q.value() for q in a1])
        errs = np.array([q.error() for q in a1])
        w = np.nan_to_num(sens(I, cval) * errs, nan=0.0, posinf=1e300, neginf=0.0)
        j = int(np.argmax(w))
        a1[j].refine()
        if o.used == before:
            errs[j] = 0.0
            if np.all(errs <= 0):
                break

    I = np.array([q.value() for q in a1])
    est_model = total_of(I, cval)
    if not np.isfinite(est_model):
        return None
    # ---- exact residual for the correction
    try:
        delta, _, _ = _exact_prod_resid(o, a1, cval, Fb, d, Kr, seed=seed)
        mres2 = float(np.mean(np.abs(delta)) / ref_scale)
    except Exception:
        delta, mres2 = np.zeros(1), math.inf
    corr = float(np.mean(delta)) if np.all(np.isfinite(delta)) else 0.0
    if verbose:
        print(f"  product accept: c={cval:.6g} mres={mres:.2e} mres2={mres2:.2e} "
              f"pts={o.used} model={est_model:.12e} corr={corr:.3e}")
    return est_model + corr


# --------------------------------------------------------------------------
def try_ridge(o, d, P0, f0, scale, seed=0, n=None, verbose=False):
    n = n or NODES.get(d, 6)
    if o.remaining < 10 * d + 8 * n + 128:
        return None
    span = max(float(np.max(f0) - np.min(f0)), scale, 1e-30)
    ref_scale = scale + span
    m = 4
    idx = np.unique(np.linspace(0, len(P0) - 1, m).astype(int))
    h = 1e-6
    G = np.empty((len(idx), d))
    for r, k in enumerate(idx):
        x = P0[k]
        E = h * np.eye(d)
        G[r] = (ev(o, np.clip(np.tile(x, (d, 1)) + E, 0, 1))
                - ev(o, np.clip(np.tile(x, (d, 1)) - E, 0, 1))) / (2 * h)
    nrm = np.linalg.norm(G, axis=1)
    mx = float(np.max(nrm)) if len(nrm) else 0.0
    if mx <= 0:
        return None
    keep = nrm > 1e-7 * mx
    Gk = G[keep] / nrm[keep][:, None]
    _, S, Vt = np.linalg.svd(Gk, full_matrices=False)
    if S.size > 1 and S[1] > 0.05 * S[0]:
        return None
    v = Vt[0] / np.linalg.norm(Vt[0])

    def mk(vv):
        bs = BoxSpline(vv)
        s_lo, s_hi = bs.off, bs.off + bs.S
        if not (s_hi > s_lo) or not np.isfinite(s_lo):
            return None
        rng_ = s_hi - s_lo
        pos = vv >= 0
        flat = np.abs(vv) < 1e-15

        def xof(s):
            L = (np.asarray(s, float) - s_lo) / rng_
            L = np.clip(L, 0.0, 1.0)
            X = np.where(pos[None, :], L[:, None], 1.0 - L[:, None])
            if flat.any():
                X[:, flat] = 0.5
            return np.clip(X, 0.0, 1.0)
        return bs, s_lo, s_hi, rng_, xof

    # ---- Newton refinement of the direction using exact residuals
    K = min(24, max(10, o.remaining // 40))
    xs = P0[:K]
    fs = f0[:K]
    gp = None
    mres = math.inf
    for it in range(2):
        if not _ok():
            break
        made = mk(v)
        if made is None:
            return None
        bs, s_lo, s_hi, rng_, xof = made
        s = np.clip(xs @ v, s_lo + 1e-12 * rng_, s_hi - 1e-12 * rng_)
        X = xof(s)
        fX = ev(o, X)
        delta = fs - fX
        mres = float(np.mean(np.abs(delta)) / (ref_scale + np.max(np.abs(fX))))
        if verbose:
            print(f"  ridge newton {it}: mres={mres:.3e} pts={o.used}")
        if mres <= 1e-13 or o.remaining < 3 * K + 8 * n:
            break
        if gp is None or it == 0:
            eps = 1e-5 * rng_
            sp = np.clip(s + eps, s_lo, s_hi)
            sm = np.clip(s - eps, s_lo, s_hi)
            gp = (ev(o, xof(sp)) - ev(o, xof(sm))) / (2 * eps)
        A = gp[:, None] * (xs - X)
        try:
            dv, *_ = np.linalg.lstsq(A, delta, rcond=None)
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(dv)):
            break
        vn = v + dv
        nn = float(np.linalg.norm(vn))
        if nn <= 0 or np.linalg.norm(dv) > 0.3 * nn:
            break
        v = vn / nn
    if not (mres <= 1e-8):
        if verbose:
            print(f"  ridge REJECT: mres={mres:.3e}")
        return None

    made = mk(v)
    if made is None:
        return None
    bs, s_lo, s_hi, rng_, xof = made
    q = A1D(lambda s: ev(o, xof(s)), n=n, lo=s_lo, hi=s_hi, wfun=bs)
    q.add_initial([s_lo + t for t in bs.knots()])
    Kr = 64
    reserve = Kr + 8
    guard = 0
    while o.remaining > reserve + 2 * n and guard < 200000 and _ok():
        guard += 1
        before = o.used
        if q.refine() is None or o.used == before:
            break
    est = q.value()
    if not np.isfinite(est):
        return None
    # exact residual on a fresh set for the correction
    Kx = min(Kr, len(P0))
    xs2 = P0[:Kx]
    s2 = np.clip(xs2 @ v, s_lo + 1e-12 * rng_, s_hi - 1e-12 * rng_)
    try:
        d2 = f0[:Kx] - ev(o, xof(s2))
        mres2 = float(np.mean(np.abs(d2)) / ref_scale)
        corr = float(np.mean(d2)) if mres2 <= 1e-4 else 0.0
    except Exception:
        mres2, corr = math.inf, 0.0
    if verbose:
        print(f"  ridge accept: mres={mres:.2e} mres2={mres2:.2e} pts={o.used} "
              f"model={est:.12e} corr={corr:.3e}")
    return est + corr


# --------------------------------------------------------------------------
def _legendre_index_set(d, M, lmax=28):
    """Hyperbolic-cross multi-indices, ordered by prod(1+l_i)."""
    S = 1
    while math.comb(S + d + 1, d) <= 8 * M + 200 and S < 60:
        S += 1
    idx = []

    def rec(i, rem, acc):
        if i == d - 1:
            for l in range(rem + 1):
                if l <= lmax:
                    acc.append(l)
                    idx.append(tuple(acc))
                    acc.pop()
            return
        for l in range(min(rem, lmax) + 1):
            acc.append(l)
            rec(i + 1, rem - l, acc)
            acc.pop()
    rec(0, S, [])
    idx.sort(key=lambda t: (_p1(t), sum(t), t))
    return idx[:M]


def _p1(t):
    r = 1
    for x in t:
        r *= (1 + x)
    return r


def poly_basis(X, idx):
    N, d = X.shape
    maxl = max(max(t) for t in idx)
    z = 2.0 * X - 1.0
    T = np.empty((d, maxl + 1, N))
    for i in range(d):
        T[i, 0] = 1.0
        if maxl >= 1:
            T[i, 1] = z[:, i] * math.sqrt(3.0)
        for k in range(1, maxl):
            T[i, k + 1] = ((2 * k + 1) * z[:, i] * T[i, k] - k * T[i, k - 1]) * \
                math.sqrt((2.0 * k + 3) / (2.0 * k + 1.0)) / (k + 1.0)
    B = np.empty((N, len(idx)))
    for j, t in enumerate(idx):
        col = np.ones(N)
        for i in range(d):
            if t[i]:
                col = col * T[i, t[i]]
        B[:, j] = col
    return B


def poly_cv(o, d, P0, f0, scale, seed=0, verbose=False):
    """Scrambled-Sobol QMC + a well-conditioned polynomial least-squares
    (control-variate) rule.  Falls back to the plain QMC mean whenever the
    correction looks untrustworthy."""
    rem = o.remaining
    n0 = len(P0)
    n = min(rem, SEQ.shape[0] - n0)
    if n >= 8 and _ok(2.0):
        P = SEQ[n0:n0 + n]
        f = ev(o, P)
        X = np.vstack([P0, P]) if n0 else P
        y = np.concatenate([f0, f]) if n0 else f
    else:
        X, y = P0, f0
    N = len(y)
    if N < 8:
        return float(np.mean(y)) if N else 0.0
    qmc_mean = float(np.mean(y))
    M = max(1, min(N // 8, 400))
    idx = _legendre_index_set(d, M)
    B = poly_basis(X, idx)
    try:
        from scipy.linalg import qr
        Q, R, piv = qr(B, mode="economic", pivoting=True)
    except Exception:
        return qmc_mean
    diag = np.abs(np.diag(R))
    if diag.size == 0 or diag[0] <= 0:
        return qmc_mean
    keep = int(np.sum(diag > 1e-7 * diag[0]))
    keep = max(1, keep)
    try:
        z = np.linalg.solve(R[:keep, :keep], Q[:, :keep].T @ y)
    except np.linalg.LinAlgError:
        return qmc_mean
    if not np.all(np.isfinite(z)):
        return qmc_mean
    c = np.zeros(len(idx))
    c[piv[:keep]] = z
    r = y - B @ c
    est = float(c[0] + np.mean(r))
    if not np.isfinite(est):
        return qmc_mean
    # safety clamp: never let the control variate move the answer far beyond
    # what the sampling noise can justify.
    rms = float(np.sqrt(np.mean(r ** 2)))
    sd = float(np.std(y)) / math.sqrt(N)
    # |est_cv - true| <= ||f-p||_1 <= rms, so only use it when the polynomial
    # actually explains the integrand better than the sampling noise scale.
    if not (rms < sd):
        return qmc_mean
    lim = 3.0 * sd + 1e-15
    delta = est - qmc_mean
    if abs(delta) > lim:
        est = qmc_mean + math.copysign(lim, delta)
    if verbose:
        print(f"  poly_cv: N={N} M={len(idx)} keep={keep} rms={float(np.sqrt(np.mean(r**2))):.3e} "
              f"cv={float(c[0] + np.mean(r)):.8e} qmc={qmc_mean:.8e} sd={sd:.3e}")
    return est


def integrate(o, d, budget, seed=0, verbose=False):
    global P_FULL
    global SEQ, P_FULL, P0N
    n0 = min(128, max(16, budget // 8))
    n0 = 2 ** int(math.floor(math.log2(n0)))
    SEQ = sobol_seq(d, seed + 12345, max(budget, 1024))
    P_FULL = SEQ
    P0N = n0
    P0 = SEQ[:n0]
    f0 = ev(o, P0)
    scale = float(np.max(np.abs(f0)))
    if not np.isfinite(scale) or scale == 0.0:
        scale = 1e-30

    if d >= 2:
        try:
            e = try_product(o, d, P0, f0, scale, seed=seed, verbose=verbose)
            if e is not None and math.isfinite(e):
                return e
        except Exception as exc:
            if verbose:
                import traceback; traceback.print_exc()
    try:
        e = try_ridge(o, d, P0, f0, scale, seed=seed, verbose=verbose)
        if e is not None and math.isfinite(e):
            return e
    except Exception as exc:
        if verbose:
            import traceback; traceback.print_exc()
    return poly_cv(o, d, P0, f0, scale, seed=seed, verbose=verbose)


# ==========================================================================
class Oracle:
    """Batched point->value oracle speaking the JSON-Lines protocol."""

    def __init__(self, d, budget, max_batch=MAXB):
        self.d = d
        self.budget = int(budget)
        self.used = 0
        self.max_batch = max_batch
        self.vals = []

    @property
    def remaining(self):
        return self.budget - self.used

    def __call__(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        n = X.shape[0]
        if n == 0:
            return np.empty(0)
        if n > self.remaining:
            raise RuntimeError("budget exhausted")
        X = np.ascontiguousarray(X[:n])
        np.clip(X, 0.0, 1.0, out=X)
        if not np.all(np.isfinite(X)):
            raise RuntimeError("non-finite query point")
        out = np.empty(n)
        for s in range(0, n, self.max_batch):
            chunk = X[s:s + self.max_batch]
            sys.stdout.write(json.dumps({"type": "query",
                                         "points": chunk.tolist()}) + "\n")
            sys.stdout.flush()
            self.used += len(chunk)
            while True:
                line = sys.stdin.readline()
                if not line:
                    raise RuntimeError("oracle closed")
                msg = json.loads(line)
                if msg.get("type") == "values":
                    break
            v = np.asarray(msg.get("values", []), dtype=np.float64)
            if v.size != len(chunk):
                raise RuntimeError("bad value count")
            out[s:s + len(chunk)] = v
        self.vals.append(out)
        if not np.all(np.isfinite(out)):
            raise RuntimeError("non-finite values")
        return out

    def naive_mean(self):
        if not self.vals:
            return None
        v = np.concatenate(self.vals)
        v = v[np.isfinite(v)]
        return float(v.mean()) if v.size else None


def main():
    line = sys.stdin.readline()
    if not line:
        return
    init = json.loads(line)
    d = int(init["dimension"])
    budget = int(init["budget"])
    seed = int(init.get("seed", 0) or 0)
    o = Oracle(d, budget)
    est = float("nan")
    try:
        est = integrate(o, d, budget, seed)
    except Exception:
        est = float("nan")
    if est is None or not np.isfinite(est):
        est = o.naive_mean()
    if est is None or not np.isfinite(est):
        try:
            if o.remaining >= 8 and SEQ is not None:
                n = min(o.remaining, 2 ** int(math.floor(math.log2(max(1, o.remaining)))))
                est = float(ev(o, SEQ[P0N:P0N + n]).mean())
        except Exception:
            pass
    if est is None or not np.isfinite(est):
        est = 0.0
    sys.stdout.write(json.dumps({"type": "result", "estimate": float(est)}) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
