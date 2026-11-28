#!/usr/bin/env python3
"""Structure-adaptive cubature for an unknown black-box integrand on [0,1]^d.

Every function evaluation we request (pilot, probes, selectors, refinement) is
charged against one global point budget.

Method
------
1.  A scrambled Sobol pilot gives a baseline QMC estimate, a value scale and
    the most active sample points.
2.  Cheap exact identities probe the global shape of the integrand:
      * rank-1 gradient covariance  ->  f(x) = h(v.x)          (ridge)
      * multiplicative swap identity ->  f(x) = prod g_j(x_j)  (separable)
      * additive swap identity       ->  f(x) = sum  g_j(x_j)  (additive)
    A hypothesis is trusted only when the identity holds to floating point
    noise on points that were not used to build it.
3.  Trusted structure is integrated with a high order adaptive 1-D Gauss rule
    along the structure (ridges use the exact distribution of the projection
    of the cube onto v), which is far more accurate per point than any generic
    d-dimensional rule.
4.  Otherwise we fall back to scrambled Sobol QMC over every point not spent.
"""

import json
import math
import sys
import time

import numpy as np

T0 = time.time()
SOFT_TIME = 15.0          # stop asking for new work after this many seconds
BATCHCAP = 512
DEBUG = bool(__import__('os').environ.get('CUBA_DEBUG'))


def _dbg(*a):
    if DEBUG:
        try:
            sys.stderr.write(' '.join(str(x) for x in a) + "\n")
        except Exception:
            pass


class BudgetExhausted(Exception):
    pass


class TimeUp(Exception):
    pass


# ---------------------------------------------------------------- protocol --
def _send(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def _recv():
    while True:
        line = sys.stdin.readline()
        if not line:
            raise EOFError("stdin closed")
        s = line.strip()
        if s:
            return json.loads(s)


class Oracle(object):
    """The only interface to the hidden integrand, with strict accounting."""

    def __init__(self, d, budget):
        self.d = int(d)
        self.budget = int(budget)
        self.used = 0
        self.u_sum = 0.0
        self.u_n = 0
        self.fmax = 0.0

    def remaining(self):
        return self.budget - self.used

    def eval(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        n = int(X.shape[0])
        if n == 0:
            return np.zeros(0)
        if X.shape[1] != self.d:
            raise ValueError("bad point shape")
        if n > self.remaining():
            raise BudgetExhausted()
        if time.time() - T0 > SOFT_TIME:
            raise TimeUp()
        X = np.nan_to_num(X, nan=0.5, posinf=1.0, neginf=0.0)
        np.clip(X, 0.0, 1.0, out=X)
        rows = [[float(x) for x in row] for row in X]
        vals = []
        for s in range(0, n, BATCHCAP):
            _send({"type": "query", "points": rows[s:s + BATCHCAP]})
            msg = _recv()
            if msg.get("type") != "values":
                raise ValueError("unexpected message type")
            vals.extend([float(v) for v in msg.get("values", [])])
        v = np.asarray(vals, dtype=float)
        if v.shape[0] != n:
            raise ValueError("value count mismatch")
        if not np.all(np.isfinite(v)):
            v = np.where(np.isfinite(v), v, 0.0)
        self.used += n
        if n:
            self.fmax = max(self.fmax, float(np.max(np.abs(v))))
        return v

    def add_uniform(self, v):
        v = np.asarray(v, dtype=float)
        self.u_sum += float(np.sum(v))
        self.u_n += int(v.shape[0])

    def uniform_mean(self):
        if self.u_n:
            return self.u_sum / self.u_n
        return None


# ------------------------------------------------------------ point sets ----
def make_sobol(d, n, seed):
    n = max(1, int(n))
    try:
        from scipy.stats import qmc
        m = max(1, int(math.ceil(math.log2(max(2, n)))))
        eng = qmc.Sobol(d=d, scramble=True, seed=seed)
        X = eng.random_base2(m)
        X = np.asarray(X, dtype=float)
    except Exception:
        X = None
    if X is None or X.shape[0] < n:
        rng = np.random.default_rng(seed)
        X = rng.random((n, d))
    return np.clip(X[:n], 0.0, 1.0)


def max_step(x, p):
    """Largest s >= 0 with both x + s*p and x - s*p inside [0,1]^d."""
    x = np.asarray(x, dtype=float)
    p = np.asarray(p, dtype=float)
    q = np.abs(p)
    with np.errstate(divide="ignore", invalid="ignore"):
        up = np.where(p > 0, (1.0 - x) / q, np.where(p < 0, x / q, np.inf))
        dn = np.where(p > 0, x / q, np.where(p < 0, (1.0 - x) / q, np.inf))
    lim = min(float(np.min(up)), float(np.min(dn)))
    if not np.isfinite(lim):
        return 1.0
    return max(0.0, lim)


# ------------------------------------------------- adaptive 1-D quadrature --
def adaptive_1d(feval, a, b, evals, npg=8, breakpoints=()):
    """Adaptive composite Gauss-Legendre quadrature over [a,b].

    feval takes a numpy array of nodes inside [a,b] (batched) and returns the
    values.  Exceptions raised inside feval (budget / time) are caught and the
    integral of the current partition is returned.

    Each interval is compared against its two children (Gauss on the halves),
    which gives a genuine discrepancy estimate, so the refinement budget
    concentrates on kinks, jumps and spikes instead of being spread evenly.
    """
    a = float(a)
    b = float(b)
    evals = int(evals)
    if not (b > a):
        return 0.0, 0
    gx, gw = np.polynomial.legendre.leggauss(npg)

    def nodes_of(lo, hi):
        return 0.5 * (hi - lo) * gx + 0.5 * (hi + lo)

    def integ_of(lo, hi, sv):
        return float(np.dot(sv, 0.5 * (hi - lo) * gw))

    if evals < 3 * npg:
        if evals >= npg:
            try:
                return integ_of(a, b, feval(nodes_of(a, b))), npg
            except (BudgetExhausted, TimeUp, ValueError):
                return 0.0, 0
        return 0.0, 0

    bps = sorted(float(p) for p in breakpoints if a < p < b)
    edges = [a] + bps + [b]
    m0 = int(max(2, min(10, evals // (4 * npg))))
    if len(edges) - 1 > m0:
        idx = np.unique(np.round(np.linspace(0, len(edges) - 1, m0 + 1)).astype(int))
        edges = [edges[int(i)] for i in idx]
    keep = [edges[0]]
    for e in edges[1:]:
        if e - keep[-1] > 1e-12 * (b - a):
            keep.append(e)
    if len(keep) < 2:
        keep = [a, b]
    segs = [[keep[i], keep[i + 1]] for i in range(len(keep) - 1)]

    state = {'used': 0}

    try:
        fv = feval(np.concatenate([nodes_of(s[0], s[1]) for s in segs]))
    except (BudgetExhausted, TimeUp, ValueError):
        return 0.0, 0
    state['used'] += npg * len(segs)
    vals = [integ_of(s[0], s[1], fv[k * npg:(k + 1) * npg]) for k, s in enumerate(segs)]
    store = [fv[k * npg:(k + 1) * npg] for k in range(len(segs))]
    # provisional ranking error: local variation times width.  Real interval
    # discrepancies replace it as soon as an interval has been bisected once.
    errs = [(s[1] - s[0]) * float(np.max(np.abs(sv)) - np.min(np.abs(sv)))
            for s, sv in zip(segs, store)]
    total = float(np.sum(vals))

    def split_point(lo, hi, sv):
        """Midpoint, or the middle of a dominant jump in the node values."""
        v = np.asarray(sv, dtype=float)
        if v.size < 2:
            return 0.5 * (lo + hi)
        dv = np.abs(np.diff(v))
        rngv = float(np.max(v) - np.min(v))
        if rngv > 0.0:
            k = int(np.argmax(dv))
            if float(dv[k]) > 0.6 * rngv:
                t = nodes_of(lo, hi)
                return float(0.5 * (t[k] + t[k + 1]))
        return 0.5 * (lo + hi)

    def bisect(ids):
        nonlocal segs, vals, store, errs, total
        jobs = []
        allnodes = []
        for i in ids:
            lo, hi = segs[i]
            mid = split_point(lo, hi, store[i])
            mid = min(max(mid, lo + 1e-13 * (b - a)), hi - 1e-13 * (b - a))
            for (l, h) in ((lo, mid), (mid, hi)):
                jobs.append((i, l, h))
                allnodes.append(nodes_of(l, h))
        try:
            f2 = feval(np.concatenate(allnodes))
        except (BudgetExhausted, TimeUp, ValueError):
            return False
        state['used'] += 2 * npg * len(ids)
        child = {}
        for j, (i, l, h) in enumerate(jobs):
            sv = f2[j * npg:(j + 1) * npg]
            child.setdefault(i, []).append([l, h, integ_of(l, h, sv), sv])
        nsegs, nvals, nstore, nerrs = [], [], [], []
        for i, (lo, hi) in enumerate(segs):
            if i in child:
                ch = child[i]
                disc = 0.5 * abs(vals[i] - sum(c[2] for c in ch))
                for (l, h, v, sv) in ch:
                    nsegs.append([l, h])
                    nvals.append(v)
                    nstore.append(sv)
                    nerrs.append(disc)
            else:
                nsegs.append([lo, hi])
                nvals.append(vals[i])
                nstore.append(store[i])
                nerrs.append(errs[i])
        segs, vals, store, errs = nsegs, nvals, nstore, nerrs
        total = float(np.sum(vals))
        return True

    # provisional pass: bisect every interval once so that the cheap ranking
    # error is replaced by a genuine discrepancy wherever it matters
    todo = sorted(range(len(segs)), key=lambda i: -errs[i])
    limit = max(1, min(len(todo), (evals - state['used']) // (6 * npg)))
    if not bisect(todo[:limit]):
        return total, state['used']

    while True:
        left = evals - state['used']
        if left < 2 * npg:
            break
        if float(np.sum(errs)) <= 1e-14 * max(abs(total), 1e-300):
            break
        nref = int(max(1, min(4, left // (2 * npg))))
        order = sorted(range(len(segs)), key=lambda i: -errs[i])[:nref]
        if not bisect(order):
            break
    return total, state['used']


# ------------------------------------------------------- structure probes ---
def pick_bases(X1, V1, K, d, rng):
    order = np.argsort(-np.abs(V1))
    bases = []
    for i in order:
        x = np.asarray(X1[i], dtype=float)
        ok = True
        for b in bases:
            if np.max(np.abs(x - b)) < 0.12:
                ok = False
                break
        if ok:
            bases.append(x.copy())
        if len(bases) >= K:
            break
    while len(bases) < K:
        bases.append(np.asarray(rng.uniform(0.15, 0.85, size=d), dtype=float))
    return bases


def central_gradients(oracle, bases, d, delta=0.02):
    nb = len(bases)
    pts = []
    for b in bases:
        b2 = np.clip(np.asarray(b, dtype=float), delta, 1.0 - delta)
        for i in range(d):
            p = b2.copy()
            p[i] += delta
            pts.append(p)
            p = b2.copy()
            p[i] -= delta
            pts.append(p)
    fv = oracle.eval(np.asarray(pts, dtype=float))
    G = fv.reshape(nb, d, 2)
    return (G[:, :, 0] - G[:, :, 1]) / (2.0 * delta)


def richardson_gradient(oracle, x0, d, K=6, delta0=0.015):
    """High order extrapolated gradient (central differences + Richardson)."""
    x0 = np.clip(np.asarray(x0, dtype=float), delta0, 1.0 - delta0)
    steps = [delta0 / (2.0 ** k) for k in range(K)]
    pts = []
    for dk in steps:
        for i in range(d):
            p = x0.copy()
            p[i] += dk
            pts.append(p)
            p = x0.copy()
            p[i] -= dk
            pts.append(p)
    fv = oracle.eval(np.asarray(pts, dtype=float))
    A = fv.reshape(K, d, 2)
    D = (A[:, :, 0] - A[:, :, 1]) / (2.0 * np.asarray(steps)[:, None])
    T = D.copy()
    for j in range(1, K):
        fac = 4.0 ** j - 1.0
        T[:K - j, :] = T[1:K - j + 1, :] + (T[1:K - j + 1, :] - T[:K - j, :]) / fac
    return T[0, :]


def ridge_test(oracle, v, X1, V1, scale, rng, d, tol=1e-8):
    """Is f constant on hyperplanes orthogonal to v?  (exact identity test)"""
    tp = []
    order = np.argsort(-np.abs(V1))
    for i in order[:6]:
        tp.append(np.clip(np.asarray(X1[i], dtype=float), 0.12, 0.88))
    for _ in range(4):
        tp.append(np.asarray(rng.uniform(0.2, 0.8, size=d), dtype=float))
    num = 0.0
    den = 0.0
    ntest = 0
    for x in tp:
        w = rng.standard_normal(d)
        w = w - float(np.dot(w, v)) * v
        nw = float(np.linalg.norm(w))
        if nw < 1e-8:
            continue
        w = w / nw
        dd = min(0.12, 0.8 * max_step(x, w), 0.8 * max_step(x, v))
        if dd < 0.02:
            continue
        pts = np.asarray([x, x + dd * w, x - dd * w, x + dd * v], dtype=float)
        try:
            fv = oracle.eval(pts)
        except (BudgetExhausted, TimeUp):
            break
        ntest += 1
        num = max(num, abs(float(fv[1]) - float(fv[0])), abs(float(fv[2]) - float(fv[0])))
        den = max(den, abs(float(fv[3]) - float(fv[0])))
    if ntest == 0:
        return False
    ratio = num / max(den, 1e-300)
    _dbg('ridge_test ratio', ratio, 'num', num, 'den', den, 'ntest', ntest)
    return bool(ratio <= tol)


def box_spline_weights(u):
    ss = np.zeros(1)
    sg = np.ones(1)
    for ui in u:
        ss = np.concatenate([ss, ss + ui])
        sg = np.concatenate([sg, -sg])
    return ss, sg


def ridge_integral(oracle, v, d, budget):
    """int_[0,1]^d h(v.x) dx  =  int h(s) w(s) ds, w the box-spline density."""
    v = np.asarray(v, dtype=float)
    nv = float(np.linalg.norm(v))
    if not np.isfinite(nv) or nv <= 0:
        return None
    v = v / nv
    vm = np.abs(v)
    vmax = float(vm.max())
    act = vm > 1e-7 * vmax
    if not np.any(act):
        return None
    u = np.where(act, np.maximum(vm, 1e-7 * vmax), 0.0)
    idx = np.where(act)[0]
    u = u[idx]
    m = int(len(u))
    V = float(np.sum(u))
    if V <= 0:
        return None
    a = np.where(v > 0, 0.0, 1.0)
    b = np.where(v > 0, 1.0, 0.0)
    a = np.where(act, a, 0.5)
    b = np.where(act, b, 0.5)
    ss, sg = box_spline_weights(u)
    fact = 1.0 / math.factorial(m - 1)
    pu = float(np.prod(u))
    if pu <= 0 or not np.isfinite(pu):
        return None

    def wfun(s):
        s = np.asarray(s, dtype=float)
        if m == 1:
            return np.where((s >= 0.0) & (s <= u[0]), 1.0 / pu, 0.0)
        t = s[:, None] - ss[None, :]
        np.maximum(t, 0.0, out=t)
        t = t ** (m - 1)
        return (t * sg[None, :]).sum(axis=1) * fact / pu

    def feval(lam):
        lam = np.asarray(lam, dtype=float)
        X = a[None, :] + lam[:, None] * (b - a)[None, :]
        fv = oracle.eval(X)
        return fv * wfun(lam * V) * V

    bps = ()
    if (1 << m) <= 64:
        bps = [float(p) / V for p in ss if 0.0 < p / V < 1.0]
    try:
        val, _ = adaptive_1d(feval, 0.0, 1.0, int(budget), npg=8, breakpoints=bps)
    except (BudgetExhausted, TimeUp, ValueError):
        return None
    if not np.isfinite(val):
        return None
    return val


def cross_test(oracle, X1, V1, scale, rng, d):
    """Exact swap identities for separable / additive structure."""
    sc = max(scale, 1e-300)
    order = np.argsort(-np.abs(V1))
    cand = [(i, np.asarray(X1[i], dtype=float)) for i in order if abs(float(V1[i])) > 0.02 * sc]
    cand = cand[:24]
    if len(cand) < 2:
        cand = [(i, np.asarray(X1[i], dtype=float)) for i in order[:6]]
    pairs = []
    for i in range(len(cand)):
        if len(pairs) >= 6:
            break
        for j in range(i + 1, len(cand)):
            if len(pairs) >= 6:
                break
            if float(np.max(np.abs(cand[i][1] - cand[j][1]))) > 0.12:
                pairs.append((cand[i], cand[j]))
    if len(pairs) < 2 and len(cand) >= 2:
        pairs.append((cand[0], cand[1]))
    if not pairs:
        return 1.0, 1.0, 0
    sep_max = 0.0
    add_max = 0.0
    ntest = 0
    for ((ia, xa), (ib, xb)) in pairs:
        for jc in range(d):
            try:
                pa = xa.copy()
                pa[jc] = xb[jc]
                pb = xb.copy()
                pb[jc] = xa[jc]
                fv = oracle.eval(np.asarray([pa, pb], dtype=float))
            except (BudgetExhausted, TimeUp):
                return sep_max, add_max, ntest
            fa = float(V1[ia])
            fb = float(V1[ib])
            fpa = float(fv[0])
            fpb = float(fv[1])
            ntest += 1
            dsep = max(abs(fa * fb), abs(fpa * fpb), 1e-8 * sc * sc)
            sep_max = max(sep_max, abs(fa * fb - fpa * fpb) / dsep)
            dadd = max(abs(fa) + abs(fb), abs(fpa) + abs(fpb), 1e-8 * sc)
            add_max = max(add_max, abs(fa + fb - fpa - fpb) / dadd)
    return sep_max, add_max, ntest


def coordinate_ascent(oracle, x0, f0, d, m=12, sweeps=2):
    x = np.asarray(x0, dtype=float).copy()
    fx = float(f0)
    for _ in range(sweeps):
        improved = False
        for j in range(d):
            ts = np.linspace(0.0, 1.0, m)
            X = np.tile(x, (m, 1))
            X[:, j] = ts
            try:
                fv = oracle.eval(X)
            except (BudgetExhausted, TimeUp):
                return x, fx
            k = int(np.argmax(np.abs(fv)))
            if abs(float(fv[k])) > abs(fx):
                improved = True
                x = X[k].copy()
                fx = float(fv[k])
        if not improved:
            break
    return x, fx


def slice_estimates(oracle, xstar, d, budget, min_per=40):
    """1-D adaptive integrals of the coordinate slices through xstar."""
    per = int(max(min_per, budget // d))
    out = []
    carry = 0
    for j in range(d):
        left_dims = d - 1 - j
        avail = min(per + carry, oracle.remaining() - left_dims * 24)
        if avail < 24:
            return None
        base = np.asarray(xstar, dtype=float).copy()

        def feval(ts, j=j, base=base):
            X = np.tile(base, (len(ts), 1))
            X[:, j] = np.asarray(ts, dtype=float)
            return oracle.eval(X)

        val, used = adaptive_1d(feval, 0.0, 1.0, avail, npg=8)
        if not np.isfinite(val):
            return None
        carry = max(0, carry + per - used)
        out.append(val)
    return out


# --------------------------------------------------------------- main flow --
def solve(d, budget, seed, oracle):
    N1 = 256 if budget >= 512 else max(16, budget // 8)
    Xsob = make_sobol(d, budget, seed)
    if Xsob.shape[0] < N1:
        N1 = int(Xsob.shape[0])
    X1 = np.asarray(Xsob[:N1], dtype=float).copy()
    V1 = oracle.eval(X1)
    oracle.add_uniform(V1)
    if N1 == 0:
        return 0.0
    if d == 1:
        def feval1(t):
            return oracle.eval(np.asarray(t, dtype=float).reshape(-1, 1))
        val, _ = adaptive_1d(feval1, 0.0, 1.0, oracle.remaining(), npg=8)
        if np.isfinite(val) and oracle.used > 0:
            return float(val)
        return float(np.mean(V1))
    scale = float(np.max(np.abs(V1)))
    if scale <= 0.0:
        scale = 1e-300
    qmc = float(np.mean(V1))
    rng = np.random.default_rng((int(seed) % 1000000) + 12345)

    # ---- ridge hypothesis -------------------------------------------------
    ridge = None
    try:
        nb = 4
        if oracle.remaining() < 2 * d * nb + 240:
            nb = max(1, (oracle.remaining() - 240) // (2 * d))
        if nb >= 1:
            bases = pick_bases(X1, V1, nb, d, rng)
            grads = central_gradients(oracle, bases, d)
            gn = np.linalg.norm(grads, axis=1)
            gmax = float(np.max(gn)) if len(gn) else 0.0
            if gmax > 1e-12 * scale:
                keep = gn > gmax * 1e-6
                Mm = grads[keep].T @ grads[keep]
                ev, evec = np.linalg.eigh(Mm)
                lam1 = float(ev[-1])
                _dbg('gradPCA', ev[-1], ev[-2] if d > 1 else 0.0, 'gmax', gmax)
                if lam1 > 0 and (d < 2 or float(ev[-2]) / lam1 < 1e-6):
                    order = np.argsort(-gn)
                    refine = []
                    for i in order:
                        if i not in refine:
                            refine.append(i)
                        if len(refine) >= 2:
                            break
                    if oracle.remaining() > 2 * 6 * d * len(refine) + 200:
                        rg = [richardson_gradient(oracle, bases[i], d) for i in refine]
                        rn = [float(np.linalg.norm(g)) for g in rg]
                        if max(rn) > 0:
                            Mr = np.zeros((d, d))
                            for g, nrm in zip(rg, rn):
                                if nrm > max(rn) * 1e-6:
                                    Mr += np.outer(g, g)
                            ev2, ev2v = np.linalg.eigh(Mr)
                            l1 = float(ev2[-1])
                            _dbg('richPCA', ev2[-1], ev2[-2] if d > 1 else 0.0)
                            if l1 > 0 and (d < 2 or float(ev2[-2]) / l1 < 1e-8):
                                v = np.asarray(ev2v[:, -1], dtype=float)
                                v = v / max(float(np.linalg.norm(v)), 1e-300)
                                if ridge_test(oracle, v, X1, V1, scale, rng, d):
                                    res = oracle.remaining() - 120
                                    if res > 96:
                                        ridge = ridge_integral(oracle, v, d, res)
                                        _dbg('RIDGE', ridge)
    except (BudgetExhausted, TimeUp, ValueError):
        ridge = None
    if ridge is not None and np.isfinite(ridge):
        return float(ridge)

    # ---- separable / additive hypothesis ----------------------------------
    try:
        sep_max, add_max, ntest = cross_test(oracle, X1, V1, scale, rng, d)
    except (BudgetExhausted, TimeUp, ValueError):
        sep_max, add_max, ntest = 1.0, 1.0, 0
    _dbg('cross', sep_max, add_max, ntest)
    sep_ok = ntest >= 2 and sep_max < 1e-9
    add_ok = ntest >= 2 and add_max < 1e-9
    if sep_ok or add_ok:
        try:
            ib = int(np.argmax(np.abs(V1)))
            xstar, fstar = coordinate_ascent(oracle, X1[ib], float(V1[ib]), d)
            if abs(fstar) > 0.0:
                res = oracle.remaining() - 96
                if res >= 24 * d:
                    A = slice_estimates(oracle, xstar, d, res)
                    if A is not None:
                        if add_ok and (not sep_ok or add_max <= sep_max):
                            est = float(np.sum(A)) - (d - 1) * float(fstar)
                        else:
                            prod = 1.0
                            ok = True
                            for A_j in A:
                                prod *= A_j / fstar
                                if not np.isfinite(prod) or abs(prod) > 1e250:
                                    ok = False
                                    break
                            est = float(fstar) * prod if ok else None
                        _dbg('SEP/ADD est', est, 'fstar', fstar)
                        if est is not None and np.isfinite(est) and abs(est) < 1e250:
                            return float(est)
        except (BudgetExhausted, TimeUp, ValueError, ZeroDivisionError, OverflowError):
            pass

    # ---- generic fallback: scrambled Sobol QMC over remaining points -------
    try:
        rest = oracle.remaining()
        avail = int(Xsob.shape[0]) - N1
        n = int(min(rest, avail))
        if n > 0:
            X2 = np.asarray(Xsob[N1:N1 + n], dtype=float)
            V2 = oracle.eval(X2)
            oracle.add_uniform(V2)
    except (BudgetExhausted, TimeUp, ValueError):
        pass
    m = oracle.uniform_mean()
    if m is None or not np.isfinite(m):
        return qmc
    return float(m)


def main():
    line = sys.stdin.readline()
    if not line:
        return
    init = json.loads(line)
    d = int(init.get("dimension", 1))
    budget = int(init.get("budget", 2048))
    seed = int(init.get("seed", 0))
    oracle = Oracle(d, budget)
    est = None
    try:
        est = solve(d, budget, seed, oracle)
    except Exception:
        est = None
    if est is None or not np.isfinite(est):
        est = oracle.uniform_mean()
    if est is None or not np.isfinite(est):
        est = 0.0
    _send({"type": "result", "estimate": float(est)})


if __name__ == "__main__":
    main()
