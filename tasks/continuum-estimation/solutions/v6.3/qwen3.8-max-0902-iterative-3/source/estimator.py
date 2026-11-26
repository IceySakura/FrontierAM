"""Continuum-limit estimator for y(x) = t + a*x^p + b*x^q + c*x^(q+OFF).

The observations are exact (deterministic simulator), so with n = 4 or 5 levels
the parameter vector theta = (t, a, b, c, p, q) is confined to a (6 - n)
dimensional manifold: every admissible theta reproduces the data exactly.  The
posterior of theta given the data is therefore the prior restricted to that
manifold, with the co-area density

    pi(p, q) \\propto prior(theta(p, q)) / sqrt(det(M^T M)),

where M = [1, x^p, x^q, x^(q+OFF)] is the n x 4 design matrix (for n = 4,
sqrt(det(M^T M)) = |det M|).  For n = 5 the manifold is the curve
F(p, q) = det([M | y]) = 0, along which the density reduces to
prior / |dF/dp|; it is located by scanning for sign changes of F and bisecting.

The reported estimate is a posterior quantile of t (see QUANTILE in solve.py)
taken under the density re-weighted by 1 / max(1, |t|), because the scored loss
is |z - t| / max(1, |t|).

Only numpy is used, so the interpreter start-up stays small.
"""
import numpy as np

OFF = 1.5           # third exponent offset: r = q + OFF
GRID4 = 520         # per-axis grid for the n = 4 manifold
GRID5 = 880         # per-axis scan for the n = 5 curve
COARSE = 0.55       # grid scale-down factor used when short on time

# Scan windows for (p, q); the prior does the real restricting.
P_LO, P_HI = 0.02, 6.0
Q_LO, Q_HI = 0.12, 8.6


# --------------------------------------------------------------------------
# prior
# --------------------------------------------------------------------------
def _expit(z):
    z = np.asarray(z, float)
    out = np.empty(z.shape)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def softbox(v, lo, hi, w):
    """Smooth indicator: ~1 on [lo, hi], logistic taper of width w outside."""
    if hi <= lo:
        return np.ones(np.shape(v))
    if w <= 0:
        return np.where((v >= lo) & (v <= hi), 1.0, 0.0)
    return _expit((v - lo) / w) * _expit((hi - v) / w)


# Nominal prior: bounds inferred from the public development examples, with
# logistic tapers wide enough that a truth slightly outside a box is only
# down-weighted, never excluded.
DEFAULT_SPEC = dict(
    t=(-3.05, 3.05), wt=0.15,
    a=(-1.53, 1.53), wa=0.075,
    b=(-0.92, -0.085), wb=0.020,
    c=(-0.41, -0.040), wc=0.0088,
    p=(0.47, 3.05), wp=0.063,
    q=(0.95, 4.60), wq=0.088,
    gap=(0.38, 99.0), wgap=0.030,
)


class Prior:
    """Product of soft boxes over (t, a, b, c, p, q) and the gap q - p."""

    def __init__(self, spec=None):
        self.s = dict(DEFAULT_SPEC)
        if spec:
            self.s.update(spec)

    def w(self, t, a, b, c, p, q):
        s = self.s
        return (softbox(t, *s['t'], s['wt']) *
                softbox(a, *s['a'], s['wa']) *
                softbox(b, *s['b'], s['wb']) *
                softbox(c, *s['c'], s['wc']) *
                softbox(p, *s['p'], s['wp']) *
                softbox(q, *s['q'], s['wq']) *
                softbox(q - p, *s['gap'], s['wgap']))

    def widen(self, k):
        """Stretch every box about its centre by k and soften the tapers."""
        s = {}
        for key in ('t', 'a', 'b', 'c', 'p', 'q'):
            lo, hi = self.s[key]
            mid = 0.5 * (lo + hi)
            half = 0.5 * (hi - lo) * k
            s[key] = (mid - half, mid + half)
        lo, hi = self.s['gap']
        s['gap'] = (lo / k, hi * k)
        for key in ('wt', 'wa', 'wb', 'wc', 'wp', 'wq', 'wgap'):
            s[key] = self.s[key] * k
        return Prior(s)


# --------------------------------------------------------------------------
# model algebra
# --------------------------------------------------------------------------
def design(x, p, q):
    """Stacked design matrices [1, x^p, x^q, x^(q+OFF)], shape (K, n, 4)."""
    x = np.asarray(x, float)
    p = np.asarray(p, float).reshape(-1, 1)
    q = np.asarray(q, float).reshape(-1, 1)
    M = np.empty((p.shape[0], x.size, 4))
    M[:, :, 0] = 1.0
    M[:, :, 1] = x[None, :] ** p
    M[:, :, 2] = x[None, :] ** q
    M[:, :, 3] = x[None, :] ** (q + OFF)
    return M


def det_augmented(x, y, p, q):
    """det([M | y]) for each (p, q) pair; its zero set is the n = 5 manifold."""
    M = design(x, p, q)
    yb = np.broadcast_to(np.asarray(y, float)[None, :, None],
                         (M.shape[0], M.shape[1], 1))
    return np.linalg.det(np.concatenate([M, yb], axis=2))


def _solve_best_subset(M, y):
    """Solve M theta = y using the best-conditioned 4-row subset of each stack.

    Returns (theta, |det| of the chosen subset, max residual over all rows).
    """
    K, n = M.shape[0], M.shape[1]
    yb = np.broadcast_to(np.asarray(y, float), (K, n))
    best_th = np.zeros((K, 4))
    best_det = np.full(K, -1.0)
    best_res = np.full(K, np.inf)
    for drop in range(n):
        ii = [i for i in range(n) if i != drop]
        Ms = M[:, ii, :]
        try:
            d = np.abs(np.linalg.det(Ms))
        except np.linalg.LinAlgError:
            continue
        ok = np.isfinite(d) & (d > best_det) & (d > 0.0)
        if not ok.any():
            continue
        try:
            th = np.asarray(np.linalg.solve(Ms[ok], yb[ok][:, ii][:, :, None])[:, :, 0])
        except np.linalg.LinAlgError:
            continue
        th = np.where(np.isfinite(th), th, 0.0)
        res = np.abs(np.einsum('kij,kj->ki', M[ok], th) - yb[ok]).max(axis=1)
        best_th[ok] = th
        best_res[ok] = res
        best_det[ok] = d[ok]
    return best_th, np.maximum(best_det, 0.0), best_res


def manifold_n4(x, y, scale=1.0):
    """Whole 2-D manifold for n = 4 -> (p, q, t, a, b, c, |det M|)."""
    ngrid = max(80, int(round(GRID4 * scale)))
    ps = np.linspace(P_LO, P_HI, ngrid)
    qs = np.linspace(Q_LO, Q_HI, ngrid)
    P, Q = np.meshgrid(ps, qs, indexing='ij')
    P = P.ravel(); Q = Q.ravel()
    keep = (Q - P) > 0.02
    P = P[keep]; Q = Q[keep]
    yb = np.asarray(y, float)
    out = [[] for _ in range(7)]
    chunk = 60000
    for s in range(0, len(P), chunk):
        p = P[s:s + chunk]; q = Q[s:s + chunk]
        try:
            M = design(x, p, q)
            detM = np.linalg.det(M)
            ok = np.isfinite(detM) & (np.abs(detM) > 1e-15)
            if not ok.any():
                continue
            Mk = M[ok]
            th = np.linalg.solve(Mk, np.broadcast_to(yb, (Mk.shape[0], yb.size))[:, :, None])[:, :, 0]
            good = np.isfinite(th).all(axis=1)
            if not good.any():
                continue
            th = th[good]; pk = p[ok][good]; qk = q[ok][good]; dk = np.abs(detM[ok][good])
        except np.linalg.LinAlgError:
            continue
        out[0].append(pk); out[1].append(qk)
        for i in range(4):
            out[2 + i].append(th[:, i])
        out[6].append(dk)
    if not out[0]:
        return None
    return tuple(np.concatenate(o) for o in out)


def manifold_n5(x, y, scale=1.0):
    """Trace det([M|y]) = 0 for n = 5 -> (p, q, t, a, b, c, |dF/dp|)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    nq = max(120, int(round(GRID5 * scale)))
    nps = max(120, int(round(GRID5 * scale)))
    qg = np.linspace(Q_LO, Q_HI, nq)
    pg = np.linspace(P_LO, P_HI, nps)
    P2, Q2 = np.meshgrid(pg, qg)
    flat_P = P2.ravel(); flat_Q = Q2.ravel()
    valid = np.flatnonzero((flat_P < flat_Q - 0.02) & (flat_P > 0))
    F = np.full(flat_P.shape, np.nan)
    chunk = 150000
    for s in range(0, len(valid), chunk):
        ii = valid[s:s + chunk]
        F[ii] = det_augmented(x, y, flat_P[ii], flat_Q[ii])
    F = F.reshape(P2.shape)
    sg = np.sign(F)
    ri, ci = np.nonzero((sg[:, :-1] * sg[:, 1:]) < 0)
    if len(ri) == 0:
        return None
    lo = pg[ci]; hi = pg[ci + 1]; qc = qg[ri]
    flo = det_augmented(x, y, lo, qc)
    for _ in range(46):                      # vectorised bisection
        mid = 0.5 * (lo + hi)
        fm = det_augmented(x, y, mid, qc)
        same = (fm * flo) > 0
        lo = np.where(same, mid, lo)
        hi = np.where(same, hi, mid)
        flo = np.where(same, fm, flo)
    P = 0.5 * (lo + hi); Q = qc
    M = design(x, P, Q)
    th, _det, res = _solve_best_subset(M, y)
    ok = res < 1e-9 * max(1.0, float(np.abs(y).max()))
    P, Q, th = P[ok], Q[ok], th[ok]
    if len(P) == 0:
        return None
    h = 1e-5
    dF = np.abs(det_augmented(x, y, P + h, Q) -
                det_augmented(x, y, P - h, Q)) / (2 * h)
    good = np.isfinite(dF) & (dF > 0)
    P, Q, th, dF = P[good], Q[good], th[good], dF[good]
    if len(P) == 0:
        return None
    return P, Q, th[:, 0], th[:, 1], th[:, 2], th[:, 3], dF


def manifold(x, y, scale=1.0):
    n = len(x)
    if n == 4:
        return manifold_n4(x, y, scale)
    if n == 5:
        return manifold_n5(x, y, scale)
    return None


# --------------------------------------------------------------------------
# posterior over t
# --------------------------------------------------------------------------
def wquantile(v, w, frac=0.5):
    o = np.argsort(v)
    v = v[o]; cw = np.cumsum(w[o])
    tot = cw[-1]
    if not np.isfinite(tot) or tot <= 0:
        return None
    return float(v[min(int(np.searchsorted(cw, frac * tot)), len(v) - 1)])


def posterior_t(man, prior, loss_weight=True):
    """(t values, weights) restricted to the prior-consistent manifold."""
    if man is None:
        return None, None
    p, q, t, a, b, c, jac = man
    w = prior.w(t, a, b, c, p, q) / np.maximum(jac, 1e-300)
    if loss_weight:
        w = w / np.maximum(1.0, np.abs(t))
    m = np.isfinite(w) & np.isfinite(t) & (w > 0)
    if not m.any():
        return None, None
    return t[m], w[m]


def grid_fit(x, y, prior, sigma=None, scale=1.0):
    """Generic fallback for n not in {4, 5}: soft least squares over a (p, q)
    grid with the same prior and co-area weighting."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    n = x.size
    ngrid = max(60, int(round(300 * scale)))
    ps = np.linspace(P_LO, P_HI, ngrid)
    qs = np.linspace(Q_LO, Q_HI, ngrid)
    P, Q = np.meshgrid(ps, qs, indexing='ij')
    P = P.ravel(); Q = Q.ravel()
    keep = (Q - P) > 0.02
    P = P[keep]; Q = Q[keep]
    sc = float(np.max(np.abs(y))) or 1.0
    if sigma is None:
        sigma = 1e-7 * sc
    T = []; W = []
    chunk = 60000
    for s in range(0, len(P), chunk):
        p = P[s:s + chunk]; q = Q[s:s + chunk]
        M = design(x, p, q)
        G = np.einsum('kni,knj->kij', M, M)
        h = (M * y[None, :, None]).sum(axis=1)
        try:
            th = np.linalg.solve(G + 1e-14 * np.eye(4)[None], h[:, :, None])[:, :, 0]
        except np.linalg.LinAlgError:
            continue
        r = np.einsum('kni,ki->kn', M, th) - y[None, :]
        rss = np.sum(r * r, axis=1)
        detG = np.linalg.det(G)
        ok = np.isfinite(rss) & (detG > 0)
        if not ok.any():
            continue
        w = (prior.w(th[ok, 0], th[ok, 1], th[ok, 2], th[ok, 3], p[ok], q[ok])
             * np.exp(-0.5 * rss[ok] / (sigma ** 2)) / np.sqrt(detG[ok]))
        T.append(th[ok, 0]); W.append(w)
    if not T:
        return None, None
    t = np.concatenate(T); w = np.concatenate(W)
    m = np.isfinite(w) & np.isfinite(t) & (w > 0)
    if not m.any():
        return None, None
    tt = t[m]; ww = w[m]
    return tt, ww / np.maximum(1.0, np.abs(tt))
