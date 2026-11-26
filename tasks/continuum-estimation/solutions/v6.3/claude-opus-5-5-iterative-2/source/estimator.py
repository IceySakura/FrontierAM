"""Bayesian continuum estimator for y = t + a x^p + b x^q + c x^(q+1.5), exact data."""
import numpy as np

def log_softbox(v, lo, hi, w):
    return -np.logaddexp(0.0, -(v - lo) / w) - np.logaddexp(0.0, -(hi - v) / w)

PRIOR = dict(p=(0.5, 2.9, 0.04), d=(0.4, 2.4, 0.04), a=(-1.6, 1.6, 0.04),
             b=(-0.95, 0.0, 0.02), c=(-0.42, 0.0, 0.01))
GRID = dict(dp=0.005, dd=0.005, margin=6.0)
EXTRA = 1.5
LAST = None
KEEP_LAST = False

def log_prior1(v, spec):
    """spec: (lo, hi, w) soft box, or list of (weight, lo, hi, w) mixture components."""
    if isinstance(spec, tuple):
        if spec[0] == 'pl':  # ('pl', m, M, k, lo_hard, hi_hard, w): flat on [m,M], power-law tails, soft hard limits
            _, m, M, k, lh, hh, w = spec
            R = M - m
            dist = np.maximum(m - v, 0.0) + np.maximum(v - M, 0.0)
            out = -k * np.log1p(dist / R)
            if lh is not None:
                out = out - np.logaddexp(0.0, -(v - lh) / w)
            if hh is not None:
                out = out - np.logaddexp(0.0, -(hh - v) / w)
            return out
        if len(spec) == 4:  # (lo, hi, w, k): soft box times v**(-k)
            lo, hi, w, k = spec
            return log_softbox(v, lo, hi, w) - k * np.log(np.maximum(v, 1e-3))
        return log_softbox(v, *spec)
    acc = None
    for wt, lo, hi, w in spec:
        term = np.log(wt / (hi - lo)) + log_softbox(v, lo, hi, w)
        acc = term if acc is None else np.logaddexp(acc, term)
    return acc

def _range(spec):
    if isinstance(spec, tuple):
        if spec[0] == 'pl':
            _, m, M, k, lh, hh, w = spec
            R = M - m
            lo = m - R * 0.6 if lh is None else max(lh, m - R * 0.6)
            hi = M + R * 0.6 if hh is None else min(hh, M + R * 0.6)
            return (lo, hi, w)
        return spec[:3]
    return (min(c[1] for c in spec), max(c[2] for c in spec), max(c[3] for c in spec))

def _axis(lo, hi, w, step, margin):
    a0 = max(lo - margin * w, 0.02)
    a1 = hi + margin * w
    m = int(np.ceil((a1 - a0) / step)) + 1
    return np.linspace(a0, a1, m)

def _logprior_coef(a, b, c, pr):
    return log_prior1(a, pr['a']) + log_prior1(b, pr['b']) + log_prior1(c, pr['c'])

def weighted_median(v, w):
    o = np.argsort(v)
    v = v[o]; w = w[o]
    cw = np.cumsum(w)
    tot = cw[-1]
    k = np.searchsorted(cw, 0.5 * tot)
    return float(v[min(k, len(v) - 1)])

def _design(L, P, Q, xs15):
    # L: (n,), P,Q: broadcastable arrays -> (..., n, 4)
    e1 = np.exp(P[..., None] * L)
    e2 = np.exp(Q[..., None] * L)
    e3 = e2 * xs15
    return np.stack([np.ones_like(e1), e1, e2, e3], axis=-1)

def posterior_samples(x, y, pr=PRIOR, grid=GRID):
    """Return (t_values, weights) sampling the posterior of t."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    o = np.argsort(x)[::-1]
    x = x[o]; y = y[o]
    n = len(x)
    s = x.max()
    xs = x / s
    L = np.log(xs)
    xs15 = xs ** EXTRA
    y0 = y[-1]
    yc = y - y0
    P = _axis(*_range(pr['p']), grid['dp'], grid['margin'])
    D = _axis(*_range(pr['d']), grid['dd'], grid['margin'])
    ls = np.log(s)
    if n == 4:
        PP, DD = np.meshgrid(P, D, indexing='ij')
        QQ = PP + DD
        X = _design(L, PP, QQ, xs15)
        sign, logdet = np.linalg.slogdet(X)
        ok = sign != 0
        Xs = np.where(ok[..., None, None], X, np.eye(4))
        beta = np.linalg.solve(Xs, np.broadcast_to(yc, X.shape[:-1]))
        a = beta[..., 1] * np.exp(-PP * ls)
        b = beta[..., 2] * np.exp(-QQ * ls)
        c = beta[..., 3] * np.exp(-(QQ + EXTRA) * ls)
        lw = (log_prior1(PP, pr['p']) + log_prior1(DD, pr['d']) + _logprior_coef(a, b, c, pr)
              - logdet - (3 * PP + 2 * DD + EXTRA) * ls)
        lw = np.where(ok & np.isfinite(lw), lw, -np.inf)
        t = beta[..., 0] + y0
        lw = lw.ravel(); t = t.ravel()
        extra = dict(p=PP.ravel(), d=DD.ravel(), a=a.ravel(), b=b.ravel(), c=c.ravel())
    elif n == 5:
        PP, DD = np.meshgrid(P, D, indexing='ij')
        QQ = PP + DD
        X = _design(L, PP, QQ, xs15)
        M = np.concatenate([X, np.broadcast_to(yc, X.shape[:-1])[..., None]], axis=-1)
        G = np.linalg.det(M)
        G = np.where(G == 0, 1e-300, G)
        dp = P[1] - P[0]; dd = D[1] - D[0]
        G00 = G[:-1, :-1]; G10 = G[1:, :-1]; G01 = G[:-1, 1:]; G11 = G[1:, 1:]
        pi = P[:-1][:, None] * np.ones_like(G00); dj = D[:-1][None, :] * np.ones_like(G00)
        # edges: bottom (00-10, along p at d_j), top (01-11 at d_j+1), left (00-01 along d at p_i), right (10-11 at p_i+1)
        def cross(g1, g2):
            return (np.sign(g1) != np.sign(g2)), g1 / (g1 - g2)
        mb, fb = cross(G00, G10); mt, ft = cross(G01, G11)
        ml, fl = cross(G00, G01); mr, fr = cross(G10, G11)
        cnt = mb.astype(int) + mt + ml + mr
        pts_p = np.stack([pi + fb * dp, pi + ft * dp, pi + 0 * fl, pi + dp + 0 * fr], axis=-1)
        pts_d = np.stack([dj + 0 * fb, dj + dd + 0 * ft, dj + fl * dd, dj + fr * dd], axis=-1)
        msk = np.stack([mb, mt, ml, mr], axis=-1)
        segs = []
        idx2 = np.nonzero(cnt == 2)
        if len(idx2[0]):
            mm = msk[idx2]
            pp = pts_p[idx2]; ddv = pts_d[idx2]
            order = np.argsort(~mm, axis=1, kind='stable')[:, :2]
            r = np.arange(len(mm))
            segs.append((pp[r, order[:, 0]], ddv[r, order[:, 0]], pp[r, order[:, 1]], ddv[r, order[:, 1]],
                         idx2[0], idx2[1]))
        idx4 = np.nonzero(cnt == 4)
        for i, j in zip(*idx4):
            gc = 0.25 * (G00[i, j] + G10[i, j] + G01[i, j] + G11[i, j])
            if np.sign(gc) == np.sign(G00[i, j]):
                pairs = [(0, 3), (2, 1)]
            else:
                pairs = [(0, 2), (1, 3)]
            for k1, k2 in pairs:
                segs.append((np.array([pts_p[i, j, k1]]), np.array([pts_d[i, j, k1]]),
                             np.array([pts_p[i, j, k2]]), np.array([pts_d[i, j, k2]]),
                             np.array([i]), np.array([j])))
        if not segs:
            return np.array([]), np.array([])
        p1 = np.concatenate([s_[0] for s_ in segs]); d1 = np.concatenate([s_[1] for s_ in segs])
        p2 = np.concatenate([s_[2] for s_ in segs]); d2 = np.concatenate([s_[3] for s_ in segs])
        ci = np.concatenate([s_[4] for s_ in segs]); cj = np.concatenate([s_[5] for s_ in segs])
        pm = 0.5 * (p1 + p2); dm = 0.5 * (d1 + d2)
        seglen = np.hypot(p2 - p1, d2 - d1)
        u = (pm - P[ci]) / dp; v = (dm - D[cj]) / dd
        g00 = G00[ci, cj]; g10 = G10[ci, cj]; g01 = G01[ci, cj]; g11 = G11[ci, cj]
        gp = ((g10 - g00) * (1 - v) + (g11 - g01) * v) / dp
        gd = ((g01 - g00) * (1 - u) + (g11 - g10) * u) / dd
        gn = np.hypot(gp, gd)
        qm = pm + dm
        Xm = _design(L, pm, qm, xs15)
        Q_, R_ = np.linalg.qr(Xm)
        rhs = np.einsum('kij,i->kj', Q_, yc)
        beta = np.linalg.solve(R_, rhs)
        a = beta[:, 1] * np.exp(-pm * ls)
        b = beta[:, 2] * np.exp(-qm * ls)
        c = beta[:, 3] * np.exp(-(qm + EXTRA) * ls)
        lw = (log_prior1(pm, pr['p']) + log_prior1(dm, pr['d']) + _logprior_coef(a, b, c, pr)
              + np.log(seglen + 1e-300) - np.log(gn + 1e-300) - (3 * pm + 2 * dm + EXTRA) * ls)
        lw = np.where(np.isfinite(lw) & (seglen > 0), lw, -np.inf)
        t = beta[:, 0] + y0
        extra = dict(p=pm, d=dm, a=a, b=b, c=c)
    else:
        raise ValueError('unsupported n')
    good = np.isfinite(lw) & np.isfinite(t)
    t = t[good]; lw = lw[good]
    extra = {k: v[good] for k, v in extra.items()}
    global LAST
    if len(t) == 0:
        LAST = extra
        return t, lw
    w = np.exp(lw - lw.max())
    extra['w'] = w; extra['t'] = t
    LAST = extra
    return t, w

def estimate(x, y, pr=PRIOR, grid=GRID, stat='wmedian'):
    t, w = posterior_samples(x, y, pr, grid)
    if len(t) == 0 or not np.isfinite(w.sum()) or w.sum() <= 0:
        return None
    if stat == 'mean':
        return float(np.sum(t * w) / np.sum(w))
    ww = w / np.maximum(1.0, np.abs(t))
    return weighted_median(t, ww)


def _widen(pr, f):
    out = {}
    for k, spec in pr.items():
        lo, hi, w = _range(spec)
        c = 0.5 * (lo + hi); h = 0.5 * (hi - lo) * f
        out[k] = (c - h, c + h, w * f)
    out['p'] = (max(out['p'][0], 0.05), out['p'][1], out['p'][2])
    out['d'] = (max(out['d'][0], 0.05), out['d'][1], out['d'][2])
    return out


def robust_estimate(x, y, pr=None, grid=None):
    pr = PRIOR if pr is None else pr
    grid = GRID if grid is None else grid
    x = np.asarray(x, float); y = np.asarray(y, float)
    n = len(x)
    if n < 4:
        return None
    if np.ptp(y) == 0:
        return float(y[0])
    if n > 5:
        o = np.argsort(x)[:5]
        x = x[o]; y = y[o]
    # sanity bound: the continuum value cannot sit far outside the observed trend
    guard = max(10.0 * float(np.ptp(y)), 3.0)
    ymin_x = float(y[np.argmin(x)])
    for f in (1.0, 2.0, 4.0):
        prf = pr if f == 1.0 else _widen(pr, f)
        grf = grid if f == 1.0 else dict(grid, dp=grid['dp'] * f, dd=grid['dd'] * f)
        try:
            z = estimate(x, y, prf, grf)
        except Exception:
            z = None
        if z is not None and np.isfinite(z) and abs(z - ymin_x) <= guard:
            return z
    return None
