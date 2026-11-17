"""Posterior-median estimate of T for y = T + a x^p + b x^q + c x^(q+1.5)."""
import numpy as np

# prior: name -> (lo, hi, edge_width) logistic soft box, or a list of (mass, box)
# mixture components (each box normalised by its width); 'absa'/'absT' act on |a|, |T|.


def _tail(core, lo, hi, eps):
    # density 1 on the core box plus eps on [lo, hi] (hedge against wider true ranges)
    return [(core[1] - core[0], core), (eps * (hi - lo), (lo, hi, core[2]))]


DEFAULT = dict(
    p=[(2.5, (0.5, 3.0, 0.02)), (0.02, (0.3, 0.5, 0.02))],
    d=[(0.3, (0.4, 0.7, 0.02)), (0.7, (0.4, 2.25, 0.02)), (0.2 * 0.378 * 0.75, (2.25, 3.0, 0.02))],
    absa=_tail((0.2, 1.55, 0.02), 0.0, 2.5, 0.1),
    b=_tail((-0.93, -0.07, 0.01), -1.5, 0.0, 0.1),
    c=_tail((-0.42, -0.03, 0.01), -0.8, 0.0, 0.1),
    absT=(0.0, 1e6, 1.0))


def soft_box(v, lo, hi, w):
    if w <= 0:
        return ((v >= lo) & (v <= hi)).astype(float)
    with np.errstate(over='ignore'):
        return 1.0 / (1.0 + np.exp(-(v - lo) / w)) / (1.0 + np.exp((v - hi) / w))


def _dens(spec, v):
    # spec: (lo, hi, w) soft box, or a callable density, or list of (weight, spec) mixture
    if callable(spec):
        return spec(v)
    if isinstance(spec, list):
        return sum(wt * _dens(sp, v) / (sp[1] - sp[0] if not callable(sp) else 1.0) for wt, sp in spec)
    return soft_box(v, *spec)


def prior_pd(P, Dd, pr):
    return _dens(pr['p'], P) * _dens(pr['d'], Dd)


def prior_theta(th, pr):
    T, a, b, c = th[..., 0], th[..., 1], th[..., 2], th[..., 3]
    w = _dens(pr['absa'], np.abs(a)) * _dens(pr['b'], b) * _dens(pr['c'], c)
    return w * _dens(pr['absT'], np.abs(T))


def design(x, P, Q):
    lx = np.log(x)[None, :]
    return np.stack([np.ones(P.shape + x.shape), np.exp(P[:, None] * lx), np.exp(Q[:, None] * lx),
                     np.exp((Q[:, None] + 1.5) * lx)], -1)


def solve_ls(A, y):
    # batched least squares via QR; A (m, n, 4)
    Qm, R = np.linalg.qr(A)
    rhs = np.einsum('mni,n->mi', Qm, y)
    return np.linalg.solve(R, rhs[..., None])[..., 0]


def grid_eval(x, y, pr, p_lo, p_hi, d_lo, d_hi, n_p, n_d):
    ps = np.linspace(p_lo, p_hi, n_p)
    ds = np.linspace(d_lo, d_hi, n_d)
    P, Dd = np.meshgrid(ps, ds, indexing='ij')
    Pf, Df = P.ravel(), Dd.ravel()
    n = len(x)
    if n == 4:
        A = design(x, Pf, Pf + Df)
        det = np.linalg.det(A)
        ok = np.abs(det) > 1e-300
        th = np.zeros((len(Pf), 4))
        th[ok] = np.linalg.solve(A[ok], np.broadcast_to(y, (ok.sum(), 4))[..., None])[..., 0]
        w = np.zeros(len(Pf))
        w[ok] = prior_pd(Pf[ok], Df[ok], pr) * prior_theta(th[ok], pr) / np.abs(det[ok])
        w *= (ps[1] - ps[0]) * (ds[1] - ds[0])
        return th[:, 0], w, Pf, Df, th
    # n == 5: integrate along curve det([A y]) = 0
    A = design(x, Pf, Pf + Df)
    M = np.concatenate([A, np.broadcast_to(y, (A.shape[0], n))[..., None]], -1)
    D = np.linalg.det(M).reshape(n_p, n_d)
    dp, dd = ps[1] - ps[0], ds[1] - ds[0]
    gp, gd = np.gradient(D, dp, dd)
    Ps, Ds, Gs, Ss = [], [], [], []
    for axis in (1, 0):
        if axis == 1:
            s0, s1 = D[:, :-1], D[:, 1:]
            g0, g1, h0, h1 = gd[:, :-1], gd[:, 1:], gp[:, :-1], gp[:, 1:]
        else:
            s0, s1 = D[:-1, :], D[1:, :]
            g0, g1, h0, h1 = gp[:-1, :], gp[1:, :], gd[:-1, :], gd[1:, :]
        idx = np.nonzero(s0 * s1 < 0)
        if len(idx[0]) == 0:
            continue
        fr = s0[idx] / (s0[idx] - s1[idx])
        gpar = (1 - fr) * g0[idx] + fr * g1[idx]
        gperp = (1 - fr) * h0[idx] + fr * h1[idx]
        keep = np.abs(gpar) >= np.abs(gperp)
        i, j, fr, gpar = idx[0][keep], idx[1][keep], fr[keep], gpar[keep]
        if axis == 1:
            pp = ps[i]; dv = ds[j] + fr * dd; step = dp
        else:
            pp = ps[i] + fr * dp; dv = ds[j]; step = dd
        Ps.append(pp); Ds.append(dv); Gs.append(np.abs(gpar)); Ss.append(np.full(len(pp), step))
    if not Ps:
        z = np.zeros(0)
        return z, z, z, z, np.zeros((0, 4))
    pp = np.concatenate(Ps); dv = np.concatenate(Ds); g = np.concatenate(Gs); st = np.concatenate(Ss)
    A2 = design(x, pp, pp + dv)
    th = solve_ls(A2, y)
    w = prior_pd(pp, dv, pr) * prior_theta(th, pr) * st / np.maximum(g, 1e-300)
    return th[:, 0], w, pp, dv, th


def wmedian(v, w):
    o = np.argsort(v)
    v, w = v[o], w[o]
    c = np.cumsum(w)
    return float(np.interp(0.5 * c[-1], c, v))


def estimate(x, y, pr=DEFAULT, n1=260, n2=420, rel=True, return_post=False):
    x = np.asarray(x, float); y = np.asarray(y, float)
    o = np.argsort(x); x, y = x[o], y[o]
    p_lo, p_hi = 0.1, 4.0
    d_lo, d_hi = 0.05, 3.5
    T, w, P, Dd, th = grid_eval(x, y, pr, p_lo, p_hi, d_lo, d_hi, n1, n1)
    if len(w) == 0 or not np.isfinite(w).all() or w.sum() <= 0:
        return None
    # refine around support
    m = w > w.max() * 1e-6
    sp = (p_hi - p_lo) / (n1 - 1); sd = (d_hi - d_lo) / (n1 - 1)
    a0, a1 = max(p_lo, P[m].min() - 2 * sp), min(p_hi, P[m].max() + 2 * sp)
    b0, b1 = max(d_lo, Dd[m].min() - 2 * sd), min(d_hi, Dd[m].max() + 2 * sd)
    T2, w2, P2, D2, th2 = grid_eval(x, y, pr, a0, a1, b0, b1, n2, n2)
    if len(w2) and np.isfinite(w2).all() and w2.sum() > 0:
        T, w, P, Dd, th = T2, w2, P2, D2, th2
    ww = w / np.maximum(1.0, np.abs(T)) if rel else w
    est = wmedian(T, ww)
    if return_post:
        return est, (T, w, P, Dd, th)
    return est


WIDE = dict(p=(0.2, 4.0, 0.05), d=(0.1, 3.4, 0.05), absa=(0.0, 5.0, 0.05),
            b=(-3.0, 0.5, 0.05), c=(-2.0, 0.5, 0.05), absT=(0.0, 20.0, 0.1))


def fallback_fit(x, y):
    # last-resort least-squares fit of T + a x^p (+ b x^(p+d)) over a coarse (p, d) grid
    x = np.asarray(x, float); y = np.asarray(y, float)
    o = np.argsort(x); x, y = x[o], y[o]
    n = len(x)
    if n < 3:
        return float(y[0])
    best = None
    for p in np.linspace(0.3, 4.0, 38):
        for dd in np.linspace(0.2, 3.0, 15):
            cols = [np.ones(n), x ** p]
            if n >= 4:
                cols.append(x ** (p + dd))
            A = np.stack(cols, 1)
            th, *_ = np.linalg.lstsq(A, y, rcond=None)
            r = float(np.sum((A @ th - y) ** 2))
            if best is None or r < best[0]:
                best = (r, th[0])
    return float(best[1])


def robust_estimate(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    n = len(x)
    if n in (4, 5):
        for pr in (DEFAULT, WIDE):
            try:
                z = estimate(x, y, pr)
            except TimeoutError:
                raise
            except Exception:
                z = None
            if z is not None and np.isfinite(z):
                return float(z)
    elif n > 5:
        # use the 5 finest levels
        o = np.argsort(x)[:5]
        for pr in (DEFAULT, WIDE):
            try:
                z = estimate(x[o], y[o], pr)
            except TimeoutError:
                raise
            except Exception:
                z = None
            if z is not None and np.isfinite(z):
                return float(z)
    return fallback_fit(x, y)
