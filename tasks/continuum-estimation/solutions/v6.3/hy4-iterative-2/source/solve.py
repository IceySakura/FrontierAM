#!/usr/bin/env python3
"""Continuum-limit estimation from positive discretization levels.

The observations are modelled as

    y(x) = t + a*x^p + b*x^q + c*x^(q+3/2),     p, q > 0,

which is the smooth convergent error trend of the benchmark.  Given only
four or five samples, (p,q) and the amplitudes are not identifiable, so the
limit t is obtained by integrating the posterior over the exact-fit manifold
with a prior on the exponents and amplitudes (flat prior on t).  For five
samples the manifold is a curve (detected by a sign change of the orthogonal
residual); for four samples it is the whole (p,q) domain.

All computation is offline and single-threaded (NumPy / SciPy).
"""
import json
import math
import os
import sys

import numpy as np

# ----------------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------------
GAP = 1.5

# Prior: soft boxes on p, d=q-p, a, b, c.  Entries:
#   pq   : (pmin,pmax,dmin,dmax, sigma_p, sigma_d)
#   coef : ((amin,amax),(bmin,bmax),(cmin,cmax), sigma_a, sigma_b, sigma_c)
PRIOR = dict(
    pq=(0.5, 2.9, 0.3, 2.2, 0.06, 0.07),
    coef=((-1.5, 1.5), (-1.05, 0.03), (-0.44, 0.03), 0.025, 0.04, 0.025),
)
PRIOR_WIDE = dict(
    pq=(0.3, 3.8, 0.15, 3.2, 0.45, 0.5),
    coef=((-2.6, 2.6), (-1.7, 0.9), (-0.9, 0.6), 0.6, 0.5, 0.3),
)

NCURVE_P = 320
NCURVE_Q = 1000
NGRID4 = 480
JAC_EXP = 0.5


# ----------------------------------------------------------------------------
# linear algebra helpers (batched over exponent hypotheses)
# ----------------------------------------------------------------------------
def build_A(x, p, q, gap):
    X = x[None, :]
    P = np.asarray(p, float)[:, None]
    Q = np.asarray(q, float)[:, None]
    A = np.empty((P.shape[0], len(x), 4))
    A[:, :, 0] = 1.0
    A[:, :, 1] = X ** P
    A[:, :, 2] = X ** Q
    A[:, :, 3] = X ** (Q + gap)
    return A


def qr_solve(A, yc):
    """Batched least squares via reduced QR; also returns log|det R|."""
    N, n, _ = A.shape
    Q, R = np.linalg.qr(A, mode='reduced')
    qty = np.einsum('nji,nj->ni', Q, np.broadcast_to(yc, (N, n)))
    diagR = np.diagonal(R, axis1=1, axis2=2)
    bad = np.min(np.abs(diagR), axis=1) < 1e-280
    if np.any(bad):
        R = R.copy()
        R[bad] += np.eye(4) * 1e-10
    beta = np.linalg.solve(R, qty)
    resid = np.broadcast_to(yc, (N, n)) - np.einsum('nij,nj->ni', A, beta)
    logdetR = np.sum(np.log(np.maximum(np.abs(
        np.diagonal(R, axis1=1, axis2=2)), 1e-300)), axis=1)
    return beta, resid, logdetR


def signed_residual(A, yc):
    """Signed component of yc orthogonal to span(A), from full SVD."""
    U, sv, _ = np.linalg.svd(A, full_matrices=True)
    u = U[:, :, -1]
    return np.einsum('ni,i->n', u, yc)


def softbox(v, lo, hi, s):
    return -0.5 * ((np.maximum(lo - v, 0.0) / s) ** 2 +
                   (np.maximum(v - hi, 0.0) / s) ** 2)


def _softbox_sum(p, q, beta, cfg):
    d = q - p
    pmin, pmax, dmin, dmax, sp, sd = cfg['pq']
    lp = softbox(p, pmin, pmax, sp) + softbox(d, dmin, dmax, sd)
    a, b, c = beta[:, 1], beta[:, 2], beta[:, 3]
    abox, bbox, cbox, sa, sb, sc = cfg['coef']
    lp += (softbox(a, abox[0], abox[1], sa) +
           softbox(b, bbox[0], bbox[1], sb) +
           softbox(c, cbox[0], cbox[1], sc))
    return lp


def _log_norm(cfg):
    pmin, pmax, dmin, dmax, sp, sd = cfg['pq']
    abox, bbox, cbox, sa, sb, sc = cfg['coef']
    widths = [(pmax - pmin, sp), (dmax - dmin, sd),
              (abox[1] - abox[0], sa), (bbox[1] - bbox[0], sb),
              (cbox[1] - cbox[0], sc)]
    return sum(math.log(w + 2.5066282746310002 * s) for w, s in widths)


def log_prior(p, q, beta, cfg):
    """Log prior density.  A dict with components 'mix' is a two-component
    mixture: a sharp core plus a heavy-tailed broad component, each with its
    own normalization."""
    if 'mix' not in cfg:
        return _softbox_sum(p, q, beta, cfg)
    (w1, c1), (w2, c2) = cfg['mix']
    l1 = math.log(w1) + _softbox_sum(p, q, beta, c1) - _log_norm(c1)
    l2 = math.log(w2) + _softbox_sum(p, q, beta, c2) - _log_norm(c2)
    m = np.maximum(l1, l2)
    return m + np.log(np.exp(l1 - m) + np.exp(l2 - m))


def posterior_stats(t, lw):
    lw = np.where(np.isfinite(lw), lw, -1e300)
    m = float(np.max(lw))
    if not np.isfinite(m) or m < -1e290:
        return None
    w = np.exp(lw - m)
    s = w.sum()
    if not np.isfinite(s) or s <= 0:
        return None
    w /= s
    order = np.argsort(t)
    cs = np.cumsum(w[order])
    j = int(np.searchsorted(cs, 0.5))
    j = min(j, len(t) - 1)
    median = float(t[order][j])
    mean = float(np.dot(w, t))
    return median, mean, float(w.max()), w


# ----------------------------------------------------------------------------
# n = 5: integrate along exact-fit curve
# ----------------------------------------------------------------------------
def estimate5(x, y, cfg, gap=GAP, np_=NCURVE_P, nq=NCURVE_Q, verbose=False):
    ybar = float(np.mean(y))
    yc = y - ybar
    yr = float(np.linalg.norm(yc))
    if yr < 1e-13 * max(1.0, float(np.max(np.abs(y)))):
        return (ybar, ybar), None
    pmin, pmax, dmin, dmax, _, _ = cfg['pq']
    ps = np.linspace(max(0.15, pmin - 0.35), pmax + 0.35, np_)
    qlo = max(0.12, dmin - 0.45)
    qhi = pmax + dmax + 0.8
    qs = np.linspace(qlo, qhi, nq)
    brackets = []
    for p in ps:
        s = signed_residual(build_A(x, np.full(nq, p), qs, gap), yc)
        idx = np.where(s[:-1] * s[1:] < 0)[0]
        for i in idx:
            brackets.append((p, qs[i], qs[i + 1], s[i], s[i + 1]))
    if not brackets:
        return None, None
    bp = np.array([b[0] for b in brackets])
    lo = np.array([b[1] for b in brackets])
    hi = np.array([b[2] for b in brackets])
    flo = np.array([b[3] for b in brackets])
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        fm = signed_residual(build_A(x, bp, mid, gap), yc)
        take = np.sign(fm) == np.sign(flo)
        lo = np.where(take, mid, lo)
        flo = np.where(take, fm, flo)
        hi = np.where(~take, mid, hi)
    q = 0.5 * (lo + hi)
    d = q - bp
    keep = ((d > max(0.02, dmin - 0.55)) & (d < dmax + 1.0) &
            (q < qhi + 0.01) & (bp > 0.15))
    bp, q = bp[keep], q[keep]
    if len(q) == 0:
        return None, None
    A = build_A(x, bp, q, gap)
    beta, resid, logdetR = qr_solve(A, yc)
    r2 = np.einsum('ni,ni->n', resid, resid)
    tol = max(1e-3 * yr, 1e-12 * max(float(np.linalg.norm(y)), 1.0))
    ok = r2 < tol ** 2
    r2 = r2[ok]
    bp, q, beta, logdetR = bp[ok], q[ok], beta[ok], logdetR[ok]
    if len(q) == 0:
        return None, None
    h = 2e-6
    sp = signed_residual(build_A(x, bp, q + h, gap), yc)
    sm = signed_residual(build_A(x, bp, q - h, gap), yc)
    dsdq = np.maximum(np.abs((sp - sm) / (2 * h)), 1e-300)
    lp = log_prior(bp, q, beta, cfg)
    soft_noise = max(1e-4 * yr, 1e-13 * max(float(np.linalg.norm(y)), 1.0))
    lw = lp - JAC_EXP * logdetR - np.log(dsdq) - r2 / (2.0 * soft_noise ** 2)
    t = beta[:, 0] + ybar
    st = posterior_stats(t, lw)
    if st is None:
        return None, None
    median, mean, wmax, w = st
    diag = dict(median=median, mean=mean, maxw=wmax,
                effective=float(1.0 / np.sum(w ** 2)), np=len(t),
                maxlp=float(np.max(lp)), resid_min=float(np.sqrt(np.min(r2))))
    if verbose:
        diag.update(w=w, t=t, p=bp, q=q, beta=beta)
    return (median, mean), diag


# ----------------------------------------------------------------------------
# n = 4: integrate over the 2D (p,d) domain
# ----------------------------------------------------------------------------
def estimate4(x, y, cfg, gap=GAP, ng=NGRID4, verbose=False):
    ybar = float(np.mean(y))
    yc = y - ybar
    yr = float(np.linalg.norm(yc))
    if yr < 1e-13 * max(1.0, float(np.max(np.abs(y)))):
        return (ybar, ybar), None
    pmin, pmax, dmin, dmax, _, _ = cfg['pq']
    ps = np.linspace(max(0.1, pmin - 0.45), pmax + 0.45, ng)
    ds = np.linspace(max(0.08, dmin - 0.45), dmax + 0.6, ng)
    P, D = np.meshgrid(ps, ds, indexing='ij')
    p = P.ravel()
    q = (P + D).ravel()
    ts, lws, lps = [], [], []
    sig = 1e-9 * max(abs(ybar), yr, 1.0)
    for k in range(0, len(p), 20000):
        pk, qk = p[k:k + 20000], q[k:k + 20000]
        A = build_A(x, pk, qk, gap)
        beta, resid, logdetR = qr_solve(A, yc)
        r2 = np.einsum('ni,ni->n', resid, resid)
        lp = log_prior(pk, qk, beta, cfg)
        lw = lp - JAC_EXP * logdetR - r2 / (2 * sig ** 2)
        ts.append(beta[:, 0])
        lws.append(lw)
        lps.append(lp)
    t = np.concatenate(ts) + ybar
    lw = np.concatenate(lws)
    st = posterior_stats(t, lw)
    if st is None:
        return None, None
    median, mean, wmax, w = st
    diag = dict(median=median, mean=mean, maxw=wmax,
                effective=float(1.0 / np.sum(w ** 2)), np=len(t),
                maxlp=float(np.max(np.concatenate(lps))))
    if verbose:
        diag.update(w=w, t=t, p=p, q=q)
    return (median, mean), diag


def bayes(x, y, cfg, gap=GAP, verbose=False):
    if len(x) == 5:
        return estimate5(x, y, cfg, gap, verbose=verbose)
    if len(x) == 4:
        return estimate4(x, y, cfg, gap, verbose=verbose)
    return None, None


# ----------------------------------------------------------------------------
# model-free fallbacks
# ----------------------------------------------------------------------------
def fit2_nls(x, y):
    """Least-squares fit of y = c0 + c1 x^p + c2 x^q with free orders."""
    from scipy.optimize import least_squares
    def fun(v):
        return v[0] + v[1] * np.power(x, v[2]) + v[3] * np.power(x, v[4]) - y
    best = None
    for p0 in (0.7, 1.2, 1.8, 2.4, 3.2):
        for q0 in (1.7, 2.5, 3.4, 4.6):
            if q0 <= p0 + 0.2:
                continue
            A = np.column_stack([np.ones_like(x), x ** p0, x ** q0])
            c = np.linalg.lstsq(A, y, rcond=None)[0]
            try:
                r = least_squares(
                    fun, [c[0], c[1], p0, c[2], q0],
                    bounds=([-np.inf, -np.inf, 0.15, -np.inf, 0.25],
                            [np.inf, np.inf, 8.0, np.inf, 12.0]),
                    xtol=1e-13, ftol=1e-13, gtol=1e-13, max_nfev=400)
                if best is None or r.cost < best.cost:
                    best = r
            except Exception:
                continue
    return float(best.x[0]) if best is not None else None


def richardson(x, y):
    """Simple robust extrapolation: fit c0 + c1 x^p for a few p, take c0."""
    # p estimated from the three smallest levels by solving the ratio equation
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    est = []
    for p in (0.7, 1.0, 1.4, 2.0, 2.6, 3.2):
        A = np.column_stack([np.ones_like(x), x ** p])
        c = np.linalg.lstsq(A, y, rcond=None)[0]
        est.append(c[0])
    est.append(ys[0])
    return float(np.median(est))


PRIOR_NORM = dict(
    pq=(0.4, 3.2, 0.25, 2.6, 0.15, 0.20),
    coef=((-4.0, 4.0), (-4.0, 0.8), (-1.5, 0.8), 0.5, 0.3, 0.20),
)


def bayes_scaled(x, y):
    xr = float(np.max(x))
    ybar = float(np.mean(y))
    S = float(np.max(y) - np.min(y))
    if S < 1e-13 * max(1.0, float(np.max(np.abs(y)))):
        return (ybar, ybar), None
    u = x / xr
    yv = (y - ybar) / S
    pair, diag = bayes(u, yv, PRIOR_NORM)
    if pair is None:
        return None, diag
    return (ybar + S * pair[0], ybar + S * pair[1]), diag


def write_output(path, value):
    value = float(value)
    if not math.isfinite(value):
        value = 0.0
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump({'estimate': value}, f, allow_nan=False)
        f.write('\n')
    os.replace(tmp, path)


def main():
    inp = sys.argv[1] if len(sys.argv) > 1 else '/workspace/inputs/input.json'
    out = sys.argv[2] if len(sys.argv) > 2 else '/workspace/result/output.json'
    fallback = 0.0
    try:
        with open(inp) as f:
            obj = json.load(f)
        x = np.array([float(v) for v in obj['x']], dtype=float)
        y = np.array([float(v) for v in obj['y']], dtype=float)
        order = np.argsort(x)
        x, y = x[order], y[order]
        fallback = float(y[0])
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
            raise ValueError('nonfinite input')
        if np.all(y == y[0]):
            write_output(out, y[0])
            return
        # sanity range for extrapolated limits
        span = float(np.max(y) - np.min(y))
        eps = 1e-6 * max(1.0, float(np.max(np.abs(y))))
        lo = float(np.min(y)) - 4.0 * span - eps
        hi = float(np.max(y)) + 4.0 * span + eps

        dy = np.diff(y)
        nonmono = bool(np.any(dy > 0) and np.any(dy < 0))
        z = None
        diag = None
        if nonmono:
            try:
                v = fit2_nls(x, y)
                if v is not None and math.isfinite(v) and lo <= v <= hi:
                    z = v
            except Exception:
                pass
            if z is None:
                try:
                    z = richardson(x, y)
                except Exception:
                    z = fallback
            write_output(out, z)
            return
        try:
            pair, diag = bayes(x, y, PRIOR)
            if pair is not None:
                cand = 0.75 * pair[0] + 0.25 * pair[1]
                if math.isfinite(cand) and lo <= cand <= hi:
                    z = cand
                    # If the sharp prior is inconsistent with every exact-fit
                    # hypothesis, hedge with a much wider prior.
                    mlp = diag.get('maxlp', 0.0) if diag else 0.0
                    if mlp < -4.0:
                        try:
                            pair2, diag2 = bayes(x, y, PRIOR_WIDE)
                            if pair2 is not None:
                                c2 = 0.75 * pair2[0] + 0.25 * pair2[1]
                                if math.isfinite(c2) and lo <= c2 <= hi:
                                    wt = 1.0 / (1.0 + math.exp(-(mlp + 9.0) / 1.5))
                                    wt = max(0.0, min(1.0, wt))
                                    z = wt * cand + (1.0 - wt) * c2
                        except Exception:
                            pass
        except Exception:
            pass
        cand_n = None
        try:
            pair_n, diag_n = bayes_scaled(x, y)
            if pair_n is not None:
                cn = 0.75 * pair_n[0] + 0.25 * pair_n[1]
                if math.isfinite(cn) and lo <= cn <= hi:
                    cand_n = cn
        except Exception:
            pass
        if z is not None and cand_n is not None:
            mlp = diag.get('maxlp', 0.0) if diag else 0.0
            u = (mlp + 6.0) / 1.5
            w = 0.0 if u < -60 else (1.0 if u > 60 else 1.0 / (1.0 + math.exp(-u)))
            if float(np.max(x)) > 0.9:
                w = min(w, 0.15)
            z = w * z + (1.0 - w) * cand_n
        elif z is None and cand_n is not None:
            z = cand_n
        if z is None:
            try:
                pair, diag = bayes(x, y, PRIOR_WIDE)
                if pair is not None:
                    cand = 0.75 * pair[0] + 0.25 * pair[1]
                    if math.isfinite(cand) and lo <= cand <= hi:
                        z = cand
            except Exception:
                pass
        if z is None:
            try:
                v = fit2_nls(x, y)
                if v is not None and math.isfinite(v) and lo <= v <= hi:
                    z = v
            except Exception:
                pass
        if z is None:
            z = richardson(x, y)
        # Hedge with a model-free two-power extrapolation.
        try:
            v2 = fit2_nls(x, y)
            if v2 is not None and math.isfinite(v2) and lo <= v2 <= hi:
                z = 0.60 * z + 0.40 * v2
        except Exception:
            pass
        if not math.isfinite(z):
            z = fallback
        write_output(out, z)
    except Exception:
        write_output(out, fallback if math.isfinite(fallback) else 0.0)


if __name__ == '__main__':
    main()
