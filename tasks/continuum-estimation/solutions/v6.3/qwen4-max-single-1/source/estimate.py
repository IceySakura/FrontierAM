"""Continuum limit estimation from positive discretization levels.

Model (exact, noise-free observations):
    y(x) = t + a*x**p + b*x**q + c*x**(q + 1.5)
The limit t is the x->0 value. With 4 or 5 observations the 6 parameters are
under-determined, so we integrate t over the exact-fit manifold weighted by a
weak prior on the convergence orders and coefficients, and report the weighted
median (optimal for the absolute-error metric).
"""
import json
import math
import os
import sys

import numpy as np
from scipy.optimize import minimize_scalar

USE_DET = True
SIGNFLOOR = 0.05

# (lo, hi, soft-width) for each parameter; b and c are negative in this family.
PRIOR = dict(
    p=(0.5, 3.0, 0.15),
    delta=(0.4, 2.0, 0.15),
    absa=(0.2, 1.5, 0.12),
    b=(-0.9, -0.1, 0.08),
    c=(-0.4, -0.05, 0.05),
)


def boxw(u, lo, hi, w):
    u = np.asarray(u, float)
    out = np.ones_like(u)
    lo_m = u < lo
    hi_m = u > hi
    out[lo_m] = np.exp(-((lo - u[lo_m]) / w) ** 2)
    out[hi_m] = np.exp(-((u[hi_m] - hi) / w) ** 2)
    return out


def coef_weight(a, b, c, p, q, P=PRIOR):
    pl, ph, pw = P['p']
    dl, dh, dw = P['delta']
    al, ah, aw = P['absa']
    bl, bh, bw = P['b']
    cl, ch, cw = P['c']
    wb = boxw(b, bl, bh, bw) + SIGNFLOOR * boxw(-b, bl, bh, bw)
    wc = boxw(c, cl, ch, cw) + SIGNFLOOR * boxw(-c, cl, ch, cw)
    return (boxw(p, pl, ph, pw) * boxw(q - p, dl, dh, dw)
            * boxw(np.abs(a), al, ah, aw) * wb * wc)


def wmedian(vals, w):
    vals = np.asarray(vals, float)
    w = np.asarray(w, float)
    if len(vals) == 0:
        return None
    o = np.argsort(vals)
    v = vals[o]
    ww = w[o]
    cs = np.cumsum(ww)
    tot = cs[-1]
    if tot <= 0:
        return float(np.median(v))
    cdf = (cs - 0.5 * ww) / tot
    return float(np.interp(0.5, cdf, v))


def basis(x, p, q):
    return np.stack([np.ones_like(x), x ** p, x ** q, x ** (q + 1.5)], axis=-1)


def est_n4(x, y, P=PRIOR, np_=56, nd=56):
    ps = np.linspace(0.3, 3.6, np_)
    ds = np.linspace(0.25, 2.6, nd)
    PP, DD = np.meshgrid(ps, ds, indexing='ij')
    Pflat = PP.ravel()
    Qflat = (PP + DD).ravel()
    G = len(Pflat)
    m = len(x)
    A = np.stack([np.ones((G, m)), x[None, :] ** Pflat[:, None],
                  x[None, :] ** Qflat[:, None],
                  x[None, :] ** (Qflat[:, None] + 1.5)], axis=-1)
    yb = np.broadcast_to(y, (G, m))
    if m == 4:
        try:
            coef = np.linalg.solve(A, yb[..., None])[..., 0]
        except np.linalg.LinAlgError:
            coef = np.array([np.linalg.lstsq(Ai, y, rcond=None)[0] for Ai in A])
        resid = np.zeros(G)
    else:
        coef = np.empty((G, 4))
        resid = np.empty(G)
        for i in range(G):
            cc, _, _, _ = np.linalg.lstsq(A[i], y, rcond=None)
            coef[i] = cc
            resid[i] = np.linalg.norm(y - A[i] @ cc)
    t = coef[:, 0]
    a = coef[:, 1]
    b = coef[:, 2]
    c = coef[:, 3]
    ok = np.all(np.isfinite(coef), axis=1)
    w = coef_weight(a, b, c, Pflat, Qflat, P)
    if USE_DET and m == 4:
        det = np.abs(np.linalg.det(A))
        w = w / np.maximum(det, 1e-300)
    if m > 4:
        ys = max(1.0, float(np.max(np.abs(y))))
        w = w * np.exp(-(resid / (1e-4 * ys)) ** 2)
    w = w * ok
    msk = w > 1e-9
    if not np.any(msk):
        return float(np.mean(y))
    return wmedian(t[msk], w[msk])


def _null_resid_batch(x, y, p, qs):
    n = len(qs)
    m = len(x)
    A = np.stack([np.ones((n, m)), np.broadcast_to(x ** p, (n, m)),
                  x[None, :] ** qs[:, None],
                  x[None, :] ** (qs[:, None] + 1.5)], axis=-1)
    U, s, Vt = np.linalg.svd(A, full_matrices=True)
    nv = U[:, :, -1]
    return np.abs(nv @ y)


def est_n5(x, y, P=PRIOR, np_=64, nq=150):
    yscale = max(1.0, float(np.max(np.abs(y))))
    tol = 1e-6 * yscale
    ts = []
    ws = []
    ps = np.linspace(0.3, 3.6, np_)
    for p in ps:
        qs = np.linspace(p + 0.25, p + 3.0, nq)
        Rs = _null_resid_batch(x, y, p, qs)
        cand = np.where((Rs[1:-1] <= Rs[:-2]) & (Rs[1:-1] <= Rs[2:])
                        & (Rs[1:-1] < tol * 50))[0] + 1
        for i in cand:
            lo = qs[i - 1]
            hi = qs[i + 1]

            def fun(q):
                A = basis(x, p, q)
                coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
                r = y - A @ coef
                return float(r @ r)

            res = minimize_scalar(fun, bounds=(lo, hi), method='bounded',
                                  options={'xatol': 1e-12})
            q = res.x
            A = basis(x, p, q)
            coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            r = y - A @ coef
            if np.linalg.norm(r) < tol:
                t, a, b, c = coef
                w = float(coef_weight(a, b, c, p, q, P))
                if w > 1e-9:
                    ts.append(t)
                    ws.append(w)
    if not ts:
        return est_n4(x, y, P)
    return wmedian(ts, ws)


def estimate(x, y, P=PRIOR):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    o = np.argsort(x)
    x = x[o]
    y = y[o]
    z = None
    try:
        if len(x) == 4:
            z = est_n4(x, y, P)
        else:
            z = est_n5(x, y, P)
    except Exception:
        z = None
    if z is None or not np.isfinite(z):
        try:
            A = np.column_stack([np.ones_like(x), x ** 2])
            co, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            z = float(co[0])
        except Exception:
            z = float(y[0])
    B = max(1.0, float(np.max(np.abs(y))))
    lo = float(np.min(y)) - B
    hi = float(np.max(y)) + B
    return float(min(max(z, lo), hi))


def main():
    inp = sys.argv[1] if len(sys.argv) > 1 else '/workspace/inputs/input.json'
    out = sys.argv[2] if len(sys.argv) > 2 else '/workspace/result/output.json'
    with open(inp) as f:
        obj = json.load(f)
    z = estimate(obj['x'], obj['y'])
    if not math.isfinite(z):
        z = 0.0
    d = os.path.dirname(out)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(out, 'w') as f:
        json.dump({"estimate": z}, f, allow_nan=False)


if __name__ == '__main__':
    main()
